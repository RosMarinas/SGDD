"""
Diffusion Decoder - Bidirectional Transformer with Cross-Attention

Implements a 6-layer bidirectional transformer decoder with:
- Rotary Position Embeddings (RoPE)
- Cross-attention to semantic vectors
- Sinusoidal timestep embeddings
- Weight tying between input and output embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).

    Implements rotary position encoding which applies rotation to query and key
    vectors based on their position. Provides better extrapolation than learned
    absolute positional embeddings.

    Based on: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048):
        """
        Initialize rotary embeddings.

        Args:
            dim: Dimension of each attention head (must be even)
            max_position_embeddings: Maximum sequence length
        """
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        """Build cached frequency tensor for given sequence length"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple:
        """
        Apply rotary embeddings to input.

        Args:
            x: Input tensor [batch, seq_len, heads, head_dim]
            seq_len: Sequence length

        Returns:
            cos: Cosine component for rotation
            sin: Sine component for rotation
        """
        # Rebuild cache if sequence length exceeds cached size
        if seq_len > self.max_position_embeddings:
            self._build_cache(seq_len)
            self.max_position_embeddings = seq_len

        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the dimensions for RoPE"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    """
    Apply rotary position embeddings to query and key.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine component [batch, heads, seq_len, head_dim]
        sin: Sine component [batch, heads, seq_len, head_dim]

    Returns:
        q_embed: Rotated query
        k_embed: Rotated key
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPESelfAttention(nn.Module):
    """
    Multi-head self-attention with Rotary Position Embeddings (RoPE).
    
    Replaces nn.MultiheadAttention to allow manual injection of RoPE 
    into Query and Key vectors before attention computation.
    Uses F.scaled_dot_product_attention for efficiency (FlashAttention compatible).
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout_p = dropout
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens [batch, seq_len, hidden_dim]
            cos, sin: RoPE embeddings [batch, heads, seq_len, head_dim] (broadcastable)
            attn_mask: Attention mask [batch, seq_len] (0 for pad, 1 for keep) or broadcastable shape.
        """
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape to [batch, heads, seq_len, head_dim]
        # Note: We transpose to [batch, heads, seq_len, head_dim] for SDPA
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Query and Key
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Prepare attention mask for SDPA if needed
        sdpa_mask = attn_mask
        if attn_mask is not None:
            # If mask is [batch, seq_len], we need to expand/invert it for SDPA
            # SDPA expects:
            # - None (no mask)
            # - Boolean mask (True = ignore/mask out, False = keep) - WAIT, PyTorch docs say:
            #   "Binary mask where True values indicate elements that should be computed (keep), 
            #    and False values indicate elements that should be ignored (mask)." -> NO, that's for some ops.
            # Let's check SDPA docs: "attn_mask: ... binary mask (True = attend, False = ignore)... OR additive mask".
            # Usually for padding mask [batch, seq_len] with 1=keep, 0=pad:
            # We want to mask out where it is 0.
            
            if attn_mask.dim() == 2:
                # Expand to [batch, 1, 1, seq_len] to broadcast over heads and query sequence
                # Shape: [batch, heads, queries, keys] -> [batch, 1, 1, seq_len]
                # We construct an additive mask: 0 for keep, -inf for mask.
                # attn_mask is 1 for keep, 0 for pad.
                # (1 - 1) * -inf = 0
                # (1 - 0) * -inf = -inf
                sdpa_mask = (1.0 - attn_mask.unsqueeze(1).unsqueeze(1)) * -1e9
            
        # Scaled Dot Product Attention
        # Uses FlashAttention if available and conditions met
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False # SGDD is bidirectional
        )
        
        # Reshape back to [batch, seq_len, hidden_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(out)


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embeddings.

    Encodes the diffusion timestep using sinusoidal embeddings,
    similar to DDPM. This helps the model distinguish between
    different noise levels.
    """

    def __init__(self, dim: int):
        """
        Initialize time embedding.

        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal timestep embeddings.

        Args:
            timestep: Timestep values [batch_size]

        Returns:
            embedding: Time embeddings [batch_size, dim]
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timestep.device) * -emb)
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class DecoderLayer(nn.Module):
    """
    Single decoder layer with self-attention, cross-attention, and FFN.

    This is a bidirectional transformer layer (no causal mask) with:
    - Multi-head self-attention over tokens (with RoPE)
    - Multi-head cross-attention to semantic vector
    - Position-wise feed-forward network
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Initialize decoder layer.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        # Self-attention (bidirectional + RoPE)
        self.self_attn = RoPESelfAttention(
            hidden_dim, num_heads, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Cross-attention to semantic vector
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        semantic_vector: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.

        Args:
            x: Input tokens [batch, seq_len, hidden_dim]
            semantic_vector: Semantic vector [batch, hidden_dim]
            cos, sin: RoPE embeddings
            attn_mask: Optional attention mask [seq_len, seq_len]

        Returns:
            output: Processed tokens [batch, seq_len, hidden_dim]
        """
        # Self-attention with RoPE
        attn_output = self.self_attn(x, cos, sin, attn_mask=attn_mask)
        x = self.norm1(x + attn_output)

        # Cross-attention: semantic_vector is [batch, hidden_dim]
        # Need to expand to [batch, 1, hidden_dim] for attention
        semantic_expanded = semantic_vector.unsqueeze(1)  # [batch, 1, hidden_dim]
        cross_output, _ = self.cross_attn(x, semantic_expanded, semantic_expanded)
        x = self.norm2(x + cross_output)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)

        return x


class DiffusionDecoder(nn.Module):
    """
    Diffusion Decoder - 6-layer bidirectional transformer.

    Takes noisy tokens and semantic vectors as input, predicts clean tokens.
    Uses RoPE for position encoding and cross-attention for semantic conditioning.
    """

    def __init__(
        self,
        vocab_size: int = 50265,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        max_len: int = 64,
        dropout: float = 0.1,
        roberta_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Initialize diffusion decoder.

        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
            roberta_embeddings: Optional RoBERTa embeddings for initialization
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        if roberta_embeddings is not None:
            self.token_emb.weight.data.copy_(roberta_embeddings)

        # RoPE positional embeddings
        self.rotary_emb = RotaryEmbedding(hidden_dim // num_heads, max_position_embeddings=max_len)

        # Time embedding
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output projection to vocabulary
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Tie output weights with input embeddings (weight tying)
        self.output_proj.weight = self.token_emb.weight

        # Initialize output layer with RoBERTa embeddings if provided
        if roberta_embeddings is not None:
            self.output_proj.weight.data.copy_(roberta_embeddings)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        semantic_vector: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            semantic_vector: Semantic vector [batch, hidden_dim]
            timestep: Diffusion timestep [batch]

        Returns:
            logits: Predicted logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embed tokens
        x = self.token_emb(input_ids)  # [batch, seq_len, hidden_dim]
        x = self.dropout(x)

        # Add time embedding (broadcast to sequence length)
        time_emb = self.time_emb(timestep)  # [batch, hidden_dim]
        x = x + time_emb.unsqueeze(1)  # [batch, seq_len, hidden_dim]

        # Calculate RoPE embeddings for the current sequence length
        cos, sin = self.rotary_emb(x, seq_len)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, semantic_vector, cos, sin)

        # Project to vocabulary logits
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]

        return logits

    def get_num_params(self) -> int:
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing DiffusionDecoder components...")

    # Test RotaryEmbedding
    print("\n1. Testing RotaryEmbedding...")
    rope = RotaryEmbedding(dim=64, max_position_embeddings=64)
    print("[OK] RotaryEmbedding initialized")

    # Test DecoderLayer
    print("\n2. Testing DecoderLayer...")
    layer = DecoderLayer(hidden_dim=512, num_heads=8, ffn_dim=2048)
    print("[OK] DecoderLayer initialized")

    # Test forward pass
    batch_size = 2
    seq_len = 64
    x = torch.randn(batch_size, seq_len, 512)
    semantic_vector = torch.randn(batch_size, 512)
    
    # Need cos/sin for layer test
    rope = RotaryEmbedding(dim=64, max_position_embeddings=64)
    cos, sin = rope(x, seq_len)

    output = layer(x, semantic_vector, cos, sin)
    print(f"[OK] Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, 512)

    # Test DiffusionDecoder
    print("\n3. Testing DiffusionDecoder...")
    decoder = DiffusionDecoder(
        vocab_size=50265,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        max_len=64,
    )
    print("[OK] DiffusionDecoder initialized")

    # Count parameters
    num_params = decoder.get_num_params()
    num_trainable = decoder.get_num_trainable_params()
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")

    # Test forward pass
    input_ids = torch.randint(0, 50265, (batch_size, seq_len))
    timestep = torch.randint(0, 1000, (batch_size,))

    logits = decoder(input_ids, semantic_vector, timestep)
    print(f"[OK] Forward pass successful")
    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, 50265)

    print("\n[SUCCESS] All tests passed!")
