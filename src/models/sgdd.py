"""
SGDD Model - Full Semantic-Guided Discrete Diffusion Model

Combines frozen semantic encoder, diffusion process, and trainable decoder
into a complete model for text generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

from .encoder import SemanticEncoder
from .decoder import DiffusionDecoder
from .diffusion import CosineNoiseSchedule, DiscreteDiffusion


@dataclass
class SGDDConfig:
    """Configuration for SGDD model"""
    # Encoder
    encoder_model: str = "roberta-base"
    hidden_dim: int = 512

    # Decoder
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 2048
    max_len: int = 64
    dropout: float = 0.1

    # Diffusion
    num_diffusion_steps: int = 1000
    vocab_size: int = 50265

    # Training
    cfg_prob: float = 0.1  # Probability of unconditional training (CFG)


class SGDDModel(nn.Module):
    """
    Semantic-Guided Discrete Diffusion Model.

    Combines:
    - Frozen semantic encoder (RoBERTa)
    - Noise schedule (cosine)
    - Trainable diffusion decoder

    Supports classifier-free guidance (CFG) for better semantic conditioning.
    """

    def __init__(self, config: SGDDConfig):
        """
        Initialize SGDD model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Semantic encoder (frozen)
        self.encoder = SemanticEncoder(
            model_name=config.encoder_model,
            hidden_dim=config.hidden_dim,
        )

        # Get RoBERTa embeddings for decoder initialization
        # Note: RoBERTa embeddings are 768-dim, need to project to hidden_dim
        # We'll use the first hidden_dim dimensions as initialization
        roberta = self.encoder.roberta
        roberta_embeddings = roberta.embeddings.word_embeddings.weight.data
        # Project from 768 to hidden_dim (512) by taking first 512 dimensions
        roberta_embeddings_proj = roberta_embeddings[:, :config.hidden_dim]
        # This is a simplification - in practice, you might want a proper projection

        # Diffusion decoder (trainable)
        self.decoder = DiffusionDecoder(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            max_len=config.max_len,
            dropout=config.dropout,
            roberta_embeddings=roberta_embeddings_proj,
        )

        # Noise schedule
        self.noise_schedule = CosineNoiseSchedule(
            num_timesteps=config.num_diffusion_steps
        )

        # Diffusion process
        self.diffusion = DiscreteDiffusion(
            noise_schedule=self.noise_schedule,
            num_classes=config.vocab_size,
        )

        # Tokenizer from encoder
        self.tokenizer = self.encoder.tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        cfg_uncond: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            timestep: Diffusion timestep [batch] (if None, sample random)
            cfg_uncond: If True, randomly drop semantic vector with cfg_prob for CFG training.
                        This is the standard classifier-free guidance training strategy,
                        NOT forcing all samples to be unconditional.

        Returns:
            logits: Predicted logits [batch, seq_len, vocab_size]
            target_tokens: Target tokens for loss [batch, seq_len]
            loss_mask: Loss mask [batch, seq_len]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Sample random timestep if not provided
        if timestep is None:
            timestep = torch.rand(batch_size, device=device)
            # Convert to discrete timestep
            timestep = (timestep * self.noise_schedule.num_timesteps).long()

        # Extract semantic vector from encoder
        semantic_vector = self.encoder(input_ids, attention_mask)
        # [batch, hidden_dim]

        # Apply classifier-free guidance during training
        if self.training and cfg_uncond:
            # Randomly select samples to make unconditional
            uncond_mask = torch.rand(batch_size, device=device) < self.config.cfg_prob
            semantic_vector = torch.where(
                uncond_mask.unsqueeze(1),
                torch.zeros_like(semantic_vector),
                semantic_vector
            )

        # Forward diffusion: add noise to input tokens
        x_start = input_ids.clone()
        x_t = self.diffusion.q_sample(x_start, timestep)
        # [batch, seq_len]

        # Decoder: predict clean tokens from noisy tokens
        logits = self.decoder(x_t, semantic_vector, timestep)
        # [batch, seq_len, vocab_size]

        # Compute loss mask (only on tokens that were noised)
        loss_mask = self.diffusion.get_loss_weights(x_start, x_t, timestep)
        # [batch, seq_len]

        return logits, x_start, loss_mask

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_tokens: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked cross-entropy loss.

        Args:
            logits: Predicted logits [batch, seq_len, vocab_size]
            target_tokens: Target tokens [batch, seq_len]
            loss_mask: Loss mask [batch, seq_len]

        Returns:
            loss: Scalar loss
        """
        # Compute cross-entropy loss for each position
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            target_tokens.view(-1),
            reduction='none'
        ).view(logits.shape[:2])
        # [batch, seq_len]

        # Apply loss mask (only compute loss on noised tokens)
        loss = (loss * loss_mask).sum() / loss_mask.sum()

        return loss

    @torch.no_grad()
    def generate(
        self,
        input_text: str,
        num_steps: int = 16,
        guidance_scale: float = 2.0,
        temperature: float = 1.0,
        max_length: int = 64,
    ) -> str:
        """
        Generate text from semantic vector using MaskGIT-style decoding.

        Args:
            input_text: Input text to encode
            num_steps: Number of decoding steps
            guidance_scale: CFG guidance scale
            temperature: Sampling temperature
            max_length: Maximum generation length

        Returns:
            generated_text: Generated text
        """
        self.eval()
        device = next(self.parameters()).device

        # Encode input text
        encoded = self.tokenizer(
            input_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Get semantic vector (conditional)
        semantic_vector_cond = self.encoder(input_ids, attention_mask)
        # [1, hidden_dim]

        # Start from all MASK tokens
        mask_token_id = self.tokenizer.mask_token_id
        current_tokens = torch.full(
            (1, max_length),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )
        # [1, max_length]

        # MaskGIT iterative decoding
        timesteps = torch.linspace(0, 1, num_steps + 1)[1:]
        # [num_steps]

        for step_idx, t in enumerate(timesteps):
            # Convert continuous time to discrete timestep
            timestep = torch.tensor(
                [int(t * self.noise_schedule.num_timesteps)],
                device=device,
            )

            # Number of tokens to unmask at this step
            # Cosine schedule: more tokens in early steps
            num_masked = (current_tokens == mask_token_id).sum().item()
            num_to_unmask = int((1 - t) * num_masked / 2)
            num_to_unmask = max(1, num_to_unmask)

            # Conditional prediction (with semantic vector)
            logits_cond = self.decoder(current_tokens, semantic_vector_cond, timestep)
            probs_cond = F.softmax(logits_cond / temperature, dim=-1)
            # [1, seq_len, vocab_size]

            # Unconditional prediction (without semantic vector)
            semantic_vector_uncond = torch.zeros_like(semantic_vector_cond)
            logits_uncond = self.decoder(current_tokens, semantic_vector_uncond, timestep)
            probs_uncond = F.softmax(logits_uncond / temperature, dim=-1)
            # [1, seq_len, vocab_size]

            # Classifier-free guidance
            probs = probs_cond + guidance_scale * (probs_cond - probs_uncond)
            # [1, seq_len, vocab_size]

            # Sample tokens for masked positions
            masked_positions = (current_tokens == mask_token_id)
            # [1, seq_len]

            if masked_positions.sum() == 0:
                # No masked tokens left, break
                break

            # Get probabilities at masked positions
            # Need to index properly: probs is [1, seq_len, vocab_size]
            probs_at_masked = probs[0][masked_positions[0]]
            # [num_masked, vocab_size]

            # Clamp probabilities to avoid numerical issues (untrained model)
            probs_at_masked = torch.clamp(probs_at_masked, min=0, max=1)
            # Renormalize
            probs_at_masked = probs_at_masked / probs_at_masked.sum(dim=-1, keepdim=True)

            # Sample from predicted distribution
            sampled_tokens = torch.multinomial(probs_at_masked, num_samples=1).squeeze(-1)
            # [num_masked]

            # Confidence-based token selection
            # Use max probability as confidence
            confidence = probs_at_masked.max(dim=-1)[0]
            # [num_masked]

            # Select top-k most confident positions to fill
            if num_to_unmask < len(sampled_tokens):
                _, topk_indices = torch.topk(confidence, num_to_unmask)
                tokens_to_fill = sampled_tokens[topk_indices]

                # Update only selected positions
                mask_indices = torch.where(masked_positions[0])[0]
                positions_to_update = mask_indices[topk_indices]
                current_tokens[0, positions_to_update] = tokens_to_fill
            else:
                # Fill all masked positions
                mask_indices = torch.where(masked_positions[0])[0]
                current_tokens[0, mask_indices] = sampled_tokens

        # Decode tokens to text
        generated_ids = current_tokens[0].cpu().numpy()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def get_num_params(self) -> int:
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing SGDDModel...")

    config = SGDDConfig()
    model = SGDDModel(config)
    print("[OK] Model initialized")

    # Count parameters
    total_params = model.get_num_params()
    trainable_params = model.get_num_trainable_params()
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 50265, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    logits, target_tokens, loss_mask = model(input_ids, attention_mask, cfg_uncond=True)
    print(f"[OK] Forward pass successful")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Target tokens shape: {target_tokens.shape}")
    print(f"  Loss mask shape: {loss_mask.shape}")

    # Test loss computation
    loss = model.compute_loss(logits, target_tokens, loss_mask)
    print(f"[OK] Loss computed: {loss.item():.4f}")

    # Test generation (simple)
    model.eval()
    with torch.no_grad():
        generated = model.generate("Hello world", num_steps=4, max_length=32)
    print(f"[OK] Generation successful")
    print(f"  Generated: {generated}")

    print("\n[SUCCESS] All tests passed!")
