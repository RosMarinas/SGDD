"""
Semantic Encoder - Frozen RoBERTa-base for extracting semantic vectors.

This module implements a frozen semantic encoder that uses RoBERTa-base
to extract deep semantic representations from input text via mean pooling.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SemanticEncoder(nn.Module):
    """
    Frozen semantic encoder using RoBERTa-base.

    Extracts semantic vectors from input text using mean pooling over
    the final hidden states, then optionally projects to the decoder dimension.

    If hidden_dim == 768 (RoBERTa's output dimension), the adapter is bypassed
    for faster training. Otherwise, a learnable adapter is used to project
    from 768 to the target dimension.

    Attributes:
        roberta: Frozen RoBERTa-base model
        adapter: Optional linear projection layer (768 -> hidden_dim)
        tokenizer: RoBERTa tokenizer (for convenience)
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        hidden_dim: int = 512,
        kl_weight: float = 0.001,
        kl_anneal_steps: int = 10000,
        kl_threshold: float = 0.0,
    ):
        """
        Initialize the semantic encoder with Variational Information Bottleneck.

        Args:
            model_name: HuggingFace model name (default: "roberta-base")
            hidden_dim: Output dimension for semantic vector (default: 512)
            kl_weight: Weight for KL divergence loss (default: 0.001)
            kl_anneal_steps: Number of steps for KL annealing (default: 10000)
            kl_threshold: Minimum KL value (free bits) to prevent posterior collapse (default: 0.0)
        """
        super().__init__()

        # Load RoBERTa from HuggingFace
        self.roberta = AutoModel.from_pretrained(model_name)

        # FREEZE all parameters - critical for preserving pretrained knowledge
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Store tokenizer for convenience
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Store dimensions
        self.hidden_dim = hidden_dim
        self.output_dim = 768  # RoBERTa hidden size

        # VIB parameters
        self.kl_weight = kl_weight
        self.kl_anneal_steps = kl_anneal_steps
        self.kl_threshold = kl_threshold
        self._current_step = 0  # For KL annealing

        # Variational Information Bottleneck
        # Two separate projections for mu and logvar
        # Using BatchNorm1d instead of LayerNorm to reduce anisotropy (Cone Effect)
        self.mu_layer = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.logvar_layer = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        # Initialize logvar to small values (near-deterministic start)
        # This prevents posterior collapse at the beginning of training
        # CRITICAL FIX: Initialize bias to -10 to make sigma very small (~0.0067)
        # This ensures the signal is not drowned in noise at the start
        for layer in self.logvar_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.001)
                nn.init.constant_(layer.bias, -10.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_kl: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Extract semantic vector using Variational Information Bottleneck.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_kl: If True, return (z, kl_loss)

        Returns:
            semantic_vector: Latent vector z ~ N(mu, sigma^2) [batch_size, hidden_dim]
            kl_loss: KL divergence per sample [batch_size] (only if return_kl=True)
        """
        # Extract RoBERTa representations
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get last hidden state: [batch_size, seq_len, 768]
        last_hidden_state = outputs.last_hidden_state

        # Mean pooling over sequence length
        # Expand attention_mask for broadcasting: [batch_size, seq_len, 1]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        # Sum hidden states weighted by mask
        sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)

        # Count actual tokens (not padding)
        token_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        # Compute mean: [batch_size, 768]
        mean_pooled = sum_hidden / token_counts

        # Variational Information Bottleneck
        mu = self.mu_layer(mean_pooled)  # [batch_size, hidden_dim]
        logvar = self.logvar_layer(mean_pooled)  # [batch_size, hidden_dim]

        # Clamp logvar for stability (prevent extreme values)
        logvar = torch.clamp(logvar, min=-10, max=2)

        # KL annealing: gradually increase KL weight over training
        if self.training:
            anneal_ratio = min(1.0, self._current_step / self.kl_anneal_steps)
            current_kl_weight = self.kl_weight * anneal_ratio
        else:
            current_kl_weight = self.kl_weight

        # Reparameterization trick: z = mu + sigma * epsilon
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps  # [batch_size, hidden_dim]
        else:
            # Inference: use mu (deterministic)
            z = mu

        # KL divergence: D_KL(N(mu, sigma^2) || N(0, 1))
        # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # Apply KL threshold (free bits)
        if self.kl_threshold > 0:
            kl_per_sample = torch.max(kl_per_sample, torch.tensor(self.kl_threshold, device=kl_per_sample.device))

        kl_weighted = kl_per_sample * current_kl_weight  # [batch_size]

        if return_kl:
            return z, kl_weighted
        return z

    def get_tokenizer(self) -> AutoTokenizer:
        """Return the tokenizer for external use."""
        return self.tokenizer

    def encode_text(self, text: str, max_length: int = 64) -> torch.Tensor:
        """
        Convenience method to encode a single text string.

        Args:
            text: Input text string
            max_length: Maximum sequence length

        Returns:
            semantic_vector: [1, hidden_dim]
        """
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Move to same device as model
        input_ids = encoded["input_ids"].to(self.roberta.device)
        attention_mask = encoded["attention_mask"].to(self.roberta.device)

        # Extract semantic vector
        with torch.no_grad():
            semantic_vector = self.forward(input_ids, attention_mask)

        return semantic_vector

    def encode_from_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode from token IDs (alias for forward method).

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            semantic_vector: [batch_size, hidden_dim]
        """
        return self.forward(input_ids, attention_mask)

    def increment_step(self):
        """Increment step counter for KL annealing (call each training step)"""
        if self.training:
            self._current_step += 1


if __name__ == "__main__":
    # Quick test
    print("Testing SemanticEncoder...")

    encoder = SemanticEncoder(model_name="roberta-base", hidden_dim=512)
    print("[OK] Encoder initialized")

    # Test forward pass
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, 50265, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    semantic_vector = encoder(input_ids, attention_mask)
    print("[OK] Forward pass successful")
    print(f"  Output shape: {semantic_vector.shape}")
    print(f"  Expected shape: ({batch_size}, 512)")

    assert semantic_vector.shape == (batch_size, 512), "Output shape mismatch!"
    print("[OK] Shape test passed")

    # Test frozen parameters
    for name, param in encoder.roberta.named_parameters():
        assert param.requires_grad == False, f"Parameter {name} is not frozen!"
    print("[OK] All RoBERTa parameters are frozen")

    # Test projection layers are trainable
    for name, param in encoder.mu_layer.named_parameters():
        assert param.requires_grad == True, f"Mu layer parameter {name} should be trainable"
    print("[OK] Projection layers are trainable")

    print("\n[SUCCESS] All tests passed!")
