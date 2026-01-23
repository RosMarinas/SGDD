"""
Semantic Encoder - Frozen RoBERTa-base for extracting semantic vectors.

This module implements a frozen semantic encoder that uses RoBERTa-base
to extract deep semantic representations from input text via mean pooling.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import os


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
        whitening_stats_path: str = None,
    ):
        """
        Initialize the semantic encoder with a dual-path VIB architecture.

        This architecture uses a stable direct path for the main semantic signal
        and a parallel VIB path for regularization, preventing KL collapse.

        Args:
            model_name: HuggingFace model name (default: "roberta-base")
            hidden_dim: Output dimension for semantic vector (default: 512)
            kl_weight: Weight for KL divergence loss (default: 0.001)
            kl_anneal_steps: Number of steps for KL annealing (default: 10000)
            kl_threshold: Minimum KL value (free bits)
            whitening_stats_path: Path to whitening statistics file (optional)
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

        # --- Whitening (Optimization) ---
        self.register_buffer("whitening_mean", torch.zeros(768))
        self.register_buffer("whitening_matrix", torch.eye(768))
        self.use_whitening = False

        if whitening_stats_path and os.path.exists(whitening_stats_path):
            print(f"Loading whitening stats from {whitening_stats_path}...")
            stats = torch.load(whitening_stats_path, map_location="cpu")
            self.whitening_mean.copy_(stats["mean"])
            self.whitening_matrix.copy_(stats["whitening_matrix"])
            self.use_whitening = True
            print("Whitening enabled.")

        # --- Dual-Path Architecture ---

        # 1. Direct Path: Stable signal for the decoder
        self.adapter = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. VIB Path: Regularization and stochasticity
        self.mu_layer = nn.Linear(768, hidden_dim)
        self.logvar_layer = nn.Linear(768, hidden_dim)

        # --- Initialization ---
        
        # Initialize adapter for good initial signal
        for layer in self.adapter:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize mu layer near zero
        nn.init.xavier_uniform_(self.mu_layer.weight, gain=0.01)
        nn.init.zeros_(self.mu_layer.bias)
        
        # Initialize logvar for very small initial sigma to prevent noise overwhelming signal
        # CRITICAL FIX: Initialize bias to -10 to make sigma very small (~0.0067)
        nn.init.xavier_uniform_(self.logvar_layer.weight, gain=0.001)
        nn.init.constant_(self.logvar_layer.bias, -10.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_kl: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Extract semantic vector using the dual-path VIB architecture.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_kl: If True, return (z, kl_loss)

        Returns:
            semantic_vector: Latent vector z [batch_size, hidden_dim]
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
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)
        token_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_hidden / token_counts # [batch_size, 768]

        # --- Whitening ---
        if self.use_whitening:
            # Apply (x - mu) W^T
            # Note: whitening_matrix is W, we use linear transformation convention x @ W.T
            mean_pooled = torch.matmul(mean_pooled - self.whitening_mean, self.whitening_matrix.t())

        # --- Dual-Path Forward ---

        # 1. Direct path signal
        direct_signal = self.adapter(mean_pooled)

        # 2. VIB path for regularization
        mu = self.mu_layer(mean_pooled)
        logvar = self.logvar_layer(mean_pooled)
        logvar = torch.clamp(logvar, min=-10, max=2) # Stability

        # Reparameterization trick for the VIB path
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            vib_signal = mu + std * eps
        else:
            # At inference, use mu from VIB path for determinism
            vib_signal = mu

        # Combine signals: z = direct_path + vib_path
        # The direct path provides a stable signal, while the VIB path adds
        # learned, structured noise, guided by the KL loss.
        z = direct_signal + vib_signal
        
        # --- KL Divergence Calculation ---

        # KL annealing
        current_kl_weight = self.kl_weight
        if self.training:
            anneal_ratio = min(1.0, self._current_step / self.kl_anneal_steps)
            current_kl_weight *= anneal_ratio

        # D_KL(N(mu, sigma^2) || N(0, 1))
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # Apply KL threshold (free bits)
        if self.kl_threshold > 0:
            kl_per_sample = torch.max(kl_per_sample, torch.tensor(self.kl_threshold, device=kl_per_sample.device))

        kl_weighted = kl_per_sample * current_kl_weight

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
    print("Testing SemanticEncoder (Dual-Path VIB)...")

    encoder = SemanticEncoder(model_name="roberta-base", hidden_dim=512, kl_threshold=2.0)
    print("[OK] Encoder initialized")

    # Test forward pass
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, 50265, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    encoder.train() # Test training mode
    z_train, kl_loss = encoder(input_ids, attention_mask, return_kl=True)
    print("\n--- Training Mode ---")
    print("[OK] Forward pass successful")
    print(f"  Output shape (z): {z_train.shape}")
    print(f"  KL Loss shape:    {kl_loss.shape}")
    assert z_train.shape == (batch_size, 512), "Training output shape mismatch!"
    assert kl_loss.shape == (batch_size,), "KL loss shape mismatch!"

    encoder.eval() # Test eval mode
    z_eval = encoder(input_ids, attention_mask)
    print("\n--- Eval Mode ---")
    print("[OK] Forward pass successful")
    print(f"  Output shape (z): {z_eval.shape}")
    assert z_eval.shape == (batch_size, 512), "Eval output shape mismatch!"

    # Test frozen parameters
    for name, param in encoder.roberta.named_parameters():
        assert not param.requires_grad, f"Parameter {name} is not frozen!"
    print("\n[OK] All RoBERTa parameters are frozen")

    # Test trainable parameters
    assert all(p.requires_grad for p in encoder.adapter.parameters()), "Adapter should be trainable"
    assert all(p.requires_grad for p in encoder.mu_layer.parameters()), "Mu layer should be trainable"
    assert all(p.requires_grad for p in encoder.logvar_layer.parameters()), "Logvar layer should be trainable"
    print("[OK] All projection layers are trainable")

    print("\n[SUCCESS] All tests passed!")
