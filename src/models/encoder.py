"""
Semantic Encoder - Frozen Encoder (BGE-M3 / RoBERTa) for extracting semantic vectors.

This module implements a frozen semantic encoder that uses a pre-trained model (default: BGE-M3)
to extract deep semantic representations from input text via mean pooling.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import os
import warnings

class SemanticEncoder(nn.Module):
    """
    Frozen semantic encoder using BGE-M3 (default).

    Extracts semantic vectors from input text using mean pooling over
    the final hidden states, then optionally projects to the decoder dimension.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        hidden_dim: int = 512,
        kl_weight: float = 0.001,
        kl_anneal_steps: int = 10000,
        kl_threshold: float = 0.0,
        whitening_stats_path: str = None,
    ):
        """
        Initialize the semantic encoder with a dual-path VIB architecture.

        Args:
            model_name: HuggingFace model name (default: "BAAI/bge-m3")
            hidden_dim: Output dimension for semantic vector (default: 512)
            kl_weight: Weight for KL divergence loss
            kl_anneal_steps: Number of steps for KL annealing
            kl_threshold: Minimum KL value (free bits)
            whitening_stats_path: Path to whitening statistics file (optional)
        """
        super().__init__()

        # Load Pre-trained Model
        print(f"Loading encoder model: {model_name}...")
        self.model = AutoModel.from_pretrained(model_name)

        # FREEZE all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Store tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Determine input dimension from model config
        self.input_dim = self.model.config.hidden_size
        self.hidden_dim = hidden_dim
        print(f"Encoder Input Dim: {self.input_dim}, Output Dim: {hidden_dim}")

        # VIB parameters
        self.kl_weight = kl_weight
        self.kl_anneal_steps = kl_anneal_steps
        self.kl_threshold = kl_threshold
        self._current_step = 0

        # --- Whitening (Optimization) ---
        # Only enable whitening if stats match dimensions
        self.use_whitening = False
        
        if whitening_stats_path and os.path.exists(whitening_stats_path):
            print(f"Loading whitening stats from {whitening_stats_path}...")
            stats = torch.load(whitening_stats_path, map_location="cpu")
            
            # Check dimension match
            stats_dim = stats["mean"].shape[0]
            if stats_dim != self.input_dim:
                warnings.warn(
                    f"Whitening stats dimension ({stats_dim}) does not match "
                    f"model dimension ({self.input_dim}). Whitening disabled."
                )
            else:
                self.register_buffer("whitening_mean", stats["mean"])
                self.register_buffer("whitening_matrix", stats["whitening_matrix"])
                self.use_whitening = True
                print("Whitening enabled.")

        # --- Dual-Path Architecture ---

        # 1. Direct Path: Stable signal for the decoder
        self.adapter = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. VIB Path: Regularization and stochasticity
        self.mu_layer = nn.Linear(self.input_dim, hidden_dim)
        self.logvar_layer = nn.Linear(self.input_dim, hidden_dim)

        # --- Initialization ---
        self._init_weights()

    def _init_weights(self):
        # Initialize adapter
        for layer in self.adapter:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize mu layer near zero
        nn.init.xavier_uniform_(self.mu_layer.weight, gain=0.01)
        nn.init.zeros_(self.mu_layer.bias)
        
        # Initialize logvar for very small initial sigma
        nn.init.xavier_uniform_(self.logvar_layer.weight, gain=0.001)
        nn.init.constant_(self.logvar_layer.bias, -10.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_kl: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        last_hidden_state = outputs.last_hidden_state

        # Use CLS token (first token) for BGE-M3 / dense retrieval models
        # This preserves semantic meaning better than mean pooling for these models
        pooled_output = last_hidden_state[:, 0, :]

        # Whitening
        if self.use_whitening:
            pooled_output = torch.matmul(pooled_output - self.whitening_mean, self.whitening_matrix.t())

        # Dual-Path
        direct_signal = self.adapter(pooled_output)
        mu = self.mu_layer(pooled_output)
        logvar = self.logvar_layer(pooled_output)
        logvar = torch.clamp(logvar, min=-10, max=2)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            vib_signal = mu + std * eps
        else:
            vib_signal = mu

        z = direct_signal + vib_signal
        
        # KL Divergence
        current_kl_weight = self.kl_weight
        if self.training:
            anneal_ratio = min(1.0, self._current_step / self.kl_anneal_steps)
            current_kl_weight *= anneal_ratio

        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        if self.kl_threshold > 0:
            kl_per_sample = torch.max(kl_per_sample, torch.tensor(self.kl_threshold, device=kl_per_sample.device))

        kl_weighted = kl_per_sample * current_kl_weight

        if return_kl:
            return z, kl_weighted
        return z

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer

    def encode_text(self, text: str, max_length: int = 64) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)

        with torch.no_grad():
            semantic_vector = self.forward(input_ids, attention_mask)
        return semantic_vector

    def increment_step(self):
        if self.training:
            self._current_step += 1

if __name__ == "__main__":
    print("Testing SemanticEncoder with BGE-M3...")
    # NOTE: This might fail if BAAI/bge-m3 is not cached and no internet, but usually safe
    try:
        encoder = SemanticEncoder(model_name="BAAI/bge-m3", hidden_dim=512)
        print("[OK] Encoder initialized with BGE-M3")
        
        batch_size = 2
        seq_len = 32
        # XLM-R vocab size is larger, max ID is 250001
        input_ids = torch.randint(0, 250002, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        z = encoder(input_ids, attention_mask)
        print(f"[OK] Forward pass output shape: {z.shape}")
        assert z.shape == (batch_size, 512)
        
    except Exception as e:
        print(f"Skipping BGE-M3 test due to load error (network?): {e}")
        print("Falling back to roberta-base test")
        encoder = SemanticEncoder(model_name="roberta-base", hidden_dim=512)
        input_ids = torch.randint(0, 50265, (2, 32))
        attention_mask = torch.ones_like(input_ids)
        z = encoder(input_ids, attention_mask)
        print(f"[OK] RoBERTa Forward pass output shape: {z.shape}")