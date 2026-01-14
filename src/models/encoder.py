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
    ):
        """
        Initialize the semantic encoder.

        Args:
            model_name: HuggingFace model name (default: "roberta-base")
            hidden_dim: Output dimension for semantic vector (default: 512)
                        If set to 768, no adapter is used (faster training).
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

        # Optionally add adapter only if dimensions don't match
        # This allows users to skip the projection for faster training
        if hidden_dim != 768:
            # Learnable Adapter: 768 (RoBERTa) -> hidden_dim (decoder)
            # We use an MLP to better project the semantic space
            self.adapter = nn.Sequential(
                nn.Linear(768, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        else:
            # Identity layer - bypass projection for faster training
            self.adapter = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract semantic vector from input text.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            semantic_vector: Projected semantic vector [batch_size, hidden_dim]
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

        # Project to decoder dimension using the learnable adapter: [batch_size, hidden_dim]
        semantic_vector = self.adapter(mean_pooled)

        return semantic_vector

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

    # Test adapter is trainable
    for name, param in encoder.adapter.named_parameters():
        assert param.requires_grad == True, f"Adapter parameter {name} should be trainable"
    print("[OK] Adapter is trainable")

    print("\n[SUCCESS] All tests passed!")
