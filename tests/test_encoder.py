"""
Unit tests for SemanticEncoder
"""

import pytest
import torch
from src.models.encoder import SemanticEncoder


class TestSemanticEncoder:
    """Test suite for SemanticEncoder"""

    @pytest.fixture
    def encoder(self):
        """Create a semantic encoder for testing"""
        return SemanticEncoder(model_name="roberta-base", hidden_dim=512)

    def test_output_shape(self, encoder):
        """Test that encoder produces correct output shape"""
        batch_size = 4
        seq_len = 64
        input_ids = torch.randint(0, 50265, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        semantic_vector = encoder(input_ids, attention_mask)

        assert semantic_vector.shape == (batch_size, 512), \
            f"Expected shape ({batch_size}, 512), got {semantic_vector.shape}"

    def test_frozen_parameters(self, encoder):
        """Test that RoBERTa parameters are frozen"""
        for name, param in encoder.roberta.named_parameters():
            assert param.requires_grad == False, \
                f"Parameter {name} should be frozen but is trainable"

    def test_projection_layer_trainable(self, encoder):
        """Test that projection layer is trainable"""
        assert encoder.proj.weight.requires_grad == True, \
            "Projection layer should be trainable"

    def test_forward_with_different_batch_sizes(self, encoder):
        """Test forward pass with different batch sizes"""
        seq_len = 64

        for batch_size in [1, 2, 8, 16]:
            input_ids = torch.randint(0, 50265, (batch_size, seq_len))
            attention_mask = torch.ones_like(input_ids)

            semantic_vector = encoder(input_ids, attention_mask)

            assert semantic_vector.shape == (batch_size, 512), \
                f"Failed for batch_size={batch_size}"

    def test_forward_with_different_seq_lengths(self, encoder):
        """Test forward pass with different sequence lengths"""
        batch_size = 4

        for seq_len in [32, 64, 128]:
            input_ids = torch.randint(0, 50265, (batch_size, seq_len))
            attention_mask = torch.ones_like(input_ids)

            semantic_vector = encoder(input_ids, attention_mask)

            assert semantic_vector.shape == (batch_size, 512), \
                f"Failed for seq_len={seq_len}"

    def test_mean_pooling_with_padding(self, encoder):
        """Test that mean pooling correctly handles padding"""
        batch_size = 2
        seq_len = 64

        # Create input with different actual lengths
        input_ids = torch.randint(0, 50265, (batch_size, seq_len))

        # First sample: all tokens are valid
        # Second sample: only half are valid
        attention_mask = torch.ones_like(input_ids)
        attention_mask[1, 32:] = 0

        semantic_vector = encoder(input_ids, attention_mask)

        assert semantic_vector.shape == (batch_size, 512)
        # Vectors should be different due to different pooling
        assert not torch.allclose(semantic_vector[0], semantic_vector[1])

    def test_encode_text_convenience_method(self, encoder):
        """Test the convenience method for encoding text"""
        text = "This is a test sentence."
        semantic_vector = encoder.encode_text(text, max_length=64)

        assert semantic_vector.shape == (1, 512)

    def test_device_compatibility(self, encoder):
        """Test that encoder works on different devices"""
        if torch.cuda.is_available():
            encoder = encoder.cuda()
            input_ids = torch.randint(0, 50265, (2, 64)).cuda()
            attention_mask = torch.ones_like(input_ids)

            semantic_vector = encoder(input_ids, attention_mask)

            assert semantic_vector.shape == (2, 512)
            assert semantic_vector.device.type == "cuda"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
