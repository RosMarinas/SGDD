"""
Test script to verify the optional adapter functionality in SemanticEncoder.

This script tests:
1. hidden_dim=512 (with adapter - default behavior)
2. hidden_dim=768 (without adapter - faster training)
3. Verify output shapes match expected dimensions
"""

import torch
import sys
sys.path.insert(0, 'src')

from models.encoder import SemanticEncoder

def test_encoder_with_adapter():
    """Test encoder with hidden_dim=512 (with adapter)"""
    print("[TEST 1] Testing encoder with adapter (hidden_dim=512)...")
    encoder = SemanticEncoder(model_name="roberta-base", hidden_dim=512)

    # Check that adapter is MLP (not Identity)
    assert not isinstance(encoder.adapter, torch.nn.Identity), "Adapter should be MLP for hidden_dim=512"
    print("  ✓ Adapter is MLP (as expected)")

    # Check adapter has trainable parameters
    adapter_params = sum(p.numel() for p in encoder.adapter.parameters() if p.requires_grad)
    assert adapter_params > 0, "Adapter should have trainable parameters"
    print(f"  ✓ Adapter has {adapter_params:,} trainable parameters")

    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 50265, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    output = encoder(input_ids, attention_mask)
    assert output.shape == (batch_size, 512), f"Expected shape (2, 512), got {output.shape}"
    print(f"  ✓ Output shape correct: {output.shape}")

    print("[SUCCESS] Test 1 passed!\n")


def test_encoder_without_adapter():
    """Test encoder with hidden_dim=768 (without adapter)"""
    print("[TEST 2] Testing encoder without adapter (hidden_dim=768)...")
    encoder = SemanticEncoder(model_name="roberta-base", hidden_dim=768)

    # Check that adapter is Identity
    assert isinstance(encoder.adapter, torch.nn.Identity), "Adapter should be Identity for hidden_dim=768"
    print("  ✓ Adapter is Identity (as expected)")

    # Check that Identity has no trainable parameters
    adapter_params = sum(p.numel() for p in encoder.adapter.parameters() if p.requires_grad)
    assert adapter_params == 0, "Identity adapter should have no trainable parameters"
    print(f"  ✓ Adapter has {adapter_params} trainable parameters (faster!)")

    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 50265, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    output = encoder(input_ids, attention_mask)
    assert output.shape == (batch_size, 768), f"Expected shape (2, 768), got {output.shape}"
    print(f"  ✓ Output shape correct: {output.shape}")

    print("[SUCCESS] Test 2 passed!\n")


def test_encoder_other_dimensions():
    """Test encoder with other dimensions (should use adapter)"""
    print("[TEST 3] Testing encoder with other dimensions (hidden_dim=256)...")

    encoder = SemanticEncoder(model_name="roberta-base", hidden_dim=256)

    # Check that adapter is MLP
    assert not isinstance(encoder.adapter, torch.nn.Identity), "Adapter should be MLP for hidden_dim=256"
    print("  ✓ Adapter is MLP (as expected)")

    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 50265, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    output = encoder(input_ids, attention_mask)
    assert output.shape == (batch_size, 256), f"Expected shape (2, 256), got {output.shape}"
    print(f"  ✓ Output shape correct: {output.shape}")

    print("[SUCCESS] Test 3 passed!\n")


def test_parameter_comparison():
    """Compare parameter counts with and without adapter"""
    print("[TEST 4] Comparing parameter counts...")

    encoder_with_adapter = SemanticEncoder(model_name="roberta-base", hidden_dim=512)
    encoder_without_adapter = SemanticEncoder(model_name="roberta-base", hidden_dim=768)

    params_with = sum(p.numel() for p in encoder_with_adapter.parameters() if p.requires_grad)
    params_without = sum(p.numel() for p in encoder_without_adapter.parameters() if p.requires_grad)

    print(f"  Trainable params with adapter (512dim): {params_with:,}")
    print(f"  Trainable params without adapter (768dim): {params_without:,}")

    # Without adapter should have significantly fewer trainable params
    assert params_without < params_with, "768dim version should have fewer trainable params"
    reduction = (params_with - params_without) / params_with * 100
    print(f"  ✓ Using 768dim reduces trainable params by {reduction:.1f}%")

    print("[SUCCESS] Test 4 passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Testing Optional Adapter Functionality")
    print("="*60 + "\n")

    try:
        test_encoder_with_adapter()
        test_encoder_without_adapter()
        test_encoder_other_dimensions()
        test_parameter_comparison()

        print("="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nSummary:")
        print("  ✓ hidden_dim=512: Uses MLP adapter (768→512)")
        print("  ✓ hidden_dim=768: Uses Identity (bypasses adapter)")
        print("  ✓ Other dimensions: Use MLP adapter")
        print("  ✓ Users can now choose faster training by setting hidden_dim=768")
        print("\nRecommendation:")
        print("  For faster training, set hidden_dim=768 in SGDDConfig")
        print("  and ensure decoder also uses hidden_dim=768")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
