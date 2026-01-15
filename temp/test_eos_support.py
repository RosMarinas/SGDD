"""Test EOS token support with checkpoint 4

This script tests whether the model can generate EOS tokens properly
after enabling compute_eos_loss support.
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from models.sgdd import SGDDModel, SGDDConfig
from transformers import RobertaTokenizer


def test_eos_generation():
    """Test that model generates EOS tokens"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = "checkpoints/4/best_model.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please ensure checkpoint 4 exists before running this test.")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint if available
    if "config" in checkpoint:
        checkpoint_config_dict = checkpoint["config"]
        # Extract model config
        model_config = checkpoint_config_dict.get("model", {})

        # Map config keys to SGDDConfig fields
        config_mapping = {
            "encoder_name": "encoder_model",
            "semantic_dim": "hidden_dim",
            "max_length": "max_len",
            # Keep other keys as-is
        }

        valid_keys = {}
        for k, v in model_config.items():
            # Use mapped key or original key
            config_key = config_mapping.get(k, k)
            # Check if this is a valid SGDDConfig field
            if config_key in [f.name for f in SGDDConfig.__dataclass_fields__.values()]:
                valid_keys[config_key] = v

        print(f"Loaded config from checkpoint with {len(valid_keys)} valid keys")
        print(f"Config: {valid_keys}")
    else:
        valid_keys = {}
        print("No config in checkpoint, using defaults")

    # Create model config with EOS support enabled
    # Start with checkpoint config, then override with EOS settings
    config_dict = {
        "compute_eos_loss": True,  # Enable EOS support
        "eos_token_id": 2,
    }
    config_dict.update(valid_keys)

    config = SGDDConfig(**config_dict)

    # Create model
    print("Creating model...")
    model = SGDDModel(config).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded successfully")
    print(f"EOS token ID: {config.eos_token_id}")
    print(f"Compute EOS loss: {config.compute_eos_loss}")

    # Test cases
    test_texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming artificial intelligence",
        "Artificial intelligence will change the world",
        "This is a test of the emergency broadcast system",
    ]

    eos_count = 0
    total_tests = len(test_texts)
    results = []

    print(f"\n{'='*80}")
    print(f"Running {total_tests} tests...")
    print(f"{'='*80}\n")

    for i, text in enumerate(test_texts, 1):
        print(f"[Test {i}/{total_tests}]")
        print(f"Input: {text}")

        # Get original token count (without special tokens)
        original_tokens = len(model.tokenizer.encode(text, add_special_tokens=False))
        print(f"Original tokens: {original_tokens}")

        # Generate
        with torch.no_grad():
            generated = model.generate(
                text,
                num_steps=16,
                guidance_scale=2.0,
                max_length=64,
                temperature=1.0,
            )

        print(f"Output: {generated}")

        # Get generated token count
        generated_tokens = len(model.tokenizer.encode(generated, add_special_tokens=False))
        print(f"Generated tokens: {generated_tokens}")

        # Check if output was truncated (indicates EOS generation)
        if generated_tokens < 64:  # max_length used in generate
            print(f"[+] EOS likely generated (output < max_length)")
            eos_count += 1
            result = "EOS"
        else:
            print(f"[-] No EOS generated (output = max_length)")
            result = "No EOS"

        results.append({
            "input": text,
            "output": generated,
            "original_tokens": original_tokens,
            "generated_tokens": generated_tokens,
            "eos_generated": result
        })
        print()

    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"EOS Generation Rate: {eos_count}/{total_tests} ({eos_count/total_tests*100:.1f}%)")
    print(f"{'='*80}\n")

    # Detailed results
    print("Detailed Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['input'][:50]}...")
        print(f"   Original: {result['original_tokens']} tokens")
        print(f"   Generated: {result['generated_tokens']} tokens")
        print(f"   EOS: {result['eos_generated']}")
        print(f"   Output: {result['output'][:100]}...")

    # Analysis
    print(f"\n{'='*80}")
    print("Analysis:")
    if eos_count == 0:
        print("- Model is NOT generating EOS tokens")
        print("- This is expected if the model was trained WITHOUT compute_eos_loss")
        print("- To enable EOS generation, resume training with compute_eos_loss=True")
    elif eos_count < total_tests // 2:
        print(f"- Model is generating EOS tokens in {eos_count}/{total_tests} cases")
        print("- Partial EOS learning - may need more training")
    else:
        print(f"- Model is generating EOS tokens in {eos_count}/{total_tests} cases")
        print("- Good EOS generation capability!")
    print(f"{'='*80}")


def test_tokenizer_behavior():
    """Test how RoBERTa tokenizer handles EOS tokens"""
    print("\n" + "="*80)
    print("Testing RoBERTa Tokenizer Behavior")
    print("="*80 + "\n")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    test_text = "Hello world"
    print(f"Input text: '{test_text}'")

    # Tokenize
    encoded = tokenizer(
        test_text,
        max_length=20,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)

    print(f"Input IDs: {input_ids.tolist()}")
    print(f"Attention mask: {attention_mask.tolist()}")

    # Decode special tokens
    print(f"\nToken breakdown:")
    for i, (token_id, mask) in enumerate(zip(input_ids.tolist(), attention_mask.tolist())):
        if mask == 1:
            token = tokenizer.decode([token_id])
            special_token = tokenizer.convert_ids_to_tokens(token_id)
            # Handle potential unicode issues
            try:
                print(f"  Position {i}: ID={token_id}, Token='{token}', Special='{special_token}'")
            except UnicodeEncodeError:
                print(f"  Position {i}: ID={token_id}, Token='{token}', Special=[Unicode token]")
        else:
            print(f"  Position {i}: ID={token_id}, Token='<PAD>', Padding")

    print(f"\nEOS token ID: {tokenizer.eos_token_id} ('{tokenizer.eos_token}')")
    print(f"PAD token ID: {tokenizer.pad_token_id} ('{tokenizer.pad_token}')")
    print(f"BOS token ID: {tokenizer.bos_token_id} ('{tokenizer.bos_token}')")


if __name__ == "__main__":
    print("SGDD EOS Token Support Test")
    print("="*80)

    # First test tokenizer behavior
    test_tokenizer_behavior()

    # Then test EOS generation
    test_eos_generation()
