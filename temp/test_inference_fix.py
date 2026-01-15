"""
Test inference fixes
Use WikiText model from checkpoints/4
"""

import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sgdd import SGDDModel, SGDDConfig as ModelConfig
from src.utils.config import SGDDConfig
from src.utils.data import WikipediaDataset
from torch.utils.data import DataLoader


def load_model(checkpoint_path: str):
    """Load model"""
    # Load config
    config_path = Path(checkpoint_path) / "config.yaml"
    file_config = SGDDConfig.from_yaml(config_path)

    # Convert to ModelConfig for SGDDModel
    model_config = ModelConfig(
        encoder_model=file_config.model.encoder_name,
        hidden_dim=file_config.model.semantic_dim,
        num_layers=file_config.model.num_layers,
        num_heads=file_config.model.num_heads,
        ffn_dim=file_config.model.ffn_dim,
        max_len=file_config.model.max_length,
        dropout=file_config.model.dropout,
        num_diffusion_steps=file_config.model.num_diffusion_steps,
        use_self_conditioning=file_config.model.use_self_conditioning,
        compute_pad_loss=file_config.model.compute_pad_loss,
    )

    print(f"Config loaded successfully:")
    print(f"  - semantic_dim: {model_config.hidden_dim}")
    print(f"  - max_length: {model_config.max_len}")
    print(f"  - num_layers: {model_config.num_layers}")

    # Create model
    model = SGDDModel(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load weights
    model_path = Path(checkpoint_path) / "best_model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\nModel loaded successfully!")
    print(f"  - checkpoint: {checkpoint_path}")
    print(f"  - device: {device}")

    return model, device, model_config


def test_generation(model, device, config, num_samples=5):
    """Test text generation"""
    print(f"\n{'='*60}")
    print("Testing Generation Quality")
    print(f"{'='*60}\n")

    # Load test data
    dataset = WikipediaDataset(
        split="validation",
        num_samples=1000,
        min_length=20,
        max_length=config.max_len,
        max_token_length=config.max_len,
    )

    # Test combinations of parameters
    test_configs = [
        # (cfg_scale, rep_penalty, top_k, temperature)
        (2.0, 1.0, -1, 1.0),  # Baseline: no improvements
        (2.0, 1.2, -1, 1.0),  # + Repetition penalty only
        (2.0, 1.0, 50, 1.0),  # + Top-k only
        (2.0, 1.2, 50, 0.9),  # + Rep penalty + Top-k + Lower temp
        (2.0, 1.2, 50, 0.8),  # + Stronger improvements
    ]

    for cfg_scale, rep_penalty, top_k, temp in test_configs:
        print(f"\n{'='*60}")
        print(f"CFG={cfg_scale}, RepPen={rep_penalty}, TopK={top_k}, Temp={temp}")
        print(f"{'='*60}\n")

        for i in range(num_samples):
            sample = dataset[i]
            input_text = sample["text"]

            # Generate
            with torch.no_grad():
                generated = model.generate(
                    input_text=input_text,
                    num_steps=16,
                    guidance_scale=cfg_scale,
                    temperature=temp,
                    max_length=config.max_len,
                    repetition_penalty=rep_penalty,
                    top_k=top_k,
                )

            print(f"Example {i+1}:")
            print(f"  Input:     {input_text[:100]}...")
            print(f"  Generated: {generated[:100]}...")
            print()

            if i >= 2:  # Only show first 3
                break


if __name__ == "__main__":
    checkpoint_path = "checkpoints/4"

    print("Loading model...")
    model, device, config = load_model(checkpoint_path)

    print("\nStarting tests...")
    test_generation(model, device, config, num_samples=5)
