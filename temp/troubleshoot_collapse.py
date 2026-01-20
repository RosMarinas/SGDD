"""
Troubleshoot Posterior Collapse
Run with: uv run python temp/troubleshoot_collapse.py --config configs/phase1_vib.yaml
Or: uv run python temp/troubleshoot_collapse.py --config configs/phase1_wiki.yaml
"""

import torch
import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.sgdd import SGDDModel
from utils.config import SGDDConfig

def analyze_collapse(config_path: str, checkpoint_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    print(f"Loading config from: {config_path}")
    config = SGDDConfig.from_yaml(config_path)

    # Determine checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_dir = Path(config.checkpoint_dir)
        # Find the most recent checkpoint
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
        else:
            checkpoint_path = str(checkpoint_dir / "checkpoint_final.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Please run training first.")
        return

    print(f"Loading checkpoint: {checkpoint_path}")

    # Initialize model with loaded config (SGDDModel expects full SGDDConfig)
    model = SGDDModel(config).to(device)

    # Load state dict
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded successfully.")
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()

    # Print configuration info
    print("\n" + "="*60)
    print("Configuration Summary:")
    print("="*60)
    print(f"Encoder: {config.model.encoder_name}")
    print(f"Semantic Dimension: {config.model.semantic_dim}")
    print(f"KL Weight: {config.model.kl_weight}")
    print(f"KL Threshold: {config.model.kl_threshold}")
    print(f"Contrastive Weight: {config.model.contrastive_weight}")
    print(f"Diffusion Steps: {config.model.num_diffusion_steps}")
    print("="*60)

    # Test texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "Artificial intelligence is transforming the world.",
        "I love eating pizza with extra cheese.",
        "The weather today is sunny and warm.",
    ]

    print("\nEncoding texts...")

    # Encode
    encoded = model.tokenizer(
        texts,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        # Get stats from encoder directly
        outputs = model.encoder.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)
        token_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_hidden / token_counts

        mu = model.encoder.mu_layer(mean_pooled)
        logvar = model.encoder.logvar_layer(mean_pooled)

        # Calculate KL
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # Get z (deterministic)
        z = mu

    print("\n--- Latent Space Analysis ---")

    # 1. Check Mu statistics
    mu_norm = torch.norm(mu, dim=1)
    print(f"\nMu Norms (should be > 0, if ~0 then collapse):")
    for i, norm in enumerate(mu_norm):
        print(f"  Text {i+1}: {norm.item():.4f}")

    # 2. Check Logvar statistics
    sigma = torch.exp(0.5 * logvar)
    print(f"\nSigma Means (should be close to 1 if prior is standard normal):")
    for i, s in enumerate(sigma.mean(dim=1)):
        print(f"  Text {i+1}: {s.item():.4f}")

    # 3. Check KL Divergence
    print(f"\nKL Divergence (should be > 0):")
    for i, kl in enumerate(kl_per_sample):
        print(f"  Text {i+1}: {kl.item():.4f}")
    print(f"Average KL: {kl_per_sample.mean().item():.4f}")

    # 4. Check diversity of Z
    # Compute pairwise cosine similarity
    z_norm = z / z.norm(dim=1, keepdim=True)
    similarity = torch.mm(z_norm, z_norm.t())
    print("\nPairwise Cosine Similarity of Mu (lower is better for diversity):")
    print(similarity)

    avg_sim = (similarity.sum() - len(texts)) / (len(texts) * (len(texts) - 1))
    print(f"\nAverage Pairwise Similarity: {avg_sim.item():.4f}")

    if avg_sim > 0.95:
        print("WARNING: High similarity suggests representations are collapsing to a single point!")
    elif avg_sim < 0.1:
        print("Note: Low similarity suggests good separation.")

    if kl_per_sample.mean() < 0.1:
        print("WARNING: Very low KL divergence suggests posterior collapse to prior!")

    # 5. Generate Text
    print("\n--- Generation Test ---")
    print("Checking if different inputs generate different outputs...")

    num_inference_steps = config.inference.num_inference_steps
    max_length = config.model.max_length

    generated = model.generate(
        semantic_vector=z,
        num_steps=num_inference_steps,
        max_length=max_length
    )

    for i, text in enumerate(generated):
        print(f"  Input {i+1}: {texts[i]}")
        print(f"  Gen   {i+1}: {text}")
        print("-" * 40)

    unique_outputs = len(set(generated))
    print(f"\nUnique outputs: {unique_outputs}/{len(texts)}")

    if unique_outputs == 1:
        print("CRITICAL: All outputs are identical! Posterior collapse confirmed.")
    else:
        print("Outputs are diverse. No complete collapse.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Troubleshoot Posterior Collapse in SGDD Model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file (e.g., configs/phase1_vib.yaml)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (optional, will auto-detect if not provided)"
    )

    args = parser.parse_args()
    analyze_collapse(args.config, args.checkpoint)
