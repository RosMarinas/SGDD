
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from src.models.encoder import SemanticEncoder
from src.utils.data import get_dataloader

class WhiteningTransform(nn.Module):
    """
    Helper module to apply whitening transformation.
    """
    def __init__(self, mean, whitening_matrix):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("whitening_matrix", whitening_matrix)
        
    def forward(self, x):
        # x: [batch, dim]
        # centered = x - mean
        # whitened = centered @ W.T
        return torch.matmul(x - self.mean, self.whitening_matrix.t())

def compute_statistics(encoder, dataloader, device, num_samples=1000):
    """
    Compute mean and whitening matrix from dataloader.
    """
    print(f"Collecting {num_samples} samples for statistics...")
    all_embeddings = []
    count = 0
    
    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Extract raw RoBERTa mean pool
            outputs = encoder.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            last_hidden_state = outputs.last_hidden_state
            
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)
            token_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_hidden / token_counts # [batch, 768]
            
            all_embeddings.append(mean_pooled.cpu())
            count += input_ids.size(0)
            if count >= num_samples:
                break
                
    all_embeddings = torch.cat(all_embeddings, dim=0)[:num_samples]
    print(f"Collected shape: {all_embeddings.shape}")
    
    # Compute Mean
    mu = torch.mean(all_embeddings, dim=0)
    
    # Compute Covariance
    X_centered = all_embeddings - mu
    cov = torch.matmul(X_centered.T, X_centered) / (X_centered.shape[0] - 1)
    
    # Compute Whitening Matrix (ZCA)
    # SVD on covariance
    try:
        U, S, V = torch.svd(cov)
    except RuntimeError:
        # Fallback for stability if needed (e.g. on cpu sometimes)
        U, S, V = torch.linalg.svd(cov)
        
    epsilon = 1e-5
    S_inv_sqrt = torch.diag(1.0 / torch.sqrt(S + epsilon))
    
    # ZCA Whitening: W = U * S^-1/2 * U^T
    W = torch.mm(torch.mm(U, S_inv_sqrt), U.t())
    
    return mu, W, all_embeddings

def test_whitening_optimization():
    """
    Main test function to visualize whitening effect.
    """
    output_dir = "tests/visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Setup
    encoder = SemanticEncoder()
    encoder.to(device)
    
    # Use 'wikipedia' dataset
    # Loading a small subset for testing
    dataloader = get_dataloader(
        dataset_name="wikipedia",
        split="train",
        batch_size=32,
        num_samples=2000, # Sufficient for covariance estimation test
        num_workers=0 # Avoid multiprocessing issues in simple tests
    )
    
    # 2. Compute Stats
    mu, W, raw_embeddings = compute_statistics(encoder, dataloader, device, num_samples=1000)
    
    # 3. Apply Whitening
    whitener = WhiteningTransform(mu, W)
    whitened_embeddings = whitener(raw_embeddings)
    
    # 4. Analyze & Visualize
    
    # Take a single sample for detailed line plot
    idx = 0
    raw_vec = raw_embeddings[idx].numpy()
    whitened_vec = whitened_embeddings[idx].numpy()
    
    # Global statistics
    raw_cov = torch.cov(raw_embeddings.T)
    white_cov = torch.cov(whitened_embeddings.T)
    
    # Extract diagonal (variance) and off-diagonal (correlations)
    raw_vars = torch.diagonal(raw_cov)
    white_vars = torch.diagonal(white_cov)
    
    raw_off_diag = raw_cov[~torch.eye(768, dtype=bool)].abs().mean()
    white_off_diag = white_cov[~torch.eye(768, dtype=bool)].abs().mean()
    
    print("\n--- Quantitative Analysis ---")
    print(f"Original Mean Variance: {raw_vars.mean():.4f}")
    print(f"Original Max Variance:  {raw_vars.max():.4f}")
    print(f"Original Mean Abs Corr: {raw_off_diag:.4f}")
    print("-" * 30)
    print(f"Whitened Mean Variance: {white_vars.mean():.4f} (Target ~1.0)")
    print(f"Whitened Max Variance:  {white_vars.max():.4f}")
    print(f"Whitened Mean Abs Corr: {white_off_diag:.4f} (Target ~0.0)")
    
    # Plotting
    plt.figure(figsize=(15, 12))
    
    # Subplot 1: Single Vector Comparison
    plt.subplot(3, 1, 1)
    plt.plot(raw_vec, alpha=0.7, label=f"Original (Max={raw_vec.max():.2f})", color='red', linewidth=0.8)
    plt.plot(whitened_vec, alpha=0.7, label=f"Whitened (Max={whitened_vec.max():.2f})", color='green', linewidth=0.8)
    plt.title("Single Vector Representation: Original vs Whitened")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Subplot 2: Variance Spectrum (Eigenvalues)
    # Perform PCA on both to see explained variance ratio
    _, S_raw, _ = torch.svd(raw_cov)
    _, S_white, _ = torch.svd(white_cov)
    
    plt.subplot(3, 1, 2)
    plt.plot(S_raw.numpy(), label="Original Eigenvalues", color='red', alpha=0.7)
    plt.plot(S_white.numpy(), label="Whitened Eigenvalues", color='green', alpha=0.7)
    plt.yscale('log')
    plt.title("Spectrum of Covariance Matrix (Log Scale)")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Subplot 3: Heatmap of first 50x50 Covariance
    plt.subplot(3, 2, 5)
    plt.imshow(raw_cov[:50, :50].numpy(), cmap='coolwarm', vmin=-0.5, vmax=0.5)
    plt.title("Original Covariance (Top-Left 50x50)")
    plt.colorbar()
    
    plt.subplot(3, 2, 6)
    plt.imshow(white_cov[:50, :50].numpy(), cmap='coolwarm', vmin=-0.5, vmax=0.5)
    plt.title("Whitened Covariance (Top-Left 50x50)")
    plt.colorbar()
    
    output_path = os.path.join(output_dir, "whitening_analysis.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nVisualization saved to {output_path}")
    
    # Check for spikes removal
    # Define a spike as a dimension with value > 5 * std
    raw_std = raw_embeddings.std()
    white_std = whitened_embeddings.std()
    
    raw_spikes = (raw_embeddings.abs() > 5 * raw_std).float().mean().item()
    white_spikes = (whitened_embeddings.abs() > 5 * white_std).float().mean().item()
    
    print(f"\nSpike Analysis (> 5*sigma):")
    print(f"Original Spike Ratio: {raw_spikes*100:.4f}%")
    print(f"Whitened Spike Ratio: {white_spikes*100:.4f}%")

if __name__ == "__main__":
    test_whitening_optimization()
