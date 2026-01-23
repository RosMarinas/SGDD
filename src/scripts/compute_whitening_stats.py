
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.getcwd())

from src.models.encoder import SemanticEncoder
from src.utils.data import get_dataloader

def compute_whitening_stats(
    dataset_name="wikipedia",
    batch_size=64,
    num_samples=100000, 
    output_path="data/whitening_stats.pt",
    device="cuda" if torch.cuda.is_available() else "cpu",
    **dataset_kwargs
):
    print(f"Using device: {device}")
    
    # 1. Initialize Encoder
    print("Initializing Encoder...")
    # We only need the base RoBERTa, so we can initialize SemanticEncoder
    # effectively just to access its internal .roberta and tokenizer
    encoder = SemanticEncoder()
    encoder.to(device)
    encoder.eval()
    
    # 2. Initialize Dataloader
    print(f"Loading {dataset_name} dataset...")
    dataloader = get_dataloader(
        dataset_name=dataset_name,
        split="train",
        batch_size=batch_size,
        num_workers=4,
        num_samples=num_samples,
        **dataset_kwargs
    )
    
    print(f"Collecting embeddings from {len(dataloader.dataset)} samples...")
    
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Extract raw RoBERTa outputs
            outputs = encoder.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            last_hidden_state = outputs.last_hidden_state # [batch, seq, 768]
            
            # Mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)
            token_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_hidden / token_counts # [batch, 768]
            
            all_embeddings.append(mean_pooled.cpu())
            
    # Concatenate
    all_embeddings = torch.cat(all_embeddings, dim=0) # [N, 768]
    print(f"Collected shape: {all_embeddings.shape}")
    
    # 3. Compute Statistics
    print("Computing Mean and Whitening Matrix (ZCA)...")
    
    # Mean
    mu = torch.mean(all_embeddings, dim=0) # [768]
    
    # Center
    X_centered = all_embeddings - mu
    
    # Covariance: Sigma = (1 / (N-1)) * X^T * X
    # Note: For very large N, doing this on GPU might be faster, but let's check memory.
    # 768^2 float32 is tiny. 100k * 768 is ~300MB. CPU is fine.
    cov = torch.matmul(X_centered.T, X_centered) / (X_centered.shape[0] - 1)
    
    # SVD for stability
    # Sigma = U * S * V^T (U=V since Sigma is symmetric)
    try:
        U, S, V = torch.svd(cov)
    except RuntimeError:
        # Fallback to linalg.svd if needed
        U, S, V = torch.linalg.svd(cov)
    
    epsilon = 1e-5
    
    # ZCA Whitening Matrix: W = U * S^{-1/2} * U^T
    S_inv_sqrt = torch.diag(1.0 / torch.sqrt(S + epsilon))
    W = torch.mm(torch.mm(U, S_inv_sqrt), U.t())
    
    stats = {
        "mean": mu,
        "whitening_matrix": W,
        "covariance": cov,
        "eigenvalues": S,
        "n_samples": all_embeddings.shape[0]
    }
    
    # 4. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(stats, output_path)
    print(f"Saved statistics to {output_path}")
    
    # Verification stats
    print("\n--- Verification ---")
    print(f"Original Max Var: {torch.diagonal(cov).max().item():.4f}")
    
    # Apply to a sample batch to check
    x_test = all_embeddings[:1000] - mu
    x_white = torch.mm(x_test, W.t())
    cov_white = torch.cov(x_white.T)
    diag_white = torch.diagonal(cov_white)
    
    print(f"Whitened Mean Var: {diag_white.mean().item():.4f} (Target 1.0)")
    print(f"Whitened Max Var:  {diag_white.max().item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikipedia", help="Dataset name: wikipedia, qqp, mixed")
    parser.add_argument("--samples", type=int, default=100000, help="Total number of samples to use")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    
    # Mixed dataset specific args
    parser.add_argument("--wiki_ratio", type=float, default=0.6, help="Ratio of Wikipedia samples")
    parser.add_argument("--alpaca_ratio", type=float, default=0.3, help="Ratio of Alpaca samples")
    parser.add_argument("--oasst_ratio", type=float, default=0.1, help="Ratio of OASST1 samples")
    
    args = parser.parse_args()
    
    dataset_kwargs = {}
    if args.dataset == "mixed":
        # Calculate absolute numbers based on total samples and ratios
        dataset_kwargs["wiki_num_samples"] = int(args.samples * args.wiki_ratio)
        dataset_kwargs["alpaca_num_samples"] = int(args.samples * args.alpaca_ratio)
        dataset_kwargs["oasst1_num_samples"] = int(args.samples * args.oasst_ratio)
        print(f"Mixed Dataset Config: Wiki={dataset_kwargs['wiki_num_samples']}, Alpaca={dataset_kwargs['alpaca_num_samples']}, OASST={dataset_kwargs['oasst1_num_samples']}")
    
    compute_whitening_stats(
        dataset_name=args.dataset,
        num_samples=args.samples,
        batch_size=args.batch_size,
        **dataset_kwargs
    )
