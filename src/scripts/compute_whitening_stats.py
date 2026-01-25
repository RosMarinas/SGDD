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
    dataset_name="bookcorpus",
    batch_size=64,
    num_samples=100000, 
    output_path="data/whitening_stats_bge.pt",
    device="cuda" if torch.cuda.is_available() else "cpu",
    **dataset_kwargs
):
    print(f"Using device: {device}")
    
    # 1. Initialize Encoder
    print("Initializing Encoder (BGE-M3)...")
    encoder = SemanticEncoder(model_name="BAAI/bge-m3")
    encoder.to(device)
    encoder.eval()
    
    # 2. Initialize Dataloader
    print(f"Loading {dataset_name} dataset...")
    dataloader = get_dataloader(
        dataset_name=dataset_name,
        split="train",
        batch_size=batch_size,
        num_workers=4,
        tokenizer_name="BAAI/bge-m3",
        **dataset_kwargs
    )
    
    print(f"Collecting embeddings...")
    
    all_embeddings = []
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Extract raw encoder outputs
            outputs = encoder.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            last_hidden_state = outputs.last_hidden_state # [batch, seq, 1024]
            
            # Use CLS pooling for BGE-M3
            pooled_output = last_hidden_state[:, 0, :] # [batch, 1024]
            
            all_embeddings.append(pooled_output.cpu())
            count += input_ids.size(0)
            if count >= num_samples:
                break
            
    # Concatenate
    all_embeddings = torch.cat(all_embeddings, dim=0) # [N, 1024]
    print(f"Collected shape: {all_embeddings.shape}")
    
    # 3. Compute Statistics
    print("Computing Mean and Whitening Matrix (ZCA)...")
    
    # Mean
    mu = torch.mean(all_embeddings, dim=0) # [1024]
    
    # Center
    X_centered = all_embeddings - mu
    
    # Covariance
    cov = torch.matmul(X_centered.T, X_centered) / (X_centered.shape[0] - 1)
    
    # SVD
    try:
        U, S, V = torch.svd(cov)
    except RuntimeError:
        U, S, V = torch.linalg.svd(cov)
    
    epsilon = 1e-5
    
    # ZCA Whitening Matrix
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
    
    x_test = all_embeddings[:min(1000, all_embeddings.shape[0])] - mu
    x_white = torch.mm(x_test, W.t())
    cov_white = torch.cov(x_white.T)
    diag_white = torch.diagonal(cov_white)
    
    print(f"Whitened Mean Var: {diag_white.mean().item():.4f} (Target 1.0)")
    print(f"Whitened Max Var:  {diag_white.max().item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bookcorpus", help="Dataset name: bookcorpus")
    parser.add_argument("--samples", type=int, default=50000, help="Total number of samples to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--dataset_path", type=str, default="data/BookCorpus/final_dataset_1.4B", help="Path to dataset")
    
    args = parser.parse_args()
    
    compute_whitening_stats(
        dataset_name=args.dataset,
        num_samples=args.samples,
        batch_size=args.batch_size,
        dataset_path=args.dataset_path
    )