
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.encoder import SemanticEncoder
from src.utils.data import BookCorpusDataset, collate_fn_bookcorpus

def visualize_encoder_batch():
    """
    Visualizes the encoder outputs for a batch of 10 samples from the dataset.
    """
    # Setup
    output_dir = "tests/visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Initialize Encoder
    print("Initializing SemanticEncoder...")
    encoder = SemanticEncoder(
        model_name="BAAI/bge-m3", 
        hidden_dim=1024
    )
    encoder.to(device)
    encoder.eval()

    # 2. Load Dataset (BookCorpus)
    print("Loading Dataset (BookCorpus)...")
    # We only need a few samples
    dataset = BookCorpusDataset(
        dataset_path="data/BookCorpus/final_dataset_1.4B",
        max_token_length=64,
        split="train"
    )
    
    # Select 20 samples randomly
    np.random.seed(42)
    indices = np.random.choice(len(dataset), size=20, replace=False)
    samples = [dataset[i] for i in indices]
    
    # Collate
    batch = collate_fn_bookcorpus(samples)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    texts = batch["texts"]

    print(f"Selected {len(texts)} samples.")
    for i, text in enumerate(texts):
        print(f"  {i+1}. {text[:50]}...")

    # 3. Forward Pass & Capture Intermediates
    print("Running forward pass...")
    
    with torch.no_grad():
        # A. Base Model Output
        outputs = encoder.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state # [B, seq_len, input_dim]
        
        # B. CLS Pooling
        # BGE-M3 uses CLS token
        cls_pooled = last_hidden_state[:, 0, :] # [B, input_dim]
        
        # C. Adapter / VIB
        # Direct Path
        direct_signal = encoder.adapter(cls_pooled) # [B, hidden_dim]
        
        # VIB Path
        mu = encoder.mu_layer(cls_pooled) # [B, hidden_dim]
        logvar = encoder.logvar_layer(cls_pooled)
        
        # Final Z (Validation mode: z = direct + mu)
        z = direct_signal + mu
        
    # Convert to numpy
    cls_pooled_np = cls_pooled.cpu().numpy()
    z_np = z.cpu().numpy()
    
    # 4. Visualizations
    plt.figure(figsize=(20, 12))
    
    # Plot 1: Cosine Similarity Matrix of Final Embeddings (Z)
    # Normalize rows
    z_norm = z_np / np.linalg.norm(z_np, axis=1, keepdims=True)
    sim_matrix = np.dot(z_norm, z_norm.T)
    
    plt.subplot(2, 2, 1)
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Cosine Similarity Matrix (Final Z)")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    
    # Plot 2: Heatmap of Z (Embeddings)
    plt.subplot(2, 2, 2)
    sns.heatmap(z_np, cmap="viridis", cbar=True)
    plt.title(f"Final Embeddings Z (Shape: {z_np.shape})")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Sample Index")

    # Plot 3: Distribution of values in Z
    plt.subplot(2, 2, 3)
    for i in range(len(z_np)):
        sns.kdeplot(z_np[i], alpha=0.3, label=f"S{i}")
    plt.title("Value Distribution in Z per Sample")
    plt.xlabel("Value")
    # plt.legend() # Too cluttered with 10 lines
    
    # Plot 4: Standard Deviation across dimensions (Anisotropy check)
    plt.subplot(2, 2, 4)
    # Std of each dimension across batch
    dim_std = np.std(z_np, axis=0)
    plt.plot(dim_std, alpha=0.7)
    plt.title("Standard Deviation per Dimension (Batch Diversity)")
    plt.xlabel("Dimension Index")
    plt.ylabel("Std Dev")
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "batch_10_visualization.png")
    plt.savefig(save_path)
    print(f"\nVisualization saved to: {save_path}")

    # Text report
    print("\n--- Sample Texts ---")
    for i, text in enumerate(texts):
        print(f"Sample {i}: {text}")

if __name__ == "__main__":
    visualize_encoder_batch()
