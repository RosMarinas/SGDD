
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.encoder import SemanticEncoder

def visualize_encoder_steps():
    """
    Visualizes the intermediate outputs of the SemanticEncoder:
    1. RoBERTa output (Last Hidden State)
    2. Mean Pooled output
    3. Whitened output (after ZCA)
    4. VIB output (Final z)
    """
    # Setup
    output_dir = "tests/visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    whitening_path = "data/whitening_stats.pt"
    if not os.path.exists(whitening_path):
        print(f"Warning: Whitening stats not found at {whitening_path}. Visualization will skip whitening effect (identity).")
        whitening_path = None
    else:
        print(f"Found whitening stats at {whitening_path}")

    # Initialize Encoder
    print("Initializing SemanticEncoder with BGE-M3...")
    encoder = SemanticEncoder(
        model_name="BAAI/bge-m3", 
        hidden_dim=1024,  # New standard dimension
        whitening_stats_path=whitening_path
    )
    encoder.eval()
    
    input_dim = encoder.input_dim
    print(f"Model Input Dimension: {input_dim}")
    
    # Input Data
    text = "Output dimension for semantic vector"
    print(f"Input text: {text}")
    
    tokenizer = encoder.tokenizer
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print("Executing manual forward pass to capture intermediates...")
    
    with torch.no_grad():
        # 1. Base Model Output
        outputs = encoder.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state # [1, seq_len, input_dim]
        
        # 2. CLS Pooling Output (First Token)
        cls_pooled = last_hidden_state[:, 0, :] # [1, input_dim]
        
        # 3. Whitened Output (if enabled)
        if encoder.use_whitening:
            whitened_pooled = torch.matmul(cls_pooled - encoder.whitening_mean, encoder.whitening_matrix.t())
        else:
            whitened_pooled = cls_pooled
            
        # 4. Dual-Path Outputs
        # Direct Path
        direct_signal = encoder.adapter(whitened_pooled) # [1, 1024]
        
        # VIB Path
        mu = encoder.mu_layer(whitened_pooled) # [1, 1024]
        vib_signal = mu # Inference mode
        
        # Final z
        z = direct_signal + vib_signal # [1, 1024]
        
    # Convert to numpy
    hidden_np = last_hidden_state[0].cpu().numpy()      # [seq, input_dim]
    cls_pooled_np = cls_pooled[0].cpu().numpy()         # [input_dim]
    direct_np = direct_signal[0].cpu().numpy()          # [1024]
    vib_np = vib_signal[0].cpu().numpy()                # [1024]
    z_np = z[0].cpu().numpy()                           # [1024]
    
    # Visualization
    plt.figure(figsize=(15, 20))
    
    # Plot 1: Encoder Last Hidden State (Heatmap)
    plt.subplot(5, 1, 1)
    sns.heatmap(hidden_np, cmap="viridis", cbar=True)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    plt.yticks(ticks=np.arange(len(tokens)) + 0.5, labels=tokens, rotation=0)
    plt.title(f"1. BGE-M3 Last Hidden State (Shape: {hidden_np.shape})")
    plt.xlabel(f"Hidden Dimension ({input_dim})")
    
    # Plot 2: CLS Pooled Output
    plt.subplot(5, 1, 2)
    plt.plot(cls_pooled_np, alpha=0.8, color='blue', linewidth=0.5)
    plt.title(f"2. CLS Vector (Pooled) - Avg: {cls_pooled_np.mean():.4f}, Std: {cls_pooled_np.std():.4f}")
    plt.xlim(0, input_dim)
    plt.grid(True, alpha=0.3)

    # Plot 3: Direct Path Output
    plt.subplot(5, 1, 3)
    plt.plot(direct_np, alpha=0.8, color='green', linewidth=0.5)
    plt.title(f"3. Direct Path Output (Adapter) - Avg: {direct_np.mean():.4f}, Std: {direct_np.std():.4f}")
    plt.xlim(0, 1024)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: VIB Path Output
    plt.subplot(5, 1, 4)
    plt.plot(vib_np, alpha=0.8, color='orange', linewidth=0.5)
    plt.title(f"4. VIB Path Output (Mu) - Avg: {vib_np.mean():.4f}, Std: {vib_np.std():.4f}")
    plt.xlim(0, 1024)
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Final Combined Output (z)
    plt.subplot(5, 1, 5)
    plt.plot(z_np, alpha=0.8, color='red', linewidth=0.5)
    plt.title(f"5. Final Semantic Vector z (Direct + VIB) - Avg: {z_np.mean():.4f}, Std: {z_np.std():.4f}")
    plt.xlim(0, 1024)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "bge_m3_dual_path_visualization.png")
    plt.savefig(save_path)
    print(f"\nVisualization saved to: {save_path}")
    
    # Print numerical stats
    print("\n--- Statistics ---")
    print("1. BGE-M3 Hidden State:")
    print(f"   Shape: {hidden_np.shape}")
    print(f"   Range: [{hidden_np.min():.4f}, {hidden_np.max():.4f}]")
    print("2. CLS Vector:")
    print(f"   Shape: {cls_pooled_np.shape}")
    print(f"   Range: [{cls_pooled_np.min():.4f}, {cls_pooled_np.max():.4f}]")
    print("3. Direct Path:")
    print(f"   Shape: {direct_np.shape}")
    print(f"   Range: [{direct_np.min():.4f}, {direct_np.max():.4f}]")
    print("4. VIB Path:")
    print(f"   Shape: {vib_np.shape}")
    print(f"   Range: [{vib_np.min():.4f}, {vib_np.max():.4f}]")
    print("5. Final z:")
    print(f"   Shape: {z_np.shape}")
    print(f"   Range: [{z_np.min():.4f}, {z_np.max():.4f}]")
if __name__ == "__main__":
    visualize_encoder_steps()