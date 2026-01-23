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
    print("Initializing SemanticEncoder...")
    encoder = SemanticEncoder(
        model_name="roberta-base", 
        hidden_dim=512,
        whitening_stats_path=whitening_path
    )
    encoder.eval()
    
    # Input Data
    text = "The quick brown fox jumps over the lazy dog."
    print(f"Input text: {text}")
    
    tokenizer = encoder.tokenizer
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print("Executing manual forward pass to capture intermediates...")
    
    with torch.no_grad():
        # 1. RoBERTa Output
        outputs = encoder.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state # [1, seq_len, 768]
        
        # 2. Mean Pooling Output
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)
        token_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        raw_mean_pooled = sum_hidden / token_counts # [1, 768]
        
        # 3. Whitened Output
        # Logic copied from SemanticEncoder.forward
        if encoder.use_whitening:
            whitened_pooled = torch.matmul(raw_mean_pooled - encoder.whitening_mean, encoder.whitening_matrix.t())
        else:
            whitened_pooled = raw_mean_pooled
            
        # 4. VIB Output (Final z)
        # Direct Path
        direct_signal = encoder.adapter(whitened_pooled)
        
        # VIB Path
        mu = encoder.mu_layer(whitened_pooled)
        vib_signal = mu # Inference mode
        
        z = direct_signal + vib_signal # [1, 512]
        
    # Convert to numpy
    hidden_np = last_hidden_state[0].numpy()      # [seq, 768]
    raw_pooled_np = raw_mean_pooled[0].numpy()    # [768]
    whitened_np = whitened_pooled[0].numpy()      # [768]
    z_np = z[0].numpy()                           # [512]
    
    # Visualization
    plt.figure(figsize=(15, 16))
    
    # Plot 1: RoBERTa Last Hidden State (Heatmap)
    plt.subplot(4, 1, 1)
    sns.heatmap(hidden_np, cmap="viridis", cbar=True)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    plt.yticks(ticks=np.arange(len(tokens)) + 0.5, labels=tokens, rotation=0)
    plt.title(f"1. RoBERTa Last Hidden State (Shape: {hidden_np.shape})")
    plt.xlabel("Hidden Dimension (768)")
    
    # Plot 2: Raw Mean Pooled Output
    plt.subplot(4, 1, 2)
    plt.plot(raw_pooled_np, alpha=0.7, color='blue', linewidth=0.5)
    plt.title(f"2. Mean Pooled Output (Raw) - Avg: {raw_pooled_np.mean():.4f}, Std: {raw_pooled_np.std():.4f}")
    plt.xlim(0, 768)
    plt.grid(True, alpha=0.3)

    # Plot 3: Whitened Output
    plt.subplot(4, 1, 3)
    plt.plot(whitened_np, alpha=0.7, color='green', linewidth=0.5)
    plt.title(f"3. Whitened Output (ZCA) - Avg: {whitened_np.mean():.4f}, Std: {whitened_np.std():.4f}")
    plt.xlim(0, 768)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: VIB Output (Final z)
    plt.subplot(4, 1, 4)
    plt.plot(z_np, alpha=0.7, color='red', linewidth=0.5)
    plt.title(f"4. VIB Final Output z (Shape: {z_np.shape}) - Avg: {z_np.mean():.4f}, Std: {z_np.std():.4f}")
    plt.xlim(0, 512)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "encoder_steps_visualization.png")
    plt.savefig(save_path)
    print(f"\nVisualization saved to: {save_path}")
    
    # Print numerical stats
    print("\n--- Statistics ---")
    print("1. RoBERTa Hidden State:")
    print(f"   Shape: {hidden_np.shape}")
    print(f"   Range: [{hidden_np.min():.4f}, {hidden_np.max():.4f}]")
    print("2. Raw Mean Pooled:")
    print(f"   Shape: {raw_pooled_np.shape}")
    print(f"   Range: [{raw_pooled_np.min():.4f}, {raw_pooled_np.max():.4f}]")
    print("3. Whitened Output:")
    print(f"   Shape: {whitened_np.shape}")
    print(f"   Range: [{whitened_np.min():.4f}, {whitened_np.max():.4f}]")
    print("4. VIB Output (z):")
    print(f"   Shape: {z_np.shape}")
    print(f"   Range: [{z_np.min():.4f}, {z_np.max():.4f}]")
if __name__ == "__main__":
    visualize_encoder_steps()