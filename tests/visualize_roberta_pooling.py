
import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns
from transformers import AutoModel, AutoTokenizer
import numpy as np

def visualize_roberta_mean_pooling():
    """
    Extracts and visualizes the 768-dim vector directly from RoBERTa + Mean Pooling,
    bypassing the VIB/Adapter projection layers of the SemanticEncoder.
    """
    # Setup output directory
    output_dir = "tests/visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading RoBERTa-base and Tokenizer...")
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    roberta = AutoModel.from_pretrained(model_name)
    roberta.eval() # Set to eval mode
    
    # Input text
    text = "instantly connect with friends and family around the world."
    print(f"\nInput text: '{text}'")
    
    # 1. Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print(f"Tokens: {tokens}")
    
    # 2. RoBERTa Forward Pass
    print("Running RoBERTa forward pass...")
    with torch.no_grad():
        outputs = roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state # Shape: [1, seq_len, 768]
        last_layer_norm=roberta.encoder.layer[-1].output.LayerNorm
        bias=last_layer_norm.bias.detach().numpy()
        top_indices=np.argsort(np.abs(bias))[-50:]
        gamma = last_layer_norm.weight.detach().numpy()
        
        
    print(f"RoBERTa last_hidden_state shape: {last_hidden_state.shape}")
    print(f"Bias 最大的维度索引: {top_indices}")
    print(f"对应的值: {bias[top_indices]}")
    print(f"Index 588 的 Weight: {gamma[588]}")
    print(f"Index 77 的 Weight: {gamma[77]}")
    print(f"Weight 的平均值: {gamma.mean()}")
    # 3. Mean Pooling (Copying logic from src/models/encoder.py)
    # This is the exact logic used in the encoder before VIB
    print("Applying Mean Pooling...")
    
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)
    token_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    mean_pooled = sum_hidden / token_counts # Shape: [1, 768]
    
    print(f"Mean Pooled vector shape: {mean_pooled.shape}")
    
    # Verify shape is strictly 768
    assert mean_pooled.shape[1] == 768, f"Expected 768 dimensions, got {mean_pooled.shape[1]}"

    # 4. Visualization
    
    # A. Token Embeddings (Heatmap) - Pre-pooling
    plt.figure(figsize=(14, 8))
    # Convert to numpy and transpose for better visualization (Dimensions x Tokens) or keep (Tokens x Dimensions)
    # Let's do Tokens (y) x Dimensions (x)
    data = last_hidden_state[0].numpy()
    
    sns.heatmap(data, cmap='viridis', cbar_kws={'label': 'Activation'})
    plt.yticks(range(len(tokens)), tokens, rotation=0)
    plt.xlabel('Embedding Dimension (0-767)')
    plt.title('RoBERTa Token Embeddings (Before Pooling)')
    plt.tight_layout()
    plot_path_tokens = os.path.join(output_dir, "roberta_raw_tokens_768d.png")
    plt.savefig(plot_path_tokens)
    plt.close()
    print(f"Saved token embeddings plot to: {plot_path_tokens}")

    # B. Mean Pooled Vector (Line Plot) - Post-pooling
    plt.figure(figsize=(12, 5))
    vec_data = mean_pooled[0].numpy()
    
    plt.plot(vec_data, alpha=0.8, linewidth=0.8)
    plt.xlabel('Dimension Index (0-767)')
    plt.ylabel('Activation Value')
    plt.title('Mean Pooled RoBERTa Vector (768 Dimensions)\n(Direct Output, No VIB/Projection)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path_vec = os.path.join(output_dir, "roberta_mean_pooled_768d.png")
    plt.savefig(plot_path_vec)
    plt.close()
    print(f"Saved mean pooled vector plot to: {plot_path_vec}")

if __name__ == "__main__":
    visualize_roberta_mean_pooling()
