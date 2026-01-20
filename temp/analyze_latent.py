
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Add project root to sys.path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.encoder import SemanticEncoder

def analyze_latent_space():
    print("Initializing SemanticEncoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize encoder
    # Using default parameters as per the code
    encoder = SemanticEncoder(
        model_name="roberta-base",
        hidden_dim=512,
        kl_weight=0.001
    ).to(device)
    encoder.eval()

    print("Encoder initialized.")

    # Sample texts to test
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "To be or not to be, that is the question.",
        "I love machine learning and deep neural networks.",
        "The weather today is sunny with a chance of rain.",
        "Python is a great programming language for data science.",
        "Quantum computing will solve problems that are currently intractable.",
        "A neural network consists of layers of interconnected nodes.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of data for training."
    ]

    print(f"\nProcessing {len(texts)} sample texts...")

    # encode_text method returns [1, hidden_dim]
    latent_vectors = []
    
    # We can also use forward directly to get KL
    # Let's batch them for efficiency and better statistics
    tokenizer = encoder.get_tokenizer()
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    print("\nRunning forward pass (Deterministic Inference Mode)...")
    with torch.no_grad():
        # Default behavior in eval mode is deterministic (z = mu)
        z = encoder(input_ids, attention_mask)
        latent_vectors = z.cpu().numpy()

    print(f"Latent vectors shape: {latent_vectors.shape}")
    
    # Statistics of the latent space (Deterministic)
    print("\n--- Latent Space Statistics (Deterministic / Mu) ---")
    print(f"Mean: {np.mean(latent_vectors):.4f}")
    print(f"Std:  {np.std(latent_vectors):.4f}")
    print(f"Min:  {np.min(latent_vectors):.4f}")
    print(f"Max:  {np.max(latent_vectors):.4f}")
    
    # Check if it looks normalized (it shouldn't necessarily if untrained/initialized lightly, 
    # but the code uses BatchNorm1d so it might be close to 0 mean 1 std per feature batch-wise?)
    # Wait, BatchNorm is in the projection layers. 
    
    # Now let's check the VIB properties (KL Divergence)
    # We need to switch to training mode or manually invoke the components to see mu and logvar
    # Or just use return_kl=True which works in forward()
    
    print("\nRunning forward pass with KL (training mode simulation for VIB)...")
    # We temporarily set training to True to enable sampling if we want to check stochasticity,
    # but return_kl=True works regardless of training mode in the code provided?
    # Let's check the code:
    # "if return_kl: return z, kl_weighted"
    # "if self.training: std = ... z = mu + std*eps ... else: z = mu"
    
    # To see the distribution properties (mu, logvar), we want to inspect them.
    # The public API doesn't return mu/logvar directly, but we can hook or just modify usage.
    # Actually, looking at `encoder.py`:
    # mu = self.mu_layer(mean_pooled)
    # logvar = self.logvar_layer(mean_pooled)
    
    # Let's verify KL output
    with torch.no_grad():
        # Even in eval mode, it calculates KL based on mu and logvar
        z_out, kl_loss = encoder(input_ids, attention_mask, return_kl=True)
        kl_values = kl_loss.cpu().numpy()

    print("\n--- KL Divergence Statistics ---")
    print(f"KL Loss (per sample) Mean: {np.mean(kl_values):.4f}")
    print(f"KL Loss (per sample) Min:  {np.min(kl_values):.4f}")
    print(f"KL Loss (per sample) Max:  {np.max(kl_values):.4f}")

    # Let's inspect pairwise similarities to check for collapse (isotropy check)
    # Normalize vectors
    norms = np.linalg.norm(latent_vectors, axis=1, keepdims=True)
    normalized_z = latent_vectors / (norms + 1e-9)
    
    cosine_sim_matrix = np.dot(normalized_z, normalized_z.T)
    
    # Get off-diagonal elements
    off_diag_mask = ~np.eye(cosine_sim_matrix.shape[0], dtype=bool)
    off_diag_sims = cosine_sim_matrix[off_diag_mask]
    
    print("\n--- Anisotropy / Collapse Check ---")
    print(f"Mean Pairwise Cosine Similarity: {np.mean(off_diag_sims):.4f}")
    print("  (Close to 1.0 indicates representation collapse)")
    print("  (Close to 0.0 indicates good separation/isotropy)")

    # --- NEW: Check Variance/Noise Initialization ---
    print("\n--- Variance Initialization Check ---")
    # Access the logvar layer directly to see initialization
    with torch.no_grad():
        # Get the mean pooled vectors again manually to feed into logvar layer
        outputs = encoder.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)
        token_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_hidden / token_counts
        
        # Pass through logvar layer
        logvar = encoder.logvar_layer(mean_pooled)
        sigma = torch.exp(0.5 * logvar)
        
    print(f"LogVar Mean: {logvar.mean().item():.4f}")
    print(f"Sigma Mean:  {sigma.mean().item():.4f}")
    print("  (If Sigma is close to 1.0 at start, the signal is drowned in noise!)")
    print("  (Target for start of training: Sigma should be small, e.g., < 0.1)")
    
    print("\nSample Output Vector (first 10 dims of first sample):")
    print(latent_vectors[0, :10])

if __name__ == "__main__":
    analyze_latent_space()
