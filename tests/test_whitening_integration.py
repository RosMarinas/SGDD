import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from src.models.encoder import SemanticEncoder
from src.utils.data import get_dataloader

class WhitenedSemanticEncoder(SemanticEncoder):
    """
    Experimental Encoder with Whitening.
    """
    def __init__(self, whitening_stats=None, **kwargs):
        super().__init__(**kwargs)
        
        self.register_buffer("whitening_mean", torch.zeros(768))
        self.register_buffer("whitening_matrix", torch.eye(768))
        self.use_whitening = False
        
        if whitening_stats is not None:
            self.load_whitening_stats(whitening_stats)

    def load_whitening_stats(self, stats):
        self.whitening_mean = stats["mean"]
        self.whitening_matrix = stats["whitening_matrix"]
        self.use_whitening = True
        # Move to correct device if needed
        # (Buffers handle this automatically on .to(device))

    def forward(self, input_ids, attention_mask, return_kl=False):
        # 1. RoBERTa
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state
        
        # 2. Mean Pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)
        token_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_hidden / token_counts
        
        # 3. Apply Whitening (if enabled)
        if self.use_whitening:
            # (x - mu) @ W.T
            mean_pooled = torch.matmul(mean_pooled - self.whitening_mean, self.whitening_matrix.t())
        
        # 4. Rest of the pipeline (Adapter + VIB)
        direct_signal = self.adapter(mean_pooled)
        mu = self.mu_layer(mean_pooled)
        logvar = self.logvar_layer(mean_pooled)
        logvar = torch.clamp(logvar, min=-10, max=2)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            vib_signal = mu + std * eps
        else:
            vib_signal = mu
            
        z = direct_signal + vib_signal
        
        # KL Calculation
        current_kl_weight = self.kl_weight
        if self.training:
            anneal_ratio = min(1.0, self._current_step / self.kl_anneal_steps)
            current_kl_weight *= anneal_ratio

        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        if self.kl_threshold > 0:
            kl_per_sample = torch.max(kl_per_sample, torch.tensor(self.kl_threshold, device=kl_per_sample.device))

        kl_weighted = kl_per_sample * current_kl_weight

        if return_kl:
            return z, kl_weighted
        return z

def compute_stats_on_fly(dataloader, device, num_samples=200):
    print("Computing stats on the fly...")
    encoder = SemanticEncoder().to(device)
    encoder.eval()
    
    all_embs = []
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            out = encoder.roberta(input_ids, attention_mask)
            mask = attention_mask.unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embs.append(emb)
            count += len(emb)
            if count >= num_samples: break
            
    all_embs = torch.cat(all_embs)[:num_samples]
    mu = all_embs.mean(0)
    X = all_embs - mu
    cov = (X.T @ X) / (len(X)-1)
    
    # SVD
    U, S, V = torch.svd(cov)
    epsilon = 1e-5
    S_inv_sqrt = torch.diag(1.0 / torch.sqrt(S + epsilon))
    W = U @ S_inv_sqrt @ U.t()
    
    return {"mean": mu, "whitening_matrix": W}

def test_integration():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Integration on {device}")
    
    # 1. Get Data & Stats
    dataloader = get_dataloader("wikipedia", batch_size=16, num_samples=300, num_workers=0)
    stats = compute_stats_on_fly(dataloader, device)
    
    # 2. Init Whitened Encoder
    print("Initializing Whitened Encoder...")
    model = WhitenedSemanticEncoder(whitening_stats=stats, hidden_dim=512)
    model.to(device)
    model.train()
    
    # 3. Run Forward Pass
    print("Running forward pass...")
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    
    z, kl = model(input_ids, mask, return_kl=True)
    
    # 4. Checks
    print("\n--- Integration Results ---")
    print(f"Output Z shape: {z.shape}")
    print(f"KL Loss shape: {kl.shape}")
    print(f"KL Loss mean: {kl.mean().item():.4f}")
    
    assert z.shape == (input_ids.shape[0], 512)
    assert not torch.isnan(z).any(), "Output contains NaNs"
    assert not torch.isinf(z).any(), "Output contains Infs"
    
    print("[SUCCESS] Whitened Encoder produces valid outputs.")

if __name__ == "__main__":
    test_integration()
