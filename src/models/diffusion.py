"""
Noise Schedule and Discrete Diffusion Process

Implements cosine noise schedule and discrete token diffusion for text.
Based on MaskGIT and D3PM approaches.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class CosineNoiseSchedule(nn.Module):
    """
    Cosine noise schedule for discrete diffusion.

    Uses cosine schedule from Nichol & Dhariwal (2021) which provides
    smoother signal-to-noise ratio transitions compared to linear schedule.

    Attributes:
        num_timesteps: Number of discrete diffusion timesteps
        alphas_cumprod: Cumulative product of (1 - beta) at each timestep
        betas: Noise schedule values at each timestep
    """

    def __init__(self, num_timesteps: int = 1000, s: float = 0.008):
        """
        Initialize cosine noise schedule.

        Args:
            num_timesteps: Number of discrete timesteps (default: 1000)
            s: Small offset to prevent beta_0 from being too small (default: 0.008)
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        self.s = s

        # Compute discrete cosine schedule
        # Based on: https://arxiv.org/abs/2102.09672
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps)

        # Alpha_bar = cos^2(((t/T) + s) / (1 + s) * pi/2)
        alphas_cumprod = torch.cos(
            ((t / num_timesteps) + s) / (1 + s) * math.pi * 0.5
        ) ** 2

        # Normalize so alpha_bar_0 = 1
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        # Compute betas: beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        # Clamp betas to valid range
        betas = torch.clamp(betas, min=1e-8, max=0.999)

        # Register as buffers for device compatibility
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("betas", betas)

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get cumulative product of (1 - beta) up to timestep t.

        Args:
            t: Timestep indices [batch_size]

        Returns:
            alpha_bar_t: [batch_size]
        """
        # Ensure t is within valid range
        t = torch.clamp(t, 0, self.num_timesteps - 1)
        return self.alphas_cumprod[t]

    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get (1 - beta_t) at timestep t.

        Args:
            t: Timestep indices [batch_size]

        Returns:
            alpha_t: [batch_size]
        """
        t = torch.clamp(t, 0, self.num_timesteps - 1)
        return 1 - self.betas[t]

    def get_beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get noise schedule value at timestep t.

        Args:
            t: Timestep indices [batch_size]

        Returns:
            beta_t: [batch_size]
        """
        t = torch.clamp(t, 0, self.num_timesteps - 1)
        return self.betas[t]


class DiscreteDiffusion:
    """
    Discrete diffusion process for tokens.

    Implements forward diffusion process that adds noise to clean tokens
    by independently replacing them with random tokens based on alpha_bar.

    This is MaskGIT-style discrete diffusion where at each timestep,
    each token has probability alpha_bar of being kept and (1 - alpha_bar)
    of being replaced with a random token.
    """

    def __init__(
        self,
        noise_schedule: CosineNoiseSchedule,
        num_classes: int = 50265,  # RoBERTa vocab size
    ):
        """
        Initialize discrete diffusion process.

        Args:
            noise_schedule: CosineNoiseSchedule instance
            num_classes: Vocabulary size (number of possible tokens)
        """
        self.noise_schedule = noise_schedule
        self.num_classes = num_classes

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to clean tokens.

        For each token independently:
        - With probability alpha_bar_t: keep original token
        - With probability (1 - alpha_bar_t): replace with random token

        Args:
            x_start: Clean tokens [batch_size, seq_len]
            t: Timestep [batch_size]
            noise: Random tokens (if None, sample uniformly)

        Returns:
            x_t: Noisy tokens [batch_size, seq_len]
        """
        batch_size, seq_len = x_start.shape
        device = x_start.device

        # Sample random tokens if not provided
        if noise is None:
            noise = torch.randint(
                0, self.num_classes, (batch_size, seq_len), device=device
            )

        # Get alpha_bar_t for each sample in batch
        alpha_bar = self.noise_schedule.get_alpha_bar(t)  # [batch_size]

        # Create mask: which tokens to keep vs replace
        # For each position, sample uniform [0,1] and compare with alpha_bar
        random_matrix = torch.rand(batch_size, seq_len, device=device)
        keep_mask = random_matrix < alpha_bar.unsqueeze(1)  # [batch_size, seq_len]

        # Mix original and noise based on keep_mask
        # Where keep_mask is True, keep original; otherwise, use noise
        x_t = torch.where(keep_mask, x_start, noise)

        return x_t

    def get_loss_weights(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        compute_pad_loss: bool = False,
    ) -> torch.Tensor:
        """
        Compute weighting for each token in loss calculation.

        Args:
            x_start: Original clean tokens [batch_size, seq_len]
            x_t: Noisy tokens [batch_size, seq_len]
            t: Timestep [batch_size]
            compute_pad_loss: If True, compute loss on all positions including PAD.
                              This teaches the model to output PAD tokens appropriately.

        Returns:
            loss_mask: Binary mask [batch_size, seq_len]
                       1 for positions to compute loss, 0 otherwise
        """
        if compute_pad_loss:
            # All positions contribute to loss (including PAD)
            # This teaches the model when to output PAD/EOS tokens
            return torch.ones_like(x_start.float(), device=x_start.device)
        else:
            # Only compute loss on tokens that were actually noised (changed)
            # This improves efficiency by focusing learning on difficult tokens
            mask = (x_t != x_start).float()
            return mask


if __name__ == "__main__":
    # Quick test
    print("Testing CosineNoiseSchedule and DiscreteDiffusion...")

    # Test noise schedule
    print("\n1. Testing CosineNoiseSchedule...")
    schedule = CosineNoiseSchedule(num_timesteps=1000)
    print("[OK] Noise schedule initialized")

    # Test alpha_bar is decreasing
    alpha_bar_0 = schedule.get_alpha_bar(torch.tensor([0]))
    alpha_bar_1000 = schedule.get_alpha_bar(torch.tensor([999]))
    print(f"  alpha_bar at t=0: {alpha_bar_0.item():.4f}")
    print(f"  alpha_bar at t=999: {alpha_bar_1000.item():.4f}")
    assert alpha_bar_0 > alpha_bar_1000, "alpha_bar should decrease"
    print("[OK] alpha_bar is decreasing")

    # Test discrete diffusion
    print("\n2. Testing DiscreteDiffusion...")
    diffusion = DiscreteDiffusion(schedule, num_classes=50265)
    print("[OK] DiscreteDiffusion initialized")

    # Test forward diffusion
    batch_size = 4
    seq_len = 64
    x_start = torch.randint(0, 50265, (batch_size, seq_len))
    t = torch.tensor([0, 250, 500, 999])

    x_t = diffusion.q_sample(x_start, t)
    print(f"[OK] Forward diffusion successful")
    print(f"  Input shape: {x_start.shape}")
    print(f"  Output shape: {x_t.shape}")

    # At t=0, should be mostly unchanged
    similarity_0 = (x_t[0] == x_start[0]).float().mean()
    print(f"  Similarity at t=0: {similarity_0.item():.2%}")
    assert similarity_0 > 0.95, "At t=0, most tokens should be unchanged"

    # At t=1000, should be mostly random
    similarity_1000 = (x_t[3] == x_start[3]).float().mean()
    print(f"  Similarity at t=999: {similarity_1000.item():.2%}")
    assert similarity_1000 < 0.10, "At t=999, most tokens should be changed"

    print("[OK] Noise level increases with t")

    # Test loss weights
    loss_mask = diffusion.get_loss_weights(x_start, x_t, t)
    print(f"[OK] Loss mask computed")
    print(f"  Loss mask shape: {loss_mask.shape}")
    print(f"  Non-zero ratio at t=0: {loss_mask[0].mean().item():.2%}")
    print(f"  Non-zero ratio at t=999: {loss_mask[3].mean().item():.2%}")

    print("\n[SUCCESS] All tests passed!")
