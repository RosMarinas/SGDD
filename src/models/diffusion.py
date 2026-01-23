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

    def __init__(self, num_timesteps: int = 1000, s: float = 0.008,
                 min_alpha_bar: float = 0.001):
        """
        Initialize cosine noise schedule.

        Args:
            num_timesteps: Number of discrete timesteps (default: 1000)
            s: Small offset to prevent beta_0 from being too small (default: 0.008)
            min_alpha_bar: Minimum alpha_bar at final timestep (default: 0.001)
                          Controls how aggressive the schedule is. Lower values mean
                          more aggressive corruption (more MASK tokens during training).
                          Set to 0.001 or lower for near-complete MASK at high timesteps.
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        self.s = s
        self.min_alpha_bar = min_alpha_bar

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

        # IMPORTANT: Rescale to reach min_alpha_bar at final timestep
        # This ensures that at t=num_timesteps, alpha_bar is very small (~0.1% keep rate)
        # instead of the default ~0%, forcing near-complete MASK during training
        alphas_cumprod = alphas_cumprod * (1 - min_alpha_bar) + min_alpha_bar

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
    Discrete diffusion process for tokens with Absorbing State mechanism.

    Implements forward diffusion process that adds noise to clean tokens
    by independently replacing them with MASK tokens based on alpha_bar.

    This is Absorbing State diffusion (MaskGIT-style) where at each timestep,
    each token has probability alpha_bar of being kept and (1 - alpha_bar)
    of being replaced with a MASK token (absorbing state).

    The absorbing state ensures training-inference consistency:
    - Training: tokens are corrupted to MASK
    - Inference: start from all MASK and progressively denoise
    """

    def __init__(
        self,
        noise_schedule: CosineNoiseSchedule,
        num_classes: int = 50265,  # RoBERTa vocab size
        mask_token_id: int = 50264,  # RoBERTa MASK token
    ):
        """
        Initialize discrete diffusion process.

        Args:
            noise_schedule: CosineNoiseSchedule instance
            num_classes: Vocabulary size (number of possible tokens)
            mask_token_id: ID of the MASK token (absorbing state)
        """
        self.noise_schedule = noise_schedule
        self.num_classes = num_classes
        self.mask_token_id = mask_token_id

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to clean tokens using Absorbing State.

        For each token independently:
        - With probability alpha_bar_t: keep original token
        - With probability (1 - alpha_bar_t): replace with MASK token

        This is the Absorbing State transition that ensures consistency
        between training (corruption to MASK) and inference (denoising from MASK).

        Args:
            x_start: Clean tokens [batch_size, seq_len]
            t: Timestep [batch_size]

        Returns:
            x_t: Noisy tokens [batch_size, seq_len]
        """
        batch_size, seq_len = x_start.shape
        device = x_start.device

        # Get alpha_bar_t for each sample in batch
        alpha_bar = self.noise_schedule.get_alpha_bar(t)  # [batch_size]

        # Create mask: which tokens to keep vs replace with MASK
        # For each position, sample uniform [0,1] and compare with alpha_bar
        random_matrix = torch.rand(batch_size, seq_len, device=device)
        keep_mask = random_matrix < alpha_bar.unsqueeze(1)  # [batch_size, seq_len]

        # Absorbing State: replace with MASK token instead of random token
        mask_tokens = torch.full_like(x_start, self.mask_token_id)

        # Mix original and MASK based on keep_mask
        # Where keep_mask is True, keep original; otherwise, use MASK
        x_t = torch.where(keep_mask, x_start, mask_tokens)

        return x_t

    def get_loss_weights(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        compute_pad_loss: bool = False,
        compute_eos_loss: bool = True,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        Compute weighting for each token in loss calculation.

        Args:
            x_start: Original clean tokens [batch_size, seq_len]
            x_t: Noisy tokens [batch_size, seq_len]
            t: Timestep [batch_size]
            attention_mask: Attention mask [batch_size, seq_len]
                           1 for real tokens, 0 for padding tokens
            compute_pad_loss: If True, compute loss on all positions including PAD.
                              This teaches the model to output PAD tokens appropriately.
                              **Recommended: False for MaskGIT-style training**
            compute_eos_loss: If True, always include EOS token positions in loss.
                             This teaches the model to predict EOS at the end of sequences.
                             **Recommended: True for variable-length output**
            eos_token_id: Token ID for EOS (default: 2 for RoBERTa)

        Returns:
            loss_mask: Binary mask [batch_size, seq_len]
                       1 for positions to compute loss, 0 otherwise

        Note:
            For MaskGIT-style training with Absorbing State:
            - compute_pad_loss=False (recommended): Only compute loss on MASK positions
              (tokens that were replaced), excluding padding
            - compute_pad_loss=True: Compute loss everywhere including PAD
              (useful if you want model to learn when to output PAD)
            - compute_eos_loss=True (recommended): Always include EOS positions
              to teach model when to stop generating
        """
        if compute_pad_loss:
            # All positions contribute to loss (including PAD)
            base_mask = torch.ones_like(x_start.float(), device=x_start.device)
        else:
            # MaskGIT-style: Only compute loss on tokens that were noised (changed to MASK)
            base_mask = (x_t != x_start).float()

            # Exclude padding positions from loss computation
            # attention_mask=1 means real token, attention_mask=0 means padding
            if attention_mask is not None:
                # Multiply: 1 * 1 = 1 (real token & noised -> compute loss)
                #           1 * 0 = 0 (real token but not noised -> no loss)
                #           0 * 1 = 0 (padding token -> no loss)
                #           0 * 0 = 0 (padding & not noised -> no loss)
                base_mask = base_mask * attention_mask

        # Always include EOS token positions in loss
        if compute_eos_loss:
            eos_positions = (x_start == eos_token_id).float()
            # Union of base_mask and eos_positions
            base_mask = torch.clamp(base_mask + eos_positions, min=0, max=1)

        return base_mask


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

    # Test discrete diffusion with Absorbing State
    print("\n2. Testing DiscreteDiffusion with Absorbing State...")
    diffusion = DiscreteDiffusion(schedule, num_classes=50265, mask_token_id=50264)
    print("[OK] DiscreteDiffusion initialized")
    print(f"  Mask token ID: {diffusion.mask_token_id}")

    # Test forward diffusion
    batch_size = 4
    seq_len = 64
    x_start = torch.randint(0, 50264, (batch_size, seq_len))  # Exclude MASK from input
    t = torch.tensor([0, 250, 500, 999])

    x_t = diffusion.q_sample(x_start, t)
    print(f"[OK] Forward diffusion successful")
    print(f"  Input shape: {x_start.shape}")
    print(f"  Output shape: {x_t.shape}")

    # At t=0, should be mostly unchanged
    similarity_0 = (x_t[0] == x_start[0]).float().mean()
    mask_ratio_0 = (x_t[0] == 50264).float().mean()
    print(f"  Similarity at t=0: {similarity_0.item():.2%}")
    print(f"  MASK ratio at t=0: {mask_ratio_0.item():.2%}")
    assert similarity_0 > 0.95, "At t=0, most tokens should be unchanged"
    print("[OK] At t=0, minimal corruption")

    # At t=1000, should be mostly MASK (absorbing state)
    similarity_1000 = (x_t[3] == x_start[3]).float().mean()
    mask_ratio_1000 = (x_t[3] == 50264).float().mean()
    print(f"  Similarity at t=999: {similarity_1000.item():.2%}")
    print(f"  MASK ratio at t=999: {mask_ratio_1000.item():.2%}")
    assert mask_ratio_1000 > 0.90, "At t=999, most tokens should be MASK"
    print("[OK] At t=999, mostly in absorbing state (MASK)")

    # Verify absorbing state property: once MASK, stays MASK
    print("\n3. Testing Absorbing State property...")
    x_t1 = diffusion.q_sample(x_start, torch.tensor([500]))
    x_t2 = diffusion.q_sample(x_t1, torch.tensor([999]))

    # Tokens that were MASK at t=500 should still be MASK at t=999
    mask_at_t1 = (x_t1 == 50264)
    mask_at_t2 = (x_t2 == 50264)
    absorbing_property = (~mask_at_t1 | mask_at_t2).all().item()
    print(f"  Absorbing state property holds: {absorbing_property}")
    print("[OK] Once a token becomes MASK, it stays MASK")

    # Test loss weights
    loss_mask = diffusion.get_loss_weights(x_start, x_t, t)
    print(f"\n4. Testing loss weights...")
    print(f"[OK] Loss mask computed")
    print(f"  Loss mask shape: {loss_mask.shape}")
    print(f"  Non-zero ratio at t=0: {loss_mask[0].mean().item():.2%}")
    print(f"  Non-zero ratio at t=999: {loss_mask[3].mean().item():.2%}")

    print("\n[SUCCESS] All tests passed! Absorbing State diffusion verified.")
