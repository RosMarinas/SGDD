"""
Unit tests for CosineNoiseSchedule and DiscreteDiffusion
"""

import pytest
import torch
from src.models.diffusion import CosineNoiseSchedule, DiscreteDiffusion


class TestCosineNoiseSchedule:
    """Test suite for CosineNoiseSchedule"""

    @pytest.fixture
    def schedule(self):
        """Create a cosine noise schedule for testing"""
        return CosineNoiseSchedule(num_timesteps=1000)

    def test_alpha_bar_monotonic_decrease(self, schedule):
        """Test that alpha_bar decreases monotonically"""
        t = torch.arange(0, 1000)
        alpha_bars = schedule.get_alpha_bar(t)

        # Check that each value is <= the previous
        for i in range(1, len(alpha_bars)):
            assert alpha_bars[i] <= alpha_bars[i-1], \
                f"alpha_bar[{i}]={alpha_bars[i]} > alpha_bar[{i-1}]={alpha_bars[i-1]}"

    def test_alpha_bar_range(self, schedule):
        """Test that alpha_bar is in valid range [0, 1]"""
        t = torch.arange(0, 1000)
        alpha_bars = schedule.get_alpha_bar(t)

        assert alpha_bars[0] <= 1.0, "alpha_bar[0] should be <= 1"
        assert alpha_bars[0] >= 0.0, "alpha_bar[0] should be >= 0"
        assert alpha_bars[-1] <= 1.0, "alpha_bar[-1] should be <= 1"
        assert alpha_bars[-1] >= 0.0, "alpha_bar[-1] should be >= 0"

    def test_alpha_bar_at_boundaries(self, schedule):
        """Test alpha_bar values at boundaries"""
        alpha_bar_0 = schedule.get_alpha_bar(torch.tensor([0]))
        alpha_bar_end = schedule.get_alpha_bar(torch.tensor([999]))

        # At t=0, should be close to 1 (no noise)
        assert alpha_bar_0.item() > 0.99, f"alpha_bar[0] should be ~1.0, got {alpha_bar_0}"

        # At t=999, should be close to 0 (max noise)
        assert alpha_bar_end.item() < 0.01, f"alpha_bar[999] should be ~0.0, got {alpha_bar_end}"

    def test_beta_range(self, schedule):
        """Test that beta values are in valid range"""
        betas = schedule.betas

        assert torch.all(betas >= 0), "All betas should be >= 0"
        assert torch.all(betas <= 1), "All betas should be <= 1"
        # Last few betas may reach 0.999 due to clamping
        assert torch.all(betas <= 0.9999), "All betas should be <= 0.9999 (clamped)"

    def test_different_num_timesteps(self):
        """Test schedule with different number of timesteps"""
        for num_timesteps in [100, 500, 1000, 2000]:
            schedule = CosineNoiseSchedule(num_timesteps=num_timesteps)
            assert schedule.num_timesteps == num_timesteps

            # Test monotonicity
            t = torch.arange(0, num_timesteps)
            alpha_bars = schedule.get_alpha_bar(t)
            for i in range(1, len(alpha_bars)):
                assert alpha_bars[i] <= alpha_bars[i-1]


class TestDiscreteDiffusion:
    """Test suite for DiscreteDiffusion"""

    @pytest.fixture
    def diffusion(self):
        """Create a discrete diffusion process for testing"""
        schedule = CosineNoiseSchedule(num_timesteps=1000)
        return DiscreteDiffusion(schedule, num_classes=50265)

    def test_q_sample_shape(self, diffusion):
        """Test that q_sample produces correct output shape"""
        batch_size = 4
        seq_len = 64
        x_start = torch.randint(0, 50265, (batch_size, seq_len))
        t = torch.randint(0, 1000, (batch_size,))

        x_t = diffusion.q_sample(x_start, t)

        assert x_t.shape == x_start.shape, \
            f"Expected shape {x_start.shape}, got {x_t.shape}"

    def test_q_sample_at_t_zero(self, diffusion):
        """Test that at t=0, tokens are mostly unchanged"""
        x_start = torch.randint(0, 50265, (2, 64))
        t = torch.zeros(2, dtype=torch.long)

        x_t = diffusion.q_sample(x_start, t)

        # At t=0, most tokens should be unchanged
        similarity = (x_t == x_start).float().mean()
        assert similarity > 0.95, \
            f"At t=0, most tokens should be unchanged, but similarity={similarity}"

    def test_q_sample_at_max_t(self, diffusion):
        """Test that at max t, tokens are mostly changed"""
        x_start = torch.randint(0, 50265, (2, 64))
        t = torch.full((2,), 999)

        x_t = diffusion.q_sample(x_start, t)

        # At max t, most tokens should be changed
        similarity = (x_t == x_start).float().mean()
        assert similarity < 0.10, \
            f"At max t, most tokens should be changed, but similarity={similarity}"

    def test_noise_increases_with_t(self, diffusion):
        """Test that noise level increases with t"""
        x_start = torch.randint(0, 50265, (10, 64))
        t_values = torch.tensor([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])

        x_t = diffusion.q_sample(x_start, t_values)
        similarities = (x_t == x_start).float().mean(dim=1)

        # Similarity should generally decrease as t increases
        # Allow some tolerance for randomness
        for i in range(1, len(similarities)):
            assert similarities[i] <= similarities[0] + 0.1, \
                f"Similarity at t={t_values[i]} ({similarities[i]}) should be <= similarity at t=0 ({similarities[0]})"

    def test_custom_noise(self, diffusion):
        """Test q_sample with custom noise"""
        x_start = torch.randint(0, 50265, (2, 64))
        noise = torch.randint(0, 50265, (2, 64))
        t = torch.tensor([500, 500])

        x_t = diffusion.q_sample(x_start, t, noise=noise)

        # Check that output is either from x_start or noise
        for i in range(x_start.shape[0]):
            for j in range(x_start.shape[1]):
                assert x_t[i, j] == x_start[i, j] or x_t[i, j] == noise[i, j], \
                    f"Output should be from x_start or noise"

    def test_loss_weights_shape(self, diffusion):
        """Test that get_loss_weights produces correct shape"""
        batch_size = 4
        seq_len = 64
        x_start = torch.randint(0, 50265, (batch_size, seq_len))
        t = torch.randint(0, 1000, (batch_size,))

        x_t = diffusion.q_sample(x_start, t)
        loss_mask = diffusion.get_loss_weights(x_start, x_t, t)

        assert loss_mask.shape == (batch_size, seq_len), \
            f"Expected shape ({batch_size}, {seq_len}), got {loss_mask.shape}"

    def test_loss_weights_binary(self, diffusion):
        """Test that loss weights are binary (0 or 1)"""
        x_start = torch.randint(0, 50265, (4, 64))
        t = torch.randint(0, 1000, (4,))

        x_t = diffusion.q_sample(x_start, t)
        loss_mask = diffusion.get_loss_weights(x_start, x_t, t)

        # Check that values are 0 or 1
        unique_values = torch.unique(loss_mask)
        assert len(unique_values) <= 2, "Loss mask should be binary"
        assert all(v in [0.0, 1.0] for v in unique_values), "Loss mask should only contain 0 and 1"

    def test_loss_weights_detect_changes(self, diffusion):
        """Test that loss weights detect token changes"""
        x_start = torch.randint(0, 50265, (2, 64))
        t = torch.tensor([0, 999])  # Low noise and high noise

        x_t = diffusion.q_sample(x_start, t)
        loss_mask = diffusion.get_loss_weights(x_start, x_t, t)

        # At t=0, few tokens should be changed
        assert loss_mask[0].sum() < 10, "At t=0, few tokens should have loss_mask=1"

        # At t=999, many tokens should be changed
        assert loss_mask[1].sum() > 50, "At t=999, many tokens should have loss_mask=1"

    def test_device_compatibility(self, diffusion):
        """Test that diffusion works on GPU if available"""
        if torch.cuda.is_available():
            # Move noise schedule tensors to GPU
            diffusion.noise_schedule.alphas_cumprod = diffusion.noise_schedule.alphas_cumprod.cuda()
            diffusion.noise_schedule.betas = diffusion.noise_schedule.betas.cuda()

            x_start = torch.randint(0, 50265, (2, 64)).cuda()
            t = torch.randint(0, 1000, (2,)).cuda()

            x_t = diffusion.q_sample(x_start, t)
            loss_mask = diffusion.get_loss_weights(x_start, x_t, t)

            assert x_t.device.type == "cuda"
            assert loss_mask.device.type == "cuda"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
