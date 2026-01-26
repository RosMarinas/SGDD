"""
SGDD Model - Full Semantic-Guided Discrete Diffusion Model

Combines frozen semantic encoder, diffusion process, and trainable decoder
into a complete model for text generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .encoder import SemanticEncoder
from .decoder import DiffusionDecoder
from .diffusion import CosineNoiseSchedule, DiscreteDiffusion


@dataclass
class SGDDConfig:
    """Configuration for SGDD model"""
    # Encoder
    encoder_model: str = "BAAI/bge-m3"
    semantic_dim: int = 1024  # High dimension to preserve semantics
    kl_weight: float = 0.001  # KL divergence weight
    kl_anneal_steps: int = 10000  # KL annealing steps
    whitening_stats_path: Optional[str] = None


    # Decoder
    decoder_dim: int = 256  # Reduced dimension for lightweight model
    num_layers: int = 4  # Reduced layers
    num_heads: int = 4  # 256 / 64 = 4 heads
    ffn_dim: int = 1024  # 4 * 256
    max_len: int = 128
    dropout: float = 0.1

    # Diffusion
    num_diffusion_steps: int = 1000
    vocab_size: int = 250002  # BGE-M3 / XLM-R vocab size

    # Training
    cfg_prob: float = 0.1  # Probability of unconditional training (CFG)
    use_self_conditioning: bool = True  # Enable self-conditioning during training
    compute_pad_loss: bool = False  # Compute loss on PAD positions (teaches model to output PAD)
    compute_eos_loss: bool = True  # Compute loss on EOS positions (teaches model to output EOS)
    eos_token_id: int = 2  # RoBERTa / XLM-R EOS token ID


class SGDDModel(nn.Module):
    """
    Semantic-Guided Discrete Diffusion Model.

    Combines:
    - Frozen semantic encoder (BGE-M3)
    - Noise schedule (cosine)
    - Trainable diffusion decoder

    Supports classifier-free guidance (CFG) for better semantic conditioning.
    """

    def __init__(self, config: SGDDConfig):
        """
        Initialize SGDD model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Semantic encoder (frozen) with Variational Information Bottleneck
        self.encoder = SemanticEncoder(
            model_name=config.encoder_model,
            hidden_dim=config.semantic_dim,  # Output dimension of encoder (VIB)
            kl_weight=config.kl_weight,
            kl_anneal_steps=config.kl_anneal_steps,
            whitening_stats_path=config.whitening_stats_path,
        )

        # Tokenizer from encoder
        self.tokenizer = self.encoder.tokenizer

        # Decoder Embedding Initialization with Pretrained Encoder Weights
        # Initialize decoder embeddings with projected encoder embeddings
        # Source: Encoder Dim (1024) -> Target: Decoder Dim (256)
        
        # Get source dimension from encoder
        source_dim = self.encoder.input_dim
        
        encoder_embeddings_proj = self._project_embeddings(
            self.encoder.model.embeddings.word_embeddings.weight,
            source_dim=source_dim,
            target_dim=config.decoder_dim,
        )

        # Diffusion decoder (trainable)
        self.decoder = DiffusionDecoder(
            vocab_size=config.vocab_size,
            hidden_dim=config.decoder_dim,
            semantic_dim=config.semantic_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            max_len=config.max_len,
            dropout=config.dropout,
            pretrained_embeddings=encoder_embeddings_proj,
        )

        # Noise schedule
        self.noise_schedule = CosineNoiseSchedule(
            num_timesteps=config.num_diffusion_steps
        )

        # Diffusion process
        self.diffusion = DiscreteDiffusion(
            noise_schedule=self.noise_schedule,
            num_classes=config.vocab_size,
            mask_token_id=self.tokenizer.mask_token_id,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        cfg_uncond: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model with AdaLN and optional self-conditioning.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            timestep: Diffusion timestep [batch] (if None, sample random)
            cfg_uncond: If True, randomly drop semantic vector with cfg_prob for CFG training.
                        This is the standard classifier-free guidance training strategy,
                        NOT forcing all samples to be unconditional.

        Returns:
            logits: Predicted logits [batch, seq_len, vocab_size]
            target_tokens: Target tokens for loss [batch, seq_len]
            loss_mask: Loss mask [batch, seq_len]
            kl_loss: KL divergence [batch]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Sample random timestep if not provided
        # IMPORTANT: Use squared sampling to bias towards high timesteps (more MASK)
        # This ensures the model learns to generate from heavily corrupted inputs,
        # which is crucial for inference where we start from all MASK tokens.
        # Without this bias, the model would only see 50% MASK at t=500 and
        # fail to learn to rely on semantic vector z for generation.
        if timestep is None:
            # Sample from [0, 1] with square bias towards 1 (high corruption)
            timestep = torch.rand(batch_size, device=device)
            timestep = timestep ** 0.5  # Square root bias: higher timesteps more likely
            # Convert to discrete timestep
            timestep = (timestep * self.noise_schedule.num_timesteps).long()

        # Extract semantic vector from encoder with KL loss
        semantic_vector, kl_loss = self.encoder(
            input_ids,
            attention_mask,
            return_kl=True
        )
        # [batch, hidden_dim]

        # Apply classifier-free guidance during training
        if self.training and cfg_uncond:
            # Randomly select samples to make unconditional
            uncond_mask = torch.rand(batch_size, device=device) < self.config.cfg_prob
            semantic_vector = torch.where(
                uncond_mask.unsqueeze(1),
                torch.zeros_like(semantic_vector),
                semantic_vector
            )

        # Forward diffusion: add noise to input tokens
        x_start = input_ids.clone()
        x_t = self.diffusion.q_sample(x_start, timestep)
        # [batch, seq_len]

        # Self-conditioning: 50% probability during training (if enabled in config)
        prev_pred = None
        if self.training and self.config.use_self_conditioning:
            use_sc = torch.rand(1).item() < 0.5
            if use_sc:
                with torch.no_grad():
                    # Get previous prediction without self-conditioning
                    prev_logits = self.decoder(x_t, semantic_vector, timestep, prev_pred=None)
                    prev_pred = prev_logits.argmax(dim=-1)

        # Decoder: predict clean tokens from noisy tokens (with optional self-conditioning)
        logits = self.decoder(x_t, semantic_vector, timestep, prev_pred=prev_pred)
        # [batch, seq_len, vocab_size]

        # Compute loss mask with EOS support
        # Pass attention_mask to exclude padding positions from loss computation
        loss_mask = self.diffusion.get_loss_weights(
            x_start, x_t, timestep,
            attention_mask=attention_mask,
            compute_pad_loss=self.config.compute_pad_loss,
            compute_eos_loss=self.config.compute_eos_loss,
            eos_token_id=self.config.eos_token_id
        )
        # [batch, seq_len]

        return logits, x_start, loss_mask, kl_loss

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_tokens: torch.Tensor,
        loss_mask: torch.Tensor,
        kl_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked cross-entropy loss + KL divergence.

        Args:
            logits: Predicted logits [batch, seq_len, vocab_size]
            target_tokens: Target tokens [batch, seq_len]
            loss_mask: Loss mask [batch, seq_len]
            kl_loss: KL divergence per sample [batch]

        Returns:
            loss: Scalar loss (reconstruction + KL)
        """
        # Compute cross-entropy loss for each position
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            target_tokens.view(-1),
            reduction='none'
        ).view(logits.shape[:2])
        # [batch, seq_len]

        # Apply loss mask (only compute loss on noised tokens)
        recon_loss = (loss * loss_mask).sum() / loss_mask.sum()

        # Add KL divergence (kl_loss is already weighted by kl_weight in encoder)
        kl_loss_mean = kl_loss.mean()
        total_loss = recon_loss + kl_loss_mean

        return total_loss

    def _project_embeddings(
        self,
        encoder_embeddings: torch.Tensor,
        source_dim: int,
        target_dim: int = 512,
    ) -> torch.Tensor:
        """
        Project encoder embeddings from source_dim to target_dim for decoder initialization.

        Uses a simple linear projection (without bias) to preserve semantic structure
        while reducing dimensionality. This is done once during initialization and
        the projection matrix is discarded afterwards.

        Args:
            encoder_embeddings: Source word embeddings [vocab_size, source_dim]
            source_dim: Source dimension (768 for RoBERTa, 1024 for BGE-M3)
            target_dim: Target dimension (512 for decoder)

        Returns:
            projected_embeddings: Projected embeddings [vocab_size, target_dim]
        """
        device = encoder_embeddings.device

        # Create a projection layer without bias
        # Using Xavier initialization for better gradient flow
        projection = nn.Linear(source_dim, target_dim, bias=False)
        nn.init.xavier_uniform_(projection.weight)

        # Move to same device as embeddings
        projection = projection.to(device)

        # Project embeddings: [vocab_size, source_dim] -> [vocab_size, target_dim]
        with torch.no_grad():
            projected_embeddings = projection(encoder_embeddings)

        # Clean up: we don't need the projection layer anymore
        del projection

        return projected_embeddings

    @torch.no_grad()
    def generate(
        self,
        input_text: Optional[str | list[str]] = None,
        semantic_vector: Optional[torch.Tensor] = None,
        num_steps: int = 16,
        guidance_scale: float = 2.0,
        temperature: float = 1.0,
        max_length: int = 64,
        repetition_penalty: float = 1.0,
        top_k: int = -1,
        sample_z: bool = False,
    ) -> str | list[str]:
        """
        Generate text from semantic vector using MaskGIT-style decoding.

        Args:
            input_text: Input text or list of texts to encode. Optional if semantic_vector provided.
            semantic_vector: Pre-computed semantic vector [batch, hidden_dim]. Optional if input_text provided.
            num_steps: Number of decoding steps
            guidance_scale: CFG guidance scale
            temperature: Sampling temperature
            max_length: Maximum generation length
            repetition_penalty: Penalty for repeating tokens
            top_k: Only sample from top-k tokens (-1 = disabled)
            sample_z: If True, sample z ~ N(mu, sigma) for stochastic inference. If False, use z=mu (deterministic).

        Returns:
            generated_text: Generated text or list of texts
        """
        self.eval()
        device = next(self.parameters()).device

        # Handle input
        if semantic_vector is not None:
            # Use provided semantic vector
            pass
        elif input_text is not None:
            # Encode input text
            if isinstance(input_text, str):
                input_texts = [input_text]
                is_single = True
            else:
                input_texts = input_text
                is_single = False

            encoded = self.tokenizer(
                input_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            if sample_z:
                # Stochastic: temporarily enable training mode for sampling
                was_training = self.encoder.training
                self.encoder.train()
                semantic_vector, _ = self.encoder(input_ids, attention_mask, return_kl=True)
                self.encoder.train(was_training)
            else:
                # Deterministic: uses mu (default)
                semantic_vector, _ = self.encoder(input_ids, attention_mask, return_kl=True)
        else:
            raise ValueError("Must provide either input_text or semantic_vector")

        # Get batch size from semantic vector
        batch_size = semantic_vector.shape[0]
        
        # Start from all MASK tokens
        mask_token_id = self.tokenizer.mask_token_id
        current_tokens = torch.full(
            (batch_size, max_length),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )
        # [batch, max_length]

        # MaskGIT iterative decoding: t goes from 1.0 down to 0.0
        # We perform num_steps iterations.
        # t_schedule: [1.0, ..., >0]
        t_schedule = torch.linspace(1.0, 0.0, num_steps + 1)
        
        for step_idx in range(num_steps):
            t_curr = t_schedule[step_idx]
            t_next = t_schedule[step_idx + 1]
            
            # Convert continuous time to discrete timestep for model input
            # Clamp to [0, num_timesteps-1]
            timestep_val = int(t_curr * (self.noise_schedule.num_timesteps - 1))
            timestep_val = max(0, min(self.noise_schedule.num_timesteps - 1, timestep_val))
            
            timestep = torch.full(
                (batch_size,), 
                timestep_val,
                device=device, 
                dtype=torch.long
            )

            # Prepare self-conditioning from previous step
            prev_pred = None
            if step_idx > 0:
                # Use previous iteration's tokens as self-conditioning
                prev_pred = current_tokens.clone()
            
            # Calculate target mask ratio for the end of this step
            # We want to match the noise schedule at t_next
            timestep_next_val = int(t_next * (self.noise_schedule.num_timesteps - 1))
            timestep_next_val = max(0, min(self.noise_schedule.num_timesteps - 1, timestep_next_val))
            
            # Get alpha_bar at t_next
            # alpha_bar represents probability of KEEPING a token
            # So mask ratio = 1 - alpha_bar
            alpha_bar_next = self.noise_schedule.get_alpha_bar(torch.tensor([timestep_next_val], device=device))
            target_mask_ratio = 1.0 - alpha_bar_next.item()
            
            # Ensure final step clears everything (ratio=0)
            if step_idx == num_steps - 1:
                target_mask_ratio = 0.0

            # Conditional prediction
            logits_cond = self.decoder(current_tokens, semantic_vector, timestep, prev_pred=prev_pred)

            # Unconditional prediction (compute in-place to save memory)
            semantic_vector_uncond = torch.zeros_like(semantic_vector)
            with torch.no_grad():  # Don't need gradients for unconditional
                logits_uncond = self.decoder(current_tokens, semantic_vector_uncond, timestep, prev_pred=prev_pred)

            # Classifier-free guidance
            guided_logits = logits_cond + guidance_scale * (logits_cond - logits_uncond)

            # Debug logging for first step (first sample only)
            if step_idx == 0:
                print(f"[DEBUG] Step 0 sample 0: logits range=[{logits_cond[0].min():.2f}, {logits_cond[0].max():.2f}]")

            # Free memory
            del logits_cond, logits_uncond
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for b in range(batch_size):
                    unique_tokens, counts = torch.unique(
                        current_tokens[b][current_tokens[b] != mask_token_id],
                        return_counts=True
                    )
                    for token, count in zip(unique_tokens, counts):
                        guided_logits[b, :, token] /= (repetition_penalty ** count)

            probs = F.softmax(guided_logits / temperature, dim=-1)

            # Sample tokens for masked positions
            masked_positions = (current_tokens == mask_token_id)
            # [batch, seq_len]

            if masked_positions.sum() == 0:
                break
            
            # Determine which tokens to unmask
            # We unmask based on confidence
            
            for b in range(batch_size):
                mask_indices = torch.where(masked_positions[b])[0]
                num_currently_masked = len(mask_indices)
                if num_currently_masked == 0:
                    continue
                
                # Calculate how many tokens should remain masked
                num_to_keep_masked = int(max_length * target_mask_ratio)
                num_to_unmask = max(1, num_currently_masked - num_to_keep_masked)
                
                # Don't unmask more than we have
                num_to_unmask = min(num_to_unmask, num_currently_masked)
                
                probs_at_masked = probs[b][mask_indices]
                # [num_masked_b, vocab]
                
                # Sample candidate tokens
                # Top-k sampling
                if top_k > 0 and top_k < probs_at_masked.size(-1):
                    top_k_probs, top_k_indices = torch.topk(
                        probs_at_masked,
                        k=min(top_k, probs_at_masked.size(-1)),
                        dim=-1
                    )
                    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                    sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
                    sampled_tokens = top_k_indices.gather(1, sampled_indices.unsqueeze(-1)).squeeze(-1)
                else:
                    sampled_tokens = torch.multinomial(probs_at_masked, num_samples=1).squeeze(-1)
                
                # Confidence-based selection for unmasking
                # We pick the `num_to_unmask` tokens with highest confidence
                # Confidence is the probability of the sampled token
                # gathered_probs = probs_at_masked.gather(1, sampled_tokens.unsqueeze(1)).squeeze(1)
                # Note: 'confidence' usually means max probability, but here we used sampled tokens.
                # Standard MaskGIT uses max probability as confidence score.
                
                confidence_scores = probs_at_masked.max(dim=-1)[0]
                
                # Select top confidence indices to unmask
                _, top_conf_indices = torch.topk(confidence_scores, num_to_unmask)
                
                # Indices in the original sequence
                indices_to_update = mask_indices[top_conf_indices]
                tokens_to_fill = sampled_tokens[top_conf_indices]
                
                current_tokens[b, indices_to_update] = tokens_to_fill

        # Decode tokens
        generated_ids = current_tokens.cpu().numpy()
        eos_token_id = self.tokenizer.eos_token_id
        
        decoded_texts = []
        for b in range(batch_size):
            ids = generated_ids[b]
            eos_positions = np.where(ids == eos_token_id)[0]
            if len(eos_positions) > 0:
                ids = ids[:eos_positions[0]]
            
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            decoded_texts.append(text)

        if input_text is not None and isinstance(input_text, str):
            return decoded_texts[0]
        return decoded_texts

    def get_num_params(self) -> int:
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing SGDDModel...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        config = SGDDConfig()
        model = SGDDModel(config).to(device)
        print(f"[OK] Model initialized with {config.encoder_model}")
        
        # Test forward pass with correct vocab size
        vocab_size = config.vocab_size
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        logits, target_tokens, loss_mask, kl_loss = model(input_ids, attention_mask, cfg_uncond=True)
        print(f"[OK] Forward pass successful")
        print(f"  Logits shape: {logits.shape}")
        
    except Exception as e:
        print(f"Error testing SGDDModel: {e}")
        exit(1)

    # Count parameters
    total_params = model.get_num_params()
    trainable_params = model.get_num_trainable_params()
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test loss computation
    loss = model.compute_loss(logits, target_tokens, loss_mask, kl_loss)
    print(f"[OK] Loss computed: {loss.item():.4f}")

    # Test generation (simple)
    model.eval()
    with torch.no_grad():
        generated = model.generate("Hello world", num_steps=4, max_length=32)
    print(f"[OK] Generation successful")
    print(f"  Generated: {generated}")

    print("\n[SUCCESS] All tests passed!")