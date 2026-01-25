# Semantic-Guided Discrete Diffusion (SGDD) Language Model

A lightweight, non-autoregressive discrete diffusion language model that uses a frozen semantic encoder (BGE-M3) and trainable diffusion decoder (AdaLN-Zero) for short text generation.

## Project Status

**BGE-M3 + BookCorpus Update Complete** ✅

- **Total Parameters**: ~650M (including frozen BGE-M3 567M)
- **Trainable Parameters**: ~67M
- **Framework**: PyTorch 2.5+ + Transformers 4.40+
- **Hardware**: RTX 4070 Ti Super

## Architecture

### 1. Semantic Encoder (Frozen)
- **BGE-M3** (BAAI/bge-m3, 567M parameters, frozen)
- [CLS] pooling for high-quality semantic extraction
- **Variational Information Bottleneck (VIB)** for latent space regularization
- 1024-dim semantic vector

### 2. Diffusion Decoder (Trainable)
- **Lightweight Strategy**: 2-layer bidirectional Transformer
- **AdaLN-Zero** architecture (Adaptive Layer Normalization)
- 256 hidden dimension, 4 attention heads
- Rotary Position Embeddings (RoPE)
- ~67M trainable parameters (mostly embeddings)

### 3. Diffusion Process
- Cosine noise schedule (1000 steps)
- Discrete token diffusion (Masking)
- MaskGIT-style inference (16 steps)
- Classifier-Free Guidance (CFG)

## Implemented Features

### Core Models (`src/models/`)
- [x] `encoder.py` - BGE-M3 encoder with VIB
- [x] `decoder.py` - AdaLN-Zero decoder
- [x] `diffusion.py` - Discrete diffusion process
- [x] `sgdd.py` - Complete SGDD model

### Training Infrastructure (`src/utils/`)
- [x] `config.py` - Configuration system (YAML support)
- [x] `data.py` - Data pipeline (**BookCorpus** support)
- [x] `metrics.py` - Evaluation metrics
- [x] `checkpoints.py` - Checkpoint management

### Training & Evaluation (`src/`)
- [x] `train.py` - Complete training loop with WandB logging
- [x] `evaluate.py` - Model evaluation script

## Quick Start

### Install Dependencies

```bash
uv sync
```

### Training on BookCorpus

```bash
# Train on BookCorpus (text reconstruction)
uv run python src/train.py --config configs/phase1_vib.yaml
```

### Evaluation

```bash
# Evaluate trained model
uv run python src/evaluate.py --config configs/phase1_vib.yaml --checkpoint checkpoints/phase1_vib_bge
```

## Model Specifications

| Component | Parameters | Status |
|-----------|------------|--------|
| BGE-M3 Encoder (frozen) | ~567M | [x] |
| SGDD Decoder (trainable) | ~67M | [x] |
| **Total** | **~634M** | [x] |
| **Trainable** | **~67M** | [x] |

## Project Structure

```
SGDD/
├── src/
│   ├── models/              # Core model components
│   │   ├── encoder.py       # Semantic encoder (BGE-M3 + VIB)
│   │   ├── decoder.py       # Diffusion decoder (AdaLN)
│   │   ├── diffusion.py     # Noise schedule
│   │   └── sgdd.py          # Complete model
│   ├── utils/               # Training utilities
│   │   ├── config.py        # Configuration system
│   │   ├── data.py          # Data pipeline (BookCorpus)
│   │   ├── metrics.py       # Evaluation metrics
│   │   └── checkpoints.py   # Checkpoint management
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
├── configs/                 # Configuration files
│   ├── phase1_vib.yaml      # Main training config
│   └── phase1_whitening.yaml # Whitening optimization
├── data/                    # Dataset directory
├── checkpoints/             # Model checkpoints
├── plan.md                  # Implementation plan
└── README.md                # This file
```

## Tech Stack

- Python 3.11+
- PyTorch 2.5+
- Transformers 4.40+
- Datasets 3.0+
- WandB (Logging)
- uv (Package Manager)

---

**Last Updated**: 2026-01-25