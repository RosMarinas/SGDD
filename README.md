# Semantic-Guided Discrete Diffusion (SGDD) Language Model

A lightweight, non-autoregressive discrete diffusion language model that uses a frozen semantic encoder and trainable diffusion decoder for short text generation.

## Project Status

**Phase 0-3 Complete** (100%) - Ready for Training! ✅

- **Total Parameters**: ~176M
- **Trainable Parameters**: ~51M
- **Test Coverage**: 22 unit tests passing
- **Framework**: PyTorch 2.9.1 + Transformers 4.57
- **Hardware**: RTX 4070 Ti Super

## Architecture

### 1. Semantic Encoder (Frozen)
- RoBERTa-base (125M parameters, frozen)
- Mean pooling for semantic vector extraction
- 768-dim -> 512-dim projection

### 2. Diffusion Decoder (Trainable)
- 6-layer bidirectional Transformer
- 512 hidden dimension, 8 attention heads
- Rotary Position Embeddings (RoPE)
- Cross-attention mechanism
- ~51M trainable parameters

### 3. Diffusion Process
- Cosine noise schedule (1000 steps)
- Discrete token diffusion
- MaskGIT-style inference (16 steps)
- Classifier-Free Guidance (CFG)

## Implemented Features

### Core Models (`src/models/`)
- [x] `encoder.py` - Semantic encoder
- [x] `diffusion.py` - Noise schedule and diffusion process
- [x] `decoder.py` - Bidirectional transformer decoder
- [x] `sgdd.py` - Complete SGDD model

### Training Infrastructure (`src/utils/`)
- [x] `config.py` - Configuration system (YAML support)
- [x] `data.py` - Data pipeline (Wikipedia & QQP datasets)
- [x] `sampling.py` - MaskGIT sampling with CFG
- [x] `metrics.py` - Evaluation metrics (BLEU, EM, Perplexity)
- [x] `checkpoints.py` - Checkpoint management

### Training & Evaluation (`src/`)
- [x] `train.py` - Complete training loop with WandB logging
- [x] `evaluate.py` - Model evaluation script

### Test Suite (`tests/`)
- [x] `test_encoder.py` - Encoder tests (8 tests)
- [x] `test_diffusion.py` - Diffusion tests (14 tests)

## Quick Start

### Install Dependencies

```bash
uv sync
```

### Phase 1: Wikipedia Reconstruction Training

```bash
# Train on Wikipedia dataset (text reconstruction)
uv run python src/train.py --config configs/phase1_wiki.yaml
```

### Phase 2: QQP Paraphrase Fine-tuning

```bash
# Fine-tune on QQP dataset (paraphrase generation)
uv run python src/train.py --config configs/phase2_qqp.yaml --resume checkpoints/phase1_wiki/best_model.pt
```

### Evaluation

```bash
# Evaluate trained model
uv run python src/evaluate.py --checkpoint checkpoints/phase1_wiki --dataset wikipedia --num_samples 1000
```

### Run Tests

```bash
# Test encoder
uv run pytest tests/test_encoder.py -v

# Test diffusion
uv run pytest tests/test_diffusion.py -v

# Run all tests
uv run pytest tests/ -v

# Test complete model
uv run python -m src.models.sgdd
```

## Model Specifications

| Component | Parameters | Status |
|-----------|------------|--------|
| RoBERTa Encoder (frozen) | ~125M | [x] |
| SGDD Decoder (trainable) | ~51M | [x] |
| **Total** | **~176M** | [x] |
| **Trainable** | **~51M** | [x] |

## Performance Metrics

- **Inference**: ~3-4 seconds per generation (16 steps)
- **Memory**: ~2GB GPU memory (FP16)
- **Training**: TBD (Phase 4 pending)

## Project Structure

```
Semantic_Guided_Discrete_Diffusion/
├── src/
│   ├── models/              # Core model components
│   │   ├── encoder.py       # Semantic encoder (RoBERTa)
│   │   ├── decoder.py       # Diffusion decoder
│   │   ├── diffusion.py     # Noise schedule
│   │   └── sgdd.py          # Complete model
│   ├── utils/               # Training utilities
│   │   ├── config.py        # Configuration system
│   │   ├── data.py          # Data pipeline
│   │   ├── sampling.py      # MaskGIT sampling
│   │   ├── metrics.py       # Evaluation metrics
│   │   └── checkpoints.py   # Checkpoint management
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
├── configs/                 # Configuration files
│   ├── phase1_wiki.yaml     # Wikipedia reconstruction
│   └── phase2_qqp.yaml      # QQP paraphrase
├── tests/                   # Unit tests
│   ├── test_encoder.py
│   └── test_diffusion.py
├── checkpoints/             # Model checkpoints (created during training)
├── plan.md                  # Detailed implementation plan
└── README.md                # This file
```

## Tech Stack

- Python 3.11+
- PyTorch 2.9.1 (CUDA 12.8)
- Transformers 4.57
- Datasets 4.4
- pytest 9.0

## Documentation

See [plan.md](plan.md) for detailed implementation plan

## Contributing

Currently in development, core model implementation is complete.

## License

MIT License

---

**Last Updated**: 2026-01-11
