"""
检查Mu向量相似度和各向异性指标

运行方式:
uv run python temp/check_isotropy.py --config configs/phase1_vib.yaml
"""

import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


from models.sgdd import SGDDModel, SGDDConfig as ModelConfig
from models.encoder import SemanticEncoder
from utils.config import SGDDConfig as Config
from utils.data import get_dataloader


def compute_cosine_similarity(matrix):
    """计算余弦相似度矩阵"""
    # 归一化
    norm = matrix.norm(dim=1, keepdim=True)
    normalized = matrix / (norm + 1e-8)

    # 计算相似度矩阵
    sim_matrix = torch.mm(normalized, normalized.t())

    return sim_matrix


def analyze_isotropy(vectors, title="Mu Vectors"):
    """分析向量的各向异性"""
    batch_size, dim = vectors.shape

    # 1. 计算余弦相似度矩阵
    sim_matrix = compute_cosine_similarity(vectors)

    # 2. 提取上三角矩阵（排除对角线）
    mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
    off_diagonal_sims = sim_matrix[mask]

    # 3. 统计指标
    mean_sim = off_diagonal_sims.mean().item()
    std_sim = off_diagonal_sims.std().item()
    max_sim = off_diagonal_sims.max().item()
    min_sim = off_diagonal_sims.min().item()

    # 4. 计算各向异性指标
    # - 平均相似度越接近0，各向同性越好
    # - 标准差越小，分布越均匀
    isotropy_score = abs(mean_sim)  # 越小越好

    print(f"\n{'='*60}")
    print(f"{title} Isotropy Analysis")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"Vector dimension: {dim}")
    print(f"\nCosine Similarity Statistics (off-diagonal):")
    print(f"  Mean:   {mean_sim:+.4f} (closer to 0 is better)")
    print(f"  Std:    {std_sim:.4f}")
    print(f"  Max:    {max_sim:+.4f}")
    print(f"  Min:    {min_sim:+.4f}")
    print(f"\nIsotropy Score: {isotropy_score:.4f} (lower is better)")
    print(f"  Target: < 0.1 for good isotropy")
    print(f"  Current: {'✓ GOOD' if isotropy_score < 0.1 else '✗ NEEDS IMPROVEMENT'}")

    return {
        "mean_similarity": mean_sim,
        "std_similarity": std_sim,
        "max_similarity": max_sim,
        "min_similarity": min_sim,
        "isotropy_score": isotropy_score,
    }


def check_model_isotropy(config_path: str, num_batches: int = 10):
    """检查模型训练时的各向异性指标"""
    # 加载配置
    config = Config.from_yaml(config_path)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建模型
    print("\nLoading model...")
    from models.encoder import SemanticEncoder
    temp_encoder = SemanticEncoder(
        model_name=config.model.encoder_name,
        hidden_dim=config.model.semantic_dim,
    )
    vocab_size = temp_encoder.tokenizer.vocab_size
    del temp_encoder

    model_config = ModelConfig(
        encoder_model=config.model.encoder_name,
        hidden_dim=config.model.semantic_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        ffn_dim=config.model.ffn_dim,
        max_len=config.model.max_length,
        dropout=config.model.dropout,
        num_diffusion_steps=config.model.num_diffusion_steps,
        vocab_size=vocab_size,
        cfg_prob=config.training.cfg_drop_prob,
        use_self_conditioning=config.model.use_self_conditioning,
        compute_pad_loss=config.model.compute_pad_loss,
        kl_weight=getattr(config.model, 'kl_weight', 0.001),
        kl_anneal_steps=getattr(config.model, 'kl_anneal_steps', 10000),
        kl_threshold=getattr(config.model, 'kl_threshold', 0.0),
        contrastive_weight=getattr(config.model, 'contrastive_weight', 0.0),
    )

    model = SGDDModel(model_config).to(device)

    # 检查是否使用了 BatchNorm
    print("\n" + "="*60)
    print("Architecture Check")
    print("="*60)

    has_batchnorm = False
    for name, module in model.encoder.mu_layer.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            has_batchnorm = True
            print(f"✓ Found BatchNorm1d in mu_layer: {name}")
            break

    for name, module in model.encoder.logvar_layer.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            print(f"✓ Found BatchNorm1d in logvar_layer: {name}")
            break

    if not has_batchnorm:
        print("✗ No BatchNorm1d found - using LayerNorm instead")

    # 检查是否配置了对比损失
    contrastive_weight = model.config.contrastive_weight
    if contrastive_weight > 0:
        print(f"\n✓ Isotropy regularization enabled")
        print(f"  Contrastive weight: {contrastive_weight}")
    else:
        print(f"\n✗ Isotropy regularization disabled (weight=0)")

    # 加载数据
    print("\nLoading validation data...")
    dataloader_kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": 0,
        "pin_memory": False,
    }

    if config.data.dataset == "wikipedia":
        dataset_kwargs = {
            "num_samples": min(config.data.wiki_num_samples, 1000),
            "min_length": config.data.wiki_min_length,
            "max_length": config.data.wiki_max_length,
        }
    elif config.data.dataset == "qqp":
        dataset_kwargs = {
            "num_samples": min(config.data.qqp_num_samples, 1000),
            "min_length": config.data.qqp_min_length,
        }
    elif config.data.dataset == "mixed":
        dataset_kwargs = {
            "wiki_num_samples": min(config.data.wiki_num_samples, 500),
            "wiki_min_length": config.data.wiki_min_length,
            "wiki_max_length": config.data.wiki_max_length,
            "alpaca_num_samples": min(config.data.alpaca_num_samples, 500),
            "alpaca_min_length": config.data.alpaca_min_length,
            "oasst1_num_samples": min(config.data.oasst1_num_samples, 500),
            "oasst1_min_length": config.data.oasst1_min_length,
        }

    val_loader = get_dataloader(
        dataset_name=config.data.dataset,
        split="validation",
        **dataloader_kwargs,
        **dataset_kwargs,
    )

    # 分析各向异性
    print(f"\nAnalyzing isotropy on {num_batches} batches...")
    model.eval()

    all_mu_vectors = []
    all_z_vectors = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing")):
            if batch_idx >= num_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 提取语义向量
            semantic_vector, kl_loss = model.encoder(
                input_ids,
                attention_mask,
                return_kl=True
            )

            # 收集向量
            all_z_vectors.append(semantic_vector.cpu())

            # 提取 mu（需要在训练模式下才能访问）
            # 我们可以通过关闭 training mode 的 sampling 来获取 mu
            model.encoder.train(False)
            with torch.no_grad():
                # 重新计算以获取中间表示
                outputs = model.encoder.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                last_hidden_state = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)
                token_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_pooled = sum_hidden / token_counts

                mu = model.encoder.mu_layer(mean_pooled)
                all_mu_vectors.append(mu.cpu())

            model.encoder.train(False)

    # 合并所有向量
    all_mu = torch.cat(all_mu_vectors, dim=0)
    all_z = torch.cat(all_z_vectors, dim=0)

    # 分析
    print("\n" + "="*60)
    print("FINAL ISOTROPY ANALYSIS")
    print("="*60)

    mu_metrics = analyze_isotropy(all_mu, "Mu Vectors (before reparameterization)")
    z_metrics = analyze_isotropy(all_z, "Z Vectors (after reparameterization)")

    # 总结
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nConfig: {config_path}")
    print(f"Architecture: BatchNorm1d" if has_batchnorm else "Architecture: LayerNorm")
    print(f"Isotropy Regularization: {contrastive_weight}")
    print(f"\nMu Isotropy Score: {mu_metrics['isotropy_score']:.4f}")
    print(f"Z Isotropy Score: {z_metrics['isotropy_score']:.4f}")

    if mu_metrics['isotropy_score'] < 0.1:
        print("\n✓ EXCELLENT: Mu vectors show good isotropy!")
    elif mu_metrics['isotropy_score'] < 0.3:
        print("\n⚠ MODERATE: Some improvement possible")
    else:
        print("\n✗ POOR: High anisotropy detected - vectors are too similar")

    print("\nRecommendations:")
    if not has_batchnorm and mu_metrics['isotropy_score'] > 0.3:
        print("  - Consider using BatchNorm1d instead of LayerNorm in encoder")
    if contrastive_weight == 0 and mu_metrics['isotropy_score'] > 0.3:
        print("  - Enable isotropy regularization with contrastive_weight > 0")
    if mu_metrics['isotropy_score'] < 0.1:
        print("  - Current configuration is working well!")

    return mu_metrics, z_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check isotropy of semantic vectors")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to analyze")

    args = parser.parse_args()

    check_model_isotropy(args.config, args.num_batches)
