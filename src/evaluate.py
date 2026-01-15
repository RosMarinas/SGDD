"""评估脚本

用于评估训练好的模型,计算各种指标并生成示例。
"""

import os
import sys
from pathlib import Path
import torch
import argparse
from typing import Dict, Any
import json
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.sgdd import SGDDModel, SGDDConfig as ModelConfig
from utils.config import SGDDConfig
from utils.data import get_dataloader
from utils.metrics import evaluate_generation, compute_bleu, exact_match_score, format_metrics
from utils.checkpoints import load_best_model
from utils.sampling import maskgit_sample


def evaluate_model(
    model: SGDDModel,
    dataloader,
    device: torch.device,
    max_samples: int = 1000,
    num_inference_steps: int = 16,
    cfg_scale: float = 2.0,
) -> Dict[str, Any]:
    """评估模型

    Args:
        model: SGDD模型
        dataloader: 数据加载器
        device: 设备
        max_samples: 最大评估样本数
        num_inference_steps: 推理步数
        cfg_scale: CFG引导强度

    Returns:
        评估指标字典
    """
    model.eval()

    all_generated_texts = []
    all_target_texts = []
    all_input_texts = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if total_samples >= max_samples:
                break

            # 移动数据到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", batch.get("attention_mask_q1")).to(device)

            # 获取目标
            if "target_ids" in batch:
                target_ids = batch["target_ids"].to(device)
                target_mask = batch.get("target_mask", attention_mask).to(device)
                target_texts = batch.get("texts_q2", [])
                input_texts = batch.get("texts_q1", [])
            else:
                target_ids = input_ids
                target_mask = attention_mask
                target_texts = batch.get("texts", [])
                input_texts = target_texts

            # 计算损失
            semantic_vector = model.encoder.encode_from_tokens(input_ids, attention_mask)
            logits = model.decoder(target_ids, semantic_vector, timestep=torch.zeros(input_ids.size(0), dtype=torch.long, device=device))
            loss = model.compute_loss(logits, target_ids, target_mask)

            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

            # 生成文本
            generated = model.generate(
                input_text="",  # 不使用,因为我们会直接使用semantic_vector
                num_steps=num_inference_steps,
                guidance_scale=cfg_scale,
                max_length=model.config.max_len,
            )

            # 由于模型generate方法需要input_text,我们需要手动使用maskgit_sample
            # 或者直接使用模型的generate方法
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                # 获取单个样本的语义向量
                single_semantic = semantic_vector[i : i + 1]

                # 使用模型的generate方法
                input_text = input_texts[i] if i < len(input_texts) else ""
                generated_text = model.generate(
                    input_text=input_text,
                    num_steps=num_inference_steps,
                    guidance_scale=cfg_scale,
                    max_length=model.config.max_len,
                )

                all_generated_texts.append(generated_text)
                all_target_texts.append(target_texts[i] if i < len(target_texts) else "")
                all_input_texts.append(input_text)

    # 计算指标
    metrics = {}

    # 平均损失
    metrics["loss"] = total_loss / total_samples if total_samples > 0 else 0.0

    # 精确匹配
    if len(all_generated_texts) > 0 and len(all_target_texts) > 0:
        em = exact_match_score(all_generated_texts, all_target_texts)
        metrics["exact_match"] = em

        # BLEU分数
        bleu_scores = compute_bleu(all_generated_texts, all_target_texts)
        metrics.update(bleu_scores)

    # 生成示例
    examples = []
    num_examples = min(10, len(all_input_texts))
    for i in range(num_examples):
        examples.append(
            {
                "input": all_input_texts[i],
                "target": all_target_texts[i],
                "generated": all_generated_texts[i],
            }
        )

    return metrics, examples


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Evaluate SGDD model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory (overrides config)")
    parser.add_argument("--dataset", type=str, help="Dataset to evaluate on (overrides config)")
    parser.add_argument("--split", type=str, help="Data split (overrides config)")
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config)")
    parser.add_argument("--num_samples", type=int, help="Number of samples to evaluate (overrides config)")
    parser.add_argument("--num_inference_steps", type=int, help="Number of inference steps (overrides config)")
    parser.add_argument("--cfg_scale", type=float, help="CFG guidance scale (overrides config)")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON file for results")
    args = parser.parse_args()

    # 加载配置
    print(f"Loading config from {args.config}...")
    config = SGDDConfig.from_yaml(args.config)
    print(config)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 确定checkpoint路径
    checkpoint_path = args.checkpoint or config.checkpoint_dir
    print(f"\nLoading model from {checkpoint_path}...")

    # 创建临时编码器以获取vocab_size
    from models.encoder import SemanticEncoder
    temp_encoder = SemanticEncoder(
        model_name=config.model.encoder_name,
        hidden_dim=config.model.semantic_dim,
    )
    vocab_size = temp_encoder.tokenizer.vocab_size
    print(f"Detected vocab_size: {vocab_size} from {config.model.encoder_name}")
    del temp_encoder

    # 创建模型配置
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
        cfg_prob=0.0,  # 评估时不需要CFG dropout
        use_self_conditioning=config.model.use_self_conditioning,
        compute_pad_loss=config.model.compute_pad_loss,
    )

    # 创建模型
    model = SGDDModel(model_config).to(device)

    # 加载最佳模型
    try:
        best_metric = load_best_model(checkpoint_path, model, device)
        print(f"Model loaded successfully (best metric: {best_metric:.4f})")
    except FileNotFoundError:
        print(f"Warning: Best model not found, looking for latest checkpoint...")
        # 尝试加载最新检查点
        checkpoint_dir = Path(checkpoint_path)
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            checkpoint_file = checkpoints[0]
            print(f"Loading checkpoint: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Checkpoint loaded successfully")
        else:
            print("Error: No checkpoints found!")
            return

    # 确定评估参数（命令行参数优先）
    dataset = args.dataset or config.data.dataset
    split = args.split or "validation"
    batch_size = args.batch_size or config.training.batch_size * 2  # 评估时使用2倍batch size
    num_samples = args.num_samples or 1000
    num_inference_steps = args.num_inference_steps or config.inference.num_inference_steps
    cfg_scale = args.cfg_scale or config.inference.cfg_scale

    # 加载数据
    print(f"\nLoading {dataset} dataset ({split} split)...")

    if dataset == "wikipedia":
        dataset_kwargs = {
            "num_samples": config.data.wiki_num_samples,
            "min_length": config.data.wiki_min_length,
            "max_length": config.data.wiki_max_length,
        }
    elif dataset == "qqp":
        dataset_kwargs = {
            "num_samples": config.data.qqp_num_samples,
            "min_length": config.data.qqp_min_length,
        }
    elif dataset == "mixed":
        # 混合数据集配置
        dataset_kwargs = {
            "wiki_num_samples": config.data.wiki_num_samples,
            "wiki_min_length": config.data.wiki_min_length,
            "wiki_max_length": config.data.wiki_max_length,
            "alpaca_num_samples": config.data.alpaca_num_samples,
            "alpaca_min_length": config.data.alpaca_min_length,
            "oasst1_num_samples": config.data.oasst1_num_samples,
            "oasst1_min_length": config.data.oasst1_min_length,
            "max_token_length": config.model.max_length,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    dataloader = get_dataloader(
        dataset_name=dataset,
        split=split,
        batch_size=batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        **dataset_kwargs,
    )

    print(f"Dataset loaded: {len(dataloader.dataset)} samples")

    # 评估
    print(f"\nEvaluating model...")
    print(f"  Num samples: {num_samples}")
    print(f"  Inference steps: {num_inference_steps}")
    print(f"  CFG scale: {cfg_scale}")

    metrics, examples = evaluate_model(
        model,
        dataloader,
        device,
        max_samples=num_samples,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(format_metrics(metrics))

    # 打印示例
    print("\n" + "=" * 60)
    print("Generated Examples")
    print("=" * 60)
    for i, example in enumerate(examples):
        print(f"\nExample {i + 1}:")
        print(f"  Input:    {example['input'][:100]}...")
        print(f"  Target:   {example['target'][:100]}...")
        print(f"  Generated: {example['generated'][:100]}...")

    # 保存结果
    output_path = args.output or str(Path(config.checkpoint_dir) / "evaluation_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    results = {
        "metrics": metrics,
        "examples": examples,
        "config": {
            "dataset": dataset,
            "split": split,
            "num_samples": num_samples,
            "num_inference_steps": num_inference_steps,
            "cfg_scale": cfg_scale,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
