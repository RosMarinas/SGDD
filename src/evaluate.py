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
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--config", type=str, help="Path to config file (optional, loaded from checkpoint)")
    parser.add_argument("--dataset", type=str, default="wikipedia", choices=["wikipedia", "qqp"], help="Dataset to evaluate on")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"], help="Data split")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--num_inference_steps", type=int, default=16, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="CFG guidance scale")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    print(f"\nLoading model from {args.checkpoint}...")

    # 创建模型配置
    model_config = ModelConfig()

    # 创建模型
    model = SGDDModel(model_config).to(device)

    # 加载最佳模型
    try:
        best_metric = load_best_model(args.checkpoint, model, device)
        print(f"Model loaded successfully (best metric: {best_metric:.4f})")
    except FileNotFoundError:
        print(f"Warning: Best model not found, looking for latest checkpoint...")
        # 尝试加载最新检查点
        checkpoint_dir = Path(args.checkpoint)
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            checkpoint_path = checkpoints[0]
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Checkpoint loaded successfully")
        else:
            print("Error: No checkpoints found!")
            return

    # 加载数据
    print(f"\nLoading {args.dataset} dataset ({args.split} split)...")

    if args.dataset == "wikipedia":
        dataset_kwargs = {
            "num_samples": 10000,
            "min_length": 20,
            "max_length": 200,
        }
    else:  # qqp
        dataset_kwargs = {
            "num_samples": 10000,
            "min_length": 10,
        }

    dataloader = get_dataloader(
        dataset_name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=4,
        **dataset_kwargs,
    )

    print(f"Dataset loaded: {len(dataloader.dataset)} samples")

    # 评估
    print(f"\nEvaluating model...")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  CFG scale: {args.cfg_scale}")

    metrics, examples = evaluate_model(
        model,
        dataloader,
        device,
        max_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
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
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "metrics": metrics,
            "examples": examples,
            "args": vars(args),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
