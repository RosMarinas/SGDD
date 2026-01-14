"""训练脚本
HF_ENDPOINT=https://hf-mirror.com python src/train.py --config configs/phase1_wiki.yaml
支持Wikipedia重构和QQP改写任务的训练,包含WandB日志、检查点保存、评估等功能。
"""

import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from typing import Optional, Dict, Any
from dataclasses import asdict
import wandb

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from models.sgdd import SGDDModel, SGDDConfig as ModelConfig
from utils.config import SGDDConfig as Config
from utils.data import get_dataloader
from utils.metrics import evaluate_generation, format_metrics
from utils.checkpoints import save_checkpoint, load_checkpoint, save_best_model, cleanup_old_checkpoints


def set_seed(seed: int) -> None:
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: SGDDModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    config: Config,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """训练一个epoch

    Args:
        model: SGDD模型
        dataloader: 数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        config: 配置
        epoch: 当前epoch
        scaler: 混合精度scaler

    Returns:
        训练指标字典
    """
    from contextlib import nullcontext

    model.train()

    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    # 梯度累积:只在循环开始时清零一次
    optimizer.zero_grad()

    # 统一的autocast context
    autocast = torch.amp.autocast('cuda') if (config.training.use_fp16 and scaler is not None) else nullcontext()

    for batch_idx, batch in enumerate(progress_bar):
        # 移动数据到设备
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # 获取目标 (QQP数据集需要特殊处理)
        if "target_ids" in batch:
            # QQP: 使用target_ids作为目标
            target_ids = batch["target_ids"].to(device)
            # 用input_ids提取语义向量
            semantic_input_ids = input_ids
            semantic_attention_mask = attention_mask
        else:
            # Wikipedia: 重构任务,目标是input_ids
            target_ids = input_ids
            semantic_input_ids = input_ids
            semantic_attention_mask = attention_mask

        # 统一的前向传播和反向传播
        with autocast:
            logits, predicted_tokens, loss_mask = model(
                semantic_input_ids,
                semantic_attention_mask,
                cfg_uncond=True,
            )
            loss = model.compute_loss(logits, target_ids, loss_mask)
            raw_loss = loss.item()
            # 梯度累积:loss除以累积步数以保持梯度尺度一致
            loss = loss / config.training.gradient_accumulation_steps

        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积:每accumulation_steps步更新一次参数,或者在最后一个batch也要更新
        is_accumulation_step = (batch_idx + 1) % config.training.gradient_accumulation_steps == 0
        is_last_batch = (batch_idx + 1) == len(dataloader)

        if is_accumulation_step or is_last_batch:
            # 梯度裁剪:只在更新前裁剪一次(对累积后的总梯度进行裁剪)
            if config.training.grad_clip > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

            # 更新参数
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # 更新学习率
            if scheduler is not None:
                scheduler.step()

            # 清零梯度为下一次累积做准备
            optimizer.zero_grad()

        # 记录损失
        total_loss += raw_loss
        num_batches += 1

        # 更新进度条
        progress_bar.set_postfix({"loss": raw_loss})

        # 日志记录（按照epoch统计loss，仅在训练过程中用于监控）
        if (batch_idx + 1) % config.training.log_interval == 0:
            avg_loss = total_loss / num_batches  # 当前epoch的平均loss
            step = epoch * len(dataloader) + batch_idx + 1
            if config.training.use_wandb:
                wandb.log({
                    "train/loss": avg_loss,  # 当前epoch的累计平均loss
                    "train/batch_loss": raw_loss,  # 当前batch的瞬时loss
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/step": step,
                })

    # 返回平均损失
    avg_loss = total_loss / num_batches
    return {"loss": avg_loss}


def train(config: Config, resume_from: Optional[str] = None) -> None:
    """主训练函数

    Args:
        config: 配置
        resume_from: 从检查点恢复训练
    """
    print("=" * 60)
    print("SGDD Training")
    print("=" * 60)
    print(config)

    # 设置种子
    set_seed(config.seed)

    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # 创建目录
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 初始化WandB
    if config.training.use_wandb:
        wandb.init(
            project=config.training.wandb_project,
            entity=config.training.wandb_entity,
            name=config.training.wandb_run_name,
            config=asdict(config) if hasattr(config, "__dict__") else vars(config),
        )

    # 创建模型
    print("\nInitializing model...")

    # 创建临时编码器以获取vocab_size
    from models.encoder import SemanticEncoder
    temp_encoder = SemanticEncoder(
        model_name=config.model.encoder_name,
        hidden_dim=config.model.semantic_dim,
    )
    vocab_size = temp_encoder.tokenizer.vocab_size
    print(f"Detected vocab_size: {vocab_size} from {config.model.encoder_name}")
    del temp_encoder  # 释放临时编码器

    model_config = ModelConfig(
        encoder_model=config.model.encoder_name,
        hidden_dim=config.model.semantic_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        ffn_dim=config.model.ffn_dim,
        max_len=config.model.max_length,
        dropout=config.model.dropout,
        num_diffusion_steps=config.model.num_diffusion_steps,
        vocab_size=vocab_size,  # 从tokenizer动态获取
        cfg_prob=config.training.cfg_drop_prob,
        use_self_conditioning=config.model.use_self_conditioning,
        compute_pad_loss=config.model.compute_pad_loss,
    )
    model = SGDDModel(model_config).to(device)

    total_params = model.get_num_params()
    trainable_params = model.get_num_trainable_params()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # 混合精度
    scaler = torch.amp.GradScaler('cuda') if config.training.use_fp16 else None

    # 加载数据
    print("\nLoading datasets...")
    dataloader_kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
    }

    # 根据数据集类型设置参数
    if config.data.dataset == "wikipedia":
        dataset_kwargs = {
            "num_samples": config.data.wiki_num_samples,
            "min_length": config.data.wiki_min_length,
            "max_length": config.data.wiki_max_length,
        }
    elif config.data.dataset == "qqp":
        dataset_kwargs = {
            "num_samples": config.data.qqp_num_samples,
            "min_length": config.data.qqp_min_length,
        }
    elif config.data.dataset == "mixed":
        # 混合数据集需要传递所有数据源的配置
        dataset_kwargs = {
            "wiki_num_samples": config.data.wiki_num_samples,
            "wiki_min_length": config.data.wiki_min_length,
            "wiki_max_length": config.data.wiki_max_length,
            "alpaca_num_samples": config.data.alpaca_num_samples,
            "alpaca_min_length": config.data.alpaca_min_length,
            "oasst1_num_samples": config.data.oasst1_num_samples,
            "oasst1_min_length": config.data.oasst1_min_length,
        }
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")

    train_loader = get_dataloader(
        dataset_name=config.data.dataset,
        split="train",
        **dataloader_kwargs,
        **dataset_kwargs,
    )

    # 验证时使用更大的批大小(验证时不需要反向传播,显存占用更小)
    # 使用训练时的2倍以加快验证速度
    val_dataloader_kwargs = dataloader_kwargs.copy()
    val_dataloader_kwargs["batch_size"] = config.training.batch_size * 2

    val_loader = get_dataloader(
        dataset_name=config.data.dataset,
        split="validation",
        **val_dataloader_kwargs,
        **dataset_kwargs,
    )

    # 在数据加载后创建学习率调度器,以动态计算总步数
    # 注意:需要考虑梯度累积,实际更新步数 = batch数 / accumulation_steps
    steps_per_epoch = len(train_loader) // config.training.gradient_accumulation_steps
    num_training_steps = config.training.num_epochs * steps_per_epoch
    print(f"Total training steps: {num_training_steps}")
    print(f"Steps per epoch: {steps_per_epoch} (batches: {len(train_loader)}, accum: {config.training.gradient_accumulation_steps})")

    if config.training.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=1e-6,
        )
        print(f"Using Cosine Annealing LR scheduler with T_max={num_training_steps}")
    elif config.training.lr_scheduler == "linear":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_steps=config.training.warmup_steps,
        )
        print(f"Using Linear LR scheduler with warmup_steps={config.training.warmup_steps}")
    else:
        scheduler = None
        print("No learning rate scheduler (constant LR)")

    # 从检查点恢复
    start_epoch = 0
    best_metric = float('inf')  # 初始化为无穷大,因为loss越小越好
    global_step = 0  # 初始化global_step

    if resume_from is not None:
        print(f"\nResuming from checkpoint: {resume_from}")
        training_state = load_checkpoint(
            resume_from,
            model,
            optimizer,
            scheduler,
            device,
        )
        start_epoch = training_state["epoch"] + 1
        global_step = training_state.get("global_step", start_epoch * len(train_loader))
        print(f"Resumed from epoch {start_epoch}, global_step={global_step}")
    else:
        # 如果不是从checkpoint恢复,根据start_epoch计算global_step
        global_step = start_epoch * len(train_loader)

    # 训练循环
    print("\nStarting training...")

    for epoch in range(start_epoch, config.training.num_epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{config.training.num_epochs}")
        print(f"{'=' * 60}")

        # 训练
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config,
            epoch,
            scaler,
        )

        print(f"\nTrain metrics: {format_metrics(train_metrics, prefix='train_')}")

        # 每个epoch结束后进行验证
        print("\nRunning validation...")
        val_metrics = evaluate_generation(
            model,
            val_loader,
            device,
            max_samples=1000,  # 只评估100个样本以加快速度
        )
        print(f"Val metrics: {format_metrics(val_metrics, prefix='val_')}")

        # 记录到WandB
        if config.training.use_wandb:
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()})

        # 保存最佳模型 (基于验证loss,越小越好)
        # evaluate_generation返回{"loss": avg_loss, "perplexity": perplexity}
        val_loss = val_metrics.get("loss", float('inf'))
        if val_loss < best_metric:
            best_metric = val_loss
            save_best_model(
                model,
                best_metric,
                "val_loss",
                checkpoint_dir,
                asdict(config) if hasattr(config, "__dict__") else vars(config),
                metric_higher_is_better=False,  # loss越小越好
            )
            print(f"New best model saved with val_loss={best_metric:.4f}")

        # 保存检查点
        if config.training.save_epochs > 0:
            should_save = (epoch + 1) % config.training.save_epochs == 0
        else:
            save_freq = max(1, config.training.save_interval // len(train_loader))
            should_save = (epoch + 1) % save_freq == 0
            
        if should_save or epoch == config.training.num_epochs - 1:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                train_metrics["loss"],
                asdict(config) if hasattr(config, "__dict__") else vars(config),
                checkpoint_path,
            )

            # 清理旧检查点
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n=5)

        global_step += len(train_loader)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best metric: {best_metric:.4f}")
    print("=" * 60)

    if config.training.use_wandb:
        wandb.finish()


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Train SGDD model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # 加载配置
    config = Config.from_yaml(args.config)

    # 开始训练
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
