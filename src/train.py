"""训练脚本
HF_ENDPOINT=https://hf-mirror.com python src/train.py --config configs/phase1_vib.yaml
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
from typing import Optional, Dict, Any, List, Tuple
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
    scaler: Optional[torch.cuda.amp.GradScaler],
    val_loader: DataLoader,
    val_iter_wrapper: List[Any],
    best_metric_wrapper: Dict[str, float],
    global_step: int,
    checkpoint_dir: Path,
) -> Tuple[Dict[str, float], int]:
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
        val_loader: 验证集加载器
        val_iter_wrapper: 包含验证集迭代器的列表 [iterator]
        best_metric_wrapper: 包含最佳指标的字典 {'value': float}
        global_step: 全局步数
        checkpoint_dir: 检查点目录

    Returns:
        训练指标字典, 更新后的global_step
    """
    from contextlib import nullcontext

    model.train()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    # 梯度累积:只在循环开始时清零一次
    optimizer.zero_grad()

    # 统一的autocast context
    autocast = torch.amp.autocast('cuda') if  (config.training.use_fp16 and scaler is not None) else nullcontext()

    # Determine validation batch count for ~1000 samples
    val_batch_size = config.training.batch_size * 2
    num_val_batches = max(1, 1000 // val_batch_size)
    
    # 验证间隔: 优先使用 config 中的 eval_interval, 如果没设置则默认 1000
    eval_interval = getattr(config.training, 'eval_interval', 1000)

    for batch_idx, batch in enumerate(progress_bar):
        global_step += 1
        
        # 移动数据到设备
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # BookCorpus: 重构任务, 目标是 input_ids
        target_ids = input_ids
        semantic_input_ids = input_ids
        semantic_attention_mask = attention_mask

        # 统一的前向传播和反向传播
        with autocast:
            # Forward pass: returns logits, target, loss_mask, kl_loss
            logits, predicted_tokens, loss_mask, kl_loss = model(
                semantic_input_ids,
                semantic_attention_mask,
                cfg_uncond=True,
            )

            # Compute loss with KL divergence
            loss = model.compute_loss(logits, target_ids, loss_mask, kl_loss)
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

        current_grad_norm = 0.0 # 初始化当前步的梯度范数

        if is_accumulation_step or is_last_batch:
            # 梯度裁剪:只在更新前裁剪一次(对累积后的总梯度进行裁剪)
            if config.training.grad_clip > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                # 捕获梯度范数
                current_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip).item()
            
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

        # Increment annealing step for KL divergence
        model.encoder.increment_step()

        # Track losses separately
        total_loss += raw_loss
        kl_value = kl_loss.mean().item()
        total_kl_loss += kl_value
        num_batches += 1

        # 更新进度条
        progress_bar.set_postfix({"loss": raw_loss, "kl": kl_value})

        # 日志记录（按照epoch统计loss，仅在训练过程中用于监控）
        if (batch_idx + 1) % config.training.log_interval == 0:
            avg_loss = total_loss / num_batches  # 当前epoch的平均loss
            avg_kl = total_kl_loss / num_batches
            if config.training.use_wandb:
                # 计算当前 KL 权重 (用于日志)
                # Handle _current_step as tensor (buffer) or int
                current_step_val = model.encoder._current_step.item() if isinstance(model.encoder._current_step, torch.Tensor) else model.encoder._current_step
                kl_anneal_ratio = min(1.0, current_step_val / model.encoder.kl_anneal_steps)
                current_kl_weight = model.encoder.kl_weight * kl_anneal_ratio
                
                log_dict = {
                    "train/loss": avg_loss,  # 当前epoch的累计平均loss
                    "train/reconstruction_loss": avg_loss - avg_kl,
                    "train/kl_loss": avg_kl,
                    "train/unweighted_kl": avg_kl / current_kl_weight if current_kl_weight > 0 else 0.0,
                    "train/kl_weight": current_kl_weight,
                    "train/batch_loss": raw_loss,  # 当前batch的瞬时loss
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/step": global_step,
                }
                
                # 记录梯度范数 (如果这一步进行了更新)
                if current_grad_norm > 0:
                    log_dict["train/grad_norm"] = current_grad_norm
                
                # 记录scaler scale
                if scaler is not None:
                    log_dict["train/scaler_scale"] = scaler.get_scale()
                
                wandb.log(log_dict)

        # --- 按步数进行验证 ---
        if global_step % eval_interval == 0:
            # 准备验证数据 (从迭代器获取下 num_val_batches 个 batch)
            val_batches = []
            for _ in range(num_val_batches):
                try:
                    batch = next(val_iter_wrapper[0])
                    val_batches.append(batch)
                except StopIteration:
                    # 重新初始化迭代器
                    val_iter_wrapper[0] = iter(val_loader)
                    try:
                        batch = next(val_iter_wrapper[0])
                        val_batches.append(batch)
                    except StopIteration:
                        break # 数据集为空

            if val_batches:
                # 运行验证
                val_metrics = evaluate_generation(
                    model,
                    val_batches, # 传递 batch 列表 (它是可迭代的，兼容 dataloader 接口)
                    device,
                    max_samples=None # 我们已经控制了 batch 数量
                )
                
                print(f"\nStep {global_step} Validation: {format_metrics(val_metrics, prefix='val_')}")

                if config.training.use_wandb:
                    wandb.log({f"val/{k}": v for k, v in val_metrics.items()})
                
                # 保存最佳模型
                val_loss = val_metrics.get("loss", float('inf'))
                if val_loss < best_metric_wrapper['value']:
                    best_metric_wrapper['value'] = val_loss
                    save_best_model(
                        model,
                        val_loss,
                        "val_loss",
                        checkpoint_dir,
                        asdict(config) if hasattr(config, "__dict__") else vars(config),
                        metric_higher_is_better=False,
                    )
                    print(f"New best model saved with val_loss={val_loss:.4f}")

                # 滚动保存检查点 (只保留最近3个)
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    raw_loss, # 使用当前 batch loss 或 val_loss
                    asdict(config) if hasattr(config, "__dict__") else vars(config),
                    checkpoint_path,
                )
                cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3)
                
            # 恢复训练模式
            model.train()

    # 返回平均损失
    avg_loss = total_loss / num_batches
    avg_kl = total_kl_loss / num_batches

    return {
        "loss": avg_loss,
        "reconstruction_loss": avg_loss - avg_kl,
        "kl_loss": avg_kl,
    }, global_step


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
    vocab_size = len(temp_encoder.tokenizer)
    print(f"Detected vocab_size: {vocab_size} from {config.model.encoder_name}")
    del temp_encoder  # 释放临时编码器

    model_config = ModelConfig(
        encoder_model=config.model.encoder_name,
        semantic_dim=config.model.semantic_dim,
        decoder_dim=config.model.decoder_dim,
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
        # VIB config
        kl_weight=getattr(config.model, 'kl_weight', 0.001),
        kl_anneal_steps=getattr(config.model, 'kl_anneal_steps', 10000),
    )
    model = SGDDModel(model_config).to(device)
    
    # 监控模型梯度和参数 (仅限解码器，因为编码器大部分参数已冻结)
    # log="all" 会记录参数直方图和梯度直方图
    # log_freq 决定了记录的频率
    if config.training.use_wandb:
        wandb.watch(model.decoder, log="all", log_freq=config.training.log_interval)

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
        "tokenizer_name": config.model.encoder_name,
    }

    dataset_kwargs = {
        "dataset_path": config.data.dataset_path,
        "max_token_length": config.data.max_token_length,
        "min_length": config.data.min_length,
    }

    train_loader = get_dataloader(
        dataset_name=config.data.dataset,
        split="train",
        **dataloader_kwargs,
        **dataset_kwargs,
    )

    # 验证时使用更大的批大小
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
        global_step = start_epoch * (len(train_loader) // config.training.gradient_accumulation_steps)

    # 初始化验证迭代器和最佳指标包装器 (用于在 train_epoch 中修改)
    val_iter_wrapper = [iter(val_loader)]
    best_metric_wrapper = {'value': best_metric}

    # 训练循环
    print("\nStarting training...")

    for epoch in range(start_epoch, config.training.num_epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{config.training.num_epochs}")
        print(f"{'=' * 60}")

        # 训练 (验证和保存现在在 train_epoch 中按 step 进行)
        train_metrics, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config,
            epoch,
            scaler,
            val_loader,
            val_iter_wrapper,
            best_metric_wrapper,
            global_step,
            checkpoint_dir,
        )

        print(f"\nEpoch {epoch + 1} completed. Train metrics: {format_metrics(train_metrics, prefix='train_')}")

    # 更新最终的最佳指标用于打印
    best_metric = best_metric_wrapper['value']

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
