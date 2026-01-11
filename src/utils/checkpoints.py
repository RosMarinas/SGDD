"""检查点保存和加载系统

支持模型、优化器、调度器状态的保存和加载。
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
import json
import shutil


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    loss: float,
    config: Dict[str, Any],
    checkpoint_path: str | Path,
    save_optimizer: bool = True,
) -> None:
    """保存训练检查点

    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器 (可选)
        epoch: 当前epoch
        step: 当前步数
        loss: 当前损失
        config: 配置字典
        checkpoint_path: 检查点保存路径
        save_optimizer: 是否保存优化器状态
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # 准备保存内容
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "config": config,
        "model_state_dict": model.state_dict(),
    }

    if save_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # 保存
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """加载训练检查点

    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器 (可选)
        scheduler: 学习率调度器 (可选)
        device: 设备

    Returns:
        包含训练状态的字典
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 加载
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型状态
    model.load_state_dict(checkpoint["model_state_dict"])

    # 加载优化器状态
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 加载调度器状态
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # 返回训练状态
    training_state = {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", 0.0),
        "config": checkpoint.get("config", {}),
    }

    print(f"Checkpoint loaded: epoch={training_state['epoch']}, step={training_state['step']}")

    return training_state


def save_best_model(
    model: torch.nn.Module,
    metric: float,
    metric_name: str,
    checkpoint_dir: str | Path,
    config: Dict[str, Any],
) -> None:
    """保存最佳模型

    Args:
        model: 模型
        metric: 指标值 (越大越好)
        metric_name: 指标名称
        checkpoint_dir: 检查点目录
        config: 配置字典
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型
    model_path = checkpoint_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metric": metric,
            "metric_name": metric_name,
            "config": config,
        },
        model_path,
    )

    # 保存指标
    metrics_path = checkpoint_dir / "best_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"best_metric": metric, "metric_name": metric_name}, f, indent=2)

    print(f"Best model saved with {metric_name}={metric:.4f}")


def load_best_model(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
) -> float:
    """加载最佳模型

    Args:
        checkpoint_dir: 检查点目录
        model: 模型
        device: 设备

    Returns:
        最佳指标值
    """
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / "best_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found: {model_path}")

    # 加载
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metric = checkpoint.get("metric", 0.0)
    metric_name = checkpoint.get("metric_name", "unknown")

    print(f"Best model loaded with {metric_name}={metric:.4f}")

    return metric


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """查找最新的检查点

    Args:
        checkpoint_dir: 检查点目录

    Returns:
        最新检查点的路径,如果不存在则返回None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    # 查找所有检查点
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))

    if not checkpoints:
        return None

    # 按修改时间排序
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return checkpoints[0]


def cleanup_old_checkpoints(
    checkpoint_dir: str | Path,
    keep_last_n: int = 5,
) -> None:
    """清理旧检查点,只保留最近N个

    Args:
        checkpoint_dir: 检查点目录
        keep_last_n: 保留的检查点数量
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    # 查找所有检查点
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))

    if len(checkpoints) <= keep_last_n:
        return

    # 按修改时间排序
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # 删除旧的检查点
    for checkpoint in checkpoints[keep_last_n:]:
        checkpoint.unlink()
        print(f"Deleted old checkpoint: {checkpoint}")


def export_model_for_inference(
    model: torch.nn.Module,
    export_path: str | Path,
    config: Dict[str, Any],
) -> None:
    """导出模型用于推理

    Args:
        model: 模型
        export_path: 导出路径
        config: 配置字典
    """
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # 只保存模型状态和配置 (不需要优化器等训练状态)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        export_path,
    )

    print(f"Model exported for inference to {export_path}")
