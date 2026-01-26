"""评估指标计算

实现损失计算、精确匹配、BLEU分数等指标。
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from collections import Counter
import math


def compute_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """计算交叉熵损失

    Args:
        logits: 模型输出 [batch, seq_len, vocab_size]
        target: 目标tokens [batch, seq_len]
        ignore_index: 忽略的索引 (用于padding)

    Returns:
        损失值
    """
    # 重塑为 [batch * seq_len, vocab_size]
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target.view(-1)

    # 计算损失
    loss = F.cross_entropy(logits_flat, target_flat, ignore_index=ignore_index, reduction="mean")

    return loss


def compute_masked_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """计算masked位置的交叉熵损失

    Args:
        logits: 模型输出 [batch, seq_len, vocab_size]
        target: 目标tokens [batch, seq_len]
        mask: 损失掩码 [batch, seq_len] (True表示计算损失)
        ignore_index: 忽略的索引

    Returns:
        损失值
    """
    # 重塑
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target.view(-1)
    mask_flat = mask.view(-1)

    # 只在mask为True的位置计算损失
    loss_per_token = F.cross_entropy(
        logits_flat,
        target_flat,
        ignore_index=ignore_index,
        reduction="none",
    )

    # 应用mask
    masked_loss = loss_per_token * mask_flat.float()

    # 计算平均值 (只考虑masked位置)
    num_masked = mask_flat.sum().item()
    if num_masked > 0:
        loss = masked_loss.sum() / num_masked
    else:
        loss = torch.tensor(0.0, device=logits.device)

    return loss


def exact_match_score(
    predicted: List[str],
    target: List[str],
) -> float:
    """计算精确匹配分数

    Args:
        predicted: 预测文本列表
        target: 目标文本列表

    Returns:
        精确匹配率 (0到1之间)
    """
    if len(predicted) != len(target):
        raise ValueError("Predicted and target must have the same length")

    exact_matches = sum(1 for p, t in zip(predicted, target) if p.strip() == t.strip())
    return exact_matches / len(target) if len(target) > 0 else 0.0


def compute_bleu(
    predicted: List[str],
    target: List[str],
    max_order: int = 4,
    smooth: bool = False,
) -> Dict[str, float]:
    """计算BLEU分数

    Args:
        predicted: 预测文本列表
        target: 目标文本列表
        max_order: n-gram的最大阶数
        smooth: 是否应用平滑

    Returns:
        包含BLEU分数的字典
    """
    if len(predicted) != len(target):
        raise ValueError("Predicted and target must have the same length")

    # 分词
    predicted_tokens = [p.lower().split() for p in predicted]
    target_tokens = [[t.lower().split()] for t in target]  # 注意: 每个目标是一个列表

    # 计算每个n-gram的精度
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order

    for prediction, targets in zip(predicted_tokens, target_tokens):
        # 对于每个参考翻译,找到最佳匹配
        for target in targets:
            for order in range(1, max_order + 1):
                # 提取n-gram
                predicted_ngrams = Counter(
                    [tuple(prediction[i : i + order]) for i in range(len(prediction) - order + 1)]
                )
                target_ngrams = Counter(
                    [tuple(target[i : i + order]) for i in range(len(target) - order + 1)]
                )

                # 计算匹配数
                matches = 0
                for ngram in predicted_ngrams:
                    matches += min(predicted_ngrams[ngram], target_ngrams.get(ngram, 0))

                matches_by_order[order - 1] += matches
                possible_matches_by_order[order - 1] += max(len(prediction) - order + 1, 0)

    # 计算精度
    precisions = [0.0] * max_order
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
            else:
                precisions[i] = 0.0

    # 计算几何平均
    if min(precisions) > 0:
        p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0.0

    # 计算长度惩罚
    prediction_len = sum(len(p) for p in predicted_tokens)
    target_len = sum(min(len(t), len(p)) for t, p in zip(target_tokens, predicted_tokens))

    if prediction_len > 0 and target_len > 0:
        ratio = prediction_len / target_len
        if ratio > 1.0:
            bp = 1.0
        else:
            bp = math.exp(1 - 1 / ratio)
    else:
        bp = 0.0

    # BLEU分数
    bleu = bp * geo_mean

    return {
        "bleu": bleu,
        "bleu1": precisions[0],
        "bleu2": precisions[1] if max_order > 1 else 0.0,
        "bleu3": precisions[2] if max_order > 2 else 0.0,
        "bleu4": precisions[3] if max_order > 3 else 0.0,
        "brevity_penalty": bp,
    }


def compute_perplexity(loss: float) -> float:
    """从损失计算困惑度

    Args:
        loss: 交叉熵损失

    Returns:
        困惑度
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def evaluate_generation(
    model,
    dataloader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """评估生成模型 (BookCorpus 重构任务)

    Args:
        model: SGDD模型
        dataloader: 数据加载器
        device: 设备
        max_samples: 最大评估样本数 (可选)

    Returns:
        包含评估指标的字典
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_samples is not None and total_samples >= max_samples:
                break

            # 移动数据到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 前向传播 (使用与训练一致的逻辑)
            # 注意: 验证时不使用 CFG dropout (cfg_uncond=False)
            logits, target_tokens, loss_mask, kl_loss = model(
                input_ids,
                attention_mask,
                cfg_uncond=False,
            )

            # 计算损失
            loss = model.compute_loss(logits, target_tokens, loss_mask, kl_loss)

            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

    # 计算平均损失
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    perplexity = compute_perplexity(avg_loss)

    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity,
    }

    return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """格式化指标用于日志记录

    Args:
        metrics: 指标字典
        prefix: 前缀 (例如 'train_', 'val_')

    Returns:
        格式化的字符串
    """
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{prefix}{key}: {value:.4f}")
        else:
            parts.append(f"{prefix}{key}: {value}")

    return " | ".join(parts)
