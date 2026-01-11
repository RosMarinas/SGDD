"""MaskGIT采样和推理

实现迭代式mask decoding和分类器无关引导(CFG)。
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict
import math


def compute_cosine_schedule(step: int, total_steps: int) -> float:
    """计算余弦调度比率

    Args:
        step: 当前步数
        total_steps: 总步数

    Returns:
        unmask比率 (0到1之间)
    """
    # 余弦调度: 从小比率开始,逐渐增加
    ratio = 0.5 * (1 + math.cos(math.pi * step / total_steps))
    return ratio


def maskgit_sample(
    model,
    semantic_vector: torch.Tensor,
    num_steps: int = 16,
    temperature: float = 1.0,
    cfg_scale: float = 2.0,
    use_cfg: bool = True,
    sampling_strategy: str = "confidence",
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    tokenizer=None,
) -> Dict[str, torch.Tensor]:
    """MaskGIT迭代采样

    Args:
        model: SGDD模型
        semantic_vector: 语义向量 [batch, semantic_dim]
        num_steps: 迭代步数
        temperature: 采样温度
        cfg_scale: CFG引导强度
        use_cfg: 是否使用CFG
        sampling_strategy: 采样策略 ('confidence' or 'multinomial')
        top_k: Top-k采样 (可选)
        top_p: Top-p采样 (可选)
        tokenizer: Tokenizer (用于解码)

    Returns:
        包含生成结果的字典
    """
    batch_size = semantic_vector.shape[0]
    device = semantic_vector.device
    vocab_size = model.tokenizer.vocab_size if hasattr(model, "tokenizer") else 50265
    max_length = semantic_vector.shape[1] if len(semantic_vector.shape) > 1 else 64

    # 获取最大长度
    max_length = model.decoder.max_length if hasattr(model, "decoder") else 64

    # 初始化: 完全masked的tokens
    # 假设mask_token_id = vocab_size - 1 (通常是50265 for RoBERTa)
    mask_token_id = vocab_size - 1
    current_tokens = torch.full((batch_size, max_length), mask_token_id, device=device)
    mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=device)

    # 迭代解码
    for step in range(num_steps):
        # 计算当前步的unmask比率
        if step == num_steps - 1:
            # 最后一步: unmask所有tokens
            unmask_ratio = 1.0
        else:
            unmask_ratio = compute_cosine_schedule(step, num_steps)

        # 计算要unmask的token数量
        num_to_unmask = int(unmask_ratio * max_length)
        num_masked = mask.sum(dim=1).max().item()

        if num_masked == 0:
            # 所有tokens都已unmask
            break

        # 模型预测
        with torch.no_grad():
            if use_cfg:
                # 条件预测
                logits_cond = model(current_tokens, semantic_vector, use_cfg=False)

                # 无条件预测 (用零向量替换语义向量)
                semantic_uncond = torch.zeros_like(semantic_vector)
                logits_uncond = model(current_tokens, semantic_uncond, use_cfg=False)

                # CFG组合
                logits = logits_cond + cfg_scale * (logits_cond - logits_uncond)
            else:
                logits = model(current_tokens, semantic_vector, use_cfg=False)

        # 只在masked位置采样
        logits_masked = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        # 应用温度
        logits_masked = logits_masked / temperature

        # 采样策略
        if sampling_strategy == "confidence":
            # 基于置信度的采样 (选择置信度最高的token)
            probs = F.softmax(logits_masked, dim=-1)
            confidences, predicted_tokens = torch.max(probs, dim=-1)

            # 选择要unmask的位置 (基于置信度)
            confidences_masked = confidences.masked_fill(~mask, float("-inf"))
            _, top_indices = torch.topk(confidences_masked, k=min(num_to_unmask, num_masked), dim=1)

            # Unmask选定的位置
            for b in range(batch_size):
                indices = top_indices[b]
                current_tokens[b, indices] = predicted_tokens[b, indices]
                mask[b, indices] = False

        elif sampling_strategy == "multinomial":
            # 多项式采样
            if top_k is not None:
                # Top-k采样
                indices_to_remove = logits_masked < torch.topk(logits_masked, top_k, dim=-1)[0][..., -1:]
                logits_masked = logits_masked.masked_fill(indices_to_remove, float("-inf"))

            if top_p is not None:
                # Top-p采样
                sorted_logits, sorted_indices = torch.sort(logits_masked, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累积概率超过top_p的tokens
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits_masked = logits_masked.masked_fill(indices_to_remove, float("-inf"))

            # 采样
            probs = F.softmax(logits_masked, dim=-1)
            predicted_tokens = torch.multinomial(probs.view(-1, vocab_size), num_samples=1).squeeze(-1)
            predicted_tokens = predicted_tokens.view(batch_size, max_length)

            # 选择要unmask的位置
            confidences = torch.max(probs, dim=-1)[0]
            confidences_masked = confidences.masked_fill(~mask, float("-inf"))
            _, top_indices = torch.topk(confidences_masked, k=min(num_to_unmask, num_masked), dim=1)

            # Unmask选定的位置
            for b in range(batch_size):
                indices = top_indices[b]
                current_tokens[b, indices] = predicted_tokens[b, indices]
                mask[b, indices] = False

    return {
        "tokens": current_tokens,
        "mask": mask,
    }


@torch.no_grad()
def generate_with_cfg(
    model,
    input_texts: list[str],
    num_steps: int = 16,
    temperature: float = 1.0,
    cfg_scale: float = 2.0,
    use_cfg: bool = True,
    batch_size: int = 8,
) -> Dict[str, any]:
    """使用CFG批量生成文本

    Args:
        model: SGDD模型
        input_texts: 输入文本列表
        num_steps: 迭代步数
        temperature: 采样温度
        cfg_scale: CFG引导强度
        use_cfg: 是否使用CFG
        batch_size: 批大小

    Returns:
        包含生成结果的字典
    """
    model.eval()

    # 分批处理
    all_generated_texts = []
    all_generated_tokens = []

    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i : i + len(input_texts)][
            :batch_size
        ]  # Fix slicing to avoid index error

        # 编码语义向量
        semantic_vector = model.encoder.encode_text(batch_texts)

        # 采样
        results = maskgit_sample(
            model=model,
            semantic_vector=semantic_vector,
            num_steps=num_steps,
            temperature=temperature,
            cfg_scale=cfg_scale,
            use_cfg=use_cfg,
            tokenizer=model.tokenizer,
        )

        # 解码
        generated_tokens = results["tokens"]
        generated_texts = model.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        all_generated_texts.extend(generated_texts)
        all_generated_tokens.append(generated_tokens)

    # 合并tokens
    all_generated_tokens = torch.cat(all_generated_tokens, dim=0)

    return {
        "generated_texts": all_generated_texts,
        "generated_tokens": all_generated_tokens,
    }
