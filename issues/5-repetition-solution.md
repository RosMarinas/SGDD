# 解决重复Token问题的综合方案

## 问题分析

当前生成结果出现大量重复token:
- "asonason", "905905", "303303"
- 模式坍缩到少数几个高频token

## 根本原因

1. **模型训练不足** (val_loss=2.07)
   - 模型未充分收敛,学到的token分布不准确
   - 预测倾向过于集中

2. **采样策略问题**
   - MaskGIT的置信度采样可能陷入局部最优
   - 高频token被不断强化

3. **缺少多样性约束**
   - 没有机制惩罚重复模式

## 已实施的解决方案

### 方案1: 重复惩罚 (已实现 ✅)

**位置**: `src/models/sgdd.py:350-360`

**原理**: 对已出现的token进行logit惩罚

```python
if repetition_penalty > 1.0:
    # 统计token频率
    unique_tokens, counts = torch.unique(
        current_tokens[current_tokens != mask_token_id],
        return_counts=True
    )
    # 惩罚已出现token: logit /= repetition_penalty^count
    for token, count in zip(unique_tokens, counts):
        guided_logits[:, :, token] /= (repetition_penalty ** count)
```

**参数建议**:
- `1.0` - 无惩罚(默认)
- `1.2` - 轻度惩罚,平衡多样性和质量
- `1.5` - 强惩罚,强制多样性

**测试结果**:
```
rep_penalty=1.0: "asonason asonason asonason"
rep_penalty=1.2: "asonason SARason 31 January 1931ason"
rep_penalty=1.5: "asonason SARason April April 31ason 1931ason Marason"
```

✅ 改善: token多样性增加
⚠️ 但仍有重复,说明问题不仅是采样策略

---

## 其他推荐方案

### 方案2: Top-K采样 (推荐实施)

**原理**: 只从前K个概率最高的token中采样

**优点**:
- 简单有效
- 防止模型采样到极低频的糟糕token
- 提高多样性

**实施**: 修改`src/models/sgdd.py`第375行采样逻辑

```python
# Before:
sampled_tokens = torch.multinomial(probs_at_masked, num_samples=1).squeeze(-1)

# After (添加top-k):
if top_k > 0:
    # 只保留top-k概率
    top_k_probs, top_k_indices = torch.topk(probs_at_masked, k=min(top_k, probs_at_masked.size(-1)))
    # 重新归一化
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    sampled_tokens = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
    # 映射回原始词汇表索引
    sampled_tokens = top_k_indices.gather(1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
else:
    sampled_tokens = torch.multinomial(probs_at_masked, num_samples=1).squeeze(-1)
```

**建议参数**: `top_k=50` 或 `top_k=100`

---

### 方案3: Top-P (Nucleus) 采样 (推荐实施)

**原理**: 动态选择累积概率达到p的最小token集合

**优点**:
- 自适应调整
- 在分布尖锐时用少token,分布平坦时用多token

**实施**: 类似top-k,但用累积概率

```python
if top_p < 1.0:
    sorted_probs, sorted_indices = torch.sort(probs_at_masked, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # 找到累积概率超过top_p的索引
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    # 移除这些索引
    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled_tokens = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
    sampled_tokens = sorted_indices.gather(1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
```

**建议参数**: `top_p=0.9` 或 `top_p=0.95`

---

### 方案4: 降低Temperature (简单有效)

**原理**: 降低采样温度使分布更集中,减少随机性

**当前**: `temperature=1.0`

**建议**: 尝试 `temperature=0.8` 或 `temperature=0.7`

**注意**: 温度过低会损害多样性,需平衡

---

### 方案5: 改进训练 (根本解决)

#### 5.1 继续训练当前模型

**问题**: checkpoints/4 的val_loss=2.07还比较高

**建议**:
- 继续训练直到val_loss < 1.5
- 监控exact match指标

#### 5.2 使用更好的checkpoint

**发现**: checkpoints/3 (val_loss=1.37) 可能更好

**建议**: 先测试checkpoints/3的表现

#### 5.3 调整模型配置

**当前**: `semantic_dim=128` (较小)

**问题**: 可能限制表达能力

**建议**:
- 增加到 `semantic_dim=256` 或 `512`
- 重新训练

#### 5.4 改进训练策略

**当前CFG训练**: 10%无条件批次

**建议**:
- 提高到15-20% unconditional
- 加强CFG对语义的依赖

---

### 方案6: 改进MaskGIT解码策略

#### 6.1 调整Unmask调度

**当前**: `num_to_unmask = int((1 - t) * num_masked / 2)`

**问题**: 可能太保守

**建议更激进的调度**:
```python
# 当前: 线性调度
num_to_unmask = int((1 - t) * num_masked / 2)

# 改进: 平方根调度(早期更快)
num_to_unmask = int((1 - t**0.5) * num_masked / 1.5)
```

#### 6.2 增加解码步数

**当前**: `num_steps=16`

**建议**: 尝试 `num_steps=24` 或 `num_steps=32`

---

### 方案7: 后处理去重

**原理**: 对生成结果应用简单的去重规则

**实施**:
```python
def post_process(generated_text: str) -> str:
    # 移除连续重复的词
    words = generated_text.split()
    result = []
    prev_word = None
    for word in words:
        if word != prev_word:
            result.append(word)
        prev_word = word
    return ' '.join(result)
```

---

## 推荐的实施顺序

### 阶段1: 快速修复(已实施 ✅)
- ✅ 重复惩罚 (rep_penalty=1.2-1.5)

### 阶段2: 采样改进(推荐立即实施)
1. ⭐ **Top-K采样** (k=50) - 最简单有效
2. ⭐ **降低Temperature** (0.8) - 与top-k配合
3. Top-P采样 (p=0.9) - 可选,与top-k二选一

### 阶段3: 训练改进(根本解决)
1. 测试checkpoints/3
2. 继续训练直到val_loss < 1.5
3. 如果仍不满意,增加semantic_dim重新训练

### 阶段4: 高级优化
1. 调整unmask调度
2. 增加解码步数到24-32
3. 后处理去重

---

## 立即可执行的命令

### 测试Top-K + Temperature

修改`temp/test_inference_fix.py`:
```python
generated = model.generate(
    input_text=input_text,
    num_steps=16,
    guidance_scale=2.0,
    temperature=0.8,  # 降低
    max_length=config.max_len,
    repetition_penalty=1.2,  # 轻度惩罚
)
```

### 测试更好的checkpoint

```bash
# 修改test_inference_fix.py
checkpoint_path = "checkpoints/3"  # 使用val_loss=1.37的模型
```

---

## 预期改善

### 当前状态
- 重复token: "asonason asonason"
- 语义不连贯
- Exact match: 0%

### 实施后预期
- **阶段2** (Top-K + Temperature + Rep Penalty):
  - 重复减少80%
  - 语义开始连贯
  - Exact match: 10-20%

- **阶段3** (更好的checkpoint):
  - 重复减少95%
  - 语义基本连贯
  - Exact match: 40-60%

- **阶段4** (完整优化):
  - 重复基本消除
  - 语义完全连贯
  - Exact match: 80%+

---

## 关键代码位置

1. **重复惩罚**: `src/models/sgdd.py:350-360` (已添加 ✅)
2. **采样逻辑**: `src/models/sgdd.py:375-377` (需添加top-k)
3. **配置参数**: `configs/phase1_wiki.yaml:64-66`
4. **测试脚本**: `temp/test_inference_fix.py`

---

## 时间戳

- 2026-01-15: 实施重复惩罚方案
- 2026-01-15: 测试rep_penalty=1.0, 1.2, 1.5
- 待实施: Top-K采样, Temperature调优
