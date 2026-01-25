# 实施计划：Semantic-Guided Discrete Diffusion (SGDD) 模型





## 2026-01-25 数据接口更新 (BookCorpus) - ✅ 已完成

根据最新需求，数据接口已更新为适配新的BookCorpus数据集。

### 变更点
1. **数据集**: 切换到 `data/BookCorpus/final_dataset_1.4B`
2. **格式**: 支持HuggingFace Arrow/Parquet格式 (load_from_disk)
3. **接口**: 重写 `src/utils/data.py`，移除旧的Wikipedia/QQP/Mixed接口，新增 `BookCorpusDataset`
4. **配置**: 更新 `DataConfig`，移除旧数据集参数，新增 `dataset_path` 和 `max_token_length` (默认64)
5. **训练**: 更新 `train.py` 适配新的参数传递

---

## 2026-01-23 Encoder升级计划 (BGE-M3) - ✅ 已完成

鉴于RoBERTa效果一般，已切换到BGE-M3 (`BAAI/bge-m3`) 作为语义编码器。

### 变更点
1. **模型**: `roberta-base` (768d) -> `BAAI/bge-m3` (1024d)
2. **架构**: **Lightweight Decoder Strategy**
   - Semantic Vector: **1024d** (保留完整语义)
   - Decoder Hidden: **256d** (大幅减少Embedding参数)
   - Decoder Layers: **2** (强迫依赖语义向量)
   - Attention Heads: **4**
   - FFN Dim: **1024**
3. **参数量**:
   - Embedding: ~64M (250k * 256)
   - Decoder Body: ~3M
   - **总可训练参数**: ~67M (符合预期)
4. **适配层**: 线性映射 1024->256 (用于初始化)
5. **词表**: 50265 -> 250002 (XLM-R Tokenizer)
6. **白化**: 默认禁用

### 实施步骤
1. ✅ 修改 `src/models/encoder.py`: 适配1024d输入，使用[CLS] pooling，rename roberta->model
2. ✅ 修改 `src/models/decoder.py`: 支持分离的 `semantic_dim` 和 `hidden_dim`
3. ✅ 修改 `src/models/sgdd.py`: 更新 `SGDDConfig` 默认值，支持投影 1024->256
4. ✅ 更新 `src/utils/config.py`: 添加 `decoder_dim` 支持
5. ✅ 更新 `configs/phase1_vib.yaml`: 应用轻量级配置
6. ✅ 测试: 验证显存占用和前向传播形状

---

## 项目概述

构建一个轻量级的、非自回归的离散扩散语言模型，该模型：
- 使用冻结的 RoBERTa-base 提取语义向量作为输入
- 生成短文本（≤64 tokens），基于语义向量进行条件生成
- 使用 MaskGIT 风格的迭代解码和分类器无关引导（CFG）
- 目标参数量：~40-60M 可训练参数，适配 RTX 4070 Ti Super

## 架构组件

### 1. 语义编码器（Semantic Encoder - 冻结）
- **模型**: RoBERTa-base (HuggingFace `roberta-base`)
- **输出**: 均值池化的语义向量 Z ∈ ℝ⁷⁶⁸ → 投影到 ℝ⁵¹²
- **参数状态**: 冻结 (requires_grad=False)
- **文件**: `src/models/encoder.py`

### 2. 扩散解码器（Diffusion Decoder - 可训练）
- **架构**: 6层双向 Transformer（无因果掩码）
- **维度**: 512 hidden, 8 attention heads, 64 max length
- **关键特性**:
  - 交叉注意力机制（Query=token, Key/Value=语义向量 Z）
  - RoPE（旋转位置编码）
  - 时间步嵌入（正弦波）
  - 使用 RoBERTa 权重初始化输入/输出嵌入层
- **文件**: `src/models/decoder.py`

### 3. 噪声调度与扩散过程
- **调度**: 余弦噪声调度（1000个离散时间步）
- **过程**: 离散token扩散（基于alpha_bar的token替换）
- **损失**: 仅在masked token上计算交叉熵
- **文件**: `src/models/diffusion.py`

### 4. 完整模型
- **组合**: Encoder → Semantic Vector → Noise → Decoder
- **训练**: 分类器无关引导（10%无条件批次）
- **文件**: `src/models/sgdd.py`

## 训练策略

### Phase 1: 文本重构（Wikipedia）
- **任务**: 输入文本 → 编码 → 解码 → 相同文本
- **数据集**: Wikipedia片段（100k样本）
- **目标**: 验证模型能够遵循语义向量指令
- **成功指标**: Loss < 2.0, Exact Match > 80%

### Phase 2: 改写（Quora Question Pairs）
- **任务**: 输入文本A → 编码 → 解码 → 文本B（相同含义）
- **数据集**: QQP（问题对）
- **目标**: 测试语义泛化能力
- **成功指标**: BLEU分数提升

## 推理

- **算法**: MaskGIT迭代解码（**16步**以获得更高质量）
- **引导**: 分类器无关引导，scale w=2.0
- **过程**:
  1. 从完全masked的token开始
  2. 迭代unmask token（余弦调度）
  3. 基于置信度的token选择
  4. 应用CFG: p = p_cond + w(p_cond - p_uncond)
- **文件**: `src/utils/sampling.py`
- **预期推理时间**: 每次生成约3-4秒

## 项目结构

```
SGDD/
├── src/
│   ├── models/
│   │   ├── encoder.py           # 冻结RoBERTa编码器
│   │   ├── decoder.py           # 双向Transformer解码器
│   │   ├── diffusion.py         # 噪声调度和扩散过程
│   │   └── sgdd.py              # 完整模型（编码器+解码器）
│   ├── utils/
│   │   ├── config.py            # 配置管理
│   │   ├── data.py              # 数据加载和预处理
│   │   ├── sampling.py          # MaskGIT推理和CFG
│   │   ├── metrics.py           # 评估指标
│   │   └── checkpoints.py       # 检查点保存/加载
│   ├── train.py                 # 训练脚本
│   └── evaluate.py              # 评估脚本
├── tests/                       # 综合测试套件
│   ├── test_encoder.py          # 编码器单元测试
│   └── test_diffusion.py        # 扩散过程测试
├── configs/                     # YAML配置
│   ├── phase1_wiki.yaml         # Phase1训练配置（Wikipedia重构）
│   ├── phase2_qqp.yaml          # Phase2训练配置（QQP改写）
│   └── phase1_wiki_test.yaml    # Phase1测试配置
├── issues/                      # 问题追踪
│   ├── 1-printlog.md            # 训练日志相关问题（已修复）
│   └── 2-architecture.md        # 架构相关问题
├── data/                        # 数据集（自动下载）
├── checkpoints/                 # 模型检查点
│   ├── first/                   # Phase1检查点
│   └── second/                  # Phase2检查点
├── logs/                        # WandB日志
├── pyproject.toml               # 项目依赖和配置
├── README.md                    # 项目说明
├── CLAUDE.md                    # Claude开发规范
├── SGDD.md                      # SGDD模型详细说明
└── plan.md                      # 本文件（实施计划）

```