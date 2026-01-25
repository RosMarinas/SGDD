# 实施计划：Semantic-Guided Discrete Diffusion (SGDD) 模型

## 2026-01-25 冒烟测试与修复 - ✅ 已完成

完成了对模型的全面冒烟测试，并修复了在测试过程中发现的生成问题。

### 修复与改进
1. **位置编码修复**: 在 `DiffusionDecoder` 中添加了可学习的绝对位置编码 (`Absolute Positional Embeddings`)。这是为了解决在全MASK输入（推理初始状态）下，模型无法区分不同位置，导致输出重复字符的问题。
2. **生成逻辑修正**: 修正了 `SGDDModel.generate` 中的 MaskGIT 调度逻辑。现在的生成过程正确地从 $t=1.0$ (全噪声) 迭代到 $t=0.0$ (无噪声)，并使用噪声调度器计算每一步的目标掩码率，确保了推理与训练过程的一致性。
3. **冒烟测试**: 创建了 `tests/smoke_test.py`，使用真实的 BookCorpus 数据集子集（64个样本）进行过拟合测试。
   - **结果**: 模型能够成功将 Loss 从 ~14.0 降低到 ~0.16，并能准确重建训练样本，生成的文本连贯且结构正确。

---

## 2026-01-25 架构与代码清理 - ✅ 已完成

完成了从 RoBERTa 到 BGE-M3 以及从 Wikipedia 到 BookCorpus 的全面迁移，并清理了冗余代码。

### 变更点
1. **代码清理**: 移除 `src/utils/data.py` 和 `src/evaluate.py` 中对 Wikipedia, QQP, Mixed 数据集的兼容逻辑。
2. **模型解耦**: 将 `DiffusionDecoder` 中的 `roberta_embeddings` 参数重命名为 `pretrained_embeddings`。
3. **Encoder清理**: 移除 `src/models/encoder.py` 中对 RoBERTa 的引用和 fallback 测试逻辑。
4. **脚本更新**: 重写 `src/scripts/compute_whitening_stats.py` 以适配 BGE-M3 和 BookCorpus。
5. **配置更新**: 更新 `configs/phase1_whitening.yaml` 以使用新数据集和模型。
6. **文档同步**: 更新 `README.md` 和 `SGDD.md` 以反映当前项目状态。

---

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

### 架构细节
- **模型**: `BAAI/bge-m3` (1024d)
- **Decoder**: 2层, 256d, 4 heads
- **参数量**: ~67M 可训练参数
- **词表**: XLM-R (250k tokens)

---

## 项目概述

构建一个轻量级的、非自回归的离散扩散语言模型，该模型：
- 使用冻结的 BGE-M3 提取语义向量作为输入
- 生成短文本（≤64 tokens），基于语义向量进行条件生成
- 使用 MaskGIT 风格的迭代解码和分类器无关引导（CFG）
- 目标参数量：~40-70M 可训练参数，适配 RTX 4070 Ti Super


## 项目结构

```
SGDD/
├── src/
│   ├── models/
│   │   ├── encoder.py           # 冻结BGE-M3编码器 (+ VIB)
│   │   ├── decoder.py           # AdaLN-Zero解码器
│   │   ├── diffusion.py         # 噪声调度和扩散过程
│   │   └── sgdd.py              # 完整模型
│   ├── utils/
│   │   ├── config.py            # 配置管理
│   │   ├── data.py              # 数据加载 (BookCorpus)
│   │   ├── metrics.py           # 评估指标
│   │   └── checkpoints.py       # 检查点保存/加载
│   ├── train.py                 # 训练脚本
│   └── evaluate.py              # 评估脚本
├── tests/                       # 测试套件
├── configs/                     # YAML配置
│   ├── phase1_vib.yaml          # 主训练配置
│   └── phase1_whitening.yaml    # 白化优化配置
├── issues/                      # 问题追踪
├── data/                        # 数据集
├── checkpoints/                 # 模型检查点
├── pyproject.toml               # 项目依赖
├── README.md                    # 项目说明
├── SGDD.md                      # 模型详细说明
└── plan.md                      # 本文件
```
