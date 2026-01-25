# 项目规划文档：Semantic-Guided Discrete Diffusion (SGDD)

## 1. 项目目标 (Objective)

构建一个轻量级的、非自回归的离散扩散语言模型（Discrete Diffusion LM）。

- **核心功能**：输入一段文本的语义向量（Semantic Vector），输出一段符合该语义的短文本。
- **核心优势**：利用扩散模型的全局感受野，在短句生成上实现比自回归模型更好的结构完整性和语义一致性。
- **验证指标**：模型能否准确还原或改写具有相同语义的句子，且语法通顺。

---

## 2. 模型规格与 Scaling Law 分析

根据 **SMDM (Scaling Masked Diffusion Models)** 的研究结论：

1. **Scaling Law**：MDM 的验证集 Loss 随计算量呈幂律下降，且与自回归模型（ARM）的 Scaling 速率相当。
2. **参数效率**：在相同计算预算下，MDM 的最佳模型参数量大约是 ARM 的一半。这意味着 MDM 能够更高效地利用参数容量。
3. **计算代价**：虽然参数更少，但 MDM 训练达到相同 Loss 需要的计算量约为 ARM 的 16 倍。

**推荐参数量**：**40M ~ 70M (Trainable)**。
- **理由**：配合强力 Encoder，整体“智商”足够处理短句语义，且可在单卡（RTX 4070 Ti Super）上快速迭代。

---

## 3. 架构详细规范 (Architecture Spec)

### A. 语义编码器 (Semantic Encoder) - **Frozen**

- **模型**: `BAAI/bge-m3` (HuggingFace: `BAAI/bge-m3`)。
- **作用**: 提取输入文本的深层语义。
- **输出处理**: 使用 `[CLS]` 向量作为语义表示，并通过 **VIB (Variational Information Bottleneck)** 进行正则化。
- **输出维度**: $Z  \mathbb{R}^{1024}$。
- **参数状态**: **冻结 (Requires_grad = False)**。

### B. 扩散解码器 (Diffusion Decoder) - **Trainable**

- **结构**: Bidirectional Transformer (AdaLN-Zero 架构)。
- **层数 (Layers)**: 2 层 (Lightweight Strategy)。
- **隐藏层维度 (Hidden Dim)**: 256。
- **注意力头数 (Heads)**: 4。
- **上下文长度 (Context Len)**: 64 - 128。
- **条件注入**: **AdaLN (Adaptive Layer Normalization)**。将语义向量 $Z$ 和时间步 $t$ 注入到每一层的 LayerNorm 中。
- **位置编码**: **RoPE (Rotary Positional Embeddings)**。
- **参数初始化**: 使用 BGE-M3 (XLM-RoBERTa 架构) 的 Word Embeddings 进行投影初始化。

### C. 词表 (Tokenizer)

- **类型**: XLM-RoBERTa Tokenizer (BGE-M3 默认)。
- **词表大小**: 250,002。

---

## 4. 数据集选取策略 (Data Strategy)

### 文本重构 (Text Reconstruction)

- **任务**: 输入文本 A $\rightarrow$ Encoder $\rightarrow$ 向量 $Z$ $\rightarrow$ Decoder $\rightarrow$ 还原文本 A。
- **数据集**: **BookCorpus** (data/BookCorpus/final_dataset_1.4B)。
- **优点**: 高质量长文本切分的短句，适合学习语言规律和语义对齐。

---

## 5. 训练配置 (Training Config)

- **Objective**: Cross-Entropy on Masked Tokens。
- **Noise Schedule**: **Cosine Schedule**。
- **VIB Loss**: 引入 KL Divergence 约束隐空间分布。
- **Self-Conditioning**: 50% 概率开启自我调节以稳定生成。
- **CFG (Classifier-Free Guidance)**: 15% 概率将语义向量 $Z$ 置零。

---

## 6. 推理与评估 (Inference & Eval)

- **采样算法**: **MaskGIT Iterative Decoding**。
- **步数**: 16 步 (平衡速度与质量)。
- **Guidance Scale**: $w = 2.0$。

---

## 7. 环境与说明
1. 本环境使用uv作为包管理器，pyproject.toml中有一些现成的库，请通过修改pyproject.toml来添加新的依赖。
2. 显卡为 RTX 4070 Ti Super 16GB，系统为 Windows，支持 CUDA。
3. 文件管理：代码在 `src/`，数据集在 `data/`，权重在 `checkpoints/`。