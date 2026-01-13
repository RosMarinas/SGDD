

------

# 项目规划文档：Semantic-Guided Discrete Diffusion (SGDD)

## 1. 项目目标 (Objective)

构建一个轻量级的、非自回归的离散扩散语言模型（Discrete Diffusion LM）。

- **核心功能**：输入一段文本的语义向量（Semantic Vector），输出一段符合该语义的短文本
- **核心优势**：利用扩散模型的全局感受野，在短句生成上实现比自回归模型更好的结构完整性和语义一致性。
- **验证指标**：模型能否准确还原或改写具有相同语义的句子，且语法通顺。

------

## 2. 模型规格与 Scaling Law 分析

根据 **SMDM (Scaling Masked Diffusion Models)** 的研究结论：

1. **Scaling Law**：MDM 的验证集 Loss 随计算量呈幂律下降，且与自回归模型（ARM）的 Scaling 速率相当 1。
2. **参数效率**：在相同计算预算下，MDM 的最佳模型参数量大约是 ARM 的一半 2。这意味着 MDM 能够更高效地利用参数容量。
3. **计算代价**：虽然参数更少，但 MDM 训练达到相同 Loss 需要的计算量约为 ARM 的 16 倍（因为需要多次去噪）。

结论与参数推荐：

为了“快速复现”，我们不需要追求 SOTA 的大模型，而是追求训练效率与收敛速度。

- **推荐参数量**：**40M ~ 60M (Trainable)**。
- **理由**：
  - 这相当于 4-6 层 BERT 的大小，在单张消费级显卡（如 RTX 3090/4090）上可以在数小时内完成训练。
  - 配合一个冻结的（Frozen）强力 Encoder（如 RoBERTa-base, ~85M），整体“智商”足够处理短句语义。

------

## 3. 架构详细规范 (Architecture Spec)

### A. 语义编码器 (Semantic Encoder) - **Frozen**

- **模型**: `RoBERTa-base` (HuggingFace: `roberta-base`)。
- **作用**: 提取输入文本的深层语义。
- **输出处理**: `Mean Pooling` 得到 $Z \in \mathbb{R}^{768}$。
- **参数状态**: **冻结 (Requires_grad = False)**。

### B. 扩散解码器 (Diffusion Decoder) - **Trainable**

- **结构**: Bidirectional Transformer Encoder (无 Causal Mask)。
- **层数 (Layers)**: 6 层。
- **隐藏层维度 (Hidden Dim)**: 512 (通过 Linear 层将 $Z$ 的 768 映射为 512)。
- **注意力头数 (Heads)**: 8。
- **上下文长度 (Context Len)**: 64 (覆盖你的 48 token 需求，留有余地)。
- **条件注入**: **Cross-Attention** (Query = Noisy Tokens, Key/Value = Semantic Vector $Z$)。
- **位置编码**: **RoPE (Rotary Positional Embeddings)** (符合 LLaDA 设计，增强位置感知)。
- **参数初始化**: **使用 `roberta-base` 的 Word Embeddings 初始化 Input/Output Layer** (加速收敛的关键)。

### C. 词表 (Tokenizer)

- **类型**: BPE (Byte-Pair Encoding)。
- **配置**: 复用 `roberta-base` tokenizer (Vocab size $\approx$ 50k)。

------

## 4. 数据集选取策略 (Data Strategy)

为了验证“语义 -> 文字”的能力，我们需要数据集具备**“语义不变，表述多样”**或**“高质量文本重构”**的特性。

### 方案 A：文本重构 (Text Reconstruction) - **推荐首选**

- **任务**: 输入文本 A $\rightarrow$ Encoder $\rightarrow$ 向量 $Z$ $\rightarrow$ Decoder $\rightarrow$ 还原文本 A。
- **数据集**: **Wikipedia Snippets** 或 **BookCorpus** (切分为短句)。
- **优点**: 数据近乎无限，无需标注。这能最快验证“模型是否学会了听从向量 $Z$ 的指挥”。如果模型能完美还原，说明架构成功。

### 方案 B：语义改写 (Paraphrasing) - **进阶验证**

- **任务**: 输入文本 A $\rightarrow$ Encoder $\rightarrow$ 向量 $Z$ $\rightarrow$ Decoder $\rightarrow$ 生成文本 B (与 A 意思相同但写法不同)。
- **数据集**: **Quora Question Pairs (QQP)** 或 **ParaNMT-50M**。
- **优点**: 真正验证“语义生成”而非“死记硬背”。
- **实现**: 训练时输入 Pair 中的 Sentence 1，让 Decoder 预测 Sentence 2。

**路线**：先用 **方案 A (Wiki)** 跑通流程（Loss 下降，能还原句子），然后用 **方案 B (QQP)** 微调看生成效果。

------

## 5. 训练配置 (Training Config)

- **Objective**: Cross-Entropy on Masked Tokens (只计算被 Mask 掉位置的 Loss) 4。
- **Noise Schedule**: **Cosine Schedule** (相比 Linear，更适合短文本的平滑去噪)。
- **Masking Strategy**: 动态独立掩码 (每个 step 随机采样 $t \sim U[0,1]$)。
- **CFG (无分类器引导)**:
  - **训练时**: 10% 概率将语义向量 $Z$ 置为全零向量 $\emptyset$ 。
  - 这迫使模型在无条件时学习单词的统计规律，在有条件时学习语义对齐。

------

## 6. 推理与评估 (Inference & Eval)

- **采样算法**: **MaskGIT Iterative Decoding**。
- **步数**: 8 步 (平衡速度与质量)。
- **Guidance Scale**: $w = 2.0$ (根据 $s = s_{uncond} + w(s_{cond} - s_{uncond})$)。
- **长度控制**: 输出 64 token，在遇到 `[EOS]` 后截断。

## 7. 环境与说明
1. 本环境使用uv作为包管理器，pyproject.toml中有一些现成的库，请通过修改pyproject.toml来添加新的依赖。
2. 显卡设置为一张4070tisuper，系统为windows，有cuda，可以使用
3. 文件管理，所有代码文件放在src目录下，数据集放在data目录下，模型权重放在checkpoints目录下。src目录下可以创建子目录来管理不同模块的代码，但是最多不超多一层，例如src/models, src/utils等.

