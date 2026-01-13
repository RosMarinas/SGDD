# 2-Issue:基于 AdaLN-Zero 与 Self-Conditioning 的单向量文本生成架构

## 1. 背景与现状 (Context)

当前项目旨在使用单个全局语义向量 (Global Semantic Vector, $Z$) 来指导扩散模型重构或生成文本。

目前的基准实现方案存在以下特征：

- **Encoder**: 使用 Frozen RoBERTa + Mean Pooling 提取 $Z \in \mathbb{R}^{512}$。
- **Conditioning**: 使用简单的 `Concat` 或 `Add` 将 $Z$ 注入到去噪网络中。
- **Backbone**: 标准 Transformer。
- **Training**: 标准扩散损失 $\mathcal{L}_{\text{simple}}$。

## 2. 问题陈述 (Problem Statement)

当前的架构在处理“单向量 $\to$ 长序列”生成任务时存在三个核心痛点：

1. **条件控制力弱 (Weak Conditioning)**: 简单的 `Concat` 使得全局语义信号随着网络层数加深被逐渐“稀释”，导致生成文本容易跑题或忽略语义约束。
2. **长度信息缺失 (Length Agnostic)**: 单个向量 $Z$ 丢失了序列长度信息。目前模型无法动态决定生成文本的长度，导致生成结果要么被截断，要么有过多的 Padding。
3. **曝光偏差 (Exposure Bias)**: 训练时模型总是依赖真实的 $x_t$，而推理时必须依赖自身预测的（含有误差的）中间态。这导致长文本生成时容易出现语义崩坏或重复循环。

## 3. 架构改进提案 (Proposed Architecture)

本提案建议从底层逻辑重构扩散模型的主干与训练流程，采用 **DiT (Diffusion Transformer)** 架构范式。

### 3.1 架构详细设计

#### A. 长度

在扩散过程开始前，我们需要知道“画布”有多大。

- 所有训练数据统一 `Padding` 到固定的 `MAX_LEN = 128`。
- 确保词表中包含 `[EOS]` 和 `[PAD]`。
- **关键点**：`Label` 中保留 `[PAD]`，**计算 Loss 时即使是 PAD 位置也要计算梯度**。这是教会模型“闭嘴”的唯一方式。

#### B. 主干: DiT with AdaLN-Zero

摒弃将条件作为 Token 输入的做法，将其作为调节器。

对于每一个 Transformer Block：

$$\begin{aligned} \text{Condition} &= \text{MLP}([t, Z]) \rightarrow (\gamma, \beta) \\ x_{norm} &= \text{LayerNorm}(x) \cdot (1 + \gamma) + \beta \\ x_{out} &= \text{Attention}(x_{norm}) + \dots \end{aligned}$$

- **Zero Initialization**: 初始化时将 $\gamma, \beta$ 的最后一层权重设为 0。这使得整个 Block 在训练初期近似恒等映射（Identity），极大加速收敛。

#### C. 自条件化 (Self-Conditioning)

修改模型的输入接口，使其接受之前的预测结果。

- **Input**: $x_{input} = \text{Concat}(x_t, x_{prev\_pred})$
- **Training Loop**:
  - 50% 概率: $x_{prev\_pred} = \emptyset$ (零向量)
  - 50% 概率: $x_{prev\_pred} = \text{StopGrad}(\text{Model}(x_t, \dots))$

## 4. 预期指标与验证 (Validation)

我们将在验证集上监控以下指标来评估重构效果：

1. **Reconstruction Accuracy (BLEU/ROUGE)**: 评估语义重构的准确性（主要目标）。
2. **Perplexity (PPL)**: 评估生成文本的通顺度。
3. **Repetition Rate (4-gram)**: 监控是否出现死循环（自条件化应能显著降低此指标）。
4. **Length Error (MSE)**: 评估长度预测器的准确性。

## 5. 参考文献 (References)

- **DiT**: *Scalable Diffusion Models with Transformers* (Peebles et al., ICCV 2023) - [AdaLN-Zero 来源]
- **Bit Diffusion**: *Analog Bits: Generating Discrete Data using Diffusion Models* (Chen et al., ICLR 2023) - [Self-Conditioning 来源]
- **LLaDA**: *Large Language Diffusion Models* (Nie et al., 2024) - [Masking 策略参考]

