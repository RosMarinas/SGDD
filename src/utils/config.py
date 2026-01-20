"""配置管理系统

支持YAML配置文件的加载和保存,包含模型、训练、数据和推理的所有超参数。
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """模型架构配置"""

    # 编码器配置
    encoder_name: str = "roberta-base"
    encoder_freeze: bool = True
    semantic_dim: int = 512  # 对应 SGDDConfig 的 hidden_dim

    # VIB配置
    kl_weight: float = 0.001  # KL divergence weight
    kl_anneal_steps: int = 10000  # KL annealing steps
    kl_threshold: float = 2.0  # Free bits (min KL) - Increased to prevent vanishing
    contrastive_weight: float = 0.1  # Weight for isotropy regularization - Increased to prevent collapse

    # 解码器配置 (字段名与 SGDDConfig 保持一致)
    num_layers: int = 6  # 对应 SGDDConfig 的 num_layers
    num_heads: int = 8  # 对应 SGDDConfig 的 num_heads
    ffn_dim: int = 2048  # 对应 SGDDConfig 的 ffn_dim
    max_length: int = 128  # 对应 SGDDConfig 的 max_len
    dropout: float = 0.1  # 对应 SGDDConfig 的 dropout

    # 扩散配置
    num_diffusion_steps: int = 1000
    noise_schedule: str = "cosine"  # cosine, linear

    # 训练优化配置 (字段名与 SGDDConfig 保持一致)
    use_self_conditioning: bool = True  # 对应 SGDDConfig 的 use_self_conditioning
    compute_pad_loss: bool = False  # 对应 SGDDConfig 的 compute_pad_loss
    compute_eos_loss: bool = True  # 对应 SGDDConfig 的 compute_eos_loss
    word_dropout_prob: float = 0.3  # Word dropout概率 (0.0-0.5), 强制解码器依赖语义向量z
    mask_token_id: int = 50264  # RoBERTa MASK token ID


@dataclass
class TrainingConfig:
    """训练超参数配置"""

    # 基础训练参数
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    grad_clip: float = 1.0

    # 学习率调度
    lr_scheduler: str = "cosine"  # cosine, linear, constant
    warmup_steps: int = 1000

    # 混合精度
    use_fp16: bool = True

    # CFG训练
    cfg_drop_prob: float = 0.1  # 10% unconditional batches

    # 日志和保存
    log_interval: int = 10
    save_interval: int = 1000
    save_epochs: int = 1
    eval_interval: int = 500

    # WandB
    use_wandb: bool = True
    wandb_project: str = "sgdd"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None


@dataclass
class DataConfig:
    """数据配置"""

    # 数据集选择
    dataset: str = "wikipedia"  # wikipedia, qqp, mixed

    # Wikipedia配置
    wiki_num_samples: int = 100000
    wiki_min_length: int = 20
    wiki_max_length: int = 128

    # QQP配置
    qqp_num_samples: int = 100000
    qqp_min_length: int = 10

    # 混合数据配置
    mixing_strategy: str = "none"  # "none" | "fast_validation" | "scale_up" | "full_scale"
    total_samples: int = 100000

    # Alpaca配置
    alpaca_num_samples: int = 0
    alpaca_min_length: int = 20

    # OpenAssistant配置
    oasst1_num_samples: int = 0
    oasst1_min_length: int = 20

    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True

    # 训练/验证分割
    val_split: float = 0.03


@dataclass
class InferenceConfig:
    """推理配置"""

    # MaskGIT采样
    num_inference_steps: int = 16  # 16步解码
    temperature: float = 1.0

    # CFG
    cfg_scale: float = 2.0  # 引导强度

    # 采样策略
    sampling_strategy: str = "confidence"  # confidence, multinomial
    top_k: Optional[int] = None
    top_p: Optional[float] = None


@dataclass
class SGDDConfig:
    """完整配置"""

    # 子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # 系统配置
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # 任务类型
    task: str = "reconstruction"  # reconstruction, paraphrase

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SGDDConfig":
        """从YAML文件加载配置"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # 递归构建配置对象
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "SGDDConfig":
        """从字典递归构建配置"""
        # 处理子配置
        if "model" in data and isinstance(data["model"], dict):
            data["model"] = ModelConfig(**data["model"])
        if "training" in data and isinstance(data["training"], dict):
            data["training"] = TrainingConfig(**data["training"])
        if "data" in data and isinstance(data["data"], dict):
            data["data"] = DataConfig(**data["data"])
        if "inference" in data and isinstance(data["inference"], dict):
            data["inference"] = InferenceConfig(**data["inference"])

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """保存配置到YAML文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 转换为可序列化的字典
        data = asdict(self)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def __str__(self) -> str:
        """打印配置摘要"""
        lines = ["=" * 60, "SGDD Configuration", "=" * 60]

        # 模型配置
        lines.append("\n[Model]")
        lines.append(f"  Encoder: {self.model.encoder_name} (frozen={self.model.encoder_freeze})")
        lines.append(f"  Decoder: {self.model.num_layers} layers, {self.model.semantic_dim} dim, {self.model.num_heads} heads")
        lines.append(f"  Max Length: {self.model.max_length}")
        lines.append(f"  Diffusion Steps: {self.model.num_diffusion_steps}")
        lines.append(f"  Self-Conditioning: {self.model.use_self_conditioning}")
        lines.append(f"  Compute PAD Loss: {self.model.compute_pad_loss}")

        # 训练配置
        lines.append("\n[Training]")
        lines.append(f"  Learning Rate: {self.training.learning_rate}")
        lines.append(f"  Batch Size: {self.training.batch_size}")
        lines.append(f"  Epochs: {self.training.num_epochs}")
        lines.append(f"  FP16: {self.training.use_fp16}")
        lines.append(f"  CFG Drop Prob: {self.training.cfg_drop_prob}")

        # 数据配置
        lines.append("\n[Data]")
        lines.append(f"  Dataset: {self.data.dataset}")
        if self.data.dataset == "wikipedia":
            lines.append(f"  Samples: {self.data.wiki_num_samples}")
        elif self.data.dataset == "qqp":
            lines.append(f"  Samples: {self.data.qqp_num_samples}")

        # 推理配置
        lines.append("\n[Inference]")
        lines.append(f"  Steps: {self.inference.num_inference_steps}")
        lines.append(f"  CFG Scale: {self.inference.cfg_scale}")
        lines.append(f"  Temperature: {self.inference.temperature}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)
