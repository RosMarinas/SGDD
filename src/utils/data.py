"""数据加载和预处理管道

支持Wikipedia和QQP数据集的加载、tokenization和批处理。
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from datasets import load_dataset
from typing import Dict, Optional, Tuple
import random
from pathlib import Path
import os

# ### 新增：延迟导入 ModelScope 组件以避免版本兼容性问题 ###
# from modelscope.msdatasets import MsDataset
# from modelscope.hub.snapshot_download import snapshot_download


# 数据集缓存目录
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class WikipediaDataset(Dataset):
    """Wikipedia文本重构数据集"""

    def __init__(
        self,
        num_samples: int = 100000,
        min_length: int = 20,
        max_length: int = 200,
        max_token_length: int = 64,
        split: str = "train",
    ):
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        self.max_token_length = max_token_length

        # ### 修改 1：从 ModelScope 下载并加载 Tokenizer ###
        # 原始代码: self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", ...)
        print("Loading Tokenizer...")
        try:
            # 延迟导入以避免版本兼容性问题
            from modelscope.hub.snapshot_download import snapshot_download
            # 使用 ModelScope 上的 roberta-base 镜像
            model_dir = snapshot_download('AI-ModelScope/roberta-base', cache_dir=str(DATA_DIR / "model_cache"))
            self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        except Exception as e:
            print(f"ModelScope tokenizer load failed: {e}, trying HuggingFace...")
            # 回退尝试（如果本地有缓存）
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        # ### 修改 2：使用 MsDataset 加载 Wikitext ###
        print(f"Loading Wikipedia dataset ({split} split)...")

        try:
            # 延迟导入以避免版本兼容性问题
            from modelscope.msdatasets import MsDataset
            # ModelScope 的 cache 机制由其内部管理，这里主要关注加载
            # 对应 HF 的 wikitext-103-raw-v1
            ms_ds = MsDataset.load(
                'wikitext',
                subset_name='wikitext-103-raw-v1',
                split='train',
                trust_remote_code=True
            )
            # 转换为 HuggingFace Dataset 对象
            wiki_data = ms_ds.to_hf_dataset()
            print(f"Dataset loaded from ModelScope.")
        except Exception as e:
            print(f"ModelScope dataset load failed: {e}, trying HuggingFace...")
            # 回退到 HuggingFace
            wiki_data = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
            print(f"Dataset loaded from HuggingFace.")

        # 过滤和预处理 (原逻辑保持不变)
        self.samples = self._filter_and_preprocess(wiki_data, num_samples)

        # 分割训练/验证集 (原逻辑保持不变)
        if split == "validation":
            self.samples = self.samples[: int(len(self.samples) * 0.05)]
        else:
            self.samples = self.samples[int(len(self.samples) * 0.05) :]

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _filter_and_preprocess(self, dataset, num_samples: int) -> list:
        """过滤和预处理数据"""
        samples = []

        for item in dataset:
            text = item["text"].strip()

            # 过滤空文本和过短的文本
            if len(text) < self.min_length or len(text) > self.max_length:
                continue

            # 过滤标题和特殊格式
            if text.startswith("=") or text.startswith("{{"):
                continue

            samples.append(text)

            if len(samples) >= num_samples:
                break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        text = self.samples[idx]

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text": text,
        }


class QQPDataset(Dataset):
    """Quora Question Pairs改写数据集"""

    def __init__(
        self,
        num_samples: int = 100000,
        min_length: int = 10,
        max_token_length: int = 64,
        split: str = "train",
    ):
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_token_length = max_token_length

        # ### 修改 3：Tokenizer 同样改为从 ModelScope 加载 (或者复用上面的路径) ###
        # 这里为了代码独立性再写一次，实际运行会自动使用缓存，不会重复下载
        print("Loading Tokenizer...")
        try:
            # 延迟导入以避免版本兼容性问题
            from modelscope.hub.snapshot_download import snapshot_download
            model_dir = snapshot_download('AI-ModelScope/roberta-base', cache_dir=str(DATA_DIR / "model_cache"))
            self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        except Exception as e:
            print(f"ModelScope tokenizer load failed: {e}, trying HuggingFace...")
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # ### 修改 4：使用 MsDataset 加载 QQP ###
        print(f"Loading QQP dataset ({split} split)...")

        try:
            # 延迟导入以避免版本兼容性问题
            from modelscope.msdatasets import MsDataset
            # ModelScope 上 Quora 数据集通常叫 'quora' 或 'quora-qqp'
            ms_ds = MsDataset.load(
                'quora',
                split='train'
            )
            # 转换为 HF Dataset
            qqp_data = ms_ds.to_hf_dataset()
            print(f"Dataset loaded from ModelScope.")
        except Exception as e:
            print(f"ModelScope dataset load failed: {e}, trying HuggingFace...")
            # 回退到 HuggingFace
            qqp_data = load_dataset('quora', 'quora', split='train')
            print(f"Dataset loaded from HuggingFace.")

        # 过滤和预处理 (原逻辑保持不变)
        self.samples = self._filter_and_preprocess(qqp_data, num_samples)

        # 分割训练/验证集
        if split == "validation":
            self.samples = self.samples[: int(len(self.samples) * 0.05)]
        else:
            self.samples = self.samples[int(len(self.samples) * 0.05) :]

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _filter_and_preprocess(self, dataset, num_samples: int) -> list:
        """过滤和预处理数据"""
        samples = []

        for item in dataset:
            # ### 注意：ModelScope 转换后的格式通常与 HF 一致，但以防万一可以打印 item 查看 ###
            # 标准 HF QQP 结构: {'is_duplicate': int, 'questions': {'text': [str, str], ...}}
            
            if item["is_duplicate"] != 1:
                continue

            q1 = item["questions"]["text"][0].strip()
            q2 = item["questions"]["text"][1].strip()

            if len(q1) < self.min_length or len(q2) < self.min_length:
                continue

            if q1 == q2:
                continue

            samples.append((q1, q2))

            if len(samples) >= num_samples:
                break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        q1, q2 = self.samples[idx]

        encoded_q1 = self.tokenizer(
            q1,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        encoded_q2 = self.tokenizer(
            q2,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded_q1["input_ids"].squeeze(0),
            "attention_mask_q1": encoded_q1["attention_mask"].squeeze(0),
            "target_ids": encoded_q2["input_ids"].squeeze(0),
            "attention_mask_q2": encoded_q2["attention_mask"].squeeze(0),
            "text_q1": q1,
            "text_q2": q2,
        }

# Collate functions 和 get_dataloader 保持不变
def collate_fn_wiki(batch: list) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "texts": [item["text"] for item in batch],
    }

def collate_fn_qqp(batch: list) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask_q1 = torch.stack([item["attention_mask_q1"] for item in batch])
    target_ids = torch.stack([item["target_ids"] for item in batch])
    attention_mask_q2 = torch.stack([item["attention_mask_q2"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask_q1,
        "target_ids": target_ids,
        "target_mask": attention_mask_q2,
        "texts_q1": [item["text_q1"] for item in batch],
        "texts_q2": [item["text_q2"] for item in batch],
    }

class AlpacaDataset(Dataset):
    """Alpaca指令数据集 - 支持文本重构任务"""

    def __init__(
        self,
        num_samples: int = 30000,
        min_length: int = 20,
        max_token_length: int = 128,
        split: str = "train",
    ):
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_token_length = max_token_length

        # 加载Tokenizer
        print("Loading Tokenizer for Alpaca...")
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            model_dir = snapshot_download('AI-ModelScope/roberta-base', cache_dir=str(DATA_DIR / "model_cache"))
            self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        except Exception as e:
            print(f"ModelScope tokenizer load failed: {e}, trying HuggingFace...")
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # 加载Alpaca数据集
        print(f"Loading Alpaca dataset ({split} split)...")
        alpaca_data = load_dataset('tatsu-lab/alpaca', split='train')

        # 过滤和预处理
        self.samples = self._filter_and_preprocess(alpaca_data, num_samples)

        # 分割训练/验证集
        if split == "validation":
            self.samples = self.samples[: int(len(self.samples) * 0.05)]
        else:
            self.samples = self.samples[int(len(self.samples) * 0.05) :]

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _filter_and_preprocess(self, dataset, num_samples: int) -> list:
        """过滤和预处理数据"""
        samples = []

        for item in dataset:
            # 拼接Instruction和Input
            instruction = item['instruction'].strip()
            input_text = item.get('input', '').strip() if item.get('input') else ''
            output = item['output'].strip()

            # 拼接源文本
            if input_text:
                source_text = f"{instruction}\n{input_text}"
            else:
                source_text = instruction

            # 检查长度 (基于token数量)
            encoded_output = self.tokenizer(
                output,
                max_length=self.max_token_length,
                truncation=False,
                return_tensors=None
            )

            token_length = len(encoded_output["input_ids"])
            if token_length < self.min_length or token_length > self.max_token_length:
                continue

            # 对于重构任务,我们使用output作为重构目标
            samples.append(output)

            if len(samples) >= num_samples:
                break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        text = self.samples[idx]

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text": text,
        }


class OASSTDataset(Dataset):
    """OpenAssistant对话数据集 - 支持文本重构任务"""

    def __init__(
        self,
        num_samples: int = 10000,
        min_length: int = 20,
        max_token_length: int = 128,
        split: str = "train",
    ):
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_token_length = max_token_length

        # 加载Tokenizer
        print("Loading Tokenizer for OASST1...")
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            model_dir = snapshot_download('AI-ModelScope/roberta-base', cache_dir=str(DATA_DIR / "model_cache"))
            self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        except Exception as e:
            print(f"ModelScope tokenizer load failed: {e}, trying HuggingFace...")
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # 加载OpenAssistant数据集
        print(f"Loading OpenAssistant OASST1 dataset ({split} split)...")
        oasst_data = load_dataset('OpenAssistant/oasst1', split='train')

        # 过滤和预处理
        self.samples = self._filter_and_preprocess(oasst_data, num_samples)

        # 分割训练/验证集
        if split == "validation":
            self.samples = self.samples[: int(len(self.samples) * 0.05)]
        else:
            self.samples = self.samples[int(len(self.samples) * 0.05) :]

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _filter_and_preprocess(self, dataset, num_samples: int) -> list:
        """过滤和预处理数据"""
        samples = []

        for item in dataset:
            # 只提取assistant的回复
            if item['role'] != 'assistant':
                continue

            # 过滤低质量回复
            if item.get('rank') is None or item['rank'] < 0:
                continue

            # 只保留英文
            if item.get('lang') != 'en':
                continue

            text = item['text'].strip()

            # 过滤过短的回复 (纯标点或表情符号)
            if len(text) < 5:
                continue

            # 检查token长度
            encoded = self.tokenizer(
                text,
                max_length=self.max_token_length,
                truncation=False,
                return_tensors=None
            )

            token_length = len(encoded["input_ids"])
            if token_length < self.min_length or token_length > self.max_token_length:
                continue

            samples.append(text)

            if len(samples) >= num_samples:
                break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        text = self.samples[idx]

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text": text,
        }


class MixedDataset(Dataset):
    """混合数据集 - 按比例混合多个数据源"""

    def __init__(
        self,
        wiki_config: dict = None,
        alpaca_config: dict = None,
        oasst1_config: dict = None,
        split: str = "train",
    ):
        """
        Args:
            wiki_config: Wikipedia数据集配置
            alpaca_config: Alpaca数据集配置
            oasst1_config: OASST1数据集配置
            split: train或validation
        """
        self.split = split

        # 加载各数据源
        print("Loading mixed dataset...")
        self.datasets = {}
        self.sample_ratios = {}

        if wiki_config and wiki_config.get('num_samples', 0) > 0:
            print("Loading Wikipedia subset...")
            self.datasets['wikipedia'] = WikipediaDataset(
                num_samples=wiki_config['num_samples'],
                min_length=wiki_config.get('min_length', 20),
                max_length=wiki_config.get('max_length', 128),
                max_token_length=wiki_config.get('max_token_length', 128),
                split=split,
            )
            self.sample_ratios['wikipedia'] = wiki_config['num_samples']

        if alpaca_config and alpaca_config.get('num_samples', 0) > 0:
            print("Loading Alpaca subset...")
            self.datasets['alpaca'] = AlpacaDataset(
                num_samples=alpaca_config['num_samples'],
                min_length=alpaca_config.get('min_length', 20),
                max_token_length=alpaca_config.get('max_token_length', 128),
                split=split,
            )
            self.sample_ratios['alpaca'] = alpaca_config['num_samples']

        if oasst1_config and oasst1_config.get('num_samples', 0) > 0:
            print("Loading OASST1 subset...")
            self.datasets['oasst1'] = OASSTDataset(
                num_samples=oasst1_config['num_samples'],
                min_length=oasst1_config.get('min_length', 20),
                max_token_length=oasst1_config.get('max_token_length', 128),
                split=split,
            )
            self.sample_ratios['oasst1'] = oasst1_config['num_samples']

        # 计算总样本数和比例
        self.total_samples = sum(self.sample_ratios.values())
        self.ratios = {k: v / self.total_samples for k, v in self.sample_ratios.items()}

        print(f"Dataset composition: {self.sample_ratios}")
        print(f"Total samples: {self.total_samples}")

        # 构建索引映射
        self.index_mapping = self._build_index_mapping()

    def _build_index_mapping(self):
        """构建全局索引到(数据源,局部索引)的映射"""
        mapping = []
        for source_name, count in self.sample_ratios.items():
            dataset_len = len(self.datasets[source_name])
            for i in range(count):
                # 循环采样
                local_idx = i % dataset_len
                mapping.append((source_name, local_idx))

        # 打乱顺序
        if self.split == "train":
            random.shuffle(mapping)

        return mapping

    def __len__(self) -> int:
        return len(self.index_mapping)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        source_name, local_idx = self.index_mapping[idx]
        sample = self.datasets[source_name][local_idx]

        # 添加数据源标记
        sample['source'] = source_name

        return sample


def collate_fn_mixed(batch: list) -> Dict[str, torch.Tensor]:
    """混合数据集的collate函数"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "texts": [item["text"] for item in batch],
        "sources": [item.get("source", "unknown") for item in batch],
    }


def get_dataloader(
    dataset_name: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    if dataset_name == "wikipedia":
        dataset = WikipediaDataset(split=split, **dataset_kwargs)
        collate_fn = collate_fn_wiki
    elif dataset_name == "qqp":
        dataset = QQPDataset(split=split, **dataset_kwargs)
        collate_fn = collate_fn_qqp
    elif dataset_name == "mixed":
        # 混合数据集需要特殊处理
        wiki_config = {
            'num_samples': dataset_kwargs.get('wiki_num_samples', 0),
            'min_length': dataset_kwargs.get('wiki_min_length', 20),
            'max_length': dataset_kwargs.get('wiki_max_length', 128),
            'max_token_length': dataset_kwargs.get('max_token_length', 128),
        }
        alpaca_config = {
            'num_samples': dataset_kwargs.get('alpaca_num_samples', 0),
            'min_length': dataset_kwargs.get('alpaca_min_length', 20),
            'max_token_length': dataset_kwargs.get('max_token_length', 128),
        }
        oasst1_config = {
            'num_samples': dataset_kwargs.get('oasst1_num_samples', 0),
            'min_length': dataset_kwargs.get('oasst1_min_length', 20),
            'max_token_length': dataset_kwargs.get('max_token_length', 128),
        }

        dataset = MixedDataset(
            wiki_config=wiki_config if wiki_config['num_samples'] > 0 else None,
            alpaca_config=alpaca_config if alpaca_config['num_samples'] > 0 else None,
            oasst1_config=oasst1_config if oasst1_config['num_samples'] > 0 else None,
            split=split,
        )
        collate_fn = collate_fn_mixed
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return dataloader