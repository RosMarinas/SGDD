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

# ### 新增：导入 ModelScope 组件 ###

from modelscope.msdatasets import MsDataset
from modelscope.hub.snapshot_download import snapshot_download


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
        print("Loading Tokenizer from ModelScope...")
        try:
            # 使用 ModelScope 上的 roberta-base 镜像
            model_dir = snapshot_download('AI-ModelScope/roberta-base', cache_dir=str(DATA_DIR / "model_cache"))
            self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        except Exception as e:
            print(f"ModelScope tokenizer load failed: {e}")
            # 回退尝试（如果本地有缓存）
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        # ### 修改 2：使用 MsDataset 加载 Wikitext ###
        print(f"Loading Wikipedia dataset ({split} split) from ModelScope...")
        
        # ModelScope 的 cache 机制由其内部管理，这里主要关注加载
        # 对应 HF 的 wikitext-103-raw-v1
        ms_ds = MsDataset.load(
            'wikitext', 
            subset_name='wikitext-103-raw-v1', 
            split='train' ,
            trust_remote_code=True# ModelScope 有时 split 命名习惯不同，但 wikitext 通常一致
        )
        
        # 关键步骤：转换为 HuggingFace Dataset 对象，这样你后面的代码不用改
        wiki_data = ms_ds.to_hf_dataset()
        
        print(f"Dataset loaded from ModelScope.")

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
        model_dir = snapshot_download('AI-ModelScope/roberta-base', cache_dir=str(DATA_DIR / "model_cache"))
        self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)

        # ### 修改 4：使用 MsDataset 加载 QQP ###
        print(f"Loading QQP dataset ({split} split) from ModelScope...")
        
        # ModelScope 上 Quora 数据集通常叫 'quora' 或 'quora-qqp'
        # 我们使用 'quora'，它对应 HF 的版本
        ms_ds = MsDataset.load(
            'quora', 
            split='train'
        )
        
        # 转换为 HF Dataset
        qqp_data = ms_ds.to_hf_dataset()
        print(f"Dataset loaded from ModelScope.")

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