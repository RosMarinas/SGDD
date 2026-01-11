"""数据加载和预处理管道

支持Wikipedia和QQP数据集的加载、tokenization和批处理。
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from datasets import load_dataset
from typing import Dict, Optional, Tuple
import random


class WikipediaDataset(Dataset):
    """Wikipedia文本重构数据集

    任务: 输入文本 → 编码 → 解码 → 相同文本
    """

    def __init__(
        self,
        num_samples: int = 100000,
        min_length: int = 20,
        max_length: int = 200,
        max_token_length: int = 64,
        split: str = "train",
    ):
        """
        Args:
            num_samples: 使用样本数量
            min_length: 最小文本长度(字符)
            max_length: 最大文本长度(字符)
            max_token_length: 最大token长度
            split: train or validation
        """
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        self.max_token_length = max_token_length

        # 加载tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # 加载数据集
        print(f"Loading Wikipedia dataset ({split} split)...")
        wiki_data = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

        # 过滤和预处理
        self.samples = self._filter_and_preprocess(wiki_data, num_samples)

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

        # 移除batch维度
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text": text,  # 保存原始文本用于调试
        }


class QQPDataset(Dataset):
    """Quora Question Pairs改写数据集

    任务: 输入问题A → 编码 → 解码 → 问题B(相同含义)
    """

    def __init__(
        self,
        num_samples: int = 100000,
        min_length: int = 10,
        max_token_length: int = 64,
        split: str = "train",
    ):
        """
        Args:
            num_samples: 使用样本数量
            min_length: 最小问题长度(字符)
            max_token_length: 最大token长度
            split: train or validation
        """
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_token_length = max_token_length

        # 加载tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # 加载数据集
        print(f"Loading QQP dataset ({split} split)...")
        qqp_data = load_dataset("quora", split="train")

        # 过滤和预处理
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
            # 只选择标记为重复的问题对
            if item["is_duplicate"] != 1:
                continue

            q1 = item["questions"]["text"][0].strip()
            q2 = item["questions"]["text"][1].strip()

            # 过滤过短的问题
            if len(q1) < self.min_length or len(q2) < self.min_length:
                continue

            # 过滤完全相同的问题
            if q1 == q2:
                continue

            samples.append((q1, q2))

            if len(samples) >= num_samples:
                break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        q1, q2 = self.samples[idx]

        # Tokenize两个问题
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

        # 移除batch维度
        return {
            "input_ids": encoded_q1["input_ids"].squeeze(0),
            "attention_mask_q1": encoded_q1["attention_mask"].squeeze(0),
            "target_ids": encoded_q2["input_ids"].squeeze(0),
            "attention_mask_q2": encoded_q2["attention_mask"].squeeze(0),
            "text_q1": q1,
            "text_q2": q2,
        }


def collate_fn_wiki(batch: list) -> Dict[str, torch.Tensor]:
    """Wikipedia数据集的collate函数"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "texts": [item["text"] for item in batch],
    }


def collate_fn_qqp(batch: list) -> Dict[str, torch.Tensor]:
    """QQP数据集的collate函数"""
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
    """数据加载器工厂函数

    Args:
        dataset_name: 数据集名称 ('wikipedia' or 'qqp')
        split: 'train' or 'validation'
        batch_size: 批大小
        num_workers: DataLoader worker数量
        pin_memory: 是否固定内存
        **dataset_kwargs: 传递给数据集的额外参数

    Returns:
        DataLoader对象
    """
    # 创建数据集
    if dataset_name == "wikipedia":
        dataset = WikipediaDataset(split=split, **dataset_kwargs)
        collate_fn = collate_fn_wiki
    elif dataset_name == "qqp":
        dataset = QQPDataset(split=split, **dataset_kwargs)
        collate_fn = collate_fn_qqp
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return dataloader
