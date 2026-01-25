"""数据加载和预处理管道

支持BookCorpus数据集的加载、tokenization和批处理。
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, AutoTokenizer
from datasets import load_from_disk
from typing import Dict, Optional, Tuple
from pathlib import Path
import os

# 数据集缓存目录
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class BookCorpusDataset(Dataset):
    """BookCorpus文本重构数据集"""

    def __init__(
        self,
        dataset_path: str,
        tokenizer_name: str = "BAAI/bge-m3",
        max_token_length: int = 64,
        min_length: int = 5,
        split: str = "train",
    ):
        self.max_token_length = max_token_length
        self.min_length = min_length
        self.split = split

        # 加载 Tokenizer
        print(f"Loading Tokenizer: {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 加载数据集
        print(f"Loading BookCorpus dataset from {dataset_path} ({split} split)...")
        try:
            # 加载保存到磁盘的数据集
            full_dataset = load_from_disk(dataset_path)
            
            if split in full_dataset:
                self.dataset = full_dataset[split]
            else:
                # 如果没有显式的split，根据split参数进行切分或者直接使用
                # 这里假设数据集结构是 {'train': ..., 'validation': ...}
                # 如果结构不同，可能需要调整
                print(f"Warning: Split '{split}' not found in dataset dict. Keys: {full_dataset.keys()}")
                # 尝试默认行为，或者如果是一个 Dataset 对象而不是 DatasetDict
                if hasattr(full_dataset, 'features'): # 单个 Dataset
                     # 这里简单处理，如果是 validation 且只有单一dataset，可能需要按比例切分，
                     # 但通常 load_from_disk 对应的是 save_to_disk 的结果，通常是 DatasetDict
                     self.dataset = full_dataset
                else:
                    raise ValueError(f"Split {split} not found in dataset")
            
            print(f"Dataset loaded. Size: {len(self.dataset)}")
            
        except Exception as e:
            print(f"Failed to load dataset from {dataset_path}: {e}")
            raise e

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        # datasets 库支持高效的磁盘访问
        item = self.dataset[idx]
        text = item["text"]

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


def collate_fn_bookcorpus(batch: list) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "texts": [item["text"] for item in batch],
    }


def get_dataloader(
    dataset_name: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    tokenizer_name: str = "BAAI/bge-m3",
    **dataset_kwargs,
) -> DataLoader:
    
    if dataset_name == "bookcorpus":
        # 提取相关参数
        dataset_path = dataset_kwargs.get('dataset_path', 'data/BookCorpus/final_dataset_1.4B')
        max_token_length = dataset_kwargs.get('max_token_length', 64)
        min_length = dataset_kwargs.get('min_length', 5)
        
        dataset = BookCorpusDataset(
            dataset_path=dataset_path,
            tokenizer_name=tokenizer_name,
            max_token_length=max_token_length,
            min_length=min_length,
            split=split
        )
        collate_fn = collate_fn_bookcorpus
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Only 'bookcorpus' is supported.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return dataloader
