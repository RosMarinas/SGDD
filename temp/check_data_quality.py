"""数据质量检查脚本

用于验证混合数据集的加载情况,打印样本示例和统计信息。
完成后可以删除此脚本。
"""

import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.data import WikipediaDataset, AlpacaDataset, OASSTDataset, MixedDataset
from transformers import RobertaTokenizer
import numpy as np


def analyze_dataset(dataset, name: str, num_samples: int = 5):
    """Analyze and print dataset information"""
    print(f"\n{'='*80}")
    print(f"Dataset: {name}")
    print(f"{'='*80}")
    print(f"Total samples: {len(dataset)}")

    if len(dataset) == 0:
        print("WARNING: Dataset is empty!")
        return

    # Analyze length distribution
    print(f"\nAnalyzing length distribution...")
    lengths = []
    sample_texts = []

    for i in range(min(len(dataset), 1000)):  # Sample 1000 samples
        sample = dataset[i]
        text = sample.get('text', '')
        sample_texts.append(text)

        # Calculate token length
        if 'input_ids' in sample:
            token_length = (sample['input_ids'] != 0).sum().item()  # Exclude padding
        else:
            token_length = len(text.split())
        lengths.append(token_length)

    lengths = np.array(lengths)

    print(f"\nToken Length Statistics:")
    print(f"  Min: {lengths.min()}")
    print(f"  Max: {lengths.max()}")
    print(f"  Mean: {lengths.mean():.2f}")
    print(f"  Median: {np.median(lengths):.2f}")
    print(f"  Std: {lengths.std():.2f}")

    # Length distribution bins
    bins = [0, 20, 40, 60, 80, 100, 128, 256]
    hist, _ = np.histogram(lengths, bins=bins)
    print(f"\nLength Distribution:")
    for i in range(len(bins) - 1):
        count = hist[i]
        pct = count / len(lengths) * 100
        print(f"  {bins[i]:3d}-{bins[i+1]:3d} tokens: {count:4d} samples ({pct:5.1f}%)")

    # Print sample examples
    print(f"\nSample Examples (first {num_samples}):")
    print(f"{'-'*80}")

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        text = sample.get('text', '')

        if 'input_ids' in sample:
            input_ids = sample['input_ids']
            token_length = (input_ids != 0).sum().item()
        else:
            token_length = len(text.split())

        # Truncate long text
        display_text = text if len(text) <= 200 else text[:200] + "..."

        print(f"\nSample #{i+1} ({token_length} tokens):")
        print(f"  {display_text}")

    print(f"{'-'*80}\n")


def check_mixed_dataset(dataset: MixedDataset):
    """Check mixed dataset distribution"""
    print(f"\n{'='*80}")
    print(f"Mixed Dataset Analysis")
    print(f"{'='*80}")

    # Count samples from each source
    source_counts = {}
    for source_name, _ in dataset.index_mapping:
        source_counts[source_name] = source_counts.get(source_name, 0) + 1

    print(f"\nData Source Distribution:")
    total = len(dataset)
    for source, count in source_counts.items():
        pct = count / total * 100
        print(f"  {source:15s}: {count:6d} samples ({pct:5.1f}%)")

    print(f"\nTotal: {total} samples")

    # Print samples from each source
    print(f"\nSample Examples from Each Source:")
    print(f"{'-'*80}")

    for source in source_counts.keys():
        # Find first sample from this source
        for i, (src, idx) in enumerate(dataset.index_mapping):
            if src == source:
                sample = dataset[i]
                text = sample.get('text', '')

                if 'input_ids' in sample:
                    input_ids = sample['input_ids']
                    token_length = (input_ids != 0).sum().item()
                else:
                    token_length = len(text.split())

                display_text = text if len(text) <= 200 else text[:200] + "..."

                print(f"\n{source} example (#{i+1}, {token_length} tokens):")
                print(f"  {display_text}")
                break

    print(f"{'-'*80}\n")


def main():
    """主函数"""
    # 设置UTF-8编码输出
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "="*80)
    print("SGDD Data Quality Check Tool")
    print("="*80)

    # 配置
    max_token_length = 128

    # 1. Check Wikipedia dataset
    print("\nLoading Wikipedia dataset...")
    wiki_dataset = WikipediaDataset(
        num_samples=1000,  # Small scale test
        min_length=20,
        max_length=128,
        max_token_length=max_token_length,
        split="train",
    )
    analyze_dataset(wiki_dataset, "Wikipedia", num_samples=3)

    # 2. Check Alpaca dataset
    print("\nLoading Alpaca dataset...")
    try:
        alpaca_dataset = AlpacaDataset(
            num_samples=1000,  # Small scale test
            min_length=20,
            max_token_length=max_token_length,
            split="train",
        )
        analyze_dataset(alpaca_dataset, "Alpaca", num_samples=3)
    except Exception as e:
        print(f"WARNING: Alpaca dataset loading failed: {e}")
        print("Tip: Make sure datasets library is installed: uv pip install datasets")

    # 3. Check OASST1 dataset
    print("\nLoading OASST1 dataset...")
    try:
        oasst1_dataset = OASSTDataset(
            num_samples=500,  # Small scale test
            min_length=20,
            max_token_length=max_token_length,
            split="train",
        )
        analyze_dataset(oasst1_dataset, "OASST1", num_samples=3)
    except Exception as e:
        print(f"WARNING: OASST1 dataset loading failed: {e}")
        print("Tip: Make sure datasets library is installed: uv pip install datasets")

    # 4. Check mixed dataset
    print("\nLoading mixed dataset...")
    try:
        mixed_dataset = MixedDataset(
            wiki_config={'num_samples': 600, 'min_length': 20, 'max_length': 128, 'max_token_length': max_token_length},
            alpaca_config={'num_samples': 300, 'min_length': 20, 'max_token_length': max_token_length},
            oasst1_config={'num_samples': 100, 'min_length': 20, 'max_token_length': max_token_length},
            split="train",
        )
        check_mixed_dataset(mixed_dataset)

        # Print mixed dataset sample examples
        print(f"\nMixed Dataset Sample Examples (first 5):")
        print(f"{'-'*80}")
        for i in range(min(5, len(mixed_dataset))):
            sample = mixed_dataset[i]
            source = sample.get('source', 'unknown')
            text = sample.get('text', '')

            if 'input_ids' in sample:
                input_ids = sample['input_ids']
                token_length = (input_ids != 0).sum().item()
            else:
                token_length = len(text.split())

            display_text = text if len(text) <= 200 else text[:200] + "..."

            print(f"\nSample #{i+1} [{source}] ({token_length} tokens):")
            print(f"  {display_text}")

        print(f"{'-'*80}\n")

    except Exception as e:
        print(f"WARNING: Mixed dataset loading failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("Data Quality Check Complete!")
    print("="*80)
    print("\nTips:")
    print("  - If all datasets loaded successfully, you can start training")
    print("  - If any dataset failed, check network connection and dependencies")
    print("  - You can delete this script after checking: rm temp/check_data_quality.py")
    print()


if __name__ == "__main__":
    main()
