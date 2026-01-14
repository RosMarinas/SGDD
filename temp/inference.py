"""
简单的推理脚本
用于从训练好的模型生成文本
"""

import sys
from pathlib import Path
import torch
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.sgdd import SGDDModel, SGDDConfig
from utils.checkpoints import load_best_model


def load_model(checkpoint_path: str, device: str = "cuda"):
    """加载训练好的模型

    Args:
        checkpoint_path: 检查点目录路径
        device: 设备 ("cuda" 或 "cpu")

    Returns:
        model: 加载好的模型
    """
    print(f"Loading model from {checkpoint_path}...")

    # 创建默认配置
    config = SGDDConfig()
    model = SGDDModel(config).to(device)

    # 加载最佳模型
    try:
        best_metric = load_best_model(checkpoint_path, model, device)
        print(f"✓ Model loaded successfully (best metric: {best_metric:.4f})")
    except FileNotFoundError as e:
        print(f"✗ Error loading model: {e}")
        print("Trying to load from checkpoint file directly...")

        # 尝试直接加载
        import os
        checkpoint_dir = Path(checkpoint_path)
        best_model_path = checkpoint_dir / "best_model.pt"

        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print("✓ Model loaded from best_model.pt")
        else:
            raise FileNotFoundError(f"No model found in {checkpoint_path}")

    model.eval()
    return model


def generate_text(
    model: SGDDModel,
    input_text: str,
    num_steps: int = 16,
    guidance_scale: float = 2.0,
    temperature: float = 1.0,
    max_length: int = 256,
) -> str:
    """生成文本

    Args:
        model: SGDD模型
        input_text: 输入文本
        num_steps: 推理步数
        guidance_scale: CFG引导强度
        temperature: 采样温度
        max_length: 最大长度

    Returns:
        generated_text: 生成的文本
    """
    with torch.no_grad():
        generated = model.generate(
            input_text=input_text,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            temperature=temperature,
            max_length=max_length,
        )
    return generated


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Generate text from trained SGDD model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/second",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input text to generate from (if not provided, will use interactive mode)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=16,
        help="Number of inference steps (default: 16)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.0,
        help="CFG guidance scale (default: 2.0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum generation length (default: 256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    args = parser.parse_args()

    # 检查设备
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU instead")
        device = "cpu"

    print(f"Using device: {device}\n")

    # 加载模型
    model = load_model(args.checkpoint, device)

    # 生成参数
    print(f"Generation parameters:")
    print(f"  - Num steps: {args.num_steps}")
    print(f"  - CFG scale: {args.cfg_scale}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Max length: {args.max_length}")
    print()

    # 如果提供了输入文本,直接生成
    if args.input:
        print(f"Input: {args.input}")
        print("-" * 60)
        generated = generate_text(
            model,
            args.input,
            num_steps=args.num_steps,
            guidance_scale=args.cfg_scale,
            temperature=args.temperature,
            max_length=args.max_length,
        )
        print(f"Generated: {generated}")
    else:
        # 交互模式
        print("Interactive mode (press Ctrl+C to exit)")
        print("=" * 60)

        examples = [
            "The history of artificial intelligence dates back to the 1950s.",
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Neural networks are computing systems inspired by biological neural networks.",
        ]

        try:
            while True:
                print("\nOptions:")
                print("  1. Use example input")
                print("  2. Enter custom input")
                print("  3. Exit")

                choice = input("\nChoose option (1-3): ").strip()

                if choice == "1":
                    # 使用示例
                    print("\nExample inputs:")
                    for i, ex in enumerate(examples, 1):
                        print(f"  {i}. {ex[:80]}...")

                    idx = input(f"\nSelect example (1-{len(examples)}): ").strip()
                    try:
                        idx = int(idx) - 1
                        if 0 <= idx < len(examples):
                            input_text = examples[idx]
                        else:
                            print("Invalid selection!")
                            continue
                    except ValueError:
                        print("Invalid input!")
                        continue

                elif choice == "2":
                    # 自定义输入
                    input_text = input("\nEnter input text: ").strip()
                    if not input_text:
                        print("Input cannot be empty!")
                        continue

                elif choice == "3":
                    print("Goodbye!")
                    break

                else:
                    print("Invalid choice!")
                    continue

                # 生成文本
                print(f"\nInput: {input_text}")
                print("-" * 60)
                generated = generate_text(
                    model,
                    input_text,
                    num_steps=args.num_steps,
                    guidance_scale=args.cfg_scale,
                    temperature=args.temperature,
                    max_length=args.max_length,
                )
                print(f"Generated: {generated}")
                print("=" * 60)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
