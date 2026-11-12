"""MedBharat-1 Training Entry Point.

Usage:
  # Train from scratch (full curriculum)
  python train.py --data_dir /path/to/data --output_dir ./checkpoints

  # Resume from checkpoint
  python train.py --data_dir /path/to/data --output_dir ./checkpoints --resume ./checkpoints/checkpoint_step1000

  # Train specific phase only
  python train.py --data_dir /path/to/data --output_dir ./checkpoints --start_phase 3
"""

import argparse
import sys
from pathlib import Path

import torch

from model.config import MedBharatConfig
from model.tokenizer import build_tokenizer, load_tokenizer
from training.trainer import MedBharatTrainer


def train_tokenizer_if_needed(data_dir: str, output_dir: str) -> str:
    """Train tokenizer if it doesn't exist yet."""
    tok_path = Path(output_dir) / "medbharat_tokenizer.json"
    if tok_path.exists():
        print(f"Tokenizer found at {tok_path}")
        return str(tok_path)

    print("Training tokenizer...")
    corpus_files = []
    data_path = Path(data_dir)
    for ext in ["*.jsonl.gz", "*_corpus.txt.gz", "*.jsonl", "*.txt.gz"]:
        corpus_files.extend(str(f) for f in data_path.rglob(ext))

    if not corpus_files:
        print(f"ERROR: No data files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(corpus_files)} corpus files")
    build_tokenizer(
        corpus_files=corpus_files,
        vocab_size=48_000,
        output_path=str(tok_path),
    )
    return str(tok_path)


def main():
    parser = argparse.ArgumentParser(description="Train MedBharat-1")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--start_phase", type=int, default=None, help="Start from specific phase (1-6)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--eval_only", action="store_true", help="Run benchmark only")
    args = parser.parse_args()

    # Config
    config = MedBharatConfig()
    print(f"MedBharat-1 Configuration:")
    print(f"  {config.total_params_estimate}")
    print(f"  Max seq len: {config.max_seq_len}")
    print(f"  Experts: {config.n_experts} ({', '.join(config.expert_names)})")
    print(f"  Top-K: {config.top_k_experts}")
    if torch.cuda.is_available():
        device_name = "cuda"
        print(f"  Device: cuda ({torch.cuda.get_device_name()})")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "mps"
        print(f"  Device: MPS (Apple Silicon)")
    else:
        device_name = "cpu"
        print(f"  Device: CPU")

    # Tokenizer
    tok_path = train_tokenizer_if_needed(args.data_dir, args.output_dir)
    tokenizer = load_tokenizer(tok_path)
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    if args.eval_only:
        # Run benchmark only
        from model.transformer import MedBharatModel
        from eval.benchmark import MedBharatBenchmark

        device = device_name
        model = MedBharatModel(config).to(device)

        if args.resume:
            model.load_state_dict(
                torch.load(Path(args.resume) / "model.pt", map_location=device, weights_only=True)
            )

        benchmark = MedBharatBenchmark(model, tokenizer, device)
        results = benchmark.evaluate()
        benchmark.print_results(results)
        benchmark.save_results(results, str(Path(args.output_dir) / "benchmark_results.json"))
        return

    # Trainer
    trainer = MedBharatTrainer(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        peak_lr=args.lr,
        use_wandb=args.wandb,
        resume_from=args.resume,
    )

    # Train
    trainer.train(start_phase=args.start_phase)

    # Run benchmark after training
    print("\nRunning MedBharat Benchmark...")
    from eval.benchmark import MedBharatBenchmark
    benchmark = MedBharatBenchmark(
        trainer.model, tokenizer, trainer.device
    )
    results = benchmark.evaluate()
    benchmark.print_results(results)
    benchmark.save_results(results, str(Path(args.output_dir) / "benchmark_results.json"))


if __name__ == "__main__":
    main()
