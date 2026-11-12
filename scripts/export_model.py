"""Export MedBharat-1 model for OCI deployment.

Converts checkpoint to optimized format for inference:
  - Strips optimizer state (saves ~60% disk space)
  - Optional: converts to float16 (halves model size)
  - Copies tokenizer alongside weights
"""

import argparse
import shutil
from pathlib import Path

import torch

from model.config import MedBharatConfig
from model.transformer import MedBharatModel


def export(checkpoint_dir: str, output_dir: str, fp16: bool = True):
    checkpoint = Path(checkpoint_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Exporting from {checkpoint} -> {output}")

    # Load model
    config = MedBharatConfig()
    model = MedBharatModel(config)
    state_dict = torch.load(
        checkpoint / "model.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)

    # Convert to fp16 if requested
    if fp16:
        model = model.half()
        print("Converted to float16")

    # Save only model weights (no optimizer, no scaler)
    torch.save(model.state_dict(), output / "model.pt")

    # Copy tokenizer if exists
    for tok_name in ["medbharat_tokenizer.json", "tokenizer.json"]:
        tok_path = checkpoint / tok_name
        if tok_path.exists():
            shutil.copy2(tok_path, output / "medbharat_tokenizer.json")
            print(f"Copied tokenizer: {tok_name}")
            break
    else:
        # Check parent directory
        for parent_tok in checkpoint.parent.glob("*tokenizer*.json"):
            shutil.copy2(parent_tok, output / "medbharat_tokenizer.json")
            print(f"Copied tokenizer from parent: {parent_tok.name}")
            break

    # Save config
    import json
    with open(output / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    # Print sizes
    model_size = (output / "model.pt").stat().st_size / 1e6
    print(f"\nExported model: {model_size:.0f} MB")
    print(f"Output directory: {output}")
    print(f"\nReady for OCI deployment:")
    print(f"  docker build -t medbharat-1 -f deploy/Dockerfile .")
    print(f"  docker run -v {output.absolute()}:/app/weights -p 8000:8000 medbharat-1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MedBharat-1 for deployment")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--output", default="./weights", help="Output directory")
    parser.add_argument("--fp32", action="store_true", help="Keep fp32 (default: convert to fp16)")
    args = parser.parse_args()
    export(args.checkpoint, args.output, fp16=not args.fp32)
