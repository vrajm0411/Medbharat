"""MedBharat-1 Training Loop.

Features:
  - Mixed precision (fp16) for P100 GPU
  - Gradient accumulation for effective larger batch
  - Gradient checkpointing to fit in 16GB VRAM
  - Curriculum phase management
  - Checkpoint save/resume (critical for Kaggle 30h sessions)
  - Wandb logging (optional)
"""

import os
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from model.config import MedBharatConfig
from model.transformer import MedBharatModel
from training.dataset import MedBharatDataset, CurriculumDataManager


class MedBharatTrainer:
    """Handles the full training loop with curriculum phases."""

    def __init__(
        self,
        config: MedBharatConfig,
        data_dir: str,
        output_dir: str,
        tokenizer,
        # Training hyperparameters
        batch_size: int = 8,
        grad_accum_steps: int = 4,
        peak_lr: float = 3e-4,
        min_lr: float = 1e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
        # Logging
        log_every: int = 10,
        save_every: int = 500,
        use_wandb: bool = False,
        # Resume
        resume_from: str = None,
    ):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.effective_batch = batch_size * grad_accum_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.log_every = log_every
        self.save_every = save_every
        self.use_wandb = use_wandb

        # Device (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.use_amp = self.device.type == "cuda"  # AMP only on CUDA
        print(f"Using device: {self.device}")

        # Model
        self.model = MedBharatModel(config).to(self.device)
        self.model.enable_gradient_checkpointing()

        # Print param counts
        param_counts = self.model.count_parameters()
        print("Model Parameters:")
        for k, v in param_counts.items():
            print(f"  {k}: {v / 1e6:.1f}M")

        # Optimizer (AdamW with weight decay)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=peak_lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )

        # Mixed precision scaler (only for CUDA)
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Curriculum manager
        self.curriculum = CurriculumDataManager(data_dir)

        # Training state
        self.global_step = 0
        self.current_phase = 1
        self.phase_step = 0
        self.tokens_seen = 0
        self.best_loss = float("inf")
        self.total_steps = 50_000  # default estimate, updated in train()

        # Wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project="medbharat-1", config=vars(config))
            except ImportError:
                self.use_wandb = False

        # Resume from checkpoint
        if resume_from:
            self._load_checkpoint(resume_from)

    def _get_lr(self) -> float:
        """Cosine learning rate schedule with warmup."""
        if self.global_step < self.warmup_steps:
            return self.peak_lr * self.global_step / max(1, self.warmup_steps)

        # Cosine decay
        progress = (self.global_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, progress)
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def _set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _save_checkpoint(self, name: str = None):
        """Save model checkpoint + training state. Keeps only last 2 checkpoints."""
        import shutil

        name = name or f"checkpoint_step{self.global_step}"
        ckpt_dir = self.output_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights only (skip optimizer to save disk space)
        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")

        # Save training state
        state = {
            "global_step": self.global_step,
            "current_phase": self.current_phase,
            "phase_step": self.phase_step,
            "tokens_seen": self.tokens_seen,
            "best_loss": self.best_loss,
            "config": vars(self.config) if hasattr(self.config, '__dict__') else {},
        }
        with open(ckpt_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)

        print(f"Checkpoint saved: {ckpt_dir}")

        # Clean old checkpoints — keep only last 2 + best_model
        existing = sorted([
            d for d in self.output_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint_step")
        ], key=lambda d: d.name)
        while len(existing) > 2:
            old = existing.pop(0)
            shutil.rmtree(old)
            print(f"  Removed old checkpoint: {old.name}")

    def _load_checkpoint(self, ckpt_path: str):
        """Resume from a checkpoint."""
        ckpt_dir = Path(ckpt_path)
        print(f"Loading checkpoint from {ckpt_dir}")

        # Load model
        model_path = ckpt_dir / "model.pt"
        if model_path.exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )

        # Load optimizer
        opt_path = ckpt_dir / "optimizer.pt"
        if opt_path.exists():
            self.optimizer.load_state_dict(
                torch.load(opt_path, map_location=self.device, weights_only=True)
            )

        # Load scaler
        scaler_path = ckpt_dir / "scaler.pt"
        if scaler_path.exists():
            self.scaler.load_state_dict(
                torch.load(scaler_path, map_location=self.device, weights_only=True)
            )

        # Load training state
        state_path = ckpt_dir / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.global_step = state.get("global_step", 0)
            self.current_phase = state.get("current_phase", 1)
            self.phase_step = state.get("phase_step", 0)
            self.tokens_seen = state.get("tokens_seen", 0)
            self.best_loss = state.get("best_loss", float("inf"))

        print(f"Resumed: step={self.global_step}, phase={self.current_phase}")

    @torch.no_grad()
    def validate(self, data_files, max_batches: int = 50) -> dict:
        """Run validation and return loss + perplexity."""
        self.model.eval()
        dataset = MedBharatDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.max_seq_len,
            shuffle_files=False,
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=0, pin_memory=(self.device.type == "cuda"),
        )

        total_loss = 0.0
        total_tokens = 0
        n_batches = 0

        for batch in dataloader:
            if n_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            language_ids = batch["language_ids"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids, language_ids=language_ids, labels=labels,
                )

            loss_val = outputs["loss"].item()
            batch_tokens = (labels != self.config.pad_token_id).sum().item()
            total_loss += loss_val * batch_tokens
            total_tokens += batch_tokens
            n_batches += 1

            del input_ids, labels, language_ids, outputs

        self.model.train()

        if total_tokens == 0:
            return {"val_loss": float("inf"), "val_perplexity": float("inf")}

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 20))  # clamp to avoid overflow
        return {"val_loss": avg_loss, "val_perplexity": perplexity}

    def train_phase(self, phase: int):
        """Train one curriculum phase."""
        phase_config = self.curriculum.get_phase_config(phase)
        data_files = self.curriculum.get_phase_files(phase)

        if not data_files:
            print(f"Phase {phase} ({phase_config['name']}): No data files found, skipping")
            return

        print(f"\n{'='*60}")
        print(f"PHASE {phase}: {phase_config['name']}")
        print(f"Data files: {len(data_files)}")
        print(f"Epochs: {phase_config['epochs']}")
        print(f"{'='*60}\n")

        self.current_phase = phase
        self.phase_step = 0

        for epoch in range(phase_config["epochs"]):
            dataset = MedBharatDataset(
                data_files=data_files,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                shuffle_files=True,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=0,  # streaming dataset, no multiprocessing
                pin_memory=(self.device.type == "cuda"),
            )

            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                language_ids = batch["language_ids"].to(self.device)

                # Forward with mixed precision
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=input_ids,
                        language_ids=language_ids,
                        labels=labels,
                    )
                    loss = outputs["total_loss"] / self.grad_accum_steps

                # Backward
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                    # Update learning rate
                    lr = self._get_lr()
                    self._set_lr(lr)

                    # Step
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    self.phase_step += 1
                    self.tokens_seen += self.effective_batch * self.config.max_seq_len

                    actual_loss = loss.item() * self.grad_accum_steps
                    epoch_loss += actual_loss
                    epoch_steps += 1

                    # Logging
                    if self.global_step % self.log_every == 0:
                        avg_loss = epoch_loss / max(1, epoch_steps)
                        tokens_m = self.tokens_seen / 1e6
                        print(
                            f"Phase {phase} | Epoch {epoch+1} | "
                            f"Step {self.global_step} | "
                            f"Loss: {actual_loss:.4f} | "
                            f"Avg: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Tokens: {tokens_m:.1f}M | "
                            f"AuxLoss: {outputs['aux_loss'].item():.4f}"
                        )

                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "loss": actual_loss,
                                "avg_loss": avg_loss,
                                "lr": lr,
                                "phase": phase,
                                "tokens_seen": self.tokens_seen,
                                "aux_loss": outputs["aux_loss"].item(),
                            }, step=self.global_step)

                    # Checkpoint + validation
                    if self.global_step % self.save_every == 0:
                        # Run validation
                        val_metrics = self.validate(data_files, max_batches=30)
                        print(
                            f"  Validation | Loss: {val_metrics['val_loss']:.4f} | "
                            f"Perplexity: {val_metrics['val_perplexity']:.2f}"
                        )
                        if self.use_wandb:
                            import wandb
                            wandb.log(val_metrics, step=self.global_step)

                        self._save_checkpoint()
                        if val_metrics["val_loss"] < self.best_loss:
                            self.best_loss = val_metrics["val_loss"]
                            self._save_checkpoint("best_model")

                # Memory cleanup
                del input_ids, labels, language_ids, outputs, loss
                if batch_idx % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            print(f"Phase {phase} Epoch {epoch+1} complete | Avg Loss: {epoch_loss / max(1, epoch_steps):.4f}")

        # Save phase checkpoint
        self._save_checkpoint(f"phase{phase}_complete")

    def train(self, start_phase: int = None):
        """Run full curriculum training."""
        start = start_phase or self.current_phase

        # Estimate total steps for LR schedule
        self.total_steps = self.global_step + 50_000  # rough estimate

        print(f"Starting MedBharat-1 curriculum training from phase {start}")
        print(f"Config: {self.config.total_params_estimate}")
        print(f"Effective batch size: {self.effective_batch}")
        print(f"Device: {self.device}")

        for phase in range(start, 7):
            self.train_phase(phase)

        # Final save
        self._save_checkpoint("final_model")
        print("\nTraining complete!")
        print(f"Total steps: {self.global_step}")
        print(f"Total tokens: {self.tokens_seen / 1e9:.2f}B")
        print(f"Best loss: {self.best_loss:.4f}")
