"""
Training loop for XRayGPT.

Implements:
  - Mixed precision training (FP16)
  - Gradient accumulation
  - Learning rate warmup + cosine decay
  - Early stopping
  - Checkpoint saving/loading
  - Validation with generation
"""

import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.data.dataset import IUXRayDataset, collate_fn
from src.models.xraygpt import XRayGPT
from src.utils.config import XRayGPTConfig
from src.utils.logging import setup_logger


class Trainer:
    """Handles the full training lifecycle for XRayGPT."""

    def __init__(
        self,
        model: XRayGPT,
        config: XRayGPTConfig,
        train_dataset: IUXRayDataset,
        val_dataset: IUXRayDataset,
        log_file: str | None = None,
    ):
        self.model = model
        self.config = config
        self.train_cfg = config.training
        self.logger = setup_logger("xraygpt.trainer", log_file=log_file)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
        self.model.print_param_summary()

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Optimizer — only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )

        # Learning rate scheduler: linear warmup → cosine decay
        total_steps = (
            len(self.train_loader)
            // self.train_cfg.gradient_accumulation_steps
            * self.train_cfg.num_epochs
        )
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.train_cfg.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - self.train_cfg.warmup_steps, 1),
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.train_cfg.warmup_steps],
        )

        # Mixed precision
        self.use_fp16 = self.train_cfg.fp16 and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_fp16)

        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0

        # Checkpoint directory
        self.ckpt_dir = Path(config.paths.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> dict:
        """Run the full training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"  Epochs: {self.train_cfg.num_epochs}")
        self.logger.info(f"  Batch size: {self.train_cfg.batch_size}")
        self.logger.info(f"  Accumulation steps: {self.train_cfg.gradient_accumulation_steps}")
        self.logger.info(f"  Effective batch size: "
                         f"{self.train_cfg.batch_size * self.train_cfg.gradient_accumulation_steps}")
        self.logger.info(f"  Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"  Validation samples: {len(self.val_loader.dataset)}")

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.train_cfg.num_epochs + 1):
            # Train
            train_loss = self._train_epoch(epoch)
            history["train_loss"].append(train_loss)

            # Validate
            if epoch % self.train_cfg.eval_every == 0:
                val_loss, sample_reports = self._validate(epoch)
                history["val_loss"].append(val_loss)

                # Log sample generations
                if sample_reports:
                    self.logger.info(f"  Sample generation: {sample_reports[0][:200]}...")

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                    self.logger.info(f"  New best validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.train_cfg.patience:
                        self.logger.info(
                            f"Early stopping at epoch {epoch} "
                            f"(patience={self.train_cfg.patience})"
                        )
                        break

            # Save periodic checkpoint
            if epoch % self.train_cfg.save_every == 0:
                self._save_checkpoint(epoch)

        self.logger.info("Training complete!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        return history

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with autocast(device_type=self.device.type, enabled=self.use_fp16):
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = outputs["loss"] / self.train_cfg.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.train_cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.train_cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.train_cfg.gradient_accumulation_steps
            num_batches += 1

            # Logging
            if (batch_idx + 1) % self.config.logging.log_every == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                self.logger.info(
                    f"  Epoch {epoch} | Step {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | Time: {elapsed:.1f}s"
                )

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        self.logger.info(
            f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s"
        )
        return avg_loss

    @torch.no_grad()
    def _validate(self, epoch: int) -> tuple[float, list[str]]:
        """Run validation and generate sample reports."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with autocast(device_type=self.device.type, enabled=self.use_fp16):
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            total_loss += outputs["loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Generate sample reports from first val batch
        sample_reports = []
        first_batch = next(iter(self.val_loader))
        pixel_values = first_batch["pixel_values"][:2].to(self.device)
        sample_reports = self.model.generate(pixel_values)

        self.logger.info(f"Epoch {epoch} | Val Loss: {avg_loss:.4f}")
        return avg_loss, sample_reports

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "config": self.config,
        }

        # Save epoch checkpoint
        path = self.ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"  Saved checkpoint: {path}")

        # Save best model
        if is_best:
            best_path = self.ckpt_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"  Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load a checkpoint and return the epoch number."""
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.global_step = checkpoint["global_step"]

        epoch = checkpoint["epoch"]
        self.logger.info(f"Resumed from epoch {epoch}, best val loss: {self.best_val_loss:.4f}")
        return epoch
