import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.gameplay_dataset_reader import GameFrameDataset
from models.st_transformer import (MaskedCrossEntropyLoss,
                                   SpatioTemporalTransformer)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and process the YAML config file"""
    with open(config_path, "r") as f:
        # Load YAML with OmegaConf to handle the ${} references
        config = OmegaConf.load(f)

    # Convert to dictionary
    config = OmegaConf.to_container(config, resolve=True)

    return config


class TransformerTrainer:
    def __init__(
        self,
        config: Dict[str, Any],
        model: SpatioTemporalTransformer,
        train_dataset: GameFrameDataset,
        val_dataset: GameFrameDataset,
        device: str = "cuda",
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.global_step = 0

        # Initialize dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["optimizer"]["learning_rate"],
            betas=(
                config["optimizer"]["betas"]["beta1"],
                config["optimizer"]["betas"]["beta2"],
            ),
        )

        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["num_epochs"],
            eta_min=config["optimizer"]["min_lr"],
        )

        # Initialize logging
        self.setup_logging()

    def setup_logging(self):
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join("experiments", f"game_engine_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.exp_dir, "training.log")),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=self.exp_dir)

    def save_checkpoint(self, epoch: int):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        path = os.path.join(self.exp_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        return checkpoint["epoch"]

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        """Log metrics to both tensorboard and logging file"""
        for name, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{name}", value, step)

        metrics_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"{prefix} step {step}: {metrics_str}")

    def train_epoch(self, epoch: int):
        self.model.train()
        epoch_losses = []

        with tqdm(self.train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, (tokens, actions) in enumerate(pbar):
                # Init empty mask
                mask = torch.zeros((tokens.shape[0], tokens.shape[-1]))

                # Tokens values for last image
                labels = tokens[:, -1, :].to(self.device)

                # Use ciriculum learning for mask ratio
                mask_ratio = max(
                    random.random()
                    * np.sin(
                        min(
                            self.global_step
                            / self.config["training"]["ciriculum_warmup_steps"],
                            1.0,
                        )
                        * np.pi
                        / 2
                    ),
                    self.config["training"]["min_train_mask_ratio"],
                )
                for i in range(len(tokens)):
                    for j in range(tokens.shape[-1]):
                        # replace with mask token
                        if random.random() <= mask_ratio:
                            tokens[i, -1, j] = self.model.mask_token
                            mask[i, j] = 1.0

                mask = mask.to(self.device)

                actions += self.model.vocab_size + 1

                assert tokens.size(1) == actions.size(
                    1
                ), "images and actions do not align"

                # Concat image tokens and actions
                sequence = torch.cat([actions.unsqueeze(-1), tokens], dim=-1).to(
                    self.device
                )

                # concat all timesteps
                sequence = rearrange(sequence, "B T L -> B (T L)")

                # Forward pass
                logits = self.model(sequence)

                # CSE loss
                loss = MaskedCrossEntropyLoss(logits, labels, mask)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                metrics = {"loss": loss.item()}

                epoch_losses.append(loss.item())

                if (batch_idx + 1) % self.config["training"]["log_interval"] == 0:
                    self.log_metrics(metrics, self.global_step)

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                self.global_step += 1

        return sum(epoch_losses) / len(epoch_losses)

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        val_losses = []

        for tokens, actions in tqdm(self.val_loader, desc="Validation"):
            # Init empty mask
            mask = torch.zeros((tokens.shape[0], tokens.shape[-1]))

            # Tokens values for last image
            labels = tokens[:, -1, :].to(self.device)

            # Use ciriculum learning for mask ratio
            mask_ratio = max(
                random.random()
                * np.sin(
                    min(
                        self.global_step
                        / self.config["training"]["ciriculum_warmup_steps"],
                        1.0,
                    )
                    * np.pi
                    / 2
                ),
                self.config["training"]["min_train_mask_ratio"],
            )
            for i in range(len(tokens)):
                for j in range(tokens.shape[-1]):
                    # replace with mask token
                    if random.random() <= mask_ratio:
                        tokens[i, -1, j] = self.model.mask_token
                        mask[i, j] = 1.0

            mask = mask.to(self.device)

            actions += self.model.vocab_size + 1

            assert tokens.size(1) == actions.size(1), "images and actions do not align"

            # Concat image tokens and actions
            sequence = torch.cat([actions.unsqueeze(-1), tokens], dim=-1).to(
                self.device
            )

            # concat all timesteps
            sequence = rearrange(sequence, "B T L -> B (T L)")

            # Forward pass
            logits = self.model(sequence)

            # CSE loss
            loss = MaskedCrossEntropyLoss(logits, labels, mask)
            val_losses.append(loss.item())

        avg_loss = sum(val_losses) / len(val_losses)
        metrics = {"val_loss": avg_loss}
        self.log_metrics(metrics, self.global_step, prefix="val")

        return avg_loss

    def train(self):
        self.logger.info("Starting training...")
        self.logger.info(f"Config: {self.config}")

        best_val_loss = float("inf")

        for epoch in range(self.config["training"]["num_epochs"]):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            self.scheduler.step()

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch)

            self.logger.info(
                f"Epoch {epoch} - Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
            )
