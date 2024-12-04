import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.gameplay_dataset_reader import GameFrameDataset, PreprocessingConfig
from models.st_transformer import SpatioTemporalTransformer
from models.tokenizer import Decoder, Encoder, Tokenizer


# Config for tokenizer encoder and decoder
@dataclass
class EncoderDecoderConfig:
    resolution: int
    in_channels: int
    z_channels: int
    ch: int
    ch_mult: tuple
    num_res_blocks: int
    attn_resolutions: tuple
    out_ch: int
    dropout: float


# Config for ST-Transformer
@dataclass
class TransformerConfig:
    dim: int
    num_heads: int
    num_layers: int
    context_length: int
    tokens_per_image: int
    vocab_size: int
    num_actions: int
    mask_token: int
    attn_drop: float
    proj_drop: float
    ffn_drop: float


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
        tokenizer: Tokenizer,
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
            lr=config["learning_rate"],
            betas=(config["beta1"], config["beta2"]),
        )

        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["num_epochs"], eta_min=config["min_lr"]
        )

        # Initialize logging
        self.setup_logging()

    def setup_logging(self):
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join("experiments", f"vqvae_{timestamp}")
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
