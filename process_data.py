import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.gameplay_dataset_reader import GameFrameDataset
from models.tokenizer import Decoder, Encoder, EncoderDecoderConfig, Tokenizer
from utils.train_utils import load_config


def main():
    device = "cuda"
    # load configs
    tokenizer_config = load_config("config/tokenizer/config.yaml")
    # Create encoder/decoder config from loaded configuration
    encoder_decoder_config = EncoderDecoderConfig(
        resolution=tokenizer_config["encoder"]["config"]["resolution"],
        in_channels=tokenizer_config["encoder"]["config"]["in_channels"],
        z_channels=tokenizer_config["encoder"]["config"]["z_channels"],
        ch=tokenizer_config["encoder"]["config"]["ch"],
        ch_mult=tuple(tokenizer_config["encoder"]["config"]["ch_mult"]),
        num_res_blocks=tokenizer_config["encoder"]["config"]["num_res_blocks"],
        attn_resolutions=tuple(
            tokenizer_config["encoder"]["config"]["attn_resolutions"]
        ),
        out_ch=tokenizer_config["encoder"]["config"]["out_ch"],
        dropout=tokenizer_config["encoder"]["config"]["dropout"],
    )

    # Initialize datasets
    train_dataset = GameFrameDataset(
        shard_dir="gameplay_data/train/", preload_shards=True
    )
    # Initialize dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )
    # Initialize model components using config values
    encoder = Encoder(config=encoder_decoder_config)
    decoder = Decoder(config=encoder_decoder_config)

    # Initialize Tokenizer using config values
    tokenizer = Tokenizer(
        vocab_size=tokenizer_config["vocab_size"],
        embed_dim=tokenizer_config["embed_dim"],
        encoder=encoder,
        decoder=decoder,
        with_lpips=False,
    ).to(device)
    checkpoint = torch.load(
        "checkpoint_epoch_15.pt", map_location=device, weights_only=True
    )
    tokenizer.load_state_dict(checkpoint["model_state_dict"], strict=False)

    tokenizer.eval()

    tokens_list = []
    actions_list = []

    for batch in tqdm(train_loader):
        images = batch["image"].to(device)
        actions = batch["action"]
        with torch.no_grad():
            output = tokenizer.encode(images, should_preprocess=True)

        tokens = output.tokens

        tokens_list.append(tokens.cpu().detach().numpy())
        actions_list.append(actions)
    # Convert tokens_list to a contiguous NumPy array
    tokens_array = np.concatenate(tokens_list, axis=0)
    actions_array = np.concatenate(actions_list, axis=0)

    # Optional: Save the tokens array to a file
    np.save("tokens.npy", tokens_array)
    np.save("actions.npy", actions_array)

    print(f"Tokens array shape: {tokens_array.shape}")
    print(f"Actions shape: {actions_array.shape}")


if __name__ == "__main__":
    main()
