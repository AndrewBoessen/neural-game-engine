import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange

from data.token_dataset_reader import TokenDataset
from models.st_transformer import SpatioTemporalTransformer
from models.tokenizer import Decoder, Encoder, EncoderDecoderConfig, Tokenizer
from utils.train_utils import TransformerTrainer, load_config


def tensor_to_video(tensor, output_path="output.mp4", fps=30):
    """
    Convert a tensor of RGB images to a video.

    Args:
    tensor (torch.Tensor): Input tensor of shape (num_frames, height, width, 3) or (num_frames, 3, height, width)
    output_path (str): Path to save the output video
    fps (int): Frames per second of the output video
    """
    # Ensure tensor is on CPU and convert to numpy
    if isinstance(tensor, torch.Tensor):
        # If tensor is in (C, H, W) format, rearrange to (H, W, C)
        if tensor.dim() == 4 and tensor.shape[1] == 3:
            tensor = tensor.permute(0, 2, 3, 1)

        # Convert to numpy and ensure values are in 0-255 range
        images = tensor.cpu().detach().numpy()

        # Normalize if values are in 0-1 range
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8)
    else:
        images = images.astype(np.uint8)

    # Get video dimensions
    num_frames, height, width, _ = images.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames to video
    for frame in images:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    # Release the video writer
    out.release()

    print(f"Video saved to {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load configs
    tokenizer_config = load_config("config/tokenizer/config.yaml")
    transformer_config = load_config("config/engine/config.yaml")
    train_config = load_config("config/train_config.yaml")

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
    train_dataset = TokenDataset(
        image_tokens_path="token_data/train/tokens.npy",
        action_tokens_path="token_data/train/actions.npy",
        context_length=transformer_config["context_length"],
        stride=1,
    )
    val_dataset = TokenDataset(
        image_tokens_path="token_data/val/tokens.npy",
        action_tokens_path="token_data/val/actions.npy",
        context_length=transformer_config["context_length"],
        stride=1,
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
    )

    checkpoint = torch.load(
        train_config["tokenizer_checkpoint"], map_location=device, weights_only=True
    )
    tokenizer.load_state_dict(checkpoint["model_state_dict"], strict=False)

    tokens, _ = train_dataset[100]
    z_q = rearrange(
        tokenizer.embedding(tokens),
        "b (h w) e -> b e h w",
        b=len(tokens),
        e=512,
        h=16,
        w=16,
    ).contiguous()
    print(z_q.device)

    image = tokenizer.decode(z_q, should_postprocess=True)

    tensor_to_video(image)


if __name__ == "__main__":
    main()
