import argparse

import cv2
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from data.token_dataset_reader import TokenDataset
from models.maskgit import gen_image
from models.st_transformer import SpatioTemporalTransformer
from models.tokenizer import Decoder, Encoder, EncoderDecoderConfig, Tokenizer
from utils.train_utils import load_config


def tensor_to_video(tensor, output_path="output.mp4", fps=15):
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Video Generation Inference Script")
    parser.add_argument(
        "--gen_iterations",
        type=int,
        default=8,
        help="Number of iterations in gen_image function (default: 8)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Temperature parameter for gen_image (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output video filename (default: output.mp4)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load configs
    tokenizer_config = load_config("config/tokenizer/config.yaml")
    transformer_config = load_config("config/engine/config.yaml")
    train_config = load_config("config/train_config.yaml")

    mask_token = transformer_config["mask_token"]
    tokens_per_image = transformer_config["tokens_per_image"]

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
    ).to(device)

    engine = SpatioTemporalTransformer(
        dim=transformer_config["dim"],
        num_heads=transformer_config["num_heads"],
        num_layers=transformer_config["num_layers"],
        context_length=transformer_config["context_length"],
        tokens_per_image=transformer_config["tokens_per_image"],
        vocab_size=transformer_config["vocab_size"],
        num_actions=transformer_config["num_actions"],
        mask_token=transformer_config["mask_token"],
        attn_drop=transformer_config["attn_drop"],
        proj_drop=transformer_config["proj_drop"],
        ffn_drop=transformer_config["ffn_drop"],
    ).to(device)

    engine_checkpoint = torch.load(
        "checkpoint_epoch_77.pt", map_location=device, weights_only=True
    )
    engine.load_state_dict(engine_checkpoint["model_state_dict"], strict=True)

    engine.eval()
    tokenizer.eval()

    checkpoint = torch.load(
        train_config["tokenizer_checkpoint"], map_location=device, weights_only=True
    )
    tokenizer.load_state_dict(checkpoint["model_state_dict"], strict=False)

    frames_to_gen = 40
    sample_start = 420
    tokens, actions = val_dataset[sample_start]
    next_actions = []
    for i in range(frames_to_gen // 20):
        _, new_actions = val_dataset[sample_start + 1 + i]
        next_actions.append(new_actions)
    next_actions = torch.stack(next_actions, dim=0).reshape(-1)
    next_actions += engine.vocab_size + 1
    context = tokens

    mask = torch.ones(tokens_per_image) * mask_token

    actions += engine.vocab_size + 1

    gen_frames = []
    for i in tqdm(range(frames_to_gen)):
        tokens[-1] = mask  # mask last image
        actions[-1] = next_actions[i]

        sequence = torch.cat([actions.unsqueeze(-1), tokens], dim=-1)

        sequence = rearrange(sequence.unsqueeze(0), "b t l -> b (t l)").to(device)

        with torch.no_grad():
            image = gen_image(
                sequence,
                engine,
                args.gen_iterations,
                tokens_per_image,
                mask_token,
                temperature=args.temperature,
            )

        tokens[-1] = image[:, -tokens_per_image:].squeeze(0)
        gen_frames.append(tokens[-1])

        # shift all tokens to left one spot
        tokens = torch.roll(tokens, shifts=-1, dims=0)
        actions = torch.roll(actions, shifts=-1, dims=0)

    gen_frames = torch.stack(gen_frames)
    results = torch.cat([context, gen_frames])

    embeddings = tokenizer.embedding(results.to(device))
    embeddings = rearrange(embeddings, "b (h w) e -> b e h w", h=16, w=16)

    decoded_images = []
    for i in range(len(embeddings)):
        image = tokenizer.decode(embeddings[i], should_postprocess=True)
        decoded_images.append(image.cpu().detach())
    tensor_to_video(torch.stack(decoded_images), output_path=args.output)


if __name__ == "__main__":
    main()
