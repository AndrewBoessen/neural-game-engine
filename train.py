from utils.train_utils import load_config, TransformerTrainer
from data.gameplay_dataset_reader import GameFrameDataset
from models.tokenizer import Encoder, Decoder, Tokenizer, EncoderDecoderConfig
from models.st_transformer import SpatioTemporalTransformer


def main():
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
            tokenizer_config["encoder"]["config"]["attn_resolutions"]),
        out_ch=tokenizer_config["encoder"]["config"]["out_ch"],
        dropout=tokenizer_config["encoder"]["config"]["dropout"],
    )

    # Initialize datasets
    train_dataset = GameFrameDataset(
        shard_dir="gameplay_data/train/", sequence_length=transformer_config["context_length"], preload_shards=True
    )
    val_dataset = GameFrameDataset(
        shard_dir="gameplay_data/val/", sequence_length=transformer_config["context_length"], preload_shards=True
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
        with_lpips=True,
    )

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
        ffn_drop=transformer_config["ffn_drop"]
    )

    trainer = TransformerTrainer(
        train_config, engine, tokenizer, train_dataset, val_dataset)

    trainer.train()


if __name__ == "__main__":
    main()
