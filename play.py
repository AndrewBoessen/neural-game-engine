"""credit to: https://github.com/eloialonso/diamond/blob/main/src/game/game.py"""

from collections import OrderedDict
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch
from einops import rearrange
from PIL import Image
from torch.amp import autocast
from torchinfo import summary
from tqdm import tqdm

from data.token_dataset_reader import TokenDataset
from models.maskgit import gen_image
from models.st_transformer import SpatioTemporalTransformer
from models.tokenizer import Decoder, Encoder, EncoderDecoderConfig, Tokenizer
from utils.train_utils import load_config


class Game:
    def __init__(
        self,
        tokenizer: Tokenizer,
        engine: SpatioTemporalTransformer,
        states: torch.Tensor,
        actions: torch.Tensor,
        keymap: Dict[Tuple[int], int],
        size: Tuple[int, int],
        fps: int,
        verbose: bool,
        device: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.engine = engine
        self.states = states
        self.actions = actions
        self.original_context = [states, actions]
        self.keymap = keymap
        self.height, self.width = size
        self.fps = fps
        self.verbose = verbose
        self.keymap = OrderedDict()
        self.device = device
        for keys, act in sorted(keymap.items(), key=lambda keys_act: -len(keys_act[0])):
            self.keymap[keys] = act

        print("\nControls (general):\n")
        print("⏎ : reset env")
        print(". : pause/unpause")
        print("e : step-by-step (when paused)")
        print("← : move left")
        print("→ : move right")
        print("\n")

        self.mask = torch.ones(self.engine.tokens_per_image) * self.engine.mask_token
        self.vocab_size = self.engine.vocab_size

    def run(self) -> None:
        pygame.init()

        header_height = 150 if self.verbose else 0
        font_size = 16
        screen = pygame.display.set_mode((self.width, self.height + header_height))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("mono", font_size)
        header_rect = pygame.Rect(0, 0, self.width, header_height)

        def step(action):
            self.states = torch.roll(self.states, shifts=-1, dims=0)
            self.actions = torch.roll(self.actions, shifts=-1, dims=0)

            self.states[-1] = self.mask  # mask last state
            self.actions[-1] = action  # prompt with new action

            sequence = torch.cat(
                [self.actions.unsqueeze(-1) + self.vocab_size + 1, self.states], dim=-1
            ).to(self.device)
            sequence = rearrange(sequence.unsqueeze(0), "b t l -> b (t l)").to(
                self.device
            )
            with torch.no_grad():
                with autocast(str(self.device), enabled=True, dtype=torch.float16):
                    image_tokens = gen_image(
                        sequence,
                        self.engine,
                        8,
                        self.engine.tokens_per_image,
                        self.engine.mask_token,
                        temperature=0.4,
                    )
                    self.states[-1] = image_tokens[
                        :, -self.engine.tokens_per_image :
                    ].squeeze(0)
                    z_q = rearrange(
                        self.tokenizer.embedding(
                            self.states[-1].unsqueeze(0).to(self.device)
                        ),
                        "b (h w) e -> b e h w",
                        b=1,
                        e=512,
                        h=16,
                        w=16,
                    ).contiguous()

                    obs = self.tokenizer.decode(z_q, should_postprocess=True)

                    return obs

        def clear_header():
            pygame.draw.rect(screen, pygame.Color("black"), header_rect)
            pygame.draw.rect(screen, pygame.Color("white"), header_rect, 1)

        def draw_text(text, idx_line, idx_column, num_cols):
            pos = (
                5 + idx_column * int(self.width // num_cols),
                5 + idx_line * font_size,
            )
            assert (0 <= pos[0] <= self.width) and (0 <= pos[1] <= header_height)
            screen.blit(font.render(text, True, pygame.Color("white")), pos)

        def draw_game(obs):
            assert obs.ndim == 4 and obs.size(0) == 1
            img = Image.fromarray(obs[0].mul(255).byte().permute(1, 2, 0).cpu().numpy())
            pygame_image = np.array(
                img.resize((self.width, self.height), resample=Image.NEAREST)
            ).transpose((1, 0, 2))
            surface = pygame.surfarray.make_surface(pygame_image)
            screen.blit(surface, (0, header_height))

        def reset():
            nonlocal obs, info, do_reset, ep_return, ep_length
            self.states, self.actions = self.original_context
            obs = step(0)
            do_reset = False
            ep_return = 0
            ep_length = 0

        obs, info, do_reset, ep_return, ep_length = (None,) * 5

        reset()
        do_wait = False
        should_stop = False

        while not should_stop:
            do_one_step = False
            action = 0  # noop
            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_stop = True

                if event.type != pygame.KEYDOWN:
                    continue

                if event.key == pygame.K_RETURN:
                    do_reset = True

                if event.key == pygame.K_PERIOD:
                    do_wait = not do_wait
                    print("Game paused." if do_wait else "Game resumed.")

                if event.key == pygame.K_e:
                    do_one_step = True

            if action == 0:
                pressed = pygame.key.get_pressed()
                for keys, action in self.keymap.items():
                    if all([pressed[key] for key in keys]):
                        break
                else:
                    action = 0

            if do_reset:
                reset()

            if do_wait and not do_one_step:
                continue

            # get obs
            next_obs = step(action)

            ep_length += 1

            draw_game(obs)

            pygame.display.flip()
            clock.tick(self.fps)

            obs = next_obs


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

    # Load Model Checkpoints
    engine_checkpoint = torch.load(
        "checkpoint_epoch_77.pt", map_location=device, weights_only=True
    )
    engine.load_state_dict(engine_checkpoint["model_state_dict"], strict=True)
    summary(engine)
    checkpoint = torch.load(
        train_config["tokenizer_checkpoint"], map_location=device, weights_only=True
    )
    tokenizer.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # compile models
    tokenizer = torch.compile(tokenizer, backend="aot_eager")
    engine = torch.compile(engine, backend="aot_eager")

    engine.eval()
    tokenizer.eval()

    states, actions = val_dataset[200]

    keymap = {(pygame.K_RIGHT,): 1, (pygame.K_LEFT,): 2}

    game = Game(
        tokenizer,
        engine,
        states,
        actions,
        keymap,
        size=(512, 512),
        fps=15,
        verbose=False,
        device=device,
    )

    game.run()


if __name__ == "__main__":
    main()
