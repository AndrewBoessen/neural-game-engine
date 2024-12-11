import torch
import torch.nn.functional as F
from torch import nn

from models.st_transformer import (FeedForward, LayerNorm, PredictionHead,
                                   SpatialBlock, TemporalBlock)


class SpatioTemporalLayer(nn.Module):
    """
    Layer for SpatioTemporal Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        tokens_per_image: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
    ):
        """
        Initialize Layer Params

        :param dim: model dimension
        :param num_heads: number of attention heads
        :param tokens_per_image: tokens in single image
        :param attn_drop: attention dropout
        :param proj_drop: output projection dropout
        :param ffn_drop: feed forward net dropout
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.tokens = tokens_per_image
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.ffn_drop = ffn_drop

        self.layer = nn.Sequential(
            SpatialBlock(
                self.dim,
                self.tokens,
                self.num_heads,
                attn_drop=self.attn_drop,
                proj_drop=self.proj_drop,
            ),
            TemporalBlock(
                self.dim,
                self.tokens,
                self.num_heads,
                attn_drop=self.attn_drop,
                proj_drop=self.proj_drop,
            ),
            LayerNorm(self.dim),
            FeedForward(self.dim, dropout=self.ffn_drop),
        )

    def forward(self, x: torch.Tensor):
        """
        Apply SpatioTemporal Layer

        :param x: input sequence
        """
        return self.layer(x)


class SpatioTemporalTransformer(nn.Module):
    """
    ST-Transformer Model
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        context_length: int,
        tokens_per_image: int,
        vocab_size: int,
        num_actions: int,
        mask_token: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
    ):
        """
        Initialize Model Params

        :param dim: model dimension
        :param num_heads: number of attention heads
        :param num_layers: number of st-layers
        :param context_length: max number of image, action pairs
        :param tokens_per_image: number of tokens in single image
        :param vocab_size: token vocab size
        :param num_actions: number of actions
        :param mask_token: mask token id in vocab
        :param attn_drop: attention dropout
        :param proj_drop: output projection dropout
        :param ffn_drop: feed forward network dropout
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.context_length = context_length
        self.tokens_per_image = tokens_per_image
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        self.mask_token = mask_token
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.ffn_drop = ffn_drop

        # Token Embeddings
        self.embedding = nn.Embedding(
            self.vocab_size + 1 + self.num_actions, self.dim
        )  # plus one for mask token and add actions to end

        # Positional Embedings
        self.spatial_embedding = nn.Embedding(self.tokens_per_image, self.dim)
        self.temporal_embedding = nn.Embedding(self.context_length, self.dim)

        # ST Layers
        self.layers = nn.ModuleList(
            [
                SpatioTemporalLayer(
                    self.dim,
                    self.num_heads,
                    self.tokens_per_image,
                    self.attn_drop,
                    self.proj_drop,
                    self.ffn_drop,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Prediction MLP Head
        self.prediction_head = PredictionHead(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor):
        """
        Apply ST-Transformer to sequence of tokens

        :param x: input tokens
        """
        # Embed images and actions
        embeddings = self.embedding(x)

        # Tile positional embeddings
        # Create a range tensor for indices

        batch_size, seq_len = x.shape[0], x.shape[1]

        # Calculate image and sequence indices
        image_indices = torch.div(
            torch.arange(seq_len, device=x.device),
            self.tokens_per_image + 1,
            rounding_mode="floor",
        )

        # Create a mask for action tokens (tokens at start of each sequence)
        action_mask = (
            torch.remainder(
                torch.arange(seq_len, device=x.device), self.tokens_per_image + 1
            )
            == 0
        )

        # Temporal embedding for all tokens
        temporal_embeddings = self.temporal_embedding(image_indices)

        # Expand temporal embeddings to match batch size
        temporal_embeddings = temporal_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # Add temporal embeddings to the main embeddings
        embeddings[:, :, : self.dim] = (
            embeddings[:, :, : self.dim] + temporal_embeddings
        )

        # Spatial embedding for non-action tokens
        spatial_mask = ~action_mask
        spatial_indices = (
            torch.remainder(
                torch.arange(seq_len, device=x.device)[spatial_mask],
                self.tokens_per_image + 1,
            )
            - 1
        )

        # Create spatial embeddings for non-action tokens
        if torch.any(spatial_mask):
            spatial_embeddings = self.spatial_embedding(spatial_indices)

        # Expand and add spatial embeddings
        spatial_embeddings = spatial_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        embeddings[:, spatial_mask, : self.dim] = (
            embeddings[:, spatial_mask, : self.dim] + spatial_embeddings
        )

        # Attention Layers
        for layer in self.layers:
            embeddings = layer(embeddings)

        # Tokens for current image in sequence
        tokens_to_predict = embeddings[:, -self.tokens_per_image :, :]
        logits = self.prediction_head(tokens_to_predict)

        # Replace unmasked tokens with one hot encoding
        input_tokens = x[:, -self.tokens_per_image :]

        # Create a mask for tokens that should be predicted (masked tokens)
        mask = input_tokens == self.mask_token

        one_hot = (
            F.one_hot(input_tokens, num_classes=self.vocab_size + 1)[
                :, :, : self.vocab_size
            ]
            * 1e4
        )

        # Replace logits for unmasked tokens with one-hot encoding
        logits = torch.where(mask.unsqueeze(-1), logits, one_hot)

        return logits
