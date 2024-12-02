import torch
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
        self.image_embedding = nn.Embedding(
            self.vocab_size + 1, self.dim
        )  # plus one for mask token
        self.action_embedding = nn.Embedding(self.num_actions, self.dim)

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
        embeddings = torch.zeros((x.shape[0], x.shape[1], self.dim), device=x.device)
        for i in range(x.shape[1]):
            if i % (self.tokens_per_image + 1) == 0:
                action_tokens = x[:, i]
                embeddings[:, i, : self.dim] = self.action_embedding(
                    action_tokens.to(torch.int32)
                )
                # Only apply temporal positional embedding to action
                embeddings[:, i, : self.dim] += self.temporal_embedding(
                    torch.tensor(i // (self.tokens_per_image + 1), device=x.device)
                )
            else:
                image_tokens = x[:, i]
                embeddings[:, i, : self.dim] = self.image_embedding(
                    image_tokens.to(torch.int32)
                )
                # apply both spatial and temporal positional embeddings to image
                embeddings[:, i, : self.dim] += self.temporal_embedding(
                    torch.tensor(i // (self.tokens_per_image + 1), device=x.device)
                )
                embeddings[:, i, : self.dim] += self.spatial_embedding(
                    torch.tensor(i % (self.tokens_per_image + 1) - 1, device=x.device)
                )

        # Attention Layers
        for layer in self.layers:
            embeddings = layer(embeddings)

        # Tokens for current image in sequence
        tokens_to_predict = embeddings[:, -self.tokens_per_image :, :]
        token_pred = self.prediction_head(tokens_to_predict)

        # Replace unmasked tokens with one hot encoding
        input_tokens = x[:, -self.tokens_per_image :]

        # Create a one-hot encoding for input tokens
        one_hot_tokens = torch.zeros_like(token_pred)
        one_hot_tokens.scatter_(2, input_tokens.unsqueeze(2).to(dtype=torch.int64), 1.0)

        # Create a mask for tokens that should be predicted (masked tokens)
        mask = input_tokens == self.mask_token

        # Replace non-masked tokens with their one-hot encoding
        # Convert one hot encoding to softmax mask (-inf)
        token_pred = torch.where(
            mask.unsqueeze(2),
            token_pred,
            torch.where(
                one_hot_tokens == 1.0,
                one_hot_tokens,
                torch.tensor(-float("inf"), device=token_pred.device),
            ),
        )

        return token_pred
