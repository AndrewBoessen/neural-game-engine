import torch
from torch import nn

from models.st_transformer import LayerNorm, TemporalBlock, SpatialBlock, FeedForward, PredictionHead


class SpatioTemporalLayer(nn.Module):
    """
    Layer for SpatioTemporal Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        tokens_per_image: int,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        ffn_drop: float = 0.
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
            FeedForward(self.dim, dropout=self.ffn_drop)
        )

    def forward(self, x: torch.Tensor):
        """
        Apply SpatioTemporal Layer

        :param x: input sequence
        """
        return self.layer(x)
