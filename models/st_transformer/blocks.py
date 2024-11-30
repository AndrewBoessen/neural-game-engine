"""
ST-Transformer Blocks
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import einops
from typing import Tuple


class LayerNorm(nn.Module):
    """
    Layer Normalization with optional learnable affine transformation
    """

    def __init__(
        self,
        normalized_shape: int | Tuple,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        """
        Initialize LayerNorm parameters
        :param normalized_shape dimension to normalize over
        :param eps small constant for numerical stability
        :param elementwise_affine whether to use learnable scale and shift
        """
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor):
        """
        Apply Layer Normalization
        :param x input tensor
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Affine transformation if enabled
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU()
    ):
        """
        Initialize FFN parameters
        :param dim input and output dimension
        :param hidden_dim hidden layer dimension (defaults to 4*dim if None)
        :param dropout dropout probability
        :param activation activation function
        """
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        """
        Apply Feed-Forward Network
        :param x input tensor
        """
        return self.net(x)


class Attention(nn.Module):
    """
    Multihead Self Attention
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qk_scale: float = None,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            mask: bool = False
    ):
        """
        Initialize attention params and weights

        :param dim  model dimension
        :param num_heads number of heads
        :param qk_scale constant to scale by
        :param attn_drop dropout ratio
        :param proj_drop output dropout ratio
        :param mask use masked attention
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # dimension for single head
        self.scale = qk_scale or head_dim ** -0.5  # scale factor for qk products
        self.mask = mask

        # qkv embedding projection
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)  # output projection
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        """
        Apply MHSA to input

        :param x input
        """
        qkv = self.qkv(x)  # QKV projection B, L, 3 * E
        # Split QKV embeddings into heads
        q, k, v = einops.rearrange(
            qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
        # use flash attention
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            # MHSA with optional mask and dropout
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop, is_causal=self.mask, scale=self.scale)

        # concatonate heads
        x = einops.rearrange(x, 'B H L D -> B L (H D)')  # B L E
        x = self.proj(x)  # output projection
        x = self.proj_drop(x)  # output dropout
        return x
