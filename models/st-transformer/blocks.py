"""
ST-Transformer Blocks
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import einops


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
