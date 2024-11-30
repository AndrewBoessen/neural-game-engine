import torch
import pytest
import numpy as np

# Import the previously defined blocks
from models.st_transformer import LayerNorm, FeedForward, Attention


class TestLayerNorm:
    def test_initialization(self):
        """Test LayerNorm initialization with different input shapes"""
        # Single integer initialization
        ln1 = LayerNorm(10)
        assert ln1.normalized_shape == (10,)
        assert ln1.weight is not None
        assert ln1.bias is not None

        # Tuple initialization
        ln2 = LayerNorm((10, 20))
        assert ln2.normalized_shape == (10, 20)

        # Disable elementwise affine
        ln3 = LayerNorm(10, elementwise_affine=False)
        assert ln3.weight is None
        assert ln3.bias is None

    def test_forward_normalization(self):
        """Test LayerNorm normalization properties"""
        ln = LayerNorm(10)

        # Create a test tensor
        x = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0]
        ], dtype=torch.float32)

        # Apply LayerNorm
        x_norm = ln(x)

        # Check mean and variance
        np.testing.assert_almost_equal(
            x_norm.mean(dim=-1).detach().numpy(),
            np.zeros(3),
            decimal=1
        )
        np.testing.assert_almost_equal(
            x_norm.std(dim=-1).detach().numpy(),
            np.ones(3),
            decimal=1
        )

    def test_elementwise_affine(self):
        """Test learnable affine transformation"""
        ln = LayerNorm(3)

        # Set custom weights and biases
        ln.weight.data = torch.tensor([2.0, 2.0, 2.0])
        ln.bias.data = torch.tensor([1.0, 1.0, 1.0])

        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]
        ], dtype=torch.float32)

        x_norm = ln(x)

        # The output should be scaled by weight and shifted by bias
        expected = (x - x.mean(dim=-1, keepdim=True)) / \
            torch.sqrt(x.var(dim=-1, keepdim=True) + ln.eps)
        expected = expected * ln.weight + ln.bias

        torch.testing.assert_close(x_norm, expected)


class TestFeedForward:
    def test_initialization(self):
        """Test FeedForward network initialization"""
        # Default initialization
        ffn1 = FeedForward(dim=10)
        # Linear, Activation, Dropout, Linear, Dropout
        assert len(ffn1.net) == 5

        # Custom hidden dimension
        ffn2 = FeedForward(dim=10, hidden_dim=20)
        assert ffn2.net[0].out_features == 20

    def test_forward(self):
        """Test FeedForward forward pass"""
        ffn = FeedForward(dim=10, hidden_dim=20)

        x = torch.randn(2, 5, 10)
        output = ffn(x)

        # Check output shape
        assert output.shape == x.shape

    def test_dropout(self):
        """Test dropout behavior"""
        ffn = FeedForward(dim=10, dropout=0.5)

        x = torch.ones(100, 10, 10)
        ffn.train()  # Set to training mode
        output = ffn(x)

        # In training mode, some activations should be zero
        assert not torch.allclose(output, x)


class TestAttention:
    def test_initialization(self):
        """Test Attention block initialization"""
        attn = Attention(dim=64, num_heads=8)

        assert attn.num_heads == 8
        assert attn.qkv.out_features == 64 * 3
        assert attn.proj.in_features == 64
        assert attn.proj.out_features == 64

    def test_forward_shape(self):
        """Test attention forward pass shape"""
        attn = Attention(dim=64, num_heads=8)

        # Create random input
        x = torch.randn(2, 10, 64)
        output = attn(x)

        # Check output shape matches input
        assert output.shape == x.shape

    def test_masked_attention(self):
        """Test masked attention"""
        attn = Attention(dim=64, num_heads=8, mask=True)

        x = torch.randn(2, 10, 64)
        output = attn(x)

        # Masked attention should work without errors
        assert output.shape == x.shape


@pytest.mark.parametrize("dim", [32, 64, 128])
@pytest.mark.parametrize("num_heads", [4, 8])
def test_attention_configurations(dim, num_heads):
    """Test Attention with different dimensions and head configurations"""
    attn = Attention(dim=dim, num_heads=num_heads)
    x = torch.randn(2, 10, dim)
    output = attn(x)
    assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__])
