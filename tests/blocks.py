import torch
import pytest
import numpy as np

# Import the previously defined blocks
from models.st_transformer import LayerNorm, FeedForward, Attention, PredictionHead, SpatialBlock


class TestSpatialBlock:
    @pytest.fixture
    def spatial_block(self):
        """
        Fixture to create a standard SpatialBlock for testing
        """
        return SpatialBlock(
            dim=64,           # model dimension
            tokens_per_image=10,  # 10 tokens per image
            num_heads=4,      # 4 attention heads
            attn_drop=0.1,    # 10% attention dropout
            proj_drop=0.1     # 10% projection dropout
        )

    def test_initialization(self, spatial_block):
        """
        Test that the SpatialBlock is initialized correctly
        """
        assert isinstance(spatial_block, SpatialBlock)
        assert spatial_block.dim == 64
        assert spatial_block.num_tokens == 11  # tokens_per_image + 1
        assert spatial_block.num_heads == 4

    def test_forward_shape_preservation(self, spatial_block):
        """
        Test that the forward pass preserves input tensor shape
        """
        # Create a sample input tensor
        # Batch size of 2, total tokens of 22 (11 tokens * 2), dimension of 64
        x = torch.randn(2, 22, 64)

        # Pass through SpatialBlock
        output = spatial_block(x)

        # Check output shape matches input shape
        assert output.shape == x.shape

    def test_residual_connection(self, spatial_block):
        """
        Test that the forward pass includes a residual connection
        """
        # Create a sample input tensor
        x = torch.randn(2, 22, 64)

        # Pass through SpatialBlock
        output = spatial_block(x)

        # Check that output is not exactly the same as the input
        # but close enough to indicate a residual connection
        assert not torch.allclose(output, x, atol=1e-7)

    def test_no_grad_modification(self, spatial_block):
        """
        Test that the forward pass does not modify input gradient requirement
        """
        # Create a sample input tensor that requires gradient
        x = torch.randn(2, 22, 64, requires_grad=True)

        # Pass through SpatialBlock
        output = spatial_block(x)

        # Perform a dummy loss and backpropagation
        loss = output.sum()
        loss.backward()

        # Ensure input still requires gradient
        assert x.requires_grad

    def test_different_input_sizes(self, spatial_block):
        """
        Test the block works with different input sizes
        """
        test_sizes = [
            (1, 11, 64),    # Single batch, standard number of tokens
            (4, 44, 64),    # Multiple batches, more tokens
            (3, 33, 64)     # Different batch and token combination
        ]

        for batch, total_tokens, dim in test_sizes:
            x = torch.randn(batch, total_tokens, dim)
            output = spatial_block(x)

            assert output.shape == x.shape, f"Failed for shape {x.shape}"


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


class TestPredictionHead:
    def test_initialization(self):
        """Test PredictionHead initialization"""
        # Without softmax
        pred_head1 = PredictionHead(
            dim=512, vocab_size=10000, apply_softmax=False)
        assert pred_head1.projection.in_features == 512
        assert pred_head1.projection.out_features == 10000
        assert pred_head1.apply_softmax is False

        # With softmax
        pred_head2 = PredictionHead(
            dim=512, vocab_size=10000, apply_softmax=True)
        assert pred_head2.apply_softmax is True

    def test_forward_without_softmax(self):
        """Test forward pass without softmax"""
        pred_head = PredictionHead(
            dim=512, vocab_size=10000, apply_softmax=False)

        # Create random input
        x = torch.randn(2, 10, 512)
        output = pred_head(x)

        # Check output shape
        assert output.shape == (2, 10, 10000)

        # Check output is raw logits (not probabilities)
        assert not torch.all(output >= 0) and not torch.all(output <= 1)

    def test_forward_with_softmax(self):
        """Test forward pass with softmax"""
        pred_head = PredictionHead(
            dim=512, vocab_size=10000, apply_softmax=True)

        # Create random input
        x = torch.randn(2, 10, 512)
        output = pred_head(x)

        # Check output shape
        assert output.shape == (2, 10, 10000)

        # Check softmax properties
        # 1. All values should be between 0 and 1
        assert torch.all(output >= 0) and torch.all(output <= 1)

        # 2. Probabilities along last dimension should sum to 1
        prob_sums = output.sum(dim=-1)
        np.testing.assert_almost_equal(
            prob_sums.detach().numpy(),
            np.ones((2, 10)),
            decimal=6
        )

    def test_softmax_temperature(self):
        """Test temperature scaling in softmax"""
        pred_head = PredictionHead(
            dim=512, vocab_size=10000, apply_softmax=True)

        # Create random input
        x = torch.randn(2, 10, 512)

        # Different temperature values
        output_default = pred_head(x)
        output_low_temp = pred_head(x, temperature=0.1)
        output_high_temp = pred_head(x, temperature=10.0)

        # Check shape consistency
        assert output_default.shape == output_low_temp.shape == output_high_temp.shape

        # Low temperature should make distribution more peaked
        # High temperature should make distribution more uniform
        assert torch.max(output_low_temp) > torch.max(output_default)
        assert torch.max(output_high_temp) < torch.max(output_default)


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

# Parametrized tests for different configurations


@pytest.mark.parametrize("dim", [256, 512, 1024])
@pytest.mark.parametrize("vocab_size", [5000, 10000, 20000])
@pytest.mark.parametrize("apply_softmax", [True, False])
def test_prediction_head_configurations(dim, vocab_size, apply_softmax):
    """Test PredictionHead with different configurations"""
    pred_head = PredictionHead(
        dim=dim, vocab_size=vocab_size, apply_softmax=apply_softmax)

    x = torch.randn(2, 10, dim)
    output = pred_head(x)

    assert output.shape == (2, 10, vocab_size)

    if apply_softmax:
        prob_sums = output.sum(dim=-1)
        np.testing.assert_almost_equal(
            prob_sums.detach().numpy(),
            np.ones((2, 10)),
            decimal=6
        )


if __name__ == "__main__":
    pytest.main([__file__])
