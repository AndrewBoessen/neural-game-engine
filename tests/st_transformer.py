import torch
import pytest

# Import the previously defined blocks
from models.st_transformer import SpatioTemporalLayer, LayerNorm, FeedForward, SpatialBlock, TemporalBlock


class TestSpatioTemporalLayer:
    @pytest.fixture
    def spatiotemporal_layer(self):
        """
        Fixture to create a standard SpatioTemporalLayer for testing
        """
        return SpatioTemporalLayer(
            dim=64,           # model dimension
            num_heads=4,      # 4 attention heads
            tokens_per_image=10,  # 10 tokens per image
            attn_drop=0.1,    # 10% attention dropout
            proj_drop=0.1,    # 10% projection dropout
            ffn_drop=0.1      # 10% feed-forward dropout
        )

    def test_initialization(self, spatiotemporal_layer):
        """
        Test that the SpatioTemporalLayer is initialized correctly
        """
        assert isinstance(spatiotemporal_layer, SpatioTemporalLayer)
        assert spatiotemporal_layer.dim == 64
        assert spatiotemporal_layer.num_heads == 4
        assert spatiotemporal_layer.tokens == 10

    def test_layer_components(self, spatiotemporal_layer):
        """
        Test that the layer contains the correct components
        """
        assert len(spatiotemporal_layer.layer) == 4

        # Check first component is SpatialBlock
        assert isinstance(spatiotemporal_layer.layer[0], SpatialBlock)
        spatial_block = spatiotemporal_layer.layer[0]
        assert spatial_block.dim == 64
        assert spatial_block.num_tokens == 11  # tokens + 1
        assert spatial_block.num_heads == 4

        # Check second component is TemporalBlock
        assert isinstance(spatiotemporal_layer.layer[1], TemporalBlock)
        temporal_block = spatiotemporal_layer.layer[1]
        assert temporal_block.dim == 64
        assert temporal_block.num_tokens == 11  # tokens + 1
        assert temporal_block.num_heads == 4

        # Check third component is LayerNorm
        assert isinstance(spatiotemporal_layer.layer[2], LayerNorm)

        # Check fourth component is FeedForward
        assert isinstance(spatiotemporal_layer.layer[3], FeedForward)

    def test_forward_shape_preservation(self, spatiotemporal_layer):
        """
        Test that the forward pass preserves input tensor shape
        """
        # Create a sample input tensor
        # Batch size of 2, sequence length of 22 (11 tokens * 2), dimension of 64
        x = torch.randn(2, 22, 64)

        # Pass through SpatioTemporalLayer
        output = spatiotemporal_layer(x)

        # Check output shape matches input shape
        assert output.shape == x.shape

    def test_dropout_parameters(self, spatiotemporal_layer):
        """
        Test that dropout parameters are correctly set
        """
        # Check spatial and temporal blocks have correct dropout
        spatial_block = spatiotemporal_layer.layer[0]
        temporal_block = spatiotemporal_layer.layer[1]
        feedforward = spatiotemporal_layer.layer[3]

        assert spatial_block.attn_drop == 0.1
        assert spatial_block.proj_drop == 0.1
        assert temporal_block.attn_drop == 0.1
        assert temporal_block.proj_drop == 0.1
        assert feedforward.net[2].p == 0.1  # First dropout layer
        assert feedforward.net[4].p == 0.1  # Second dropout layer

    def test_different_input_sizes(self, spatiotemporal_layer):
        """
        Test the layer works with different input sizes
        """
        test_sizes = [
            (1, 11, 64),    # Single batch, standard number of tokens
            (4, 44, 64),    # Multiple batches, more tokens
            (3, 33, 64)     # Different batch and token combination
        ]

        for batch, total_tokens, dim in test_sizes:
            x = torch.randn(batch, total_tokens, dim)
            output = spatiotemporal_layer(x)

            assert output.shape == x.shape, f"Failed for shape {x.shape}"

    def test_no_grad_modification(self, spatiotemporal_layer):
        """
        Test that the forward pass does not modify input gradient requirement
        """
        # Create a sample input tensor that requires gradient
        x = torch.randn(2, 22, 64, requires_grad=True)

        # Pass through SpatioTemporalLayer
        output = spatiotemporal_layer(x)

        # Perform a dummy loss and backpropagation
        loss = output.sum()
        loss.backward()

        # Ensure input still requires gradient
        assert x.requires_grad

    def test_output_not_identical_to_input(self, spatiotemporal_layer):
        """
        Test that the layer transforms the input
        """
        # Create a sample input tensor
        x = torch.randn(2, 22, 64)

        # Pass through SpatioTemporalLayer
        output = spatiotemporal_layer(x)

        # Check that output is not exactly the same as the input
        assert not torch.allclose(output, x, atol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__])
