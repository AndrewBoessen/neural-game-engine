import torch
import pytest

# Import the previously defined blocks
from models.st_transformer import SpatioTemporalTransformer, SpatioTemporalLayer, LayerNorm, FeedForward, SpatialBlock, TemporalBlock, PredictionHead, MaskedCrossEntropyLoss


class TestMaskedCrossEntropyLoss:
    """
    Comprehensive test suite for MaskedCrossEntropyLoss function
    """

    @pytest.fixture
    def sample_inputs(self):
        """
        Fixture to provide common test input tensors
        """
        batch_size, seq_length, vocab_size = 2, 5, 10

        # Random logits
        logits = torch.randn(batch_size, seq_length, vocab_size)

        # Random labels
        labels = torch.randint(0, vocab_size, (batch_size, seq_length))

        # Varied mask for different sequences
        mask = torch.tensor([
            [0, 1, 1, 0, 0],  # First sequence: 2nd and 3rd tokens masked
            [1, 0, 0, 1, 0]   # Second sequence: 1st and 4th tokens masked
        ], dtype=torch.bool)

        return logits, labels, mask

    def test_basic_functionality(self, sample_inputs):
        """
        Test basic functionality of MaskedCrossEntropyLoss
        """
        logits, labels, mask = sample_inputs

        # Calculate loss
        loss = MaskedCrossEntropyLoss(logits, labels, mask)

        # Verify loss is not NaN and is a reasonable value
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_no_masked_tokens(self, sample_inputs):
        """
        Test scenario with no masked tokens
        """
        logits, labels, _ = sample_inputs

        # Mask with all zeros (no masked tokens)
        mask = torch.zeros_like(labels, dtype=torch.bool)

        # Calculate loss
        loss = MaskedCrossEntropyLoss(logits, labels, mask)

        # Loss should be zero when no tokens are masked
        assert loss.item() == 0.0

    def test_all_tokens_masked(self, sample_inputs):
        """
        Test scenario with all tokens masked
        """
        logits, labels, _ = sample_inputs

        # Mask with all ones (all tokens masked)
        mask = torch.ones_like(labels, dtype=torch.bool)

        # Calculate loss
        loss = MaskedCrossEntropyLoss(logits, labels, mask)

        # Verify loss is not NaN and is a reasonable value
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_shape_mismatch(self, sample_inputs):
        """
        Test error handling for tensor shape mismatches
        """
        logits, labels, _ = sample_inputs

        # Create labels with incorrect shape
        incorrect_labels = torch.randint(
            0, logits.size(-1), (labels.size(0), labels.size(1) + 1))

        # Mask with incorrect shape
        mask = torch.ones_like(incorrect_labels, dtype=torch.bool)

        # Should raise a ValueError due to shape mismatch
        with pytest.raises(ValueError):
            MaskedCrossEntropyLoss(logits, incorrect_labels, mask)

    def test_numerical_stability(self):
        """
        Test numerical stability with extreme logit values
        """

        # Create logits with very large values
        logits = torch.tensor([[
            [1e10, -1e10, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1e10, -1e10, 0, 0, 0],
            [0, 0, 1e10, -1e10, 0, 0, 0, 0, 0, 0]
        ]], dtype=torch.float32)

        labels = torch.tensor([[0, 5, 2]])
        mask = torch.tensor([[1, 1, 1]])

        # Calculate loss
        loss = MaskedCrossEntropyLoss(logits, labels, mask)

        # Verify loss is not NaN and is a reasonable value
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_different_masking_patterns(self):
        """
        Test loss calculation with different masking patterns
        """
        batch_size, seq_length, vocab_size = 2, 5, 10

        # Random logits
        logits = torch.randn(batch_size, seq_length, vocab_size)

        # Random labels
        labels = torch.randint(0, vocab_size, (batch_size, seq_length))

        # Varied mask with different patterns
        mask = torch.tensor([
            [0, 1, 1, 0, 0],  # First sequence: 2nd and 3rd tokens masked
            [1, 0, 0, 1, 0]   # Second sequence: 1st and 4th tokens masked
        ], dtype=torch.bool)

        # Calculate loss
        loss = MaskedCrossEntropyLoss(logits, labels, mask)

        # Verify loss is not NaN and is a reasonable value
        assert not torch.isnan(loss)
        assert loss.item() > 0


class TestSpatioTemporalTransformer:
    @pytest.fixture
    def transformer_model(self):
        """
        Fixture to create a standard SpatioTemporalTransformer for testing
        """
        return SpatioTemporalTransformer(
            dim=64,               # model dimension
            num_heads=4,          # 4 attention heads
            num_layers=2,         # 2 transformer layers
            context_length=5,     # 5 context steps
            tokens_per_image=10,  # 10 tokens per image
            vocab_size=1000,      # 1000 vocab tokens
            num_actions=20,       # 20 possible actions
            mask_token=0,         # 0 as mask token
            attn_drop=0.1,        # 10% attention dropout
            proj_drop=0.1,        # 10% projection dropout
            ffn_drop=0.1          # 10% feed-forward dropout
        )

    def test_initialization(self, transformer_model):
        """
        Test that the SpatioTemporalTransformer is initialized correctly
        """
        assert transformer_model.dim == 64
        assert transformer_model.num_heads == 4
        assert transformer_model.num_layers == 2
        assert transformer_model.context_length == 5
        assert transformer_model.tokens_per_image == 10
        assert transformer_model.vocab_size == 1000
        assert transformer_model.num_actions == 20
        assert transformer_model.mask_token == 0

    def test_embedding_layers(self, transformer_model):
        """
        Test embedding layers initialization
        """
        # Image embedding
        assert isinstance(transformer_model.image_embedding,
                          torch.nn.Embedding)
        # vocab_size + 1 for mask
        assert transformer_model.image_embedding.num_embeddings == 1001
        assert transformer_model.image_embedding.embedding_dim == 64

        # Action embedding
        assert isinstance(transformer_model.action_embedding,
                          torch.nn.Embedding)
        assert transformer_model.action_embedding.num_embeddings == 20
        assert transformer_model.action_embedding.embedding_dim == 64

        # Spatial embedding
        assert isinstance(transformer_model.spatial_embedding,
                          torch.nn.Embedding)
        assert transformer_model.spatial_embedding.num_embeddings == 10
        assert transformer_model.spatial_embedding.embedding_dim == 64

        # Temporal embedding
        assert isinstance(
            transformer_model.temporal_embedding, torch.nn.Embedding)
        assert transformer_model.temporal_embedding.num_embeddings == 5
        assert transformer_model.temporal_embedding.embedding_dim == 64

    def test_layers(self, transformer_model):
        """
        Test transformer layers
        """
        assert len(transformer_model.layers) == 2
        for layer in transformer_model.layers:
            assert isinstance(layer, SpatioTemporalLayer)

    def test_prediction_head(self, transformer_model):
        """
        Test prediction head
        """
        assert isinstance(transformer_model.prediction_head, PredictionHead)
        assert transformer_model.prediction_head.projection.in_features == 64
        assert transformer_model.prediction_head.projection.out_features == 1000

    def test_forward_shape(self, transformer_model):
        """
        Test forward pass shape
        """
        # Create input tensor: batch_size x sequence_length
        # Sequence length is (tokens_per_image + 1) * context_length
        batch_size = 4
        # 11 tokens per step * 5 steps
        input_tokens = torch.randint(0, 20, (batch_size, 55))

        # Add some mask tokens
        input_tokens[0, 10] = 0  # mask a token in first batch

        # Forward pass
        output = transformer_model(input_tokens)

        # Check output shape: batch_size x tokens_per_image x vocab_size
        assert output.shape == (batch_size, 10, 1000)

    def test_mask_token_prediction(self, transformer_model):
        """
        Test mask token prediction functionality
        """
        batch_size = 4
        input_tokens = torch.randint(0, 20, (batch_size, 55))

        # Explicitly set some tokens as mask tokens
        mask_positions = torch.randint(1, 10, (batch_size,))
        for i, pos in enumerate(mask_positions):
            input_tokens[i, 11 * 4 + pos] = 0.0  # mask token in last image

        # Forward pass
        output = transformer_model(input_tokens)

        # Verify mask token positions have non-zero logits
        for i, pos in enumerate(mask_positions):
            mask_token_pred = output[i, pos]
            assert not torch.all(mask_token_pred == -float('inf'))

    def test_device_compatibility(self, transformer_model):
        """
        Test model works with different device configurations
        """
        # Test CPU
        input_tokens_cpu = torch.randint(0, 20, (4, 55))
        output_cpu = transformer_model(input_tokens_cpu)
        assert output_cpu.shape == (4, 10, 1000)

        # Test GPU if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            transformer_model_gpu = transformer_model.to(device)
            input_tokens_gpu = input_tokens_cpu.to(device)

            output_gpu = transformer_model_gpu(input_tokens_gpu)
            assert output_gpu.shape == (4, 10, 1000)
            assert output_gpu.device.type == 'cuda'

    def test_no_grad_modification(self, transformer_model):
        """
        Test that the forward pass does not modify input gradient requirement
        """
        input_tokens = torch.randint(0, 20, (4, 55)).to(torch.float32)

        input_tokens.requires_grad = True

        # Forward pass
        output = transformer_model(input_tokens)

        # Dummy loss to test backpropagation
        loss = output.sum()
        loss.backward()

        # Ensure input still requires gradient
        assert input_tokens.requires_grad


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
