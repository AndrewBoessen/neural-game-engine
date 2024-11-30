import pytest
import numpy as np
import torch

from models.maskgit import get_mask


def test_get_mask_cosine_schedule():
    """Test cosine schedule mask generation"""
    token_confidence = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1]).unsqueeze(0)
    mask = get_mask(
        iteration=5,
        total_iterations=10,
        total_tokens=5,
        token_confidence=token_confidence,
        schedule="cosine"
    )

    assert mask.shape == token_confidence.shape
    assert mask.dtype == torch.float32

    # assert mask and schedule align
    # mask out 3 tokens for iteration 5
    assert sum(mask.squeeze(0)) == np.floor(np.cos(np.pi / 4) *
                                            len(token_confidence.squeeze(0)))


def test_get_mask_linear_schedule():
    """Test linear schedule mask generation"""
    token_confidence = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1]).unsqueeze(0)
    mask = get_mask(
        iteration=5,
        total_iterations=10,
        total_tokens=5,
        token_confidence=token_confidence,
        schedule="linear"
    )

    assert mask.shape == token_confidence.shape
    assert mask.dtype == torch.float32


def test_get_mask_invalid_schedule():
    """Test that an invalid schedule raises a ValueError"""
    token_confidence = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1]).unsqueeze(0)

    with pytest.raises(ValueError, match="Invalid schedule function"):
        get_mask(
            iteration=5,
            total_iterations=10,
            total_tokens=5,
            token_confidence=token_confidence,
            schedule="invalid_schedule"
        )


def test_get_mask_edge_cases():
    """Test edge cases for mask generation"""
    # First iteration
    token_confidence = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1]).unsqueeze(0)
    mask_first = get_mask(
        iteration=0,
        total_iterations=10,
        total_tokens=5,
        token_confidence=token_confidence,
        schedule="cosine"
    )

    # Last iteration
    mask_last = get_mask(
        iteration=10,
        total_iterations=10,
        total_tokens=5,
        token_confidence=token_confidence,
        schedule="cosine"
    )

    # All tokens masked in first iteration
    assert torch.sum(mask_first.squeeze(0)) == len(
        token_confidence.squeeze(0))
    # No tokens masked in last iteration
    assert torch.sum(mask_last.squeeze(0)) == 0


def test_get_mask_confidence_threshold():
    """Test that tokens below confidence threshold are masked"""
    token_confidence = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1]).unsqueeze(0)
    mask = get_mask(
        iteration=5,
        total_iterations=10,
        total_tokens=5,
        token_confidence=token_confidence,
        schedule="cosine"
    )

    # Verify that masked tokens are those with lowest confidence
    sorted_indices = torch.argsort(token_confidence.squeeze(0))
    masked_indices = torch.where(mask.squeeze(0) == 1)

    assert set(masked_indices[0].tolist()).issubset(
        set(sorted_indices[:torch.sum(mask).int()].tolist()))
