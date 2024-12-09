import numpy as np
import torch


def get_mask(
    iteration: int,
    total_iterations: int,
    total_tokens: int,
    token_confidence: torch.Tensor,
    schedule: str = "cosine",
) -> torch.Tensor:
    """
    Get masks for current iteration in generation process

    :param iteration: current interation number
    :param total_iterations: max iteration
    :param total_tokens: number of tokens per image
    :param token_confidence: confidence values of tokens in previous iteration
    :param schedule: schedule function to use
    :return: binary masks
    :raises ValueError: invalid schedule function
    """
    if schedule == "cosine":

        def policy_func(x):
            return np.cos(x * np.pi / 2)

    elif schedule == "linear":

        def policy_func(x):
            return 1 - x

    else:
        raise ValueError(f"Invalid schedule function: {schedule}")

    # number of tokens to mask at current iteration
    n: int = (
        np.ceil(policy_func(iteration / total_iterations) * (total_tokens + 1)).astype(
            np.int32
        )
        - 1
    )

    # confidence value for top nth token
    confidence_threshold, _ = (
        torch.kthvalue(token_confidence, n) if n > 0 else (-1, None)
    )

    # mask out all tokens with confidence below threshold
    mask = (token_confidence <= confidence_threshold).float()
    return mask
