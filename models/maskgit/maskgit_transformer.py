import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

from models.maskgit import get_mask


def get_confidence(token_dist: torch.Tensor, temp: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample token distributions and return confidence values and tokens

    :param token_dist: output from transformer
    :param temp: temperature
    :return: confidence values and sampled tokens
    """
    # token probability distributions
    prob_dist = F.softmax(token_dist / temp, dim=-1)  # B, N, C
    # sample token from dist
    sampled_tokens = torch.multinomial(prob_dist, num_samples=1)

    # get confidence score from token prob in each dist
    conf_vals = prob_dist[sampled_tokens]

    return conf_vals, sampled_tokens


def gen_image(
    input_seq: torch.Tensor,
    model: nn.Module,
    total_iterations: int,
    total_tokens: int,
    mask_token_id: int
) -> torch.Tensor:
    """
    Generate tokens for a single image

    :param input_seq: transformer input sequence
    :param model: st-transformer model
    :param total_iterations: max number of iterations
    :param total_tokens: total tokens in one image
    :param mask_token_id: mask token id
    :return: image tokens
    """
    for iteration in range(total_iterations):
        model_pred = model(input_seq)
        # sample tokens and get conf scores
        conf_vals, tokens = get_confidence(model_pred)
        # next iteration mask
        mask = get_mask(iteration, total_iterations, total_tokens, conf_vals)

        # update sequence with predicted tokens
        input_seq[:, -total_tokens] = torch.where(mask ==
                                                  1, mask_token_id, tokens)
    return input_seq
