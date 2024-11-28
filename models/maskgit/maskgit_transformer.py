import torch
import torch.nn.functional as F
from typing import Tuple


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
