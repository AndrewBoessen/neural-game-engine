import torch
from torch import nn


def MaskedCrossEntropyLoss(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Cross Entropy Loss for Masked Transformer Model

    :param logits: ouput from model prediction head pre softmax
    :param labels: target token indicies
    :param mask: input token mask
    :return: loss value
    """

    # Validate input shapes
    if labels.shape != mask.shape:
        raise ValueError(
            f"Labels and mask must have same shape. Got {labels.shape} and {mask.shape}"
        )

    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"Logits and labels must have matching dimensions. Got {logits.shape} and {labels.shape}"
        )

    # Reshape logits and labels to calculate loss
    # logits: (batch_size * sequence_length, vocab_size)
    # labels: (batch_size * sequence_length)
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    # Flatten the logits and labels
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)

    # Compute loss for each token
    losses = loss_fct(flat_logits, flat_labels)

    # Reshape losses back to original shape
    losses = losses.view(labels.size())

    # Apply the mask to zero out losses for non-masked tokens
    # mask should be the same shape as labels, with 1s for masked tokens
    masked_losses = losses * mask.float()

    # Calculate the total loss
    # Divide by the number of masked tokens to normalize
    num_masked_tokens = mask.sum().float()

    if num_masked_tokens == 0:
        return torch.tensor(0.0, device=losses.device)

    total_loss = masked_losses.sum() / num_masked_tokens

    return total_loss
