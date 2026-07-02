import torch
import torch.nn.functional as F


def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
    """
    Original code: https://github.com/apple/ml-destseg/blob/main/model/losses.py#L13

    Accepts raw logits (not sigmoid outputs). Uses binary_cross_entropy_with_logits
    which is autocast-safe and numerically more stable than BCE(sigmoid(x), y).
    """
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    probs = torch.sigmoid(inputs)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
