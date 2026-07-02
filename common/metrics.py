import torch
from torchmetrics import Metric


class MaxF1Score(Metric):
    """Maximum image-level F1 score over all classification thresholds.

    Sweeps every unique prediction score as a threshold and reports the best
    F1 = 2·P·R / (P + R), equivalent to reporting the peak of the PR curve.
    No threshold needs to be chosen in advance.
    """

    higher_is_better = True
    is_differentiable = False
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.preds.append(preds.detach().cpu().float())
        self.target.append(target.detach().cpu().long())

    def compute(self) -> torch.Tensor:
        preds = torch.cat(self.preds) if isinstance(self.preds, list) else self.preds
        target = (torch.cat(self.target) if isinstance(self.target, list) else self.target).long()

        # Sort predictions descending — each position is a candidate threshold
        order = torch.argsort(preds, descending=True)
        target_sorted = target[order]

        tp = target_sorted.cumsum(0).float()
        fp = (1 - target_sorted).cumsum(0).float()
        fn = target_sorted.sum().float() - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        return f1.max()

    def compute_threshold(self) -> float:
        """Return the score threshold that achieves the best F1 on accumulated predictions."""
        preds = torch.cat(self.preds) if isinstance(self.preds, list) else self.preds
        target = (torch.cat(self.target) if isinstance(self.target, list) else self.target).long()

        order = torch.argsort(preds, descending=True)
        sorted_preds = preds[order]
        target_sorted = target[order]

        tp = target_sorted.cumsum(0).float()
        fp = (1 - target_sorted).cumsum(0).float()
        fn = target_sorted.sum().float() - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        return sorted_preds[f1.argmax()].item()
