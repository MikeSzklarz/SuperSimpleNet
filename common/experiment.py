import csv
import json
import logging
import re
from pathlib import Path


def _sanitize(s: str) -> str:
    return re.sub(r"[\s/\\]+", "_", str(s))


def _make_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _append_csv(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class ExperimentDir:
    """Central manager for a single training experiment's output directory.

    Creates a flat, self-contained directory named after the experiment
    parameters (dataset, category, supervision, seed, timestamp) and provides
    helpers to write all output artefacts into it.
    """

    def __init__(self, base_results_path: Path, config: dict):
        run_name = config.get("run_name")
        if run_name:
            self.experiment_id = _sanitize(run_name)
        else:
            dataset = _sanitize(config.get("dataset", "unknown"))
            category = _sanitize(config.get("category") or config.get("dataset", "none"))
            self.experiment_id = f"{dataset}__{category}"

        self.root = Path(base_results_path) / self.experiment_id
        self.checkpoints = self.root / "checkpoints"
        self.logs = self.root / "logs"
        self.scores = self.root / "scores"
        self.metrics_dir = self.root / "metrics"
        self.visual = self.root / "visual"
        self.tp_dir = self.visual / "true_positives"
        self.fp_dir = self.visual / "false_positives"
        self.tn_dir = self.visual / "true_negatives"
        self.fn_dir = self.visual / "false_negatives"

    def create_dirs(self) -> None:
        for d in [self.checkpoints, self.logs, self.scores, self.metrics_dir, self.visual]:
            d.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: dict) -> None:
        with open(self.root / "config.json", "w") as f:
            json.dump(_make_serializable(config), f, indent=2)

    def add_file_logging(self) -> None:
        """Attach a file handler to the 'ssn' logger, writing to logs/training.log.

        Removes any previously attached FileHandler first so that log output
        from earlier experiments (e.g. prior categories/folds in the same
        process) doesn't keep accumulating across every subsequent training.log.
        """
        logger = logging.getLogger("ssn")
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
                h.close()
        fh = logging.FileHandler(self.logs / "training.log", mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    def append_metrics_history(self, epoch: int, metrics: dict) -> None:
        """Append one row to logs/metrics_history.csv (called after each eval epoch)."""
        row = {"epoch": epoch, **{k: round(float(v), 6) for k, v in metrics.items()}}
        _append_csv(self.logs / "metrics_history.csv", row)

    def append_loss(
        self,
        epoch: int,
        avg_loss: float,
        lrs: list,
        grad_norm,
        loss_components: dict | None = None,
        synth_stats: dict | None = None,
    ) -> None:
        """Append one row to logs/losses.csv (called after each training epoch).

        loss_components / synth_stats are optional epoch-mean breakdowns (seg
        vs. cls loss terms, synthetic-anomaly injection stats) -- kept
        optional so a stalled avg_loss can be attributed to a specific term
        without re-deriving it from training.log text later.
        """
        row = {
            "epoch": epoch,
            "avg_loss": round(avg_loss, 6),
            "lr_adapt": round(lrs[0], 8) if len(lrs) > 0 else None,
            "lr_seg": round(lrs[1], 8) if len(lrs) > 1 else None,
            "lr_dec": round(lrs[2], 8) if len(lrs) > 2 else None,
            "grad_norm": round(float(grad_norm), 6) if grad_norm is not None else None,
        }
        if loss_components:
            row.update({k: round(v, 6) for k, v in loss_components.items()})
        if synth_stats:
            row.update({k: round(v, 6) if isinstance(v, float) else v for k, v in synth_stats.items()})
        _append_csv(self.logs / "losses.csv", row)

    def save_per_image_scores(self, rows: list[dict]) -> None:
        """Write scores/per_image.csv with one row per test image."""
        if not rows:
            return
        path = self.scores / "per_image.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def save_scores_json(self, score_dict: dict) -> None:
        """Write scores/scores.json in the legacy nested format."""
        with open(self.scores / "scores.json", "w") as f:
            json.dump(score_dict, f)

    def save_summary_metrics(self, metrics: dict) -> None:
        """Write metrics/summary.csv with the final evaluation metrics."""
        path = self.metrics_dir / "summary.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)

    def save_by_defect_metrics(self, per_image_rows: list[dict]) -> None:
        """Compute per-defect-type AUROC and AP, write metrics/by_defect.csv."""
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
        except ImportError:
            return

        defect_groups: dict[str, dict] = {}
        for row in per_image_rows:
            dt = row.get("defect_type", "unknown")
            if dt not in defect_groups:
                defect_groups[dt] = {"labels": [], "scores": []}
            defect_groups[dt]["labels"].append(row["ground_truth_label"])
            defect_groups[dt]["scores"].append(row["score"])

        output_rows = []
        for defect_type, data in sorted(defect_groups.items()):
            labels = data["labels"]
            scores = data["scores"]
            n_pos = sum(labels)
            n_neg = len(labels) - n_pos
            out_row = {
                "defect_type": defect_type,
                "n_total": len(labels),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "auroc": None,
                "ap": None,
            }
            if n_pos > 0 and n_neg > 0:
                try:
                    out_row["auroc"] = round(roc_auc_score(labels, scores), 6)
                    out_row["ap"] = round(average_precision_score(labels, scores), 6)
                except Exception:
                    pass
            output_rows.append(out_row)

        if not output_rows:
            return
        path = self.metrics_dir / "by_defect.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)
