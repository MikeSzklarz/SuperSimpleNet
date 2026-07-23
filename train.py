import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

from datamodules.base import Supervision

LOG_WANDB = False

if LOG_WANDB:
    import wandb

from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, seed_everything

from torchmetrics import AveragePrecision, Metric
from anomalib.utils.metrics import AUROC, AUPRO

from datamodules import ksdd2, sensum
from datamodules.ksdd2 import KSDD2, NumSegmented
from datamodules.sensum import Sensum, RatioSegmented
from datamodules.mvtec import MVTec
from datamodules.visa import Visa
from datamodules.custom import Custom

from model.supersimplenet import SuperSimpleNet

from common.visualizer import Visualizer
from common.results_writer import ResultsWriter
from common.loss import focal_loss
from common.logger import setup_logger, add_file_handler, get_logger
from common.metrics import MaxF1Score
from common.experiment import ExperimentDir, _sanitize


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _load_config_files(argv: list[str]) -> list[str]:
    """Expand --config <path> flags before argparse sees them.

    Each config file is one argument per line (blank lines and # comments ignored).
    Multiple --config flags are allowed and merged in order; CLI args added after
    --config override values from the file.
    """
    result = []
    i = 0
    while i < len(argv):
        if argv[i] in ("--config", "-c"):
            i += 1
            if i >= len(argv):
                print("ERROR: --config requires a path argument", file=sys.stderr)
                sys.exit(1)
            config_path = Path(argv[i])
            if not config_path.exists():
                print(
                    f"ERROR: config file not found: {config_path}\n"
                    f"  Looked in: {config_path.resolve()}",
                    file=sys.stderr,
                )
                sys.exit(1)
            with open(config_path) as f:
                lines = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]
            result.extend(lines)
        else:
            result.append(argv[i])
        i += 1
    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Config files can be loaded with --config path/to/file.txt (one arg per line).
    Multiple --config flags are supported and merged in order. The legacy
    @path/to/file.txt syntax also still works.
    """
    parser = argparse.ArgumentParser(
        description="SuperSimpleNet — train and evaluate anomaly detection",
        fromfile_prefix_chars="@",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Config files: use  --config configs/custom_fully_sup.txt  to load arguments "
            "from a file (one argument per line, # comments supported). "
            "CLI arguments after --config override values from the file."
        ),
    )

    # ---- dataset / paths ----
    ds = parser.add_argument_group("Dataset")
    ds.add_argument(
        "--dataset",
        required=True,
        choices=["mvtec", "visa", "ksdd2", "sensum", "custom"],
        help="Which dataset to train on",
    )
    ds.add_argument(
        "--mode",
        default="unsup",
        choices=["unsup", "sup"],
        help="Supervision mode (unsup = unsupervised, sup = mixed/full supervision)",
    )
    ds.add_argument(
        "--datasets-folder",
        type=Path,
        default=Path("./datasets"),
        help="Root folder that contains all dataset sub-directories",
    )
    ds.add_argument(
        "--data-root",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to the custom dataset root directory (required when --dataset custom). "
            "Pass this on the command line rather than in a config file so it stays "
            "machine-specific. The directory must exist and contain train/ and test/ subfolders."
        ),
    )
    ds.add_argument(
        "--good-test-split",
        type=float,
        default=0.2,
        metavar="FRAC",
        help="Fraction of train/good images withheld for test when no test/good folder exists",
    )
    ds.add_argument(
        "--defect-train-split",
        type=float,
        default=0.0,
        metavar="FRAC",
        help=(
            "Fraction of test/<defect>/ images automatically used for supervised training. "
            "Required for weakly/mixed/fully supervision (e.g. 0.5). "
            "The remaining fraction is always reserved for test evaluation. "
            "No manual file organisation needed."
        ),
    )
    ds.add_argument(
        "--category",
        default=None,
        help="Override category name (custom dataset only; defaults to folder name)",
    )
    ds.add_argument(
        "--supervision",
        choices=["unsupervised", "weakly", "mixed", "fully"],
        default=None,
        help=(
            "Supervision level for custom dataset training. "
            "unsupervised = normal samples only; "
            "weakly = anomalous training samples, masks used when available; "
            "mixed = same as weakly; "
            "fully = all anomalous training samples must have masks. "
            "Default: unsupervised when --mode unsup, mixed when --mode sup."
        ),
    )

    # ---- output / book-keeping ----
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        metavar="PATH",
        help="Root directory where all experiment outputs are written (default: ./results)",
    )
    out.add_argument(
        "--run-name",
        default=None,
        help=(
            "Name for this experiment's output subfolder under results-save-path "
            "(e.g. 'custom-fully-supervised'). Set this in your config file to give "
            "each experiment a meaningful, stable name. If omitted, a short name is "
            "generated from dataset and category."
        ),
    )
    out.add_argument(
        "--setup-name",
        default="superSimpleNet",
        help="Legacy sub-directory name (unused when --run-name is set)",
    )
    out.add_argument("--wandb-project", default="ssn", help="Weights & Biases project name")

    # ---- model architecture ----
    arch = parser.add_argument_group("Architecture")
    arch.add_argument("--backbone", default="wide_resnet50_2", help="Torchvision backbone name")
    arch.add_argument(
        "--layers",
        nargs="+",
        default=["layer2", "layer3"],
        metavar="LAYER",
        help="Backbone layers to extract features from",
    )
    arch.add_argument("--patch-size", type=int, default=3, help="Avg-pool kernel for neighbourhood aggregation")

    # ---- training schedule ----
    tr = parser.add_argument_group("Training")
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--batch", type=int, default=32, help="Batch size for training and evaluation")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    tr.add_argument("--eval-step-size", type=int, default=4, help="Run evaluation every N epochs")
    tr.add_argument("--image-size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))

    # ---- optimiser ----
    opt = parser.add_argument_group("Optimiser")
    opt.add_argument("--seg-lr", type=float, default=0.0002)
    opt.add_argument("--dec-lr", type=float, default=0.0002)
    opt.add_argument("--adapt-lr", type=float, default=0.0001)
    opt.add_argument("--gamma", type=float, default=0.4, help="LR scheduler decay factor")

    # ---- anomaly generation ----
    ano = parser.add_argument_group("Anomaly generation")
    ano.add_argument("--noise-std", type=float, default=0.015, help="Std dev of Gaussian feature noise")
    ano.add_argument(
        "--perlin-thr",
        type=float,
        default=None,
        metavar="THR",
        help="Perlin noise binarisation threshold (dataset-specific default applied if omitted)",
    )
    ano.add_argument(
        "--no-anomaly",
        default="empty",
        choices=["empty", "full", "none"],
        help="What to do when 50%% chance selects 'no anomaly': empty=zero mask, full=full mask, none=keep",
    )
    ano.add_argument("--noise", default=True, action=argparse.BooleanOptionalAction)
    ano.add_argument("--perlin", default=True, action=argparse.BooleanOptionalAction)
    ano.add_argument("--bad", default=True, action=argparse.BooleanOptionalAction,
                     help="Apply noise to already-anomalous samples")
    ano.add_argument("--overlap", default=False, action=argparse.BooleanOptionalAction,
                     help="Allow synthetic noise to overlap existing anomalous regions")

    # ---- training flags ----
    fl = parser.add_argument_group("Training flags")
    fl.add_argument("--clip-grad", default=None, action=argparse.BooleanOptionalAction,
                    help="Clip gradients to norm=1 (default: False for unsup, True for sup)")
    fl.add_argument("--stop-grad", default=None, action=argparse.BooleanOptionalAction,
                    help="Stop gradient from anomaly map into decoder head (default: True for unsup, False for sup)")
    fl.add_argument("--adapt-cls-feat", default=False, action=argparse.BooleanOptionalAction,
                    help="Use adapted features for classification head (ICPR extension)")
    fl.add_argument("--flips", default=None, action=argparse.BooleanOptionalAction,
                    help="Augment anomalous training samples with flips (default: False for unsup, True for sup)")
    fl.add_argument("--amp", action="store_true", default=False,
                    help="Enable automatic mixed precision (fp16) training — CUDA only")

    # ---- supervised-only ----
    sup = parser.add_argument_group("Supervised options (ksdd2 / sensum only)")
    sup.add_argument("--dt", type=int, nargs=2, default=None, metavar=("WEIGHT", "POWER"),
                     help="Distance transform params (default: 3 2 for sup mode)")
    sup.add_argument("--dilate", type=int, default=None,
                     help="Mask dilation kernel size (default: 7 for sup mode)")
    sup.add_argument("--ratio", type=float, default=None,
                     help="Labeled-defect knob: ksdd2 = count of segmented samples (0/16/53/126/246), "
                          "sensum = fraction segmented (0.0-1.0), custom = fraction of masked training "
                          "defects that keep their masks (0.0-1.0, mixed supervision only; "
                          "weakly forces 0, fully ignores it; default 1.0)")

    return parser.parse_args(_load_config_files(sys.argv[1:]))


def _build_config(args: argparse.Namespace) -> dict:
    """Convert parsed args into the config dict used by the rest of the code."""
    is_sup = args.mode == "sup"

    # mode-specific defaults for flags not explicitly set
    clip_grad = args.clip_grad if args.clip_grad is not None else is_sup
    stop_grad = args.stop_grad if args.stop_grad is not None else (not is_sup)
    flips = args.flips if args.flips is not None else is_sup
    dt = tuple(args.dt) if args.dt is not None else ((3, 2) if is_sup else None)
    dilate = args.dilate if args.dilate is not None else (7 if is_sup else None)

    # dataset-specific perlin threshold defaults
    if args.perlin_thr is not None:
        perlin_thr = args.perlin_thr
    elif args.dataset == "mvtec":
        perlin_thr = 0.2
    else:
        perlin_thr = 0.6

    config = {
        "wandb_project": args.wandb_project,
        "datasets_folder": args.datasets_folder,
        "num_workers": args.num_workers,
        "setup_name": args.setup_name,
        "backbone": args.backbone,
        "layers": args.layers,
        "patch_size": args.patch_size,
        "noise": args.noise,
        "perlin": args.perlin,
        "no_anomaly": args.no_anomaly,
        "bad": args.bad,
        "overlap": args.overlap,
        "adapt_cls_feat": args.adapt_cls_feat,
        "noise_std": args.noise_std,
        "perlin_thr": perlin_thr,
        "image_size": tuple(args.image_size),
        "seed": args.seed,
        "batch": args.batch,
        "epochs": args.epochs,
        "flips": flips,
        "seg_lr": args.seg_lr,
        "dec_lr": args.dec_lr,
        "adapt_lr": args.adapt_lr,
        "gamma": args.gamma,
        "stop_grad": stop_grad,
        "clip_grad": clip_grad,
        "eval_step_size": args.eval_step_size,
        "results_save_path": args.results_dir,
        "run_name": args.run_name,
        "dt": dt,
        "dilate": dilate,
        "use_amp": args.amp,
    }

    return config


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: SuperSimpleNet,
    epochs: int,
    datamodule: LightningDataModule,
    device: str,
    image_metrics: dict[str, Metric],
    pixel_metrics: dict[str, Metric],
    th: float = 0.5,
    clip_grad: bool = True,
    eval_step_size: int = 4,
    use_amp: bool = False,
    exp: "ExperimentDir" = None,
    skip_pixel_metrics: bool = False,
):
    log = get_logger()
    model.to(device)
    optimizer, scheduler = model.get_optimizers()

    amp_enabled = use_amp and device == "cuda"
    if use_amp and not amp_enabled:
        log.warning("  AMP requested but device is CPU — AMP disabled.")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    autocast_device = "cuda" if device == "cuda" else "cpu"

    log.info("")
    log.info("╔═══════════════════════════════════════════════════════════╗")
    log.info("║                     Training Started                     ║")
    log.info("╚═══════════════════════════════════════════════════════════╝")
    log.info("  Epochs: %d   Eval every: %d epochs   Device: %s",
             epochs, eval_step_size, device)
    log.info("  Gradient clipping: %s", clip_grad)
    log.info("  Mixed precision (AMP): %s", "ENABLED (fp16)" if amp_enabled else "DISABLED")
    log.info("")

    results = {}
    model.train()
    train_loader = datamodule.train_dataloader()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # pre-processing diagnostic: how much synthetic (Perlin+noise) anomaly
        # area is actually being fed to the discriminator this epoch, and how
        # many batches ended up with zero anomalous samples at all (relevant
        # at small batch sizes / weak-mixed supervision, where an unlucky
        # batch can starve the seg head of positive-class gradient signal).
        epoch_synth_area_sum = 0.0
        epoch_synth_batches = 0
        epoch_batches_no_positive = 0
        log.info("─── Training: Epoch %d / %d ──────────────────────────────────", epoch + 1, epochs)

        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{epochs}",
            miniters=int(1),
            unit="batch",
        ) as prog_bar:
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()

                image_batch = batch["image"].to(device)

                # best downsampling proposed by DestSeg
                mask = batch["mask"].type(torch.float32).to(device)
                mask = F.interpolate(
                    mask.unsqueeze(1),
                    size=(model.fh, model.fw),
                    mode="bilinear",
                    align_corners=True,
                )
                mask = torch.where(
                    mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
                )

                label = batch["label"].to(device).type(torch.float32)
                is_segmented = batch["is_segmented"].to(device).type(torch.float32)

                with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=amp_enabled):
                    anomaly_map, score, mask, label = model.forward(
                        image_batch, mask, label
                    )

                    gen_stats = model.anomaly_generator
                    if gen_stats.last_injected_area_frac is not None:
                        epoch_synth_area_sum += gen_stats.last_injected_area_frac
                        epoch_synth_batches += 1
                        if not gen_stats.last_batch_has_positive:
                            epoch_batches_no_positive += 1

                    seg_focal = focal_loss(anomaly_map, mask, reduction=None)

                    # use this shape to apply weights from distance transform if enabled
                    seg_l1 = torch.zeros_like(anomaly_map)

                    # adjusted truncated l1: mask + flipped sign (ano->pos, good->neg)
                    normal_scores = anomaly_map[mask == 0]
                    seg_l1[mask == 0] = torch.clip(normal_scores + th, min=0)

                    anomalous_scores = anomaly_map[mask > 0]
                    seg_l1[mask > 0] = torch.clip(-anomalous_scores + th, min=0)

                    if "loss_mask" in batch:
                        loss_mask = batch["loss_mask"].type(torch.float32).to(device)

                        # resize loss_mask to fit the loss
                        loss_mask = F.interpolate(
                            loss_mask.unsqueeze(1),
                            size=seg_focal.shape[-2:],
                            mode="bilinear",
                            align_corners=True,
                        )

                        # due to feat. duplication stack mask and multiply to get weighted loss
                        loss_mask = torch.cat((loss_mask, loss_mask))
                        seg_focal *= loss_mask
                        seg_l1 *= loss_mask

                    # due to feat. duplication
                    is_segmented = torch.cat((is_segmented, is_segmented)).type(torch.bool)

                    bad_loss = seg_l1[is_segmented][mask[is_segmented] > 0]
                    good_loss = seg_l1[is_segmented][mask[is_segmented] == 0]
                    focal_val = seg_focal[is_segmented]

                    if len(good_loss):
                        good_loss = good_loss.mean()
                    else:
                        good_loss = 0
                    if len(bad_loss):
                        bad_loss = bad_loss.mean()
                    else:
                        bad_loss = 0
                    if len(focal_val):
                        focal_val = focal_val.mean()
                    else:
                        focal_val = 0

                    # seg loss is combination of trunc l1 and focal (separately avg each l1 part due to unbalanced pixels)
                    seg_loss = good_loss + bad_loss + focal_val

                    loss = seg_loss + focal_loss(score, label)

                scaler.scale(loss).backward()

                if clip_grad:
                    # must unscale before clipping so the norm is in the original gradient space
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                else:
                    norm = None

                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.detach().cpu().item()

                output = {
                    "batch_loss": np.round(loss.data.cpu().detach().numpy(), 5),
                    "avg_loss": np.round(total_loss / (i + 1), 5),
                    "norm": norm,
                }

                prog_bar.set_postfix(**output)
                prog_bar.update(1)

            avg_loss = total_loss / len(train_loader)
            lrs = [pg["lr"] for pg in optimizer.param_groups]
            log.info(
                "  Epoch %d done  |  avg_loss=%.5f  |  grad_norm=%s  |  lr(adapt/seg/dec)=%.6f/%.6f/%.6f",
                epoch + 1,
                avg_loss,
                f"{norm:.4f}" if norm is not None else "N/A",
                lrs[0], lrs[1], lrs[2],
            )
            if epoch_synth_batches > 0:
                log.info(
                    "  Epoch %d synth-anomaly stats  |  mean injected area=%.3f%%  |  "
                    "batches w/ zero positive samples=%d/%d (%.1f%%)",
                    epoch + 1,
                    100 * epoch_synth_area_sum / epoch_synth_batches,
                    epoch_batches_no_positive, epoch_synth_batches,
                    100 * epoch_batches_no_positive / epoch_synth_batches,
                )
            log.debug("  Epoch %d raw output: %s", epoch + 1, output)

            if exp is not None:
                exp.append_loss(epoch + 1, avg_loss, lrs, norm)

            if (epoch + 1) % eval_step_size == 0:
                log.info("─── Evaluation at Epoch %d / %d ──────────────────────────", epoch + 1, epochs)
                results = test(
                    model=model,
                    datamodule=datamodule,
                    device=device,
                    image_metrics=image_metrics,
                    pixel_metrics=pixel_metrics,
                    normalize=True,
                    skip_pixel_metrics=skip_pixel_metrics,
                )
                if exp is not None:
                    exp.append_metrics_history(epoch + 1, results)
                if LOG_WANDB:
                    wandb.log({**results, **output})
            else:
                if LOG_WANDB:
                    wandb.log(output)

        scheduler.step()

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _log_peak_concentration(results: dict, log: logging.Logger) -> None:
    """Post-processing diagnostic: how spatially concentrated are predicted
    anomaly-map peaks across the test set?

    A model doing genuine content-driven localization should have peaks
    scattered around the image. Peaks clustering tightly -- especially for
    NORMAL images, which have nothing to localize -- means the model has
    learned to point at a fixed spatial location instead of image content.
    Split out normal vs. anomalous so a normal-only cluster is unmissable.
    """
    am = results["anomaly_map"].squeeze(1)  # [N, H, W]
    n_img, h_img, w_img = am.shape
    if n_img == 0:
        return
    flat = am.reshape(n_img, -1)
    peak_idx = flat.argmax(dim=1)
    peak_y = (peak_idx // w_img).float()
    peak_x = (peak_idx % w_img).float()
    diag = (h_img**2 + w_img**2) ** 0.5

    def _stats(sel: torch.Tensor) -> dict | None:
        if sel.sum() == 0:
            return None
        py, px = peak_y[sel], peak_x[sel]
        med_y, med_x = py.median(), px.median()
        dist = ((py - med_y) ** 2 + (px - med_x) ** 2).sqrt()
        return {
            "n": int(sel.sum()),
            "median_peak": (med_y.item(), med_x.item()),
            "peak_std": (
                py.std().item() if len(py) > 1 else 0.0,
                px.std().item() if len(px) > 1 else 0.0,
            ),
            "pct_within_5pct_diag": (dist < 0.05 * diag).float().mean().item() * 100,
        }

    label_bool = results["label"].bool()
    groups = [
        ("all", torch.ones(n_img, dtype=torch.bool)),
        ("normal-only", ~label_bool),
        ("anomalous-only", label_bool),
    ]

    log.info("  Anomaly-map peak concentration (post-processing diagnostic):")
    for name, sel in groups:
        s = _stats(sel)
        if s is None:
            log.info("    %-15s n=0", name)
        else:
            log.info(
                "    %-15s n=%-4d  median_peak=(y=%.0f,x=%.0f)  peak_std=(%.1f,%.1f)px  "
                "%.1f%% of peaks within 5%% of image diagonal from median",
                name, s["n"], *s["median_peak"], *s["peak_std"], s["pct_within_5pct_diag"],
            )


@torch.no_grad()
def test(
    model: SuperSimpleNet,
    datamodule: LightningDataModule,
    device: str,
    image_metrics: dict[str, Metric],
    pixel_metrics: dict[str, Metric],
    normalize: bool = True,
    image_save_path: Path = None,
    score_save_path: Path = None,
    exp: "ExperimentDir" = None,
    skip_pixel_metrics: bool = False,
):
    log = get_logger()
    model.to(device)
    model.eval()

    # for anomaly map max as image score
    seg_image_metrics = {}

    for m_name, metric in image_metrics.items():
        metric.cpu()
        metric.reset()
        seg_image_metrics[f"seg-{m_name}"] = copy.deepcopy(metric)

    for metric in pixel_metrics.values():
        metric.cpu()
        metric.reset()

    test_loader = datamodule.test_dataloader()
    n_normal = datamodule.test_data.num_neg
    n_anomalous = datamodule.test_data.num_pos
    n_total = n_normal + n_anomalous

    log.info("  Test set:  %d images total  (%d normal / %d anomalous)",
             n_total, n_normal, n_anomalous)

    results = {
        "anomaly_map": [],
        "gt_mask": [],
        "score": [],
        "seg_score": [],
        "label": [],
        "image_path": [],
        "mask_path": [],
    }
    for batch in tqdm(test_loader, position=0, leave=True, desc="Evaluating"):
        image_batch = batch["image"].to(device)
        anomaly_map, anomaly_score = model.forward(image_batch)

        anomaly_map = anomaly_map.detach().cpu()
        anomaly_score = anomaly_score.detach().cpu()

        results["anomaly_map"].append(torch.sigmoid(anomaly_map).detach().cpu())
        results["gt_mask"].append(batch["mask"].detach().cpu())

        results["score"].append(torch.sigmoid(anomaly_score))
        results["seg_score"].append(
            anomaly_map.reshape(anomaly_map.shape[0], -1).max(dim=1).values
        )
        results["label"].append(batch["label"].detach().cpu())

        results["image_path"].extend(batch["image_path"])
        results["mask_path"].extend(batch["mask_path"])

    results["anomaly_map"] = torch.cat(results["anomaly_map"])
    results["score"] = torch.cat(results["score"])
    results["seg_score"] = torch.cat(results["seg_score"])
    results["gt_mask"] = torch.cat(results["gt_mask"])
    results["label"] = torch.cat(results["label"])

    # normalize
    if normalize:
        results["anomaly_map"] = (
            results["anomaly_map"] - results["anomaly_map"].flatten().min()
        ) / (
            results["anomaly_map"].flatten().max()
            - results["anomaly_map"].flatten().min()
        )
        results["score"] = (results["score"] - results["score"].min()) / (
            results["score"].max() - results["score"].min()
        )
        results["seg_score"] = (results["seg_score"] - results["seg_score"].min()) / (
            results["seg_score"].max() - results["seg_score"].min()
        )

    _log_peak_concentration(results, log)

    results_dict = {}
    for name, metric in image_metrics.items():
        metric.update(results["score"], results["label"])
        results_dict[name] = metric.to(device).compute().item()
        metric.to("cpu")

    for name, metric in seg_image_metrics.items():
        metric.update(results["seg_score"], results["label"])
        results_dict[name] = metric.to(device).compute().item()
        metric.to("cpu")

    if skip_pixel_metrics:
        # Weakly supervised: pixel-level ground truth (even when present on disk)
        # isn't a fair target -- masks were never used to train the model, so
        # localization metrics don't measure anything meaningful here. Report NA
        # rather than a number that looks like a real score.
        for name in pixel_metrics:
            results_dict[name] = float("nan")
    else:
        for name, metric in pixel_metrics.items():
            try:
                # avoid nan in early stages
                am = results["anomaly_map"]
                am[am != am] = 0
                results["anomaly_map"] = am

                metric.update(
                    results["anomaly_map"], results["gt_mask"].type(torch.float32)
                )
                results_dict[name] = metric.to(device).compute().item()
            except RuntimeError:
                # AUPRO in some cases with early predictions crashes cuda, so just skip it in that case
                results_dict[name] = 0
            metric.to("cpu")

    log.info("  ┌─────────────────────────────┐")
    for name, value in results_dict.items():
        log.info("  │  %-20s  %6.4f  │", name, value)
    log.info("  └─────────────────────────────┘")

    # legacy per-line print kept for tqdm compatibility
    for name, value in results_dict.items():
        print(f"{name}: {value} ", end="")
    print()

    # Compute best-F1 threshold and per-image predicted labels
    max_f1_metric = image_metrics.get("I-MaxF1")
    if max_f1_metric is not None:
        best_threshold = max_f1_metric.compute_threshold()
    else:
        best_threshold = 0.5
    predicted_labels = (results["score"] >= best_threshold).long()
    results["predicted_label"] = predicted_labels

    # Build the legacy nested score dict (kept for backward compat)
    score_dict: dict = {}
    for img_path, score, seg_score, label in zip(
        results["image_path"],
        results["score"],
        results["seg_score"],
        results["label"],
    ):
        img_path = Path(img_path)
        anomaly_type = img_path.parent.name
        if anomaly_type not in score_dict:
            score_dict[anomaly_type] = {"good": {}, "bad": {}}
        kind = "bad" if label == 1 else "good"
        score_dict[anomaly_type][kind][img_path.stem] = {
            "score": score.item(),
            "seg_score": seg_score.item(),
        }

    if exp is not None:
        # Rich per-image CSV with classification outcome
        per_image_rows = []
        for img_path, score, seg_score, gt_label, pred_label in zip(
            results["image_path"],
            results["score"],
            results["seg_score"],
            results["label"],
            predicted_labels,
        ):
            per_image_rows.append({
                "image_name": Path(img_path).name,
                "defect_type": Path(img_path).parent.name,
                "ground_truth_label": gt_label.item(),
                "predicted_label": pred_label.item(),
                "score": round(score.item(), 6),
                "seg_score": round(seg_score.item(), 6),
                "is_correct": int(gt_label.item() == pred_label.item()),
            })

        exp.save_per_image_scores(per_image_rows)
        exp.save_scores_json(score_dict)
        exp.save_summary_metrics(results_dict)
        exp.save_by_defect_metrics(per_image_rows)

        log.info("  Saving visualizations → %s", exp.visual)
        outcome_dirs = {
            "TP": exp.tp_dir,
            "FP": exp.fp_dir,
            "TN": exp.tn_dir,
            "FN": exp.fn_dir,
        }
        visualizer = Visualizer(exp.visual)
        visualizer.visualize(results, outcome_dirs=outcome_dirs)
        visualizer.save_aggregate_heatmap(results, exp.visual)
    else:
        if image_save_path:
            log.info("  Saving visualizations → %s", image_save_path)
            visualizer = Visualizer(image_save_path)
            visualizer.visualize(results)
            visualizer.save_aggregate_heatmap(results, image_save_path)

        if score_save_path:
            score_save_path.mkdir(exist_ok=True, parents=True)
            with open(score_save_path / "scores.json", "w") as f:
                json.dump(score_dict, f)
            log.info("  Scores saved → %s/scores.json", score_save_path)

    return results_dict


# ---------------------------------------------------------------------------
# train + eval wrapper
# ---------------------------------------------------------------------------

def _open_experiment(config: dict) -> ExperimentDir:
    """Create the experiment dir and attach its training.log file handler.

    Must be called as soon as the experiment-defining config fields (dataset,
    category/run_name, results_save_path) are finalized for a run - i.e.
    before any dataset/model/datamodule setup logging happens - otherwise
    that early output only reaches the console (slurm .out) and is missing
    from training.log.
    """
    exp = ExperimentDir(Path(config["results_save_path"]), config)
    exp.create_dirs()
    exp.save_config(config)
    exp.add_file_logging()
    return exp


def train_and_eval(model, datamodule, config, device, exp: ExperimentDir):
    log = get_logger()

    log.info("")
    log.info("  Experiment dir: %s", exp.root)
    log.info("")

    if LOG_WANDB:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.init(project=config["wandb_project"], config=config, name=config["name"])

    image_metrics = {
        "I-MaxF1": MaxF1Score(),
        "I-AUROC": AUROC(),
        "AP-det": AveragePrecision(num_classes=1),
    }
    pixel_metrics = {
        "P-AUROC": AUROC(),
        "AUPRO": AUPRO(),
        "AP-loc": AveragePrecision(num_classes=1),
    }
    skip_pixel_metrics = config.get("supervision_str") == Supervision.WEAKLY_SUPERVISED.value
    if skip_pixel_metrics:
        log.info("  Supervision is weakly_supervised: pixel-level metrics "
                 "(P-AUROC/AUPRO/AP-loc) will be reported as NA, even if ground "
                 "truth masks are present -- they were never used for training.")

    train(
        model=model,
        epochs=config["epochs"],
        datamodule=datamodule,
        device=device,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        clip_grad=config["clip_grad"],
        eval_step_size=config["eval_step_size"],
        use_amp=config.get("use_amp", False),
        exp=exp,
        skip_pixel_metrics=skip_pixel_metrics,
    )
    if LOG_WANDB:
        wandb.finish()

    try:
        model.save_model(exp.checkpoints)
    except Exception as e:
        log.warning("Error saving checkpoint: %s", e)

    log.info("")
    log.info("─── Final Evaluation ────────────────────────────────────────")
    results = test(
        model=model,
        datamodule=datamodule,
        device=device,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        skip_pixel_metrics=skip_pixel_metrics,
        normalize=True,
        exp=exp,
    )

    log.info("")
    log.info("─── Run Complete ────────────────────────────────────────────")
    log.info("  Experiment dir: %s", exp.root)
    log.info("─────────────────────────────────────────────────────────────")

    return results


# ---------------------------------------------------------------------------
# Per-dataset main functions
# ---------------------------------------------------------------------------

def main_mvtec(device, config):
    log = get_logger()
    config = copy.deepcopy(config)
    config["dataset"] = "mvtec"
    config["ratio"] = 1

    categories = [
        "screw", "pill", "capsule", "carpet", "grid", "tile", "wood",
        "zipper", "cable", "toothbrush", "transistor", "metal_nut",
        "bottle", "hazelnut", "leather",
    ]

    results_writer = ResultsWriter(
        metrics=["AP-det", "AP-loc", "P-AUROC", "I-MaxF1", "I-AUROC", "AUPRO", "seg-AP-det", "seg-I-AUROC", "seg-I-MaxF1"]
    )

    for category in categories:
        config["category"] = category
        config["name"] = f"{category}_{config['setup_name']}"
        exp = _open_experiment(config)

        log.info("")
        log.info("╔═══════════════════════════════════════════════════════════╗")
        log.info("║  Dataset: MVTec   Category: %-29s ║", category)
        log.info("╚═══════════════════════════════════════════════════════════╝")

        seed_everything(config["seed"], workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = SuperSimpleNet(image_size=config["image_size"], config=config)

        datamodule = MVTec(
            root=Path(config["datasets_folder"]) / "mvtec",
            category=category,
            image_size=config["image_size"],
            train_batch_size=config["batch"],
            eval_batch_size=config["batch"],
            num_workers=config["num_workers"],
            seed=config["seed"],
        )
        datamodule.setup()

        results = train_and_eval(model=model, datamodule=datamodule, config=config, device=device, exp=exp)

        results_writer.add_result(category=category, last=results)
        results_writer.save(
            Path(config["results_save_path"]) / "aggregated" / config["dataset"]
        )


def main_visa(device, config):
    log = get_logger()
    config = copy.deepcopy(config)
    config["dataset"] = "visa"
    config["ratio"] = 1

    categories = [
        "candle", "capsules", "cashew", "chewinggum", "fryum",
        "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
    ]

    results_writer = ResultsWriter(
        metrics=["AP-det", "AP-loc", "P-AUROC", "I-MaxF1", "I-AUROC", "AUPRO", "seg-AP-det", "seg-I-AUROC", "seg-I-MaxF1"]
    )

    for category in categories:
        config["category"] = category
        config["name"] = f"{category}_{config['setup_name']}"
        exp = _open_experiment(config)

        log.info("")
        log.info("╔═══════════════════════════════════════════════════════════╗")
        log.info("║  Dataset: VisA    Category: %-29s ║", category)
        log.info("╚═══════════════════════════════════════════════════════════╝")

        seed_everything(config["seed"], workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = SuperSimpleNet(image_size=config["image_size"], config=config)

        datamodule = Visa(
            root=Path(config["datasets_folder"]) / "visa",
            category=category,
            image_size=config["image_size"],
            train_batch_size=config["batch"],
            eval_batch_size=config["batch"],
            num_workers=config["num_workers"],
            seed=config["seed"],
        )
        datamodule.setup()

        results = train_and_eval(model=model, datamodule=datamodule, config=config, device=device, exp=exp)

        results_writer.add_result(category=category, last=results)
        results_writer.save(
            Path(config["results_save_path"]) / "aggregated" / config["dataset"]
        )


def main_ksdd2(device, config, supervision):
    log = get_logger()
    config = copy.deepcopy(config)
    config["dataset"] = "ksdd2"
    config["category"] = "ksdd2"
    config["name"] = f"ksdd2_{config['setup_name']}"

    results_writer = ResultsWriter(
        metrics=["AP-det", "AP-loc", "P-AUROC", "I-MaxF1", "I-AUROC", "AUPRO", "seg-AP-det", "seg-I-AUROC", "seg-I-MaxF1", "ratio"]
    )

    exp = _open_experiment(config)

    log.info("")
    log.info("╔═══════════════════════════════════════════════════════════╗")
    log.info("║  Dataset: KSDD2   Ratio: %-33s ║", config.get("ratio", "?"))
    log.info("╚═══════════════════════════════════════════════════════════╝")

    seed_everything(config["seed"], workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = SuperSimpleNet(image_size=ksdd2.get_default_resolution(), config=config)

    datamodule = KSDD2(
        root=Path(config["datasets_folder"]) / "KolektorSDD2",
        supervision=supervision,
        image_size=ksdd2.get_default_resolution(),
        train_batch_size=config["batch"],
        eval_batch_size=config["batch"],
        num_workers=config["num_workers"],
        num_segmented=NumSegmented(config["ratio"]),
        seed=config["seed"],
        flips=config["flips"],
        dt=config["dt"],
        dilate=config["dilate"],
    )
    datamodule.setup()

    results = train_and_eval(model=model, datamodule=datamodule, config=config, device=device, exp=exp)

    results["ratio"] = config["ratio"]
    results_writer.add_result(category="ksdd2", last=results)
    results_writer.save(
        Path(config["results_save_path"]) / "aggregated" / config["dataset"]
    )


def main_sensum(device, config, supervision):
    log = get_logger()
    config = copy.deepcopy(config)
    config["dataset"] = "sensum"

    results_writer = ResultsWriter(
        metrics=["AP-det", "AP-loc", "P-AUROC", "I-MaxF1", "I-AUROC", "AUPRO", "seg-AP-det", "seg-I-AUROC", "seg-I-MaxF1", "fold", "ratio"]
    )

    for category in [sensum.Category.Capsule, sensum.Category.Softgel]:
        log.info("")
        log.info("╔═══════════════════════════════════════════════════════════╗")
        log.info("║  Dataset: Sensum  Category: %-29s ║", category.value)
        log.info("╚═══════════════════════════════════════════════════════════╝")

        seed_everything(config["seed"], workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for fold_num in range(3):
            config["category"] = f"{category.value}_{fold_num}"
            config["name"] = f"{category.value}_{config['setup_name']}_{fold_num}"
            config["fold"] = fold_num
            exp = _open_experiment(config)

            log.info("  ─── Fold %d / 3 ────────────────────────────────────────", fold_num + 1)

            model = SuperSimpleNet(
                image_size=sensum.get_default_resolution(category), config=config
            )

            datamodule = Sensum(
                root=Path(config["datasets_folder"]) / "SensumSODF",
                supervision=supervision,
                fold=sensum.FixedFoldNumber(fold_num),
                category=category,
                image_size=sensum.get_default_resolution(category),
                train_batch_size=config["batch"],
                eval_batch_size=config["batch"],
                num_workers=config["num_workers"],
                ratio_segmented=sensum.RatioSegmented(config["ratio"]),
                seed=config["seed"],
                flips=config["flips"],
                dt=config["dt"],
                dilate=config["dilate"],
            )
            datamodule.setup()

            results = train_and_eval(model=model, datamodule=datamodule, config=config, device=device, exp=exp)

            results["fold"] = fold_num
            results["ratio"] = config["ratio"]
            results_writer.add_result(category=f"{category.value}", last=results)
            results_writer.save(
                Path(config["results_save_path"]) / "aggregated" / config["dataset"]
            )


def main_custom(
    device,
    config,
    custom_root: Path,
    good_test_split: float,
    defect_train_split: float,
    category_name: str,
    supervision: Supervision = Supervision.UNSUPERVISED,
    mask_ratio: float = 1.0,
):
    log = get_logger()
    config = copy.deepcopy(config)
    config["dataset"] = "custom"
    config["ratio"] = mask_ratio
    config["category"] = category_name
    config["name"] = f"{category_name}_{config['setup_name']}"

    results_writer = ResultsWriter(
        metrics=["AP-det", "AP-loc", "P-AUROC", "I-MaxF1", "I-AUROC", "AUPRO", "seg-AP-det", "seg-I-AUROC", "seg-I-MaxF1"]
    )

    exp = _open_experiment(config)

    log.info("")
    log.info("╔═══════════════════════════════════════════════════════════╗")
    log.info("║  Dataset: Custom  Category: %-29s ║", category_name)
    log.info("╚═══════════════════════════════════════════════════════════╝")
    log.info("  Root:            %s", custom_root)
    log.info("  Supervision:     %s", supervision.value)
    log.info("  Good-test split:   %.0f%%", good_test_split * 100)
    log.info("  Defect-train split:%.0f%%", defect_train_split * 100)
    log.info("  Mask ratio:        %.2f", mask_ratio)
    log.info("  AMP:               %s", config.get("use_amp", False))

    seed_everything(config["seed"], workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = SuperSimpleNet(image_size=config["image_size"], config=config)

    datamodule = Custom(
        root=custom_root,
        image_size=config["image_size"],
        supervision=supervision,
        train_batch_size=config["batch"],
        eval_batch_size=config["batch"],
        num_workers=config["num_workers"],
        seed=config["seed"],
        good_test_split=good_test_split,
        defect_train_split=defect_train_split,
        mask_ratio=mask_ratio,
        flips=config.get("flips", False),
    )
    datamodule.setup()

    results = train_and_eval(model=model, datamodule=datamodule, config=config, device=device, exp=exp)

    results_writer.add_result(category=category_name, last=results)
    # include run_name so concurrent runs sharing a results dir (e.g. one per
    # supervision level) don't overwrite each other's aggregated csv
    agg_subdir = config.get("run_name") or config["dataset"]
    results_writer.save(
        Path(config["results_save_path"]) / "aggregated" / _sanitize(agg_subdir)
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.dataset == "custom":
        if args.data_root is None:
            print(
                "ERROR: --data-root is required when --dataset custom.\n"
                "  Pass the path to your dataset on the command line, e.g.:\n"
                "    python train.py @configs/custom_fully_sup.txt --data-root /path/to/your/data",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.data_root.exists():
            print(
                f"ERROR: --data-root path does not exist: {args.data_root}\n"
                "  Check the path and try again.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.data_root.is_dir():
            print(
                f"ERROR: --data-root is not a directory: {args.data_root}",
                file=sys.stderr,
            )
            sys.exit(1)
        for expected in ("train", "test"):
            if not (args.data_root / expected).exists():
                print(
                    f"WARNING: --data-root is missing expected subfolder '{expected}/': {args.data_root}",
                    file=sys.stderr,
                )

    config = _build_config(args)
    config["dataset"] = args.dataset

    log = setup_logger()  # console only; file handler attached inside train_and_eval

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    log.info("")
    log.info("╔═══════════════════════════════════════════════════════════╗")
    log.info("║          SuperSimpleNet — Training Run                    ║")
    log.info("╚═══════════════════════════════════════════════════════════╝")
    log.info("  Dataset:        %s  (%s mode)", args.dataset, args.mode)
    log.info("  Image size:     %d x %d", config["image_size"][0], config["image_size"][1])
    log.info("  Epochs:         %d  (eval every %d)", config["epochs"], config["eval_step_size"])
    log.info("  Batch size:     %d", config["batch"])
    log.info("  Seed:           %d", config["seed"])
    log.info("  Device:         %s (%s)", device.upper(), gpu_name)
    log.info("  Results dir:    %s/", config["results_save_path"])
    log.info("  Log file:       written inside per-experiment dir")
    log.info("")
    log.info("  Full config:")
    for k, v in config.items():
        log.info("    %-22s %s", k + ":", v)
    log.info("")

    if args.dataset == "mvtec":
        if args.perlin_thr is None:
            config["perlin_thr"] = 0.2
        config["supervision_str"] = Supervision.UNSUPERVISED.value
        main_mvtec(device=device, config=config)

    elif args.dataset == "visa":
        if args.perlin_thr is None:
            config["perlin_thr"] = 0.6
        config["supervision_str"] = Supervision.UNSUPERVISED.value
        main_visa(device=device, config=config)

    elif args.dataset == "ksdd2":
        ratio = args.ratio if args.ratio is not None else NumSegmented.N246.value
        if float(ratio) == 0 and args.perlin_thr is None:
            config["perlin_thr"] = 0.2
        config["ratio"] = ratio
        config["supervision_str"] = Supervision.MIXED_SUPERVISION.value
        main_ksdd2(
            device=device,
            config=config,
            supervision=Supervision.MIXED_SUPERVISION,
        )

    elif args.dataset == "sensum":
        ratio = args.ratio if args.ratio is not None else RatioSegmented.M100.value
        if float(ratio) == 0 and args.perlin_thr is None:
            config["perlin_thr"] = 0.2
        config["ratio"] = ratio
        config["supervision_str"] = Supervision.MIXED_SUPERVISION.value
        main_sensum(
            device=device,
            config=config,
            supervision=Supervision.MIXED_SUPERVISION,
        )

    elif args.dataset == "custom":
        _sup_map = {
            "unsupervised": Supervision.UNSUPERVISED,
            "weakly":       Supervision.WEAKLY_SUPERVISED,
            "mixed":        Supervision.MIXED_SUPERVISION,
            "fully":        Supervision.FULLY_SUPERVISED,
        }
        sup_str = args.supervision or ("unsupervised" if args.mode == "unsup" else "mixed")
        supervision = _sup_map[sup_str]
        config["supervision_str"] = supervision.value

        mask_ratio = args.ratio if args.ratio is not None else 1.0
        if not 0.0 <= mask_ratio <= 1.0:
            print(
                f"ERROR: --ratio for the custom dataset is a fraction and must be "
                f"in [0, 1], got {mask_ratio}",
                file=sys.stderr,
            )
            sys.exit(1)

        category_name = args.category or args.data_root.name
        main_custom(
            device=device,
            config=config,
            custom_root=args.data_root,
            good_test_split=args.good_test_split,
            defect_train_split=args.defect_train_split,
            category_name=category_name,
            supervision=supervision,
            mask_ratio=mask_ratio,
        )


if __name__ == "__main__":
    main()
