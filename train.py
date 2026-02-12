import os
import sys

from datamodules.base import Supervision

LOG_WANDB = False

import copy
import json
from pathlib import Path

if LOG_WANDB:
    import wandb

from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, seed_everything

from torchmetrics import AveragePrecision, Metric, F1Score, PrecisionRecallCurve
from anomalib.utils.metrics import AUROC, AUPRO

from datamodules import ksdd2, sensum
from datamodules.ksdd2 import KSDD2, NumSegmented
from datamodules.sensum import Sensum, RatioSegmented
from datamodules.mvtec import MVTec
from datamodules.visa import Visa
from datamodules.bowtie import BowTie

from model.supersimplenet import SuperSimpleNet

from common.visualizer import Visualizer
from common.results_writer import ResultsWriter
from common.loss import focal_loss


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
):
    model.to(device)
    optimizer, scheduler = model.get_optimizers()

    model.train()
    train_loader = datamodule.train_dataloader()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(
            total=len(train_loader),
            desc=str(epoch) + "/" + str(epochs),
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

                anomaly_map, score, mask, label = model.forward(
                    image_batch, mask, label
                )

                seg_focal = focal_loss(torch.sigmoid(anomaly_map), mask, reduction=None)

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

                loss = seg_loss + focal_loss(torch.sigmoid(score), label)

                loss.backward()

                if clip_grad:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                else:
                    norm = None

                optimizer.step()

                total_loss += loss.detach().cpu().item()

                output = {
                    "batch_loss": np.round(loss.data.cpu().detach().numpy(), 5),
                    "avg_loss": np.round(total_loss / (i + 1), 5),
                    "norm": norm,
                }

                prog_bar.set_postfix(**output)
                prog_bar.update(1)

            if (epoch + 1) % eval_step_size == 0:
                results = test(
                    model=model,
                    datamodule=datamodule,
                    device=device,
                    image_metrics=image_metrics,
                    pixel_metrics=pixel_metrics,
                    normalize=True,
                )
                if LOG_WANDB:
                    wandb.log({**results, **output})
            else:
                if LOG_WANDB:
                    wandb.log(output)
        scheduler.step()

    return results


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
):
    model.to(device)
    model.eval()

    # --- Metrics Setup ---
    # We add F1 metrics dynamically here to ensure they are on the correct device
    f1_fixed = F1Score(task="binary", threshold=0.5).to(device)
    
    # Setup standard metrics
    seg_image_metrics = {}
    for m_name, metric in image_metrics.items():
        metric.cpu()
        metric.reset()
        seg_image_metrics[f"seg-{m_name}"] = copy.deepcopy(metric)

    for metric in pixel_metrics.values():
        metric.cpu()
        metric.reset()

    test_loader = datamodule.test_dataloader()
    results = {
        "anomaly_map": [],
        "gt_mask": [],
        "score": [],
        "seg_score": [],
        "label": [],
        "image_path": [],
        "mask_path": [],
    }
    
    # --- Inference Loop ---
    for batch in tqdm(test_loader, position=0, leave=True):
        image_batch = batch["image"].to(device)
        anomaly_map, anomaly_score = model.forward(image_batch)

        anomaly_map = anomaly_map.detach().cpu()
        anomaly_score = anomaly_score.detach().cpu()

        results["anomaly_map"].append(torch.sigmoid(anomaly_map).detach().cpu())
        results["gt_mask"].append(batch["mask"].detach().cpu())

        # Scores are kept as probabilities/logits for metric calculation
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

    # --- Normalization ---
    if normalize:
        min_score = results["score"].min()
        max_score = results["score"].max()
        # Prevent division by zero
        if max_score != min_score:
            results["score"] = (results["score"] - min_score) / (max_score - min_score)
        
        min_seg = results["seg_score"].min()
        max_seg = results["seg_score"].max()
        if max_seg != min_seg:
            results["seg_score"] = (results["seg_score"] - min_seg) / (max_seg - min_seg)

        # Normalize maps (optional, mostly for viz)
        results["anomaly_map"] = (results["anomaly_map"] - results["anomaly_map"].min()) / \
                                 (results["anomaly_map"].max() - results["anomaly_map"].min() + 1e-8)

    results_dict = {}

    # --- Compute Standard Metrics ---
    for name, metric in image_metrics.items():
        metric.update(results["score"], results["label"])
        results_dict[name] = metric.to(device).compute().item()
        metric.to("cpu")

    # --- Compute F1 Scores ---
    # 1. Fixed Threshold (0.5)
    f1_fixed.update(results["score"].to(device), results["label"].to(device))
    results_dict["F1-Fixed-0.5"] = f1_fixed.compute().item()
    
    # 2. Max F1 Score (Iterate thresholds using PrecisionRecallCurve)
    pr_curve = PrecisionRecallCurve(task="binary").to(device)
    precision, recall, thresholds = pr_curve(results["score"].to(device), results["label"].to(device))
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = torch.argmax(f1_scores)
    results_dict["F1-Max"] = f1_scores[best_f1_idx].item()
    results_dict["Best-Threshold"] = thresholds[best_f1_idx].item() if best_f1_idx < len(thresholds) else 0.5

    # --- Compute Pixel Metrics (Skip if no masks/weak supervision generally implies bad masks) ---
    # Since you have no ground truth masks, these metrics (P-AUROC, etc) will be meaningless (likely 0.5 or error).
    # We keep them to prevent crashes but wrap in try-catch.
    for name, metric in pixel_metrics.items():
        try:
            metric.update(results["anomaly_map"], results["gt_mask"].type(torch.float32))
            results_dict[name] = metric.to(device).compute().item()
        except Exception:
            results_dict[name] = 0.0
        metric.to("cpu")

    # Print Results
    print("\n--- Test Results ---")
    for name, value in results_dict.items():
        print(f"{name}: {value:.4f}")
    print("--------------------")

    # --- Saving / Visualization ---
    if image_save_path:
        print(f"Visualizing to {image_save_path}")
        visualizer = Visualizer(image_save_path)
        visualizer.visualize(results)

    return results_dict


def train_and_eval(model, datamodule, config, device):
    if LOG_WANDB:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.init(project=config["wandb_project"], config=config, name=config["name"])

    image_metrics = {
        "I-AUROC": AUROC(),
        "AP-det": AveragePrecision(num_classes=1),
    }
    pixel_metrics = {
        "P-AUROC": AUROC(),
        "AUPRO": AUPRO(),
        "AP-loc": AveragePrecision(num_classes=1),
    }

    train(
        model=model,
        epochs=config["epochs"],
        datamodule=datamodule,
        device=device,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        clip_grad=config["clip_grad"],
        eval_step_size=config["eval_step_size"],
    )
    if LOG_WANDB:
        wandb.finish()

    try:
        model.save_model(
            Path(config["results_save_path"])
            / config["setup_name"]
            / "checkpoints"
            / config["dataset"]
            / config["category"]
            / str(config["ratio"]),
        )
    except Exception as e:
        print("Error saving checkpoint" + str(e))

    results = test(
        model=model,
        datamodule=datamodule,
        device=device,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        normalize=True,
        image_save_path=Path(config["results_save_path"])
        / config["setup_name"]
        / "visual"
        / config["dataset"]
        / config["category"]
        / str(config["ratio"]),
        score_save_path=Path(config["results_save_path"])
        / config["setup_name"]
        / "scores"
        / config["dataset"]
        / config["category"]
        / str(config["ratio"]),
    )

    return results


def main_mvtec(device, config):
    config = copy.deepcopy(config)
    config["dataset"] = "mvtec"
    config["ratio"] = 1

    categories = [
        "screw",
        "pill",
        "capsule",
        "carpet",
        "grid",
        "tile",
        "wood",
        "zipper",
        "cable",
        "toothbrush",
        "transistor",
        "metal_nut",
        "bottle",
        "hazelnut",
        "leather",
    ]

    results_writer = ResultsWriter(
        metrics=[
            "AP-det",
            "AP-loc",
            "P-AUROC",
            "I-AUROC",
            "AUPRO",
            "seg-AP-det",
            "seg-I-AUROC",
        ]
    )

    for category in categories:
        print(f"Training on {category}")

        config["category"] = category
        config["name"] = f"{category}_{config['setup_name']}"

        # deterministic
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

        results = train_and_eval(
            model=model, datamodule=datamodule, config=config, device=device
        )

        results_writer.add_result(
            category=category,
            last=results,
        )
        results_writer.save(
            Path(config["results_save_path"])
            / config["setup_name"]
            / config["dataset"]
            / str(config["ratio"])
        )


def main_visa(device, config):
    config = copy.deepcopy(config)
    config["dataset"] = "visa"
    config["ratio"] = 1

    categories = [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ]

    results_writer = ResultsWriter(
        metrics=[
            "AP-det",
            "AP-loc",
            "P-AUROC",
            "I-AUROC",
            "AUPRO",
            "seg-AP-det",
            "seg-I-AUROC",
        ]
    )

    for category in categories:
        print(f"Training on {category}")

        config["category"] = category
        config["name"] = f"{category}_{config['setup_name']}"

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

        results = train_and_eval(
            model=model, datamodule=datamodule, config=config, device=device
        )

        results_writer.add_result(
            category=category,
            last=results,
        )
        results_writer.save(
            Path(config["results_save_path"])
            / config["setup_name"]
            / config["dataset"]
            / str(config["ratio"])
        )


def main_ksdd2(device, config, supervision):
    config = copy.deepcopy(config)
    config["dataset"] = "ksdd2"
    config["category"] = "ksdd2"
    config["name"] = f"ksdd2_{config['setup_name']}"

    results_writer = ResultsWriter(
        metrics=[
            "AP-det",
            "AP-loc",
            "P-AUROC",
            "I-AUROC",
            "AUPRO",
            "seg-AP-det",
            "seg-I-AUROC",
            "ratio",
        ]
    )

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

    results = train_and_eval(
        model=model, datamodule=datamodule, config=config, device=device
    )

    results["ratio"] = config["ratio"]
    results_writer.add_result(
        category="ksdd2",
        last=results,
    )
    results_writer.save(
        Path(config["results_save_path"])
        / config["setup_name"]
        / config["dataset"]
        / str(config["ratio"])
    )


def main_sensum(device, config, supervision):
    config = copy.deepcopy(config)
    config["dataset"] = "sensum"

    results_writer = ResultsWriter(
        metrics=[
            "AP-det",
            "AP-loc",
            "P-AUROC",
            "I-AUROC",
            "AUPRO",
            "seg-AP-det",
            "seg-I-AUROC",
            "fold",
            "ratio",
        ]
    )

    for category in [sensum.Category.Capsule, sensum.Category.Softgel]:
        print(f"Training on {category.value}")

        seed_everything(config["seed"], workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for fold_num in range(3):
            config["category"] = f"{category.value}_{fold_num}"
            config["name"] = f"{category.value}_{config['setup_name']}_{fold_num}"
            config["fold"] = fold_num

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

            results = train_and_eval(
                model=model, datamodule=datamodule, config=config, device=device
            )

            # also log fold as a separate column
            results["fold"] = fold_num
            results["ratio"] = config["ratio"]
            results_writer.add_result(
                category=f"{category.value}",
                last=results,
            )
            results_writer.save(
                Path(config["results_save_path"])
                / config["setup_name"]
                / config["dataset"]
                / str(config["ratio"])
            )

def main_bowtie(device, config, specific_category=None):
    config = copy.deepcopy(config)
    config["dataset"] = "bowtie"
    
    if specific_category:
        categories = [specific_category]
    else:
        # Fallback list if no specific category is given
        categories = ["color_profile_1", "color_profile_1_2", "color_profile_1_2_3"]

    results_writer = ResultsWriter(
        metrics=[
            "AP-det",
            "I-AUROC",
            "F1-Fixed-0.5",
            "F1-Max",
            "Best-Threshold"
        ]
    )

    for category in categories:
        print(f"Training on {category}")

        config["category"] = category
        config["name"] = f"{category}_{config['setup_name']}"

        seed_everything(config["seed"], workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = SuperSimpleNet(image_size=config["image_size"], config=config)

        # Initialize the new BowTie DataModule
        datamodule = BowTie(
            root=Path(config["datasets_folder"]) / "BowTie-New", # Point to parent of color_profile_x
            category=category,
            image_size=config["image_size"],
            train_batch_size=config["batch"],
            eval_batch_size=config["batch"],
            num_workers=config["num_workers"],
            seed=config["seed"],
            debug=False
        )
        datamodule.setup()

        results = train_and_eval(
            model=model, datamodule=datamodule, config=config, device=device
        )

        results_writer.add_result(category=category, last=results)
        results_writer.save(
            Path(config["results_save_path"])
            / config["setup_name"]
            / config["dataset"]
        )

def run_unsup(data_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "wandb_project": "ssn",
        "datasets_folder": Path("./datasets"),
        "num_workers": 8,
        "setup_name": "superSimpleNet",
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise": True,
        "perlin": True,
        "no_anomaly": "empty",
        "bad": True,
        "overlap": True,  # makes no difference, just faster if false to avoid computation
        "adapt_cls_feat": False,  # (JIMS extension) cls features are not adapted
        "noise_std": 0.015,
        # "perlin_thr": x,
        "image_size": (256, 256),
        "seed": 42,
        "batch": 32,
        "epochs": 300,
        "flips": False,  # makes no difference, just faster if false to avoid computation
        "seg_lr": 0.0002,
        "dec_lr": 0.0002,
        "adapt_lr": 0.0001,
        "gamma": 0.4,
        "stop_grad": True,
        "clip_grad": False,
        "eval_step_size": 4,
        "results_save_path": Path("./results"),
    }
    if data_name == "visa":
        config["perlin_thr"] = 0.6
        main_visa(device=device, config=config)
    if data_name == "mvtec":
        config["perlin_thr"] = 0.2
        main_mvtec(device=device, config=config)


def run_sup(data_name, category=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "wandb_project": "ssn",
        "datasets_folder": Path("../data"),
        "num_workers": 1,
        "setup_name": "superSimpleNet",
        "dt": (3, 2),   # distance transform
        "dilate": 7,    # dilate mask
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise": True,
        "perlin": True,
        "no_anomaly": "empty",
        "bad": True,
        "overlap": False,
        "adapt_cls_feat": False,  # (JIMS extension) cls features are not adapted
        "noise_std": 0.015,
        "perlin_thr": 0.6,
        "seed": 456654,
        "batch": 32,
        "epochs": 500,
        "flips": True,
        "seg_lr": 0.0002,
        "dec_lr": 0.0002,
        "adapt_lr": 0.0001,
        "gamma": 0.4,
        "stop_grad": False,
        "clip_grad": True,
        "eval_step_size": 4,
        "results_save_path": Path("./results"),
        "image_size": (256, 256),
        "ratio": "weak_supervision",
    }
    
    if data_name == "bowtie":
        main_bowtie(device=device, config=config, specific_category=category)
    
    if data_name == "sensum":
        config["ratio"] = RatioSegmented.M100.value

        if float(config["ratio"]) == 0:
            config["perlin_thr"] = 0.2
        main_sensum(
            device=device, config=config, supervision=Supervision.MIXED_SUPERVISION
        )
    if data_name == "ksdd2":
        config["ratio"] = NumSegmented.N246.value

        if float(config["ratio"]) == 0:
            config["perlin_thr"] = 0.2
        main_ksdd2(
            device=device, config=config, supervision=Supervision.MIXED_SUPERVISION
        )


def main():
    data_name = sys.argv[1]
    category_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    if data_name != "bowtie":
        run_unsup(data_name)
        
    run_sup(data_name, category_name)


if __name__ == "__main__":
    main()
