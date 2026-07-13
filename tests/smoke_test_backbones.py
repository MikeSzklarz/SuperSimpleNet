"""Smoke test for SuperSimpleNet feature-extractor backbones (CNN and ViT).

Runs entirely without a dataset: builds the model, checks feature/map shapes in
train and eval mode, runs one optimizer step (verifying the backbone stays
frozen), and round-trips a checkpoint.

Usage (from the repo root):
    python tests/smoke_test_backbones.py                     # CNN + dinov2_vits14 (fast)
    python tests/smoke_test_backbones.py dinov2_vitl14_reg   # specific backbone(s)
    python tests/smoke_test_backbones.py radio_v2.5-l --image-size 512 512

First run downloads backbone weights via torch.hub / HuggingFace hub.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.supersimplenet import SuperSimpleNet
from model.vit_feature_extractor import (
    ViTFeatureExtractor,
    expected_patch_size,
    is_vit_backbone,
)

BASE_CONFIG = {
    "layers": ["layer2", "layer3"],
    "patch_size": 3,
    "noise": True,
    "perlin": True,
    "no_anomaly": "empty",
    "bad": True,
    "overlap": False,
    "adapt_cls_feat": False,
    "noise_std": 0.015,
    "perlin_thr": 0.2,
    "stop_grad": False,
    "adapt_lr": 0.0001,
    "seg_lr": 0.0002,
    "dec_lr": 0.0002,
    "gamma": 0.4,
    "epochs": 10,
    "vit_layers": None,
    "feat_scale": 1,
    "dinov3_path": None,
    "dinov3_weights": None,
}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def run_backbone(backbone: str, image_size: tuple[int, int], **overrides):
    print(f"\n=== {backbone} @ {image_size[0]}x{image_size[1]} "
          f"{overrides if overrides else ''} ===")
    config = {**BASE_CONFIG, "backbone": backbone, **overrides}
    model = SuperSimpleNet(image_size=image_size, config=config)

    fc, fh, fw = model.feature_extractor.feature_dim
    print(f"  feature_dim: C={fc}, {fh}x{fw}")
    if is_vit_backbone(backbone):
        p = expected_patch_size(backbone)
        scale = config["feat_scale"]
        n_layers = len(config["vit_layers"]) if config["vit_layers"] else 1
        check(fh == image_size[0] // p * scale and fw == image_size[1] // p * scale,
              f"grid {fh}x{fw} matches image/{p} * feat_scale {scale}")
        check(fc % n_layers == 0, f"channels divisible by {n_layers} layer(s)")

    b = 2
    images = torch.rand(b, 3, *image_size)

    # --- train mode: forward + one optimizer step ---
    model.train()
    mask = torch.zeros(b, 1, fh, fw)
    label = torch.zeros(b)
    anomaly_map, score, out_mask, out_label = model(images, mask=mask, label=label)
    check(anomaly_map.shape == (2 * b, 1, fh, fw), f"train anomaly_map {tuple(anomaly_map.shape)}")
    check(score.shape == (2 * b,), f"train score {tuple(score.shape)}")
    check(out_mask.shape == (2 * b, 1, fh, fw), f"train mask {tuple(out_mask.shape)}")

    optim, _ = model.get_optimizers()
    loss = anomaly_map.mean() + score.mean()
    loss.backward()
    check(all(p.grad is None for p in model.feature_extractor.parameters()),
          "backbone received no gradients (frozen)")
    check(any(p.grad is not None for p in model.feature_adaptor.parameters()),
          "adaptor received gradients")
    optim.step()

    # --- eval mode ---
    model.eval()
    with torch.no_grad():
        anomaly_map, score = model(images)
    check(anomaly_map.shape == (b, 1, *image_size), f"eval anomaly_map {tuple(anomaly_map.shape)}")
    check(score.shape == (b,), f"eval score {tuple(score.shape)}")

    # --- checkpoint round-trip: no backbone weights saved ---
    with tempfile.TemporaryDirectory() as tmp:
        model.save_model(Path(tmp))
        state = torch.load(Path(tmp) / "weights.pt")
        check(not any(k.startswith("feature_extractor") for k in state),
              "checkpoint excludes backbone weights")
        model2 = SuperSimpleNet(image_size=image_size, config=config)
        model2.load_model(Path(tmp) / "weights.pt")
        check(torch.equal(
            model2.feature_adaptor.projection.weight,
            model.feature_adaptor.projection.weight,
        ), "checkpoint reload restores trained weights")

    print(f"  PASS: {backbone}")


def run_error_paths():
    print("\n=== error paths ===")
    # non-divisible image size
    try:
        ViTFeatureExtractor("dinov2_vits14", layers=None, patch_size=3,
                            image_size=(256, 256))
        raise AssertionError("divisibility check did not fire")
    except ValueError as e:
        check("divisible" in str(e), f"divisibility error raised: {e}")
    # vit_layers on a non-dino backbone
    try:
        ViTFeatureExtractor("radio_v2.5-l", layers=[3], patch_size=3,
                            image_size=(512, 512))
        raise AssertionError("vit-layers check did not fire")
    except (ValueError, RuntimeError) as e:
        check("vit-layers" in str(e).lower() or "only supported" in str(e),
              f"vit_layers restriction raised: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backbones", nargs="*",
                        default=["wide_resnet50_2", "dinov2_vits14"],
                        help="Backbones to test (default: CNN baseline + small DINOv2)")
    parser.add_argument("--image-size", type=int, nargs=2, default=None)
    parser.add_argument("--skip-errors", action="store_true",
                        help="Skip the error-path checks")
    args = parser.parse_args()

    torch.manual_seed(0)

    for backbone in args.backbones:
        if args.image_size:
            image_size = tuple(args.image_size)
        elif is_vit_backbone(backbone):
            image_size = (448, 448) if expected_patch_size(backbone) == 14 else (512, 512)
        else:
            image_size = (256, 256)
        run_backbone(backbone, image_size)
        if is_vit_backbone(backbone) and backbone.startswith("dinov2"):
            # exercise multi-layer + feat_scale on the dino path
            run_backbone(backbone, image_size, vit_layers=[8, 11], feat_scale=2)

    if not args.skip_errors:
        run_error_paths()

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
