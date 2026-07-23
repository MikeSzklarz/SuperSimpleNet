from pathlib import Path

import cv2
import numpy as np
from anomalib.data.utils import read_image
from matplotlib import pyplot as plt

from tqdm import tqdm


_OUTCOME_LABEL = {
    (1, 1): "TP",
    (0, 1): "FP",
    (0, 0): "TN",
    (1, 0): "FN",
}

# Ground-truth and predicted-mask overlay colors. Chosen to (a) pop against
# largely desaturated/metallic photo backgrounds, (b) stay maximally distinct
# from each other for colorblind readers (CVD ΔE ~30-37, validated with the
# dataviz skill's palette checker), and (c) never appear in the `turbo`
# colormap used for the anomaly-map panel, so a color always means one thing.
_GT_COLOR = (0.878, 0.129, 0.541)   # magenta  #e0218a
_PRED_COLOR = (0.0, 0.761, 1.0)     # cyan     #00c2ff

_TURBO = plt.get_cmap("turbo")
_TITLE_FONTSIZE = 11
_SUPTITLE_FONTSIZE = 13
# Fixed inches reserved above the images for the stacked suptitle + column
# titles. Kept as an absolute size (not a fraction of figure height) so it
# stays legible even for very wide/short images, where a fraction-of-height
# margin would shrink to almost nothing.
_TITLE_AREA_IN = (_SUPTITLE_FONTSIZE + _TITLE_FONTSIZE) * 1.3 / 72 + 0.15


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple, fill_alpha: float = 0.35, outline_px: int = 2) -> np.ndarray:
    """Composite a boolean mask onto a float RGB image ([0,1]) as a
    semi-transparent fill plus a solid outline, so the region is visible
    without hiding the image underneath and its boundary always pops.
    """
    color_arr = np.array(color, dtype=np.float32)
    out = image.copy()
    out[mask] = out[mask] * (1 - fill_alpha) + color_arr * fill_alpha

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        outline = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(outline, contours, -1, 1, thickness=outline_px)
        out[outline.astype(bool)] = color_arr

    return out


def _overlay_heatmap(image: np.ndarray, anomaly_map: np.ndarray, max_alpha: float = 0.75) -> np.ndarray:
    """Composite the anomaly map onto a float RGB image ([0,1]) using the
    `turbo` colormap, with color and per-pixel alpha driven by the map after
    a per-image min/max stretch rather than the raw score. Raw anomaly
    scores rarely approach 1.0 even at genuine defects, so gating on the raw
    value left the overlay faint everywhere, hotspot included; stretching to
    the image's own range guarantees the hottest pixel always reads as fully
    saturated/opaque while low-relevance regions stay close to the photo.
    """
    lo, hi = anomaly_map.min(), anomaly_map.max()
    normalized = (anomaly_map - lo) / (hi - lo) if hi > lo else np.zeros_like(anomaly_map)
    heat_rgba = _TURBO(normalized).astype(np.float32)
    alpha = (normalized * max_alpha)[..., None]
    return image * (1 - alpha) + heat_rgba[..., :3] * alpha


class Visualizer:
    def __init__(self, save_path: Path):
        self.save_path = save_path

    def visualize(self, results: dict, outcome_dirs: dict = None):
        """Visualize all test images.

        When outcome_dirs is provided (mapping "TP"/"FP"/"TN"/"FN" → Path),
        images are routed by classification outcome then defect type, so it's
        immediately clear whether the model was right or wrong.  Without it,
        the legacy {defect_type}/{gt_label} layout is used.
        """
        predicted_labels = results.get("predicted_label")

        for item in tqdm(
            zip(
                results["image_path"],
                results["mask_path"],
                results["anomaly_map"],
                results["score"],
                results["seg_score"],
                results["label"],
                predicted_labels if predicted_labels is not None
                    else [None] * len(results["label"]),
            ),
            total=len(results["label"]),
        ):
            image_path, mask_path, anomaly_map, score, seg_score, label, pred_label = item

            anomaly_map = anomaly_map.squeeze().numpy()
            h, w = anomaly_map.shape
            image = read_image(image_path, image_size=(h, w)).astype(np.float32) / 255.0

            gt_label_str = "Anomalous" if label.item() else "Normal"
            gt_mask = None
            if mask_path:
                raw_mask = read_image(mask_path, image_size=(h, w))
                gt_mask = raw_mask[..., 0] > 127

            pred_mask = anomaly_map >= 0.5

            n_cols = 4
            titles = ["Image", f"Ground truth: {gt_label_str}", "Anomaly map", "Predicted mask"]

            image_h = h / 256 * 2
            # Each subplot cell must have the same width:height ratio as the
            # image itself, otherwise imshow (which preserves aspect ratio)
            # letterboxes the image inside its cell and leaves whitespace
            # between columns even with wspace=0.
            panel_w = image_h * (w / h)

            # With zero column padding, small images can produce panels too
            # narrow for their titles, causing adjacent titles to overlap.
            # Scale the whole figure up (preserving per-panel aspect ratio,
            # so columns stay gap-free) so the longest title always fits.
            min_panel_w = max(len(t) for t in titles) * _TITLE_FONTSIZE * 0.6 / 72
            scale = max(1.0, min_panel_w / panel_w)
            image_h *= scale
            panel_w *= scale

            fig_w = n_cols * panel_w
            # The title area is added on top of (not carved out of) the image
            # height, so it stays a fixed absolute size regardless of aspect
            # ratio instead of shrinking away for wide/short images.
            fig_h = image_h + _TITLE_AREA_IN
            top = image_h / fig_h

            fig, plots = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
            for s_plt in plots:
                s_plt.axis("off")

            fig.subplots_adjust(left=0, right=1, bottom=0, top=top, wspace=0)

            plots[0].imshow(image)
            plots[0].set_title(titles[0], fontsize=_TITLE_FONTSIZE)

            if gt_mask is not None and gt_mask.any():
                plots[1].imshow(_overlay_mask(image, gt_mask, _GT_COLOR))
            else:
                plots[1].imshow(image)
            plots[1].set_title(titles[1], fontsize=_TITLE_FONTSIZE)

            plots[2].imshow(_overlay_heatmap(image, anomaly_map))
            plots[2].set_title(titles[2], fontsize=_TITLE_FONTSIZE)

            plots[3].imshow(_overlay_mask(image, pred_mask, _PRED_COLOR))
            plots[3].set_title(titles[3], fontsize=_TITLE_FONTSIZE)

            fig.suptitle(
                f"{Path(image_path).name}   —   GT: {gt_label_str}   "
                f"Score: {round(score.item(), 4)}   SScore: {round(seg_score.item(), 4)}",
                y=0.98,
                fontsize=_SUPTITLE_FONTSIZE,
            )

            defect_type = Path(image_path).parent.name
            plot_name = f"{Path(image_path).stem}.png"

            if outcome_dirs is not None and pred_label is not None:
                outcome = _OUTCOME_LABEL[(label.item(), pred_label.item())]
                dest_dir = outcome_dirs[outcome] / defect_type
            else:
                dest_dir = self.save_path / defect_type / gt_label_str

            dest_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(dest_dir / plot_name, bbox_inches="tight")

            ano_maps_dir = dest_dir / "anomaly_maps"
            ano_maps_dir.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(ano_maps_dir / plot_name), anomaly_map * 255)

            plt.close("all")

    def save_aggregate_heatmap(self, results: dict, save_path: Path) -> None:
        """Average every predicted anomaly map in the test set (all / normal-only
        / anomalous-only) into one image per group.

        A hotspot that survives averaging over many unrelated images is a
        fixed spatial bias, not something driven by image content -- this is
        the fastest way to see that at a glance, especially for the
        normal-only panel (nothing there should be "anomalous" at all).
        """
        anomaly_maps = results["anomaly_map"].squeeze(1).numpy()  # [N, H, W]
        labels = results["label"].numpy()

        groups = [
            ("All test images", anomaly_maps),
            ("Normal only", anomaly_maps[labels == 0]),
            ("Anomalous only", anomaly_maps[labels == 1]),
        ]

        fig, plots = plt.subplots(1, len(groups), figsize=(4.5 * len(groups), 4.5))
        for ax, (title, maps) in zip(plots, groups):
            ax.axis("off")
            if len(maps) == 0:
                ax.set_title(f"{title}\n(no images)", fontsize=_TITLE_FONTSIZE)
                continue
            mean_map = maps.mean(axis=0)
            im = ax.imshow(mean_map, cmap=_TURBO, vmin=0, vmax=max(mean_map.max(), 1e-6))
            ax.set_title(f"{title}  (n={len(maps)})", fontsize=_TITLE_FONTSIZE)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(
            "Mean predicted anomaly map across test set\n"
            "(a hotspot that survives averaging = fixed-location bias, not content-driven)",
            fontsize=_SUPTITLE_FONTSIZE,
        )
        fig.tight_layout()

        save_path.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path / "_aggregate_heatmap.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
