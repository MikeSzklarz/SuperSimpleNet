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
    `turbo` colormap with per-pixel alpha driven by the score itself, so
    low-anomaly regions stay close to the original photo and only genuinely
    anomalous regions are tinted (Grad-CAM style).
    """
    clipped = np.clip(anomaly_map, 0.0, 1.0)
    heat_rgba = _TURBO(clipped).astype(np.float32)
    alpha = (clipped * max_alpha)[..., None]
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

            fig_h = h / 256 * 2
            fig_w = w / 256 * 9.6

            fig, plots = plt.subplots(1, 4, figsize=(fig_w, fig_h))
            for s_plt in plots:
                s_plt.axis("off")

            gap_frac = 5 / (fig_w * fig.dpi / 4)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=0.78, wspace=gap_frac)

            plots[0].imshow(image)
            plots[0].title.set_text("Image")

            if gt_mask is not None and gt_mask.any():
                plots[1].imshow(_overlay_mask(image, gt_mask, _GT_COLOR))
            else:
                plots[1].imshow(image)
            plots[1].title.set_text(f"Ground truth: {gt_label_str}")

            plots[2].imshow(_overlay_heatmap(image, anomaly_map))
            plots[2].title.set_text("Anomaly map")

            plots[3].imshow(_overlay_mask(image, pred_mask, _PRED_COLOR))
            plots[3].title.set_text("Predicted mask")

            fig.suptitle(
                f"{Path(image_path).name}   —   GT: {gt_label_str}   "
                f"Score: {round(score.item(), 4)}   SScore: {round(seg_score.item(), 4)}",
                y=0.98,
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
