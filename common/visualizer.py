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

            anomaly_map = anomaly_map.squeeze()
            h, w = anomaly_map.shape
            image = read_image(image_path, image_size=(h, w))
            if mask_path:
                gt_mask = read_image(mask_path, image_size=(h, w)).squeeze()
            else:
                gt_mask = np.zeros_like(anomaly_map)
            pred_mask = anomaly_map >= 0.5

            gt_label_str = "Anomalous" if label.item() else "Normal"

            fig_h = h / 256 * 2
            fig_w = w / 256 * 12

            fig, plots = plt.subplots(1, 5, figsize=(fig_w, fig_h))
            for s_plt in plots:
                s_plt.axis("off")

            fig.tight_layout()

            plots[0].imshow(image)
            plots[0].title.set_text("Image")

            plots[1].imshow(gt_mask)
            plots[1].title.set_text(f"Ground truth.\n{gt_label_str}")

            plots[2].imshow(pred_mask)
            plots[2].title.set_text("Predicted mask")

            plots[3].imshow(anomaly_map)
            plots[3].title.set_text("Anomaly map\nNorm.")

            plots[4].imshow(anomaly_map, vmax=1, vmin=0)
            plots[4].title.set_text(
                f"Anomaly map.\nScore: {round(score.item(), 4)}\nSScore: {round(seg_score.item(), 4)}"
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
            cv2.imwrite(str(ano_maps_dir / plot_name), anomaly_map.numpy() * 255)

            plt.close("all")
