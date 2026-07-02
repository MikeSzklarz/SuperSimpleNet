"""Custom dataset datamodule for MVTecAD-like folder layouts.

Expected structure
------------------
root/
├── train/
│   └── good/        (all normal training images)
├── test/
│   └── <defect>/    (defective images, one sub-dir per defect type)
└── ground_truth/
    └── <defect>/    (PNG masks; same stem as the defect image, no _mask suffix)

No manual file moving required for any supervision level.

How splits work
---------------
Good images (train/good/)
    Shuffled once with `seed`, then split:
      train  — first (1 - good_test_split) fraction
      test   — remaining good_test_split fraction

Defect images (test/<defect>/)
    Shuffled once per category (seed + category_index), then split:
      train  — first defect_train_split fraction  (used for supervised training)
      test   — remaining fraction                  (always held out for evaluation)

Supervision modes
-----------------
unsupervised  — defect_train_split must be 0; no anomalous training samples
weakly/mixed  — defect_train_split > 0; masks used when available
fully         — defect_train_split > 0; samples without masks are skipped for training
"""

import logging
from pathlib import Path

import albumentations as A
import numpy as np
from anomalib.data.utils import InputNormalizationMethod, Split
from pandas import DataFrame

from datamodules.base import Supervision
from datamodules.base.datamodule import SSNDataModule
from datamodules.base.dataset import SSNDataset

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

logger = logging.getLogger("ssn")

_SUPERVISED_MODES = {
    Supervision.WEAKLY_SUPERVISED,
    Supervision.MIXED_SUPERVISION,
    Supervision.FULLY_SUPERVISED,
}


class CustomDataset(SSNDataset):
    """Dataset for a single custom category.

    Args:
        root: path to dataset root (train/, test/, ground_truth/)
        supervision: supervision level
        transform: albumentations transforms
        split: Split.TRAIN or Split.TEST
        seed: RNG seed for reproducible splits
        good_test_split: fraction of train/good images withheld for test
        defect_train_split: fraction of test/<defect>/ images used for supervised training
        flips: augment anomalous training samples with flips
        normal_flips: apply random flips to normal training samples
    """

    def __init__(
        self,
        root: Path,
        supervision: Supervision,
        transform: A.Compose,
        split: Split,
        seed: int = 42,
        good_test_split: float = 0.2,
        defect_train_split: float = 0.0,
        flips: bool = False,
        normal_flips: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            supervision=supervision,
            transform=transform,
            split=split,
            flips=flips,
            normal_flips=normal_flips,
        )
        self.seed = seed
        self.good_test_split = good_test_split
        self.defect_train_split = defect_train_split

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _scan_images(self, directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        return sorted(p for p in directory.iterdir() if p.suffix in _IMAGE_EXTENSIONS)

    def _split_good_images(self, images: list[Path]) -> tuple[list[Path], list[Path]]:
        """Seeded shuffle → (train_good, test_good)."""
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(images))
        rng.shuffle(idx)
        n_train = max(1, int(len(images) * (1.0 - self.good_test_split)))
        train_idx = sorted(idx[:n_train])
        test_idx = sorted(idx[n_train:])
        return [images[i] for i in train_idx], [images[i] for i in test_idx]

    def _split_defect_images(
        self, images: list[Path], category_idx: int
    ) -> tuple[list[Path], list[Path]]:
        """Seeded per-category shuffle → (train_defects, test_defects).

        Uses seed + category_idx + 1 so each defect category gets its own
        independent shuffle while remaining reproducible across TRAIN/TEST calls.
        """
        if self.defect_train_split <= 0.0 or not images:
            return [], list(images)
        rng = np.random.default_rng(self.seed + category_idx + 1)
        idx = np.arange(len(images))
        rng.shuffle(idx)
        n_train = int(len(images) * self.defect_train_split)
        train_idx = sorted(idx[:n_train])
        test_idx = sorted(idx[n_train:])
        return [images[i] for i in train_idx], [images[i] for i in test_idx]

    def _make_row(
        self, img_path: Path, label: int, gt_dir: Path, defect_name: str
    ) -> dict:
        mask_path = gt_dir / defect_name / (img_path.stem + ".png")
        has_mask = mask_path.exists()
        return {
            "image_path": str(img_path),
            "mask_path": str(mask_path) if has_mask else "",
            "label_index": label,
            "is_segmented": has_mask,
        }

    # ------------------------------------------------------------------
    # make_dataset
    # ------------------------------------------------------------------

    def make_dataset(self) -> tuple[DataFrame, DataFrame]:
        train_good_dir = self.root / "train" / "good"
        test_dir = self.root / "test"
        gt_dir = self.root / "ground_truth"

        # ── Good images ──────────────────────────────────────────────────
        good_images = self._scan_images(train_good_dir)
        if not good_images:
            raise FileNotFoundError(
                f"No images found in {train_good_dir}. "
                "Ensure the dataset has a train/good/ subdirectory."
            )
        train_good, test_good = self._split_good_images(good_images)

        # ── Defect images — split per category ───────────────────────────
        # Always scan all defects; which portion goes to train vs test
        # is controlled by defect_train_split.
        defect_dirs = sorted(
            d for d in test_dir.iterdir() if d.is_dir()
        ) if test_dir.exists() else []

        train_defect_rows: list[dict] = []
        test_defect_rows: list[dict] = []
        defect_summary: dict[str, dict] = {}  # {name: {train: n, test: n}}

        require_mask = self.supervision == Supervision.FULLY_SUPERVISED

        for cat_idx, defect_dir in enumerate(defect_dirs):
            defect_name = defect_dir.name
            imgs = self._scan_images(defect_dir)
            d_train_imgs, d_test_imgs = self._split_defect_images(imgs, cat_idx)

            n_train_added = 0
            for img_path in d_train_imgs:
                row = self._make_row(img_path, 1, gt_dir, defect_name)
                if require_mask and not row["is_segmented"]:
                    logger.warning(
                        "  Skipping %s/%s for training: no mask (fully supervised)",
                        defect_name, img_path.name,
                    )
                    continue
                train_defect_rows.append(row)
                n_train_added += 1

            for img_path in d_test_imgs:
                test_defect_rows.append(self._make_row(img_path, 1, gt_dir, defect_name))

            defect_summary[defect_name] = {
                "train": n_train_added,
                "test": len(d_test_imgs),
            }

        # ── TRAIN split ──────────────────────────────────────────────────
        if self.split == Split.TRAIN:
            normal_rows = [
                {"image_path": str(p), "mask_path": "", "label_index": 0, "is_segmented": False}
                for p in train_good
            ]

            logger.info("─── Training Data Summary ───────────────────────────────────")
            logger.info("  Supervision:        %s", self.supervision.value)
            logger.info("  Root:               %s", self.root)
            logger.info("  Normal samples:     %4d  (from train/good/)", len(train_good))
            logger.info("  Withheld for test:  %4d  (good_test_split=%.0f%%)",
                        len(test_good), self.good_test_split * 100)

            if self.supervision in _SUPERVISED_MODES:
                logger.info("  defect_train_split: %.0f%%", self.defect_train_split * 100)
                logger.info("  Anomalous training samples: %4d  (from test/<defect>/)",
                            len(train_defect_rows))
                for name, counts in defect_summary.items():
                    if counts["train"] or counts["test"]:
                        logger.info("    %-30s → %2d train / %2d test",
                                    name, counts["train"], counts["test"])
                n_masked = sum(1 for r in train_defect_rows if r["is_segmented"])
                logger.info("  Masks in training anomalies: %d / %d",
                            n_masked, len(train_defect_rows))

                if not train_defect_rows:
                    raise RuntimeError(
                        f"\n\nSupervision mode '{self.supervision.value}' requires anomalous "
                        f"training images, but defect_train_split={self.defect_train_split:.0%} "
                        f"resulted in 0 training defect images.\n\n"
                        f"Fix: increase --defect-train-split (e.g. 0.5) to automatically "
                        f"reserve a fraction of test/<defect>/ images for training.\n"
                        f"Total defect images available: "
                        f"{sum(len(self._scan_images(d)) for d in defect_dirs)}"
                    )

                logger.info("─────────────────────────────────────────────────────────────")
                return DataFrame(normal_rows), DataFrame(train_defect_rows)

            # Unsupervised
            logger.info("  Anomalous samples:     0  (unsupervised — all defects go to test)")
            logger.info("─────────────────────────────────────────────────────────────")
            return DataFrame(normal_rows), DataFrame()

        # ── TEST split ───────────────────────────────────────────────────
        test_normal_rows = [
            {"image_path": str(p), "mask_path": "", "label_index": 0, "is_segmented": False}
            for p in test_good
        ]
        n_masked_test = sum(1 for r in test_defect_rows if r["is_segmented"])

        logger.info("─── Test Data Summary ───────────────────────────────────────")
        logger.info("  Root:               %s", self.root)
        logger.info("  Good images total:  %d", len(good_images))
        logger.info("  → Test (withheld):  %4d  (good_test_split=%.0f%%)",
                    len(test_good), self.good_test_split * 100)
        logger.info("  → Train:            %4d", len(train_good))
        logger.info("")
        logger.info("  Defect images (test/<defect>/):")
        for name, counts in defect_summary.items():
            logger.info("    %-30s  %2d train / %2d test",
                        name, counts["train"], counts["test"])
        logger.info("")
        logger.info("  Test split totals:")
        logger.info("    Normal (good):  %4d", len(test_normal_rows))
        logger.info("    Anomalous:      %4d  (masks: %d / %d)",
                    len(test_defect_rows), n_masked_test, len(test_defect_rows))
        logger.info("    Total:          %4d", len(test_normal_rows) + len(test_defect_rows))
        logger.info("─────────────────────────────────────────────────────────────")

        return DataFrame(test_normal_rows), DataFrame(test_defect_rows)


class Custom(SSNDataModule):
    """DataModule for a single custom category.

    Args:
        root: path to the dataset root directory
        image_size: (height, width) to resize images to
        supervision: supervision level
        train_batch_size: batch size for training
        eval_batch_size: batch size for evaluation
        num_workers: DataLoader workers (must be ≤ 1 for supervised modes)
        seed: RNG seed for reproducible splits
        good_test_split: fraction of train/good images withheld for test
        defect_train_split: fraction of test/<defect>/ images used for training
        flips: augment anomalous training samples with flips
        normalization: pixel normalization strategy
    """

    def __init__(
        self,
        root: Path | str,
        image_size: tuple[int, int],
        supervision: Supervision = Supervision.UNSUPERVISED,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        seed: int = 42,
        good_test_split: float = 0.2,
        defect_train_split: float = 0.0,
        flips: bool = False,
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
    ) -> None:
        super().__init__(
            root=root,
            supervision=supervision,
            image_size=image_size,
            normalization=normalization,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
            flips=flips,
        )

        shared_kwargs = dict(
            root=Path(root),
            supervision=supervision,
            seed=seed,
            good_test_split=good_test_split,
            defect_train_split=defect_train_split,
        )
        self.train_data = CustomDataset(
            **shared_kwargs,
            transform=self.transform_train,
            split=Split.TRAIN,
            flips=flips,
        )
        self.test_data = CustomDataset(
            **shared_kwargs,
            transform=self.transform_eval,
            split=Split.TEST,
            flips=False,  # never flip test samples
        )
