from pathlib import Path
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from anomalib.data.utils import Split, InputNormalizationMethod
import albumentations as A

from datamodules.base import Supervision
from datamodules.base.datamodule import SSNDataModule
from datamodules.base.dataset import SSNDataset

# Supported image extensions
IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

class BowTieDataset(SSNDataset):
    def __init__(
        self,
        root: Path,
        category: str,
        transform: A.Compose,
        split: Split,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        self.split_seed = seed if seed is not None else 42
        self.category = category
        
        # We force Weak Supervision logic here
        super().__init__(
            transform=transform,
            root=root,
            split=split,
            flips=True, # Enable flips for training data augmentation
            normal_flips=True,
            supervision=Supervision.WEAKLY_SUPERVISED,
            debug=debug,
        )

    def make_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scans the directory, splits defects into train/test, and handles 'good' data.
        Returns: (normal_samples, anomalous_samples) for the CURRENT split.
        """
        root_cat = Path(self.root) / self.category
        
        # 1. Gather all Good Images
        train_good_path = root_cat / "train" / "good"
        test_good_path = root_cat / "test" / "good"
        
        all_good_files = []
        if train_good_path.exists():
            all_good_files.extend([p for p in train_good_path.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS])
        if test_good_path.exists():
            all_good_files.extend([p for p in test_good_path.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS])
            
        all_good_files = sorted(list(set(all_good_files))) # Unique and sorted for reproducibility
        
        # 2. Gather all Defect Images (anything in test/ that is NOT 'good')
        test_root = root_cat / "test"
        all_defect_files = []
        if test_root.exists():
            for p in test_root.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
                    # Exclude files in a 'good' folder if it exists
                    if "good" not in p.parts:
                        all_defect_files.append(p)
                        
        all_defect_files = sorted(list(set(all_defect_files)))

        # 3. Perform Splits (Fixed Seed)
        # Split Good: 80% Train, 20% Test (unless explicit test/good structure was preferred, but this guarantees metrics work)
        train_good, test_good = train_test_split(all_good_files, test_size=0.2, random_state=self.split_seed, shuffle=True)
        
        # Split Defects: 50% Train (Supervised), 50% Test
        if len(all_defect_files) > 0:
            train_defects, test_defects = train_test_split(all_defect_files, test_size=0.5, random_state=self.split_seed, shuffle=True)
        else:
            train_defects, test_defects = [], []

        # 4. Construct DataFrames based on requested split
        if self.split == Split.TRAIN:
            files_normal = train_good
            files_anom = train_defects
        else:
            files_normal = test_good
            files_anom = test_defects

        # Create Normal DataFrame
        normal_df = pd.DataFrame()
        if files_normal:
            normal_df = pd.DataFrame({
                "image_path": [str(p) for p in files_normal],
                "label_index": 0,
                "mask_path": "",  # No masks
                "is_segmented": False 
            })

        # Create Anomalous DataFrame
        anom_df = pd.DataFrame()
        if files_anom:
            anom_df = pd.DataFrame({
                "image_path": [str(p) for p in files_anom],
                "label_index": 1,
                "mask_path": "", # No masks
                "is_segmented": False 
            })

        return normal_df, anom_df


class BowTie(SSNDataModule):
    def __init__(
        self,
        root: Path | str,
        category: str,
        image_size: tuple[int, int] | None = (256, 256),
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            supervision=Supervision.WEAKLY_SUPERVISED, # Enforce Weak Supervision
            image_size=image_size,
            normalization=normalization,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
            flips=True, 
        )

        self.train_data = BowTieDataset(
            root=Path(root),
            category=category,
            transform=self.transform_train,
            split=Split.TRAIN,
            seed=seed,
            debug=debug,
        )
        self.test_data = BowTieDataset(
            root=Path(root),
            category=category,
            transform=self.transform_eval,
            split=Split.TEST,
            seed=seed,
            debug=debug,
        )