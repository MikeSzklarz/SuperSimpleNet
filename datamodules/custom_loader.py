import random
from pathlib import Path
from pandas import DataFrame
import albumentations as A
from anomalib.data.utils import Split, LabelName, InputNormalizationMethod

from datamodules.base import Supervision
from datamodules.base.datamodule import SSNDataModule
from datamodules.base.dataset import SSNDataset

# Image extensions to search for
IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

class CustomDataset(SSNDataset):
    def __init__(
        self,
        root: Path,
        supervision: Supervision,
        transform: A.Compose,
        split: Split,
        flips: bool,
        normal_flips: bool,
        test_split_ratio: float = 0.2, # Default 20%
        seed: int = 42,                # Seed for deterministic shuffling
        dilate: int | None = None,
        dt: tuple[int, int] | None = None,
        debug: bool = False,
    ) -> None:
        self.test_split_ratio = test_split_ratio
        self.seed = seed
        super().__init__(
            transform=transform,
            root=root,
            split=split,
            flips=flips,
            normal_flips=normal_flips,
            supervision=supervision,
            dilate=dilate,
            dt=dt,
            debug=debug,
        )

    def _get_all_images(self, path: Path):
        """Recursively get all images in a directory."""
        images = []
        if not path.exists():
            return images
        for ext in IMG_EXTENSIONS:
            images.extend(list(path.rglob(f"*{ext}")))
            images.extend(list(path.rglob(f"*{ext.upper()}")))
        return [str(p) for p in images]

    def make_dataset(self) -> tuple[DataFrame, DataFrame]:
        """
        Dynamically splits 'train/good' into Train/Test using a random shuffle.
        """
        train_good_dir = self.root / "train" / "good"
        contaminated_dir = self.root / "contaminated"
        test_dir = self.root / "test"

        normal_list = []
        anomalous_list = []

        # --- DYNAMIC RANDOM SPLIT LOGIC ---
        # 1. Get ALL good training images
        all_good_images = self._get_all_images(train_good_dir)
        
        # 2. Sort then Shuffle (Deterministic based on seed)
        # Sorting first ensures the starting order is always the same OS-independent
        all_good_images = sorted(all_good_images)
        rng = random.Random(self.seed)
        rng.shuffle(all_good_images)

        # 3. Calculate Split Index
        num_total = len(all_good_images)
        num_test = int(num_total * self.test_split_ratio)
        
        # Safety: Ensure we don't end up with 0 test images if ratio > 0
        if num_test == 0 and num_total > 1 and self.test_split_ratio > 0:
            num_test = 1
            
        # 4. Slice the lists
        # Test gets the first chunk, Train gets the rest
        test_good_split = all_good_images[:num_test]
        train_good_split = all_good_images[num_test:]

        # --- LOGGING THE SPLIT ---
        # This will print when the dataset is initialized
        dataset_type = "TRAIN" if self.split == Split.TRAIN else "TEST"
        if self.split == Split.TRAIN:
            count = len(train_good_split)
            print(f"[{dataset_type} SPLIT] Using {count} Normal images ({100 - self.test_split_ratio*100:.1f}%)")
        else:
            count = len(test_good_split)
            print(f"[{dataset_type} SPLIT] Using {count} Normal images (Holdout {self.test_split_ratio*100:.1f}%)")

        # --- FILL DATAFRAMES ---
        if self.split == Split.TRAIN:
            # A. Normal Training Data
            for img in train_good_split:
                normal_list.append([str(self.root), img, "", LabelName.NORMAL, True])

            # B. Contaminated Data
            if self.supervision != Supervision.UNSUPERVISED:
                bad_images = self._get_all_images(contaminated_dir)
                for img in bad_images:
                    anomalous_list.append([str(self.root), img, "", LabelName.ABNORMAL, False])
                print(f"[{dataset_type} SPLIT] Using {len(bad_images)} Contaminated images")

        else: # TEST SPLIT
            # A. Holdout Normal Data
            for img in test_good_split:
                normal_list.append([str(self.root), img, "", LabelName.NORMAL, True])

            # B. Standard Test Defects
            if test_dir.exists():
                count_defects = 0
                for subfolder in test_dir.iterdir():
                    if not subfolder.is_dir(): continue
                    imgs = self._get_all_images(subfolder)
                    
                    if subfolder.name == "good":
                        for img in imgs:
                            normal_list.append([str(self.root), img, "", LabelName.NORMAL, True])
                    else:
                        count_defects += len(imgs)
                        for img in imgs:
                            anomalous_list.append([str(self.root), img, "", LabelName.ABNORMAL, False])
                print(f"[{dataset_type} SPLIT] Using {count_defects} Known Defect images")

        columns = ["path", "image_path", "mask_path", "label_index", "is_segmented"]
        return DataFrame(normal_list, columns=columns), DataFrame(anomalous_list, columns=columns)


class CustomMixed(SSNDataModule):
    def __init__(
        self,
        root: Path | str,
        supervision: Supervision,
        image_size: tuple[int, int] | None = None,
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        seed: int | None = None,
        flips: bool = False,
        normal_flips: bool = False,
        dilate: int | None = None,
        dt: tuple[int, int] | None = None,
        debug: bool = False,
        test_split_ratio: float = 0.2, # New Parameter
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

        self.train_data = CustomDataset(
            transform=self.transform_train,
            split=Split.TRAIN,
            root=self.root,
            supervision=supervision,
            flips=flips,
            normal_flips=normal_flips,
            dilate=dilate,
            dt=dt,
            debug=debug,
            test_split_ratio=test_split_ratio,
            seed=seed if seed else 42,
        )
        
        self.test_data = CustomDataset(
            transform=self.transform_eval,
            split=Split.TEST,
            root=self.root,
            supervision=Supervision.UNSUPERVISED, 
            flips=False,
            normal_flips=False,
            debug=debug,
            test_split_ratio=test_split_ratio,
            seed=seed if seed else 42,
        )