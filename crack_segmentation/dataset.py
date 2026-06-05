"""Crack segmentation dataset for FisherAdapTune examples.

Expected directory layout:
    <data_root>/
        images/   *.jpg / *.png / *.tif
        masks/    *.jpg / *.png / *.tif   (binary, same stem as images)

Two dataset variants are provided:
  - SAM2SegmentationDataset   → (image_tensor C×H×W, mask_tensor 1×H×W, bbox float[4])
  - SegFormerSegmentationDataset → (image_tensor C×H×W, mask_tensor 1×H×W)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _list_files(directory: Path):
    return sorted(
        f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )



# ---------------------------------------------------------------------------
# SAM2 dataset
# ---------------------------------------------------------------------------


def sam2_default_transforms(image_size: int = 256):
    """Default transforms for SAM2 fine-tuning.

    Images are resized and converted to float [0, 1].
    Masks are resized to the same size and kept as binary float.
    The bbox is computed at ``image_size`` resolution.
    """
    image_tf = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    mask_tf = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=False),
        ]
    )
    return image_tf, mask_tf


class SAM2SegmentationDataset(VisionDataset):
    """Dataset for SAM2 fine-tuning.

    Returns:
        image  : FloatTensor [3, H, W]  — pixel values in [0, 1]
        mask   : FloatTensor [1, H, W]  — binary {0, 1}
        bbox   : FloatTensor [4]        — [0, 0, W, H] covering the full image
    """

    def __init__(
        self,
        data_root: str,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        bbox_size: int = 256,
    ) -> None:
        super().__init__(data_root)
        self.bbox_size = bbox_size
        root = Path(data_root)
        self.images_files = _list_files(root / "images")
        self.gt_files = _list_files(root / "masks")
        if len(self.images_files) != len(self.gt_files):
            raise ValueError(
                f"Image/mask count mismatch: {len(self.images_files)} images vs "
                f"{len(self.gt_files)} masks in {data_root}"
            )
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Defaults if none provided
        if image_transform is None and mask_transform is None:
            image_transform, mask_transform = sam2_default_transforms(bbox_size)
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.gt_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = Image.open(self.images_files[index]).convert("RGB")
        mask = Image.open(self.gt_files[index]).convert("1")

        W, H = image.size
        bbox = np.array([0, 0, W, H], dtype=np.float32)

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask, torch.from_numpy(bbox)


# ---------------------------------------------------------------------------
# SegFormer dataset
# ---------------------------------------------------------------------------

_CRACK_MEAN = (0.511, 0.507, 0.496)
_CRACK_STD = (0.126, 0.126, 0.126)


def segformer_default_transforms(image_size: int = 256):
    """Default transforms for SegFormer fine-tuning on crack images."""
    image_tf = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(_CRACK_MEAN, _CRACK_STD),
        ]
    )
    mask_tf = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=False),
        ]
    )
    return image_tf, mask_tf


class SegFormerSegmentationDataset(VisionDataset):
    """Dataset for SegFormer fine-tuning.

    Returns:
        image : FloatTensor [3, H, W]  — normalised
        mask  : FloatTensor [1, H, W]  — binary {0, 1}
    """

    def __init__(
        self,
        data_root: str,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        image_size: int = 256,
    ) -> None:
        super().__init__(data_root)
        root = Path(data_root)
        self.images_files = _list_files(root / "images")
        self.gt_files = _list_files(root / "masks")
        if len(self.images_files) != len(self.gt_files):
            raise ValueError(
                f"Image/mask count mismatch: {len(self.images_files)} images vs "
                f"{len(self.gt_files)} masks in {data_root}"
            )
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if image_transform is None and mask_transform is None:
            image_transform, mask_transform = segformer_default_transforms(image_size)
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.gt_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images_files[index]).convert("RGB")
        mask = Image.open(self.gt_files[index]).convert("1")

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask
