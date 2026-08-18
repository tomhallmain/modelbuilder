"""
PyTorch data loading utilities for image classification.

This module provides data loaders and transforms for PyTorch training.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional

from mb.models.preprocessing import DEFAULT_PREPROCESSING, PreprocessingSpec

if TYPE_CHECKING:
    from mb.data.label_schema import LabelSchema
from mb.utils.logging_setup import get_logger

logger = get_logger(__name__)


class ImageFolderDataset(Dataset):
    """
    Custom dataset for loading images from a folder structure.
    
    Expected structure:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
    """
    
    def __init__(
        self,
        root: Path,
        transform: Optional[transforms.Compose] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    ):
        """
        Initialize the dataset.
        
        Args:
            root: Root directory containing class subdirectories
            transform: Optional transform to apply to images
            extensions: Image file extensions to include
        """
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions
        
        # Find all images and their labels
        self.samples = []
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = self.root / class_name
            for ext in self.extensions:
                # Case-insensitive search
                self.samples.extend([
                    (img_path, self.class_to_idx[class_name])
                    for img_path in class_dir.glob(f'*{ext}')
                    if img_path.is_file()
                ])
                self.samples.extend([
                    (img_path, self.class_to_idx[class_name])
                    for img_path in class_dir.glob(f'*{ext.upper()}')
                    if img_path.is_file()
                ])
        
        # Remove duplicates (case-insensitive matching)
        seen = set()
        unique_samples = []
        for img_path, label in self.samples:
            if img_path not in seen:
                seen.add(img_path)
                unique_samples.append((img_path, label))
        self.samples = unique_samples
        
        if len(self.samples) == 0:
            logger.warning(f"No images found in {root}")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image tensor, label)
        """
        from PIL import Image
        
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                # Create a dummy image and transform it
                dummy = Image.new('RGB', (224, 224), color='black')
                return self.transform(dummy), label
            return torch.zeros(3, 224, 224), label


class MultiLabelImageFolderDataset(ImageFolderDataset):
    """
    Folder dataset whose targets are multi-hot vectors rather than class indices.

    The folder an image sits in is its primary label; the manifest adds any others. Image
    loading, extension handling, and the unreadable-file fallback are inherited unchanged —
    only the target differs.
    """

    def __init__(
        self,
        root: Path,
        schema: "LabelSchema",
        manifest: Optional[Dict[str, List[str]]] = None,
        *,
        manifest_root: Optional[Path] = None,
        transform: Optional[transforms.Compose] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
    ):
        """
        Args:
            root: Split directory (``train`` or ``test``) holding the class folders.
            schema: Ordered label set; fixes the output index of every label.
            manifest: Extra labels keyed by path relative to *manifest_root*.
            manifest_root: Base the manifest's keys are relative to. Defaults to *root*'s
                parent, which is the dataset directory holding both splits.
        """
        super().__init__(root, transform=transform, extensions=extensions)
        from mb.data.label_schema import labels_for_sample

        self.schema = schema
        self.manifest = dict(manifest or {})
        self.manifest_root = Path(manifest_root) if manifest_root is not None else self.root.parent

        # Targets are resolved once here rather than per __getitem__ so that a label outside
        # the schema fails at construction, not partway through the first epoch.
        self.targets: List[List[float]] = []
        for img_path, class_idx in self.samples:
            primary = self.classes[class_idx]
            try:
                rel = Path(img_path).relative_to(self.manifest_root).as_posix()
            except ValueError:
                rel = Path(img_path).name
            names = labels_for_sample(rel, primary, self.manifest)
            self.targets.append(schema.multi_hot(names))

    def label_positive_counts(self) -> List[int]:
        """Per-label positive counts across the split, for loss weighting."""
        counts = [0] * self.schema.num_labels
        for vector in self.targets:
            for index, value in enumerate(vector):
                if value > 0:
                    counts[index] += 1
        return counts

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the image and its multi-hot target vector."""
        image, _primary_index = super().__getitem__(idx)
        return image, torch.tensor(self.targets[idx], dtype=torch.float32)


def get_train_transforms(
    image_size: int = 224,
    *,
    preprocessing: Optional[PreprocessingSpec] = None,
) -> transforms.Compose:
    """
    Get training data transforms with augmentation.

    Args:
        image_size: Target image size (assumes square)
        preprocessing: Normalization contract for the backbone being trained. Defaults to
            ImageNet statistics, which is what every torchvision backbone expects.

    Returns:
        Compose transform for training
    """
    spec = preprocessing or DEFAULT_PREPROCESSING
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(spec.normalize_mean), std=list(spec.normalize_std))
    ])


def get_val_transforms(
    image_size: int = 224,
    *,
    preprocessing: Optional[PreprocessingSpec] = None,
) -> transforms.Compose:
    """
    Get validation/test data transforms (no augmentation).

    Args:
        image_size: Target image size (assumes square)
        preprocessing: Normalization contract for the backbone being evaluated. Must match
            what training used, or scores are measured on a different input distribution
            than the model was fitted on.

    Returns:
        Compose transform for validation
    """
    spec = preprocessing or DEFAULT_PREPROCESSING
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(spec.normalize_mean), std=list(spec.normalize_std))
    ])


def create_data_loaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int,
    image_size: int = 224,
    num_workers: int = 0,
    pin_memory: bool = True,
    preprocessing: Optional[PreprocessingSpec] = None,
    label_schema: Optional["LabelSchema"] = None,
    label_manifest: Optional[Dict[str, List[str]]] = None,
    manifest_root: Optional[Path] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch data loaders for training and validation.

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation/test data directory
        batch_size: Batch size for data loading
        image_size: Target image size (assumes square)
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        preprocessing: Normalization contract for the backbone (default: ImageNet)
        label_schema: When given, build multi-label datasets with multi-hot targets over
            this schema. Omitted means single-label targets from the folder layout.
        label_manifest: Extra per-image labels, keyed relative to *manifest_root*
        manifest_root: Base for the manifest's keys (default: each split's parent)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    if label_schema is not None:
        train_dataset: Dataset = MultiLabelImageFolderDataset(
            root=train_dir,
            schema=label_schema,
            manifest=label_manifest,
            manifest_root=manifest_root,
            transform=get_train_transforms(image_size, preprocessing=preprocessing),
        )
        val_dataset: Dataset = MultiLabelImageFolderDataset(
            root=val_dir,
            schema=label_schema,
            manifest=label_manifest,
            manifest_root=manifest_root,
            transform=get_val_transforms(image_size, preprocessing=preprocessing),
        )
    else:
        train_dataset = ImageFolderDataset(
            root=train_dir,
            transform=get_train_transforms(image_size, preprocessing=preprocessing)
        )

        val_dataset = ImageFolderDataset(
            root=val_dir,
            transform=get_val_transforms(image_size, preprocessing=preprocessing)
        )
    
    # Verify classes match
    if train_dataset.classes != val_dataset.classes:
        logger.warning(
            f"Class mismatch: train={train_dataset.classes}, val={val_dataset.classes}"
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Classes: {train_dataset.classes}")
    
    return train_loader, val_loader
