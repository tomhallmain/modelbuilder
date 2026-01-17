"""
PyTorch data loading utilities for image classification.

This module provides data loaders and transforms for PyTorch training.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


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


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get training data transforms with augmentation.
    
    Args:
        image_size: Target image size (assumes square)
        
    Returns:
        Compose transform for training
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test data transforms (no augmentation).
    
    Args:
        image_size: Target image size (assumes square)
        
    Returns:
        Compose transform for validation
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])


def create_data_loaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int,
    image_size: int = 224,
    num_workers: int = 0,
    pin_memory: bool = True,
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
        **kwargs: Additional arguments (ignored for now)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = ImageFolderDataset(
        root=train_dir,
        transform=get_train_transforms(image_size)
    )
    
    val_dataset = ImageFolderDataset(
        root=val_dir,
        transform=get_val_transforms(image_size)
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
