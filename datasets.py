"""
datasets.py
Dataset Classes for Student-Teacher Anomaly Detection
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import random
import os
import glob
from typing import List, Optional, Tuple


class TripletPatchDataset(Dataset):
    """
    Dataset for triplet learning on image patches
    """

    def __init__(self, image_paths: List[str], patch_size: int = 65, patches_per_image: int = 10):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        image_path = self.image_paths[img_idx]

        # Load image
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)

        # Extract random patch (anchor)
        anchor, anchor_top, anchor_left = self._extract_random_patch(img)

        # Create positive (small translation + noise)
        positive = self._create_positive_patch(img, anchor_top, anchor_left)

        # Create negative (random patch from different image)
        negative = self._create_negative_patch(img_idx)

        return anchor, positive, negative

    def _extract_random_patch(self, image: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Extract a random patch from the image
        """
        _, h, w = image.shape
        if h < self.patch_size or w < self.patch_size:
            image = F.interpolate(
                input=image.unsqueeze(0),
                size=(max(h, self.patch_size), max(w, self.patch_size)),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            _, h, w = image.shape

        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)

        patch = image[:, top:top+self.patch_size, left:left+self.patch_size]

        return patch, top, left

    def _create_positive_patch(self, image: torch.Tensor, anchor_top: int, anchor_left: int) -> torch.Tensor:
        """
        Create positive patch with small translation and noise from anchor position
        """
        _, h, w = image.shape
        max_shift = self.patch_size // 4

        # Small random shift from anchor position
        shift_top = random.randint(-max_shift, max_shift)
        shift_left = random.randint(-max_shift, max_shift)

        # Calculate new position with bounds checking
        pos_top = max(0, min(h - self.patch_size, anchor_top + shift_top))
        pos_left = max(0, min(w - self.patch_size, anchor_left + shift_left))

        positive = image[:, pos_top:pos_top+self.patch_size, pos_left:pos_left+self.patch_size]
        positive = positive + torch.randn_like(positive) * 0.1  # Add noise

        return positive

    def _create_negative_patch(self, current_img_idx: int) -> torch.Tensor:
        """
        Create negative patch from different image
        """
        available_indices = [i for i in range(len(self.image_paths)) if i != current_img_idx]

        if not available_indices:
            # If only one image, use heavy augmentation
            neg_image = Image.open(self.image_paths[current_img_idx]).convert('RGB')
            neg_image = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])(neg_image)
        else:
            # Use different image
            neg_img_idx = random.choice(available_indices)
            neg_image = Image.open(self.image_paths[neg_img_idx]).convert('RGB')
            neg_image = self.transform(neg_image)

        negative, _, _ = self._extract_random_patch(neg_image)

        return negative


class AnomalyDataset(Dataset):
    """
    Basic anomaly detection dataset
    """

    def __init__(self, image_paths: List[str], is_anomaly: bool = False, transform=None):
        self.image_paths = image_paths
        self.is_anomaly = is_anomaly
        self.transform = transform or get_default_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = 1 if self.is_anomaly else 0
        return image, label


class MVTecDataset(Dataset):
    """
    MVTec Anomaly Detection Dataset
    """

    def __init__(self, root: str, category: str, split: str = 'train',
                 include_mask: bool = False, transform=None):
        self.root = root
        self.category = category
        self.split = split
        self.include_mask = include_mask
        self.transform = transform or get_default_transform()

        self.image_paths = []
        self.labels = []
        self.mask_paths = []

        self._load_dataset()

    def _load_dataset(self):
        category_path = os.path.join(self.root, self.category)

        if self.split == 'train':
            # Training data (only normal images)
            good_dir = os.path.join(category_path, 'train', 'good')
            if os.path.exists(good_dir):
                image_files = glob.glob(os.path.join(good_dir, '*'))
                self.image_paths.extend(image_files)
                self.labels.extend([0] * len(image_files))

        elif self.split == 'test':
            # Test data (normal + anomaly images)
            test_dir = os.path.join(category_path, 'test')

            if os.path.exists(test_dir):
                # Normal test images
                good_dir = os.path.join(test_dir, 'good')
                if os.path.exists(good_dir):
                    good_files = glob.glob(os.path.join(good_dir, '*'))
                    self.image_paths.extend(good_files)
                    self.labels.extend([0] * len(good_files))
                    self.mask_paths.extend([None] * len(good_files))

                # Anomaly images
                defect_dirs = [d for d in os.listdir(test_dir)
                              if d != 'good' and os.path.isdir(os.path.join(test_dir, d))]

                for defect_type in defect_dirs:
                    defect_dir = os.path.join(test_dir, defect_type)
                    defect_files = glob.glob(os.path.join(defect_dir, '*'))

                    self.image_paths.extend(defect_files)
                    self.labels.extend([1] * len(defect_files))

                    # Add corresponding mask paths if available
                    if self.include_mask:
                        gt_dir = os.path.join(category_path, 'ground_truth', defect_type)
                        mask_paths = []
                        for defect_file in defect_files:
                            base_name = os.path.splitext(os.path.basename(defect_file))[0]
                            mask_file = os.path.join(gt_dir, f"{base_name}_mask.png")
                            if os.path.exists(mask_file):
                                mask_paths.append(mask_file)
                            else:
                                mask_paths.append(None)
                        self.mask_paths.extend(mask_paths)
                    else:
                        self.mask_paths.extend([None] * len(defect_files))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.include_mask and self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert('L')
            mask = transforms.ToTensor()(mask)
            return image, label, mask

        return image, label


class CustomAnomalyDataset(Dataset):
    """
    Custom anomaly detection dataset from directories
    """

    def __init__(self, normal_dir: str, anomaly_dir: str = None,
                 split_ratio: float = 0.8, mode: str = 'train', transform=None):
        self.normal_dir = normal_dir
        self.anomaly_dir = anomaly_dir
        self.split_ratio = split_ratio
        self.mode = mode
        self.transform = transform or get_default_transform()

        self.image_paths = []
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        # Load normal images
        normal_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            normal_files.extend(glob.glob(os.path.join(self.normal_dir, ext)))

        # Split normal images
        random.shuffle(normal_files)
        split_idx = int(len(normal_files) * self.split_ratio)

        if self.mode == 'train':
            # Training: only normal images (first split_ratio portion)
            self.image_paths = normal_files[:split_idx]
            self.labels = [0] * len(self.image_paths)

        elif self.mode == 'test':
            # Testing: remaining normal images + all anomaly images
            self.image_paths = normal_files[split_idx:]
            self.labels = [0] * len(self.image_paths)

            # Add anomaly images if available
            if self.anomaly_dir and os.path.exists(self.anomaly_dir):
                anomaly_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    anomaly_files.extend(glob.glob(os.path.join(self.anomaly_dir, ext)))

                self.image_paths.extend(anomaly_files)
                self.labels.extend([1] * len(anomaly_files))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_default_transform(image_size: int = 224):
    """Get default image transform"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_augmented_transform(image_size: int = 224):
    """Get augmented image transform for training"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def create_dataset(dataset_type: str, **kwargs):
    """Factory function to create datasets"""
    if dataset_type == 'triplet':
        return TripletPatchDataset(**kwargs)
    elif dataset_type == 'anomaly':
        return AnomalyDataset(**kwargs)
    elif dataset_type == 'mvtec':
        return MVTecDataset(**kwargs)
    elif dataset_type == 'custom':
        return CustomAnomalyDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# Test code
if __name__ == '__main__':
    # Test with MVTec dataset
    mvtec_root = "/workspace/anomaly/mvtec"
    category = "bottle"

    # Find images for testing
    train_good_dir = os.path.join(mvtec_root, category, 'train', 'good')
    images = glob.glob(os.path.join(train_good_dir, '*'))[:10]
    print(images)

    # Test TripletPatchDataset
    triplet_dataset = create_dataset(
        dataset_type='triplet',
        image_paths=images,
        patch_size=64,
        patches_per_image=5
    )

    print(f"Triplet dataset length: {len(triplet_dataset)}")

    # Test a sample
    anchor, positive, negative = triplet_dataset[0]
    print(f"Anchor shape: {anchor.shape}")
    print(f"Positive shape: {positive.shape}")
    print(f"Negative shape: {negative.shape}")

    # Test MVTecDataset
    mvtec_dataset = create_dataset(
        dataset_type='mvtec',
        root=mvtec_root,
        category=category,
        split='train'
    )

    print(f"MVTec dataset length: {len(mvtec_dataset)}")

    if len(mvtec_dataset) > 0:
        image, label = mvtec_dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Label: {label}")

    print("✅ All datasets working correctly!")