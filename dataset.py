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
    dataset for triplet learning on image patches
    """

    def __init__(self, image_paths: List[str], patch_size: int = 64, patches_per_image: int = 10):
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


    def __getitem__(self,idx):
        img_idx = idx // self.patches_per_image
        img_path = self.image_paths[img_idx]

        # load image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Extract random patch (anchor)
        anchor = self._extract_random_patch(image)

        # create positive (small translation + noise)
        positive = self._create_positive_patch(image, anchor)

        # create negative (random patch from different image)
        negative = self._create_negative_patch(image, anchor)

        return anchor, positive, negative


    def _extract_random_patch(self, image: torch.Tensor) -> torch.Tensor:
        """
        extract a random patch from the image
        """
        _, h, w = image.shape
        if h < self.patch_size or w < self.patch_size:
            image = F.interpolate(image.unsqueeze(0),
                                  size=(max(h, self.patch_size), max(w, self.patch_size)),
                                  mode='bilinear', align_corners=False)

            _, h, w_ = image.shape
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)

        patch = image[:, top:top+self.patch_size, left:left+self.patch_size]

        return patch


    def _create_positive_patch(self, image: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """
        create positive patch with small translation and noise
        """
        _, h, w = image.shape
        max_shift = self.patch_size // 4

        # Find anchor position (approximate)
        top = random.randint(0, max(0, h - self.patch_size))
        left = random.randint(0, max(0, w - self.patch_size))

        # Small random shift
        shift_top = random.randint(-max_shift, max_shift)
        shift_left = random.randint(-max_shift, max_shift)

        pos_top = max(0, min(h - self.patch_size, top + shift_top))
        pos_left = max(0, min(w - self.patch_size, left + shift_left))

        positive = image[:, pos_top:pos_top+self.patch_size, pos_left:pos_left+self.patch_size]
        positive += torch.rand_like(positive) * 0.01

        return positive

    def _create_negative_patch(self, current_img_idx: int) -> torch.Tensor:
        """
        creative negative patch from different image
        """
        available_indices = [i for i in range(len(self.image_paths)) if i != current_img_idx]
        if not available_indices:
            # if only one image, use heavy augmentation
            neg_image = Image.open(self.image_paths).convert('RGB')
            neg_image = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, satuation=0.5, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])(neg_image)

        else:
            neg_img_idx = random.choice(available_indices)
            neg_image = Image.open(self.image_paths[neg_img_idx]).convert('RGB')
            neg_image = self.transform(neg_image)

        negative = self._extract_radnom_patch(neg_image)
        return negative

class AnomalyDataset(Dataset):
    """
    simple dataset for anomaly detection training/testing
    """

    def __init__(self, image_paths: List[str], transform=None, is_anomaly: bool = False):
        self.image_paths = image_paths
        self.is_anomaly = is_anomaly
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform

        if self.is_anomaly:
            return image, 1
        else:
            return image, 0

class MVTecDataset(Dataset):
    """
    dataset class specifically for MVTec AD dataset
    """
    def __init__(self, root_dir: str, category: str, split: str = 'train',
                 transform=None, include_mask: bool = False):

        """
        args:
            root_dir : Path to MVTec Dataset root path
            category : MVTec AD dataset category name(e.g., 'bottle', 'cable', etc.)
            split : train or test
            transform : transform to apply to image
            include_mask : include mask image
        """
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.include_mask = include_mask
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.image_paths, self.labels, self.mask_paths = self._load_dataset()

    def _load_dataset(self) -> Tuple[List[str]]:
        """
        load dataset file paths and labels
        """
        image_paths = []
        labels = []
        mask_paths = []

        if self.split == 'train':
            # training data - only normal images
            normal_dir = os.path.join(self.root_dir, 'train', 'good')
            if os.path.exists(normal_dir):
                for img_file in sorted(glob.glob(os.path.join(normal_dir, '*'))):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(img_file)
                        labels.append(0)
                        mask_paths.append(None)

        elif self.split == 'test':
            # Test data - normal and anomaly images
            test_dir = os.path.join(self.root_dir,  self.category, 'good')

            # normal test images
            normal_test_dir = os.path.join(test_dir, 'good')
            if os.path.exists(normal_test_dir):
                for img_file in sorted(glob.glob(os.path.join(normal_test_dir, '*'))):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(img_file)
                        labels.append(0)
                        mask_paths.append(None)

            # anomaly test image
            for defect_type in os.listdir(test_dir):
                defect_dir = os.path.join(test_dir, defect_type)
                if os.path.isdir(defect_dir) and defect_type != 'good':
                    for img_file in sorted(glob.glob(os.path.join(defect_dir, '*'))):
                        









