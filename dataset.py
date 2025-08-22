"""
dataset.py
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
    dataset for triplet learning on image patches
    """

    def __init__(self, image_paths: List[str], patch_size:int = 65, patches_per_image:int = 10):
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