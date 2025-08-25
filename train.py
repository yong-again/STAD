"""
train.py
Train functions for Student-Teacher Anomaly Detection
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict, List
from tqdm import tqdm
import os

from models import TeacherNetwork, MultiScaleStudentTeacherFramework


def train_teacher_with_distillation():
    pass
