"""
utils.py
Utility functions for Student-Teacher Anomaly Detections
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
import json
import yaml
from typing import Any, List,Tuple, Dict, Optional
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(prefer_gpu: str) -> str:
    """
    get the best available device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print(f"Using CPU")

    return device

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count Model Parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total_params,
        'total_trainable_params': total_trainable_params,
        'non_trainable_params': total_params - total_trainable_params,
    }

def createdir_experiment(base_dir: str) -> Dict[str, str]:
    """
    Create directory structure for experiment
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    experiment_dir = os.path.join(base_dir, f"{timestamp}")

    exp_count = 0
    exp_dir = os.path.join(experiment_dir, "exp")
    while os.path.exists(exp_dir):
        if exp_count == 0:
            exp_count += 1
            exp_dir = os.path.join(experiment_dir, f"exp")
        elif exp_count == 1:
            exp_count += 1
            exp_dir = os.path.join(experiment_dir, f"exp{exp_count}")
        else:
            exp_count += 1
            exp_dir = os.path.join(experiment_dir, f"exp{exp_count}")

    os.makedirs(exp_dir, exist_ok=True)

    dirs = {
        'experiment' : exp_dir,
        'checkpoints': os.path.join(exp_dir, 'checkpoints'),
        'logs': os.path.join(exp_dir, 'logs'),
        'results': os.path.join(exp_dir, 'results'),
        'visualizations': os.path.join(exp_dir, 'visualizations'),
        'configs': os.path.join(exp_dir, 'configs'),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs

def save_config(config: Dict[str, Any], save_path: str):
    """
    save configuration to file
    """
    with open(save_path, 'w') as f:
        if save_path.endswith('.json'):
            json.dump(config, f)
        elif save_path.endswith('.yaml') or save_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError("Configuration file must be 'json' or 'yaml'.")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    open configuration file
    """
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError("Configuration file must be 'json' or 'yaml'.")

    return config









if __name__ == "__main__":
    #set_random_seeds(42)
    #device = get_device('cuda')
    dirs = createdir_experiment("experiments")



