"""
utils.py
Utility functions for Student-Teacher Anomaly Detections
"""

import torch
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
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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

def find_images_in_directory(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Find all images in a directory
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', 'tif']

    image_paths = []
    for ext in extensions:
        pattern = os.path.join(directory, f"**/*{ext}")
        image_paths.extend(glob.glob(pattern, recursive=True))
        pattern = os.path.join(directory, f"**/*{ext.upper()}")
        image_paths.extend(glob.glob(pattern, recursive=True))

    return sorted(list(set(image_paths)))

def split_dataset(image_paths: List[str],
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split dataset into train/val/test
    """
    np.random.seed(random_seed)
    shuffled_paths = np.random.permutation(image_paths)

    n_total = len(shuffled_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_paths = shuffled_paths[:n_train].tolist()
    val_paths = shuffled_paths[n_train:n_train+n_val].tolist()
    test_paths = shuffled_paths[n_train+n_val:].tolist()

    return train_paths, val_paths, test_paths

def resize_image_maintain_aspect(image: np.ndarray,
                                 target_size: Tuple[int, int],
                                 pad_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize Image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)

    # Calculate padding offsets
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    # Place resized image in center
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded

def compute_image_statistics(image_paths: List[str],
                             sample_size: int = 1000) -> Dict[str, np.ndarray]:
    """
    compute mean and std statistics for a datasets
    """

    if len(image_paths) > sample_size:
        sampled_paths = np.random.choice(image_paths, size=sample_size, replace=False)
    else:
        sampled_paths = image_paths

    pixel_value = []
    for img_path in sampled_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image) / 255.0
            pixel_value.append(image.reshape(-1, 3))
        except:
            continue
    if pixel_value:
        all_pixels = np.vstack(pixel_value)
        mean = np.mean(all_pixels, axis=0)
        std = np.std(all_pixels, axis=0)
    else:
        # Default ImageNet statics
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    return {'mean' : mean, 'std' : std}

def create_anomaly_heatmap(anomaly_map: np.ndarray, colormap: str = 'hot',) -> np.ndarray:
    """
    created colored heatmap from anomaly detection
    """
    # normalize [0, 1]
    normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    # apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(normalized)

    # convert to rgb for opencv
    colored_bgr = (colored[:, :, [2, 1, 0]] * 255).astype(np.uint8)

    return colored_bgr

def overlay_heatmap_on_image(image: np.ndarray,
                             heatmap: np.ndarray,
                             alph: float = 0.7) -> np.ndarray:
    """
    overlay heatmap on image
    """
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    overlay = cv2.addWeighted(image, alph, heatmap, 1 - alph, 0)

    return overlay

def apply_morphological_operation(mask: np.ndarray,
                                 operation: str = 'close',
                                 kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological operation to clean up binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if operation == 'close':
        result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif operation == 'open':
        result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == 'erode':
        result = cv2.erode(mask, kernel, iterations=1)
    elif operation == 'dilate':
        result = cv2.dilate(mask, kernel, iterations=1)
    else:
        raise ValueError(f"Unknown operation {operation}")

    return result


def log_metrics(metrics: Dict[str, float],
                log_file: str,
                epoch: Optional[int] = None):
    """Log metrics to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, 'a') as f:
        if epoch is not None:
            f.write(f"[{timestamp}] Epoch {epoch}:\n")
        else:
            f.write(f"[{timestamp}] Metrics:\n")

        for key, value in metrics.items():
            f.write(f"  {key}: {value:.6f}\n")
        f.write("\n")

def memory_usage_monitor():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9

        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'max_allocated_gb': max_allocated
        }
    else:
        return {'message': 'CUDA not available'}

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class ProgressTracker:
    """
    Track training progress and statistics
    """

    def __init__(self, save_path : str):
        self.save_path = save_path
        self.history = {
            'train_loss' : [],
            'val_loss' : [],
            'learning_rate' : [],
            'timestamp': []
        }

    def update(self, train_loss : float, val_loss: float = None, lr: float = None):
        """
        update progress with new values
        """
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rate'].append(lr)
        self.history['timestamp'].append(datetime.now().isoformat())

        # save to file
        with open(self.save_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot_progress(self, save_path: Optional[str] = None):
        """
        plot progress
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Loss Curves
        epochs = range(1, len(self.history['train_loss']) + 1)
        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', color='blue')

        if any(v is not None for v in self.history['val_loss']):
            val_losses = [v for v in self.history['val_loss'] if v is not None]
            val_epochs = [i+1 for i, v in enumerate(self.history['val_loss']) if v is not None]
            axes[0].plot(val_epochs, val_losses, label='Validation Loss', color='red')

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True)

        # Learning Rate
        if any (v is not None for v in self.history['learning_rate']):
            lrs = [v for v in self.history['learning_rate'] if v is not None]
            lr_epochs = [i+1 for i,v in enumerate(self.history['learning_rate']) if v is not None]
            axes[1].plot(lr_epochs, lrs, color='green')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].grid(True)
            axes[1].set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class EarlyStopping:
    """
    Early stopping callback
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float):
        """
        checking if training should stop early
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter = 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_better(self, score) -> bool:
        """
        Check if current score is better than best score
        """
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration templates
    """
    return {
        'model' : {
            'patch_size' : [17, 33, 65],
            'num_students' : 3,
            'output_dim' : 128
        },
        'training': {
            'teacher':{
                'epochs':50,
                'lr': 2e-4,
                'lambda_kd': 10,
                'lambda_triplet': 1.0,
                'lambda_decorr': 0.1
            },
            'student':{
                'epochs': 100,
                'lr': 1e-4
            }
        },
        'batch_size' : 4,
        'device': 'cuda',
        'data': {
            'image_size': [256, 256],
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225]
        },
        'evaluation': {
            'metrics': ['auc_roc', 'auc_pr', 'pro_score'],
            'visualization_samples': 5
        }
    }

def setup_experiment(config: Dict[str, Any], base_dir = '../experiments') -> Dict[str, str]:
    """
    setup complete experiment configuration
    """
    # create directory structures
    dirs = createdir_experiment(base_dir)

    # set random seeds
    set_random_seeds(config.get('random_seed', 42))

    # setup logging
    log_file = os.path.join(dirs['logs'], 'training.log')

    print(f"Experiment setup complete.")
    print(f"Experiment directory: {dirs['experiment']}")
    print(f"Configuration saved to:{os.path.join(dirs['configs'], 'config_json.json')}")

    return dirs


# Context manger for GPU memory monitoring
class GPUMemoryMonitor:
    """
    Context manger for monitoring GPU memory
    """
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_memory = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            memory_used = (end_memory - self.start_memory)
            print(f"{self.description} used {memory_used:.2f} GB GPU memory")

# Decorators
def timer(func):
    """
    Decorator to time function execution
    """
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

def gpu_memory_tracker(func):
    """
    Decorator to track GPU memory usage
    """
    def wrapper(*args, **kwargs):
        with GPUMemoryMonitor(func.__name__):
            return func(*args, **kwargs)
    return wrapper

if __name__ == "__main__":
    set_random_seeds(42)
    device = get_device('cuda')
    config = get_default_config()
    setup_experiment(config)
    #image_paths = find_images_in_directory('./mvtec')
    #rint(image_paths[:10])




