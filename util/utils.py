"""
utils.py
Utility functions for Student-Teacher Anomaly Detection
"""

import torch
import numpy as np
import cv2
import os
import glob
import json
import yaml
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
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


def get_device(prefer_gpu: bool = True) -> str:
    """Get the best available device"""
    if prefer_gpu and torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print("Using CPU")

    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def create_dir_structure(base_dir: str) -> Dict[str, str]:
    """Create directory structure for experiments"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"experiment_{timestamp}")

    dirs = {
        'experiment': experiment_dir,
        'checkpoints': os.path.join(experiment_dir, 'checkpoints'),
        'logs': os.path.join(experiment_dir, 'logs'),
        'results': os.path.join(experiment_dir, 'results'),
        'visualizations': os.path.join(experiment_dir, 'visualizations'),
        'configs': os.path.join(experiment_dir, 'configs')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to file"""
    with open(save_path, 'w') as f:
        if save_path.endswith('.json'):
            json.dump(config, f, indent=2)
        elif save_path.endswith('.yaml') or save_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError("Config file must be .json or .yaml")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            raise ValueError("Config file must be .json or .yaml")

    return config


def find_images_in_directory(directory: str, extensions: List[str] = None) -> List[str]:
    """Find all images in a directory"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

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
    """Split dataset into train/val/test sets"""
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
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)

    # Calculate padding offsets
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    # Place resized image in center
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return padded


def compute_image_statistics(image_paths: List[str],
                           sample_size: int = 1000) -> Dict[str, np.ndarray]:
    """Compute mean and std statistics for a dataset"""
    if len(image_paths) > sample_size:
        sampled_paths = np.random.choice(image_paths, sample_size, replace=False)
    else:
        sampled_paths = image_paths

    pixel_values = []

    for img_path in sampled_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image) / 255.0  # Normalize to [0, 1]
            pixel_values.append(image.reshape(-1, 3))
        except:
            continue

    if pixel_values:
        all_pixels = np.vstack(pixel_values)
        mean = np.mean(all_pixels, axis=0)
        std = np.std(all_pixels, axis=0)
    else:
        # Default ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    return {'mean': mean, 'std': std}


def create_anomaly_heatmap(anomaly_map: np.ndarray,
                         colormap: str = 'hot',
                         alpha: float = 0.7) -> np.ndarray:
    """Create colored heatmap from anomaly scores"""
    # Normalize to [0, 1]
    normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(normalized)

    # Convert to BGR for OpenCV
    colored_bgr = (colored[:, :, [2, 1, 0]] * 255).astype(np.uint8)

    return colored_bgr


def overlay_heatmap_on_image(image: np.ndarray,
                           heatmap: np.ndarray,
                           alpha: float = 0.4) -> np.ndarray:
    """Overlay heatmap on original image"""
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay


def apply_morphological_operations(mask: np.ndarray,
                                 operation: str = 'close',
                                 kernel_size: int = 5) -> np.ndarray:
    """Apply morphological operations to clean up binary masks"""
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
        raise ValueError(f"Unknown operation: {operation}")

    return result


def compute_connected_components(mask: np.ndarray,
                               min_area: int = 100) -> Tuple[np.ndarray, int]:
    """Compute connected components and filter by area"""
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))

    # Filter components by area
    filtered_mask = np.zeros_like(mask)
    valid_components = 0

    for label in range(1, num_labels):  # Skip background (label 0)
        component_mask = (labels == label)
        area = np.sum(component_mask)

        if area >= min_area:
            filtered_mask[component_mask] = 1
            valid_components += 1

    return filtered_mask, valid_components


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


def create_video_from_images(image_paths: List[str],
                           output_path: str,
                           fps: int = 10,
                           resize: Optional[Tuple[int, int]] = None):
    """Create video from sequence of images"""
    if not image_paths:
        raise ValueError("No images provided")

    # Read first image to get dimensions
    first_img = cv2.imread(image_paths[0])
    if resize:
        first_img = cv2.resize(first_img, resize)

    height, width, _ = first_img.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if resize:
            img = cv2.resize(img, resize)
        out.write(img)

    out.release()
    print(f"Video saved to {output_path}")


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
    """Track training progress and statistics"""

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'timestamps': []
        }

    def update(self, train_loss: float, val_loss: float = None, lr: float = None):
        """Update progress with new values"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rate'].append(lr)
        self.history['timestamps'].append(datetime.now().isoformat())

        # Save to file
        with open(self.save_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot_progress(self, save_path: Optional[str] = None):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Loss curves
        epochs = range(1, len(self.history['train_loss']) + 1)
        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', color='blue')

        if any(v is not None for v in self.history['val_loss']):
            val_losses = [v for v in self.history['val_loss'] if v is not None]
            val_epochs = [i+1 for i, v in enumerate(self.history['val_loss']) if v is not None]
            axes[0].plot(val_epochs, val_losses, label='Val Loss', color='red')

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True)

        # Learning rate
        if any(v is not None for v in self.history['learning_rate']):
            lrs = [v for v in self.history['learning_rate'] if v is not None]
            lr_epochs = [i+1 for i, v in enumerate(self.history['learning_rate']) if v is not None]
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
    """Early stopping utility"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop early"""
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_better(self, score: float) -> bool:
        """Check if current score is better than best"""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


def generate_synthetic_anomaly(normal_image: np.ndarray,
                             anomaly_type: str = 'gaussian_noise',
                             intensity: float = 0.1,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic anomalies for testing"""
    h, w = normal_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    anomaly_image = normal_image.copy()

    if anomaly_type == 'gaussian_noise':
        # Add Gaussian noise to random regions
        noise_regions = kwargs.get('num_regions', 3)
        region_size = kwargs.get('region_size', 50)

        for _ in range(noise_regions):
            x = np.random.randint(0, max(1, w - region_size))
            y = np.random.randint(0, max(1, h - region_size))

            # Create circular mask
            center = (x + region_size//2, y + region_size//2)
            radius = region_size // 2
            cv2.circle(mask, center, radius, 255, -1)

            # Add noise
            noise = np.random.normal(0, intensity * 255, (region_size, region_size, 3))
            region = anomaly_image[y:y+region_size, x:x+region_size]
            noisy_region = np.clip(region + noise, 0, 255).astype(np.uint8)
            anomaly_image[y:y+region_size, x:x+region_size] = noisy_region

    elif anomaly_type == 'scratches':
        # Add scratch-like artifacts
        num_scratches = kwargs.get('num_scratches', 5)

        for _ in range(num_scratches):
            # Random line
            pt1 = (np.random.randint(0, w), np.random.randint(0, h))
            pt2 = (np.random.randint(0, w), np.random.randint(0, h))
            thickness = np.random.randint(2, 8)

            cv2.line(mask, pt1, pt2, 255, thickness)
            cv2.line(anomaly_image, pt1, pt2, (0, 0, 0), thickness)

    elif anomaly_type == 'blobs':
        # Add blob-like anomalies
        num_blobs = kwargs.get('num_blobs', 3)

        for _ in range(num_blobs):
            center = (np.random.randint(50, w-50), np.random.randint(50, h-50))
            axes = (np.random.randint(20, 60), np.random.randint(20, 60))
            angle = np.random.randint(0, 180)

            cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
            cv2.ellipse(anomaly_image, center, axes, angle, 0, 360,
                       (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), -1)

    return anomaly_image, mask


def benchmark_model_components(model, input_size: Tuple[int, int] = (256, 256), device: str = 'cuda'):
    """Benchmark different components of the model"""
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, *input_size, device=device)

    results = {}

    with torch.no_grad():
        # Teacher forward pass times
        for patch_size in model.patch_sizes:
            teacher = model.teachers[f'teacher_{patch_size}']

            # Warm up
            for _ in range(10):
                patches, _, _ = model.extract_patches_dense(dummy_input, patch_size)
                _ = teacher(patches[:100])  # Limit patches for timing

            # Time measurement
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            patches, h_out, w_out = model.extract_patches_dense(dummy_input, patch_size)
            features, _ = teacher(patches)
            end.record()

            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)

            results[f'teacher_{patch_size}_ms'] = elapsed
            results[f'teacher_{patch_size}_patches'] = patches.shape[0]

    return results


def validate_dataset_structure(dataset_path: str, expected_structure: Dict[str, List[str]]) -> bool:
    """Validate dataset directory structure"""
    valid = True

    for category, subdirs in expected_structure.items():
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            print(f"Missing category directory: {category_path}")
            valid = False
            continue

        for subdir in subdirs:
            subdir_path = os.path.join(category_path, subdir)
            if not os.path.exists(subdir_path):
                print(f"Missing subdirectory: {subdir_path}")
                valid = False
            else:
                # Check if directory contains images
                images = find_images_in_directory(subdir_path)
                if not images:
                    print(f"No images found in: {subdir_path}")
                    valid = False
                else:
                    print(f"Found {len(images)} images in {subdir_path}")

    return valid


def create_dataset_summary(dataset_path: str) -> Dict[str, Any]:
    """Create summary statistics for a dataset"""
    summary = {
        'total_images': 0,
        'categories': {},
        'image_sizes': [],
        'file_formats': {}
    }

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue

        category_summary = {}
        category_images = find_images_in_directory(category_path)

        category_summary['total_images'] = len(category_images)
        summary['total_images'] += len(category_images)

        # Sample some images for size analysis
        sample_size = min(100, len(category_images))
        sampled_images = np.random.choice(category_images, sample_size, replace=False)

        sizes = []
        formats = {}

        for img_path in sampled_images:
            try:
                with Image.open(img_path) as img:
                    sizes.append(img.size)
                    ext = os.path.splitext(img_path)[1].lower()
                    formats[ext] = formats.get(ext, 0) + 1
            except:
                continue

        category_summary['image_sizes'] = sizes
        category_summary['file_formats'] = formats
        summary['categories'][category] = category_summary

        # Update global statistics
        summary['image_sizes'].extend(sizes)
        for fmt, count in formats.items():
            summary['file_formats'][fmt] = summary['file_formats'].get(fmt, 0) + count

    return summary


# Configuration templates
def get_default_config() -> Dict[str, Any]:
    """Get default configuration template"""
    return {
        'model': {
            'patch_sizes': [17, 33, 65],
            'num_students': 3,
            'output_dim': 128
        },
        'training': {
            'teacher': {
                'epochs': 50,
                'lr': 2e-4,
                'lambda_kd': 1.0,
                'lambda_triplet': 1.0,
                'lambda_decorr': 0.1
            },
            'student': {
                'epochs': 100,
                'lr': 1e-4
            },
            'batch_size': 4,
            'device': 'cuda'
        },
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


def setup_experiment(config: Dict[str, Any], base_dir: str = './experiments') -> Dict[str, str]:
    """Setup complete experiment environment"""
    # Create directory structure
    dirs = create_dir_structure(base_dir)

    # Save configuration
    save_config(config, os.path.join(dirs['configs'], 'config.json'))

    # Set random seeds
    set_random_seeds(config.get('random_seed', 42))

    # Setup logging
    log_file = os.path.join(dirs['logs'], 'training.log')

    print(f"Experiment setup complete!")
    print(f"Experiment directory: {dirs['experiment']}")
    print(f"Configuration saved to: {os.path.join(dirs['configs'], 'config.json')}")

    return dirs


# Context manager for GPU memory monitoring
class GPUMemoryMonitor:
    """Context manager for monitoring GPU memory usage"""

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
            memory_used = (end_memory - self.start_memory) / 1e9
            print(f"{self.description} used {memory_used:.2f} GB GPU memory")


# Decorators
def timer(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


def gpu_memory_tracker(func):
    """Decorator to track GPU memory usage"""
    def wrapper(*args, **kwargs):
        with GPUMemoryMonitor(func.__name__):
            return func(*args, **kwargs)
    return wrapper