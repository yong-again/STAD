"""
evaluation.py
Evaluation functions and metrics for Student-Teacher Anomaly Detection
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import cv2

from model.models import MultiScaleStudentTeacherFramework
from dataset.datasets import get_default_transform


def evaluate_image_level_detection(model: MultiScaleStudentTeacherFramework,
                                 normal_images: List[str],
                                 anomaly_images: List[str],
                                 device: str = 'cuda',
                                 batch_size: int = 8) -> Dict[str, float]:
    """
    Evaluate image-level anomaly detection performance

    Args:
        model: Trained model
        normal_images: List of normal image paths
        anomaly_images: List of anomaly image paths
        device: Device to run inference on
        batch_size: Batch size for inference

    Returns:
        Dictionary with evaluation metrics
    """

    model = model.to(device)
    model.eval()

    transform = get_default_transform()

    normal_scores = []
    anomaly_scores = []

    print("Evaluating normal images...")
    normal_scores = _compute_scores_for_images(model, normal_images, transform, device, batch_size)

    print("Evaluating anomaly images...")
    anomaly_scores = _compute_scores_for_images(model, anomaly_images, transform, device, batch_size)

    # Compute metrics
    y_true = [0] * len(normal_scores) + [1] * len(anomaly_scores)
    y_scores = normal_scores + anomaly_scores

    metrics = _compute_classification_metrics(y_true, y_scores)

    print(f"\nImage-Level Detection Results:")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"Best F1: {metrics['best_f1']:.4f}")
    print(f"Best Threshold: {metrics['best_threshold']:.4f}")

    return metrics


def evaluate_pixel_level_segmentation(model: MultiScaleStudentTeacherFramework,
                                    test_images: List[str],
                                    test_masks: List[str],
                                    device: str = 'cuda') -> Dict[str, float]:
    """
    Evaluate pixel-level anomaly segmentation performance

    Args:
        model: Trained model
        test_images: List of test image paths
        test_masks: List of corresponding mask paths
        device: Device to run inference on

    Returns:
        Dictionary with segmentation metrics
    """

    model = model.to(device)
    model.eval()

    transform = get_default_transform()

    all_pixel_scores = []
    all_pixel_labels = []

    print("Evaluating pixel-level segmentation...")

    with torch.no_grad():
        for img_path, mask_path in tqdm(zip(test_images, test_masks), total=len(test_images)):
            # Load and preprocess image
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Load mask
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask.resize((256, 256))) > 0  # Binary mask

            # Get anomaly map
            anomaly_map = model.compute_anomaly_score(image_tensor)
            anomaly_map = anomaly_map.squeeze().cpu().numpy()

            # Flatten for pixel-wise evaluation
            pixel_scores = anomaly_map.flatten()
            pixel_labels = mask.flatten().astype(int)

            all_pixel_scores.extend(pixel_scores)
            all_pixel_labels.extend(pixel_labels)

    # Compute pixel-level metrics
    metrics = _compute_classification_metrics(all_pixel_labels, all_pixel_scores)

    # Compute PRO (Per-Region Overlap) score
    pro_score = compute_pro_score(test_images, test_masks, model, device)
    metrics['pro_score'] = pro_score

    print(f"\nPixel-Level Segmentation Results:")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"PRO Score: {metrics['pro_score']:.4f}")

    return metrics


def compute_pro_score(test_images: List[str],
                     test_masks: List[str],
                     model: MultiScaleStudentTeacherFramework,
                     device: str = 'cuda',
                     max_fpr: float = 0.3,
                     num_thresholds: int = 200) -> float:
    """
    Compute Per-Region Overlap (PRO) score

    Args:
        test_images: List of test image paths
        test_masks: List of mask paths
        model: Trained model
        device: Device to run on
        max_fpr: Maximum false positive rate
        num_thresholds: Number of thresholds to evaluate

    Returns:
        PRO score (normalized area under PRO curve)
    """

    model.eval()
    transform = get_default_transform()

    # Collect all anomaly scores and compute thresholds
    all_scores = []

    with torch.no_grad():
        for img_path in test_images:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            anomaly_map = model.compute_anomaly_score(image_tensor)
            all_scores.extend(anomaly_map.flatten().cpu().numpy())

    # Define thresholds
    min_score, max_score = np.min(all_scores), np.max(all_scores)
    thresholds = np.linspace(min_score, max_score, num_thresholds)

    per_region_overlaps = []
    fprs = []

    for threshold in thresholds:
        region_overlaps = []
        total_pixels = 0
        fp_pixels = 0

        with torch.no_grad():
            for img_path, mask_path in zip(test_images, test_masks):
                from PIL import Image

                # Load image and mask
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)

                mask = Image.open(mask_path).convert('L')
                mask = np.array(mask.resize((256, 256))) > 0

                # Get prediction
                anomaly_map = model.compute_anomaly_score(image_tensor)
                anomaly_map = anomaly_map.squeeze().cpu().numpy()
                prediction = anomaly_map > threshold

                # Count pixels for FPR
                total_pixels += mask.size
                fp_pixels += np.sum(prediction & ~mask)

                # Compute region overlaps
                if np.any(mask):
                    labeled_mask, num_regions = cv2.connectedComponents(mask.astype(np.uint8))

                    for region_id in range(1, num_regions + 1):
                        region = (labeled_mask == region_id)
                        region_size = np.sum(region)
                        overlap = np.sum(prediction & region)
                        overlap_ratio = overlap / region_size if region_size > 0 else 0
                        region_overlaps.append(overlap_ratio)

        # Compute metrics for this threshold
        fpr = fp_pixels / total_pixels if total_pixels > 0 else 0
        mean_overlap = np.mean(region_overlaps) if region_overlaps else 0

        if fpr <= max_fpr:
            fprs.append(fpr)
            per_region_overlaps.append(mean_overlap)

    # Compute area under PRO curve
    if len(fprs) > 1:
        pro_auc = auc(fprs, per_region_overlaps)
        # Normalize by maximum possible area
        normalized_pro = pro_auc / max_fpr
    else:
        normalized_pro = 0

    return normalized_pro


def visualize_detection_results(model: MultiScaleStudentTeacherFramework,
                              test_images: List[str],
                              test_masks: Optional[List[str]] = None,
                              num_samples: int = 5,
                              device: str = 'cuda',
                              save_path: Optional[str] = None):
    """
    Visualize anomaly detection results

    Args:
        model: Trained model
        test_images: List of test image paths
        test_masks: Optional list of mask paths
        num_samples: Number of samples to visualize
        device: Device to run inference on
        save_path: Path to save visualization
    """

    model = model.to(device)
    model.eval()

    transform = get_default_transform()

    # Select random samples
    indices = np.random.choice(len(test_images), min(num_samples, len(test_images)), replace=False)

    cols = 4 if test_masks else 3
    fig, axes = plt.subplots(num_samples, cols, figsize=(4*cols, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            from PIL import Image

            # Load image
            img_path = test_images[idx]
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Get anomaly map
            anomaly_map = model.compute_anomaly_score(image_tensor)
            anomaly_map = anomaly_map.squeeze().cpu().numpy()

            # Original image
            axes[i, 0].imshow(np.array(image.resize((256, 256))))
            axes[i, 0].set_title(f'Original {i+1}')
            axes[i, 0].axis('off')

            # Anomaly map
            im = axes[i, 1].imshow(anomaly_map, cmap='hot', interpolation='bilinear')
            axes[i, 1].set_title(f'Anomaly Map {i+1}')
            axes[i, 1].axis('off')
            plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

            # Overlay
            axes[i, 2].imshow(np.array(image.resize((256, 256))))
            axes[i, 2].imshow(anomaly_map, alpha=0.4, cmap='hot', interpolation='bilinear')
            axes[i, 2].set_title(f'Overlay {i+1}')
            axes[i, 2].axis('off')

            # Ground truth mask (if available)
            if test_masks and i < len(test_masks):
                mask_path = test_masks[idx]
                mask = Image.open(mask_path).convert('L')
                mask = np.array(mask.resize((256, 256)))
                axes[i, 3].imshow(mask, cmap='gray')
                axes[i, 3].set_title(f'Ground Truth {i+1}')
                axes[i, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def plot_roc_curves(y_true: List[int], y_scores: List[float],
                   save_path: Optional[str] = None):
    """Plot ROC and PR curves"""

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True)

    # PR curve
    ax2.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC/PR curves saved to {save_path}")

    plt.show()


def analyze_score_distribution(model: MultiScaleStudentTeacherFramework,
                             normal_images: List[str],
                             anomaly_images: List[str],
                             device: str = 'cuda',
                             save_path: Optional[str] = None):
    """Analyze and plot score distributions"""

    model = model.to(device)
    model.eval()

    transform = get_default_transform()

    normal_scores = _compute_scores_for_images(model, normal_images, transform, device)
    anomaly_scores = _compute_scores_for_images(model, anomaly_images, transform, device)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    ax1.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Score Distribution')
    ax1.legend()
    ax1.grid(True)

    # Box plot
    data = [normal_scores, anomaly_scores]
    labels = ['Normal', 'Anomaly']
    ax2.boxplot(data, labels=labels)
    ax2.set_ylabel('Anomaly Score')
    ax2.set_title('Score Box Plot')
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Score distribution plot saved to {save_path}")

    plt.show()

    # Print statistics
    print(f"\nScore Statistics:")
    print(f"Normal - Mean: {np.mean(normal_scores):.4f}, Std: {np.std(normal_scores):.4f}")
    print(f"Anomaly - Mean: {np.mean(anomaly_scores):.4f}, Std: {np.std(anomaly_scores):.4f}")
    print(f"Separation (Cohen's d): {_cohens_d(normal_scores, anomaly_scores):.4f}")


def evaluate_multiscale_contribution(model: MultiScaleStudentTeacherFramework,
                                   test_images: List[str],
                                   test_labels: List[int],
                                   device: str = 'cuda') -> Dict[str, float]:
    """Evaluate contribution of each scale to final performance"""

    model = model.to(device)
    model.eval()

    transform = get_default_transform()
    results = {}

    # Evaluate each scale individually
    for patch_size in model.patch_sizes:
        print(f"Evaluating scale {patch_size}...")

        scores = []
        with torch.no_grad():
            for img_path in tqdm(test_images):
                from PIL import Image
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)

                # Get teacher features for this scale
                teacher_features = model.forward_teacher_scale(image_tensor, patch_size)
                student_mean, student_var = model.forward_students_scale(image_tensor, patch_size)

                # Normalize features
                teacher_norm = (teacher_features - model.feature_mean) / (model.feature_std + 1e-8)
                student_norm = (student_mean - model.feature_mean) / (model.feature_std + 1e-8)

                # Compute anomaly score for this scale
                regression_error = torch.sum((student_norm - teacher_norm) ** 2, dim=-1)
                uncertainty = torch.sum(student_var, dim=-1)
                anomaly_score = regression_error + uncertainty

                score = torch.mean(anomaly_score).item()
                scores.append(score)

        # Compute AUC for this scale
        auc_score = roc_auc_score(test_labels, scores)
        results[f'scale_{patch_size}'] = auc_score
        print(f"Scale {patch_size} AUC: {auc_score:.4f}")

    # Evaluate combined (multiscale) performance
    print("Evaluating multiscale combination...")
    combined_scores = []

    with torch.no_grad():
        for img_path in tqdm(test_images):
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            anomaly_map = model.compute_anomaly_score(image_tensor)
            score = torch.mean(anomaly_map).item()
            combined_scores.append(score)

    combined_auc = roc_auc_score(test_labels, combined_scores)
    results['multiscale'] = combined_auc
    print(f"Multiscale AUC: {combined_auc:.4f}")

    return results


def benchmark_inference_speed(model: MultiScaleStudentTeacherFramework,
                            image_size: Tuple[int, int] = (256, 256),
                            batch_sizes: List[int] = [1, 4, 8],
                            num_runs: int = 100,
                            device: str = 'cuda') -> Dict[str, float]:
    """Benchmark inference speed of the model"""

    model = model.to(device)
    model.eval()

    results = {}

    for batch_size in batch_sizes:
        print(f"Benchmarking batch size {batch_size}...")

        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, *image_size, device=device)

        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = model.compute_anomaly_score(dummy_input)

        # Time inference
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None

        times = []
        for _ in range(num_runs):
            if device == 'cuda':
                start_time.record()
            else:
                import time
                start = time.time()

            with torch.no_grad():
                _ = model.compute_anomaly_score(dummy_input)

            if device == 'cuda':
                end_time.record()
                torch.cuda.synchronize()
                elapsed = start_time.elapsed_time(end_time)  # milliseconds
            else:
                elapsed = (time.time() - start) * 1000  # convert to milliseconds

            times.append(elapsed)

        avg_time = np.mean(times)
        fps = (1000 * batch_size) / avg_time  # frames per second

        results[f'batch_{batch_size}_ms'] = avg_time
        results[f'batch_{batch_size}_fps'] = fps

        print(f"Batch {batch_size}: {avg_time:.2f} ms, {fps:.2f} FPS")

    return results


# Helper functions
def _compute_scores_for_images(model: MultiScaleStudentTeacherFramework,
                             image_paths: List[str],
                             transform,
                             device: str,
                             batch_size: int = 8) -> List[float]:
    """Compute anomaly scores for a list of images"""

    scores = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []

            for img_path in batch_paths:
                from PIL import Image
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                batch_images.append(image_tensor)

            batch_tensor = torch.stack(batch_images).to(device)
            anomaly_maps = model.compute_anomaly_score(batch_tensor)

            # Compute mean score for each image
            for j in range(anomaly_maps.size(0)):
                score = torch.mean(anomaly_maps[j]).item()
                scores.append(score)

    return scores


def _compute_classification_metrics(y_true: List[int], y_scores: List[float]) -> Dict[str, float]:
    """Compute various classification metrics"""

    # AUC-ROC
    auc_roc = roc_auc_score(y_true, y_scores)

    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auc_pr = auc(recall, precision)

    # Find best threshold based on F1 score
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    best_f1 = 0
    best_threshold = 0

    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]

        if len(set(y_pred)) > 1:  # Avoid division by zero
            from sklearn.metrics import f1_score
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'best_f1': best_f1,
        'best_threshold': best_threshold
    }


def _cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(group2) - np.mean(group1)) / pooled_std
    return d


def generate_evaluation_report(model: MultiScaleStudentTeacherFramework,
                             test_data: Dict[str, List[str]],
                             device: str = 'cuda',
                             save_dir: str = './evaluation_results') -> Dict[str, any]:
    """
    Generate comprehensive evaluation report

    Args:
        model: Trained model
        test_data: Dictionary with keys 'normal_images', 'anomaly_images', 'masks' (optional)
        device: Device to run evaluation on
        save_dir: Directory to save results

    Returns:
        Comprehensive evaluation results
    """

    import os
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    # Image-level evaluation
    print("="*60)
    print("IMAGE-LEVEL EVALUATION")
    print("="*60)

    image_metrics = evaluate_image_level_detection(
        model,
        test_data['normal_images'],
        test_data['anomaly_images'],
        device
    )
    results['image_level'] = image_metrics

    # Score distribution analysis
    print("\nAnalyzing score distributions...")
    analyze_score_distribution(
        model,
        test_data['normal_images'],
        test_data['anomaly_images'],
        device,
        save_path=os.path.join(save_dir, 'score_distribution.png')
    )

    # Multiscale contribution analysis
    print("\nAnalyzing multiscale contributions...")
    all_images = test_data['normal_images'] + test_data['anomaly_images']
    all_labels = [0] * len(test_data['normal_images']) + [1] * len(test_data['anomaly_images'])

    multiscale_results = evaluate_multiscale_contribution(
        model, all_images, all_labels, device
    )
    results['multiscale_analysis'] = multiscale_results

    # Pixel-level evaluation (if masks available)
    if 'masks' in test_data and test_data['masks']:
        print("\n" + "="*60)
        print("PIXEL-LEVEL EVALUATION")
        print("="*60)

        pixel_metrics = evaluate_pixel_level_segmentation(
            model,
            test_data['anomaly_images'],
            test_data['masks'],
            device
        )
        results['pixel_level'] = pixel_metrics

    # Visualization
    print("\nGenerating visualizations...")
    visualize_detection_results(
        model,
        test_data['anomaly_images'][:10],  # First 10 anomaly images
        test_data.get('masks', None),
        num_samples=5,
        device=device,
        save_path=os.path.join(save_dir, 'detection_results.png')
    )

    # Performance benchmark
    print("\nBenchmarking inference speed...")
    speed_results = benchmark_inference_speed(model, device=device)
    results['performance'] = speed_results

    # Save results
    import json
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete! Results saved to {save_dir}")

    return results