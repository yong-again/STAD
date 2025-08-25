"""
evaluation.py
Evaluation functions and metrics for Student-Teacher Anomaly Detections
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
        normal_images: List of normal image path
        anomaly_images: List of anomaly image path
        device: Device to turn inference on
        batch_size: Batch size for inference

    Returns:
        Dictionary with evaluation metrics
    """

    model = model.to(device)
    model.eval()

    transform = get_default_transform()

    normal_score = []
    anomaly_score = []

    print("Evaluating normal images...")
    normal_scores = _compute_scores_for_images(model, normal_images, normal_images)


# Helper Functions
def _compute_scores_for_images(model: MultiScaleStudentTeacherFramework,
                               image_paths: List[str],
                               transform,
                               device: str,
                               batch_size: int = 8) -> List[float]:
    """
    compute anomaly scores for a list of images
    """

    scores = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []

            for img_path in batch_paths:
                from PIL import Image
                image = Image.open(img_path).conver('RGB')
                image_tensor = transform(image)
                batch_images.append(image_tensor)

            batch_tensor = torch.stakc(batch_images).to(device)
            anomaly_maps = model.compute_anomaly_score(batch_tensor)

            # Compute mean score for each images
            for j in range(anomaly_maps.size(0)):
                score = torch.mean(anomaly_maps[j]).item()
                scores.append(score)

    return scores

def _compute_classification_metrics(y_true: List[int], y_scores: List[float]) -> Dict[str, float]:
    pass



