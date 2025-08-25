"""
models.py
Core network architectures for Student-Teacher Anomaly Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import List, Tuple


class NetworkBuilder(nn.Module):
    """Base network builder following the paper's architecture"""

    def __init__(self, receptive_field: int = 65, output_dim: int = 128, decode_dim: int = 512):
        super().__init__()
        self.receptive_field = receptive_field
        self.output_dim = output_dim
        self.feature_extractor = self._build_feature_extractor(receptive_field, output_dim)
        self.decoder = nn.Linear(output_dim, decode_dim)

    def _build_feature_extractor(self, receptive_field: int, output_dim: int):
        """Build feature extractor based on receptive field size"""

        if receptive_field == 65:
            # Fixed Table 4 architecture - ensuring 1x1 output
            network = nn.Sequential(
                # Conv1: 65x65x3 -> 61x61x128
                nn.Conv2d(3, 128, 5, 1, 0),
                nn.LeakyReLU(0.005),
                # MaxPool: 61x61 -> 30x30
                nn.MaxPool2d(2, 2),

                # Conv2: 30x30x128 -> 26x26x128
                nn.Conv2d(128, 128, 5, 1, 0),
                nn.LeakyReLU(0.005),
                # MaxPool: 26x26 -> 13x13
                nn.MaxPool2d(2, 2),

                # Conv3: 13x13x128 -> 10x10x128 (Fixed kernel size)
                nn.Conv2d(128, 128, 4, 1, 0),
                nn.LeakyReLU(0.005),
                # MaxPool: 10x10 -> 5x5
                nn.MaxPool2d(2, 2),

                # Conv4: 5x5x128 -> 2x2x256 (Fixed to get proper size)
                nn.Conv2d(128, 256, 4, 1, 0),
                nn.LeakyReLU(0.005),

                # Conv5: 2x2x256 -> 1x1x128 (Fixed kernel size)
                nn.Conv2d(256, output_dim, 2, 1, 0),
            )

        elif receptive_field == 33:
            network = nn.Sequential(
                # Conv1: 33x33x3 -> 29x29x128
                nn.Conv2d(3, 128, 5, 1, 0),
                nn.LeakyReLU(0.005),
                # MaxPool: 29x29 -> 14x14
                nn.MaxPool2d(2, 2),

                # Conv2: 14x14x128 -> 10x10x128
                nn.Conv2d(128, 128, 5, 1, 0),
                nn.LeakyReLU(0.005),
                # MaxPool: 10x10 -> 5x5
                nn.MaxPool2d(2, 2),

                # Conv3: 5x5x128 -> 4x4x256
                nn.Conv2d(128, 256, 2, 1, 0),
                nn.LeakyReLU(0.005),

                # Conv4: 4x4x256 -> 1x1x128 (Fixed stride)
                nn.Conv2d(256, output_dim, 4, 1, 0),
            )

        elif receptive_field == 17:
            network = nn.Sequential(
                # Conv1: 17x17x3 -> 12x12x128
                nn.Conv2d(3, 128, 6, 1, 0),
                nn.LeakyReLU(0.005),

                # Conv2: 12x12x128 -> 8x8x256
                nn.Conv2d(128, 256, 5, 1, 0),
                nn.LeakyReLU(0.005),

                # Conv3: 8x8x256 -> 4x4x256
                nn.Conv2d(256, 256, 5, 1, 0),
                nn.LeakyReLU(0.005),

                # Conv4: 4x4x256 -> 1x1x128
                nn.Conv2d(256, output_dim, 4, 1, 0),
            )

        else:
            raise ValueError(f"Unsupported receptive field: {receptive_field}")

        return network

    def forward(self, x):
        features = self.feature_extractor(x)
        # Feature shape: [batch, output_dim, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [batch, output_dim]

        # Decode layer for knowledge distillation
        decoded = self.decoder(features)

        return features, decoded


class TeacherNetwork(nn.Module):
    """Teacher Network for extracting descriptive features"""

    def __init__(self, receptive_field: int = 65, output_dim: int = 128):
        super().__init__()
        self.receptive_field = receptive_field
        self.output_dim = output_dim
        self.network_builder = NetworkBuilder(receptive_field, output_dim, decode_dim=512)

        # Pre-trained ResNet for knowledge distillation
        self.pretrained_net = resnet18(pretrained=True)
        self.pretrained_net.eval()
        for param in self.pretrained_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        features, decoded = self.network_builder(x)
        return features, decoded

    def extract_pretrained_features(self, x):
        """Extract features from pretrained ResNet for knowledge distillation"""
        with torch.no_grad():
            # Resize input to ResNet input size
            resized_x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

            # Extract features from ResNet
            x = self.pretrained_net.conv1(resized_x)
            x = self.pretrained_net.bn1(x)
            x = self.pretrained_net.relu(x)
            x = self.pretrained_net.maxpool(x)

            x = self.pretrained_net.layer1(x)
            x = self.pretrained_net.layer2(x)
            x = self.pretrained_net.layer3(x)
            x = self.pretrained_net.layer4(x)

            x = self.pretrained_net.avgpool(x)
            features = torch.flatten(x, 1)

        return features


class StudentNetwork(nn.Module):
    """Student Network with same architecture as Teacher"""

    def __init__(self, receptive_field: int = 65, output_dim: int = 128):
        super().__init__()
        self.receptive_field = receptive_field
        self.output_dim = output_dim
        self.network_builder = NetworkBuilder(receptive_field, output_dim, decode_dim=512)

    def forward(self, x):
        features, _ = self.network_builder(x)  # Only return features for students
        return features


class MultiScaleStudentTeacherFramework(nn.Module):
    """Multi-scale Student-Teacher Anomaly Detection Framework"""

    def __init__(self, patch_sizes: List[int] = [17, 33, 65],
                 num_students: int = 3, output_dim: int = 128):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.num_students = num_students
        self.output_dim = output_dim

        # Teachers for each scale - Fixed: use ModuleDict instead of ModuleList with dict
        self.teachers = nn.ModuleDict({
            f'teacher_{p}': TeacherNetwork(p, output_dim)
            for p in patch_sizes
        })

        # Students for each scale - Fixed: use ModuleDict instead of ModuleList with dict
        self.students = nn.ModuleDict({
            f'student_{p}_{i}': StudentNetwork(p, output_dim)
            for p in patch_sizes for i in range(num_students)
        })

        # Normalization statistics - Fixed: stats_computed should be tensor(False), not ones(False)
        self.register_buffer('feature_mean', torch.zeros(output_dim))
        self.register_buffer('feature_std', torch.ones(output_dim))
        self.register_buffer('stats_computed', torch.tensor(False))

    def extract_patches_dense(self, x: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, int, int]:
        """Extract dense overlapping patches for fully convolutional processing"""
        batch_size, channels, height, width = x.shape

        # Ensure input is large enough
        if height < patch_size or width < patch_size:
            raise ValueError(f"Input size {height}x{width} too small for patch size {patch_size}")

        # Use unfold to extract overlapping patches
        patches = x.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
        # patches shape: [batch, channels, h_out, w_out, patch_size, patch_size]

        h_out, w_out = patches.shape[2], patches.shape[3]
        patches = patches.contiguous().view(batch_size * h_out * w_out, channels, patch_size, patch_size)

        return patches, h_out, w_out

    def forward_teacher_scale(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Process image through teacher network at specific scale"""
        teacher = self.teachers[f'teacher_{patch_size}']
        teacher.eval()  # Teachers are always in eval mode during inference

        patches, h_out, w_out = self.extract_patches_dense(x, patch_size)

        with torch.no_grad():
            features, _ = teacher(patches)

        batch_size = x.shape[0]
        features = features.view(batch_size, h_out, w_out, self.output_dim)

        return features

    def forward_students_scale(self, x: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process image through student networks at specific scale"""
        patches, h_out, w_out = self.extract_patches_dense(x, patch_size)

        student_predictions = []
        for i in range(self.num_students):
            student = self.students[f'student_{patch_size}_{i}']
            pred = student(patches)
            student_predictions.append(pred)

        student_predictions = torch.stack(student_predictions, dim=0)

        # Calculate ensemble statistics
        mean_pred = torch.mean(student_predictions, dim=0)
        var_pred = torch.var(student_predictions, dim=0)

        batch_size = x.shape[0]
        mean_pred = mean_pred.view(batch_size, h_out, w_out, self.output_dim)
        var_pred = var_pred.view(batch_size, h_out, w_out, self.output_dim)

        return mean_pred, var_pred

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale anomaly score for input image"""
        if not self.stats_computed:
            raise RuntimeError("Normalization statistics not computed. Run compute_normalization_stats first.")

        self.eval()
        anomaly_maps = []

        for patch_size in self.patch_sizes:  # Fixed: use self.patch_sizes instead of self.patch_size
            # Get teacher features
            teacher_features = self.forward_teacher_scale(x, patch_size)

            # Get student predictions
            student_mean, student_var = self.forward_students_scale(x, patch_size)

            # Normalize features
            teacher_norm = (teacher_features - self.feature_mean) / (self.feature_std + 1e-8)
            student_norm = (student_mean - self.feature_mean) / (self.feature_std + 1e-8)

            # Compute regression error (Equation 8-9)
            regression_error = torch.sum((student_norm - teacher_norm) ** 2, dim=-1)

            # Compute predictive uncertainty (Equation 10) 
            uncertainty = torch.sum(student_var, dim=-1)

            # Resize to original image size for combination
            original_h, original_w = x.shape[2], x.shape[3]

            regression_error = F.interpolate(
                regression_error.unsqueeze(1),
                size=(original_h, original_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            uncertainty = F.interpolate(
                uncertainty.unsqueeze(1),
                size=(original_h, original_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            # Combine scores - Fixed: should be addition, not multiplication
            anomaly_score = regression_error + uncertainty
            anomaly_maps.append(anomaly_score)

        # Average across scales (Equation 12)
        final_anomaly_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)

        return final_anomaly_map