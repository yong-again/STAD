"""
train.py
Train functions for Student-Teacher Anomaly Detection
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import torch.optim as optim
from tqdm import tqdm
import os

from anomaly.model import TeacherNetwork, MultiScaleStudentTeacherFramework


def train_teacher_with_distillation(
        teacher: TeacherNetwork,
        dataloader : DataLoader,
        num_epochs: int = 50,
        lr: float = 2e-4,
        device: str = 'cuda',
        save_path : Optional[str] = None,
        lambda_kd: float = 1.0,
        lambda_triplet: float = 1.0,
        lambda_decorr: float = 0.1):

    """
    Train Teacher network with knowledge distillation and triplet learning
    Args:
        teacher: Teacher network to train
        dataloader: Dataloader with triplet or single image data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_path: Path to save the trained model
        lambda_kd: Weight for knowledge distillation loss
        lambda_triplet: Weight for triplet loss
        lambda_decorr: Weight for decorrelation loss
    """

    teacher = teacher.to(device)
    teacher.train()

    optimizer = optim.Adam(teacher.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print(f"Training teacher network (patch_size: {teacher.receptive_field}...")
    print(f"Parameters: {sum(p.numel() for p in teacher.parameters())}...")

    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_data in pbar:
            if len(batch_data) == 3: # Triplet Data
                anchor, positive, negative = [x.to(device) for x in batch_data]

                optimizer.zero_grad()

                # Forward Pass
                anchor_feat, anchor_decoded = teacher(anchor)
                pos_feat, _ = teacher(positive)
                neg_feat, _ = teacher(negative)

                # Triplet Loss (Equation 2)
                margin = 1.0
                pos_dist = F.mse_loss(anchor_feat, pos_feat, reduction='none').sum(dim=1)
                neg_dist = F.mse_loss(anchor_feat, neg_feat, reduction='none').sum(dum=1)
                neg_dist_alt = F.mse_loss(pos_feat, neg_feat, reduction='none').sum(dim=1)

                min_neg_dist = torch.min(neg_dist, neg_dist_alt)
                triplet_loss = torch.clamp(margin + pos_dist - min_neg_dist, min=0.0).mean()

                # Decorrelation loss (Equations 5)
                if anchor_feat.size(0) > 1:
                    correlation_matrix = torch.corrcoef(anchor_feat.T)
                    decorr_loss = torch.sum(torch.abs(correlation_matrix) *
                                            (1 - torch.eye(correlation_matrix.size(0), device=device)))

                else:
                    decorr_loss = torch.tensor(0.0, device=device)

                # knowledge Distillation loss (Equation 1)
                pretrained_feat = teacher.extract_pretrained_features(anchor)
                kd_loss = F.mse_loss(anchor_decoded, pretrained_feat)

                # combined loss (Equation 6)
                loss = lambda_kd * kd_loss + lambda_triplet * triplet_loss + lambda_decorr * decorr_loss

            else: # single image data
                images = batch_data[0].to(device) if isinstance(batch_data, (list, tuple)) else batch_data.to(device)

                optimizer.zero_grad()

                features, decoded = teacher(images)

                # knowledge distillation loss
                pretrained_feat = teacher.extract_pretrained_features(images)
                kd_loss = F.mse_loss(decoded, pretrained_feat)

                # Decorrelation loss
                if features.size(0) > 1:
                    correlation_matrix = torch.corrcoef(features.T)
                    decorr_loss = torch.sum(torch.abs(correlation_matrix) *
                                            (1 - torch.eye(correlation_matrix.size(0), device=device)))

                else:
                    decorr_loss = torch.tensor(0.0, device=device)

                loss = lambda_kd * kd_loss + lambda_triplet * triplet_loss + lambda_decorr * decorr_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    scheduler.step()
    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:0.6f}")

    # Save model, if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_sate_dict': teacher.state_dict(),
            'train_losses': train_losses,
            'patch_size': teacher.receptive_field
        }, save_path)
        print(f"Teacher model save to {save_path}")

    return train_losses

def compute_normalization_stats(model: MultiScaleStudentTeacherFramework,
                               dataloader: DataLoader,
                               device: str = 'cuda'):
    """
    Compute normalization statistics from training data

    Args:
        model: MultiScaleStudentTeacherFramework
        dataloader: Dataloader with normal training data
        device: Device to compute on
    """

    model = model.to(device)
    model.eval()

    print("Computing normalization statistics...")

    all_features = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc='computing normalization statistics'):
            images = batch_data[0].to(device) if isinstance(batch_data, (list, tuple)) else batch_data.to(device)

            for patch_size in model.patch_sizes:
                teacher_features = model.forward_teacher_scale(images, patch_size)
                feature_flat = teacher_features.view(-1, model.output_dim)
                all_features.append(feature_flat)

    all_features = torch.cat(all_features, dim=0)

    # compute statistics

    mean = torch.mean(all_features, dim=0)
    std = torch.std(all_features, dim=0)

    model.feature_mean.copy_(mean)
    model.feature_std.copy_(std)
    model.stats_computed.fill_(True)

    print(f"Normalization statistics computed from {all_features.size(0)}")
    print(f"Feature Mean: {mean.mean().item():.4f} +- {mean.std().item():.4f}")
    print(f"Feature std: {std.mean().item():.4f} +- {std.std().item().item():.4f}")


def train_students(model: MultiScaleStudentTeacherFramework,
                   dataloader: DataLoader,
                   num_epochs: int = 100,
                   lr: float = 1e-4,
                   device: str = 'cuda',
                   save_path: str = None,):
    """
    Train Student Network to regress teacher feature
    Args:
        model : Multi-scale framework with trained teachers
        dataloader: DataLoader with normal training images
        num_epochs: Number of training epochs
        lr : Learning rate
        device: Device to train on
        save_path: Path to save the trained model
    """
    model = model.to(device)

    # Freeze Teachers
    for name, param in model.named_parameters():
        if 'teacher' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


    # compute normalization statistics first
    compute_normalization_stats(model, dataloader, device)

    # optimizer for student only
    student_params = [p for name, p in model.named_parameters() if 'student' in name and p.requires_grad]
    optimizer = optim.Adam(student_params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print("Training student Network...")
    print(f"Student parameters: {sum(p.numel() for p in student_params)}")

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        # Keep teacher is eval mode
        for name, module in model.named_modules():
            if 'teacher' in name:
                module.eva()

        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_data in pbar:
            images = batch_data[0].to(device) if isinstance(batch_data, (list, tuple)) else batch_data.to(device)
            optimizer.zero_grad()

            total_loss = 0

            for patch_size in model.patch_sizes:
                # Get Teacher features (frozen)
                with torch.no_grad():
                    teacher_features = model.forward_teacher_scale(images, patch_size)
                    teacher_norm = (teacher_features - model.feature_mean) / (model.feature_std.item())

                # Get student predictions
                student_mean, _ = model.forward_student_scale(images, patch_size)
                student_norm = (student_mean - model.feature_mean) / (model.feature_std + 1e-8)

                # Regression Loss (Equation 7)
                loss = F.mse_loss(student_norm, teacher_norm)
                total_loss += loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss
            num_batches += 1

            pbar.set_postfix({'Loss': f'{total_loss:.4f}'})

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.3f}, LR: {scheduler.get_last_lr()[0]:0.6f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'load_state_dict':model.state_dict(),
            'train_losses': train_losses,
            'patch_sizes': model.patch_sizes,
            'num_student': num_epochs,
            'output_dim': model.output_dim,
        }, save_path)

    return train_losses

def train_full_pipeline(model: MultiScaleStudentTeacherFramework,
                        train_dataloader: DataLoader,
                        triplet_dataloader: Optional[DataLoader] = None,
                        teacher_epochs: int = 50,
                        student_epochs: int = 100,
                        teacher_lr: float = 2e-4,
                        student_lr: float = 1e-4,
                        device: str = 'cuda',
                        save_dir: str = './checkpoints'
                        ):

    """
    Train the complete Student-Teacher pipepline

    Args:
        model : multi-scal framework,
        train_dataloader: Dataloader with normal training data
        triplet_dataloader: Dataloader with normal training data
        teacher_epochs: Number of epochs for teacher
        student_epochs: Number of epochs for student
        teacher_lr: Learning rate for teacher
        student_lr: Learning rate for student
        device: Device to train on
        save_dir: Path to save the trained model
    """

    os.makedirs(save_dir, exist_ok=True)

    print('='*60)
    print("STARTING FULL TRAINING PIPELINE")
    print('='*60)

    # Phase 1: Train teacher for each scale
    print('\nPhase 1: Training Teacher Network')
    print('-'*40)

    for patch_size in model.patch_sizes:
        print(f"\nTraining teacher for patch size {patch_size}...")
        teacher = model.teachers[f'teacher_{patch_size}']

        # use triplet dataloader if available, otherwise regular dataloader
        dataloader = triplet_dataloader if train_dataloader else train_dataloader

        teacher_save_path = os.path.join(save_dir, f'teacher_{patch_size}.pth')

        train_teacher_with_distillation(
            teacher=teacher,
            dataloader=dataloader,
            num_epochs=teacher_epochs,
            lr=teacher_lr,
            device=device,
            save_path=teacher_save_path,
        )
    # Phase 2: train student
    print("\nPhase 2: Training student Networks")
    print('-'*40)

    student_save_path = os.path.join(save_dir, 'complete_model.pth')

    train_students(
        model=model,
        dataloader=train_dataloader,
        num_epochs=student_epochs,
        lr=student_lr,
        device=device,
        save_path=student_save_path,
    )

    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE")
    print("="*60)

    return model

def load_trained_model(model: MultiScaleStudentTeacherFramework,
                       checkpoint_path: str,
                       device: str = 'cuda') -> MultiScaleStudentTeacherFramework:
    """
    load a trained model from checkpoint

    Args:
        model : Model instance to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to train on

    Returns:
        Loaded Model
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model Loaded from {checkpoint_path}")
    if 'patch_sizes' in checkpoint:
        print(f"Patch Sizes:{checkpoint['patch_sizes']}")

    if 'num_students' in checkpoint:
        print(f"Number of students:{checkpoint['num_students']}")

    return model

def resume_training(model: MultiScaleStudentTeacherFramework,
                    checkpoint_path: str,
                    dataloader: DataLoader,
                    additional_epochs: int = 50,
                    lr: float = 1e-4,
                    device: str = 'cuda'):
    """
    Resume Training from a checkpoint

    Args:
        model: Model Instance
        checkpoint_path: Path to checkpoint file
        dataloader: Dataloader with normal training data
        additional_epochs: Additional epochs to train
        lr: Learning rate for teacher
        device: Device to train on
    """

    # Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"Resume Training from {checkpoint_path}")

    # Continue Training students
    train_students(
        model=model,
        dataloader=dataloader,
        num_epochs=additional_epochs,
        lr=lr,
        device=device,
    )

    return model


# utility functions for training monitoring

def plot_training_losses(train_losses: list[float], save_path: Optional[str] = None):
    """
    plot training losses
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training loss plot save to {save_path}")

        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")


def get_learning_rate_scheduler(optimizer, mode: str='step'):
    """
    Get Learning rate Scheduler
    """
    if mode == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    elif mode == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    elif mode == ' reduce':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    else:
        raise ValueError(f"Unknown learning rate scheduler: {mode}")


def validate_model_ready_for_inference(model: MultiScaleStudentTeacherFramework):
    """
    validate that model is ready for inference
    """

    if not model.stats_computed:
        raise RuntimeError("Model not ready for inference. Normalization statistics not computed")

    # Check if teachers are trained (basic Check)
    for patch_size in model.patch_sizes:
        teacher = model.teachers[f'teacher_{patch_size}']
        if not any([p.requires_grad == False for p in teacher.parameters()]):
            print(f"Warning: Teacher {patch_size} may not be properly trained")

    print("Model Validation passed  ready for inference")





















