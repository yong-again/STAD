"""
main.py
Main script for Student-Teacher Anomaly Detection training and inference
"""

import argparse
import os
import sys
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

# Import our modules
from models import MultiScaleStudentTeacherFramework
from datasets import create_dataset, get_default_transform, get_augmented_transform
from training import train_full_pipeline, load_trained_model
from evaluation import generate_evaluation_report, visualize_detection_results
from utils import (get_device, setup_experiment, get_default_config,
                   find_images_in_directory, validate_dataset_structure,
                   create_dataset_summary, save_config, load_config)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Student-Teacher Anomaly Detection')

    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'demo'],
                        default='train', help='Mode to run the script in')

    # Data paths
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--dataset_type', type=str, default='mvtec',
                        choices=['mvtec', 'custom', 'anomaly'],
                        help='Type of dataset')
    parser.add_argument('--category', type=str, default=None,
                        help='Category for MVTec dataset')

    # Model configuration
    parser.add_argument('--patch_sizes', type=int, nargs='+', default=[17, 33, 65],
                        help='Patch sizes for multi-scale detection')
    parser.add_argument('--num_students', type=int, default=3,
                        help='Number of student networks per scale')
    parser.add_argument('--output_dim', type=int, default=128,
                        help='Output dimension of feature vectors')

    # Training parameters
    parser.add_argument('--teacher_epochs', type=int, default=50,
                        help='Number of epochs to train teachers')
    parser.add_argument('--student_epochs', type=int, default=100,
                        help='Number of epochs to train students')
    parser.add_argument('--teacher_lr', type=float, default=2e-4,
                        help='Learning rate for teacher training')
    parser.add_argument('--student_lr', type=float, default=1e-4,
                        help='Learning rate for student training')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')

    # Device and experiment settings
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--experiment_dir', type=str, default='./experiments',
                        help='Base directory for experiments')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for testing/inference')

    # Evaluation settings
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations during evaluation')
    parser.add_argument('--num_vis_samples', type=int, default=5,
                        help='Number of samples to visualize')

    # Demo settings
    parser.add_argument('--demo_image', type=str, default=None,
                        help='Path to single image for demo mode')

    return parser.parse_args()


def setup_data_loaders(args, config: Dict[str, Any]):
    """Setup data loaders based on configuration"""

    print("Setting up data loaders...")

    # Get transforms
    transform = get_default_transform(config['data']['image_size'][0])
    augmented_transform = get_augmented_transform(config['data']['image_size'][0])

    loaders = {}

    if args.dataset_type == 'mvtec':
        if not args.category:
            raise ValueError("Category must be specified for MVTec dataset")

        # Training data (normal images only)
        train_dataset = create_dataset(
            'mvtec',
            root_dir=args.data_path,
            category=args.category,
            split='train',
            transform=augmented_transform
        )

        # Test data (normal + anomaly images)
        test_dataset = create_dataset(
            'mvtec',
            root_dir=args.data_path,
            category=args.category,
            split='test',
            transform=transform,
            include_mask=True
        )

        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=1,  # For evaluation
            shuffle=False,
            num_workers=2
        )

        print(f"MVTec {args.category}: {len(train_dataset)} train, {len(test_dataset)} test images")

    elif args.dataset_type == 'custom':
        # Custom dataset with normal/anomaly directories
        normal_dir = os.path.join(args.data_path, 'normal')
        anomaly_dir = os.path.join(args.data_path, 'anomaly')

        if not os.path.exists(normal_dir):
            raise ValueError(f"Normal images directory not found: {normal_dir}")

        # Training data (normal images)
        train_dataset = create_dataset(
            'custom',
            normal_dir=normal_dir,
            anomaly_dir=anomaly_dir if os.path.exists(anomaly_dir) else None,
            transform=augmented_transform,
            mode='train'
        )

        # Test data
        test_dataset = create_dataset(
            'custom',
            normal_dir=normal_dir,
            anomaly_dir=anomaly_dir if os.path.exists(anomaly_dir) else None,
            transform=transform,
            mode='test'
        )

        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2
        )

        print(f"Custom dataset: {len(train_dataset)} train, {len(test_dataset)} test images")

    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    return loaders


def train_model(args, config: Dict[str, Any], experiment_dirs: Dict[str, str]):
    """Train the complete model pipeline"""

    print("=" * 60)
    print("TRAINING MODE")
    print("=" * 60)

    # Setup device
    device = get_device(prefer_gpu=(args.device != 'cpu'))
    config['training']['device'] = device

    # Setup data loaders
    data_loaders = setup_data_loaders(args, config)

    # Initialize model
    model = MultiScaleStudentTeacherFramework(
        patch_sizes=config['model']['patch_sizes'],
        num_students=config['model']['num_students'],
        output_dim=config['model']['output_dim']
    )

    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train the complete pipeline
    trained_model = train_full_pipeline(
        model=model,
        train_dataloader=data_loaders['train'],
        teacher_epochs=config['training']['teacher']['epochs'],
        student_epochs=config['training']['student']['epochs'],
        teacher_lr=config['training']['teacher']['lr'],
        student_lr=config['training']['student']['lr'],
        device=device,
        save_dir=experiment_dirs['checkpoints']
    )

    print("\nTraining completed successfully!")

    # Evaluate on test set if available
    if 'test' in data_loaders:
        print("\nEvaluating on test set...")
        # Prepare test data for evaluation
        test_data = prepare_test_data_for_evaluation(data_loaders['test'], args)

        # Generate evaluation report
        eval_results = generate_evaluation_report(
            model=trained_model,
            test_data=test_data,
            device=device,
            save_dir=experiment_dirs['results']
        )

        print(f"Evaluation results saved to {experiment_dirs['results']}")

    return trained_model


def test_model(args, config: Dict[str, Any], experiment_dirs: Dict[str, str]):
    """Test a trained model"""

    print("=" * 60)
    print("TESTING MODE")
    print("=" * 60)

    if not args.checkpoint:
        raise ValueError("Checkpoint path must be provided for testing mode")

    if not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint not found: {args.checkpoint}")

    # Setup device
    device = get_device(prefer_gpu=(args.device != 'cpu'))

    # Initialize model
    model = MultiScaleStudentTeacherFramework(
        patch_sizes=config['model']['patch_sizes'],
        num_students=config['model']['num_students'],
        output_dim=config['model']['output_dim']
    )

    # Load trained model
    model = load_trained_model(model, args.checkpoint, device)

    # Setup data loaders
    data_loaders = setup_data_loaders(args, config)

    if 'test' not in data_loaders:
        raise ValueError("Test data loader not available")

    # Prepare test data
    test_data = prepare_test_data_for_evaluation(data_loaders['test'], args)

    # Generate comprehensive evaluation report
    eval_results = generate_evaluation_report(
        model=model,
        test_data=test_data,
        device=device,
        save_dir=experiment_dirs['results']
    )

    print("Testing completed successfully!")
    print(f"Results saved to {experiment_dirs['results']}")

    return eval_results


def demo_mode(args, config: Dict[str, Any]):
    """Demo mode for single image inference"""

    print("=" * 60)
    print("DEMO MODE")
    print("=" * 60)

    if not args.checkpoint:
        raise ValueError("Checkpoint path must be provided for demo mode")

    if not args.demo_image:
        raise ValueError("Demo image path must be provided for demo mode")

    if not os.path.exists(args.demo_image):
        raise ValueError(f"Demo image not found: {args.demo_image}")

    # Setup device
    device = get_device(prefer_gpu=(args.device != 'cpu'))

    # Initialize and load model
    model = MultiScaleStudentTeacherFramework(
        patch_sizes=config['model']['patch_sizes'],
        num_students=config['model']['num_students'],
        output_dim=config['model']['output_dim']
    )

    model = load_trained_model(model, args.checkpoint, device)

    # Visualize results for demo image
    print(f"Processing demo image: {args.demo_image}")

    visualize_detection_results(
        model=model,
        test_images=[args.demo_image],
        num_samples=1,
        device=device,
        save_path='demo_result.png'
    )

    print("Demo completed! Result saved as 'demo_result.png'")


def prepare_test_data_for_evaluation(test_dataloader, args):
    """Prepare test data in the format expected by evaluation functions"""

    normal_images = []
    anomaly_images = []
    masks = []

    for batch in test_dataloader:
        if len(batch) == 3:  # With masks
            images, labels, batch_masks = batch
            for i, label in enumerate(labels):
                # Note: This is a simplified approach
                # In practice, you'd need to store actual image paths
                if label == 0:
                    normal_images.append(f"normal_image_{len(normal_images)}")
                else:
                    anomaly_images.append(f"anomaly_image_{len(anomaly_images)}")
                    if batch_masks[i] is not None:
                        masks.append(f"mask_{len(masks)}")
        else:  # Without masks
            images, labels = batch
            for label in labels:
                if label == 0:
                    normal_images.append(f"normal_image_{len(normal_images)}")
                else:
                    anomaly_images.append(f"anomaly_image_{len(anomaly_images)}")

    test_data = {
        'normal_images': normal_images,
        'anomaly_images': anomaly_images
    }

    if masks:
        test_data['masks'] = masks

    return test_data


def main():
    """Main function"""

    args = parse_args()

    # Load configuration
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        print("Using default configuration")
        config = get_default_config()

    # Override config with command line arguments
    config['model']['patch_sizes'] = args.patch_sizes
    config['model']['num_students'] = args.num_students
    config['model']['output_dim'] = args.output_dim
    config['training']['teacher']['epochs'] = args.teacher_epochs
    config['training']['student']['epochs'] = args.student_epochs
    config['training']['teacher']['lr'] = args.teacher_lr
    config['training']['student']['lr'] = args.student_lr
    config['training']['batch_size'] = args.batch_size

    # Validate dataset
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path does not exist: {args.data_path}")

    print(f"Dataset path: {args.data_path}")
    print(f"Dataset type: {args.dataset_type}")
    if args.category:
        print(f"Category: {args.category}")

    # Setup experiment directory (only for train mode)
    if args.mode == 'train':
        experiment_dirs = setup_experiment(config, args.experiment_dir)
        print(f"Experiment directory: {experiment_dirs['experiment']}")
    else:
        experiment_dirs = {
            'results': './results',
            'visualizations': './visualizations'
        }
        for dir_path in experiment_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    try:
        # Run based on mode
        if args.mode == 'train':
            model = train_model(args, config, experiment_dirs)
        elif args.mode == 'test':
            results = test_model(args, config, experiment_dirs)
        elif args.mode == 'demo':
            demo_mode(args, config)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        print("\nScript completed successfully!")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()