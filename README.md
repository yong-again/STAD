# Student-Teacher Anomaly Detection

PyTorch implementation of "Uninformed Students: Student–Teacher Anomaly Detection with Discriminative Latent Embeddings" paper.

## Overview

This repository implements a multi-scale student-teacher framework for unsupervised anomaly detection and pixel-precise anomaly segmentation. The approach uses:

- **Teacher networks** that extract descriptive features using knowledge distillation from pretrained models
- **Student networks** that learn to regress teacher outputs on normal data
- **Multi-scale architecture** that detects anomalies at different receptive field sizes
- **Ensemble learning** to estimate uncertainty and improve robustness

## Project Structure

```
student_teacher_anomaly/
├── models.py              # Network architectures
├── datasets.py            # Data loading and processing
├── training.py            # Training functions
├── evaluation.py          # Evaluation and metrics
├── utils.py               # Utility functions
├── main.py               # Main training/inference script
├── config_example.json   # Example configuration
└── README.md             # This file
```

## Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy opencv-python pillow
pip install scikit-learn matplotlib tqdm
pip install pyyaml
```

### Optional Dependencies

```bash
pip install tensorboard  # For logging
pip install wandb       # For experiment tracking
```

## Quick Start

### 1. Training on MVTec AD Dataset

```bash
# Download MVTec AD dataset and extract to ./data/mvtec_ad/

# Train on bottle category
python main.py --mode train \
               --data_path ./data/mvtec_ad \
               --dataset_type mvtec \
               --category bottle \
               --teacher_epochs 50 \
               --student_epochs 100 \
               --batch_size 4
```

### 2. Testing a Trained Model

```bash
python main.py --mode test \
               --data_path ./data/mvtec_ad \
               --dataset_type mvtec \
               --category bottle \
               --checkpoint ./experiments/experiment_20241201_120000/checkpoints/complete_model.pth
```

### 3. Demo on Single Image

```bash
python main.py --mode demo \
               --demo_image ./test_image.jpg \
               --checkpoint ./experiments/experiment_20241201_120000/checkpoints/complete_model.pth
```

## Usage Examples

### Basic Training

```python
from models import MultiScaleStudentTeacherFramework
from datasets import create_dataset, get_default_transform
from training import train_full_pipeline
from torch.utils.data import DataLoader

# Initialize model
model = MultiScaleStudentTeacherFramework(
    patch_sizes=[17, 33, 65],
    num_students=3,
    output_dim=128
)

# Setup dataset
transform = get_default_transform()
dataset = create_dataset('mvtec', root_dir='./data', category='bottle', split='train')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Train complete pipeline
trained_model = train_full_pipeline(
    model=model,
    train_dataloader=dataloader,
    teacher_epochs=50,
    student_epochs=100,
    device='cuda'
)
```

### Inference on New Images

```python
from evaluation import visualize_detection_results
import torch

# Load trained model
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Visualize results
visualize_detection_results(
    model=model,
    test_images=['image1.jpg', 'image2.jpg'],
    device='cuda',
    save_path='results.png'
)
```

### Custom Dataset

```python
from datasets import CustomAnomalyDataset

# For custom dataset with normal/anomaly folders
dataset = CustomAnomalyDataset(
    normal_dir='./data/normal',
    anomaly_dir='./data/anomaly',
    mode='train'
)
```

## Configuration

Use JSON configuration files to manage experiments:

```json
{
  "model": {
    "patch_sizes": [17, 33, 65],
    "num_students": 3,
    "output_dim": 128
  },
  "training": {
    "teacher": {
      "epochs": 50,
      "lr": 0.0002
    },
    "student": {
      "epochs": 100,
      "lr": 0.0001
    },
    "batch_size": 4
  }
}
```

Load configuration:

```bash
python main.py --config config_example.json --mode train --data_path ./data
```

## Network Architecture

### Teacher Network (Feature Extractor)

- **Patch Size 65**: Conv(5×5) → MaxPool → Conv(5×5) → MaxPool → Conv(5×5) → MaxPool → Conv(4×4) → Conv(1×1)
- **Patch Size 33**: Conv(5×5) → MaxPool → Conv(5×5) → MaxPool → Conv(2×2) → Conv(4×4)
- **Patch Size 17**: Conv(6×6) → Conv(5×5) → Conv(5×5) → Conv(4×4)

### Student Networks

- Same architecture as teachers
- Ensemble of 3 networks per scale
- Trained to regress teacher features

### Training Process

1. **Teacher Pretraining**: Knowledge distillation from ResNet-18 + triplet learning + decorrelation loss
2. **Normalization Statistics**: Compute mean/std from teacher features on training data
3. **Student Training**: MSE regression loss to match normalized teacher features

## Evaluation Metrics

- **AUC-ROC**: Area under ROC curve for image-level detection
- **AUC-PR**: Area under Precision-Recall curve
- **PRO Score**: Per-Region Overlap for pixel-level segmentation
- **Pixel-level AUC**: AUC for pixel-wise anomaly segmentation

## Multi-Scale Detection

The framework combines predictions from multiple receptive field sizes:

- **Small patches (17×17)**: Detect fine-grained anomalies
- **Medium patches (33×33)**: Detect medium-scale defects  
- **Large patches (65×65)**: Detect large-scale anomalies

Final anomaly map = Average of all scales

## Datasets

### Supported Formats

1. **MVTec AD**: Industrial inspection dataset
2. **Custom**: Normal/anomaly folder structure
3. **Triplet**: For metric learning during teacher training

### Data Structure

```
dataset/
├── normal/           # Normal images
│   ├── image1.jpg
│   └── image2.jpg
└── anomaly/          # Anomaly images (optional)
    ├── defect1.jpg
    └── defect2.jpg
```

For MVTec AD:
```
mvtec_ad/
├── bottle/
│   ├── train/good/
│   ├── test/good/
│   ├── test/broken_large/
│   └── ground_truth/broken_large/
└── cable/
    └── ...
```

## Performance Tips

### Memory Optimization

- Use smaller batch sizes for large images
- Enable gradient checkpointing for deeper models
- Use mixed precision training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(images)
```

### Speed Optimization

- Use multiple workers in DataLoader
- Pin memory for GPU training
- Use compiled models (PyTorch 2.0+):

```python
model = torch.compile(model)
```

## Advanced Usage

### Custom Loss Functions

```python
def custom_teacher_loss(teacher_feat, decoded_feat, pretrained_feat):
    kd_loss = F.mse_loss(decoded_feat, pretrained_feat)
    # Add custom terms
    return kd_loss
```

### Experiment Tracking

```python
import wandb

wandb.init(project="anomaly-detection")
wandb.config.update(config)

# Log metrics during training
wandb.log({"train_loss": loss, "epoch": epoch})
```

### Model Export

```python
# Export to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# Export to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or image resolution
2. **Slow training**: Check DataLoader num_workers, use GPU, enable pin_memory
3. **Poor performance**: Ensure sufficient training data, check learning rates
4. **Visualization errors**: Install matplotlib, pillow dependencies

### Debug Mode

```bash
python main.py --mode train --batch_size 1 --teacher_epochs 1 --student_epochs 1
```

## Citation

```bibtex
@article{bergmann2019uninformed,
  title={Uninformed Students: Student–Teacher Anomaly Detection with Discriminative Latent Embeddings},
  author={Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  journal={arXiv preprint arXiv:1911.02357},
  year={2019}
}
```

## License

This implementation is for research purposes. Please check the original paper for licensing details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review closed issues on GitHub
- Create a new issue with detailed description and reproduction steps