"""
triplet_visualizer.py
Visualize triplet patch relationships
"""
import os
import sys
sys.path.append(os.path.abspath('.'))
import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets import TripletPatchDataset
import random
import traceback

def visualize_triplet_patches(dataset, num_samples=3, save_path=None):
    """Visualize triplet relationships"""

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Get a triplet
        anchor, positive, negative = dataset[i]

        # Convert to displayable format
        def tensor_to_img(tensor):
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = tensor * std + mean
            img = torch.clamp(img, 0, 1)
            return img.permute(1, 2, 0).numpy()

        # Display triplet
        axes[i, 0].imshow(tensor_to_img(anchor))
        axes[i, 0].set_title(f'Anchor {i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(tensor_to_img(positive))
        axes[i, 1].set_title(f'Positive {i+1}\n(Near anchor)')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(tensor_to_img(negative))
        axes[i, 2].set_title(f'Negative {i+1}\n(Different image)')
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Triplet visualization saved to {save_path}")

    plt.show()

def test_triplet_relationships():
    """Test that the triplet relationships are meaningful"""

    # Create dummy dataset
    dummy_paths = ["/workspace/anomaly/mvtec/grid/train/good/000.png",
                   "/workspace/anomaly/mvtec/grid/train/good/002.png",
                   "/workspace/anomaly/mvtec/grid/train/good/003.png"]  # Would be real paths
    dataset = TripletPatchDataset(dummy_paths, patch_size=32)

    print("🔍 Testing Triplet Relationships")
    print("="*50)
    for i in range(3):
        # Test multiple triplets
        try:
            anchor, positive, negative = dataset[i]

            print(f"  Anchor shape: {anchor.shape}")
            print(f"  Positive shape: {positive.shape}")
            print(f"  Negative shape: {negative.shape}")

            # Compute similarities (simple MSE distance)
            anchor_pos_dist = torch.mean((anchor - positive) ** 2).item()
            anchor_neg_dist = torch.mean((anchor - negative) ** 2).item()

            print(f"  Anchor-Positive distance: {anchor_pos_dist:.4f}")
            print(f"  Anchor-Negative distance: {anchor_neg_dist:.4f}")

            # Check if triplet constraint is satisfied
            margin = 0.1
            triplet_loss = max(0, margin + anchor_pos_dist - anchor_neg_dist)

            print(f"  Triplet loss (margin={margin}): {triplet_loss:.6f}")

            if anchor_pos_dist < anchor_neg_dist:
                print("  ✅ Good triplet: positive closer than negative")
            else:
                print("  ⚠️  Poor triplet: negative closer than positive")

        except Exception as e:
            print(f"  ❌ Error: {traceback.format_exc()}")

if __name__ == "__main__":
    test_triplet_relationships()
    image_paths = ["/workspace/anomaly/mvtec/metal_nut/train/good/000.png",
                   "/workspace/anomaly/mvtec/metal_nut/train/good/004.png",
                   "/workspace/anomaly/mvtec/metal_nut/train/good/005.png"]
    dataset = TripletPatchDataset(image_paths, patch_size=65)
    visualize_triplet_patches(dataset, num_samples=3, save_path='/workspace/anomaly/debug/debug_img/triplet')