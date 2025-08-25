"""
shape_debug.py
Shape debugging tool to trace tensor shapes through the network
"""

import torch
import torch.nn as nn
from anomaly.model.models import NetworkBuilder

def trace_network_shapes(receptive_field: int = 65, output_dim: int = 128):
    """Trace shapes through the network architecture"""

    print(f"🔍 Tracing shapes for receptive field: {receptive_field}")
    print("=" * 60)

    # Create network
    network = NetworkBuilder(receptive_field, output_dim)

    # Create dummy input
    x = torch.randn(1, 3, receptive_field, receptive_field)
    print(f"Input shape: {x.shape}")
    print("-" * 40)

    # Trace through feature extractor
    for i, layer in enumerate(network.feature_extractor):
        x = layer(x)
        layer_name = layer.__class__.__name__

        if isinstance(layer, nn.Conv2d):
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            print(f"Layer {i:2d} ({layer_name}): {x.shape} | K:{kernel_size} S:{stride} P:{padding}")
        elif isinstance(layer, nn.MaxPool2d):
            kernel_size = layer.kernel_size
            stride = layer.stride
            print(f"Layer {i:2d} ({layer_name}): {x.shape} | K:{kernel_size} S:{stride}")
        else:
            print(f"Layer {i:2d} ({layer_name}): {x.shape}")

    print("-" * 40)
    print(f"Feature extractor output: {x.shape}")

    # Apply squeeze operations
    squeezed = x.squeeze(-1).squeeze(-1)
    print(f"After squeeze: {squeezed.shape}")

    # Check decoder compatibility
    decoder_input_dim = squeezed.shape[1]
    expected_input_dim = output_dim

    print(f"Decoder expects: {expected_input_dim} features")
    print(f"Actually got: {decoder_input_dim} features")

    if decoder_input_dim != expected_input_dim:
        print("❌ SHAPE MISMATCH DETECTED!")
        print(f"   Expected: {expected_input_dim}")
        print(f"   Got: {decoder_input_dim}")

        # Calculate the actual final spatial size
        final_spatial_h, final_spatial_w = x.shape[2], x.shape[3]
        total_features = x.shape[1] * final_spatial_h * final_spatial_w

        print(f"   Final spatial size: {final_spatial_h}x{final_spatial_w}")
        print(f"   Total flattened features: {total_features}")

        # Suggest fixes
        print("\n💡 Suggested fixes:")
        if final_spatial_h > 1 or final_spatial_w > 1:
            print(f"   1. Add Global Average Pooling: torch.nn.AdaptiveAvgPool2d(1)")
            print(f"   2. Or flatten and add Linear layer: nn.Linear({total_features}, {output_dim})")

        return False
    else:
        print("✅ Shapes are compatible!")
        return True


def fix_network_architecture(receptive_field: int = 65, output_dim: int = 128):
    """Create a fixed network architecture"""

    print(f"\n🔧 Creating fixed architecture for receptive field: {receptive_field}")

    if receptive_field == 65:
        # Calculate exact dimensions step by step
        # Input: 65x65
        # Conv1 (5x5): 65-5+1 = 61x61
        # MaxPool (2x2): 61//2 = 30x30
        # Conv2 (5x5): 30-5+1 = 26x26
        # MaxPool (2x2): 26//2 = 13x13
        # Conv3 (5x5): 13-5+1 = 9x9
        # MaxPool (2x2): 9//2 = 4x4
        # Conv4 (4x4): 4-4+1 = 1x1

        network = nn.Sequential(
            # Input: 65x65x3
            nn.Conv2d(3, 128, 5, 1, 0),  # -> 61x61x128
            nn.LeakyReLU(0.005),
            nn.MaxPool2d(2, 2),  # -> 30x30x128

            nn.Conv2d(128, 128, 5, 1, 0),  # -> 26x26x128
            nn.LeakyReLU(0.005),
            nn.MaxPool2d(2, 2),  # -> 13x13x128

            nn.Conv2d(128, 256, 5, 1, 0),  # -> 9x9x256
            nn.LeakyReLU(0.005),
            nn.MaxPool2d(2, 2),  # -> 4x4x256

            nn.Conv2d(256, 256, 4, 1, 0),  # -> 1x1x256
            nn.LeakyReLU(0.005),

            nn.Conv2d(256, output_dim, 1, 1, 0),  # -> 1x1x128
        )

    elif receptive_field == 33:
        # Input: 33x33
        # Conv1 (5x5): 33-5+1 = 29x29
        # MaxPool (2x2): 29//2 = 14x14
        # Conv2 (5x5): 14-5+1 = 10x10
        # MaxPool (2x2): 10//2 = 5x5
        # Conv3 (2x2): 5-2+1 = 4x4
        # Conv4 (4x4): 4-4+1 = 1x1

        network = nn.Sequential(
            nn.Conv2d(3, 128, 5, 1, 0),  # -> 29x29x128
            nn.LeakyReLU(0.005),
            nn.MaxPool2d(2, 2),  # -> 14x14x128

            nn.Conv2d(128, 256, 5, 1, 0),  # -> 10x10x256
            nn.LeakyReLU(0.005),
            nn.MaxPool2d(2, 2),  # -> 5x5x256

            nn.Conv2d(256, 256, 2, 1, 0),  # -> 4x4x256
            nn.LeakyReLU(0.005),

            nn.Conv2d(256, output_dim, 4, 1, 0),  # -> 1x1x128
        )

    elif receptive_field == 17:
        # Input: 17x17
        # Conv1 (6x6): 17-6+1 = 12x12
        # Conv2 (5x5): 12-5+1 = 8x8
        # Conv3 (5x5): 8-5+1 = 4x4
        # Conv4 (4x4): 4-4+1 = 1x1

        network = nn.Sequential(
            nn.Conv2d(3, 128, 6, 1, 0),  # -> 12x12x128
            nn.LeakyReLU(0.005),

            nn.Conv2d(128, 256, 5, 1, 0),  # -> 8x8x256
            nn.LeakyReLU(0.005),

            nn.Conv2d(256, 256, 5, 1, 0),  # -> 4x4x256
            nn.LeakyReLU(0.005),

            nn.Conv2d(256, output_dim, 4, 1, 0),  # -> 1x1x128
        )

    else:
        raise ValueError(f"Unsupported receptive field: {receptive_field}")

    return network


def test_fixed_architecture():
    """Test the fixed architecture"""
    print("🧪 Testing fixed architectures...")

    for receptive_field in [17, 33, 65]:
        print(f"\n📏 Testing receptive field: {receptive_field}")

        try:
            # Create fixed network
            feature_extractor = fix_network_architecture(receptive_field, 128)
            decoder = nn.Linear(128, 512)

            # Test forward pass
            x = torch.randn(4, 3, receptive_field, receptive_field)
            print(f"Input: {x.shape}")

            # Forward through feature extractor
            features = feature_extractor(x)
            print(f"Features before squeeze: {features.shape}")

            # Squeeze
            features_squeezed = features.squeeze(-1).squeeze(-1)
            print(f"Features after squeeze: {features_squeezed.shape}")

            # Through decoder
            decoded = decoder(features_squeezed)
            print(f"Decoded: {decoded.shape}")

            print(f"✅ Receptive field {receptive_field}: SUCCESS")

        except Exception as e:
            print(f"❌ Receptive field {receptive_field}: FAILED - {str(e)}")


if __name__ == "__main__":
    # Test current architectures
    for rf in [17, 33, 65]:
        success = trace_network_shapes(rf)
        if not success:
            print(f"❌ Current architecture for {rf} has shape mismatch")
        print()

    print("\n" + "=" * 60)

    # Test fixed architectures
    test_fixed_architecture()