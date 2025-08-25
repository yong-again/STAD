"""
debug_extract_patch.py
Debugging tool for extract_random_patch function
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import random


class ExtractPatchDebugger:
    """Debugger for extract_random_patch functionality"""

    def __init__(self):
        self.test_results = {}

    def _extract_random_patch(self, image: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Extract a random patch from the image - DEBUG VERSION"""
        print(f"🔍 _extract_random_patch called:")
        print(f"   Input image shape: {image.shape}")
        print(f"   Requested patch size: {patch_size}")

        _, h, w = image.shape
        print(f"   Image dimensions: H={h}, W={w}")

        # Check if image is too small
        if h < patch_size or w < patch_size:
            print(f"   ⚠️  Image too small! Resizing...")
            print(f"   Original size: {h}x{w}")

            new_h = max(h, patch_size)
            new_w = max(w, patch_size)
            print(f"   New size: {new_h}x{new_w}")

            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            _, h, w = image.shape
            print(f"   After resize: {image.shape}")

        # Calculate valid range for top-left corner
        max_top = h - patch_size
        max_left = w - patch_size

        print(f"   Valid range for top: 0 to {max_top}")
        print(f"   Valid range for left: 0 to {max_left}")

        if max_top < 0 or max_left < 0:
            print(f"   ❌ ERROR: Negative range detected!")
            print(f"   max_top: {max_top}, max_left: {max_left}")
            return None

        # Generate random coordinates
        top = random.randint(0, max_top)
        left = random.randint(0, max_left)

        print(f"   Selected coordinates: top={top}, left={left}")
        print(f"   Patch region: [{top}:{top + patch_size}, {left}:{left + patch_size}]")

        # Extract patch
        patch = image[:, top:top + patch_size, left:left + patch_size]
        print(f"   Extracted patch shape: {patch.shape}")

        # Verify patch size
        expected_shape = (image.shape[0], patch_size, patch_size)
        if patch.shape != expected_shape:
            print(f"   ❌ ERROR: Patch shape mismatch!")
            print(f"   Expected: {expected_shape}")
            print(f"   Got: {patch.shape}")
            return None

        print(f"   ✅ Patch extracted successfully")
        return patch

    def test_basic_functionality(self):
        """Test basic patch extraction functionality"""
        print("🧪 Testing Basic Functionality")
        print("=" * 50)

        # Test case 1: Normal sized image
        print("\n📋 Test 1: Normal sized image")
        image = torch.randn(3, 64, 64)
        patch_size = 32

        patch = self._extract_random_patch(image, patch_size)
        success = patch is not None and patch.shape == (3, 32, 32)
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")

        self.test_results['normal_image'] = success

        # Test case 2: Small image (needs resizing)
        print("\n📋 Test 2: Small image (needs resizing)")
        small_image = torch.randn(3, 16, 16)
        patch_size = 32

        patch = self._extract_random_patch(small_image, patch_size)
        success = patch is not None and patch.shape == (3, 32, 32)
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")

        self.test_results['small_image'] = success

        # Test case 3: Exact size image
        print("\n📋 Test 3: Exact size image")
        exact_image = torch.randn(3, 32, 32)
        patch_size = 32

        patch = self._extract_random_patch(exact_image, patch_size)
        success = patch is not None and patch.shape == (3, 32, 32)
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")

        self.test_results['exact_image'] = success

        # Test case 4: Rectangular image
        print("\n📋 Test 4: Rectangular image")
        rect_image = torch.randn(3, 40, 60)
        patch_size = 32

        patch = self._extract_random_patch(rect_image, patch_size)
        success = patch is not None and patch.shape == (3, 32, 32)
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")

        self.test_results['rectangular_image'] = success

    def test_edge_cases(self):
        """Test edge cases"""
        print("\n🚨 Testing Edge Cases")
        print("=" * 50)

        # Test case 1: Very small image
        print("\n📋 Edge Test 1: Very small image")
        tiny_image = torch.randn(3, 8, 8)
        patch_size = 32

        patch = self._extract_random_patch(tiny_image, patch_size)
        success = patch is not None and patch.shape == (3, 32, 32)
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")

        self.test_results['tiny_image'] = success

        # Test case 2: Large patch size
        print("\n📋 Edge Test 2: Large patch size")
        image = torch.randn(3, 100, 100)
        large_patch_size = 80

        patch = self._extract_random_patch(image, large_patch_size)
        success = patch is not None and patch.shape == (3, 80, 80)
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")

        self.test_results['large_patch'] = success

        # Test case 3: Single pixel image
        print("\n📋 Edge Test 3: Single pixel image")
        pixel_image = torch.randn(3, 1, 1)
        patch_size = 16

        patch = self._extract_random_patch(pixel_image, patch_size)
        success = patch is not None and patch.shape == (3, 16, 16)
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")

        self.test_results['single_pixel'] = success

    def test_randomness(self):
        """Test randomness of patch extraction"""
        print("\n🎲 Testing Randomness")
        print("=" * 50)

        image = torch.randn(3, 100, 100)
        patch_size = 32
        num_tests = 10

        coordinates = []

        print(f"\n📋 Extracting {num_tests} random patches...")

        for i in range(num_tests):
            # Mock the random coordinate generation
            h, w = image.shape[1], image.shape[2]
            max_top = h - patch_size
            max_left = w - patch_size

            top = random.randint(0, max_top)
            left = random.randint(0, max_left)
            coordinates.append((top, left))

            print(f"   Patch {i + 1}: top={top}, left={left}")

        # Check if coordinates are different
        unique_coords = set(coordinates)
        randomness_score = len(unique_coords) / len(coordinates)

        print(f"\n   Total coordinates: {len(coordinates)}")
        print(f"   Unique coordinates: {len(unique_coords)}")
        print(f"   Randomness score: {randomness_score:.2f}")

        success = randomness_score > 0.5  # At least 50% should be unique
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")

        self.test_results['randomness'] = success

    def test_batch_processing(self):
        """Test batch processing of patch extraction"""
        print("\n📦 Testing Batch Processing")
        print("=" * 50)

        batch_size = 4
        patch_size = 32

        print(f"\n📋 Processing batch of {batch_size} images...")

        batch_success = True

        for i in range(batch_size):
            # Different sized images in batch
            sizes = [(64, 64), (48, 48), (80, 60), (32, 32)]
            h, w = sizes[i]

            image = torch.randn(3, h, w)
            print(f"\n   Image {i + 1}: {image.shape}")

            patch = self._extract_random_patch(image, patch_size)

            if patch is None or patch.shape != (3, patch_size, patch_size):
                print(f"   ❌ Failed for image {i + 1}")
                batch_success = False
            else:
                print(f"   ✅ Success for image {i + 1}")

        print(f"\n   Batch processing: {'✅ PASS' if batch_success else '❌ FAIL'}")
        self.test_results['batch_processing'] = batch_success

    def visualize_patch_extraction(self, save_path: str = None):
        """Visualize patch extraction process"""
        print("\n🖼️  Visualizing Patch Extraction")
        print("=" * 50)

        # Create a test image with a pattern
        image = torch.zeros(3, 80, 80)

        # Add some patterns
        image[0, 10:20, 10:20] = 1.0  # Red square
        image[1, 30:40, 30:40] = 1.0  # Green square
        image[2, 50:60, 50:60] = 1.0  # Blue square

        # Add some diagonal lines
        for i in range(80):
            if i < 80:
                image[:, i, i] = 0.5

        patch_size = 32
        num_patches = 4

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Show original image
        axes[0, 0].imshow(image.permute(1, 2, 0))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Extract and show patches
        for i in range(num_patches):
            row = (i // 2)
            col = (i % 2) + 1

            if row < 2 and col < 3:
                # Extract patch
                _, h, w = image.shape
                max_top = h - patch_size
                max_left = w - patch_size

                top = random.randint(0, max_top)
                left = random.randint(0, max_left)

                patch = image[:, top:top + patch_size, left:left + patch_size]

                axes[row, col].imshow(patch.permute(1, 2, 0))
                axes[row, col].set_title(f'Patch {i + 1}\nPos: ({top}, {left})')
                axes[row, col].axis('off')

                # Draw rectangle on original image
                rect = plt.Rectangle((left, top), patch_size, patch_size,
                                     linewidth=2, edgecolor='red', facecolor='none')
                axes[0, 0].add_patch(rect)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Visualization saved to: {save_path}")

        plt.show()

    def test_performance(self):
        """Test performance of patch extraction"""
        print("\n⚡ Testing Performance")
        print("=" * 50)

        import time

        # Test parameters
        image_sizes = [(64, 64), (128, 128), (256, 256)]
        patch_sizes = [16, 32, 64]
        num_iterations = 100

        for img_h, img_w in image_sizes:
            for patch_size in patch_sizes:
                if patch_size > min(img_h, img_w):
                    continue

                print(f"\n📋 Image: {img_h}x{img_w}, Patch: {patch_size}x{patch_size}")

                image = torch.randn(3, img_h, img_w)

                # Time the extraction
                start_time = time.time()

                for _ in range(num_iterations):
                    patch = self._extract_random_patch(image, patch_size)

                end_time = time.time()

                avg_time = (end_time - start_time) / num_iterations * 1000  # ms
                print(f"   Average time: {avg_time:.3f} ms")

    def run_all_tests(self):
        """Run all debugging tests"""
        print("🚀 Starting Extract Random Patch Debugging")
        print("=" * 60)

        self.test_basic_functionality()
        self.test_edge_cases()
        self.test_randomness()
        self.test_batch_processing()
        self.test_performance()

        # Show visualization
        try:
            self.visualize_patch_extraction('patch_extraction_debug.png')
        except Exception as e:
            print(f"   Visualization failed: {e}")

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📋 DEBUGGING SUMMARY")
        print("=" * 60)

        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")

        print("-" * 60)
        print(f"📊 Total tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {total - passed}")
        print(f"📈 Success rate: {passed / total * 100:.1f}%")

        if passed == total:
            print("\n🎉 All tests passed! extract_random_patch is working correctly.")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")


# Helper function to test with actual TripletPatchDataset
def test_with_dataset():
    """Test extract_random_patch with actual dataset context"""
    print("\n🗂️  Testing with Dataset Context")
    print("=" * 50)

    from anomaly.dataset.datasets import TripletPatchDataset

    # Create dummy image paths (you can replace with real paths)
    dummy_paths = ["dummy1.jpg", "dummy2.jpg", "dummy3.jpg"]

    try:
        # Create dataset instance
        dataset = TripletPatchDataset(dummy_paths, patch_size=32)

        # Test the _extract_random_patch method
        test_image = torch.randn(3, 64, 64)

        # Apply transform
        test_image = dataset.transform(Image.fromarray((test_image.permute(1, 2, 0).numpy() * 255).astype('uint8')))

        patch = dataset._extract_random_patch(test_image)

        print(f"   Dataset patch extraction: {patch.shape}")
        print(f"   Expected: (3, 32, 32)")
        print(f"   Result: {'✅ PASS' if patch.shape == (3, 32, 32) else '❌ FAIL'}")

    except Exception as e:
        print(f"   Dataset test failed: {e}")
        print("   This is expected if you don't have the actual dataset module")


def main():
    """Main debugging function"""
    # Set random seed for reproducible results
    random.seed(42)
    torch.manual_seed(42)

    debugger = ExtractPatchDebugger()
    debugger.run_all_tests()

    # Test with dataset context
    test_with_dataset()


if __name__ == "__main__":
    main()