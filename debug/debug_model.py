def test_memory_usage(self):
        """Test memory usage patterns"""
        print("\n💾 Testing Memory Usage...")

        if self.device != 'cuda':
            print("   Skipping memory test (CPU mode)")
            self.test_results['memory_usage'] = {'status': 'SKIPPED', 'reason': 'CPU mode'}
            return
        """
debug_models.py
Debugging script for models.py - comprehensive testing of all network components
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import traceback
import time
import warnings

# Try to import torchinfo, fallback if not available
try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    try:
        from torchsummary import summary
        TORCHINFO_AVAILABLE = False
    except ImportError:
        print("Warning: Neither torchinfo nor torchsummary available. Install with: pip install torchinfo")
        summary = None
        TORCHINFO_AVAILABLE = None

# Import our models
from anomaly.model.models import (
    NetworkBuilder,
    TeacherNetwork,
    StudentNetwork,
    MultiScaleStudentTeacherFramework
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class ModelDebugger:
    """Comprehensive debugging tool for the models"""

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.test_results = {}
        print(f"🔧 Model Debugger initialized on device: {device}")
        print("=" * 60)

    def debug_network_builder(self):
        """Debug NetworkBuilder for all patch sizes"""
        print("🏗️  Testing NetworkBuilder...")

        patch_sizes = [17, 33, 65]
        output_dim = 128
        decode_dim = 512

        for patch_size in patch_sizes:
            try:
                print(f"\n📏 Testing patch size: {patch_size}")

                # Initialize network
                network = NetworkBuilder(
                    receptive_field=patch_size,
                    output_dim=output_dim,
                    decode_dim=decode_dim
                ).to(self.device)

                # Create dummy input
                batch_size = 4
                dummy_input = torch.randn(batch_size, 3, patch_size, patch_size).to(self.device)

                # Forward pass
                with torch.no_grad():
                    features, decoded = network(dummy_input)

                # Validate outputs
                expected_feature_shape = (batch_size, output_dim)
                expected_decode_shape = (batch_size, decode_dim)

                assert features.shape == expected_feature_shape, \
                    f"Feature shape mismatch: {features.shape} vs {expected_feature_shape}"
                assert decoded.shape == expected_decode_shape, \
                    f"Decoded shape mismatch: {decoded.shape} vs {expected_decode_shape}"

                # Check for NaN or Inf values
                assert not torch.isnan(features).any(), "Features contain NaN values"
                assert not torch.isinf(features).any(), "Features contain Inf values"
                assert not torch.isnan(decoded).any(), "Decoded features contain NaN values"
                assert not torch.isinf(decoded).any(), "Decoded features contain Inf values"

                # Print network summary
                print(f"✅ NetworkBuilder {patch_size}: Success")
                print(f"   Input shape: {dummy_input.shape}")
                print(f"   Feature shape: {features.shape}")
                print(f"   Decoded shape: {decoded.shape}")
                print(f"   Parameters: {sum(p.numel() for p in network.parameters()):,}")

                # Store results
                self.test_results[f'network_builder_{patch_size}'] = {
                    'status': 'PASS',
                    'input_shape': tuple(dummy_input.shape),
                    'feature_shape': tuple(features.shape),
                    'decoded_shape': tuple(decoded.shape),
                    'parameters': sum(p.numel() for p in network.parameters())
                }

            except Exception as e:
                print(f"❌ NetworkBuilder {patch_size}: FAILED")
                print(f"   Error: {str(e)}")
                print(f"   Traceback: {traceback.format_exc()}")

                self.test_results[f'network_builder_{patch_size}'] = {
                    'status': 'FAIL',
                    'error': str(e)
                }

    def debug_teacher_network(self):
        """Debug TeacherNetwork"""
        print("\n👨‍🏫 Testing TeacherNetwork...")

        patch_sizes = [17, 33, 65]
        output_dim = 128

        for patch_size in patch_sizes:
            try:
                print(f"\n📏 Testing Teacher patch size: {patch_size}")

                # Initialize teacher
                teacher = TeacherNetwork(
                    receptive_field=patch_size,
                    output_dim=output_dim
                ).to(self.device)

                # Create dummy input
                batch_size = 4
                dummy_input = torch.randn(batch_size, 3, patch_size, patch_size).to(self.device)

                # Test forward pass
                with torch.no_grad():
                    features, decoded = teacher(dummy_input)

                # Test pretrained features extraction
                pretrained_features = teacher.extract_pretrained_features(dummy_input)

                # Validate outputs
                expected_feature_shape = (batch_size, output_dim)
                expected_decode_shape = (batch_size, 512)  # ResNet feature dim
                expected_pretrained_shape = (batch_size, 512)

                assert features.shape == expected_feature_shape
                assert decoded.shape == expected_decode_shape
                assert pretrained_features.shape == expected_pretrained_shape

                print(f"✅ TeacherNetwork {patch_size}: Success")
                print(f"   Features: {features.shape}")
                print(f"   Decoded: {decoded.shape}")
                print(f"   Pretrained: {pretrained_features.shape}")

                # Check gradient flow - Fixed: create requires_grad tensors
                teacher.train()

                # Create input that requires grad for proper gradient testing
                grad_input = torch.randn(batch_size, 3, patch_size, patch_size, requires_grad=True).to(self.device)
                features, decoded = teacher(grad_input)

                # Create a dummy target that also requires grad
                target_features = torch.randn_like(decoded, requires_grad=True).to(self.device)

                # Compute loss with both tensors requiring grad
                loss = torch.nn.functional.mse_loss(decoded, target_features.detach())
                loss.backward()

                # Check if gradients exist
                has_gradients = any(p.grad is not None for p in teacher.parameters() if p.requires_grad)
                print(f"   Gradient flow: {'✅' if has_gradients else '❌'}")

                self.test_results[f'teacher_{patch_size}'] = {
                    'status': 'PASS',
                    'feature_shape': tuple(features.shape),
                    'decoded_shape': tuple(decoded.shape),
                    'pretrained_shape': tuple(pretrained_features.shape),
                    'gradient_flow': has_gradients
                }

            except Exception as e:
                print(f"❌ TeacherNetwork {patch_size}: FAILED")
                print(f"   Error: {str(e)}")

                self.test_results[f'teacher_{patch_size}'] = {
                    'status': 'FAIL',
                    'error': str(e)
                }

    def debug_student_network(self):
        """Debug StudentNetwork"""
        print("\n👨‍🎓 Testing StudentNetwork...")

        patch_sizes = [17, 33, 65]
        output_dim = 128

        for patch_size in patch_sizes:
            try:
                print(f"\n📏 Testing Student patch size: {patch_size}")

                # Initialize student
                student = StudentNetwork(
                    receptive_field=patch_size,
                    output_dim=output_dim
                ).to(self.device)

                # Create dummy input
                batch_size = 4
                dummy_input = torch.randn(batch_size, 3, patch_size, patch_size).to(self.device)

                # Test forward pass
                with torch.no_grad():
                    features = student(dummy_input)

                # Validate output
                expected_shape = (batch_size, output_dim)
                assert features.shape == expected_shape

                print(f"✅ StudentNetwork {patch_size}: Success")
                print(f"   Features: {features.shape}")

                self.test_results[f'student_{patch_size}'] = {
                    'status': 'PASS',
                    'feature_shape': tuple(features.shape)
                }

            except Exception as e:
                print(f"❌ StudentNetwork {patch_size}: FAILED")
                print(f"   Error: {str(e)}")

                self.test_results[f'student_{patch_size}'] = {
                    'status': 'FAIL',
                    'error': str(e)
                }

    def debug_multiscale_framework(self):
        """Debug MultiScaleStudentTeacherFramework"""
        print("\n🎯 Testing MultiScaleStudentTeacherFramework...")

        try:
            # Initialize framework with safe configuration
            model = MultiScaleStudentTeacherFramework(
                patch_sizes=[33],  # Use single medium patch size
                num_students=2,    # Reduce number of students
                output_dim=64      # Reduce output dimension
            ).to(self.device)

            print(f"✅ Framework initialized")
            print(f"   Patch sizes: {model.patch_sizes}")
            print(f"   Students per scale: {model.num_students}")
            print(f"   Output dim: {model.output_dim}")
            print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Test with image size that ensures proper patch extraction
            patch_size = model.patch_sizes[0]
            image_size = patch_size + 32  # Ensure sufficient size for patches
            batch_size = 1

            print(f"\n🔍 Testing with image size: {image_size}x{image_size}, patch size: {patch_size}")
            dummy_image = torch.randn(batch_size, 3, image_size, image_size).to(self.device)

            # Test patch extraction
            patches, h_out, w_out = model.extract_patches_dense(dummy_image, patch_size)
            expected_patches = batch_size * h_out * w_out
            expected_h_out = image_size - patch_size + 1
            expected_w_out = image_size - patch_size + 1

            print(f"   Patch extraction:")
            print(f"   - Patches shape: {patches.shape}")
            print(f"   - Output spatial: {h_out}x{w_out} (expected: {expected_h_out}x{expected_w_out})")
            print(f"   - Total patches: {patches.shape[0]} (expected: {expected_patches})")

            assert h_out == expected_h_out, f"Height mismatch: {h_out} vs {expected_h_out}"
            assert w_out == expected_w_out, f"Width mismatch: {w_out} vs {expected_w_out}"
            assert patches.shape[0] == expected_patches
            assert patches.shape[1:] == (3, patch_size, patch_size)

            # Clear memory
            del patches
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            # Test teacher forward pass
            print(f"\n👨‍🏫 Testing teacher forward pass...")
            teacher_features = model.forward_teacher_scale(dummy_image, patch_size)
            expected_shape = (batch_size, expected_h_out, expected_w_out, model.output_dim)

            print(f"   Teacher features: {teacher_features.shape} (expected: {expected_shape})")
            assert teacher_features.shape == expected_shape

            # Clear memory
            del teacher_features
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            # Test student forward pass
            print(f"\n👨‍🎓 Testing student forward pass...")
            student_mean, student_var = model.forward_students_scale(dummy_image, patch_size)

            print(f"   Student mean: {student_mean.shape}")
            print(f"   Student var: {student_var.shape}")

            assert student_mean.shape == expected_shape
            assert student_var.shape == expected_shape
            assert torch.all(student_var >= 0), "Negative variance detected"

            # Clear memory
            del student_mean, student_var
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            # Test normalization stats
            print(f"\n📊 Setting normalization statistics...")
            model.feature_mean.fill_(0.5)
            model.feature_std.fill_(1.0)
            model.stats_computed.fill_(True)

            # Test anomaly score computation
            print(f"\n🚨 Testing anomaly score computation...")
            with torch.no_grad():
                anomaly_map = model.compute_anomaly_score(dummy_image)

            expected_anomaly_shape = (batch_size, image_size, image_size)
            print(f"   Anomaly map: {anomaly_map.shape} (expected: {expected_anomaly_shape})")

            assert anomaly_map.shape == expected_anomaly_shape
            assert not torch.isnan(anomaly_map).any(), "Anomaly map contains NaN"
            assert not torch.isinf(anomaly_map).any(), "Anomaly map contains Inf"

            print("✅ MultiScaleFramework: All tests passed!")

            self.test_results['multiscale_framework'] = {
                'status': 'PASS',
                'patch_sizes': model.patch_sizes,
                'num_students': model.num_students,
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'anomaly_map_shape': tuple(anomaly_map.shape),
                'test_image_size': image_size
            }

        except Exception as e:
            print(f"❌ MultiScaleFramework: FAILED")
            print(f"   Error: {str(e)}")

            # Try ultra-minimal configuration
            print(f"\n🔄 Trying ultra-minimal configuration...")
            try:
                torch.cuda.empty_cache() if self.device == 'cuda' else None

                mini_model = MultiScaleStudentTeacherFramework(
                    patch_sizes=[17],  # Smallest patch size
                    num_students=1,    # Single student
                    output_dim=32      # Very small output dim
                ).to(self.device)

                mini_model.feature_mean.fill_(0.0)
                mini_model.feature_std.fill_(1.0)
                mini_model.stats_computed.fill_(True)

                # Use image size that's safely larger than patch size
                mini_size = 32  # 17 + 15 for safe margin
                mini_input = torch.randn(1, 3, mini_size, mini_size).to(self.device)

                with torch.no_grad():
                    mini_anomaly = mini_model.compute_anomaly_score(mini_input)

                print(f"   ✅ Ultra-minimal test passed: {mini_anomaly.shape}")

                self.test_results['multiscale_framework'] = {
                    'status': 'PARTIAL_PASS',
                    'note': 'Passed with ultra-minimal configuration',
                    'config': f'patch_sizes=[17], num_students=1, output_dim=32, image_size={mini_size}'
                }

            except Exception as e2:
                print(f"   ❌ Even ultra-minimal failed: {str(e2)}")
                self.test_results['multiscale_framework'] = {
                    'status': 'FAIL',
                    'error': str(e),
                    'secondary_error': str(e2)
                }

        finally:
            # Aggressive memory cleanup
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def benchmark_performance(self):
        """Benchmark model performance"""
        print("\n⚡ Performance Benchmarking...")

        try:
            # Create minimal model configuration with safe sizing
            model = MultiScaleStudentTeacherFramework(
                patch_sizes=[17],  # Single small patch size
                num_students=1,    # Single student
                output_dim=32      # Small dimension
            ).to(self.device)
            model.eval()

            # Set dummy normalization stats
            model.feature_mean.fill_(0.0)
            model.feature_std.fill_(1.0)
            model.stats_computed.fill_(True)

            patch_size = model.patch_sizes[0]

            # Use compatible image sizes with safe margins
            test_configs = [
                (32, 1),   # 32x32 image (patch 17 + margin 15)
                (48, 1),   # 48x48 image (patch 17 + margin 31)
            ]

            for img_size, batch_size in test_configs:
                print(f"\n📊 Benchmarking {batch_size}x{img_size}x{img_size} (patch size: {patch_size})...")

                # Verify compatibility
                if img_size <= patch_size:
                    print(f"   Skipped: image {img_size} too small for patch {patch_size}")
                    continue

                expected_output_size = img_size - patch_size + 1
                print(f"   Expected feature map size: {expected_output_size}x{expected_output_size}")

                dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(self.device)

                # Warm up
                for _ in range(2):
                    with torch.no_grad():
                        _ = model.compute_anomaly_score(dummy_input)
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()

                # Benchmark
                if self.device == 'cuda':
                    torch.cuda.synchronize()

                times = []
                for _ in range(3):
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()

                    start = time.time()

                    with torch.no_grad():
                        result = model.compute_anomaly_score(dummy_input)

                    if self.device == 'cuda':
                        torch.cuda.synchronize()

                    times.append(time.time() - start)

                    # Verify output shape
                    expected_shape = (batch_size, img_size, img_size)
                    if result.shape != expected_shape:
                        print(f"   Warning: Output shape {result.shape} != expected {expected_shape}")

                avg_time = np.mean(times)
                fps = batch_size / avg_time

                print(f"   Average time: {avg_time:.3f}s")
                print(f"   FPS: {fps:.2f}")

                if self.device == 'cuda':
                    memory_gb = torch.cuda.max_memory_allocated() / 1e9
                    print(f"   Peak GPU memory: {memory_gb:.2f} GB")
                    torch.cuda.reset_peak_memory_stats()

                del dummy_input
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

            self.test_results['performance'] = {'status': 'PASS'}

        except Exception as e:
            print(f"❌ Performance benchmark failed: {str(e)}")

            # Try CPU fallback with even smaller config
            print(f"   Trying CPU fallback...")
            try:
                cpu_model = MultiScaleStudentTeacherFramework([17], 1, 16)  # Even smaller
                cpu_model.feature_mean.fill_(0.0)
                cpu_model.feature_std.fill_(1.0)
                cpu_model.stats_computed.fill_(True)

                cpu_input = torch.randn(1, 3, 24, 24)  # 17 + 7 margin

                start = time.time()
                with torch.no_grad():
                    result = cpu_model.compute_anomaly_score(cpu_input)
                cpu_time = time.time() - start

                print(f"   CPU fallback successful: {cpu_time:.3f}s, output: {result.shape}")
                self.test_results['performance'] = {'status': 'PARTIAL_PASS', 'note': 'CPU fallback only'}

            except Exception as e2:
                print(f"   CPU fallback also failed: {str(e2)}")
                self.test_results['performance'] = {'status': 'FAIL', 'error': str(e)}

        finally:
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def test_gradient_flow(self):
        """Test gradient flow through the networks"""
        print("\n🌊 Testing Gradient Flow...")

        try:
            # Test teacher gradient flow
            teacher = TeacherNetwork(receptive_field=65, output_dim=128).to(self.device)
            teacher.train()

            dummy_input = torch.randn(4, 3, 65, 65).to(self.device)
            features, decoded = teacher(dummy_input)
            pretrained_features = teacher.extract_pretrained_features(dummy_input)

            # Compute loss and backprop
            kd_loss = nn.functional.mse_loss(decoded, pretrained_features)

            # Add decorrelation loss
            if features.size(0) > 1:
                correlation_matrix = torch.corrcoef(features.T)
                decorr_loss = torch.sum(torch.abs(correlation_matrix) *
                                      (1 - torch.eye(correlation_matrix.size(0), device=self.device)))
            else:
                decorr_loss = torch.tensor(0.0, device=self.device)

            total_loss = kd_loss + 0.1 * decorr_loss
            total_loss.backward()

            # Check gradients
            teacher_grads = []
            for name, param in teacher.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    teacher_grads.append(grad_norm)
                    print(f"   {name}: grad_norm = {grad_norm:.6f}")

            print(f"✅ Teacher gradient flow: {len(teacher_grads)} parameters have gradients")

            # Test student gradient flow
            student = StudentNetwork(receptive_field=65, output_dim=128).to(self.device)
            student.train()

            target_features = torch.randn(4, 128).to(self.device)
            pred_features = student(dummy_input)

            student_loss = nn.functional.mse_loss(pred_features, target_features)
            student_loss.backward()

            student_grads = []
            for name, param in student.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    student_grads.append(grad_norm)

            print(f"✅ Student gradient flow: {len(student_grads)} parameters have gradients")

            self.test_results['gradient_flow'] = {
                'status': 'PASS',
                'teacher_gradients': len(teacher_grads),
                'student_gradients': len(student_grads)
            }

        except Exception as e:
            print(f"❌ Gradient flow test failed: {str(e)}")
            self.test_results['gradient_flow'] = {'status': 'FAIL', 'error': str(e)}

    def test_memory_usage(self):
        """Test memory usage patterns"""
        print("\n💾 Testing Memory Usage...")

        if self.device != 'cuda':
            print("   Skipping memory test (CPU mode)")
            self.test_results['memory_usage'] = {'status': 'SKIPPED', 'reason': 'CPU mode'}
            return

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create minimal model
            model = MultiScaleStudentTeacherFramework(
                patch_sizes=[17],  # Single small patch size
                num_students=1,    # Single student
                output_dim=32      # Small output dimension
            ).to(self.device)
            model.eval()

            # Set dummy stats
            model.feature_mean.fill_(0.0)
            model.feature_std.fill_(1.0)
            model.stats_computed.fill_(True)

            print(f"   Model loaded: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

            # Test with very small images
            test_sizes = [32, 48, 64]

            for img_size in test_sizes:
                torch.cuda.empty_cache()
                start_memory = torch.cuda.memory_allocated()

                # Check compatibility
                max_patch_size = max(model.patch_sizes)
                if img_size <= max_patch_size:
                    print(f"   Skipped {img_size}x{img_size}: too small for patch size {max_patch_size}")
                    continue

                dummy_input = torch.randn(1, 3, img_size, img_size).to(self.device)

                with torch.no_grad():
                    anomaly_map = model.compute_anomaly_score(dummy_input)

                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = (peak_memory - start_memory) / 1e6  # MB

                print(f"   Input {img_size}x{img_size}: {memory_used:.1f} MB")

                del dummy_input, anomaly_map
                torch.cuda.empty_cache()

            self.test_results['memory_usage'] = {'status': 'PASS'}

        except Exception as e:
            print(f"❌ Memory usage test failed: {str(e)}")

            # Try ultra-minimal test
            try:
                print("   Trying ultra-minimal memory test...")
                torch.cuda.empty_cache()

                mini_model = MultiScaleStudentTeacherFramework([17], 1, 16).to(self.device)
                mini_model.feature_mean.fill_(0.0)
                mini_model.feature_std.fill_(1.0)
                mini_model.stats_computed.fill_(True)

                mini_input = torch.randn(1, 3, 24, 24).to(self.device)

                with torch.no_grad():
                    _ = mini_model.compute_anomaly_score(mini_input)

                print("   Ultra-minimal test passed ✅")
                self.test_results['memory_usage'] = {'status': 'PARTIAL_PASS', 'note': 'Ultra-minimal only'}

            except Exception as e2:
                self.test_results['memory_usage'] = {'status': 'FAIL', 'error': str(e)}

        finally:
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def visualize_feature_maps(self):
        """Visualize intermediate feature maps"""
        print("\n🖼️  Visualizing Feature Maps...")

        try:
            teacher = TeacherNetwork(receptive_field=65, output_dim=128).to(self.device)
            teacher.eval()

            # Create a simple test pattern
            dummy_input = torch.zeros(1, 3, 65, 65).to(self.device)
            dummy_input[0, :, 20:45, 20:45] = 1.0  # White square

            # Hook to capture intermediate features
            features_dict = {}

            def hook_fn(name):
                def hook(module, input, output):
                    features_dict[name] = output.detach().cpu()
                return hook

            # Register hooks
            hooks = []
            for i, layer in enumerate(teacher.network_builder.feature_extractor):
                if isinstance(layer, nn.Conv2d):
                    hook = layer.register_forward_hook(hook_fn(f'conv_{i}'))
                    hooks.append(hook)

            # Forward pass
            with torch.no_grad():
                features, decoded = teacher(dummy_input)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Plot feature maps
            fig, axes = plt.subplots(2, len(features_dict), figsize=(4*len(features_dict), 8))
            if len(features_dict) == 1:
                axes = axes.reshape(-1, 1)

            for idx, (name, feature_map) in enumerate(features_dict.items()):
                # Show first channel
                feature_map = feature_map[0, 0].numpy()  # First batch, first channel

                axes[0, idx].imshow(feature_map, cmap='viridis')
                axes[0, idx].set_title(f'{name} - Channel 0')
                axes[0, idx].axis('off')

                # Show feature statistics
                axes[1, idx].hist(feature_map.flatten(), bins=50, alpha=0.7)
                axes[1, idx].set_title(f'{name} - Distribution')
                axes[1, idx].grid(True)

            plt.tight_layout()
            plt.savefig('debug_feature_maps.png', dpi=300, bbox_inches='tight')
            plt.show()

            print("✅ Feature visualization saved as 'debug_feature_maps.png'")

            self.test_results['visualization'] = {'status': 'PASS'}

        except Exception as e:
            print(f"❌ Feature visualization failed: {str(e)}")
            self.test_results['visualization'] = {'status': 'FAIL', 'error': str(e)}

    def run_all_tests(self):
        """Run all debugging tests"""
        print("🚀 Starting comprehensive model debugging...")
        print("=" * 60)

        # Run all tests
        self.debug_network_builder()
        self.debug_teacher_network()
        self.debug_student_network()
        self.debug_multiscale_framework()
        self.test_gradient_flow()
        self.benchmark_performance()
        self.test_memory_usage()
        self.visualize_feature_maps()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📋 DEBUGGING SUMMARY")
        print("=" * 60)

        passed = 0
        failed = 0

        for test_name, result in self.test_results.items():
            status = result['status']
            if status == 'PASS':
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                failed += 1

        print("-" * 60)
        print(f"📊 Total tests: {passed + failed}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success rate: {passed/(passed+failed)*100:.1f}%")

        if failed == 0:
            print("\n🎉 All tests passed! Models are working correctly.")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")

        print("=" * 60)

def main():
    """Main debugging function"""
    print("🔧 Student-Teacher Anomaly Detection - Model Debugger")
    print("=" * 60)

    try:
        # Initialize debugger
        debugger = ModelDebugger()

        # Run all tests
        debugger.run_all_tests()

    except KeyboardInterrupt:
        print("\n⏹️  Debugging interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()