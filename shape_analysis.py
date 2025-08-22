"""
shape_analysis.py
Analyze shape flow through the MultiScale framework
"""

import torch
from models import MultiScaleStudentTeacherFramework
import warnings
warnings.filterwarnings("ignore")


def analyze_shape_flow():
    """Analyze the shape flow through the framework"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔍 Shape Analysis on {device}")
    print("=" * 60)

    # Create minimal model
    model = MultiScaleStudentTeacherFramework(
        patch_sizes=[17],
        num_students=1,
        output_dim=32
    ).to(device)

    # Set dummy stats
    model.feature_mean.fill_(0.0)
    model.feature_std.fill_(1.0)
    model.stats_computed.fill_(True)

    # Test with different image sizes
    test_sizes = [24, 32, 48]
    patch_size = 17

    for img_size in test_sizes:
        print(f"\n📏 Testing image size: {img_size}x{img_size}")
        print(f"    Patch size: {patch_size}")

        if img_size <= patch_size:
            print(f"    Skipped: image too small")
            continue

        x = torch.randn(1, 3, img_size, img_size).to(device)

        # Step 1: Extract patches
        patches, h_out, w_out = model.extract_patches_dense(x, patch_size)
        expected_h = img_size - patch_size + 1
        expected_w = img_size - patch_size + 1

        print(f"    Step 1 - Patch extraction:")
        print(f"    - Input: {x.shape}")
        print(f"    - Patches: {patches.shape}")
        print(f"    - Output spatial: {h_out}x{w_out} (expected: {expected_h}x{expected_w})")

        # Step 2: Teacher forward
        teacher_features = model.forward_teacher_scale(x, patch_size)
        print(f"    Step 2 - Teacher features: {teacher_features.shape}")

        # Step 3: Student forward
        student_mean, student_var = model.forward_students_scale(x, patch_size)
        print(f"    Step 3 - Student mean: {student_mean.shape}")
        print(f"    Step 3 - Student var: {student_var.shape}")

        # Step 4: Check if they match
        if teacher_features.shape != student_mean.shape:
            print(f"    ❌ MISMATCH: Teacher {teacher_features.shape} != Student {student_mean.shape}")
        else:
            print(f"    ✅ Teacher and Student shapes match")

        # Step 5: Compute scores
        teacher_norm = (teacher_features - model.feature_mean) / (model.feature_std + 1e-8)
        student_norm = (student_mean - model.feature_mean) / (model.feature_std + 1e-8)

        regression_error = torch.sum((student_norm - teacher_norm) ** 2, dim=-1)
        uncertainty = torch.sum(student_var, dim=-1)

        print(f"    Step 4 - Regression error: {regression_error.shape}")
        print(f"    Step 4 - Uncertainty: {uncertainty.shape}")

        if regression_error.shape != uncertainty.shape:
            print(f"    ❌ SCORE MISMATCH: Regression {regression_error.shape} != Uncertainty {uncertainty.shape}")

            # Try to debug the issue
            print(f"    Debug info:")
            print(f"    - teacher_norm shape: {teacher_norm.shape}")
            print(f"    - student_norm shape: {student_norm.shape}")
            print(f"    - student_var shape: {student_var.shape}")
            print(f"    - Sum along dim=-1 should give same result...")

            # Test the sum operation
            test_sum_teacher = torch.sum(teacher_norm, dim=-1)
            test_sum_student = torch.sum(student_norm, dim=-1)
            test_sum_var = torch.sum(student_var, dim=-1)

            print(f"    - sum(teacher_norm, dim=-1): {test_sum_teacher.shape}")
            print(f"    - sum(student_norm, dim=-1): {test_sum_student.shape}")
            print(f"    - sum(student_var, dim=-1): {test_sum_var.shape}")
        else:
            print(f"    ✅ Regression error and uncertainty shapes match")

        print("-" * 40)


def test_individual_components():
    """Test individual components step by step"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🧪 Individual Component Testing on {device}")
    print("=" * 60)

    # Create model
    model = MultiScaleStudentTeacherFramework([17], 1, 32).to(device)

    img_size = 32
    patch_size = 17
    x = torch.randn(1, 3, img_size, img_size).to(device)

    print(f"Input: {x.shape}")
    print(f"Patch size: {patch_size}")

    # Test extract_patches_dense
    patches, h_out, w_out = model.extract_patches_dense(x, patch_size)
    print(f"\n1. extract_patches_dense:")
    print(f"   Output: patches={patches.shape}, h_out={h_out}, w_out={w_out}")

    # Test teacher network directly
    teacher = model.teachers['teacher_17']
    with torch.no_grad():
        teacher_patch_features, _ = teacher(patches)
    print(f"\n2. Teacher network on patches:")
    print(f"   Input: {patches.shape}")
    print(f"   Output: {teacher_patch_features.shape}")

    # Reshape teacher features
    teacher_reshaped = teacher_patch_features.view(1, h_out, w_out, 32)
    print(f"   Reshaped: {teacher_reshaped.shape}")

    # Test student network directly
    student = model.students['student_17_0']
    with torch.no_grad():
        student_patch_features = student(patches)
    print(f"\n3. Student network on patches:")
    print(f"   Input: {patches.shape}")
    print(f"   Output: {student_patch_features.shape}")

    # Reshape student features
    student_reshaped = student_patch_features.view(1, h_out, w_out, 32)
    print(f"   Reshaped: {student_reshaped.shape}")

    # Test if reshaping is the same
    if teacher_reshaped.shape == student_reshaped.shape:
        print(f"   ✅ Teacher and Student reshaped shapes match")
    else:
        print(f"   ❌ Teacher {teacher_reshaped.shape} != Student {student_reshaped.shape}")


if __name__ == "__main__":
    analyze_shape_flow()
    test_individual_components()