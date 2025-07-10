import numpy as np
import pytest
from pydistort import cv2_distort_image
import cv2

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import setup

@pytest.mark.parametrize("C", [True, False])
def test_pydistort_cv2_distort_image_compare_methods(C):
    """Compare Cv2Distortion.cv2_distort_image for the two methods."""
    distortion = setup.CV2_DISTORTION(8, "weak_coefficients")

    # Create a grid image with wave pattern
    image = setup.ORI_IMAGE()
    if not C:
        image = image[:, :, 0]  # Use only one channel if C is False to check grayscale handling

    # Camera intrinsics
    K = setup.ORI_MATK()

    # Undistort with pydistort undistort and remap methods
    result_meth1 = cv2_distort_image(image, K=K, distortion=distortion, method="undistort")

    # Undistort with pydistort distort and LinearNDInterpolator methods
    result_meth2 = cv2_distort_image(image, K=K, distortion=distortion, method="distort")
    result_meth2 = np.round(result_meth2).astype(image.dtype)

    assert image.shape == result_meth1.shape, "Input and output shapes do not match"
    assert image.shape == result_meth2.shape, "Input and output shapes do not match"

    if not np.allclose(result_meth1, result_meth2, atol=1e-5):
        if setup.DISPLAY():
            # Show images for debugging if they do not match
            combined = np.hstack((image, result_meth1))
            cv2.imshow("Pydistort (undistort) (left) vs Pydistort (distort) (right)", combined)
            print("Press any key on the image window to exit")
            cv2.waitKey(0)

    if not np.allclose(result_meth1, result_meth2, atol=1e-5):
        if setup.VERBOSE():
            counter = 0
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if not np.allclose(result_meth1[i, j], result_meth2[i, j], atol=1e-5):
                        if setup.VERBOSE_LEVEL() >= 1:
                            print(f"Mismatch at pixel ({i}, {j}): {result_meth1[i, j]} vs {result_meth2[i, j]}")
                        counter += 1
            print(f"Total mismatches: {counter} / {image.shape[0] * image.shape[1]}")
    
    # Compare results

    assert np.allclose(result_meth1, result_meth2, atol=1e-5), "Undistorted images do not match"




def test_pydistort_cv2_distort_image_interpolation():
    """Test cv2_distort_image with different interpolation methods."""
    distortion = setup.CV2_DISTORTION(8, "weak_coefficients")

    # Create a grid image with wave pattern
    image = setup.ORI_IMAGE()

    # Camera intrinsics
    K = setup.ORI_MATK()

    # Test different interpolation methods for METHOD 1 (undistort)
    for interpolation in ["linear", "nearest", "cubic", "area", "lanczos4"]:
        result = cv2_distort_image(image, K=K, distortion=distortion, interpolation=interpolation, method="undistort")

        # Check if the result is not None and has the same shape as the input image
        assert result is not None, f"Result should not be None for interpolation method {interpolation}"
        assert result.shape == image.shape, f"Output shape does not match input shape for interpolation method {interpolation}"

    # Test different interpolation methods for METHOD 2 (distort)
    for interpolation in ["linear", "nearest", "clough"]:
        result = cv2_distort_image(image, K=K, distortion=distortion, interpolation=interpolation, method="distort")

        # Check if the result is not None and has the same shape as the input image
        assert result is not None, f"Result should not be None for interpolation method {interpolation}"
        assert result.shape == image.shape, f"Output shape does not match input shape for interpolation method {interpolation}"