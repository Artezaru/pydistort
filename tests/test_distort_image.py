import numpy as np
import pytest
from pydistort import distort_image
import cv2

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import setup

@pytest.mark.parametrize("C", [True, False])
def test_pydistort_distort_image_compare_methods(C):
    """Compare Cv2Distortion.distort_image for the two methods."""
    distortion = setup.CV2_DISTORTION(8, "weak_coefficients")

    # Create a grid image with wave pattern
    image = setup.ORI_IMAGE()
    if not C:
        image = image[:, :, 0]  # Use only one channel if C is False to check grayscale handling

    # Camera intrinsics
    K = setup.ORI_MATK()

    # Undistort with pydistort undistort and remap methods
    result_meth1 = distort_image(image, K=K, distortion=distortion, method="undistort")

    # Undistort with pydistort distort and LinearNDInterpolator methods
    result_meth2 = distort_image(image, K=K, distortion=distortion, method="distort")
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

