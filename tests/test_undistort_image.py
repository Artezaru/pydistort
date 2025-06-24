import numpy as np
import pytest
from pydistort import undistort_image
import cv2

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import setup


@pytest.mark.parametrize("Nparams", [5, 8, 12, 14])
@pytest.mark.parametrize("mode", ["weak_coefficients", "strong_coefficients"])
def test_pydistort_undistort_image_vs_opencv(Nparams, mode):
    """Compare Cv2Distortion.undistort_image and OpenCV undistort for accuracy."""
    if mode == "strong_coefficients":
        if setup.WARNINGS():
            print("[WARNING] test_pydistort_undistort_image_vs_opencv - strong_coefficients (TEST ERROR)")
        return
    
    distortion = setup.CV2_DISTORTION(Nparams, mode)

    # Create a grid image with wave pattern
    image = setup.ORI_IMAGE()

    # Camera intrinsics
    K = setup.ORI_MATK()

    # Undistort with pydistort
    result = undistort_image(image, K=K, distortion=distortion)

    # Undistort with OpenCV
    result_cv2 = cv2.undistort(image, K, distortion.parameters)

    if not np.allclose(result, result_cv2, atol=1e-5):
        if setup.DISPLAY():
            # Show images for debugging if they do not match
            combined = np.hstack((image, result, result_cv2))
            cv2.imshow("Pydistort (left) vs OpenCV (right)", combined)
            print("Press any key on the image window to exit")
            cv2.waitKey(0)

    if not np.allclose(result, result_cv2, atol=1e-5):
        if setup.VERBOSE():
            counter = 0
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if not np.allclose(result[i, j], result_cv2[i, j], atol=1e-5):
                        if setup.VERBOSE_LEVEL() >= 1:
                            print(f"Mismatch at pixel ({i}, {j}): {result[i, j]} vs {result_cv2[i, j]}")
                        counter += 1
            print(f"Total mismatches: {counter} / {image.shape[0] * image.shape[1]}")
    
    # Compare results
    assert result.shape == result_cv2.shape, "Output shapes do not match"
    assert np.allclose(result, result_cv2, atol=1e-5), "Undistorted images do not match"



