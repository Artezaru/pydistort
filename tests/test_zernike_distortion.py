import pytest
import numpy as np
from pydistort import ZernikeDistortion
import time
import cv2
import csv
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import setup


def test_pydistort_distort():
    """Compare Cv2Distortion.distort and distort_opencv for various Nparams."""
    distortion = setup.ZERNIKE_DISTORTION()

    # Test points
    points = setup.ORI_NORMALIZED_POINTS()

    # Set the radius to the diagonal of the image
    K = setup.ORI_MATK()
    image = setup.ORI_IMAGE()
    height, width = image.shape[:2]
    radius = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)

    distortion.radius_x = radius / K[0, 0]  # Normalize by the focal length
    distortion.radius_y = radius / K[1, 1]  # Normalize by the focal length

    # Distortion (analytic)
    result = distortion.distort(points, dx=True, dp=True) 

    assert result.distorted_points.shape == points.shape
    assert result.jacobian_dp.shape == (points.shape[0], 2, distortion.Nparams)
    assert result.jacobian_dx.shape == (points.shape[0], 2, 2)
    assert np.sum(np.isnan(result.distorted_points)) == 0


def test_pydistort_distort_unditort():
    """Check the consistency between distort and undistort"""
    distortion = setup.ZERNIKE_DISTORTION()

    # Test points
    points = setup.ORI_NORMALIZED_POINTS()

    # Set the radius to the diagonal of the image
    K = setup.ORI_MATK()
    image = setup.ORI_IMAGE()
    height, width = image.shape[:2]
    radius = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
   
    distortion.radius_x = radius / K[0, 0]  # Normalize by the focal length
    distortion.radius_y = radius / K[1, 1]  # Normalize by the focal length

    # Distortion (analytic)
    result = distortion.distort(points, dx=False, dp=False)

    # Undistortion (analytic)
    undistorted_result = distortion.undistort(result.distorted_points, dx=False, dp=False, max_iter=10, eps=1e-8)

    # Check that the undistorted points are close to the original points
    np.testing.assert_allclose(points[~np.isnan(undistorted_result.normalized_points)], undistorted_result.normalized_points[~np.isnan(undistorted_result.normalized_points)], rtol=1e-5, atol=1e-8)
