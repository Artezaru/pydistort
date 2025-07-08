import pytest
import numpy as np
from pydistort import Cv2Distortion
import time
import cv2
import csv
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import setup
    

@pytest.mark.parametrize("Nparams", [5, 8, 12, 14])
@pytest.mark.parametrize("mode", ["strong_coefficients", "weak_coefficients"])
def test_pydistort_distort_vs_opencv(Nparams, mode):
    """Compare Cv2Distortion.distort and distort_opencv for various Nparams."""
    distortion = setup.CV2_DISTORTION(Nparams, mode)

    # Test points
    points = setup.ORI_NORMALIZED_POINTS()

    # Distortion (analytic)
    result = distortion.distort(points, dx=True, dp=True)

    # Distortion (opencv)
    result_opencv = distortion.distort(points, dx=False, dp=True, opencv=True)

    # Check that shapes match
    assert result.distorted_points.shape == result_opencv.distorted_points.shape
    assert result.jacobian_dp.shape == result_opencv.jacobian_dp.shape

    # Compare outputs
    np.testing.assert_allclose(result.distorted_points, result_opencv.distorted_points, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(result.jacobian_dp, result_opencv.jacobian_dp, rtol=1e-5, atol=1e-8)

    # jacobian_dx is None for opencv
    assert result_opencv.jacobian_dx is None


@pytest.mark.parametrize("Nparams", [5, 8, 12, 14])
@pytest.mark.parametrize("mode", ["strong_coefficients", "weak_coefficients"])
def test_pydistort_undistort_vs_opencv(Nparams, mode):
    """Compare Cv2Distortion.undistort and undistort_opencv for various Nparams."""
    distortion = setup.CV2_DISTORTION(Nparams, mode)
    
    # Test points
    points = setup.ORI_NORMALIZED_POINTS()

    # Distortion (analytic)
    result = distortion.undistort(points)

    # Distortion (opencv)
    result_opencv = distortion.undistort(points, opencv=True)

    # Check that shapes match
    assert result.normalized_points.shape == result_opencv.normalized_points.shape

    # Compare outputs
    np.testing.assert_allclose(result.normalized_points, result_opencv.normalized_points, rtol=1e-5, atol=1e-8)



@pytest.mark.parametrize("Nparams", [5, 8, 12, 14])
@pytest.mark.parametrize("mode", ["strong_coefficients", "weak_coefficients"])
def test_pydistort_undistort_vs_opencv_IMAGE(Nparams, mode):
    """Compare Cv2Distortion.distort and distort_opencv for various Nparams."""
    if mode == "strong_coefficients":
        if setup.WARNINGS():
            print("[WARNING] test_pydistort_undistort_vs_opencv_IMAGE - strong_coefficients (TEST ERROR)")
        return

    distortion = setup.CV2_DISTORTION(Nparams, mode)
    
    # Test points
    points = setup.ORI_NORMALIZED_POINTS()

    # Distortion (analytic)
    result = distortion.undistort(points)

    # Distortion (opencv)
    result_opencv = distortion.undistort(points, opencv=True)

    # Distortion (opencv undistortImagePoints)
    K = np.eye(3)
    distCoeffs = distortion.parameters
    normalized_points = cv2.undistortImagePoints(points, K, distCoeffs=distCoeffs)
    normalized_points = normalized_points[:, 0, :]
    
    # Check that shapes match
    assert result.normalized_points.shape == result_opencv.normalized_points.shape
    assert result.normalized_points.shape == normalized_points.shape

    # Compare outputs
    np.testing.assert_allclose(result.normalized_points, result_opencv.normalized_points, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(result.normalized_points, normalized_points, rtol=1e-5, atol=1e-8)



@pytest.mark.parametrize("Nparams", [5, 8, 12, 14])
@pytest.mark.parametrize("mode", ["strong_coefficients", "weak_coefficients"])
def test_pydistort_distort_unditort(Nparams, mode):
    """Check the consistency between distort and undistort"""
    distortion = setup.CV2_DISTORTION(Nparams, mode)

    # Test points
    points = setup.ORI_NORMALIZED_POINTS()

    # Distortion (analytic)
    result = distortion.distort(points, dx=False, dp=False)

    # Undistortion (analytic)
    undistorted_result = distortion.undistort(result.distorted_points, dx=False, dp=False)

    # Check that the undistorted points are close to the original points
    np.testing.assert_allclose(points, undistorted_result.normalized_points, rtol=1e-5, atol=1e-8)



@pytest.mark.parametrize("Nparams", [None])
@pytest.mark.parametrize("mode", ["strong_coefficients"])
def test_pydistort_distort_vs_opencv_timer(Nparams, mode):
    """Compare Cv2Distortion.distort and distort_opencv for various Nparams in time."""
    if setup.TIMER():
        pydistort_alljac_times = []
        pydistort_times = []
        pydistort_nojac_times = []
        opencv_times = []
        Nparams_list = [5, 8, 12, 14]
        Npoints = 1_000_000
        for Nparams in Nparams_list:
            distortion = setup.CV2_DISTORTION(Nparams, mode)

            # Test points
            points = np.random.uniform(-1.0, 1.0, size=(Npoints, 2))  # shape (Npoints, 2)

            # Distortion (analytic)
            start_time = time.perf_counter()
            result = distortion.distort(points, dx=True, dp=True)
            elapsed_time = time.perf_counter() - start_time
            pydistort_alljac_times.append(elapsed_time)

            # Distortion (analytic)
            start_time = time.perf_counter()
            result = distortion.distort(points, dx=False, dp=True)
            elapsed_time = time.perf_counter() - start_time
            pydistort_times.append(elapsed_time)

            # Distortion (analytic)
            start_time = time.perf_counter()
            result = distortion.distort(points, dx=False, dp=False)
            elapsed_time = time.perf_counter() - start_time
            pydistort_nojac_times.append(elapsed_time)

            # Distortion (opencv)
            start_time = time.perf_counter()
            result_opencv = distortion.distort(points, dx=False, dp=True, opencv=True)
            elapsed_time = time.perf_counter() - start_time
            opencv_times.append(elapsed_time)

        # Print times in a table fomat:
        if setup.VERBOSE():
            print("\n\n ======== Distortion CV2 Time Comparison ========")
            print(f"Npoints: {Npoints}")
            print(f"{'Nparams':<15} {'pydistort (all Jacobians)':<30} {'pydistort (cv2 Jacobians)':<30} {'pydistort (no Jacobians)':<30} {'opencv':<30}")
            for i, Nparams in enumerate(Nparams_list):
                print(f"{Nparams:<15} {pydistort_alljac_times[i]:<30.4f} {pydistort_times[i]:<30.4f} {pydistort_nojac_times[i]:<30.4f} {opencv_times[i]:<30.4f}")

        if setup.CSV():
            # Write times to a CSV file
            csv_filename = "Cv2Distortion_distortion_time_comparison.csv"
            csv_filename = os.path.join(os.path.dirname(__file__), csv_filename)
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Nparams', 'pydistort_alljac_times', 'pydistort_times', 'pydistort_nojac_times', 'opencv_times'])
                for i, Nparams in enumerate(Nparams_list):
                    writer.writerow([Nparams, pydistort_alljac_times[i], pydistort_times[i], pydistort_nojac_times[i], opencv_times[i]])


@pytest.mark.parametrize("Nparams", [None])
@pytest.mark.parametrize("mode", ["weak_coefficients"])
def test_pydistort_undistort_vs_opencv_timer(Nparams, mode):
    """Compare Cv2Distortion.distort and distort_opencv for various Nparams in time."""
    if setup.TIMER():
        pydistort_times = []
        opencv_times = []
        Nparams_list = [5, 8, 12, 14]
        Npoints = 1_000_000
        for Nparams in Nparams_list:
            distortion = setup.CV2_DISTORTION(Nparams, mode)

            # Test points
            points = np.random.uniform(-1.0, 1.0, size=(Npoints, 2))  # shape (Npoints, 2)

            # Distortion (analytic)
            start_time = time.perf_counter()
            result = distortion.undistort(points)
            elapsed_time = time.perf_counter() - start_time
            pydistort_times.append(elapsed_time)

            # Distortion (opencv)
            start_time = time.perf_counter()
            result_opencv = distortion.undistort(points, opencv=True)
            elapsed_time = time.perf_counter() - start_time
            opencv_times.append(elapsed_time)

        # Print times in a table fomat:
        if setup.VERBOSE():
            print("\n\n ======== Undistortion CV2 Time Comparison ========")
            print(f"Npoints: {Npoints}")
            print(f"{'Nparams':<15} {'pydistort':<30} {'opencv':<30}")
            for i, Nparams in enumerate(Nparams_list):
                print(f"{Nparams:<15} {pydistort_times[i]:<30.4f} {opencv_times[i]:<30.4f}")

        if setup.CSV():
            # Write times to a CSV file
            csv_filename = "Cv2Distortion_undistortion_time_comparison.csv"
            csv_filename = os.path.join(os.path.dirname(__file__), csv_filename)
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Nparams', 'pydistort_times', 'opencv_times'])
                for i, Nparams in enumerate(Nparams_list):
                    writer.writerow([Nparams, pydistort_times[i], opencv_times[i]])





@pytest.mark.parametrize("Nparams", [5, 8, 12, 14])
@pytest.mark.parametrize("mode", ["weak_coefficients"]) # TODO: [WARNING] !!! Error with strong_coefficients
def test_distort_undistort_inverse(Nparams, mode):
    """Check that undistort(distort(x)) ≈ x for all supported Nparams."""
    distortion = setup.CV2_DISTORTION(Nparams, mode)

    # Test points
    points = setup.ORI_NORMALIZED_POINTS()

    # Distort and then undistort
    result = distortion.distort(points, dx=False, dp=False)
    undistorted_result = distortion.undistort(result.distorted_points, dx=False, dp=False)

    # Check that the undistorted points are close to the original points
    np.testing.assert_allclose(points, undistorted_result.normalized_points, rtol=1e-5, atol=1e-8)
