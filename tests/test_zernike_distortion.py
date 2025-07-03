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


def test_pydistort_distort_undistort():
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
    distortion.parameters_x /= K[0, 0]  # Normalize parameters by the focal length
    distortion.parameters_y /= K[1, 1]  # Normalize parameters by the focal length

    # Distortion (analytic)
    result = distortion.distort(points, dx=False, dp=False)

    # Undistortion (analytic)
    undistorted_result = distortion.undistort(result.distorted_points, dx=False, dp=False, max_iter=10, eps=1e-8)
    count_nan = np.sum(np.logical_or(np.isnan(undistorted_result.normalized_points[:, 0]), np.isnan(undistorted_result.normalized_points[:, 1])))
    if count_nan > 0:
        print(f"Warning: {count_nan} / {points.shape[0]} points have NaN values in the undistorted points - so {count_nan / points.shape[0] * 100:.2f}%")
    # Check that the undistorted points are close to the original points
    np.testing.assert_allclose(points[~np.isnan(undistorted_result.normalized_points)], undistorted_result.normalized_points[~np.isnan(undistorted_result.normalized_points)], rtol=1e-5, atol=1e-8)


def test_timing_distortion():
    """Timing the distortion for various Nzer."""
    if setup.TIMER():
        without_jacobian_times = []
        with_jacobian_times_dp = []
        with_jacobian_times = []
        Nzer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        Npoints = 1_000_000
        for Nzer in Nzer_list:
            distortion = ZernikeDistortion(Nzer=Nzer)
            distortion.parameters = np.random.rand(distortion.Nparams) * 0.01  # Random coefficients for testing
            distortion.radius = np.sqrt(2)  # Set radius to sqrt(2) for testing

            # Test points
            points = np.random.uniform(-1.0, 1.0, size=(Npoints, 2))  # shape (Npoints, 2)

            # Distortion (analytic)
            start_time = time.perf_counter()
            _ = distortion.distort(points, dx=False, dp=False)
            elapsed_time = time.perf_counter() - start_time
            without_jacobian_times.append(elapsed_time)

            # Distortion (analytic with jacobians dp only)
            start_time = time.perf_counter()
            _ = distortion.distort(points, dx=False, dp=True)
            elapsed_time = time.perf_counter() - start_time
            with_jacobian_times_dp.append(elapsed_time)

            # Distortion (analytic with jacobians)
            start_time = time.perf_counter()
            _ = distortion.distort(points, dx=True, dp=True)
            elapsed_time = time.perf_counter() - start_time
            with_jacobian_times.append(elapsed_time)

        # Print times in a table fomat:
        if setup.VERBOSE():
            print("\n\n ======== Distortion Zernike Timing Comparison ========\n")
            print(f"Npoints: {Npoints}")
            print(f"{'Nzer':<15} {'pydistort_nojac_times':<30} {'pydistort_dp_times':<30} {'pydistort_alljac_times':<30}")
            for i, Nzer in enumerate(Nzer_list):
                print(f"{Nzer:<15} {without_jacobian_times[i]:<30.4f} {with_jacobian_times_dp[i]:<30.4f} {with_jacobian_times[i]:<30.4f}")

        if setup.CSV():
            # Write times to a CSV file
            csv_filename = "ZernikeDistortion_distortion_time_comparison.csv"
            csv_filename = os.path.join(os.path.dirname(__file__), csv_filename)
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Nzer', 'pydistort_nojac_times', 'pydistort_dp_times', 'pydistort_alljac_times'])
                for i, Nzer in enumerate(Nzer_list):
                    writer.writerow([Nzer, without_jacobian_times[i], with_jacobian_times_dp[i], with_jacobian_times[i]])



def test_pydistort_undistort_timer():
    """Timing pydistort.undistor"""
    if setup.TIMER():
        times = []
        Nzer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        Npoints = 1_000_000
        for Nzer in Nzer_list:
            distortion = ZernikeDistortion(Nzer=Nzer)
            distortion.parameters = np.random.rand(distortion.Nparams) * 0.01  # Random coefficients for testing
            distortion.radius = np.sqrt(2)  # Set radius to sqrt(2) for testing

            # Test points
            points = np.random.uniform(-1.0, 1.0, size=(Npoints, 2))  # shape (Npoints, 2)

            # Distortion (analytic)
            start_time = time.perf_counter()
            result = distortion.undistort(points, dx=False, dp=False, max_iter=1)
            elapsed_time = time.perf_counter() - start_time
            times.append(elapsed_time)

        # Print times in a table fomat:
        if setup.VERBOSE():
            print("\n\n ======== Undistortion Zernike Timing Comparison ========\n")
            print(f"Npoints: {Npoints}")
            print(f"{'Nzer':<15} {'pydistort_times':<30}")
            for i, Nzer in enumerate(Nzer_list):
                print(f"{Nzer:<15} {times[i]:<30.4f}")

        if setup.CSV():
            # Write times to a CSV file
            csv_filename = "ZernikeDistortion_undistort_time_comparison.csv"
            csv_filename = os.path.join(os.path.dirname(__file__), csv_filename)
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Nzer', 'pydistort_times'])
                for i, Nzer in enumerate(Nzer_list):
                    writer.writerow([Nzer, times[i]])

