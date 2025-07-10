import numpy as np
import pytest
from pydistort import Cv2Distortion, cv2_undistort_points
import cv2
import time
import csv

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import setup



@pytest.mark.parametrize("Nparams", [5, 8, 12, 14])
@pytest.mark.parametrize("mode", ["weak_coefficients"])
def test_pydistort_undistort_vs_opencv(Nparams, mode):
    """Compare Cv2Distortion.cv2_undistort_points and OpenCV undistortPoints."""
    distortion = setup.CV2_DISTORTION(Nparams, mode)

    K = setup.ORI_MATK()
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Rectification matrix (identity)
    R = np.array([0.01, 0.02, 0.03])  # small rotation
    R, _ = cv2.Rodrigues(R)  # Convert to rotation matrix

    # Projection matrix (identity)
    P = np.array([[fx + 20, 0, cx - 11], [0, fy + 25, cy + 12], [0, 0, 1]])

    # 2D points
    points = setup.ORI_IMAGE_POINTS()

    # Undistort with your method
    undistorted_points = cv2_undistort_points(points, K=K, distortion=distortion, R=R, P=P)

    # Undistort with OpenCV
    points_cv = np.ascontiguousarray(points.reshape(-1, 1, 2), dtype=np.float64)
    undistorted_points_cv = cv2.undistortPoints(points_cv, K, distortion.parameters, R=R, P=P)
    undistorted_points_cv = np.asarray(undistorted_points_cv[:, 0, :], dtype=np.float64)

    # Comparaison
    np.testing.assert_allclose(undistorted_points, undistorted_points_cv, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("Nparams", [None])
@pytest.mark.parametrize("mode", ["weak_coefficients"])
def test_pydistort_undistort_vs_opencv_timer(Nparams, mode):
    """Compare pydistort.cv2_undistort_points and OpenCV undistortPoints for speed."""
    if setup.TIMER():
        pydistort_times = []
        opencv_times = []
        Nparams_list = [5, 8, 12, 14]
        Npoints = 1_000_000
        for Nparams in Nparams_list:
            distortion = setup.CV2_DISTORTION(Nparams, mode)

            K = setup.ORI_MATK()
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            # Rectification matrix (identity)
            R = np.array([0.01, 0.02, 0.03])  # small rotation
            R, _ = cv2.Rodrigues(R)  # Convert to rotation matrix

            # Projection matrix (identity)
            P = np.array([[fx + 20, 0, cx - 11], [0, fy + 25, cy + 12], [0, 0, 1]])

            # Test points
            points = setup.ORI_IMAGE_POINTS()

            # Distortion (analytic)
            start_time = time.perf_counter()
            undistorted_points = cv2_undistort_points(points, K=K, distortion=distortion, R=R, P=P)
            elapsed_time = time.perf_counter() - start_time
            pydistort_times.append(elapsed_time)

            # Distortion (opencv)
            start_time = time.perf_counter()
            points_cv = np.ascontiguousarray(points.reshape(-1, 1, 2), dtype=np.float64)
            undistorted_points_cv = cv2.undistortPoints(points_cv, K, distortion.parameters, R=R, P=P)
            undistorted_points_cv = np.asarray(undistorted_points_cv[:, 0, :], dtype=np.float64)
            elapsed_time = time.perf_counter() - start_time
            opencv_times.append(elapsed_time)

        # Print times in a table fomat:
        if setup.VERBOSE():
            print("\n\n ======== Undistort Points CV2 Complete Time Comparison ========")
            print(f"Npoints: {Npoints}")
            print(f"{'Nparams':<15} {'pydistort':<30} {'opencv':<30}")
            for i, Nparams in enumerate(Nparams_list):
                print(f"{Nparams:<15} {pydistort_times[i]:<30.4f} {opencv_times[i]:<30.4f}")

        if setup.CSV():
            # Write times to a CSV file
            csv_filename = "Cv2Distortion_cv2_undistort_points_time_comparison.csv"
            csv_filename = os.path.join(os.path.dirname(__file__), csv_filename)
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Nparams', 'pydistort_times', 'opencv_times'])
                for i, Nparams in enumerate(Nparams_list):
                    writer.writerow([Nparams, pydistort_times[i], opencv_times[i]])

