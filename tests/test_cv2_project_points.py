import numpy as np
import pytest
from pydistort import Cv2Distortion, cv2_project_points, NoDistortion, ZernikeDistortion
import cv2
import time
import csv

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import setup



def print_jacobian_differences(jac1, jac2, rtol=1e-5, atol=1e-8):
    diff = np.abs(jac1 - jac2)
    rel_diff = np.abs((jac1 - jac2) / (np.where(jac2 != 0, jac2, 1)))

    mask = (diff > atol) & (rel_diff > rtol)
    mismatches = np.argwhere(mask)

    print(f"Total mismatches: {len(mismatches)} / {jac1.size}")
    for idx in mismatches:
        i, j, k = idx
        v1 = jac1[i, j, k]
        v2 = jac2[i, j, k]
        abs_diff = diff[i, j, k]
        rel = rel_diff[i, j, k]
        print(f"[{i}, {j}, {k}] → our: {v1:.6g}, cv2: {v2:.6g}, "
              f"abs diff: {abs_diff:.2e}, rel diff: {rel:.2e}")


@pytest.mark.parametrize("Nparams", [5, 8, 12, 14])
@pytest.mark.parametrize("mode", ["strong_coefficients", "weak_coefficients"])
def test_pydistort_project_vs_opencv(Nparams, mode):
    """Compare Cv2Distortion.cv2_project_points and OpenCV projectPoints."""
    distortion = setup.CV2_DISTORTION(Nparams, mode)

    # Camera intrinsics
    fx, fy = 1000.0, 950.0
    cx, cy = 320.0, 240.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Rotation and translation
    rvec = np.array([0.01, 0.02, 0.03])  # small rotation
    tvec = np.array([0.1, -0.1, 0.2])    # small translation

    # 3D points
    points_3d = np.array([
        [0.0, 0.0, 5.0],
        [0.1, -0.1, 5.0],
        [-0.1, 0.2, 5.0],
        [0.2, 0.1, 5.0],
        [-0.2, -0.2, 5.0]
    ])

    # Project with your method
    result = cv2_project_points(points_3d, rvec=rvec, tvec=tvec, K=K, distortion=distortion, dx=True, faster_dx=False, dp=True)

    # Project with OpenCV
    object_points = np.ascontiguousarray(points_3d.reshape(-1, 1, 3), dtype=np.float64)
    image_points_cv, jacobian_cv = cv2.projectPoints(object_points, rvec, tvec, K, distortion.parameters)

    image_points_cv = np.asarray(image_points_cv[:,0,:], dtype=np.float64)
    jacobian_cv = np.asarray(jacobian_cv, dtype=np.float64) # shape (2 * Npoints, 10 + Nparams)
    
    jacobian_dp_cv = np.zeros((points_3d.shape[0], 2, 10 + distortion.Nparams), dtype=np.float64)
    jacobian_dp_cv[:, 0, :] = jacobian_cv[0::2, :10 + distortion.Nparams] # shape (Npoints, 10 + Nparams)
    jacobian_dp_cv[:, 1, :] = jacobian_cv[1::2, :10 + distortion.Nparams] # shape (Npoints, 10 + Nparams)

    jacobian_dx_cv = np.zeros((points_3d.shape[0], 2, 3), dtype=np.float64)
    jacobian_dx_cv[:, 0, :] = jacobian_cv[0::2, 3:6] @ cv2.Rodrigues(rvec)[0]  # shape (Npoints, 3)
    jacobian_dx_cv[:, 1, :] = jacobian_cv[1::2, 3:6] @ cv2.Rodrigues(rvec)[0]  # shape (Npoints, 3)


    # Comparaison
    np.testing.assert_allclose(result.image_points, image_points_cv, rtol=1e-5, atol=1e-8)
    try:
        np.testing.assert_allclose(result.jacobian_dp, jacobian_dp_cv, rtol=1e-5, atol=1e-8)
    except AssertionError as e:
        print_jacobian_differences(result.jacobian_dp, jacobian_dp_cv, rtol=1e-5, atol=1e-8)
        raise e
    try:    
        np.testing.assert_allclose(result.jacobian_dx, jacobian_dx_cv, rtol=1e-5, atol=1e-8)
    except AssertionError as e:
        print_jacobian_differences(result.jacobian_dx, jacobian_dx_cv, rtol=1e-5, atol=1e-8)
        raise e

@pytest.mark.parametrize("Nparams", [5, 8, 12, 14])
@pytest.mark.parametrize("mode", ["strong_coefficients", "weak_coefficients"])
def test_pydistort_project_faster_vs_opencv(Nparams, mode):
    """Compare Cv2Distortion.cv2_project_points and OpenCV projectPoints."""
    distortion = setup.CV2_DISTORTION(Nparams, mode)

    # Camera intrinsics
    fx, fy = 1000.0, 950.0
    cx, cy = 320.0, 240.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Rotation and translation
    rvec = np.array([0.01, 0.02, 0.03])  # small rotation
    tvec = np.array([0.1, -0.1, 0.2])    # small translation

    # 3D points
    points_3d = np.array([
        [0.0, 0.0, 5.0],
        [0.1, -0.1, 5.0],
        [-0.1, 0.2, 5.0],
        [0.2, 0.1, 5.0],
        [-0.2, -0.2, 5.0]
    ])

    # Project with your method
    result = cv2_project_points(points_3d, rvec=rvec, tvec=tvec, K=K, distortion=distortion, dx=True, faster_dx=True, dp=True)
   
    # Project with OpenCV
    object_points = np.ascontiguousarray(points_3d.reshape(-1, 1, 3), dtype=np.float64)
    image_points_cv, jacobian_cv = cv2.projectPoints(object_points, rvec, tvec, K, distortion.parameters)

    image_points_cv = np.asarray(image_points_cv[:,0,:], dtype=np.float64)
    jacobian_cv = np.asarray(jacobian_cv, dtype=np.float64) # shape (2 * Npoints, 10 + Nparams)
    
    jacobian_dp_cv = np.zeros((points_3d.shape[0], 2, 10 + distortion.Nparams), dtype=np.float64)
    jacobian_dp_cv[:, 0, :] = jacobian_cv[0::2, :10 + distortion.Nparams] # shape (Npoints, 10 + Nparams)
    jacobian_dp_cv[:, 1, :] = jacobian_cv[1::2, :10 + distortion.Nparams] # shape (Npoints, 10 + Nparams)

    jacobian_dx_cv = np.zeros((points_3d.shape[0], 2, 3), dtype=np.float64)
    jacobian_dx_cv[:, 0, :] = jacobian_cv[0::2, 3:6] @ cv2.Rodrigues(rvec)[0]  # shape (Npoints, 3)
    jacobian_dx_cv[:, 1, :] = jacobian_cv[1::2, 3:6] @ cv2.Rodrigues(rvec)[0]  # shape (Npoints, 3)

    # Comparaison
    np.testing.assert_allclose(result.image_points, image_points_cv, rtol=1e-5, atol=1e-8)
    try:
        np.testing.assert_allclose(result.jacobian_dp, jacobian_dp_cv, rtol=1e-5, atol=1e-8)
    except AssertionError as e:
        if setup.VERBOSE() and setup.VERBOSE_LEVEL() >= 1:
            print_jacobian_differences(result.jacobian_dp, jacobian_dp_cv, rtol=1e-5, atol=1e-8)
        raise e
    try:    
        np.testing.assert_allclose(result.jacobian_dx, jacobian_dx_cv, rtol=1e-5, atol=1e-8)
    except AssertionError as e:
        if setup.VERBOSE() and setup.VERBOSE_LEVEL() >= 1:
            print_jacobian_differences(result.jacobian_dx, jacobian_dx_cv, rtol=1e-5, atol=1e-8)
        raise e




@pytest.mark.parametrize("Nparams", [None])
@pytest.mark.parametrize("mode", ["strong_coefficients"])
def test_pydistort_project_vs_opencv_timer(Nparams, mode):
    """Compare cv2_project_points and opencv.projectPoints for various Nparams in time."""
    if setup.TIMER():
        pydistort_alljac_times = []
        pydistort_alljac_faster_times = []
        pydistort_times = []
        pydistort_nojac_times = []
        opencv_times = []
        Nparams_list = [5, 8, 12, 14]
        Npoints = 1_000_000
        for Nparams in Nparams_list:
            distortion = setup.CV2_DISTORTION(Nparams, mode)

            # Camera intrinsics
            fx, fy = 1000.0, 950.0
            cx, cy = 320.0, 240.0
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            # Rotation and translation
            rvec = np.array([0.01, 0.02, 0.03])  # small rotation
            tvec = np.array([0.1, -0.1, 0.2])    # small translation

            # Test points
            points = np.random.uniform(-1.0, 1.0, size=(Npoints, 2))  # shape (Npoints, 2)
            points = np.concatenate((points, 5.0 * np.ones((Npoints, 1))), axis=1) # shape (Npoints, 3)

            # Projection (analytic)
            start_time = time.perf_counter()
            result = cv2_project_points(points, rvec=rvec, tvec=tvec, K=K, distortion=distortion, dx=True, dp=True, faster_dx=False)
            elapsed_time = time.perf_counter() - start_time
            pydistort_alljac_times.append(elapsed_time)

            # Projection (analytic, faster dx)
            start_time = time.perf_counter()
            result = cv2_project_points(points, rvec=rvec, tvec=tvec, K=K, distortion=distortion, dx=True, dp=True, faster_dx=True)
            elapsed_time = time.perf_counter() - start_time
            pydistort_alljac_faster_times.append(elapsed_time)

            # Projection (analytic)
            start_time = time.perf_counter()
            result = cv2_project_points(points, rvec=rvec, tvec=tvec, K=K, distortion=distortion, dx=False, dp=True)
            elapsed_time = time.perf_counter() - start_time
            pydistort_times.append(elapsed_time)

            # Projection (analytic)
            start_time = time.perf_counter()
            result = cv2_project_points(points, rvec=rvec, tvec=tvec, K=K, distortion=distortion, dx=False, dp=False)
            elapsed_time = time.perf_counter() - start_time
            pydistort_nojac_times.append(elapsed_time)

            # Projection (opencv)
            start_time = time.perf_counter()
            object_points = np.ascontiguousarray(points.reshape(-1, 1, 3), dtype=np.float64)
            image_points_cv, jacobian_cv = cv2.projectPoints(object_points, rvec, tvec, K, distortion.parameters)

            image_points_cv = np.asarray(image_points_cv[:,0,:], dtype=np.float64)
            jacobian_cv = np.asarray(jacobian_cv, dtype=np.float64) # shape (2 * Npoints, 10 + Nparams)
            jacobian_dp_cv = np.zeros((points.shape[0], 2, 10 + distortion.Nparams), dtype=np.float64)
            jacobian_dp_cv[:, 0, :] = jacobian_cv[0::2, :10 + distortion.Nparams] # shape (Npoints, 10 + Nparams)
            jacobian_dp_cv[:, 1, :] = jacobian_cv[1::2, :10 + distortion.Nparams] # shape (Npoints, 10 + Nparams)
            elapsed_time = time.perf_counter() - start_time
            opencv_times.append(elapsed_time)

        # Print times in a table fomat:
        if setup.VERBOSE():
            print("\n\n ======== Cv2 Distortion Project Points Time Comparison ========")
            print(f"Npoints: {Npoints}")
            print(f"{'Nparams':<15} {'pydistort (all Jacobians)':<30} {'pydistort (faster Jacobians)':<30} {'pydistort (cv2 Jacobians)':<30} {'pydistort (no Jacobians)':<30} {'opencv':<30}")
            for i, Nparams in enumerate(Nparams_list):
                print(f"{Nparams:<15} {pydistort_alljac_times[i]:<30.4f} {pydistort_alljac_faster_times[i]:<30.4f} {pydistort_times[i]:<30.4f} {pydistort_nojac_times[i]:<30.4f} {opencv_times[i]:<30.4f}")

        if setup.CSV():
            # Write times to a CSV file
            csv_filename = "Cv2Distortion_cv2_project_points_time_comparison.csv"
            csv_filename = os.path.join(os.path.dirname(__file__), csv_filename)
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Nparams', 'pydistort_alljac_times', 'pydistort_alljac_faster_times', 'pydistort_times', 'pydistort_nojac_times', 'opencv_times'])
                for i, Nparams in enumerate(Nparams_list):
                    writer.writerow([Nparams, pydistort_alljac_times[i], pydistort_alljac_faster_times[i], pydistort_times[i], pydistort_nojac_times[i], opencv_times[i]])




def test_pydistort_project_zernike():
    """Compare cv2_project_points and opencv.projectPoints for various Nparams in time."""
    if setup.TIMER():
        pydistort_alljac_times = []
        pydistort_alljac_faster_times = []
        pydistort_times = []
        pydistort_nojac_times = []
        Nzer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        Npoints = 1_000_000
        for Nzer in Nzer_list:
            distortion = ZernikeDistortion(Nzer=Nzer)
            distortion.parameters = np.random.rand(distortion.Nparams) * 0.01  # Random coefficients for testing
            distortion.radius = np.sqrt(2)  # Set radius to sqrt(2) for testing

            # Test points
            points = np.random.uniform(-1.0, 1.0, size=(Npoints, 2))  # shape (Npoints, 2)

            # Camera intrinsics
            fx, fy = 1000.0, 950.0
            cx, cy = 320.0, 240.0
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            # Rotation and translation
            rvec = np.array([0.01, 0.02, 0.03])  # small rotation
            tvec = np.array([0.1, -0.1, 0.2])    # small translation

            # Test points
            points = np.random.uniform(-1.0, 1.0, size=(Npoints, 2))  # shape (Npoints, 2)
            points = np.concatenate((points, 5.0 * np.ones((Npoints, 1))), axis=1) # shape (Npoints, 3)

            # Projection (analytic)
            start_time = time.perf_counter()
            result = cv2_project_points(points, rvec=rvec, tvec=tvec, K=K, distortion=distortion, dx=True, dp=True, faster_dx=False)
            elapsed_time = time.perf_counter() - start_time
            pydistort_alljac_times.append(elapsed_time)

            # Projection (analytic, faster dx)
            start_time = time.perf_counter()
            result = cv2_project_points(points, rvec=rvec, tvec=tvec, K=K, distortion=distortion, dx=True, dp=True, faster_dx=True)
            elapsed_time = time.perf_counter() - start_time
            pydistort_alljac_faster_times.append(elapsed_time)

            # Projection (analytic)
            start_time = time.perf_counter()
            result = cv2_project_points(points, rvec=rvec, tvec=tvec, K=K, distortion=distortion, dx=False, dp=True)
            elapsed_time = time.perf_counter() - start_time
            pydistort_times.append(elapsed_time)

            # Projection (analytic)
            start_time = time.perf_counter()
            result = cv2_project_points(points, rvec=rvec, tvec=tvec, K=K, distortion=distortion, dx=False, dp=False)
            elapsed_time = time.perf_counter() - start_time
            pydistort_nojac_times.append(elapsed_time)

        # Print times in a table fomat:
        if setup.VERBOSE():
            print("\n\n ======== Zernike Distortion Project Points Time Comparison ========")
            print(f"Npoints: {Npoints}")
            print(f"{'Nzer':<15} {'pydistort (all Jacobians)':<30} {'pydistort (faster Jacobians)':<30} {'pydistort (cv2 Jacobians)':<30} {'pydistort (no Jacobians)':<30}")
            for i, Nzer in enumerate(Nzer_list):
                print(f"{Nzer:<15} {pydistort_alljac_times[i]:<30.4f} {pydistort_alljac_faster_times[i]:<30.4f} {pydistort_times[i]:<30.4f} {pydistort_nojac_times[i]:<30.4f}")

        if setup.CSV():
            # Write times to a CSV file
            csv_filename = "ZernikeDistortion_cv2_project_points_time_comparison.csv"
            csv_filename = os.path.join(os.path.dirname(__file__), csv_filename)
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Nzer', 'pydistort_alljac_times', 'pydistort_alljac_faster_times', 'pydistort_times', 'pydistort_nojac_times'])
                for i, Nzer in enumerate(Nzer_list):
                    writer.writerow([Nzer, pydistort_alljac_times[i], pydistort_alljac_faster_times[i], pydistort_times[i], pydistort_nojac_times[i]])



