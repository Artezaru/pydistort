import numpy as np
import pytest
from pydistort import Cv2Distortion, undistort_points
import cv2
import time

def get_distortion(Nparams, mode):
    """Create a Cv2Distortion object with specified parameters."""
    distortion = Cv2Distortion(Nparams=Nparams)

    if mode == "strong_coefficients":
        distortion.k1 = 47.6469
        distortion.k2 = 605.372
        distortion.p1 = 0.01304
        distortion.p2 = -0.02737
        distortion.k3 = -1799.929
        if Nparams >= 8:
            distortion.k4 = 47.765
            distortion.k5 = 500.027
            distortion.k6 = 1810.745
        if Nparams >= 12:
            distortion.s1 = -0.0277
            distortion.s2 = 1.9759
            distortion.s3 = -0.0208
            distortion.s4 = 0.3596
        if Nparams == 14:
            distortion.taux = 2.0
            distortion.tauy = 5.0

    elif mode == "weak_coefficients":
        distortion.k1 = 1e-4
        distortion.k2 = 1e-5
        distortion.p1 = 1e-5
        distortion.p2 = 1e-5
        distortion.k3 = 1e-5
        if Nparams >= 8:
            distortion.k4 = 1e-5
            distortion.k5 = 1e-5
            distortion.k6 = 1e-5
        if Nparams >= 12:
            distortion.s1 = 1e-5
            distortion.s2 = 1e-5
            distortion.s3 = 1e-5
            distortion.s4 = 1e-5
        if Nparams == 14:
            distortion.taux = 1e-5
            distortion.tauy = 1e-5

    return distortion

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
@pytest.mark.parametrize("mode", ["weak_coefficients"])
def test_pydistort_undistort_vs_opencv(Nparams, mode):
    """Compare Cv2Distortion.undistort_points and OpenCV undistortPoints."""
    distortion = get_distortion(Nparams, mode)

    # Camera intrinsics
    fx, fy = 1000.0, 950.0
    cx, cy = 320.0, 240.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Rectification matrix (identity)
    R = np.array([0.01, 0.02, 0.03])  # small rotation
    R, _ = cv2.Rodrigues(R)  # Convert to rotation matrix

    # Projection matrix (identity)
    P = np.array([[fx + 20, 0, cx - 11], [0, fy + 25, cy + 12], [0, 0, 1]])

    # 2D points
    points = np.array([
        [400.0, 300.0],
        [450.0, 350.0],
        [500.0, 400.0],
        [550.0, 450.0],
        [600.0, 500.0]
    ])

    # Undistort with your method
    result = undistort_points(points, K=K, distortion=distortion, R=R, P=P)

    # Undistort with OpenCV
    points_cv = np.ascontiguousarray(points.reshape(-1, 1, 2), dtype=np.float64)
    undistorted_points_cv = cv2.undistortPoints(points_cv, K, distortion.parameters, R=R, P=P)
    undistorted_points_cv = np.asarray(undistorted_points_cv[:, 0, :], dtype=np.float64)

    # Comparaison
    np.testing.assert_allclose(result.undistorted_points, undistorted_points_cv, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("Nparams", [None])
@pytest.mark.parametrize("mode", ["weak_coefficients"])
def test_pydistort_undistort_vs_opencv_timer(Nparams, mode):
    """Compare pydistort.undistort_points and OpenCV undistortPoints for speed."""
    pydistort_times = []
    opencv_times = []
    Nparams_list = [5, 8, 12, 14]
    Npoints = 1_000_000
    for Nparams in Nparams_list:
        distortion = get_distortion(Nparams, mode)

        # Camera intrinsics
        fx, fy = 1000.0, 950.0
        cx, cy = 320.0, 240.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Rectification matrix (identity)
        R = np.array([0.01, 0.02, 0.03])  # small rotation
        R, _ = cv2.Rodrigues(R)  # Convert to rotation matrix

        # Projection matrix (identity)
        P = np.array([[fx + 20, 0, cx - 11], [0, fy + 25, cy + 12], [0, 0, 1]])

        # Test points
        points = np.random.uniform(100.0, 800.0, size=(Npoints, 2))  # shape (Npoints, 2)

        # Distortion (analytic)
        start_time = time.time()
        result = undistort_points(points, K=K, distortion=distortion, R=R, P=P)
        elapsed_time = time.time() - start_time
        pydistort_times.append(elapsed_time)

        # Distortion (opencv)
        start_time = time.time()
        points_cv = np.ascontiguousarray(points.reshape(-1, 1, 2), dtype=np.float64)
        undistorted_points_cv = cv2.undistortPoints(points_cv, K, distortion.parameters, R=R, P=P)
        undistorted_points_cv = np.asarray(undistorted_points_cv[:, 0, :], dtype=np.float64)
        elapsed_time = time.time() - start_time
        opencv_times.append(elapsed_time)

    # Print times in a table fomat:
    print("\n\n ======== Undistort Points CV2 Complete Time Comparison ========")
    print(f"Npoints: {Npoints}")
    print(f"{'Nparams':<15} {'pydistort':<30} {'opencv':<30}")
    for i, Nparams in enumerate(Nparams_list):
        print(f"{Nparams:<15} {pydistort_times[i]:<30.4f} {opencv_times[i]:<30.4f}")
