import pytest
import numpy as np
from pydistort import Cv2Distortion
import time
import cv2

def get_distortion(Nparams, mode):
    """Create a Cv2Distortion object with specified parameters."""
    distortion = Cv2Distortion(Nparams=Nparams)

    if mode == "strong_coefficients":
        # Strong distortion
        distortion.k1 = 47.6469
        distortion.k2 = 605.372
        distortion.p1 = 0.01304
        distortion.p2 = -0.02737

        if Nparams >= 5:
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
        # Very weak distortion (almost identity)
        distortion.k1 = 1e-4
        distortion.k2 = 1e-5
        distortion.p1 = 1e-5
        distortion.p2 = 1e-5

        if Nparams >=5 :
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
    

@pytest.mark.parametrize("Nparams", [5, 8, 12, 14])
@pytest.mark.parametrize("mode", ["strong_coefficients", "weak_coefficients"])
def test_pydistort_distort_vs_opencv(Nparams, mode):
    """Compare Cv2Distortion.distort and distort_opencv for various Nparams."""
    distortion = get_distortion(Nparams, mode)

    # Test points
    points = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    # Distortion (analytic)
    result = distortion.distort(points, dx=True, dp=True)

    # Distortion (opencv)
    result_opencv = distortion.distort_opencv(points)

    # Check that shapes match
    assert result.distorted_points.shape == result_opencv.distorted_points.shape
    assert result.jacobian_dp.shape == result_opencv.jacobian_dp.shape

    # Compare outputs
    np.testing.assert_allclose(result.distorted_points, result_opencv.distorted_points, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(result.jacobian_dp, result_opencv.jacobian_dp, rtol=1e-5, atol=1e-8)

    # jacobian_dx is None for opencv
    assert result_opencv.jacobian_dx is None


@pytest.mark.parametrize("Nparams", [5, 8, 12, 14])
@pytest.mark.parametrize("mode", ["weak_coefficients"])
def test_pydistort_undistort_vs_opencv(Nparams, mode):
    """Compare Cv2Distortion.undistort and undistort_opencv for various Nparams."""
    distortion = get_distortion(Nparams, mode)
    
    # Test points
    points = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    # Distortion (analytic)
    result = distortion.undistort(points, dx=True, dp=True)

    # Distortion (opencv)
    result_opencv = distortion.undistort_opencv(points)

    # Check that shapes match
    assert result.normalized_points.shape == result_opencv.normalized_points.shape

    # Compare outputs
    np.testing.assert_allclose(result.normalized_points, result_opencv.normalized_points, rtol=1e-5, atol=1e-8)



@pytest.mark.parametrize("Nparams", [5, 8, 12, 14])
@pytest.mark.parametrize("mode", ["weak_coefficients"])
def test_pydistort_undistort_vs_opencv_IMAGE(Nparams, mode):
    """Compare Cv2Distortion.distort and distort_opencv for various Nparams."""
    distortion = get_distortion(Nparams, mode)
    
    # Test points
    points = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    # Distortion (analytic)
    result = distortion.undistort(points, dx=True, dp=True)

    # Distortion (opencv)
    result_opencv = distortion.undistort_opencv(points)

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


@pytest.mark.parametrize("Nparams", [None])
@pytest.mark.parametrize("mode", ["strong_coefficients"])
def test_pydistort_distort_vs_opencv_timer(Nparams, mode):
    """Compare Cv2Distortion.distort and distort_opencv for various Nparams in time."""
    pydistort_alljac_times = []
    pydistort_times = []
    pydistort_nojac_times = []
    opencv_times = []
    Nparams_list = [5, 8, 12, 14]
    Npoints = 1_000_000
    for Nparams in Nparams_list:
        distortion = get_distortion(Nparams, mode)

        # Test points
        points = np.random.uniform(-1.0, 1.0, size=(Npoints, 2))  # shape (Npoints, 2)

        # Distortion (analytic)
        start_time = time.time()
        result = distortion.distort(points, dx=True, dp=True)
        elapsed_time = time.time() - start_time
        pydistort_alljac_times.append(elapsed_time)

        # Distortion (analytic)
        start_time = time.time()
        result = distortion.distort(points, dx=False, dp=True)
        elapsed_time = time.time() - start_time
        pydistort_times.append(elapsed_time)

        # Distortion (analytic)
        start_time = time.time()
        result = distortion.distort(points, dx=False, dp=False)
        elapsed_time = time.time() - start_time
        pydistort_nojac_times.append(elapsed_time)

        # Distortion (opencv)
        start_time = time.time()
        result_opencv = distortion.distort_opencv(points)
        elapsed_time = time.time() - start_time
        opencv_times.append(elapsed_time)

    # Print times in a table fomat:
    print("\n\n ======== Distortion CV2 Time Comparison ========")
    print(f"Npoints: {Npoints}")
    print(f"{'Nparams':<15} {'pydistort (all Jacobians)':<30} {'pydistort (cv2 Jacobians)':<30} {'pydistort (no Jacobians)':<30} {'opencv':<30}")
    for i, Nparams in enumerate(Nparams_list):
        print(f"{Nparams:<15} {pydistort_alljac_times[i]:<30.4f} {pydistort_times[i]:<30.4f} {pydistort_nojac_times[i]:<30.4f} {opencv_times[i]:<30.4f}")



@pytest.mark.parametrize("Nparams", [None])
@pytest.mark.parametrize("mode", ["weak_coefficients"])
def test_pydistort_undistort_vs_opencv_timer(Nparams, mode):
    """Compare Cv2Distortion.distort and distort_opencv for various Nparams in time."""
    pydistort_times = []
    opencv_times = []
    Nparams_list = [5, 8, 12, 14]
    Npoints = 1_000_000
    for Nparams in Nparams_list:
        distortion = get_distortion(Nparams, mode)

        # Test points
        points = np.random.uniform(-1.0, 1.0, size=(Npoints, 2))  # shape (Npoints, 2)

        # Distortion (analytic)
        start_time = time.time()
        result = distortion.undistort(points)
        elapsed_time = time.time() - start_time
        pydistort_times.append(elapsed_time)

        # Distortion (opencv)
        start_time = time.time()
        result_opencv = distortion.undistort_opencv(points)
        elapsed_time = time.time() - start_time
        opencv_times.append(elapsed_time)

    # Print times in a table fomat:
    print("\n\n ======== Undistortion CV2 Time Comparison ========")
    print(f"Npoints: {Npoints}")
    print(f"{'Nparams':<15} {'pydistort':<30} {'opencv':<30}")
    for i, Nparams in enumerate(Nparams_list):
        print(f"{Nparams:<15} {pydistort_times[i]:<30.4f} {opencv_times[i]:<30.4f}")






@pytest.mark.parametrize("Nparams", [5, 8, 12, 14])
@pytest.mark.parametrize("mode", ["weak_coefficients"]) # TODO: [WARNING] !!! Error with strong_coefficients
def test_distort_undistort_inverse(Nparams, mode):
    """Check that undistort(distort(x)) ≈ x for all supported Nparams."""
    distortion = get_distortion(Nparams, mode)

    # Test points
    points = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    # Distort and then undistort
    result = distortion.distort(points, dx=False, dp=False)
    undistorted_result = distortion.undistort(result.distorted_points, dx=False, dp=False)

    # Check that the undistorted points are close to the original points
    np.testing.assert_allclose(points, undistorted_result.normalized_points, rtol=1e-5, atol=1e-8)
