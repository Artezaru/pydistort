import numpy as np
import pytest
import time
import cv2
from pydistort.objects import Intrinsic, IntrinsicResult  # Adjust if needed

@pytest.fixture
def default_intrinsic():
    """Creates a default Intrinsic object with known parameters."""
    matrix = np.array([[1000.0, 0.0, 320.0],
                       [0.0, 1000.0, 240.0],
                       [0.0,    0.0,   1.0]])
    return Intrinsic(matrix)

def test_transform_2d_points(default_intrinsic):
    """Test basic transformation of 2D normalized points to image coordinates."""
    points = np.array([[0.0, 0.0],
                       [0.1, 0.2]])
    result = default_intrinsic.transform(points)
    expected = np.array([[320.0, 240.0],
                         [420.0, 440.0]])
    np.testing.assert_allclose(result.image_points, expected)

def test_jacobian_dx_correct_shape_and_values(default_intrinsic):
    """Test the shape and values of the dx Jacobian for single input."""
    points = np.array([[0.0, 0.0]])
    result = default_intrinsic.transform(points, dx=True)
    dx = result.jacobian_dx[0]
    expected = np.array([[1000.0, 0.0],
                         [0.0, 1000.0]])
    assert dx.shape == (2, 2)
    np.testing.assert_allclose(dx, expected)

def test_jacobian_dp_correct_shape_and_values(default_intrinsic):
    """Test the shape and values of the dp Jacobian for single input."""
    points = np.array([[0.2, 0.3]])
    result = default_intrinsic.transform(points, dp=True)
    dp = result.jacobian_dp[0]
    expected = np.array([[0.2, 0.0, 1.0, 0.0],
                         [0.0, 0.3, 0.0, 1.0]])
    assert dp.shape == (2, 4)
    np.testing.assert_allclose(dp, expected)

def test_transform_grid_and_jacobians(default_intrinsic):
    """Test transformation and Jacobians over a 2D grid of normalized points."""
    grid = np.array([[[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
                     [[0.0, 0.1], [0.1, 0.1], [0.2, 0.1]]])  # shape (2, 3, 2)
    result = default_intrinsic.transform(grid, dx=True, dp=True)

    # Check shape of image points and Jacobians
    assert result.image_points.shape == (2, 3, 2)
    assert result.jacobian_dx.shape == (2, 3, 2, 2)
    assert result.jacobian_dp.shape == (2, 3, 2, 4)

    # Check correctness of some expected outputs
    for i in range(2):
        for j in range(3):
            x, y = grid[i, j]
            expected_point = np.array([1000 * x + 320, 1000 * y + 240])
            expected_dx = np.array([[1000.0, 0.0], [0.0, 1000.0]])
            expected_dp = np.array([[x, 0, 1, 0], [0, y, 0, 1]])
            np.testing.assert_allclose(result.image_points[i, j], expected_point)
            np.testing.assert_allclose(result.jacobian_dx[i, j], expected_dx)
            np.testing.assert_allclose(result.jacobian_dp[i, j], expected_dp)

def test_transpose_input(default_intrinsic):
    """Test transformation when input points are in transposed shape (2, N)."""
    points = np.array([[0.0, 0.1, 0.2],
                       [0.0, 0.2, 0.4]])  # Shape (2, 3)
    result = default_intrinsic.transform(points, transpose=True)

    expected = np.empty_like(points)
    expected[0] = 1000 * points[0] + 320
    expected[1] = 1000 * points[1] + 240

    assert result.image_points.shape == (2, 3)
    np.testing.assert_allclose(result.image_points, expected)

