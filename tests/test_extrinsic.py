import numpy as np
import pytest
from pydistort import Extrinsic
import time
import cv2

@pytest.fixture
def default_extrinsic():
    """Creates a default Extrinsic object with small rotation and translation."""
    rvec = np.array([0.1, 0.2, 0.3])
    tvec = np.array([0.5, 0.5, 0.5])
    return Extrinsic(rvec, tvec)

def test_transform_points_shape_and_values(default_extrinsic):
    """Test that transforming known 3D points gives expected normalized coordinates."""
    world_points = np.array([[0.0, 0.0, 1.0],
                             [1.0, 0.0, 1.0],
                             [0.0, 1.0, 1.0],
                             [0.0, 0.0, 2.0]])  # Shape (4, 3)

    result = default_extrinsic.transform(world_points)
    normalized = result.normalized_points

    # Check that the output shape is correct
    assert normalized.shape == (4, 2)

    # We can at least check the central projection property: x/z, y/z
    rotation_matrix = default_extrinsic.rotation_matrix
    translation_vector = default_extrinsic.translation_vector
    cam_points = (rotation_matrix @ world_points.T + translation_vector.reshape(3, 1)).T
    expected_normalized = cam_points[:, :2] / cam_points[:, 2:3]

    np.testing.assert_allclose(normalized, expected_normalized, rtol=1e-5)

def test_jacobian_dx_shape_and_type(default_extrinsic):
    """Test that the dx Jacobian has the correct shape for one point."""
    point = np.array([[1.0, 2.0, 3.0]])
    result = default_extrinsic.transform(point, dx=True)
    dx = result.jacobian_dx
    assert dx.shape == (1, 2, 3)
    assert isinstance(dx, np.ndarray)

def test_jacobian_dp_shape_and_type(default_extrinsic):
    """Test that the dp Jacobian has the correct shape for one point."""
    point = np.array([[1.0, 2.0, 3.0]])
    result = default_extrinsic.transform(point, dp=True)
    dp = result.jacobian_dp
    assert dp.shape == (1, 2, 6)
    assert isinstance(dp, np.ndarray)

def test_transform_grid_and_jacobians(default_extrinsic):
    """Test transformation and Jacobians over a 3D grid of points."""
    grid = np.array([[[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                      [[1.0, 2.0, 1.0], [2.0, 2.0, 1.0]]]])  # shape (1, 2, 2, 3)

    result = default_extrinsic.transform(grid, dx=True, dp=True)

    # Check shapes
    assert result.normalized_points.shape == (1, 2, 2, 2)
    assert result.jacobian_dx.shape == (1, 2, 2, 2, 3)
    assert result.jacobian_dp.shape == (1, 2, 2, 2, 6)

def test_compare_with_manual_projection():
    """Compare the output of the Extrinsic class with manual transformation."""
    rvec = np.array([0.1, 0.2, 0.3])
    tvec = np.array([0.5, -0.2, 0.3])
    extrinsic = Extrinsic(rvec, tvec)

    point = np.array([[1.0, 0.0, 0.0]])
    result = extrinsic.transform(point)
    normalized = result.normalized_points[0]

    # Manually compute rotation matrix and projection
    from cv2 import Rodrigues
    R, _ = Rodrigues(rvec)
    t = tvec.reshape((3, 1))
    cam_point = (R @ point.T + t).flatten()
    expected = cam_point[:2] / cam_point[2]

    np.testing.assert_allclose(normalized, expected, rtol=1e-6)
