import pytest
import numpy
import cv2

from pydistort import Cv2Extrinsic

@pytest.fixture
def default():
    """Creates a default Extrinsic object with known parameters."""
    rvec = numpy.array([0.1, 0.2, 0.3])
    tvec = numpy.array([0.5, 0.5, 0.5])
    return Cv2Extrinsic(rvec, tvec)

def test_parameters(default):
    """Test the parameters property returns the correct extrinsic parameters."""
    assert numpy.allclose(default.parameters, numpy.array([0.1, 0.2, 0.3, 0.5, 0.5, 0.5]))

def test_transform_2d_points(default):
    """Test basic transformation of 3D world points to normalized coordinates."""
    world_points = numpy.array([[0.0, 0.0, 1.0],
                                [1.0, 0.0, 1.0],
                                [0.0, 1.0, 1.0],
                                [0.0, 0.0, 2.0]])  # Shape (4, 3)

    result = default.transform(world_points)
    normalized = result.transformed_points

    # Check that the output shape is correct
    assert normalized.shape == (4, 2)

    # We can at least check the central projection property: x/z, y/z
    rotation_matrix = default.rotation_matrix
    translation_vector = default.translation_vector
    cam_points = (rotation_matrix @ world_points.T + translation_vector.reshape(3, 1)).T
    expected_normalized = cam_points[:, :2] / cam_points[:, 2:3]

    numpy.testing.assert_allclose(normalized, expected_normalized, rtol=1e-5)

def test_transform_inverse_transform_consistency(default):
    """Test inverse transformation of normalized coordinates back to world points."""
    world_points = numpy.random.rand(100, 3)  # Random normalized points

    # Compute the depth manually for testing
    rotation_matrix = default.rotation_matrix
    translation_vector = default.translation_vector
    cam_points = (rotation_matrix @ world_points.T + translation_vector.reshape(3, 1)).T
    depth = cam_points[:, 2]  # Use the z-coordinate as depth

    # Transform the points
    transformed = default.transform(world_points)
    inverse_transformed = default.inverse_transform(transformed.transformed_points, depth=depth)
    numpy.testing.assert_allclose(inverse_transformed.transformed_points, world_points)


def test_jacobian_analytic_numeric_match(default):
    """Test that the analytic Jacobian of the extrinsic transform matches the numeric approximation."""
    points = numpy.random.rand(10, 3)  # Random 3D points
    result_analytic = default.transform(points, dx=True, dp=True)
    epsilon = 1e-8

    # --- dx (∂output/∂input) ---
    dx_labels = ["X", "Y"]
    for i in range(len(dx_labels)):
        points_plus = points.copy()
        points_plus[:, i] += epsilon
        result_plus = default.transform(points_plus, dx=False, dp=False)
        jacobian_numeric = (result_plus.transformed_points - result_analytic.transformed_points) / epsilon
        try:
            numpy.testing.assert_allclose(
                result_analytic.jacobian_dx[..., i], jacobian_numeric, rtol=1e-3, atol=1e-5
            )
        except AssertionError as e:
            print(f"Jacobian mismatch with respect to input coordinate '{dx_labels[i]}'")
            raise e

    # --- dp (∂output/∂extrinsic parameters) ---
    dp_labels = ["fx", "fy", "cx", "cy", "skew"]
    param_vec = default.parameters
    for i in range(len(dp_labels)):
        param_plus = param_vec.copy()
        param_plus[i] += epsilon
        extrinsic_plus = Cv2Extrinsic()
        extrinsic_plus.parameters = param_plus
        result_plus = extrinsic_plus.transform(points, dx=False, dp=False)
        jacobian_numeric = (result_plus.transformed_points - result_analytic.transformed_points) / epsilon
        try:
            numpy.testing.assert_allclose(
                result_analytic.jacobian_dp[..., i], jacobian_numeric, rtol=1e-3, atol=1e-5
            )
        except AssertionError as e:
            print(f"Jacobian mismatch with respect to parameter '{dp_labels[i]}'")
            raise e

def test_inverse_jacobian_analytic_numeric_match(default):
    """Test that the analytic inverse Jacobian matches the numeric approximation."""
    points = numpy.random.rand(10, 2)  # Random 2D points
    result_analytic = default.inverse_transform(points, dx=True, dp=True)
    epsilon = 1e-8

    # --- dx (∂output/∂input) ---
    dx_labels = ["X", "Y"]
    for i in range(len(dx_labels)):
        points_plus = points.copy()
        points_plus[:, i] += epsilon
        result_plus = default.inverse_transform(points_plus, dx=False, dp=False)
        jacobian_numeric = (result_plus.transformed_points - result_analytic.transformed_points) / epsilon
        try:
            numpy.testing.assert_allclose(
                result_analytic.jacobian_dx[..., i], jacobian_numeric, rtol=1e-3, atol=1e-5
            )
        except AssertionError as e:
            print(f"Jacobian mismatch with respect to input coordinate '{dx_labels[i]}'")
            raise e

    # --- dp (∂output/∂extrinsic parameters) ---
    dp_labels = ["fx", "fy", "cx", "cy", "skew"]
    param_vec = default.parameters
    for i in range(len(dp_labels)):
        param_plus = param_vec.copy()
        param_plus[i] += epsilon
        extrinsic_plus = Cv2Extrinsic()
        extrinsic_plus.parameters = param_plus
        result_plus = extrinsic_plus.inverse_transform(points, dx=False, dp=False)
        jacobian_numeric = (result_plus.transformed_points - result_analytic.transformed_points) / epsilon
        try:
            numpy.testing.assert_allclose(
                result_analytic.jacobian_dp[..., i], jacobian_numeric, rtol=1e-3, atol=1e-5
            )
        except AssertionError as e:
            print(f"Jacobian mismatch with respect to parameter '{dp_labels[i]}'")
            raise e

def test_custom_jacobian_view(default):
    """Test that custom Jacobian views can be created and accessed correctly."""
    points = numpy.random.rand(10, 3)  # Random 3D points
    result = default.transform(points, dx=True, dp=True)

    numpy.testing.assert_allclose(
        result.jacobian_dr, result.jacobian_dp[..., 0:3], rtol=1e-3, atol=1e-5
    )
    numpy.testing.assert_allclose(
        result.jacobian_dt, result.jacobian_dp[..., 3:6], rtol=1e-3, atol=1e-5
    )

def test_rays_creation(default):
    """Test the creation of rays from the camera to the scene."""
    points = numpy.random.rand(1, 2)  # Random 2D points

    rvec = default.rotation_vector
    tvec = default.translation_vector
    
    def ray_one_point(point, rvec, tvec):
        """Create a ray for a single point."""
        rmat, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to rotation matrix
        origin = (- rmat.T @ tvec.reshape(3, 1)).flatten()  # Camera origin in world coordinates
        world = (rmat.T @ (numpy.array([point[0], point[1], 1]) - tvec).reshape(3, 1)).flatten()  # World coordinates of the point
        ray = numpy.zeros((6,), dtype=numpy.float64)
        ray[:3] = world  # World coordinates of the point
        ray[3:] = (world - origin) / numpy.linalg.norm(world - origin)  # Direction vector
        return ray
    
    expected_rays = numpy.array([ray_one_point(point, rvec, tvec) for point in points])
    rays = default.compute_rays(points)

    numpy.testing.assert_allclose(rays, expected_rays, rtol=1e-5, atol=1e-8)
    assert rays.shape == (points.shape[0], 6)  # Each ray should have 6 components (origin + direction)