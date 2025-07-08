import pytest
import numpy

from pydistort import SkewIntrinsic

@pytest.fixture
def default():
    """Creates a default Intrinsic object with known parameters."""
    matrix = numpy.array([[1000.0, 100.0, 320.0],
                          [0.0, 1200.0, 240.0],
                          [0.0,    0.0,   1.0]])
    return SkewIntrinsic(matrix)

def test_parameters(default):
    """Test the parameters property returns the correct intrinsic parameters."""
    assert numpy.allclose(default.parameters, numpy.array([1000.0, 1200.0, 320.0, 240.0, 100.0]))  # Including skew parameter

def test_transform_2d_points(default):
    """Test basic transformation of 2D normalized points to image coordinates."""
    points = numpy.array([[0.0, 0.0],
                          [0.1, 0.2]])
    result = default.transform(points)
    expected = numpy.array([[320.0, 240.0],
                            [440.0, 480.0]])
    numpy.testing.assert_allclose(result.transformed_points, expected)    

def test_inverse_transform_2d_points(default):
    """Test inverse transformation of image coordinates back to normalized points."""
    image_points = numpy.array([[320.0, 240.0],
                                [440.0, 480.0]])
    result = default.inverse_transform(image_points)
    expected = numpy.array([[0.0, 0.0],
                            [0.1, 0.2]])
    numpy.testing.assert_allclose(result.transformed_points, expected)

def test_transform_inverse_transform_consistency(default):
    """Test that transforming and then inverse transforming returns original points."""
    points = numpy.random.rand(100, 2)  # Random normalized points
    transformed = default.transform(points)
    inverse_transformed = default.inverse_transform(transformed.transformed_points)
    numpy.testing.assert_allclose(inverse_transformed.transformed_points, points)

def test_jacobian_analytic_numeric_match(default):
    """Test that the analytic Jacobian of the intrinsic transform matches the numeric approximation."""
    points = numpy.random.rand(10, 2)  # Random 2D points
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

    # --- dp (∂output/∂intrinsic parameters) ---
    dp_labels = ["fx", "fy", "cx", "cy", "skew"]
    param_vec = default.parameters
    for i in range(len(dp_labels)):
        param_plus = param_vec.copy()
        param_plus[i] += epsilon
        intrinsic_plus = SkewIntrinsic()
        intrinsic_plus.parameters = param_plus
        result_plus = intrinsic_plus.transform(points, dx=False, dp=False)
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

    # --- dp (∂output/∂intrinsic parameters) ---
    dp_labels = ["fx", "fy", "cx", "cy", "skew"]
    param_vec = default.parameters
    for i in range(len(dp_labels)):
        param_plus = param_vec.copy()
        param_plus[i] += epsilon
        intrinsic_plus = SkewIntrinsic()
        intrinsic_plus.parameters = param_plus
        result_plus = intrinsic_plus.inverse_transform(points, dx=False, dp=False)
        jacobian_numeric = (result_plus.transformed_points - result_analytic.transformed_points) / epsilon
        try:
            numpy.testing.assert_allclose(
                result_analytic.jacobian_dp[..., i], jacobian_numeric, rtol=1e-3, atol=1e-5
            )
        except AssertionError as e:
            print(f"Jacobian mismatch with respect to parameter '{dp_labels[i]}'")
            raise e

