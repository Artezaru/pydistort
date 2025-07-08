import pytest
import numpy

from pydistort import Cv2Distortion

@pytest.fixture
def default():
    """Creates a default Distortion object with known parameters."""
    return Cv2Distortion(parameters=numpy.array([1e-3, 2e-3, 1e-4, 1e-4, 2e-5]), Nparams=5)

def test_parameters(default):
    """Test the parameters property returns the correct distortion parameters."""
    assert numpy.allclose(default.parameters, numpy.array([
        1e-3,  # k1
        2e-3,  # k2
        1e-4,  # p1
        1e-4,  # p2
        2e-5   # k3
    ]))

def test_jacobian_analytic_numeric_match(default):
    """Test that the analytic Jacobian of the distortion transform matches the numeric approximation."""
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

    # --- dp (∂output/∂distortion parameters) ---
    dp_labels = ["k1", "k2", "p1", "p2", "k3"]
    param_vec = default.parameters
    for i in range(len(dp_labels)):
        param_plus = param_vec.copy()
        param_plus[i] += epsilon
        distortion_plus = Cv2Distortion()
        distortion_plus.parameters = param_plus
        result_plus = distortion_plus.transform(points, dx=False, dp=False)
        jacobian_numeric = (result_plus.transformed_points - result_analytic.transformed_points) / epsilon
        try:
            numpy.testing.assert_allclose(
                result_analytic.jacobian_dp[..., i], jacobian_numeric, rtol=1e-3, atol=1e-5
            )
        except AssertionError as e:
            print(f"Jacobian mismatch with respect to parameter '{dp_labels[i]}'")
            raise e
