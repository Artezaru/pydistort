import pytest
import numpy
from pydistort.core import TransformComposition
from pydistort import Cv2Intrinsic, Cv2Extrinsic, project_points



def test_compare_project_points_and_composition():
    """Fixture to create a TransformComposition with intrinsic and extrinsic parameters."""
    intrinsic = Cv2Intrinsic(intrinsic_matrix=[[1000, 0, 320],
                                               [0, 1000, 240],
                                               [0, 0, 1]])
    extrinsic = Cv2Extrinsic(rotation_vector=numpy.random.rand(3), 
                             translation_vector=numpy.random.rand(3))
    
    points = numpy.random.rand(100, 3)
    
    transform = TransformComposition([extrinsic, intrinsic])
    transform_result = transform.transform(points, dx=True, dp=True)

    project_result = project_points(points, extrinsic, None, intrinsic, dx=True, dp=True)

    numpy.testing.assert_allclose(
        transform_result.transformed_points,
        project_result.image_points,
        rtol=1e-5,
        err_msg="TransformComposition and project_points results do not match."
    )

    jdex = transform_result.jacobian_dp[..., 0:6]
    jdint = transform_result.jacobian_dp[..., 6:10]

    numpy.testing.assert_allclose(
        jdex,
        project_result.jacobian_dextrinsic,
        rtol=1e-5,
        err_msg="Jacobian with respect to extrinsic parameters does not match."
    )

    numpy.testing.assert_allclose(
        jdint,
        project_result.jacobian_dintrinsic,
        rtol=1e-5,
        err_msg="Jacobian with respect to intrinsic parameters does not match."
    )





