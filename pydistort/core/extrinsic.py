from dataclasses import dataclass
import numpy

from .transform import Transform, TransformResult




@dataclass
class ExtrinsicResult(TransformResult):
    r"""
    Subclass of :class:`pydistort.core.TransformResult` to represent the result of the extrinsic transformation.

    This class is used to store the result of transforming the ``world_3dpoints`` to ``normalized_points``, and the optional Jacobians.

    - ``transformed_points``: The transformed normalized points in the camera coordinate system. Shape (..., 2).
    - ``jacobian_dx``: The Jacobian of the normalized points with respect to the input world 3D points if ``dx`` is True. Otherwise None. Shape (..., 2, 3), where the last dimension represents (dx, dy, dz).
    - ``jacobian_dp``: The Jacobian of the normalized points with respect to the extrinsic parameters if ``dp`` is True. Otherwise None. Shape (..., 2, Nparams), where Nparams is the number of extrinsic parameters.

    Some properties are provided for convenience:

    - ``normalized_points``: Alias for ``transformed_points`` to represent the transformed normalized points. Shape (..., 2).

    .. warning::

        If ``transpose`` is set to True during the transformation, the output points will have shape (output_dim, ...) instead of (..., output_dim), same for the Jacobian matrices.

    """
    @property
    def normalized_points(self) -> numpy.ndarray:
        r"""
        Get the transformed normalized points.

        Returns
        -------
        numpy.ndarray
            The transformed normalized points in the camera coordinate system. Shape (..., 2).
        """
        return self.transformed_points
    


@dataclass
class InverseExtrinsicResult(TransformResult):
    r"""
    Subclass of :class:`pydistort.core.TransformResult` to represent the result of the inverse extrinsic transformation.

    This class is used to store the result of transforming the ``normalized_points`` back to ``world_3dpoints``, and the optional Jacobians.

    - ``transformed_points``: The transformed world 3D points in the camera coordinate system. Shape (..., 3).
    - ``jacobian_dx``: The Jacobian of the world 3D points with respect to the input normalized points if ``dx`` is True. Otherwise None. Shape (..., 3, 2), where the last dimension represents (dx, dy).
    - ``jacobian_dp``: The Jacobian of the world 3D points with respect to the extrinsic parameters if ``dp`` is True. Otherwise None. Shape (..., 3, Nparams), where Nparams is the number of extrinsic parameters.

    Some properties are provided for convenience:

    - ``world_3dpoints``: Alias for ``transformed_points`` to represent the transformed world 3D points. Shape (..., 3).

    .. warning::

        If ``transpose`` is set to True during the transformation, the output points will have shape (output_dim, ...) instead of (..., output_dim), same for the Jacobian matrices.

    """
    @property
    def world_3dpoints(self) -> numpy.ndarray:
        r"""
        Get the transformed world 3D points.

        Returns
        -------
        numpy.ndarray
            The transformed world 3D points in the camera coordinate system. Shape (..., 3).
        """
        return self.transformed_points
    


class Extrinsic(Transform):
    r"""
    .. note::

        This class represents the extrinsic transformation, which is the central step of the process.

    The process to correspond a 3D-world point to a 2D-image point in the stenopic camera model is as follows:

    1. The ``world_3dpoints`` (:math:`X_W`) are expressed in the camera coordinate system using the rotation and translation matrices to obtain the ``camera_3dpoints`` (:math:`X_C`).
    2. The ``camera_3dpoints`` (:math:`X_C`) are normalized by dividing by the third coordinate to obtain the ``normalized_points`` (:math:`x_N`).
    3. The ``normalized_points`` (:math:`x_N`) are distorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``distorted_points`` (:math:`x_D`).
    4. The ``distorted_points`` (:math:`x_D`) are projected onto the image plane using the intrinsic matrix K to obtain the ``image_points`` (:math:`x_I`).

    This tranformation can be decomposed into 3 main steps:

    1. **Extrinsic**: Transform the ``world_3dpoints`` to ``normalized_points`` using the extrinsic parameters (rotation and translation), (this class).
    2. **Distortion**: Transform the ``normalized_points`` to ``distorted_points`` using the distortion model.
    3. **Intrinsic**: Transform the ``distorted_points`` to ``image_points`` using the intrinsic matrix K.

    .. note::

        This class is the base class for all extrinsic models.

        The subclasses should implement the following methods:

        - "parameters": property to return the extrinsic parameters in a 1D numpy array.
        - "is_set": to check if the extrinsic parameters are set.
        - "_transform": to apply extrinsic to a set of points. The transformation is applied to the ``world_3dpoints`` (:math:`X_W`) to obtain the ``normalized_points`` (:math:`x_N`).
        - "_inverse_transform": to remove extrinsic from a set of points. The transformation is applied to the ``normalized_points`` (:math:`x_N`) to obtain the ``world_3dpoints`` (:math:`X_W`).

    """
    # =============================================
    # Properties for ABC Transform Class
    # =============================================
    @property
    def input_dim(self) -> int:
        return 3 # The input is a 2D point (x, y)
    
    @property
    def output_dim(self) -> int:
        return 2 # The output is a 2D point (x, y)
    
    @property
    def result_class(self) -> type:
        return ExtrinsicResult
    
    @property
    def inverse_result_class(self) -> type:
        return InverseExtrinsicResult
