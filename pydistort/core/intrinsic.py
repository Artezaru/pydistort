from dataclasses import dataclass
import numpy

from .transform import Transform, TransformResult




@dataclass
class IntrinsicResult(TransformResult):
    r"""
    Subclass of :class:`pydistort.core.TransformResult` to represent the result of the intrinsic transformation.

    This class is used to store the result of transforming the ``distorted_points`` (or ``normalized_points`` if no distortion is applied) to ``image_points``, and the optional Jacobians.

    - ``transformed_points``: The transformed image points in the camera coordinate system. Shape (..., 2).
    - ``jacobian_dx``: The Jacobian of the image points with respect to the input distorted points if ``dx`` is True. Otherwise None. Shape (..., 2, 2), where the last dimension represents (dx, dy).
    - ``jacobian_dp``: The Jacobian of the image points with respect to the intrinsic parameters if ``dp`` is True. Otherwise None. Shape (..., 2, Nparams), where Nparams is the number of intrinsic parameters.

    Some properties are provided for convenience:

    - ``image_points``: Alias for ``transformed_points`` to represent the transformed image points. Shape (..., 2).

    .. note::

        If no distortion is applied, the ``distorted_points`` are equal to the ``normalized_points``.

    .. warning::

        If ``transpose`` is set to True during the transformation, the output points will have shape (output_dim, ...) instead of (..., output_dim), same for the Jacobian matrices.

    """
    @property
    def image_points(self) -> numpy.ndarray:
        r"""
        Get the transformed image points.

        Returns
        -------
        numpy.ndarray
            The transformed image points in the camera coordinate system. Shape (..., 2).
        """
        return self.transformed_points
    


@dataclass
class InverseIntrinsicResult(TransformResult):
    r"""
    Subclass of :class:`pydistort.core.TransformResult` to represent the result of the inverse intrinsic transformation.

    This class is used to store the result of transforming the ``image_points`` back to ``distorted_points`` (or ``normalized_points`` if no distortion is applied), and the optional Jacobians.

    - ``transformed_points``: The transformed distorted points in the camera coordinate system. Shape (..., 2).
    - ``jacobian_dx``: The Jacobian of the distorted points with respect to the input image points if ``dx`` is True. Otherwise None. Shape (..., 2, 2), where the last dimension represents (dx, dy).
    - ``jacobian_dp``: The Jacobian of the distorted points with respect to the intrinsic parameters if ``dp`` is True. Otherwise None. Shape (..., 2, Nparams), where Nparams is the number of intrinsic parameters.

    Some properties are provided for convenience:

    - ``distorted_points``: Alias for ``transformed_points`` to represent the transformed distorted points. Shape (..., 2).

    .. note::

        If no distortion is applied, the ``distorted_points`` are equal to the ``normalized_points``.

    .. warning::

        If ``transpose`` is set to True during the transformation, the output points will have shape (output_dim, ...) instead of (..., output_dim), same for the Jacobian matrices.

    """
    @property
    def distorted_points(self) -> numpy.ndarray:
        r"""
        Get the transformed distorted points.

        Returns
        -------
        numpy.ndarray
            The transformed distorted points in the camera coordinate system. Shape (..., 2).
        """
        return self.transformed_points




class Intrinsic(Transform):
    r"""
    .. note::

        This class represents the intrinsic transformation, which is the central step of the process.

    The process to correspond a 3D-world point to a 2D-image point in the stenopic camera model is as follows:

    1. The ``world_3dpoints`` (:math:`X_W`) are expressed in the camera coordinate system using the rotation and translation matrices to obtain the ``camera_3dpoints`` (:math:`X_C`).
    2. The ``camera_3dpoints`` (:math:`X_C`) are normalized by dividing by the third coordinate to obtain the ``normalized_points`` (:math:`x_N`).
    3. The ``normalized_points`` (:math:`x_N`) are distorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``distorted_points`` (:math:`x_D`).
    4. The ``distorted_points`` (:math:`x_D`) are projected onto the image plane using the intrinsic matrix K to obtain the ``image_points`` (:math:`x_I`).

    This tranformation can be decomposed into 3 main steps:

    1. **Extrinsic**: Transform the ``world_3dpoints`` to ``normalized_points`` using the extrinsic parameters (rotation and translation).
    2. **Distortion**: Transform the ``normalized_points`` to ``distorted_points`` using the distortion model.
    3. **Intrinsic**: Transform the ``distorted_points`` to ``image_points`` using the intrinsic matrix K, (this class).

    .. note::

        This class is the base class for all intrinsic models.

        The subclasses should implement the following methods:

        - "parameters": property to return the intrinsic parameters in a 1D numpy array.
        - "is_set": to check if the intrinsic parameters are set.
        - "_transform": to apply intrinsic to a set of points. The transformation is applied to the ``distorted_points`` (:math:`x_D`) to obtain the ``image_points`` (:math:`x_I`).
        - "_inverse_transform": to remove intrinsic from a set of points. The transformation is applied to the ``image_points`` (:math:`x_I`) to obtain the ``distorted_points`` (:math:`x_D`).

    """
    # =============================================
    # Properties for ABC Transform Class
    # =============================================
    @property
    def input_dim(self) -> int:
        return 2 # The input is a 2D point (x, y)
    
    @property
    def output_dim(self) -> int:
        return 2 # The output is a 2D point (x, y)
    
    @property
    def result_class(self) -> type:
        return IntrinsicResult
    
    @property
    def inverse_result_class(self) -> type:
        return InverseIntrinsicResult
    
    @property
    def _jacobian_short_hand(self) -> dict:
        return {}
    

