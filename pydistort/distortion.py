from typing import Optional, Tuple
from abc import abstractmethod
from dataclasses import dataclass
import numpy

from .transform import Transform, TransformResult, InverseTransformResult




@dataclass
class DistortionResult(TransformResult):
    r"""
    Subclass of TransformResult to represent the result of the distortion transformation.

    This class is used to store the result of transforming the ``normalized_points`` to ``distorted_points``, and the optional Jacobians.

    - ``transformed_points``: The transformed distorted points in the camera coordinate system. Shape (..., 2).
    - ``jacobian_dx``: The Jacobian of the distorted points with respect to the input normalized points if ``dx`` is True. Otherwise None. Shape (..., 2, 2), where the last dimension represents (dx, dy).
    - ``jacobian_dp``: The Jacobian of the distorted points with respect to the distortion parameters if ``dp`` is True. Otherwise None. Shape (..., 2, Nparams), where Nparams is the number of distortion parameters.

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
    
@dataclass
class InverseDistortionResult(InverseTransformResult):
    r"""
    Subclass of InverseTransformResult to represent the result of the inverse distortion transformation.

    This class is used to store the result of transforming the ``distorted_points`` back to ``normalized_points``, and the optional Jacobians.

    - ``transformed_points``: The transformed normalized points in the camera coordinate system. Shape (..., 2).
    - ``jacobian_dx``: The Jacobian of the normalized points with respect to the input distorted points if ``dx`` is True. Otherwise None. Shape (..., 2, 2), where the last dimension represents (dx, dy).
    - ``jacobian_dp``: The Jacobian of the normalized points with respect to the distortion parameters if ``dp`` is True. Otherwise None. Shape (..., 2, Nparams), where Nparams is the number of distortion parameters.

    Some properties are provided for convenience:

    - ``normalized_points``: Alias for ``transformed_points`` to represent the transformed normalized points. Shape (..., 2).

    .. note::

        If no distortion is applied, the ``normalized_points`` are equal to the ``distorted_points``.

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





class Distortion(Transform):
    r"""
    .. note::

        This class represents the distortion transformation, which is the central step of the process.

    The process to correspond a 3D-world point to a 2D-image point in the stenopic camera model is as follows:

    1. The ``world_3dpoints`` (:math:`X_W`) are expressed in the camera coordinate system using the rotation and translation matrices to obtain the ``camera_3dpoints`` (:math:`X_C`).
    2. The ``camera_3dpoints`` (:math:`X_C`) are normalized by dividing by the third coordinate to obtain the ``normalized_points`` (:math:`x_N`).
    3. The ``normalized_points`` (:math:`x_N`) are distorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``distorted_points`` (:math:`x_D`).
    4. The ``distorted_points`` (:math:`x_D`) are projected onto the image plane using the intrinsic matrix K to obtain the ``image_points`` (:math:`x_I`).

    This tranformation can be decomposed into 3 main steps:

    1. **Extrinsic**: Transform the ``world 3dpoints`` to ``normalized_points`` using the extrinsic parameters (rotation and translation).
    2. **Distortion**: Transform the ``normalized_points`` to ``distorted_points`` using the distortion model.
    3. **Intrinsic**: Transform the ``distorted_points`` to ``image_points`` using the intrinsic matrix K.

    .. note::

        This class is the base class for all distortion models.

        The subclasses should implement the following methods:

        - "parameters": property to return the distortion parameters in a numpy array.
        - "Nparams": property to return the number of distortion parameters.
        - "is_set": to check if the distortion parameters are set.
        - "_transform": to apply distortion to a set of points. The transformation is applied to the ``normalized_points`` (:math:`x_N`) to obtain the ``distorted_points`` (:math:`x_D`).
        - "_inverse_transform": to remove distortion from a set of points. The transformation is applied to the ``distorted_points`` (:math:`x_D`) to obtain the ``normalized_points`` (:math:`x_N`).

    Some aliases are provided for convenience:

    - ``distort``: Alias for the `transform` method to apply distortion to a set of points.
    - ``undistort``: Alias for the `inverse_transform` method to remove distortion from a set of points.

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
        return DistortionResult
    
    @property
    def inverse_result_class(self) -> type:
        return InverseDistortionResult
    
    # =============================================
    # Aliases for Distortion Class
    # =============================================
    def distort(self, *args, **kwargs) -> DistortionResult:
        r"""
        Alias for the `transform` method to apply distortion to a set of points.

        .. seealso::

            - `pydistort.Transform.transform` for more details on the transformation process.

        Parameters
        ----------
        *args, **kwargs
            Arguments and keyword arguments to be passed to the `transform` method.

        Returns
        -------
        DistortionResult
            The result of the distortion transformation.
        """
        return self.transform(*args, **kwargs)
    
    def undistort(self, *args, **kwargs) -> InverseDistortionResult:
        r"""
        Alias for the `inverse_transform` method to remove distortion from a set of points.

        .. seealso::

            - `pydistort.Transform.inverse_transform` for more details on the inverse transformation process.

        Parameters
        ----------
        *args, **kwargs
            Arguments and keyword arguments to be passed to the `inverse_transform` method.

        Returns
        -------
        InverseDistortionResult
            The result of the inverse distortion transformation.
        """
        return self.inverse_transform(*args, **kwargs)

    # =============================================
    # Additive Properties for Distortion Class
    # =============================================
    @property
    @abstractmethod
    def parameters(self) -> Optional[numpy.ndarray]:
        r"""
        Property to return the distortion parameters.

        This property should be implemented in the subclasses to return the distortion parameters as a numpy array.

        Returns
        -------
        numpy.ndarray
            The distortion parameters as a numpy array or None if not set.
        """
        raise NotImplementedError("The parameters property must be implemented in the subclass.")
    
    # =============================================
    # Common Properties and Methods for Distortion Class
    # =============================================  
    def __repr__(self) -> str:
        r"""
        String representation of the Distortion class.

        Returns
        -------
        str
            A string representation of the distortion model.
        """
        return f"{self.__class__.__name__} with {self.Nparams} parameters: {self.parameters if self.is_set() else 'not set'}"
