from abc import abstractmethod
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
        - "_compute_rays": to compute the rays emitted from the camera to the scene. This is used to compute the rays in the world coordinate system.

    """

    _result_class = ExtrinsicResult
    _inverse_result_class = InverseExtrinsicResult
    _jacobian_short_hand = {}

    # =============================================
    # Properties for ABC Transform Class
    # =============================================
    @property
    def input_dim(self) -> int:
        return 3 # The input is a 2D point (x, y)
    
    @property
    def output_dim(self) -> int:
        return 2 # The output is a 2D point (x, y)
    
    # =============================================
    # Additional ABC Methods
    # =============================================
    def compute_rays(
        self, 
        normalized_points: numpy.ndarray,
        *,
        transpose: bool = False,
        _skip: bool = False,
        **kwargs
        ) -> numpy.ndarray:
        r"""
        Compute the rays emitted from the camera to the scene.

        The rays are the concatenation of the normalized points and the direction of the rays in the world coordinate system.

        .. code-block:: python

            rays = compute_rays(normalized_points)

            rays.shape  # (..., 6)
            # The last dimension is the ray structure: (origin_x, origin_y, origin_z, direction_x, direction_y, direction_z)
            # Where the coordinates of the origin and the direction are in the world coordinate system.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points in the camera coordinate system. Shape (..., 2).

        transpose : bool, optional
            If True, the input and output arrays are transposed to shape (2, ...) and (6, ...), respectively. Default is False.

        _skip : bool, optional
            If True, skip the checks and transformations. Default is False.

        kwargs : dict
            Additional arguments to be passed to the transformation method.

        Returns
        -------
        numpy.ndarray
            The rays in the world coordinate system containing an origin (the normalized point) and a direction (the ray direction). Shape (..., 6).

        """
        if not _skip:
            # Check the boolean flags
            if not isinstance(transpose, bool):
                raise TypeError(f"transpose must be a boolean, got {type(transpose)}")
            
            # Check if the transformation is set
            if not self.is_set():
                raise ValueError("Transformation parameters are not set. Please set the parameters before transforming points.")
            
            # Convert input points to float64
            points = numpy.asarray(normalized_points, dtype=numpy.float64)

            # Check the shape of the input points
            if points.ndim < 2:
                raise ValueError(f"Input points must have at least 2 dimensions, got {points.ndim} dimensions.")
            
            # Transpose the input points if requested
            if transpose:
                points = numpy.moveaxis(points, 0, -1) # (output_dim, ...) -> (..., output_dim)
            
            # Save the shape of the input points
            shape = points.shape # (..., output_dim)

            # Check the last dimension of the input points
            if shape[-1] != self.output_dim:
                raise ValueError(f"Input points must have {self.output_dim} dimensions, got {shape[-1]} dimensions.")
            
            # Flatten the input points to 2D for processing
            points = points.reshape(-1, self.output_dim) # (Npoints, output_dim)

        # Apply the inverse transformation
        rays = self._compute_rays(points, **kwargs) # (Npoints, 6)

        if not _skip:
            # Reshape the transformed points to the original shape
            rays = rays.reshape(*shape[:-1], 6) # (Npoints, 6) -> (..., 6)

            # Transpose the transformed points if requested
            if transpose:
                rays = numpy.moveaxis(rays, -1, 0) # (..., 6) -> (6, ...)

        # Return the result as a InverseTransformResult object
        return rays
    

    @abstractmethod
    def _compute_rays(self, normalized_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        Computes the rays in the world coordinate system for the given normalized points.

        A ray is the concatenation of the normalized points with a z-coordinate of 1.0 representing the origin of the ray in the world coordinate system and a direction vector of (0, 0, 1) representing the direction of the ray in the world coordinate system.

        The ray structure is as follows:

        - The first 3 elements are the origin of the ray in the world coordinate system (the normalized points with z=1).
        - The last 3 elements are the direction of the ray in the world coordinate system, which is always (0, 0, 1) for the no extrinsic model. The direction vector is normalized.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points in the camera coordinate system. Shape (Npoints, 2).

        Returns
        -------
        numpy.ndarray
            The rays in the world coordinate system. Shape (Npoints, 6).
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
