from abc import ABC, abstractmethod
from typing import Optional
from numbers import Number
import numpy


class DistortionResult(object):
    r"""
    Class to represent the result of the distortion transformation.

    This class is used to store the result of the distortion transformation and its jacobian matrices.

    The number of parameters depends on the distortion model used.

    .. note::

        ``...`` in the shape of the arrays means that the array can have any number of dimensions.
        Classically, the ``...`` can be replaced by :math:`N` which is the number of points.

    Parameters
    ----------
    distorted_points : numpy.ndarray
        The transformed distorted points in normalized coordinates. 
        It will be a 2D array of shape (..., 2) if ``transpose`` is False and a 2D array of shape (2, ...) if ``transpose`` is True.

    jacobian_dx : Optional[numpy.ndarray]
        The Jacobian of the distorted points with respect to the normalized points if ``dx`` is True. Otherwise None.
        It will be a 2D array of shape (..., 2, 2) if ``transpose`` are False and a 2D array of shape (2, ..., 2) if ``transpose`` is True.

    jacobian_dp : Optional[numpy.ndarray]
        The Jacobian of the distorted points with respect to the distortion parameters if ``dp`` is True. Otherwise None.
        It will be a 2D array of shape (..., 2, Nparams) if ``transpose`` are False and a 2D array of shape (2, ..., Nparams) if ``transpose`` is True.
    """
    def __init__(self, distorted_points: numpy.ndarray, jacobian_dx: Optional[numpy.ndarray], jacobian_dp: Optional[numpy.ndarray]):
        self.distorted_points = distorted_points
        self.jacobian_dx = jacobian_dx
        self.jacobian_dp = jacobian_dp


class UndistortResult(object):
    r"""
    Class to represent the result of the undistortion transformation.

    This class is used to store the result of the undistortion transformation.

    The number of parameters depends on the distortion model used.

    .. note::

        ``...`` in the shape of the arrays means that the array can have any number of dimensions.
        Classically, the ``...`` can be replaced by :math:`N` which is the number of points.

    Parameters
    ----------
    normalized_points : numpy.ndarray
        The transformed normalized points in normalized coordinates.
        It will be a 2D array of shape (..., 2) if ``transpose`` is False and a 2D array of shape (2, ...) if ``transpose`` is True.
    """
    def __init__(self, normalized_points: numpy.ndarray):
        self.normalized_points = normalized_points





# =============================================
# Distortion class
# =============================================
class Distortion(object):
    r"""
    Abstract base class for distortion models.

    This class defines the interface for distortion models used for cameras.

    In the pinhole camera model, the distortion is represented by a set of coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}`.
    The process to correspond a 3D-world point to a 2D-image point is as follows:

    1. The ``world_3dpoints`` (:math:`X_W`) are expressed in the camera coordinate system using the rotation and translation matrices to obtain the ``camera_3dpoints`` (:math:`X_C`).
    2. The ``camera_3dpoints`` (:math:`X_C`) are normalized by dividing by the third coordinate to obtain the ``normalized_points`` (:math:`x_N`).
    3. The ``normalized_points`` (:math:`x_N`) are distorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``distorted_points`` (:math:`x_D`).
    4. The ``distorted_points`` (:math:`x_D`) are projected onto the image plane using the intrinsic matrix K to obtain the ``image_points`` (:math:`x_I`).

    .. note::

        This class manage the transformation between the ``normalized_points`` and the ``distorted_points``.

    .. math::

        \begin{align*}
        X_C &= R \cdot X_W + T \\
        x_N &= \frac{X_C}{X_C[2]} \\
        x_D &= \text{distort}(x_N, \lambda_1, \lambda_2, \lambda_3, \ldots) \\
        x_I &= K \cdot x_D
        \end{align*}
        
    The subclasses should implement the following methods:

    - "parameters": property to return the distortion parameters.
    - "Nparams": property to return the number of distortion parameters.
    - "_is_set": to check if the distortion parameters are set.
    - "_distort": to apply distortion to a set of points. The transformation is applied to the ``normalized_points`` (:math:`x_N`) to obtain the ``distorted_points`` (:math:`x_D`).
    - "_undistort": to remove distortion from a set of points. The transformation is applied to the ``distorted_points`` (:math:`x_D`) to obtain the ``normalized_points`` (:math:`x_N`).
    
    """

    @property
    def Nparams(self) -> int:
        r"""
        Property to return the number of distortion parameters.

        This property should be implemented in the subclasses to return the number of distortion parameters.

        Returns
        -------
        int
            The number of distortion parameters.
        """
        raise NotImplementedError("The Nparams property must be implemented in the subclass.")

    @abstractmethod
    def is_set(self) -> bool:
        r"""
        Check if the distortion parameters are set.

        This method should be implemented in the subclasses to check if the distortion parameters are set.

        Returns
        -------
        bool
            True if the distortion parameters are set, False otherwise.
        """
        raise NotImplementedError("The is_set method must be implemented in the subclass.")


    def distort(self, normalized_points: numpy.ndarray, transpose: bool = False, dx: bool = False, dp: bool = False, **kwargs) -> DistortionResult:
        r"""
        Transform the given ``normalized points`` to ``distorted points`` using the distortion model.

        The given points ``normalized points`` are assumed to be in the camera coordinate system and expressed in normalized coordinates with shape (..., 2).
        
        .. note::

            ``...`` in the shape of the arrays means that the array can have any number of dimensions.
            Classically, the ``...`` can be replaced by :math:`N` which is the number of points.

        The equations used to transform the points are:

        .. math::

            (x_D, y_D) = \text{distort}(x_N, y_N, \lambda_1, \lambda_2, \lambda_3, \ldots)

        The output ``distorted points`` are in the camera coordinate system and expressed in normalized coordinates with shape (..., 2).

        .. warning::

            The points are converting to float type before applying the distortion model.

        The method also computes 2 Jacobian matrices if requested:

        - ``dx``: Jacobian of the image points with respect to the normalized points.
        - ``dp``: Jacobian of the image points with respect to the distortion parameters.

        The jacobian matrice with respect to the normalized points is a (..., 2, 2) matrix where :

        .. code-block:: python

            jacobian_dx[..., 0, 0]  # ∂x_D/∂x_N -> Jacobian of the coordinates x_D with respect to the coordinates x_N.
            jacobian_dx[..., 0, 1]  # ∂x_D/∂y_N
            
            jacobian_dx[..., 1, 0]  # ∂y_D/∂x_N -> Jacobian of the coordinates y_D with respect to the coordinates x_N.
            jacobian_dx[..., 1, 1]  # ∂y_D/∂y_N

        The Jacobian matrice with respect to the distortion parameters is a (..., 2, Nparams) matrix where :

        .. code-block:: python

            jacobian_dp[..., 0, 0]  # ∂x_D/∂lambda_1 -> Jacobian of the coordinates x_D with respect to the distortion parameters.
            jacobian_dp[..., 0, 1]  # ∂x_D/∂lambda_2
            jacobian_dp[..., 0, 2]  # ∂x_D/∂lambda_3
            ...

            jacobian_dp[..., 1, 0]  # ∂y_D/∂lambda_1 -> Jacobian of the coordinates y_D with respect to the distortion parameters.
            jacobian_dp[..., 1, 1]  # ∂y_D/∂lambda_2
            jacobian_dp[..., 1, 2]  # ∂y_D/∂lambda_3
            ...

        where Nparams is the number of distortion parameters.

        .. note::

            For consistency, an alias for this method is ``transform``.
        
        Parameters
        ----------
        normalized_points : numpy.ndarray
            Array of normalized points to be transformed with shape (..., 2).

        transpose : bool, optional
            If True, the input points are assume to have shape (2, ...).
            In this case, the output points will have shape (2, ...) as well and the jacobian matrices will have shape (2, ..., 2) and (2, ..., Nparams) respectively.
            Default is False.

        dx : bool, optional
            If True, the Jacobian of the distorted points with respect to the normalized points is computed. Default is False.
            The output will be a 2D array of shape (..., 2, 2) if ``transpose`` is False.
            If ``dx`` is False, the output will be None.

        dp : bool, optional
            If True, the Jacobian of the distorted points with respect to the distortion parameters is computed. Default is False.
            The output will be a 2D array of shape (..., 2, Nparams) if ``transpose`` is False.
            If ``dp`` is False, the output will be None.

        kwargs : dict, optional
            Additional keyword arguments to be passed to the distortion model.

        Returns
        -------
        distortion_result : DistortionResult

            The result of the distortion transformation containing the image points and the jacobian matrices.
            This object has the following attributes:

            image_points : numpy.ndarray
                The transformed image points in pixels. It will be a 2D array of shape (..., 2) if ``transpose`` is False.

            jacobian_dx : Optional[numpy.ndarray]
                The Jacobian of the image points with respect to the normalized points if ``dx`` is True. Otherwise None.
                It will be a 2D array of shape (..., 2, 2) if ``transpose`` is False.

            jacobian_dp : Optional[numpy.ndarray]
                The Jacobian of the image points with respect to the distortion parameters if ``dp`` is True. Otherwise None.
                It will be a 2D array of shape (..., 2, Nparams) if ``transpose`` is False.

        Developer Notes
        ----------------

        The subclasses should implement the following methods:

        - "_distort": to apply distortion to a set of points. 
        
        The transformation is applied to the ``normalized_points`` (:math:`x_N`) with shape (2, N) to obtain the ``distorted_points`` (:math:`x_D`) with shape (2, N).                        
        """
        # Check the boolean parameters
        if not isinstance(transpose, bool):
            raise ValueError("The transpose parameter must be a boolean.")
        if not isinstance(dx, bool):
            raise ValueError("The dx parameter must be a boolean.")
        if not isinstance(dp, bool):
            raise ValueError("The dp parameter must be a boolean.")
        
        # Check if the distortion parameters are set
        if not self.is_set():
            raise ValueError("The distortion parameters is not set. Please set the distortion parameters before using this method.")
        
        # Create the array of points
        points = numpy.asarray(normalized_points, dtype=numpy.float64) 

        # Transpose the points if needed
        if transpose:
            points = numpy.moveaxis(points, 0, -1) # (2, ...) -> (..., 2)

        # Extract the original shape
        shape = points.shape # (..., 2)

        # Flatten the points along the last axis
        points_flat = points.reshape(-1, shape[-1]) # shape (..., 2) -> shape (Npoints, 2)

        # Check the shape of the points
        if points_flat.ndim !=2 or points_flat.shape[1] != 2:
            raise ValueError(f"The points must be in the shape (Npoints, 2) or (2, Npoints) if ``transpose`` is True. Got {points_flat.shape} instead and transpose is {transpose}.")
        
        distorted_points_flat, jacobian_flat_dx, jacobian_flat_dp = self._distort(points_flat, dx=dx, dp=dp, **kwargs) # shape (Npoints, 2), (Npoints, 2, 2), (Npoints, 2, Nparams)

        # Reshape the image points back to the original shape
        Nparams = jacobian_flat_dp.shape[-1] if dp else None # Nparams = 4 or None
        distorted_points = distorted_points_flat.reshape(shape) # (Npoints, 2) -> (..., 2)
        jacobian_dx = jacobian_flat_dx.reshape((*shape, 2)) if dx else None # (Npoints, 2, 2) -> (..., 2, 2)
        jacobian_dp = jacobian_flat_dp.reshape((*shape, Nparams)) if dp else None # (Npoints, 2, Nparams) -> (..., 2, Nparams)

        # Transpose the points back to the original shape if needed
        if transpose:
            distorted_points = numpy.moveaxis(distorted_points, -1, 0) # (..., 2) -> (2, ...)
            jacobian_dx = numpy.moveaxis(jacobian_dx, -2, 0) if dx else None # (..., 2, 2) -> (2, ..., 2)
            jacobian_dp = numpy.moveaxis(jacobian_dp, -2, 0) if dp else None # (..., 2, Nparams) -> (2, ..., Nparams)

        # Return the image points and the jacobian matrices
        result = DistortionResult(distorted_points, jacobian_dx, jacobian_dp)
        return result
    

    def transform(self, normalized_points: numpy.ndarray, transpose: bool = False, dx: bool = False, dp: bool = False, **kwargs) -> DistortionResult:
        return self.distort(normalized_points, transpose=transpose, dx=dx, dp=dp, **kwargs)
    

    @abstractmethod
    def _distort(self, normalized_points: numpy.ndarray, dx: bool = False, dp: bool = False, **kwargs) -> tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Abstract method to apply distortion to a set of points.

        The subclasses should implement this method to apply the distortion model to the given points.

        .. note::

            For ``_distort`` the input is always in the shape (Npoints, 2) with float64 type.
            The output must be (Npoints, 2) for the distorted points and (Npoints, 2, 2) for the jacobian with respect to the normalized points and (Npoints, 2, Nparams) for the jacobian with respect to the distortion parameters.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            Array of normalized points to be transformed with shape (Npoints, 2).

        dx : bool, optional
            If True, the Jacobian of the distorted points with respect to the normalized points is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 2, 2).

        dp : bool, optional
            If True, the Jacobian of the distorted points with respect to the distortion parameters is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 2, Nparams).

        kwargs : dict, optional
            Additional keyword arguments to be passed to the distortion model.

        Returns
        -------
        distorted_points : numpy.ndarray
            The transformed distorted points in normalized coordinates. It will be a 2D array of shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The Jacobian of the distorted points with respect to the normalized points if ``dx`` is True. Otherwise None.
            It will be a 2D array of shape (Npoints, 2, 2).

        jacobian_dp : Optional[numpy.ndarray]
            The Jacobian of the distorted points with respect to the distortion parameters if ``dp`` is True. Otherwise None.
            It will be a 2D array of shape (Npoints, 2, Nparams).
        """
        raise NotImplementedError("The _distort method must be implemented in the subclass.")


    def undistort(self, distorted_points: numpy.ndarray, transpose: bool = False, **kwargs) -> UndistortResult:
        r"""
        Transform the given ``distorted points`` to ``normalized points`` using the distortion model.

        The given points ``distorted points`` are assumed to be in the camera coordinate system and expressed in normalized coordinates with shape (..., 2).
        
        .. note::

            ``...`` in the shape of the arrays means that the array can have any number of dimensions.
            Classically, the ``...`` can be replaced by :math:`N` which is the number of points.

        The equations used to transform the points are:

        .. math::

            (x_N, y_N) = \text{undistort}(x_D, y_D, \lambda_1, \lambda_2, \lambda_3, \ldots)

        The output ``normalized points`` are in the camera coordinate system and expressed in normalized coordinates with shape (..., 2).

        .. warning::

            The points are converting to float type before applying the distortion model.

        .. note::

            This indirect method uses non-linear optimization to find the undistorted points.
            The jacobian matrices are not computed in this method.

        .. note::

            For consistency, an alias for this method is ``inverse_transform``.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            Array of distorted points to be transformed with shape (..., 2).

        transpose : bool, optional
            If True, the input points are assume to have shape (2, ...).
            In this case, the output points will have shape (2, ...) as well.
            Default is False.

        kwargs : dict, optional
            Additional keyword arguments to be passed to the distortion model.
        
        Returns
        -------
        undistort_result : UndistortResult
            The result of the undistortion transformation containing the normalized points.
            This object has the following attributes:

            normalized_points : numpy.ndarray
                The transformed normalized points in normalized coordinates. It will be a 2D array of shape (..., 2) if ``transpose`` is False.

        Developer Notes
        ----------------
        The subclasses should implement the following methods:

        - "_undistort": to remove distortion from a set of points.

        The transformation is applied to the ``distorted_points`` (:math:`x_D`) with shape (2, N) to obtain the ``normalized_points`` (:math:`x_N`) with shape (2, N).
        """
        # Check the boolean parameters
        if not isinstance(transpose, bool):
            raise ValueError("The transpose parameter must be a boolean.")
        
        # Check if the distortion parameters are set
        if not self.is_set():
            raise ValueError("The distortion parameters is not set. Please set the distortion parameters before using this method.")
        
        # Create the array of points
        points = numpy.asarray(distorted_points, dtype=numpy.float64)

        # Transpose the points if needed
        if transpose:
            points = numpy.moveaxis(points, 0, -1)

        # Extract the original shape
        shape = points.shape

        # Flatten the points along the last axis
        points_flat = points.reshape(-1, shape[-1])

        # Check the shape of the points
        if points_flat.ndim !=2 or points_flat.shape[1] != 2:
            raise ValueError(f"The points must be in the shape (Npoints, 2) or (2, Npoints) if ``transpose`` is True. Got {points_flat.shape} instead and transpose is {transpose}.")
        
        # Apply the undistortion model to the points
        normalized_points_flat = self._undistort(points_flat, **kwargs) # shape (Npoints, 2)

        # Reshape the normalized points back to the original shape
        normalized_points = normalized_points_flat.reshape(shape) # (Npoints, 2) -> (..., 2)

        # Transpose the points back to the original shape if needed
        if transpose:
            normalized_points = numpy.moveaxis(normalized_points, -1, 0) # (..., 2) -> (2, ...)

        # Return the normalized points
        result = UndistortResult(normalized_points)
        return result
    
    def inverse_transform(self, distorted_points: numpy.ndarray, transpose: bool = False, **kwargs) -> UndistortResult:
        return self.undistort(distorted_points, transpose=transpose, **kwargs)

    @abstractmethod
    def _undistort(self, distorted_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        Abstract method to remove distortion from a set of points.

        The subclasses should implement this method to apply the undistortion model to the given points.

        .. note::

            For ``_undistort`` the input is always in the shape (Npoints, 2) with float64 type.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            Array of distorted points to be transformed with shape (Npoints, 2).

        kwargs : dict, optional
            Additional keyword arguments to be passed to the distortion model.

        Returns
        -------
        normalized_points : numpy.ndarray
            The transformed normalized points in normalized coordinates. It will be a 2D array of shape (Npoints, 2).
        """
        raise NotImplementedError("The _undistort method must be implemented in the subclass.")

        




class NoDistortion(Distortion):
    r"""
    Class to represent the no distortion model.

    This class is used to represent the no distortion model.
    It is a subclass of the Distortion class and implements the methods to apply the no distortion model.

    The no distortion model is used when there is no distortion in the camera.
    """
    def __init__(self):
        super().__init__()

    @property
    def Nparams(self) -> int:
        return 0
    
    @property
    def parameters(self) -> numpy.ndarray:
        return None

    def is_set(self) -> bool:
        return True
    
    def _distort(self, normalized_points: numpy.ndarray, dx: bool = False, dp: bool = False, **kwargs) -> tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        distorted_points = normalized_points.copy() # shape (Npoints, 2)
        jacobian_dx = None # shape (Npoints, 2, 2)
        jacobian_dp = None # shape (Npoints, 2, Nparams)
        if dx:
            jacobian_dx = numpy.zeros((normalized_points.shape[0], 2, 2), dtype=numpy.float64) # shape (Npoints, 2, 2)
            jacobian_dx[:, 0, 0] = 1.0
            jacobian_dx[:, 1, 1] = 1.0
        if dp:
            jacobian_dp = numpy.empty((normalized_points.shape[0], 2, 0), dtype=numpy.float64) # shape (Npoints, 2, 0)
        return distorted_points, jacobian_dx, jacobian_dp
    
    def _undistort(self, distorted_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        normalized_points = distorted_points.copy()
        return normalized_points