from typing import Optional
from numbers import Number
import numpy


class IntrinsicResult(object):
    r"""
    Class to represent the result of the intrinsic transformation.

    This class is used to store the result of the intrinsic transformation and its jacobian matrices.

    .. note::

        ``...`` in the shape of the arrays means that the array can have any number of dimensions.
        Classically, the ``...`` can be replaced by :math:`N` which is the number of points.

    Parameters
    ----------
    image_points : numpy.ndarray
        The transformed image points in pixels. It will be a 2D array of shape (..., 2) if ``transpose`` is False and a 2D array of shape (2, ...) if ``transpose`` is True.

    jacobian_dx : Optional[numpy.ndarray]
        The Jacobian of the image points with respect to the distorted points if ``dx`` is True. Otherwise None.
        It will be a 2D array of shape (..., 2, 2) if ``transpose`` are False and a 2D array of shape (2, ..., 2) if ``transpose`` is True.

    jacobian_dp : Optional[numpy.ndarray]
        The Jacobian of the image points with respect to the intrinsic parameters if ``dp`` is True. Otherwise None.
        It will be a 2D array of shape (..., 2, 4) if ``transpose`` are False and a 2D array of shape (2, ..., 4) if ``transpose`` is True.
    """
    def __init__(self, image_points: numpy.ndarray, jacobian_dx: Optional[numpy.ndarray], jacobian_dp: Optional[numpy.ndarray]):
        self.image_points = image_points
        self.jacobian_dx = jacobian_dx
        self.jacobian_dp = jacobian_dp

    @property
    def jacobian_df(self) -> Optional[numpy.ndarray]:
        r"""
        Get the jacobian of the image points with respect to the focal length.

        Returns
        -------
        Optional[numpy.ndarray]
            The jacobian of the image points with respect to the focal length if ``dp`` is True. Otherwise None.
            It will be a 2D array of shape (..., 2, 2) if ``transpose`` are False and a 2D array of shape (2, ..., 2) if ``transpose`` is True.
        """
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., :, 0:2] # shape (..., 2, 4) -> shape (..., 2, 2)
    
    @property
    def jacobian_dc(self) -> Optional[numpy.ndarray]:
        r"""
        Get the jacobian of the image points with respect to the principal point.

        Returns
        -------
        Optional[numpy.ndarray]
            The jacobian of the image points with respect to the principal point if ``dp`` is True. Otherwise None.
            It will be a 2D array of shape (..., 2, 2) if ``transpose`` are False and a 2D array of shape (2, ..., 2) if ``transpose`` is True.
        """
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., :, 2:4] # shape (..., 2, 4) -> shape (..., 2, 2)


class InverseIntrinsicResult(object):
    r"""
    Class to represent the result of the inverse intrinsic transformation.

    This class is used to store the result of the inverse intrinsic transformation and its jacobian matrices.

    .. note::

        ``...`` in the shape of the arrays means that the array can have any number of dimensions.
        Classically, the ``...`` can be replaced by :math:`N` which is the number of points.

    Parameters
    ----------
    distorted_points : numpy.ndarray
        The transformed distorted points in normalized coordinates. It will be a 2D array of shape (..., 2) if ``transpose`` is False and a 2D array of shape (2, ...) if ``transpose`` is True.

    jacobian_dx : Optional[numpy.ndarray]
        The Jacobian of the distorted points with respect to the image points if ``dx`` is True. Otherwise None.
        It will be a 2D array of shape (..., 2, 2) if ``transpose`` are False and a 2D array of shape (2, ..., 2) if ``transpose`` is True.

    jacobian_dp : Optional[numpy.ndarray]
        The Jacobian of the distorted points with respect to the intrinsic parameters if ``dp`` is True. Otherwise None.
        It will be a 2D array of shape (..., 2, 4) if ``transpose`` are False and a 2D array of shape (2, ..., 4) if ``transpose`` is True.
    """
    def __init__(self, distorted_points: numpy.ndarray, jacobian_dx: Optional[numpy.ndarray], jacobian_dp: Optional[numpy.ndarray]):
        self.distorted_points = distorted_points
        self.jacobian_dx = jacobian_dx
        self.jacobian_dp = jacobian_dp

    @property
    def jacobian_df(self) -> Optional[numpy.ndarray]:
        r"""
        Get the jacobian of the distorted points with respect to the focal length.

        Returns
        -------
        Optional[numpy.ndarray]
            The jacobian of the distorted points with respect to the focal length if ``dp`` is True. Otherwise None.
            It will be a 2D array of shape (..., 2, 2) if ``transpose`` are False and a 2D array of shape (2, ..., 2) if ``transpose`` is True.
        """
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., :, 0:2] # shape (..., 2, 4) -> shape (..., 2, 2)
    
    @property
    def jacobian_dc(self) -> Optional[numpy.ndarray]:
        r"""
        Get the jacobian of the distorted points with respect to the principal point.
        Returns
        -------
        Optional[numpy.ndarray]
            The jacobian of the distorted points with respect to the principal point if ``dp`` is True. Otherwise None.
            It will be a 2D array of shape (..., 2, 2) if ``transpose`` are False and a 2D array of shape (2, ..., 2) if ``transpose`` is True.
        """
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., :, 2:4] # shape (..., 2, 4) -> shape (..., 2, 2)



# =============================================
# Intrinsic class
# =============================================
class Intrinsic(object):
    r"""
    Class to represent the intrinsic parameters of a camera.

    This class defines the interface for intrinsic transformation for cameras.

    In the pinhole camera model, the intrinsic transformation is represented by a set of coefficients :math:`\{f_x, f_y, c_x, c_y}`.
    The process to correspond a 3D-world point to a 2D-image point is as follows:

    1. The ``world_3dpoints`` (:math:`X_W`) are expressed in the camera coordinate system using the rotation and translation matrices to obtain the ``camera_3dpoints`` (:math:`X_C`).
    2. The ``camera_3dpoints`` (:math:`X_C`) are normalized by dividing by the third coordinate to obtain the ``normalized_points`` (:math:`x_N`).
    3. The ``normalized_points`` (:math:`x_N`) are distorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``distorted_points`` (:math:`x_D`).
    4. The ``distorted_points`` (:math:`x_D`) are projected onto the image plane using the intrinsic matrix K to obtain the ``image_points`` (:math:`x_I`).

    .. note::

        This class manage the transformation between the ``distorted points`` and the ``image points``.

    .. math::

        \begin{align*}
        X_C &= R \cdot X_W + T \\
        x_N &= \frac{X_C}{X_C[2]} \\
        x_D &= \text{distort}(x_N, \lambda_1, \lambda_2, \lambda_3, \ldots) \\
        x_I &= K \cdot x_D
        \end{align*}
        
    The intrinsic matrix K is defined as:

    .. math::

        K = \begin{bmatrix}
        f_x & 0 & c_x \\
        0 & f_y & c_y \\
        0 & 0 & 1
        \end{bmatrix}
    
    where :math:`f_x` and :math:`f_y` are the focal lengths in pixels, and :math:`c_x` and :math:`c_y` are the coordinates of the principal point in pixels.

    Parameters
    ----------

    intrinsic_matrix : numpy.ndarray, optional
        The intrinsic matrix of the camera. It should be a 3x3 matrix. Default is None.

    Example
    -------

    Create an intrinsic object with a given intrinsic matrix:

    .. code-block:: python

        import numpy as np
        from pydistort import Intrinsic

        intrinsic_matrix = np.array([[1000, 0, 320],
                                      [0, 1000, 240],
                                      [0, 0, 1]])
        intrinsic = Intrinsic(intrinsic_matrix)

    Then you can use the intrinsic object to transform ``distorted_points`` to ``image_points``:

    .. code-block:: python

        distorted_points = np.array([[100, 200],
                                     [150, 250],
                                     [200, 300]])
        result = intrinsic.transform(distorted_points)
        image_points = result.image_points
        print(image_points)

        result = intrinsic.inverse_transform(image_points)
        distorted_points = result.distorted_points
        print(distorted_points)

    You can also access to the jacobian of the intrinsic transformation:

    .. code-block:: python

        result = intrinsic.transform(distorted_points, dx=True, dp=True)
        image_points_dx = result.jacobian_dx  # Jacobian of the image points with respect to the distorted points
        image_points_dp = result.jacobian_dp  # Jacobian of the image points with respect to the intrinsic parameters
        print(image_points_dx) 

    """
    def __init__(self, intrinsic_matrix: Optional[numpy.ndarray] = None):
        # Initialize the intrinsic parameters
        self._fx = None # focal length in pixels in x direction
        self._fy = None # focal length in pixels in y direction
        self._cx = None # principal point in pixels in x direction
        self._cy = None # principal point in pixels in y direction

        # Set the intrinsic matrix
        self.intrinsic_matrix = intrinsic_matrix

    # =============================================
    # Focal length
    # =============================================
    @property
    def focal_length_x(self) -> Optional[float]:
        r"""
        Get or set the focal length ``fx`` of the intrinsic transformation.

        The focal length is a float representing the focal length of the camera in pixels in x direction.

        This parameter is the component K[0, 0] of the intrinsic matrix K of the camera.

        .. note::

            An alias for ``focal_length_x`` is ``fx``.

        .. seealso::

            - :meth:`pydistort.Intrinsic.focal_length_y` or ``fy`` to set the focal length in pixels in y direction.

        Returns
        -------
        Optional[float]
            The focal length of the camera in pixels in x direction. (or None if not set)
        """
        return self._fx

    @focal_length_x.setter
    def focal_length_x(self, fx: Optional[Number]) -> None:
        if fx is None or numpy.isnan(fx):
            self._fx = None
            return
        if not isinstance(fx, Number):
            raise ValueError("Focal length in pixels in x direction must be a number.")
        if not numpy.isfinite(fx):
            raise ValueError("Focal length in pixels in x direction must be a finite number.")
        if fx <= 0:
            raise ValueError("Focal length in pixels in x direction must be greater than 0.")
        self._fx = float(fx)

    @property
    def fx(self) -> float:
        return self.focal_length_x
    
    @fx.setter
    def fx(self, fx: Optional[Number]) -> None:
        self.focal_length_x = fx


    @property
    def focal_length_y(self) -> Optional[float]:
        r"""
        Get or set the focal length ``fy`` of the intrinsic transformation.

        The focal length is a float representing the focal length of the camera in pixels in y direction.

        This parameter is the component K[1, 1] of the intrinsic matrix K of the camera.

        .. note::

            An alias for ``focal_length_y`` is ``fy``.

        .. seealso::

            - :meth:`pydistort.Intrinsic.focal_length_x` or ``fx`` to set the focal length in pixels in x direction.

        Returns
        -------
        Optional[float]
            The focal length of the camera in pixels in y direction. (or None if not set)
        """
        return self._fy
    
    @focal_length_y.setter
    def focal_length_y(self, fy: Optional[Number]) -> None:
        if fy is None or numpy.isnan(fy):
            self._fy = None
            return
        if not isinstance(fy, Number):
            raise ValueError("Focal length in pixels in y direction must be a number.")
        if not numpy.isfinite(fy):
            raise ValueError("Focal length in pixels in y direction must be a finite number.")
        if fy <= 0:
            raise ValueError("Focal length in pixels in y direction must be greater than 0.")
        self._fy = float(fy)
    
    @property
    def fy(self) -> float:
        return self.focal_length_y

    @fy.setter
    def fy(self, fy: Optional[Number]) -> None:
        self.focal_length_y = fy


    # =============================================
    # Principal point
    # =============================================
    @property
    def principal_point_x(self) -> Optional[float]:
        r"""
        Get or set the principal point ``cx`` of the intrinsic transformation.

        The principal point is a float representing the principal point of the camera in pixels in x direction.

        This parameter is the component K[0, 2] of the intrinsic matrix K of the camera.

        .. note::

            An alias for ``principal_point_x`` is ``cx``.

        .. seealso::

            - :meth:`pydistort.Intrinsic.principal_point_y` or ``cy`` to set the principal point in pixels in y direction.

        Returns
        -------
        Optional[float]
            The principal point of the camera in pixels in x direction. (or None if not set)
        """
        return self._cx
    
    @principal_point_x.setter
    def principal_point_x(self, cx: Optional[Number]) -> None:
        if cx is None or numpy.isnan(cx):
            self._cx = None
            return
        if not isinstance(cx, Number):
            raise ValueError("Principal point in pixels in x direction must be a number.")
        if not numpy.isfinite(cx):
            raise ValueError("Principal point in pixels in x direction must be a finite number.")
        if cx < 0:
            raise ValueError("Principal point in pixels in x direction must be greater than or equal to 0.")
        self._cx = float(cx)

    @property
    def cx(self) -> float:
        return self.principal_point_x
    
    @cx.setter
    def cx(self, cx: Optional[Number]) -> None:
        self.principal_point_x = cx

    @property
    def principal_point_y(self) -> Optional[float]:
        r"""
        Get or set the principal point ``cy`` of the intrinsic transformation.

        The principal point is a float representing the principal point of the camera in pixels in y direction.

        This parameter is the component K[1, 2] of the intrinsic matrix K of the camera.

        .. note::

            An alias for ``principal_point_y`` is ``cy``.

        .. seealso::

            - :meth:`pydistort.Intrinsic.principal_point_x` or ``cx`` to set the principal point in pixels in x direction.

        Returns
        -------
        Optional[float]
            The principal point of the camera in pixels in y direction. (or None if not set)
        """
        return self._cy
    
    @principal_point_y.setter
    def principal_point_y(self, cy: Optional[Number]) -> None:
        if cy is None or numpy.isnan(cy):
            self._cy = None
            return
        if not isinstance(cy, Number):
            raise ValueError("Principal point in pixels in y direction must be a number.")
        if not numpy.isfinite(cy):
            raise ValueError("Principal point in pixels in y direction must be a finite number.")
        if cy < 0:
            raise ValueError("Principal point in pixels in y direction must be greater than or equal to 0.")
        self._cy = float(cy)
    
    @property
    def cy(self) -> float:
        return self.principal_point_y
    
    @cy.setter
    def cy(self, cy: Optional[Number]) -> None:
        self.principal_point_y = cy

    # =============================================
    # Intrinsic matrix
    # =============================================
    @property
    def intrinsic_matrix(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the intrinsic matrix of the intrinsic transformation.

        The intrinsic matrix is a 3x3 matrix representing the intrinsic parameters of the camera.

        .. math::

            K = \begin{bmatrix}
            f_x & 0 & c_x \\
            0 & f_y & c_y \\
            0 & 0 & 1
            \end{bmatrix}

        .. note::

            An alias for ``intrinsic_matrix`` is ``K``.

        .. seealso::

            - :meth:`pydistort.Intrinsic.intrinsic_vector` or ``k`` to get the intrinsic vector of the camera.

        Returns
        -------
        Optional[numpy.ndarray]
            The intrinsic matrix of the camera. (or None if one of the parameters is not set)
        """
        if self._fx is None or self._fy is None or self._cx is None or self._cy is None:
            return None
        return numpy.array([
            [self._fx, 0, self._cx],
            [0, self._fy, self._cy],
            [0, 0, 1]
        ], dtype=numpy.float64)
    
    @intrinsic_matrix.setter
    def intrinsic_matrix(self, intrinsic_matrix: Optional[numpy.ndarray]) -> None:
        if intrinsic_matrix is None:
            self._fx = None
            self._fy = None
            self._cx = None
            self._cy = None
            return
        intrinsic_matrix = numpy.asarray(intrinsic_matrix, dtype=numpy.float64)
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be a 3x3 matrix.")
        # Check if a skew value is given
        if abs(intrinsic_matrix[0, 1]) > 1e-6:
            raise ValueError("Skew value is not supported by pydistort.")
        if abs(intrinsic_matrix[1, 0]) > 1e-6 or abs(intrinsic_matrix[2, 0]) > 1e-6 or abs(intrinsic_matrix[2, 1]) > 1e-6:
            raise ValueError("Some coefficients of the intrinsic matrix are unexpected.")
        # Set the intrinsic parameters
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]

    @property
    def K(self) -> Optional[numpy.ndarray]:
        return self.intrinsic_matrix
    
    @K.setter
    def K(self, intrinsic_matrix: Optional[numpy.ndarray]) -> None:
        self.intrinsic_matrix = intrinsic_matrix

    # =============================================
    # Intrinsic vector
    # =============================================
    @property
    def intrinsic_vector(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the intrinsic vector of the intrinsic transformation.

        The intrinsic vector is a 4x1 vector representing the intrinsic parameters of the camera.

        .. math::

            \begin{bmatrix}
            f_x \\
            f_y \\
            c_x \\
            c_y
            \end{bmatrix}

        .. note::

            An alias for ``intrinsic_vector`` is ``k``.

        .. seealso::

            - :meth:`pydistort.Intrinsic.intrinsic_matrix` or ``K`` to set the intrinsic matrix of the camera.

        Returns
        -------
        Optional[numpy.ndarray]
            The intrinsic vector of the camera. (or None if one of the parameters is not set)
        """
        if self._fx is None or self._fy is None or self._cx is None or self._cy is None:
            return None
        return numpy.array([self._fx, self._fy, self._cx, self._cy], dtype=numpy.float64)
    
    @intrinsic_vector.setter
    def intrinsic_vector(self, intrinsic_vector: Optional[numpy.ndarray]) -> None:
        if intrinsic_vector is None:
            self._fx = None
            self._fy = None
            self._cx = None
            self._cy = None
            return
        intrinsic_vector = numpy.asarray(intrinsic_vector, dtype=numpy.float64).flatten()
        if intrinsic_vector.size != 4:
            raise ValueError("Intrinsic vector must be a 4x1 vector.")
        # Set the intrinsic parameters
        self.fx = intrinsic_vector[0]
        self.fy = intrinsic_vector[1]
        self.cx = intrinsic_vector[2]
        self.cy = intrinsic_vector[3]

    @property
    def k(self) -> Optional[numpy.ndarray]:
        return self.intrinsic_vector

    @k.setter
    def k(self, intrinsic_vector: Optional[numpy.ndarray]) -> None:
        self.intrinsic_vector = intrinsic_vector

    # =============================================
    # Display intrinsic matrix
    # =============================================
    def __repr__(self) -> str:
        r"""
        Get a string representation of the intrinsic matrix.

        Returns
        -------
        str
            The string representation of the intrinsic matrix.
        """
        return f"Intrinsic matrix: fx={self._fx}, fy={self._fy}, cx={self._cx}, cy={self._cy}"

    # =============================================
    # Check if the intrinsic matrix is set
    # =============================================
    def is_set(self) -> bool:
        r"""
        Check if the intrinsic matrix is set.

        The intrinsic matrix is set if all the parameters are set.

        Returns
        -------
        bool
            True if the intrinsic matrix is set, False otherwise.
        """
        return self._fx is not None and self._fy is not None and self._cx is not None and self._cy is not None

    # =============================================
    # Transformations
    # =============================================
    def _transform(self, distorted_points: numpy.ndarray, dx: bool = False, dp: bool = False) -> tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        This methos allow to bypass the check on the input points made in the transform method.

        .. note::

            For ``_transform`` the input is always in the shape (Npoints, 2) with float64 type.
            The output must be (Npoints, 2) for the image points and (Npoints, 2, 2) for the jacobian with respect to the distorted points and (Npoints, 2, 4) for the jacobian with respect to the intrinsic parameters.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            Array of distorted points to be transformed with shape (Npoints, 2).

        dx : bool, optional
            If True, the Jacobian of the image points with respect to the distorted points is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 2, 2).

        dp : bool, optional
            If True, the Jacobian of the image points with respect to the intrinsic parameters is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 2, 4).

        kwargs : dict, optional
            Additional keyword arguments to be passed to the distortion model.

        Returns
        -------
        image_points : numpy.ndarray
            The transformed image points in pixels. It will be a 2D array of shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The Jacobian of the image points with respect to the distorted points if ``dx`` is True. Otherwise None.
            It will be a 2D array of shape (Npoints, 2, 2).

        jacobian_dp : Optional[numpy.ndarray]
            The Jacobian of the image points with respect to the intrinsic parameters if ``dp`` is True. Otherwise None.
            It will be a 2D array of shape (Npoints, 2, 4).
        """
        # Extract the useful coordinates
        x_D = distorted_points[:, 0] # shape (Npoints,)
        y_D = distorted_points[:, 1] # shape (Npoints,)

        # Compute the image points
        x_I = self._fx * x_D + self._cx # shape (Npoints,)
        y_I = self._fy * y_D + self._cy # shape (Npoints,)

        image_points_flat = numpy.empty(distorted_points.shape) # shape (Npoints, 2)
        image_points_flat[:, 0] = x_I # shape (Npoints,)
        image_points_flat[:, 1] = y_I # shape (Npoints,)
 
        # Compute the jacobian with respect to the distorted points
        if dx:
            jacobian_flat_dx = numpy.empty((*distorted_points.shape, 2), dtype=numpy.float64) # shape (Npoints, 2, 2)
            jacobian_flat_dx[:, 0, 0] = self._fx # shape (Npoints,)
            jacobian_flat_dx[:, 0, 1] = 0.0 # shape (Npoints,)
            jacobian_flat_dx[:, 1, 0] = 0.0 # shape (Npoints,)
            jacobian_flat_dx[:, 1, 1] = self._fy # shape (Npoints,)
        else:
            jacobian_flat_dx = None

        # Compute the jacobian with respect to the intrinsic parameters
        if dp:
            jacobian_flat_dp = numpy.empty((*distorted_points.shape, 4), dtype=numpy.float64) # shape (Npoints, 2, 4)
            jacobian_flat_dp[:, 0, 0] = x_D # shape (Npoints,)
            jacobian_flat_dp[:, 0, 1] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 0, 2] = 1.0 # shape (Npoints,)
            jacobian_flat_dp[:, 0, 3] = 0.0 # shape (Npoints,)

            jacobian_flat_dp[:, 1, 0] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 1, 1] = y_D # shape (Npoints,)
            jacobian_flat_dp[:, 1, 2] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 1, 3] = 1.0 # shape (Npoints,)
        else:
            jacobian_flat_dp = None

        return image_points_flat, jacobian_flat_dx, jacobian_flat_dp

        
    def transform(self, distorted_points: numpy.ndarray, transpose: bool = False, dx: bool = False, dp: bool = False) -> IntrinsicResult:
        r"""
        Transform the given ``distorted points`` to ``image points`` using the intrinsic transformation.

        The given points ``distorted points`` are assumed to be in the camera coordinate system and expressed in normalized coordinates with shape (..., 2).
        
        .. note::

            ``...`` in the shape of the arrays means that the array can have any number of dimensions.
            Classically, the ``...`` can be replaced by :math:`N` which is the number of points.

        The equations used to transform the points are:

        .. math::

            x_I = f_x \cdot x_D + c_x \\
        
        .. math::

            y_I = f_y \cdot y_D + c_y

        The output points ``image points`` are in pixels and expressed in the image coordinate system with shape (..., 2).

        .. warning::

            The points are converting to float type before applying the intrinsic matrix.

        The method also computes 2 Jacobian matrices if requested:

        - ``dx``: Jacobian of the image points with respect to the distorted points.
        - ``dp``: Jacobian of the image points with respect to the intrinsic parameters.

        The jacobian matrice with respect to the distorted points is a (..., 2 , 2) matrix where :

        .. code-block:: python

            jacobian_dx[..., 0, 0]  # ∂x_I/∂x_D -> Jacobian of the coordinates x_I with respect to the coordinates x_D.
            jacobian_dx[..., 0, 1]  # ∂x_I/∂y_D

            jacobian_dx[..., 1, 0]  # ∂y_I/∂x_D -> Jacobian of the coordinates y_I with respect to the coordinates x_D.
            jacobian_dx[..., 1, 1]  # ∂y_I/∂y_D

        The Jacobian matrice with respect to the intrinsic parameters is a (..., 2, 4) matrix where :

        .. code-block:: python

            jacobian_dp[..., 0, 0]  # ∂x_I/∂fx -> Jacobian of the coordinates x_I with respect to the focal length in pixels in x direction (fx).
            jacobian_dp[..., 0, 1]  # ∂x_I/∂fy
            jacobian_dp[..., 0, 2]  # ∂x_I/∂cx
            jacobian_dp[..., 0, 3]  # ∂x_I/∂cy

            jacobian_dp[..., 1, 0]  # ∂y_I/∂fx -> Jacobian of the coordinates y_I with respect to the focal length in pixels in x direction (fx).
            jacobian_dp[..., 1, 1]  # ∂y_I/∂fy
            jacobian_dp[..., 1, 2]  # ∂y_I/∂cx
            jacobian_dp[..., 1, 3]  # ∂y_I/∂cy

            
        Parameters
        ----------
        distorted_points : numpy.ndarray
            Array of normalized points to be transformed with shape (..., 2).

        transpose : bool, optional
            If True, the input points are assume to have shape (2, ...).
            In this case, the output points will have shape (2, ...) as well and the jacobian matrices will have shape (2, ..., 2) and (2, ..., 4) respectively.
            Default is False.

        dx : bool, optional
            If True, the Jacobian of the image points with respect to the distorted points is computed. Default is False.
            The output will be a 2D array of shape (..., 2, 2) if ``transpose`` is False.
            If ``dx`` is False, the output will be None.

        dp : bool, optional
            If True, the Jacobian of the image points with respect to the intrinsic parameters is computed. Default is False.
            The output will be a 2D array of shape (..., 2, 4) if ``transpose`` is False.
            If ``dp`` is False, the output will be None.

        Returns
        -------
        intrinsic_result : IntrinsicResult

            The result of the intrinsic transformation containing the image points and the jacobian matrices.
            This object has the following attributes:

            image_points : numpy.ndarray
                The transformed image points in pixels. It will be a 2D array of shape (..., 2) if ``transpose`` is False.

            jacobian_dx : Optional[numpy.ndarray]
                The Jacobian of the image points with respect to the distorted points if ``dx`` is True. Otherwise None.
                It will be a 2D array of shape (..., 2, 2) if ``transpose`` is False.

            jacobian_dp : Optional[numpy.ndarray]
                The Jacobian of the image points with respect to the intrinsic parameters if ``dp`` is True. Otherwise None.
                It will be a 2D array of shape (..., 2, 4) if ``transpose`` is False.

        Example
        -------

        Create an intrinsic object with a given intrinsic matrix:

        .. code-block:: python

            import numpy as np
            from pydistort import Intrinsic

            intrinsic_matrix = np.array([[1000, 0, 320],
                                         [0, 1000, 240],
                                         [0, 0, 1]])
            intrinsic = Intrinsic(intrinsic_matrix)

        Then you can use the intrinsic object to transform ``distorted_points`` to ``image_points``:

        .. code-block:: python

            distorted_points = np.array([[100, 200],
                                         [150, 250],
                                         [200, 300]]) # shape (3, 2)
            result = intrinsic.transform(distorted_points, dx=True, dp=True)

            result.image_points # shape (3, 2) -> image points in pixels
            result.jacobian_dx # shape (3, 2, 2) -> jacobian of the image points with respect to the distorted points
            result.jacobian_dp # shape (3, 2, 4) -> jacobian of the image points with respect to the intrinsic parameters
            result.jacobian_df # shape (3, 2, 2) -> jacobian of the image points with respect to the focal length (extracted from jacobian_dp)
            result.jacobian_dc # shape (3, 2, 2) -> jacobian of the image points with respect to the principal point (extracted from jacobian_dp)
                                        
        """
        # Check the boolean parameters
        if not isinstance(transpose, bool):
            raise ValueError("The transpose parameter must be a boolean.")
        if not isinstance(dx, bool):
            raise ValueError("The dx parameter must be a boolean.")
        if not isinstance(dp, bool):
            raise ValueError("The dp parameter must be a boolean.")
        
        # Check if the intrinsic matrix is set
        if not self.is_set():
            raise ValueError("The intrinsic matrix is not set. Please set the intrinsic matrix before using this method.")
        
        # Create the array of points
        points = numpy.asarray(distorted_points, dtype=numpy.float64) 

        # Transpose the points if needed
        if transpose:
            points = numpy.moveaxis(points, 0, -1) # (2, ...) -> (..., 2)

        # Extract the original shape
        shape = points.shape # (..., 2)

        # Flatten the points along the last axis
        points_flat = points.reshape(-1, shape[-1]) # shape (..., 2) -> shape (Npoints, 2)
        shape_flat = points_flat.shape

        # Check the shape of the points
        if points_flat.ndim !=2 or points_flat.shape[1] != 2:
            raise ValueError(f"The points must be in the shape (Npoints, 2) or (2, Npoints) if ``transpose`` is True. Got {points_flat.shape} instead and transpose is {transpose}.")
        
        image_points_flat, jacobian_flat_dx, jacobian_flat_dp = self._transform(points_flat, dx=dx, dp=dp) # shape (Npoints, 2), shape (Npoints, 2, 2), shape (Npoints, 2, 4)

        # Reshape the image points back to the original shape
        image_points = image_points_flat.reshape(shape) # (Npoints, 2) -> (..., 2)
        jacobian_dx = jacobian_flat_dx.reshape((*shape, 2)) if dx else None # (Npoints, 2, 2) -> (..., 2, 2)
        jacobian_dp = jacobian_flat_dp.reshape((*shape, 4)) if dp else None # (Npoints, 2, 4) -> (..., 2, 4)

        # Transpose the points back to the original shape if needed
        if transpose:
            image_points = numpy.moveaxis(image_points, -1, 0) # (..., 2) -> (2, ...)
            jacobian_dx = numpy.moveaxis(jacobian_dx, -2, 0) if dx else None # (..., 2, 2) -> (2, ..., 2)
            jacobian_dp = numpy.moveaxis(jacobian_dp, -2, 0) if dp else None # (..., 2, 4) -> (2, ..., 4)

        # Return the image points and the jacobian matrices
        result = IntrinsicResult(image_points, jacobian_dx, jacobian_dp)
        return result
    
    
    def _inverse_transform(self, image_points: numpy.ndarray, dx: bool = False, dp: bool = False) -> tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        This methos allow to bypass the check on the input points made in the inverse_transform method.

        .. note::

            For ``_inverse_transform`` the input is always in the shape (Npoints, 2) with float64 type.
            The output must be (Npoints, 2) for the distorted points and (Npoints, 2, 2) for the jacobian with respect to the image points and (Npoints, 2, 4) for the jacobian with respect to the intrinsic parameters.

        Parameters
        ----------
        image_points : numpy.ndarray
            Array of image points to be transformed with shape (Npoints, 2).

        dx : bool, optional
            If True, the Jacobian of the distorted points with respect to the image points is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 2, 2).

        dp : bool, optional
            If True, the Jacobian of the distorted points with respect to the intrinsic parameters is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 2, 4).

        Returns
        -------
        distorted_points : numpy.ndarray
            The transformed distorted points in normalized coordinates. It will be a 2D array of shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The Jacobian of the distorted points with respect to the image points if ``dx`` is True. Otherwise None.
            It will be a 2D array of shape (Npoints, 2, 2).

        jacobian_dp : Optional[numpy.ndarray]
            The Jacobian of the distorted points with respect to the intrinsic parameters if ``dp`` is True. Otherwise None.
            It will be a 2D array of shape (Npoints, 2, 4).
        """
         # Extract the useful coordinates
        x_I = image_points[:, 0] # shape (Npoints,)
        y_I = image_points[:, 1] # shape (Npoints,)

        # Compute the distorted points
        x_D = (x_I - self._cx) / self._fx # shape (Npoints,)
        y_D = (y_I - self._cy) / self._fy # shape (Npoints,)

        distorted_points_flat = numpy.empty(image_points.shape) # shape (Npoints, 2)
        distorted_points_flat[:, 0] = x_D # shape (Npoints,)
        distorted_points_flat[:, 1] = y_D # shape (Npoints,)

        # Compute the jacobian with respect to the image points
        if dx:
            jacobian_flat_dx = numpy.empty((*image_points.shape, 2), dtype=numpy.float64) # shape (Npoints, 2, 2)
            jacobian_flat_dx[:, 0, 0] = 1.0 / self._fx # shape (Npoints,)
            jacobian_flat_dx[:, 0, 1] = 0.0 # shape (Npoints,)
            jacobian_flat_dx[:, 1, 0] = 0.0 # shape (Npoints,)
            jacobian_flat_dx[:, 1, 1] = 1.0 / self._fy # shape (Npoints,)
        else:
            jacobian_flat_dx = None

        # Compute the jacobian with respect to the intrinsic parameters
        if dp:
            jacobian_flat_dp = numpy.empty((*image_points.shape, 4), dtype=numpy.float64) # shape (Npoints, 2, 4)
            jacobian_flat_dp[:, 0, 0] = -x_I / self._fx # shape (Npoints,)
            jacobian_flat_dp[:, 0, 1] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 0, 2] = -1.0 / self._fx # shape (Npoints,)
            jacobian_flat_dp[:, 0, 3] = 0.0 # shape (Npoints,)

            jacobian_flat_dp[:, 1, 0] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 1, 1] = -y_I / self._fy # shape (Npoints,)
            jacobian_flat_dp[:, 1, 2] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 1, 3] = -1.0 / self._fy # shape (Npoints,)
        else:
            jacobian_flat_dp = None

        return distorted_points_flat, jacobian_flat_dx, jacobian_flat_dp


    def inverse_transform(self, image_points: numpy.ndarray, transpose: bool = False, dx: bool = False, dp: bool = False) -> InverseIntrinsicResult:
        r"""
        Transform the given ``image points`` to ``distorted points`` using the inverse intrinsic transformation.

        The given points ``image points`` are assumed to be in the image coordinate system and expressed in pixels with shape (..., 2).
        
        .. note::

            ``...`` in the shape of the arrays means that the array can have any number of dimensions.
            Classically, the ``...`` can be replaced by :math:`N` which is the number of points.

        The equations used to transform the points are:

        .. math::

            x_D = \frac{x_I - c_x}{f_x} \\
        
        .. math::

            y_D = \frac{y_I - c_y}{f_y}

        The output points ``distorted points`` are in normalized coordinates and expressed in the camera coordinate system with shape (..., 2).

        .. warning::

            The points are converting to float type before applying the intrinsic matrix.

        The method also computes 2 Jacobian matrices if requested:

        - ``dx``: Jacobian of the distorted points with respect to the image points.
        - ``dp``: Jacobian of the distorted points with respect to the intrinsic parameters.

        The jacobian matrice with respect to the image points is a (..., 2, 2) matrix where :

        .. code-block:: python

            jacobian_dx[..., 0, 0]  # ∂x_D/∂x_I -> Jacobian of the coordinates x_D with respect to the coordinates x_I.
            jacobian_dx[..., 0, 1]  # ∂x_D/∂y_I

            jacobian_dx[..., 1, 0]  # ∂y_D/∂x_I -> Jacobian of the coordinates y_D with respect to the coordinates x_I.
            jacobian_dx[..., 1, 1]  # ∂y_D/∂y_I

        The Jacobian matrice with respect to the intrinsic parameters is a (..., 2, 4) matrix where :

        .. code-block:: python

            jacobian_dp[..., 0, 0]  # ∂x_D/∂fx -> Jacobian of the coordinates x_D with respect to the focal length in pixels in x direction (fx).
            jacobian_dp[..., 0, 1]  # ∂x_D/∂fy
            jacobian_dp[..., 0, 2]  # ∂x_D/∂cx
            jacobian_dp[..., 0, 3]  # ∂x_D/∂cy

            jacobian_dp[..., 1, 0]  # ∂y_D/∂fx -> Jacobian of the coordinates y_D with respect to the focal length in pixels in x direction (fx).
            jacobian_dp[..., 1, 1]  # ∂y_D/∂fy
            jacobian_dp[..., 1, 2]  # ∂y_D/∂cx
            jacobian_dp[..., 1, 3]  # ∂y_D/∂cy

        Parameters
        ----------
        image_points : numpy.ndarray
            Array of image points to be transformed with shape (..., 2).

        transpose : bool, optional
            If True, the input points are assume to have shape (2, ...).
            In this case, the output points will have shape (2, ...) as well and the jacobian matrices will have shape (2, ..., 2) and (2, ..., 4) respectively.
            Default is False.

        dx : bool, optional
            If True, the Jacobian of the distorted points with respect to the image points is computed. Default is False.
            The output will be a 2D array of shape (..., 2, 2) if ``transpose`` is False.
            If ``dx`` is False, the output will be None.
        
        dp : bool, optional
            If True, the Jacobian of the distorted points with respect to the intrinsic parameters is computed. Default is False.
            The output will be a 2D array of shape (..., 2, 4) if ``transpose`` is False.
            If ``dp`` is False, the output will be None.

        Returns
        -------
        inverse_intrinsic_result : InverseIntrinsicResult

            The result of the inverse intrinsic transformation containing the distorted points and the jacobian matrices.
            This object has the following attributes:

            distorted_points : numpy.ndarray
                The transformed distorted points in normalized coordinates. It will be a 2D array of shape (..., 2) if ``transpose`` is False.

            jacobian_dx : Optional[numpy.ndarray]
                The Jacobian of the distorted points with respect to the image points if ``dx`` is True. Otherwise None.
                It will be a 2D array of shape (..., 2, 2) if ``transpose`` are False.

            jacobian_dp : Optional[numpy.ndarray]
                The Jacobian of the distorted points with respect to the intrinsic parameters if ``dp`` is True. Otherwise None.
                It will be a 2D array of shape (..., 2, 4) if ``transpose`` are False.

        Example
        -------

        Create an intrinsic object with a given intrinsic matrix:

        .. code-block:: python

            import numpy as np
            from pydistort import Intrinsic

            intrinsic_matrix = np.array([[1000, 0, 320],
                                         [0, 1000, 240],
                                         [0, 0, 1]])
            intrinsic = Intrinsic(intrinsic_matrix)

        Then you can use the intrinsic object to transform ``image_points`` to ``distorted_points``:

        .. code-block:: python

            image_points = np.array([[100, 200],
                                     [150, 250],
                                     [200, 300]]) # shape (3, 2)
            result = intrinsic.inverse_transform(image_points, dx=True, dp=True)

            result.distorted_points # shape (3, 2) -> distorted points in normalized coordinates
            result.jacobian_dx # shape (3, 2, 2) -> jacobian of the distorted points with respect to the image points
            result.jacobian_dp # shape (3, 2, 4) -> jacobian of the distorted points with respect to the intrinsic parameters
            result.jacobian_df # shape (3, 2, 2) -> jacobian of the distorted points with respect to the focal length (extracted from jacobian_dp)
            result.jacobian_dc # shape (3, 2, 2) -> jacobian of the distorted points with respect to the principal point (extracted from jacobian_dp)
        """
        # Check the boolean parameters
        if not isinstance(transpose, bool):
            raise ValueError("The transpose parameter must be a boolean.")
        if not isinstance(dx, bool):
            raise ValueError("The dx parameter must be a boolean.")
        if not isinstance(dp, bool):
            raise ValueError("The dp parameter must be a boolean.")
    
        # Check if the intrinsic matrix is set
        if not self.is_set():
            raise ValueError("The intrinsic matrix is not set. Please set the intrinsic matrix before using this method.")
        
        # Create the array of points
        points = numpy.asarray(image_points, dtype=numpy.float64)

        # Transpose the points if needed
        if transpose:
            points = numpy.moveaxis(points, 0, -1) # (2, ...) -> (..., 2)

        # Extract the original shape
        shape = points.shape # (..., 2)

        # Flatten the points along the last axis
        points_flat = points.reshape(-1, shape[-1]) # shape (..., 2) -> shape (Npoints, 2)
        shape_flat = points_flat.shape

        # Check the shape of the points
        if points_flat.ndim !=2 or points_flat.shape[1] != 2:
            raise ValueError(f"The points must be in the shape (Npoints, 2) or (2, Npoints) if ``transpose`` is True. Got {points_flat.shape} instead and transpose is {transpose}.")
        
        distorted_points_flat, jacobian_flat_dx, jacobian_flat_dp = self._inverse_transform(points_flat, dx=dx, dp=dp) # shape (Npoints, 2), shape (Npoints, 2, 2), shape (Npoints, 2, 4)

        # Reshape the distorted points back to the original shape
        distorted_points = distorted_points_flat.reshape(shape) # (Npoints, 2) -> (..., 2)
        jacobian_dx = jacobian_flat_dx.reshape((*shape, 2)) if dx else None # (Npoints, 2, 2) -> (..., 2, 2)
        jacobian_dp = jacobian_flat_dp.reshape((*shape, 4)) if dp else None # (Npoints, 2, 4) -> (..., 2, 4)

        # Transpose the points back to the original shape if needed
        if transpose:
            distorted_points = numpy.moveaxis(distorted_points, -1, 0) # (..., 2) -> (2, ...)
            jacobian_dx = numpy.moveaxis(jacobian_dx, -2, 0) if dx else None # (..., 2, 2) -> (2, ..., 2)
            jacobian_dp = numpy.moveaxis(jacobian_dp, -2, 0) if dp else None # (..., 2, 4) -> (2, ..., 4)

        # Return the distorted points and the jacobian matrices
        result = InverseIntrinsicResult(distorted_points, jacobian_dx, jacobian_dp)
        return result


        