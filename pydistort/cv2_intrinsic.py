from typing import Optional, Tuple, Dict
from numbers import Number
import numpy

from .core import Intrinsic




class Cv2Intrinsic(Intrinsic):
    r"""
    Subclass of :class:`pydistort.core.Intrinsic` to represent the intrinsic transformation using OpenCV conventions.

    The equation used for the intrinsic transformation is:    

    .. math::

        \begin{align*}
        x_I &= K \cdot x_D \\
        \end{align*}

    where :math:`x_D` is the distorted points, :math:`x_I` is the image points, and :math:`K` is the intrinsic matrix defined as:

    .. math::

        K = \begin{bmatrix}
        f_x & 0 & c_x \\
        0 & f_y & c_y \\
        0 & 0 & 1
        \end{bmatrix}

    where :math:`f_x` and :math:`f_y` are the focal lengths in pixels in x and y direction, :math:`c_x` and :math:`c_y` are the principal point coordinates in pixels.

    .. note::

        If no distortion is applied, the ``distorted_points`` are equal to the ``normalized_points``.

    .. warning::

        No ``skew`` parameter is included in the intrinsic matrix in this implementation. If you need to include a skew parameter, you can use :class:`pydistort.SkewIntrinsic`.

    Two short-hand notations are provided to access the jacobian with respect to focal length and principal point:

    - ``jacobian_df``: The Jacobian of the image points with respect to the focal length parameters. It has shape (..., 2, 2), where the last dimension represents (df_x, df_y).
    - ``jacobian_dc``: The Jacobian of the image points with respect to the principal point parameters. It has shape (..., 2, 2), where the last dimension represents (dc_x, dc_y).

    Parameters
    ----------
    intrinsic_matrix : Optional[numpy.ndarray], optional
        The intrinsic matrix of the camera. It should be a 3x3 matrix. Default is None.

    Example
    -------
    Create an intrinsic object with a given intrinsic matrix:

    .. code-block:: python

        import numpy as np
        from pydistort import Cv2Intrinsic

        intrinsic_matrix = np.array([[1000, 0, 320],
                                     [0, 1000, 240],
                                     [0, 0, 1]])
        intrinsic = Cv2Intrinsic(intrinsic_matrix)

    Then you can use the intrinsic object to transform ``distorted_points`` to ``image_points``:

    .. code-block:: python

        distorted_points = np.array([[100, 200],
                                     [150, 250],
                                     [200, 300]])
        result = intrinsic.transform(distorted_points)
        image_points = result.image_points
        print(image_points)

    You can also access to the jacobian of the intrinsic transformation:

    .. code-block:: python

        result = intrinsic.transform(distorted_points, dx=True, dp=True)
        image_points_dx = result.jacobian_dx  # Jacobian of the image points with respect to the distorted points
        image_points_dp = result.jacobian_dp  # Jacobian of the image points with respect to the intrinsic parameters
        print(image_points_dx)

    The inverse transformation can be computed using the `inverse_transform` method:

    .. code-block:: python

        inverse_result = intrinsic.inverse_transform(image_points, dx=True, dp=True)
        distorted_points = inverse_result.transformed_points  # Shape (..., 2)
        print(distorted_points)
    
    .. seealso::

        For more information about the transformation process, see:

        - :meth:`pydistort.Cv2Intrinsic._transform` to transform the ``distorted_points`` to ``image_points``.
        - :meth:`pydistort.Cv2Intrinsic._inverse_transform` to transform the ``image_points`` back to ``distorted_points``.
    
    """
    def __init__(self, intrinsic_matrix: Optional[numpy.ndarray] = None):
        # Initialize the Transform base class
        super().__init__()

        # Initialize the intrinsic parameters
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None

        # Set the intrinsic matrix
        self.intrinsic_matrix = intrinsic_matrix

    # =============================================
    # Properties for ABC Transform Class
    # =============================================
    @property
    def Nparams(self) -> int:
        return 4  # The intrinsic parameters are (fx, fy, cx, cy) even if some of them are not set.
    
    @property
    def _jacobian_short_hand(self) -> Dict[str, Tuple[int, int, Optional[str]]]:
        r"""
        Short-hand notation for the Jacobian matrices with respect to the intrinsic parameters.

        - ``df``: The Jacobian of the normalized points with respect to the focal length parameters. It has shape (..., 2, 2), where the last dimension represents (df_x, df_y).
        - ``dc``: The Jacobian of the normalized points with respect to the principal point parameters. It has shape (..., 2, 2), where the last dimension represents (dc_x, dc_y).

        Returns
        -------
        Dict[str, Tuple[int, int, Optional[str]]]
            A dictionary where keys are names of the custom Jacobian views and values are tuples containing:

            - start index (int): The starting index of the parameters to include in the custom Jacobian view.
            - end index (int): The ending index of the parameters to include in the custom Jacobian view.
            - doc (Optional[str]): A documentation string for the custom Jacobian view.
        """
        return {
            "df": (0, 2, "Jacobian of the image points with respect to the focal length parameters (fx, fy)"),
            "dc": (2, 4, "Jacobian of the image points with respect to the principal point parameters (cx, cy)")
        }

    @property
    def parameters(self) -> Optional[numpy.ndarray]:
        r"""
        Get the intrinsic parameters as a vector.

        .. seealso::

            - :meth:`pydistort.Cv2Intrinsic.intrinsic_vector` or ``k`` to get the intrinsic vector of the camera.

        Returns
        -------
        Optional[numpy.ndarray]
            The intrinsic parameters as a vector of shape (4,). If any of the parameters are not set, returns None.
        """
        return self.intrinsic_vector
    
    @parameters.setter
    def parameters(self, intrinsic_vector: Optional[numpy.ndarray]) -> None:
        self.intrinsic_vector = intrinsic_vector
    
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

            - :meth:`pydistort.Cv2Intrinsic.focal_length_y` or ``fy`` to set the focal length in pixels in y direction.

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

            - :meth:`pydistort.Cv2Intrinsic.focal_length_x` or ``fx`` to set the focal length in pixels in x direction.

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

            - :meth:`pydistort.Cv2Intrinsic.principal_point_y` or ``cy`` to set the principal point in pixels in y direction.

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

            - :meth:`pydistort.Cv2Intrinsic.principal_point_x` or ``cx`` to set the principal point in pixels in x direction.

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

            - :meth:`pydistort.Cv2Intrinsic.intrinsic_vector` or ``k`` to get the intrinsic vector of the camera.

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
            raise ValueError("Skew value is not supported by Cv2Intrinsic. Use SkewIntrinsic instead.")
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

            - :meth:`pydistort.Cv2Intrinsic.intrinsic_matrix` or ``K`` to set the intrinsic matrix of the camera.

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
    # Methods for ABC Transform Class
    # =============================================
    def is_set(self) -> bool:
        r"""
        Check if the intrinsic parameters are set.

        Returns
        -------
        bool
            True if all intrinsic parameters are set, False otherwise.
        """
        return self._fx is not None and self._fy is not None and self._cx is not None and self._cy is not None
    

    def _transform(
            self, 
            distorted_points: numpy.ndarray, 
            *, 
            dx: bool = False, 
            dp: bool = False
        ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        This method is called by the :meth:`pydistort.core.Transform.transform` method to perform the intrinsic transformation.
        This method allows to transform the ``distorted_points`` to ``image_points`` using the intrinsic parameters.

        .. note::

            For ``_transform`` the input must have shape (Npoints, 2) with float64 type.
            The output has shape (Npoints, 2) for the image points and (Npoints, 2, 2) for the jacobian with respect to the distorted points and (Npoints, 2, 4) for the jacobian with respect to the intrinsic parameters.

        The equation used for the transformation is:

        .. math::

            x_I = f_x \cdot x_D + c_x 

        .. math::

            y_I = f_y \cdot y_D + c_y

        where :math:`f_x` and :math:`f_y` are the focal lengths in pixels, and :math:`c_x` and :math:`c_y` are the coordinates of the principal point in pixels.

        .. warning::

            This method is not designed to be used directly for the transformation of points.
            No checks are performed on the input points, so it is the user's responsibility to ensure that the input points are valid.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted points to be transformed. Shape (Npoints, 2).

        dx : bool, optional
            If True, compute the Jacobian of the image points with respect to the distorted points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the image points with respect to the intrinsic parameters. Default is False.

        Returns
        -------
        image_points : numpy.ndarray
            The transformed image points in the camera coordinate system. Shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The Jacobian of the image points with respect to the distorted points if ``dx`` is True. Otherwise None. Shape (Npoints, 2, 2), where the last dimension represents (dx, dy).

        jacobian_dp : Optional[numpy.ndarray]
            The Jacobian of the image points with respect to the intrinsic parameters if ``dp`` is True. Otherwise None. Shape (Npoints, 2, 4), where the last dimension represents (dfx, dfy, dcx, dcy).
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
    

    def _inverse_transform(self, image_points: numpy.ndarray, *, dx: bool = False, dp: bool = False) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        This method is called by the :meth:`pydistort.core.Transform.inverse_transform` method to perform the inverse intrinsic transformation.
        This method allows to transform the ``image_points`` back to ``distorted_points`` using the intrinsic parameters.

        .. note::

            For ``_inverse_transform`` the input must have shape (Npoints, 2) with float64 type.
            The output has shape (Npoints, 2) for the distorted points and (Npoints, 2, 2)

        The equation used for the inverse transformation is:

        .. math::

            x_D = \frac{x_I - c_x}{f_x}

        .. math::

            y_D = \frac{y_I - c_y}{f_y}

        where :math:`f_x` and :math:`f_y` are the focal lengths in pixels, and :math:`c_x` and :math:`c_y` are the coordinates of the principal point in pixels.

        .. warning::

            This method is not designed to be used directly for the transformation of points.
            No checks are performed on the input points, so it is the user's responsibility to ensure that the input points are valid.

        Parameters
        ----------
        image_points : numpy.ndarray
            The image points to be transformed. Shape (Npoints, 2).

        dx : bool, optional
            If True, compute the Jacobian of the distorted points with respect to the image points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the distorted points with respect to the intrinsic parameters. Default is False.

        Returns
        -------
        distorted_points : numpy.ndarray
            The transformed distorted points in the camera coordinate system. Shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The Jacobian of the distorted points with respect to the image points if ``dx`` is True. Otherwise None. Shape (Npoints, 2, 2), where the last dimension represents (dx, dy).
        
        jacobian_dp : Optional[numpy.ndarray]
            The Jacobian of the distorted points with respect to the intrinsic parameters if ``dp`` is True. Otherwise None. Shape (Npoints, 2, 4), where the last dimension represents (dfx, dfy, dcx, dcy).
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
            jacobian_flat_dp[:, 0, 0] = - x_D / self._fx # shape (Npoints,) because x_D = (x_I - c_x) / f_x
            jacobian_flat_dp[:, 0, 1] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 0, 2] = - 1.0 / self._fx # shape (Npoints,)
            jacobian_flat_dp[:, 0, 3] = 0.0 # shape (Npoints,)

            jacobian_flat_dp[:, 1, 0] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 1, 1] = - y_D / self._fy # shape (Npoints,) because y_D = (y_I - c_y) / f_y
            jacobian_flat_dp[:, 1, 2] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 1, 3] = -1.0 / self._fy # shape (Npoints,)
        else:
            jacobian_flat_dp = None

        return distorted_points_flat, jacobian_flat_dx, jacobian_flat_dp



        