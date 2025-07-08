from typing import Optional, Tuple
from dataclasses import dataclass
import numpy
from py3dframe import Frame
import cv2

from .core import Extrinsic, ExtrinsicResult, InverseExtrinsicResult

@dataclass
class Cv2ExtrinsicResult(ExtrinsicResult):
    r"""
    Subclass of :class:`pydistort.core.ExtrinsicResult` to represent the result of the extrinsic transformation in the cv2 format.

    This class is used to store the result of transforming the ``world_3dpoints`` to ``normalized_points``, and the optional Jacobians.

    - ``transformed_points``: The transformed normalized points in the camera coordinate system. Shape (..., 2).
    - ``jacobian_dx``: The Jacobian of the normalized points with respect to the input world 3D points if ``dx`` is True. Otherwise None. Shape (..., 2, 3), where the last dimension represents (dx, dy, dz).
    - ``jacobian_dp``: The Jacobian of the normalized points with respect to the extrinsic parameters if ``dp`` is True. Otherwise None. Shape (..., 2, 6), where the last dimension represents (rx, ry, rz, tx, ty, tz).

    Some properties are provided for convenience:

    - ``normalized_points``: Alias for ``transformed_points`` to represent the transformed normalized points. Shape (..., 2).
    - ``jacobian_dr``: Part of the Jacobian with respect to the rotation vector. Shape (..., 2, 3).
    - ``jacobian_dt``: Part of the Jacobian with respect to the translation vector. Shape (..., 2, 3).

    .. warning::

        If ``transpose`` is set to True during the transformation, the output points will have shape (output_dim, ...) instead of (..., output_dim), same for the Jacobian matrices.

    """
    @property
    def jacobian_dr(self) -> Optional[numpy.ndarray]:
        r"""
        Get the Jacobian of the normalized points with respect to the rotation vector.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to rotation (dr). Shape (..., 2, 3).
        """
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., 0:3]

    @property
    def jacobian_dt(self) -> Optional[numpy.ndarray]:
        r"""
        Get the Jacobian of the normalized points with respect to the translation vector.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to translation (dt). Shape (..., 2, 3).
        """
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., 3:6]



@dataclass
class InverseCv2ExtrinsicResult(InverseExtrinsicResult):
    r"""
    Subclass of :pydistort.core.InverseExtrinsicResult` to represent the result of the inverse extrinsic transformation in the cv2 format.

    This class is used to store the result of transforming the ``normalized_points`` back to ``world_3dpoints``, and the optional Jacobians.

    - ``transformed_points``: The transformed world 3D points in the camera coordinate system. Shape (..., 3).
    - ``jacobian_dx``: The Jacobian of the world 3D points with respect to the input normalized points if ``dx`` is True. Otherwise None. Shape (..., 3, 2), where the last dimension represents (dx, dy).
    - ``jacobian_dp``: The Jacobian of the world 3D points with respect to the extrinsic parameters if ``dp`` is True. Otherwise None. Shape (..., 3, 6), where the last dimension represents (rx, ry, rz, tx, ty, tz).

    Some properties are provided for convenience:

    - ``world_3dpoints``: Alias for ``transformed_points`` to represent the transformed world 3D points. Shape (..., 3).
    - ``jacobian_dr``: Part of the Jacobian with respect to the rotation vector. Shape (..., 3, 3).
    - ``jacobian_dt``: Part of the Jacobian with respect to the translation vector. Shape (..., 3, 3).

    .. warning::

        If ``transpose`` is set to True during the transformation, the output points will have shape (output_dim, ...) instead of (..., output_dim), same for the Jacobian matrices.

    """
    @property
    def jacobian_dr(self) -> Optional[numpy.ndarray]:
        r"""
        Get the Jacobian of the world 3D points with respect to the rotation vector.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to rotation (dr). Shape (..., 3, 3).
        """
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., 0:3]
    
    @property
    def jacobian_dt(self) -> Optional[numpy.ndarray]:
        r"""
        Get the Jacobian of the world 3D points with respect to the translation vector.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to translation (dt). Shape (..., 3, 3).
        """
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., 3:6]
    






class Cv2Extrinsic(Extrinsic):
    r"""
    Subclass of :class:`pydistort.core.Extrinsic` to represent the extrinsic transformation using OpenCV conventions.

    .. note::

        To manage only ``world_3dpoints`` to ``camera_3dpoints``, use the package py3dframe (https://github.com/Artezaru/py3dframe).

    The equation used for the extrinsic transformation in the cv2 convention is: 

    .. math::

        \begin{align*}
        X_C &= R \cdot X_W + T \\
        x_N &= \frac{X_C}{Z_C} \\
        \end{align*}

    where :math:`R` is the rotation matrix, :math:`T` is the translation vector.

    .. note::

        To compute the translation vector and the rotation vector, you can use cv2.Rodrigues() or py3dframe.Frame with convention 4.

    Parameters
    ----------
    rvec : Optional[numpy.ndarray]
        The rotation vector of the camera. Shape (3,). If None, the rotation vector is not set.

    tvec : Optional[numpy.ndarray]
        The translation vector of the camera. Shape (3,). If None, the translation vector is not set.

    Example
    -------
    Create an extrinsic object with a rotation vector and a translation vector:

    .. code-block:: python

        import numpy as np
        from pydistort import Cv2Extrinsic

        rvec = np.array([0.1, 0.2, 0.3])
        tvec = np.array([0.5, 0.5, 0.5])

        extrinsic = Cv2Extrinsic(rvec, tvec)

    Then you can use the extrinsic object to transform ``world_3dpoints`` to ``camera_points``:

    .. code-block:: python

        world_3dpoints = np.array([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9],
                                   [10, 11, 12]])

        result = extrinsic.transform(world_3dpoints)
        normalized_points = result.normalized_points
        print(normalized_points)

    You can also access to the jacobian of the extrinsic transformation:

    .. code-block:: python

        result = extrinsic.transform(world_3dpoints, dx=True, dp=True)
        normalized_points_dx = result.jacobian_dx  # Shape (..., 2, 3)
        normalized_points_dp = result.jacobian_dp  # Shape (..., 2, 6)
        print(normalized_points_dx) 
        print(normalized_points_dp)

    The inverse transformation can be computed using the `inverse_transform` method:
    By default, the depth is assumed to be 1.0 for all points, but you can provide a specific depth for each point with shape (...,).

    .. code-block:: python

        depth = np.array([1.0, 2.0, 3.0, 4.0])  # Example depth values for each point

        inverse_result = extrinsic.inverse_transform(normalized_points, dx=True, dp=True, depth=depth)
        world_3dpoints = inverse_result.transformed_points  # Shape (..., 3)
        print(world_3dpoints)

    .. note::

        The jacobian with respect to the depth is not computed.
    
    .. seealso::

        For more information about the transformation process, see:

        - :meth:`pydistort.Cv2Extrinsic._transform` to transform the ``world_3dpoints`` to ``normalized_points``.
        - :meth:`pydistort.Cv2Extrinsic._inverse_transform` to transform the ``normalized_points`` back to ``world_3dpoints``.
    
    """
    def __init__(self, rvec: Optional[numpy.ndarray] = None, tvec: Optional[numpy.ndarray] = None):
        # Initialize the Transform base class
        super().__init__()

        # Initialize the extrinsic parameters
        self._rvec = None
        self._tvec = None

        # Set the extrinsic parameters
        self.rvec = rvec
        self.tvec = tvec

    # =============================================
    # Overwrite some properties from the base class
    # =============================================
    @property
    def Nparams(self) -> int:
        return 6 # The number of parameters is 6 (3 for rotation and 3 for translation) even if parameters are not set.
    
    @property
    def result_class(self) -> type:
        return Cv2ExtrinsicResult
    
    @property
    def inverse_result_class(self) -> type:
        return InverseCv2ExtrinsicResult
    
    # =============================================
    # Implement the parameters property
    # =============================================
    @property
    def parameters(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the parameters of the extrinsic transformation.

        The parameters are a numpy array of shape (6,) representing the rotation vector and translation vector concatenated.

        Returns
        -------
        Optional[numpy.ndarray]
            The parameters of the extrinsic transformation. Shape (6,) or None if not set.
        """
        if self._rvec is None or self._tvec is None:
            return None
        return numpy.concatenate((self._rvec, self._tvec), axis=0)
    
    @parameters.setter
    def parameters(self, params: Optional[numpy.ndarray]) -> None:
        if params is None:
            self._rvec = None
            self._tvec = None
            return
        params = numpy.asarray(params, dtype=numpy.float64).flatten()
        if params.shape != (6,):
            raise ValueError("Parameters must be a 1D array of shape (6,).")
        self.rotation_vector = params[:3]  # First 3 elements are the rotation vector
        self.translation_vector = params[3:]  # Last 3 elements are the translation vector

    # =============================================
    # translation vector
    # =============================================
    @property
    def translation_vector(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the translation vector ``tvec`` of the extrinsic transformation.

        The translation vector is a numpy array of shape (3,) representing the translation of the camera in the world coordinate system.

        .. note::

            An alias for ``translation_vector`` is ``tvec``.

        .. seealso::

            - :meth:`pydistort.Cv2Extrinsic.rotation_vector` or ``rvec`` to set the rotation vector of the extrinsic transformation.

        Returns
        -------
        Optional[numpy.ndarray]
            The translation vector of the camera in the world coordinate system. (or None if not set)
        """
        return self._tvec
    
    @translation_vector.setter
    def translation_vector(self, tvec: Optional[numpy.ndarray]) -> None:
        if tvec is None:
            self._tvec = None
            return
        tvec = numpy.asarray(tvec, dtype=numpy.float64).flatten()
        if tvec.shape != (3,):
            raise ValueError("Translation vector must be a 3D vector.")
        if not numpy.isfinite(tvec).all():
            raise ValueError("Translation vector must be a finite 3D vector.")
        self._tvec = tvec

    @property
    def tvec(self) -> Optional[numpy.ndarray]:
        return self.translation_vector

    @tvec.setter
    def tvec(self, tvec: Optional[numpy.ndarray]) -> None:
        self.translation_vector = tvec

    # =============================================
    # rotation vector
    # =============================================
    @property
    def rotation_vector(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the rotation vector ``rvec`` of the extrinsic transformation.

        The rotation vector is a numpy array of shape (3,) representing the rotation of the camera in the world coordinate system.

        .. note::

            An alias for ``rotation_vector`` is ``rvec``.

        .. seealso::

            - :meth:`pydistort.Cv2Extrinsic.translation_vector` or ``tvec`` to set the translation vector of the extrinsic transformation.

        Returns
        -------
        Optional[numpy.ndarray]
            The rotation vector of the camera in the world coordinate system. (or None if not set)
        """
        return self._rvec
    
    @rotation_vector.setter
    def rotation_vector(self, rvec: Optional[numpy.ndarray]) -> None:
        if rvec is None:
            self._rvec = None
            return
        rvec = numpy.asarray(rvec, dtype=numpy.float64).flatten()
        if rvec.shape != (3,):
            raise ValueError("Rotation vector must be a 3D vector.")
        if not numpy.isfinite(rvec).all():
            raise ValueError("Rotation vector must be a finite 3D vector.")
        self._rvec = rvec

    @property
    def rvec(self) -> Optional[numpy.ndarray]:
        return self.rotation_vector
    
    @rvec.setter
    def rvec(self, rvec: Optional[numpy.ndarray]) -> None:
        self.rotation_vector = rvec

    # =============================================
    # Rotation matrix
    # =============================================
    @property
    def rotation_matrix(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the rotation matrix of the extrinsic transformation.

        The rotation matrix is a numpy array of shape (3, 3) representing the rotation of the camera in the world coordinate system.

        .. note::

            The rotation matrix is computed using the Rodrigues formula.
            An alias for ``rotation_matrix`` is ``rmat``.

        Returns
        -------
        Optional[numpy.ndarray]
            The rotation matrix of the camera in the world coordinate system. (or None if not set)
        """
        if self._rvec is None:
            return None
        return cv2.Rodrigues(self._rvec)[0]
    
    @rotation_matrix.setter
    def rotation_matrix(self, rmat: Optional[numpy.ndarray]) -> None:
        if rmat is None:
            self._rvec = None
            return
        rmat = numpy.asarray(rmat, dtype=numpy.float64)
        if rmat.shape != (3, 3):
            raise ValueError("Rotation matrix must be a 3x3 matrix.")
        if not numpy.isfinite(rmat).all():
            raise ValueError("Rotation matrix must be a finite 3x3 matrix.")
        self._rvec = cv2.Rodrigues(rmat)[0].flatten()

    @property
    def rmat(self) -> Optional[numpy.ndarray]:
        return self.rotation_matrix
    
    @rmat.setter
    def rmat(self, rmat: Optional[numpy.ndarray]) -> None:
        self.rotation_matrix = rmat

    # =============================================
    # Frame (for 3dframe)
    # =============================================
    @property
    def frame(self) -> Optional[Frame]:
        r"""
        Get or set the 3D frame of the extrinsic transformation.

        The frame is a py3dframe.Frame object representing the 3D frame of the camera in the world coordinate system.

        .. seealso::

            https://github.com/Artezaru/py3dframe for more information about the Frame class.

        Returns
        -------
        Optional[Frame]
            The 3D frame of the camera in the world coordinate system. (or None if not set)
        """
        if self._rvec is None or self._tvec is None:
            return None
        return Frame(translation=self._tvec, rotation_vector=self._rvec, convention=4)
    
    @frame.setter
    def frame(self, frame: Optional[Frame]) -> None:
        if frame is None:
            self._rvec = None
            self._tvec = None
            return
        if not isinstance(frame, Frame):
            raise ValueError("Frame must be a py3dframe.Frame object.")
        self._rvec = frame.get_global_rotation_vector(convention=4)
        self._tvec = frame.get_global_translation(convention=4)

    # =============================================
    # Display the extrinsic parameters
    # =============================================
    def __repr__(self):
        r"""
        Return a string representation of the extrinsic parameters.

        Returns
        -------
        str
            A string representation of the extrinsic parameters.
        """
        return f"Cv2 Extrinsic Pose: rvec={self._rvec}, tvec={self._tvec}"
    
    # =============================================
    # Methods for ABC Transform Class
    # =============================================
    def is_set(self) -> bool:
        r"""
        Check if the extrinsic parameters are set.

        Returns
        -------
        bool
            True if both rotation vector and translation vector are set, False otherwise.
        """
        return self._rvec is not None and self._tvec is not None
    

    def _transform(self, world_3dpoints: numpy.ndarray, *, dx: bool = False, dp: bool = False) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        This method is called by the :meth:`pydistort.core.Transform.transform` method to perform the extrinsic transformation.
        This method allows to transform the ``world_3dpoints`` to ``normalized_points`` using the extrinsic parameters (rotation and translation).

        .. note::

            For ``_transform`` the input must have shape (Npoints, 3) with float64 type.
            The output has shape (Npoints, 2) for the normalized points and (Npoints, 2, 3) for the jacobian with respect to the 3D world points and (Npoints, 2, 6) for the jacobian with respect to the extrinsic parameters.

        The equation used for the transformation is:

        .. math::

            [X_C, Y_C, Z_C]^T = R \cdot [X_W, Y_W, Z_W]^T + T
        
        .. math::

            x_N = \frac{X_C}{Z_C}

        .. math::

            y_N = \frac{Y_C}{Z_C}

        where :math:`R` is the rotation matrix, :math:`T` is the translation vector.

        .. warning::

            This method is not designed to be used directly for the transformation of points.
            No checks are performed on the input points, so it is the user's responsibility to ensure that the input points are valid.

        Parameters
        ----------
        world_3dpoints : numpy.ndarray
            Array of world 3dpoints to be transformed with shape (Npoints, 3).

        dx : bool, optional
            If True, the Jacobian of the normalized points with respect to the input 3D world points is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 2, 3).

        dp : bool, optional
            If True, the Jacobian of the normalized points with respect to the pose parameters is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 2, 6).

        Returns
        -------
        normalized_points : numpy.ndarray
            The transformed image points in pixels. It will be a 2D array of shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The Jacobian of the normalized points with respect to the input 3D world points if ``dx`` is True. Otherwise None.
            It will be a 2D array of shape (Npoints, 2, 2) if ``transpose`` is False.

        jacobian_dp : Optional[numpy.ndarray]
            The Jacobian of the normalized points with respect to the pose parameters if ``dp`` is True. Otherwise None.
            It will be a 2D array of shape (Npoints, 2, 6) if ``transpose`` is False.
        """
        # Get the number of points
        Npoints = world_3dpoints.shape[0]

        # Get the rotation matrix and translation vector
        rmat, jacobian = cv2.Rodrigues(self._rvec)
        rmat = numpy.asarray(rmat, dtype=numpy.float64) # shape (3, 3)
        jacobian = numpy.asarray(jacobian, dtype=numpy.float64) # shape (3, 9) [R11,R12,R13,R21,R22,R23,R31,R32,R33]
        rmat_dr = jacobian.reshape(3, 3, 3).transpose(1, 2, 0) # shape (3, 3, 3) # [i, j, k] = dR[i,j]/drvec[k]

        # ==================
        # Camera points
        # ==================
        # Compute the camera points
        points_camera_flat = world_3dpoints @ rmat.T + self._tvec[numpy.newaxis, :] # shape (Npoints, 3)
        X_C = points_camera_flat[:, 0] # shape (Npoints,)
        Y_C = points_camera_flat[:, 1] # shape (Npoints,)
        Z_C = points_camera_flat[:, 2] # shape (Npoints,)

        # Compute the jacobian with respect to the world points
        if dx:
            points_camera_flat_dx = numpy.broadcast_to(rmat, (Npoints, 3, 3))
            X_C_dx = points_camera_flat_dx[:, 0, :] # shape (Npoints, 3)
            Y_C_dx = points_camera_flat_dx[:, 1, :] # shape (Npoints, 3)
            Z_C_dx = points_camera_flat_dx[:, 2, :] # shape (Npoints, 3)

        # Compute the jacobian with respect to the extrinsic parameters
        if dp:
            points_camera_flat_dp = numpy.empty((Npoints, 3, 6), dtype=numpy.float64) # shape (Npoints, 3, 6)
            for k in range(3):
                points_camera_flat_dp[:, :, k] = world_3dpoints @ rmat_dr[:, :, k].T # shape (Npoints, 3)
            points_camera_flat_dp[:, :, 3] = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float64)[numpy.newaxis, :] # shape (Npoints, 3)
            points_camera_flat_dp[:, :, 4] = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float64)[numpy.newaxis, :] # shape (Npoints, 3)
            points_camera_flat_dp[:, :, 5] = numpy.array([0.0, 0.0, 1.0], dtype=numpy.float64)[numpy.newaxis, :] # shape (Npoints, 3)
            X_C_dp = points_camera_flat_dp[:, 0, :] # shape (Npoints, 6)
            Y_C_dp = points_camera_flat_dp[:, 1, :] # shape (Npoints, 6)
            Z_C_dp = points_camera_flat_dp[:, 2, :] # shape (Npoints, 6)

        # ==================
        # Normalized points
        # ==================
        if numpy.any(numpy.abs(points_camera_flat[:, 2]) < 1e-6):
            raise ValueError("The Z coordinate of the camera points is too close to zero. This may cause numerical instability.")

        # Compute the normalized points
        iZ_C = 1.0 / Z_C # shape (Npoints,)

        normalized_points_flat = numpy.empty((Npoints, 2), dtype=numpy.float64) # shape (Npoints, 2)
        x_N = X_C * iZ_C
        y_N = Y_C * iZ_C
        normalized_points_flat[:, 0] = x_N
        normalized_points_flat[:, 1] = y_N

        # Compute the jacobian with respect to the camera points
        if dx:
            jacobian_flat_dx = numpy.empty((Npoints, 2, 3), dtype=numpy.float64) # shape (Npoints, 2, 3)
            jacobian_flat_dx[:, 0, :] = (X_C_dx - x_N[:, numpy.newaxis] * Z_C_dx) * iZ_C[:, numpy.newaxis] # shape (Npoints, 3)
            jacobian_flat_dx[:, 1, :] = (Y_C_dx - y_N[:, numpy.newaxis] * Z_C_dx) * iZ_C[:, numpy.newaxis] # shape (Npoints, 3)

        # Compute the jacobian with respect to the extrinsic parameters
        if dp:
            jacobian_flat_dp = numpy.empty((Npoints, 2, 6), dtype=numpy.float64) # shape (Npoints, 2, 6)
            jacobian_flat_dp[:, 0, :] = (X_C_dp - x_N[:, numpy.newaxis] * Z_C_dp) * iZ_C[:, numpy.newaxis] # shape (Npoints, 6)
            jacobian_flat_dp[:, 1, :] = (Y_C_dp - y_N[:, numpy.newaxis] * Z_C_dp) * iZ_C[:, numpy.newaxis] # shape (Npoints, 6)

        if not dx:
            jacobian_flat_dx = None
        if not dp:
            jacobian_flat_dp = None

        return normalized_points_flat, jacobian_flat_dx, jacobian_flat_dp
    

    def _inverse_transform(self, normalized_points: numpy.ndarray, *, dx: bool = False, dp: bool = False, depth: Optional[numpy.ndarray] = None) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        This method is called by the :meth:`pydistort.core.Transform.inverse_transform` method to perform the inverse extrinsic transformation.
        This method allows to transform the ``normalized_points`` back to ``world_3dpoints`` using the extrinsic parameters (rotation and translation).

        .. note::

            For ``_inverse_transform`` the input must have shape (Npoints, 2) with float64 type.
            The output has shape (Npoints, 3) for the world 3D points and (Npoints, 3, 2)

        The equation used for the transformation is:

        .. math::

            [X_W, Y_W, Z_W]^T = R^{-1} \cdot ([X_N, Y_N, 1]^T \cdot Z_C - T)

        where :math:`R^{-1}` is the inverse of the rotation matrix, :math:`T` is the translation vector.

        The depth parameter is used to scale the normalized points to the world 3D points.
        By default, the depth is assumed to be 1.0 for all points, but you can provide a specific depth for each point with shape (...,).

        .. warning::

            This method is not designed to be used directly for the transformation of points.
            No checks are performed on the input points, so it is the user's responsibility to ensure that the input points are valid.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            Array of normalized points to be transformed with shape (Npoints, 2).

        dx : bool, optional
            If True, the Jacobian of the world 3D points with respect to the input normalized points is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 3, 2).
        
        dp : bool, optional
            If True, the Jacobian of the world 3D points with respect to the pose parameters is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 3, 6).

        depth : Optional[numpy.ndarray], optional
            The depth of the points in the world coordinate system. If None, the depth is assumed to be 1.0 for all points.
            The shape should be (...,). If provided, it will be used to scale the normalized points to the world 3D points.

        Returns
        -------
        world_3dpoints : numpy.ndarray
            The transformed world 3D points. It will be a 2D array of shape (Npoints, 3).

        jacobian_dx : Optional[numpy.ndarray]
            The Jacobian of the world 3D points with respect to the input normalized points if ``dx`` is True. Otherwise None.
            It will be a 2D array of shape (Npoints, 3, 2) if ``transpose`` is False.
        
        jacobian_dp : Optional[numpy.ndarray]
            The Jacobian of the world 3D points with respect to the pose parameters if ``dp`` is True. Otherwise None.
            It will be a 2D array of shape (Npoints, 3, 6) if ``transpose`` is False.
        """
        # Get the number of points
        Npoints = normalized_points.shape[0]

        # Get the rotation matrix and translation vector
        rmat, jacobian = cv2.Rodrigues(self._rvec)
        rmat = numpy.asarray(rmat, dtype=numpy.float64) # shape (3, 3)
        rmat_inv = rmat.T # Inverse of the rotation matrix (R^{-1} = R^{T})
        jacobian = numpy.asarray(jacobian, dtype=numpy.float64) # shape (3, 9) [R11,R12,R13,R21,R22,R23,R31,R32,R33]
        rmat_dr = jacobian.reshape(3, 3, 3).transpose(1, 2, 0) # shape (3, 3, 3) # [i, j, k] = dR[i,j]/drvec[k]
        rmat_inv_dr = rmat_dr.transpose(1, 0, 2) # shape (3, 3, 3) # [i, j, k] = dR^{-1}[i,j]/drvec[k] = dR^{T}[i,j]/drvec[k] = dR[j,i]/drvec[k]


        # ==================
        # Check depth
        # ==================
        if depth is None:
            depth = numpy.ones((Npoints,), dtype=numpy.float64)
        else:
            depth = numpy.asarray(depth, dtype=numpy.float64).flatten()
            if depth.shape != (Npoints,):
                raise ValueError("Depth must be a 1D array with the same number of points as normalized_points.")

        # ==================
        # Camera points
        # ==================
        # Compute the camera points
        X_C = normalized_points[:, 0] * depth # shape (Npoints,)
        Y_C = normalized_points[:, 1] * depth # shape (Npoints,)
        Z_C = depth # shape (Npoints,)

        points_camera_flat = numpy.empty((Npoints, 3), dtype=numpy.float64) # shape (Npoints, 3)
        points_camera_flat[:, 0] = X_C
        points_camera_flat[:, 1] = Y_C
        points_camera_flat[:, 2] = Z_C

        # Compute the jacobian with respect to the normalized points
        if dx:
            points_camera_flat_dx = numpy.empty((Npoints, 3, 2), dtype=numpy.float64) # shape (Npoints, 3, 2)
            points_camera_flat_dx[:, 0, 0] = depth # shape (Npoints, 2)
            points_camera_flat_dx[:, 0, 1] = 0.0
            points_camera_flat_dx[:, 1, 0] = 0.0
            points_camera_flat_dx[:, 1, 1] = depth # shape (Npoints, 2)
            points_camera_flat_dx[:, 2, 0] = 0.0
            points_camera_flat_dx[:, 2, 1] = 0.0

        # ===================
        # World points
        # ===================
        # Compute the world points
        world_3dpoints_flat = (points_camera_flat - self._tvec[numpy.newaxis, :]) @ rmat_inv.T # shape (Npoints, 3)
        X_W = world_3dpoints_flat[:, 0] # shape (Npoints,)
        Y_W = world_3dpoints_flat[:, 1] # shape (Npoints,)
        Z_W = world_3dpoints_flat[:, 2] # shape (Npoints,)

        # Compute the jacobian with respect to the camera points
        if dx:
            world_3dpoints_flat_dx = numpy.empty((Npoints, 3, 2), dtype=numpy.float64) # shape (Npoints, 3, 2)
            world_3dpoints_flat_dx[:, :, 0] = points_camera_flat_dx[:, :, 0] @ rmat_inv.T # shape (Npoints, 3)
            world_3dpoints_flat_dx[:, :, 1] = points_camera_flat_dx[:, :, 1] @ rmat_inv.T # shape (Npoints, 3)
    
        # Compute the jacobian with respect to the extrinsic parameters
        if dp:
            world_3dpoints_flat_dp = numpy.empty((Npoints, 3, 6), dtype=numpy.float64) # shape (Npoints, 3, 6)
            for k in range(3):
                world_3dpoints_flat_dp[:, :, k] = (points_camera_flat - self._tvec[numpy.newaxis, :]) @ rmat_inv_dr[:, :, k].T
            world_3dpoints_flat_dp[:, :, 3] = - numpy.array([1.0, 0.0, 0.0], dtype=numpy.float64)[numpy.newaxis, :] @ rmat_inv.T
            world_3dpoints_flat_dp[:, :, 4] = - numpy.array([0.0, 1.0, 0.0], dtype=numpy.float64)[numpy.newaxis, :] @ rmat_inv.T
            world_3dpoints_flat_dp[:, :, 5] = - numpy.array([0.0, 0.0, 1.0], dtype=numpy.float64)[numpy.newaxis, :] @ rmat_inv.T

        if not dx:
            world_3dpoints_flat_dx = None
        if not dp:
            world_3dpoints_flat_dp = None

        return world_3dpoints_flat, world_3dpoints_flat_dx, world_3dpoints_flat_dp