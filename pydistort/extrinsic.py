from typing import Optional
from numbers import Number
import numpy
from py3dframe import Frame
import cv2

class ExtrinsicResult(object):
    r"""
    Class to represent the result of the 3D pose projection transformation.

    This class is used to store the result of projecting 3D points using a pose (rotation + translation),
    along with optional Jacobians.

    .. note::

        ``...`` in the shape of the arrays means the array can have any number of leading dimensions.
        For example, shape (..., 3) corresponds to arbitrary shapes of 3D points.

    Parameters
    ----------
    normalized_points : numpy.ndarray
        The normalized points in the camera coordinate system. Shape (..., 2).

    jacobian_dx : Optional[numpy.ndarray]
        The Jacobian of the normalized points with respect to the input 3D world points if ``dx`` is True. Otherwise None.
        Shape (..., 2, 3).

    jacobian_dp : Optional[numpy.ndarray]
        The Jacobian of the normalized points with respect to the pose parameters (rotation and translation) if ``dp`` is True. Otherwise None.
        Shape (..., 2, 6), where the last dimension represents (dr, dt).
    """
    def __init__(self, normalized_points: numpy.ndarray, jacobian_dx: Optional[numpy.ndarray], jacobian_dp: Optional[numpy.ndarray]):
        self.normalized_points = normalized_points
        self.jacobian_dx = jacobian_dx
        self.jacobian_dp = jacobian_dp

    @property
    def jacobian_dr(self) -> Optional[numpy.ndarray]:
        r"""
        Get the Jacobian of the camera points with respect to the rotation vector.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to rotation (dr). Shape (..., 3, 3).
        """
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., :, 0:3]

    @property
    def jacobian_dt(self) -> Optional[numpy.ndarray]:
        r"""
        Get the Jacobian of the camera points with respect to the translation vector.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to translation (dt). Shape (..., 3, 3).
        """
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., :, 3:6]




class Extrinsic(object):
    r"""
    Class to represent the extrinsic parameters of a camera.

    This class defines the interface for extrinsic transformation for cameras.

    In the pinhole camera model, the extrinsic transformation is represented by a rotation matrix and a translation vector.
    The process to correspond a 3D-world point to a 2D-image point is as follows:

    1. The ``world_3dpoints`` (:math:`X_W`) are expressed in the camera coordinate system using the rotation and translation matrices to obtain the ``camera_3dpoints`` (:math:`X_C`).
    2. The ``camera_3dpoints`` (:math:`X_C`) are normalized by dividing by the third coordinate to obtain the ``normalized_points`` (:math:`x_N`).
    3. The ``normalized_points`` (:math:`x_N`) are distorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``distorted_points`` (:math:`x_D`).
    4. The ``distorted_points`` (:math:`x_D`) are projected onto the image plane using the intrinsic matrix K to obtain the ``image_points`` (:math:`x_I`).

    .. note::

        This class manage the transformation between the ``world_3dpoints`` and the ``normalized_points``.
        To manage only ``world_3dpoints`` to ``camera_3dpoints``, use the package py3dframe (https://github.com/Artezaru/py3dframe).

    .. math::

        \begin{align*}
        X_C &= R \cdot X_W + T \\
        x_N &= \frac{X_C}{X_C[2]} \\
        x_D &= \text{distort}(x_N, \lambda_1, \lambda_2, \lambda_3, \ldots) \\
        x_I &= K \cdot x_D
        \end{align*}

    where :math:`R` is the rotation matrix, :math:`T` is the translation vector.

    .. note::

        To compute the translation vector and the rotation vector, you can use cv2.Rodrigues() or py3dframe.Frame with convention 0.

    Parameters
    ----------
    rvec : numpy.ndarray
        The rotation vector of the camera. Shape (3,).
    
    tvec : numpy.ndarray
        The translation vector of the camera. Shape (3,).

    Example
    -------

    Create an extrinsic object with a rotation vector and a translation vector:

    .. code-block:: python

        import numpy as np
        from pydistort import Extrinsic

        rvec = np.array([0.1, 0.2, 0.3])
        tvec = np.array([0.5, 0.5, 0.5])

        extrinsic = Extrinsic(rvec, tvec)

    Then you can use the extrinsic object to transform ``world_3dpoints`` to ``camera_points``:

    .. code-block:: python

        world_3dpoints = np.array([[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9]])

        result = extrinsic.transform(world_3dpoints)
        normalized_points = result.normalized_points
        print(normalized_points)

    You can also access to the jacobian of the extrinsic transformation:

    .. code-block:: python

        result = extrinsic.transform(distorted_points, dx=True, dp=True)
        normalized_points_dx = result.jacobian_dx # Shape (..., 2, 3)
        normalized_points_dp = result.jacobian_dp # Shape (..., 2, 6)
        print(normalized_points_dx) 
        print(normalized_points_dp)

    """
    def __init__(self, rvec: Optional[numpy.ndarray] = None, tvec: Optional[numpy.ndarray] = None):
        # Initialize the extrinsic parameters
        self._rvec = None
        self._tvec = None

        # Set the extrinsic parameters
        self.rvec = rvec
        self.tvec = tvec

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

            - :meth:`pydistort.Extrinsic.rotation_vector` or ``rvec`` to set the rotation vector of the extrinsic transformation.

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

            - :meth:`pydistort.Extrinsic.translation_vector` or ``tvec`` to set the translation vector of the extrinsic transformation.

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
        return Frame(translation=self._tvec, rotation_vector=self._rvec, convention=0)
    
    @frame.setter
    def frame(self, frame: Optional[Frame]) -> None:
        if frame is None:
            self._rvec = None
            self._tvec = None
            return
        if not isinstance(frame, Frame):
            raise ValueError("Frame must be a py3dframe.Frame object.")
        self._rvec = frame.get_global_rotation_vector(convention=0)
        self._tvec = frame.get_global_translation(convention=0)

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
        return f"Extrinsic Pose: rvec={self._rvec}, tvec={self._tvec}"

    # =============================================
    # Check if the extrinsic matrix is set
    # =============================================
    def is_set(self) -> bool:
        r"""
        Check if the extrinsec parameters are set.

        The extrinsic parameters are set if the rotation vector and the translation vector are not None.

        Returns
        -------
        bool
            True if the extrinsic parameters are set, False otherwise.
        """
        return self._rvec is not None and self._tvec is not None
    
    # =============================================
    # Transformations
    # =============================================
    def _transform(self, world_3dpoints: numpy.ndarray, dx: bool = False, dp: bool = False) -> tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        This methos allow to bypass the check on the input points made in the transform method.

        .. note::

            For ``_transform`` the input is always in the shape (Npoints, 3) with float64 type.
            The output must be (Npoints, 2) for the normalized points and (Npoints, 2, 3) for the jacobian with respect to the 3D world points and (Npoints, 2, 6) for the jacobian with respect to the extrinsic parameters.

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

        kwargs : dict, optional
            Additional keyword arguments to be passed to the distortion model.

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
            points_camera_flat_dx = numpy.empty((Npoints, 3, 3), dtype=numpy.float64) # shape (Npoints, 3, 3)
            points_camera_flat_dx[:, :, :] = rmat[numpy.newaxis, :, :] # shape (Npoints, 3, 3)
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



    def transform(self, world_3dpoints: numpy.ndarray, transpose: bool = False, dx: bool = False, dp: bool = False) -> ExtrinsicResult:
        r"""
        Transform the given ``world 3dpoints`` to ``normalized points`` using the extrinsic parameters.

        The given points ``world 3dpoints`` are assumed to be in the world coordinate system and expressed in 3D coordinates with shape (..., 3).
        
        .. note::

            ``...`` in the shape of the arrays means that the array can have any number of dimensions.
            Classically, the ``...`` can be replaced by :math:`N` which is the number of points.

        The equations used to transform the points are:

        .. math::

            [X_C, Y_C, Z_C]^T = R \cdot [X_W, Y_W, Z_W]^T + T
        
        .. math::

            x_N = \frac{X_C}{Z_C}

        .. math::

            y_N = \frac{Y_C}{Z_C}

        where :math:`R` is the rotation matrix, :math:`T` is the translation vector.

        The output points ``normalized points`` are in the camera coordinate system and expressed in 2D coordinates with shape (..., 2).

        .. warning::

            The points are converting to float type before applying the extrinsic transformation.

        The method also computes 2 Jacobian matrices if requested:

        - ``dx``: Jacobian of the normalized points with respect to the input 3D world points.
        - ``dp``: Jacobian of the normalized points with respect to the pose parameters (rotation and translation).

        The jacobian matrice with respect to the 3D points is a (..., 2, 3) matrix where :

        .. code-block:: python

            jacobian_dx[..., 0, 0]  # ∂x_N/∂X_W -> Jacobian of the coordinates x_N with respect to the coordinates X_W.
            jacobian_dx[..., 0, 1]  # ∂x_N/∂Y_W 
            jacobian_dx[..., 0, 2]  # ∂x_N/∂Z_W

            jacobian_dx[..., 1, 0]  # ∂y_N/∂X_W -> Jacobian of the coordinates y_N with respect to the coordinates X_W.
            jacobian_dx[..., 1, 1]  # ∂y_N/∂Y_W
            jacobian_dx[..., 1, 2]  # ∂y_N/∂Z_W

        The Jacobian matrice with respect to the pose parameters is a (..., 2, 6) matrix where :

        .. code-block:: python

            jacobian_dp[..., 0, :3]  # ∂x_N/∂rvec -> Jacobian of the coordinates x_N with respect to the rotation vector.
            jacobian_dp[..., 0, 3:]  # ∂x_N/∂tvec -> Jacobian of the coordinates x_N with respect to the translation vector.

            jacobian_dp[..., 1, :3]  # ∂y_N/∂rvec -> Jacobian of the coordinates y_N with respect to the rotation vector.
            jacobian_dp[..., 1, 3:]  # ∂y_N/∂tvec -> Jacobian of the coordinates y_N with respect to the translation vector.

            
        Parameters
        ----------
        world_3dpoints : numpy.ndarray
            Array of world 3dpoints to be transformed with shape (..., 3).

        transpose : bool, optional
            If True, the input points are assume to have shape (3, ...).
            In this case, the output points will have shape (3, ...) as well and the jacobian matrices will have shape (3, ..., 2) and (3, ..., 6) respectively.
            Default is False.

        dx : bool, optional
            If True, the Jacobian of the normalized points with respect to the input 3D world points is computed. Default is False.
            The output will be a 2D array of shape (..., 2, 2) if ``transpose`` is False.
            If ``dx`` is False, the output will be None.

        dp : bool, optional
            If True, the Jacobian of the normalized points with respect to the pose parameters is computed. Default is False.
            The output will be a 2D array of shape (..., 2, 6) if ``transpose`` is False.
            If ``dp`` is False, the output will be None.

        Returns
        -------
        extrinsic_result : ExtrinsicResult

            The result of the extrinsic transformation containing the normalized points and the jacobian matrices.
            This object has the following attributes:

            normalized_points : numpy.ndarray
                The transformed normalized points in the camera coordinate system. It will be a 2D array of shape (..., 2) if ``transpose`` is False.

            jacobian_dx : Optional[numpy.ndarray]
                The Jacobian of the normalized points with respect to the input 3D world points if ``dx`` is True. Otherwise None.
                It will be a 2D array of shape (..., 2, 3) if ``transpose`` is False.

            jacobian_dp : Optional[numpy.ndarray]
                The Jacobian of the normalized points with respect to the pose parameters if ``dp`` is True. Otherwise None.
                It will be a 2D array of shape (..., 2, 6) if ``transpose`` is False.

        Example
        -------

        Create an extrinsic object with a given pose:

        .. code-block:: python

            import numpy as np
            from pydistort import Extrinsic

            rvec = np.array([0.1, 0.2, 0.3]) # rotation vector
            tvec = np.array([0.5, 0.5, 0.5]) # translation vector

            extrinsic = Extrinsic(rvec, tvec)
            # or using py3dframe
            extrinsic = Extrinsic() # Default constructor
            extrinsic.frame = Frame(translation=tvec, rotation_vector=rvec, convention=0)

        Then you can use the extrinsic object to transform ``world_3dpoints`` to ``normalized points``:

        .. code-block:: python

            world_3dpoints = np.array([[1, 2, 3],
                                        [4, 5, 6],
                                        [7, 8, 9]]) # shape (3, 3)

            result = extrinsic.transform(world_3dpoints, dx=True, dp=True)

            result.normalized_points # shape (3, 2) -> normalized points in camera coordinates
            result.jacobian_dx # shape (3, 2, 3) -> jacobian of the normalized points with respect to the world points
            result.jacobian_dp # shape (3, 2, 6) -> jacobian of the normalized points with respect to the pose parameters
            result.jacobian_dr # shape (3, 2, 3) -> jacobian of the normalized points with respect to the rotation vector (extracted from jacobian_dp)
            result.jacobian_dt # shape (3, 2, 3) -> jacobian of the normalized points with respect to the translation vector (extracted from jacobian_dp)

        """
        # Check the boolean parameters
        if not isinstance(transpose, bool):
            raise ValueError("The transpose parameter must be a boolean.")
        if not isinstance(dx, bool):
            raise ValueError("The dx parameter must be a boolean.")
        if not isinstance(dp, bool):
            raise ValueError("The dp parameter must be a boolean.")
        
        # Check if the extrinsic matrix is set
        if not self.is_set():
            raise ValueError("The extrinsic matrix is not set. Please set the extrinsic matrix before using this method.")
        
        # Create the array of points
        points = numpy.asarray(world_3dpoints, dtype=numpy.float64) 

        # Transpose the points if needed
        if transpose:
            points = numpy.moveaxis(points, 0, -1) # (3, ...) -> (..., 3)

        # Extract the original shape
        shape = points.shape # (..., 3)

        # Flatten the points along the last axis
        points_flat = points.reshape(-1, shape[-1]) # shape (..., 3) -> shape (Npoints, 3)
        shape_flat = points_flat.shape # (Npoints, 3)
        Npoints = shape_flat[0] # Npoints

        # Check the shape of the points
        if points_flat.ndim !=2 or points_flat.shape[1] != 3:
            raise ValueError(f"The points must be in the shape (Npoints, 3) or (3, Npoints) if ``transpose`` is True. Got {points_flat.shape} instead and transpose is {transpose}.")

        normalized_points_flat, jacobian_flat_dx, jacobian_flat_dp = self._transform(points_flat, dx=dx, dp=dp)

        # Reshape the normalized points back to the original shape (Warning shape is (..., 3) and not (..., 2))
        normalized_points = normalized_points_flat.reshape((*shape[:-1], 2)) # shape (Npoints, 2) -> (..., 2)
        jacobian_dx = jacobian_flat_dx.reshape((*shape[:-1], 2, 3)) if dx else None # shape (Npoints, 2, 3) -> (..., 2, 3)
        jacobian_dp = jacobian_flat_dp.reshape((*shape[:-1], 2, 6)) if dp else None # shape (Npoints, 2, 6) -> (..., 2, 6)

        # Transpose the points back to the original shape if needed
        if transpose:
            image_points = numpy.moveaxis(image_points, -1, 0) # (..., 2) -> (2, ...)
            jacobian_dx = numpy.moveaxis(jacobian_dx, -2, 0) if dx else None # (..., 2, 2) -> (2, ..., 2)
            jacobian_dp = numpy.moveaxis(jacobian_dp, -2, 0) if dp else None # (..., 2, 6) -> (2, ..., 6)

        # Return the image points and the jacobian matrices
        result = ExtrinsicResult(normalized_points, jacobian_dx, jacobian_dp)
        return result
    