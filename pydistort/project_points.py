from typing import Optional
from dataclasses import dataclass
import numpy

from .objects.transform import TransformResult, TransformComposition, TransformInvertion
from .objects.distortion import Distortion
from .no_distortion import NoDistortion
from .objects.intrinsic import Intrinsic
from .objects.extrinsic import Extrinsic


@dataclass
class ProjectPointsResult(TransformResult):
    r"""
    Subclass of TransformResult to represent the result of the projection transformation from 3D world points to 2D image points.

        This class is used to store the result of transforming the ``world_3dpoints`` to ``image_points``, and the optional Jacobians.

    - ``transformed_points``: The transformed image points in the camera pixel coordinate system.
    - ``jacobian_dx``: The Jacobian of the image points with respect to the input 3D world points if ``dx`` is True. Otherwise None. Shape (..., 2, 3).
    - ``jacobian_dp``: The Jacobian of the image points with respect to the parameters (rotation, translation, distortion, intrinsic) if ``dp`` is True. Otherwise None. Shape (..., 2, 10 + Nparams). [rx, ry, rz, tx, ty, tz, fx, fy, cx, cy, d1, d2, ..., dNparams] where Nparams is the number of distortion parameters.

    Some properties are provided for convenience:

    - ``image_points``: Alias for ``transformed_points`` to represent the transformed image points. Shape (..., 2).
    - ``jacobian_dr``: Part of the Jacobian with respect to the rotation vector. Shape (..., 2, 3).
    - ``jacobian_dt``: Part of the Jacobian with respect to the translation vector. Shape (..., 2, 3).
    - ``jacobian_df``: Part of the Jacobian with respect to the focal length parameters. Shape (..., 2, 2).
    - ``jacobian_dc``: Part of the Jacobian with respect to the principal point parameters. Shape (..., 2, 2).
    - ``jacobian_ddis``: Part of the Jacobian with respect to the distortion coefficients. Shape (..., 2, Nparams).

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
            The transformed image points in the camera pixel coordinate system. Shape (..., 2).
        """
        return self.transformed_points

    @property
    def jacobian_dr(self) -> Optional[numpy.ndarray]:
        r"""
        Get the jacobian of the image points with respect to the rotation vector.

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
        Get the jacobian of the image points with respect to the translation vector.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to translation (dt). Shape (..., 2, 3).
        """
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., 3:6]
    
    @property
    def jacobian_df(self) -> Optional[numpy.ndarray]:
        r"""
        Get the jacobian of the image points with respect to the focal length.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to focal lenght parameters (df). Shape (..., 2, 2).
        """
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., 6:8]
    
    @property
    def jacobian_dc(self) -> Optional[numpy.ndarray]:
        r"""
        Get the jacobian of the image points with respect to the principal point.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to principal point parameters (dc). Shape (..., 2, 2).
        """
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., 8:10]
    
    @property
    def jacobian_ddis(self) -> Optional[numpy.ndarray]:
        r"""
        Get the jacobian of the image points with respect to the distortion coefficients.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to distortion parameters (ddis). Shape (..., 2, Nparams).
        """
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., 10:]
    





def project_points(
        world_3dpoints: numpy.ndarray, 
        rvec: Optional[numpy.ndarray],
        tvec: Optional[numpy.ndarray], 
        K: Optional[numpy.ndarray], 
        distortion: Optional[Distortion], 
        transpose: bool = False,
        dx: bool = False, 
        dp: bool = False,
        faster_dx: bool = True,
        **kwargs
    ) -> ProjectPointsResult:
    r"""
    Project 3D points to 2D image points using the camera intrinsic and extrinsic matrix and distortion coefficients.

    The process to correspond a 3D-world point to a 2D-image point is as follows:

    1. The ``world_3dpoints`` (:math:`X_W`) are expressed in the camera coordinate system using the rotation and translation matrices to obtain the ``camera_3dpoints`` (:math:`X_C`).
    2. The ``camera_3dpoints`` (:math:`X_C`) are normalized by dividing by the third coordinate to obtain the ``normalized_points`` (:math:`x_N`).
    3. The ``normalized_points`` (:math:`x_N`) are distorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``distorted_points`` (:math:`x_D`).
    4. The ``distorted_points`` (:math:`x_D`) are projected onto the image plane using the intrinsic matrix K to obtain the ``image_points`` (:math:`x_I`).

    .. math::

        \begin{align*}
        X_C &= R \cdot X_W + T \\
        x_N &= \frac{X_C}{Z_C} \\
        x_D &= \text{distort}(x_N, \lambda_1, \lambda_2, \lambda_3, \ldots) \\
        x_I &= K \cdot x_D
        \end{align*}

    where :math:`R` is the rotation matrix, :math:`T` is the translation vector et :math:`K` is the intrinsic camera matrix.
    The intrinsic camera matrix is defined as:

    .. math::

        K = \begin{pmatrix}
        f_x & 0 & c_x \\
        0 & f_y & c_y \\
        0 & 0 & 1
        \end{pmatrix}

    The given points ``world 3dpoints`` are assumed to be in the world coordinate system and expressed in 3D coordinates with shape (..., 3).
        
    .. note::

        ``...`` in the shape of the arrays means that the array can have any number of dimensions.
        Classically, the ``...`` can be replaced by :math:`N` which is the number of points.

    .. note::

        The rotation matrix and the rotation vector must be given in the convention 0 of py3dframe.
        (see https://github.com/Artezaru/py3dframe)

    To compute the Jacobians of the image points with respect to the input 3D world points and the projection parameters, set the ``dx`` and ``dp`` parameters to True.
    The Jacobians are computed using the chain rule of differentiation and are returned in the result object.

    .. note::

        For the Jacobian with respect to the input 3D world points, a faster method than the full chain rule can be used by setting the ``faster_dx`` parameter to True.
        The method uses the fact that the Jacobian of the image points with respect to the input 3D world points can be computed directly as the matrix product of the jacobian with respect to the translation and the rotation matrix.
        This method is only used if the ``dp`` parameter is set to True.

    Parameters
    ----------
    world_3dpoints : numpy.ndarray
        The 3D points in the world coordinate system. Shape (..., 3).

    rvec : Optional[numpy.ndarray]
        The rotation vector (or rotation matrix) of the camera. Shape (3,) or (3, 3).
        If None, the identity rotation is used.

    tvec : Optional[numpy.ndarray]
        The translation vector of the camera. Shape (3,).
        If None, the zero translation is used.

    K : Optional[numpy.ndarray]
        The intrinsic camera matrix (or vector). Shape (3, 3) or (4,).
        If None, the identity intrinsic matrix is used.

    distortion : Optional[Distortion]
        The distortion model to be applied to the image points.
        If None, a zero distortion is applied.

    transpose : bool, optional
        If True, the input points are assumed to be in the shape (3, ...) instead of (..., 3). Default is False.
        In this case, the output points will be in the shape (2, ...) and the jacobians will be in the shape (2, ..., 3) and (2, ..., 10 + Nparams).
        
    dx : bool, optional
        If True, compute the Jacobian of the image points with respect to the input 3D world points with shape (..., 2, 3).
        If False, the Jacobian is not computed. default is False.

    dp : bool, optional
        If True, compute the Jacobian of the image points with respect to the projection parameters with shape (..., 2, 10 + Nparams).
        If False, the Jacobian is not computed. Default is False.

    faster_dx : bool, optional
        If True, use a faster method to compute the Jacobian of the image points with respect to the input 3D world points.
        Default is True.
        This method is only processed if the ``dp`` parameter is set to True.

    **kwargs : dict
        Additional keyword arguments to be passed to the distortion model's transform method.
        
    Returns
    -------
    ProjectPointsResult
        The result of the projection transformation.

        
    Examples
    ~~~~~~~~~~

    Create a simple example to project 3D points to 2D image points using the intrinsic and extrinsic parameters of the camera.

    .. code-block:: python

        import numpy
        from pydistort import project_points, Cv2Distortion

        # Define the 3D points in the world coordinate system
        world_3dpoints = numpy.array([[0.0, 0.0, 5.0],
                                        [0.1, -0.1, 5.0],
                                        [-0.1, 0.2, 5.0],
                                        [0.2, 0.1, 5.0],
                                        [-0.2, -0.2, 5.0]]) # shape (5, 3)

        # Define the rotation vector and translation vector
        rvec = numpy.array([0.01, 0.02, 0.03])  # small rotation
        tvec = numpy.array([0.1, -0.1, 0.2])    # small translation

        # Define the intrinsic camera matrix
        K = numpy.array([[1000.0, 0.0, 320.0],
                        [0.0, 1000.0, 240.0],
                        [0.0, 0.0, 1.0]])

        # Define the distortion model (optional)
        distortion = Cv2Distortion([0.1, 0.2, 0.3, 0.4, 0.5])

        # Project the 3D points to 2D image points
        result = project_points(world_3dpoints, rvec=rvec, tvec=tvec, K=K, distortion=distortion)
        print("Projected image points:")
        print(result.image_points) # shape (5, 2)

    You can also compute the Jacobians of the image points with respect to the input 3D world points and the projection parameters by setting the ``dx`` and ``dp`` parameters to True.

    .. code-block:: python

        # Project the 3D points to 2D image points with Jacobians
        result = project_points(world_3dpoints, rvec=rvec, tvec=tvec, K=K, distortion=distortion, dx=True, dp=True)

        print("Jacobian with respect to 3D points:")
        print(result.jacobian_dx) # shape (5, 2, 3)
        print("Jacobian with respect to projection parameters:")
        print(result.jacobian_dp) # shape (5, 2, 10 + Nparams)

    """
    # Set the default values if None
    if rvec is None:
        rvec = numpy.zeros((3,), dtype=numpy.float64)
    if tvec is None:
        tvec = numpy.zeros((3,), dtype=numpy.float64)
    if K is None:
        K = numpy.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]], dtype=numpy.float64)
    if distortion is None:
        distortion = NoDistortion()

    # Create the intrinsic extrinsic and distortion objects
    tvec = numpy.asarray(tvec, dtype=numpy.float64)
    rvec = numpy.asarray(rvec, dtype=numpy.float64)
    extrinsic = Extrinsic()
    extrinsic.translation_vector = tvec
    if rvec.size == 3:
        extrinsic.rotation_vector = rvec
    elif rvec.size == 9:
        extrinsic.rotation_matrix = rvec
    else:
        raise ValueError("rvec must be of shape (3,) or (3, 3)")

    K = numpy.asarray(K, dtype=numpy.float64)
    intrinsic = Intrinsic()
    if K.size == 4:
        intrinsic.intrinsic_vector = K
    elif K.size == 9:
        intrinsic.intrinsic_matrix = K
    else:
        raise ValueError("K must be of shape (4,) or (3, 3)")
    
    if not isinstance(distortion, Distortion):
        raise ValueError("distortion must be an instance of the Distortion class")
    if not intrinsic.is_set():
        raise ValueError("The intrinsic matrix K must be set")
    if not extrinsic.is_set():
        raise ValueError("The extrinsic matrix (rvec, tvec) must be set")
    if not distortion.is_set():
        raise ValueError("The distortion coefficients must be set")
    
    if not isinstance(faster_dx, bool):
        raise ValueError("faster_dx must be a boolean value")
    if not isinstance(transpose, bool):
        raise ValueError("transpose must be a boolean value")
    if not isinstance(dx, bool):
        raise ValueError("dx must be a boolean value")
    if not isinstance(dp, bool):
        raise ValueError("dp must be a boolean value")        

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
    
    # Initialize the jacobians
    jacobian_dx = None
    jacobian_dp = None
    
    # Realize the transformation:
    normalized_points, extrinsic_jacobian_dx, extrinsic_jacobian_dp = extrinsic._transform(points_flat, dx=dx, dp=dp or (dx and faster_dx)) # (dx is requiered for propagation of dp)
    distorted_points, distortion_jacobian_dx, distortion_jacobian_dp = distortion._transform(normalized_points, dx=dp or dx, dp=dp, **kwargs) # (dx is requiered for propagation of dp)
    image_points_flat, intrinsic_jacobian_dx, intrinsic_jacobian_dp = intrinsic._transform(distorted_points, dx=dp or dx, dp=dp) # (dx is requiered for propagation of dp)

    # Apply the chain rules to compute the Jacobians with respect to the projection parameters
    if dp:
        Nparams = distortion.Nparams
        jacobian_flat_dp = numpy.empty((Npoints, 2, 10 + Nparams), dtype=numpy.float64)
        jacobian_flat_dp[..., 6:10] = intrinsic_jacobian_dp # (focal length and principal point)
        jacobian_flat_dp[..., 10:] = numpy.einsum("nij, njk -> nik", intrinsic_jacobian_dx, distortion_jacobian_dp) # (distortion coefficients)
        jacobian_flat_dp[..., 0:6] = numpy.einsum("nij, njk -> nik", intrinsic_jacobian_dx, numpy.einsum("nij, njk -> nik", distortion_jacobian_dx, extrinsic_jacobian_dp)) # (rotation and translation)
    
    # Apply the chain rules to compute the Jacobians with respect to the input 3D world points
    if dx:
        if faster_dx and dp:
            jacobian_flat_dx = numpy.empty((Npoints, 2, 3), dtype=numpy.float64)
            jacobian_flat_dx[:,0,:] = jacobian_flat_dp[:, 0, 3:6] @ extrinsic.rotation_matrix # shape (Npoints, 3)
            jacobian_flat_dx[:,1,:] = jacobian_flat_dp[:, 1, 3:6] @ extrinsic.rotation_matrix # shape (Npoints, 3)
        else:
            jacobian_flat_dx = numpy.einsum("nij, njk -> nik", intrinsic_jacobian_dx, numpy.einsum("nij, njk -> nik", distortion_jacobian_dx, extrinsic_jacobian_dx))

    # Reshape the normalized points back to the original shape (Warning shape is (..., 3) and not (..., 2))
    image_points = image_points_flat.reshape((*shape[:-1], 2)) # shape (Npoints, 2) -> (..., 2)
    jacobian_dx = jacobian_flat_dx.reshape((*shape[:-1], 2, 3)) if dx else None # shape (Npoints, 2, 3) -> (..., 2, 3)
    jacobian_dp = jacobian_flat_dp.reshape((*shape[:-1], 2, 10 + Nparams)) if dp else None # shape (Npoints, 2, 10 + Nparams) -> (..., 2, 10 + Nparams)

    # Transpose the points back to the original shape if needed
    if transpose:
        image_points = numpy.moveaxis(image_points, -1, 0) # (..., 2) -> (2, ...)
        jacobian_dx = numpy.moveaxis(jacobian_dx, -2, 0) if dx else None # (..., 2, 2) -> (2, ..., 2)
        jacobian_dp = numpy.moveaxis(jacobian_dp, -2, 0) if dp else None # (..., 2, 4) -> (2, ..., 4)

    # Return the result
    result = ProjectPointsResult(
        transformed_points=image_points,
        jacobian_dx=jacobian_dx,
        jacobian_dp=jacobian_dp
    )
    return result







# TO TEST IF COMPOSETRANSFORM WORK ! 
# def project_points_bis(
#         world_3dpoints: numpy.ndarray, 
#         rvec: Optional[numpy.ndarray],
#         tvec: Optional[numpy.ndarray], 
#         K: Optional[numpy.ndarray], 
#         distortion: Optional[Distortion], 
#         transpose: bool = False,
#         dx: bool = False, 
#         dp: bool = False,
#         faster_dx: bool = True,
#         **kwargs
#     ) -> ProjectPointsResult:
#     # Set the default values if None
#     if rvec is None:
#         rvec = numpy.zeros((3,), dtype=numpy.float64)
#     if tvec is None:
#         tvec = numpy.zeros((3,), dtype=numpy.float64)
#     if K is None:
#         K = numpy.array([[1.0, 0.0, 0.0],
#                          [0.0, 1.0, 0.0],
#                          [0.0, 0.0, 1.0]], dtype=numpy.float64)
#     if distortion is None:
#         distortion = NoDistortion()

#     # Create the intrinsic extrinsic and distortion objects
#     tvec = numpy.asarray(tvec, dtype=numpy.float64)
#     rvec = numpy.asarray(rvec, dtype=numpy.float64)
#     extrinsic = Extrinsic()
#     extrinsic.translation_vector = tvec
#     if rvec.size == 3:
#         extrinsic.rotation_vector = rvec
#     elif rvec.size == 9:
#         extrinsic.rotation_matrix = rvec
#     else:
#         raise ValueError("rvec must be of shape (3,) or (3, 3)")

#     K = numpy.asarray(K, dtype=numpy.float64)
#     intrinsic = Intrinsic()
#     if K.size == 4:
#         intrinsic.intrinsic_vector = K
#     elif K.size == 9:
#         intrinsic.intrinsic_matrix = K
#     else:
#         raise ValueError("K must be of shape (4,) or (3, 3)")
    
#     if not isinstance(distortion, Distortion):
#         raise ValueError("distortion must be an instance of the Distortion class")
#     if not intrinsic.is_set():
#         raise ValueError("The intrinsic matrix K must be set")
#     if not extrinsic.is_set():
#         raise ValueError("The extrinsic matrix (rvec, tvec) must be set")
#     if not distortion.is_set():
#         raise ValueError("The distortion coefficients must be set")
    
#     if not isinstance(faster_dx, bool):
#         raise ValueError("faster_dx must be a boolean value")
#     if not isinstance(transpose, bool):
#         raise ValueError("transpose must be a boolean value")
#     if not isinstance(dx, bool):
#         raise ValueError("dx must be a boolean value")
#     if not isinstance(dp, bool):
#         raise ValueError("dp must be a boolean value")        

#     # Create the array of points
#     points = numpy.asarray(world_3dpoints, dtype=numpy.float64) 

#     # Transpose the points if needed
#     if transpose:
#         points = numpy.moveaxis(points, 0, -1) # (3, ...) -> (..., 3)

#     # Extract the original shape
#     shape = points.shape # (..., 3)

#     # Flatten the points along the last axis
#     points_flat = points.reshape(-1, shape[-1]) # shape (..., 3) -> shape (Npoints, 3)
#     shape_flat = points_flat.shape # (Npoints, 3)
#     Npoints = shape_flat[0] # Npoints

#     # Check the shape of the points
#     if points_flat.ndim !=2 or points_flat.shape[1] != 3:
#         raise ValueError(f"The points must be in the shape (Npoints, 3) or (3, Npoints) if ``transpose`` is True. Got {points_flat.shape} instead and transpose is {transpose}.")
    
#     # Initialize the jacobians
#     jacobian_dx = None
#     jacobian_dp = None

#     # transform compisition:
#     # extrinsic -> distortion -> intrinsic
#     transform = TransformComposition([extrinsic, distortion, intrinsic])
#     image_points_flat, jacobian_flat_dx, jacobian_flat_dp = transform._transform(points_flat, dx=dx, dp=dp, **kwargs)

#     # Reshape the normalized points back to the original shape (Warning shape is (..., 3) and not (..., 2))
#     image_points = image_points_flat.reshape((*shape[:-1], 2)) # shape (Npoints, 2) -> (..., 2)
#     jacobian_dx = jacobian_flat_dx.reshape((*shape[:-1], 2, 3)) if dx else None # shape (Npoints, 2, 3) -> (..., 2, 3)
#     jacobian_dp = jacobian_flat_dp.reshape((*shape[:-1], 2, jacobian_flat_dp.shape[-1])) if dp else None # shape (Npoints, 2, 10 + Nparams) -> (..., 2, 10 + Nparams)

#     # Transpose the points back to the original shape if needed
#     if transpose:
#         image_points = numpy.moveaxis(image_points, -1, 0) # (..., 2) -> (2, ...)
#         jacobian_dx = numpy.moveaxis(jacobian_dx, -2, 0) if dx else None # (..., 2, 2) -> (2, ..., 2)
#         jacobian_dp = numpy.moveaxis(jacobian_dp, -2, 0) if dp else None # (..., 2, 4) -> (2, ..., 4)

#     # Return the result
#     result = ProjectPointsResult(
#         transformed_points=image_points,
#         jacobian_dx=jacobian_dx,
#         jacobian_dp=jacobian_dp
#     )
#     return result