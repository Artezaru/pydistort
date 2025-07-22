from typing import Optional
from dataclasses import dataclass
import numpy

from .core.transform import TransformResult
from .core.distortion import Distortion
from .core.intrinsic import Intrinsic
from .core.extrinsic import Extrinsic
from .distortion_objects.no_distortion import NoDistortion
from .intrinsic_objects.no_intrinsic import NoIntrinsic
from .extrinsic_objects.no_extrinsic import NoExtrinsic


@dataclass
class ProjectPointsResult(TransformResult):
    r"""
    Subclass of TransformResult to represent the result of the projection transformation from 3D world points to 2D image points.

        This class is used to store the result of transforming the ``world_3dpoints`` to ``image_points``, and the optional Jacobians.

    - ``transformed_points``: The transformed image points in the camera pixel coordinate system.
    - ``jacobian_dx``: The Jacobian of the image points with respect to the input 3D world points if ``dx`` is True. Otherwise None. Shape (..., 2, 3).
    - ``jacobian_dp``: The Jacobian of the image points with respect to the parameters (extrinsic, distortion, intrinsic) if ``dp`` is True. Otherwise None. Shape (..., 2, Nextrinsic + Ndistortion + Nintrinsic).

    Some properties are provided for convenience:

    - ``image_points``: Alias for ``transformed_points`` to represent the transformed image points. Shape (..., 2).
    - ``jacobian_dextrinsic``: Alias for ``jacobian_dp[..., :Nextrinsic]`` to represent the Jacobian with respect to the extrinsic parameters. Shape (..., 2, Nextrinsic).
    - ``jacobian_ddistortion``: Alias for ``jacobian_dp[..., Nextrinsic:Nextrinsic + Ndistortion]`` to represent the Jacobian with respect to the distortion parameters. Shape (..., 2, Ndistortion).
    - ``jacobian_dintrinsic``: Alias for ``jacobian_dp[..., Nextrinsic + Ndistortion:]`` to represent the Jacobian with respect to the intrinsic parameters. Shape (..., 2, Nintrinsic).

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
    





def project_points(
        world_3dpoints: numpy.ndarray, 
        extrinsic: Optional[Extrinsic], 
        distortion: Optional[Distortion],
        intrinsic: Optional[Intrinsic],
        *,
        transpose: bool = False,
        dx: bool = False, 
        dp: bool = False,
        **kwargs
    ) -> ProjectPointsResult:
    r"""
    Project 3D points to 2D image points using the camera intrinsic, distortion and extrinsic transformations.

    .. seealso::

        To use a method usage-like OpenCV, use the :func:`pydistort.cv2_project_points` function.

    The process to correspond a 3D-world point to a 2D-image point is as follows:

    1. The ``world_3dpoints`` (:math:`X_W`) are projected to the camera coordinate system using the extrinsic parameters (rotation and translation) to obtain the ``normalized_points`` (:math:`x_N`).
    3. The ``normalized_points`` (:math:`x_N`) are distorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``distorted_points`` (:math:`x_D`).
    4. The ``distorted_points`` (:math:`x_D`) are projected onto the image plane using the intrinsic transformation to obtain the ``image_points`` (:math:`x_I`).

    .. math::

        \begin{align*}
        x_N = \text{Extrinsic}(X_W) \\
        x_D = \text{Distortion}(x_N) \\
        x_I = \text{Intrinsic}(x_D) \\
        \end{align*}

    The given points ``world 3dpoints`` are assumed to be in the world coordinate system and expressed in 3D coordinates with shape (..., 3).
        
    .. note::

        ``...`` in the shape of the arrays means that the array can have any number of dimensions.
        Classically, the ``...`` can be replaced by :math:`N` which is the number of points.

    To compute the Jacobians of the image points with respect to the input 3D world points and the projection parameters, set the ``dx`` and ``dp`` parameters to True.
    The Jacobians are computed using the chain rule of differentiation and are returned in the result object.

    To access the Jacobians, you can use the following properties of the result object:

    - ``jacobian_dx``: The Jacobian of the image points with respect to the input 3D world points. Shape (..., 2, 3).
    - ``jacobian_dp``: The Jacobian of the image points with respect to the projection parameters (extrinsic, distortion, intrinsic). Shape (..., 2, Nextrinsic + Ndistortion + Nintrinsic).
    - ``jacobian_dextrinsic``: Alias for ``jacobian_dp[..., :Nextrinsic]`` to represent the Jacobian with respect to the extrinsic parameters. Shape (..., 2, Nextrinsic).
    - ``jacobian_ddistortion``: Alias for ``jacobian_dp[..., Nextrinsic:Nextrinsic + Ndistortion]`` to represent the Jacobian with respect to the distortion parameters. Shape (..., 2, Ndistortion).
    - ``jacobian_dintrinsic``: Alias for ``jacobian_dp[..., Nextrinsic + Ndistortion:]`` to represent the Jacobian with respect to the intrinsic parameters. Shape (..., 2, Nintrinsic).

    Parameters
    ----------
    world_3dpoints : numpy.ndarray
        The 3D points in the world coordinate system. Shape (..., 3).

    extrinsic : Optional[Extrinsic]
        The extrinsic transformation to be applied to the 3D world points.
        If None, a no extrinsic transformation is applied (identity transformation).

    distortion : Optional[Distortion]
        The distortion model to be applied to the normalized points.
        If None, a zero distortion is applied (identity distortion).

    intrinsic : Optional[Intrinsic]
        The intrinsic transformation to be applied to the distorted points.
        If None, a no intrinsic transformation is applied (identity intrinsic).

    transpose : bool, optional
        If True, the input points are assumed to be in the shape (3, ...) instead of (..., 3). Default is False.
        In this case, the output points will be in the shape (2, ...) and the jacobians will be in the shape (2, ..., 3) and (2, ..., Nparams) respectively.
        
    dx : bool, optional
        If True, compute the Jacobian of the image points with respect to the input 3D world points with shape (..., 2, 3).
        If False, the Jacobian is not computed. default is False.

    dp : bool, optional
        If True, compute the Jacobian of the image points with respect to the projection parameters with shape (..., 2, Nparams).
        If False, the Jacobian is not computed. Default is False.

    **kwargs : dict
        Additional keyword arguments to be passed for the different transformations.
        
    Returns
    -------
    ProjectPointsResult
        The result of the projection transformation.
        
    Examples
    ~~~~~~~~~~

    Create a simple example to project 3D points to 2D image points using the intrinsic and extrinsic parameters of the camera.

    .. code-block:: python

        import numpy
        from pydistort import project_points, Cv2Distortion, Cv2Extrinsic, Cv2Intrinsic

        # Define the 3D points in the world coordinate system
        world_3dpoints = numpy.array([[0.0, 0.0, 5.0],
                                        [0.1, -0.1, 5.0],
                                        [-0.1, 0.2, 5.0],
                                        [0.2, 0.1, 5.0],
                                        [-0.2, -0.2, 5.0]]) # shape (5, 3)

        # Define the rotation vector and translation vector
        rvec = numpy.array([0.01, 0.02, 0.03])  # small rotation
        tvec = numpy.array([0.1, -0.1, 0.2])    # small translation
        extrinsic = Cv2Extrinsic(rvec=rvec, tvec=tvec)

        # Define the intrinsic camera matrix
        K = numpy.array([[1000.0, 0.0, 320.0],
                        [0.0, 1000.0, 240.0],
                        [0.0, 0.0, 1.0]])

        intrinsic = Cv2Intrinsic(intrinsic_matrix=K)

        # Define the distortion model (optional)
        distortion = Cv2Distortion(parameters = [0.1, 0.2, 0.3, 0.4, 0.5])

        # Project the 3D points to 2D image points
        result = project_points(world_3dpoints, extrinsic=extrinsic, distortion=distortion, intrinsic=intrinsic)
        print("Projected image points:")
        print(result.image_points) # shape (5, 2)

    You can also compute the Jacobians of the image points with respect to the input 3D world points and the projection parameters by setting the ``dx`` and ``dp`` parameters to True.

    .. code-block:: python

        # Project the 3D points to 2D image points with Jacobians
        result = project_points(world_3dpoints, extrinsic=extrinsic, distortion=distortion, intrinsic=intrinsic, dx=True, dp=True)

        print("Jacobian with respect to 3D points:")
        print(result.jacobian_dx) # shape (5, 2, 3)
        print("Jacobian with respect to projection parameters:")
        print(result.jacobian_dp) # shape (5, 2, Nparams)
        print("Jacobian with respect to extrinsic parameters:")
        print(result.jacobian_dextrinsic) # shape (5, 2, Nextrinsic) -> ordered as given by the selected extrinsic object
        print("Jacobian with respect to distortion parameters:")
        print(result.jacobian_ddistortion) # shape (5, 2, Ndistortion) -> ordered as given by the selected distortion object
        print("Jacobian with respect to intrinsic parameters:")
        print(result.jacobian_dintrinsic) # shape (5, 2, Nintrinsic) -> ordered as given by the selected intrinsic object

    This method can also be used without any extrinsic, distortion or intrinsic parameters by passing None.

    .. note::

        The output image points can be converted to the pixels coordinates in the image by swaping the axes :

        .. code-block:: python

            import numpy
            import cv2

            image = cv2.imread('image.jpg')
            image_height, image_width = image.shape[:2]

            pixel_points = numpy.indices((image_height, image_width), dtype=numpy.float64) # shape (2, H, W)
            pixel_points = pixel_points.reshape(2, -1).T  # shape (H*W, 2) WARNING: [H, W -> Y, X]
            
            image_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format

    """
    # Set the default values if None
    if intrinsic is None:
        intrinsic = NoIntrinsic()
    if extrinsic is None:
        extrinsic = NoExtrinsic()
    if distortion is None:
        distortion = NoDistortion()

    # Check the types of the parameters
    if not isinstance(intrinsic, Intrinsic):
        raise ValueError("intrinsic must be an instance of the Intrinsic class")
    if not intrinsic.is_set():
        raise ValueError("The intrinsic object must be ready to transform the points, check is_set() method.")
    if not isinstance(extrinsic, Extrinsic):
        raise ValueError("extrinsic must be an instance of the Extrinsic class")
    if not extrinsic.is_set():
        raise ValueError("The extrinsic object must be ready to transform the points, check is_set() method.")
    if not isinstance(distortion, Distortion):
        raise ValueError("distortion must be an instance of the Distortion class.")
    if not distortion.is_set():
        raise ValueError("The distortion object must be ready to transform the points, check is_set() method.")

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
    Nparams = intrinsic.Nparams + distortion.Nparams + extrinsic.Nparams # Total number of parameters

    # Check the shape of the points
    if points_flat.ndim !=2 or points_flat.shape[1] != 3:
        raise ValueError(f"The points must be in the shape (Npoints, 3) or (3, Npoints) if ``transpose`` is True. Got {points_flat.shape} instead and transpose is {transpose}.")
    
    # Initialize the jacobians
    jacobian_dx = None
    jacobian_dp = None
    
    # Realize the transformation:
    normalized_points, extrinsic_jacobian_dx, extrinsic_jacobian_dp = extrinsic._transform(points_flat, dx=dx, dp=dp)
    distorted_points, distortion_jacobian_dx, distortion_jacobian_dp = distortion._transform(normalized_points, dx=dx or dp, dp=dp, **kwargs) # (dx is requiered for propagation of dp)
    points_flat, intrinsic_jacobian_dx, intrinsic_jacobian_dp = intrinsic._transform(distorted_points, dx=dx or dp, dp=dp) # (dx is requiered for propagation of dp)

    # Apply the chain rules to compute the Jacobians with respect to the projection parameters
    if dp:
        jacobian_flat_dp = numpy.empty((Npoints, 2, Nparams), dtype=numpy.float64)
        # wrt the extrinsic parameters
        if isinstance(intrinsic, NoIntrinsic) and isinstance(distortion, NoDistortion):
            jacobian_flat_dp[..., :extrinsic.Nparams] = extrinsic_jacobian_dp
        elif isinstance(intrinsic, NoIntrinsic):
            jacobian_flat_dp[..., :extrinsic.Nparams] = numpy.matmul(distortion_jacobian_dx, extrinsic_jacobian_dp)
        elif isinstance(distortion, NoDistortion):
            jacobian_flat_dp[..., :extrinsic.Nparams] = numpy.matmul(intrinsic_jacobian_dx, extrinsic_jacobian_dp)
        else:
            jacobian_flat_dp[..., :extrinsic.Nparams] = numpy.matmul(intrinsic_jacobian_dx, numpy.matmul(distortion_jacobian_dx, extrinsic_jacobian_dp))

        # wrt the distortion parameters
        if intrinsic is None:
            jacobian_flat_dp[..., extrinsic.Nparams:extrinsic.Nparams + distortion.Nparams] = distortion_jacobian_dp
        else:
            jacobian_flat_dp[..., extrinsic.Nparams:extrinsic.Nparams + distortion.Nparams] = numpy.matmul(intrinsic_jacobian_dx, distortion_jacobian_dp)

        # wrt the intrinsic parameters
        jacobian_flat_dp[..., extrinsic.Nparams + distortion.Nparams:extrinsic.Nparams + distortion.Nparams + intrinsic.Nparams] = intrinsic_jacobian_dp # (intrinsic parameters)

    # Apply the chain rules to compute the Jacobians with respect to the input 3D world points
    if dx:
        if isinstance(intrinsic, NoIntrinsic) and isinstance(distortion, NoDistortion):
            jacobian_flat_dx = extrinsic_jacobian_dx
        elif isinstance(intrinsic, NoIntrinsic):
            jacobian_flat_dx = numpy.matmul(distortion_jacobian_dx, extrinsic_jacobian_dx)
        elif isinstance(distortion, NoDistortion):
            jacobian_flat_dx = numpy.matmul(intrinsic_jacobian_dx, extrinsic_jacobian_dx)
        else:
            jacobian_flat_dx = numpy.matmul(intrinsic_jacobian_dx, numpy.matmul(distortion_jacobian_dx, extrinsic_jacobian_dx)) # shape (Npoints, 2, 3)

    # Reshape the normalized points back to the original shape (Warning shape is (..., 3) and not (..., 2))
    image_points = points_flat.reshape((*shape[:-1], 2)) # shape (Npoints, 2) -> (..., 2)
    jacobian_dx = jacobian_flat_dx.reshape((*shape[:-1], 2, 3)) if dx else None # shape (Npoints, 2, 3) -> (..., 2, 3)
    jacobian_dp = jacobian_flat_dp.reshape((*shape[:-1], 2, Nparams)) if dp else None # shape (Npoints, 2, Nparams) -> (..., 2, Nparams)

    # Transpose the points back to the original shape if needed
    if transpose:
        image_points = numpy.moveaxis(image_points, -1, 0) # (..., 2) -> (2, ...)
        jacobian_dx = numpy.moveaxis(jacobian_dx, -2, 0) if dx else None # (..., 2, 2) -> (2, ..., 2)
        jacobian_dp = numpy.moveaxis(jacobian_dp, -2, 0) if dp else None # (..., 2, Nparams) -> (2, ..., Nparams)

    # Return the result
    result = ProjectPointsResult(
        transformed_points=image_points,
        jacobian_dx=jacobian_dx,
        jacobian_dp=jacobian_dp
    )

    # Add the short-hand properties for the jacobians
    result.add_jacobian("dextrinsic", 0, extrinsic.Nparams, f"Jacobian of the image points with respect to the extrinsic parameters (see {extrinsic.__class__.__name__}) for more details on their order")
    result.add_jacobian("ddistortion", extrinsic.Nparams, extrinsic.Nparams + distortion.Nparams, f"Jacobian of the image points with respect to the distortion parameters (see {distortion.__class__.__name__}) for more details on their order")
    result.add_jacobian("dintrinsic", extrinsic.Nparams + distortion.Nparams, Nparams, f"Jacobian of the image points with respect to the intrinsic parameters (see {intrinsic.__class__.__name__}) for more details on their order")

    return result


