from typing import Optional
import numpy


from .core.distortion import Distortion
from .distortion_objects.no_distortion import NoDistortion
from .intrinsic_objects.cv2_intrinsic import Cv2Intrinsic
from .extrinsic_objects.cv2_extrinsic import Cv2Extrinsic



def cv2_undistort_points(
        image_points: numpy.ndarray,
        K: Optional[numpy.ndarray],
        distortion: Optional[Distortion],
        R: Optional[numpy.ndarray] = None,
        P: Optional[numpy.ndarray] = None,
        transpose: bool = False,
        **kwargs
    ) -> numpy.ndarray:
    r"""

    .. warning::

        DEPRECATED: This function is deprecated and will be removed in a future version. Use :func:`pydistort.undistort_points` instead.

    Undistort 2D image points using the camera intrinsic and distortion coefficients.

    This method use the same architecture as the `cv2.undistortPoints` function from OpenCV, but it is implemented in a more flexible way to allow the use of different distortion models.
    
    .. seealso::

        - :func:`pydistort.undistort_points` for a more general undistortion function that can handle different types of points and transformations (extrinsic, intrinsic, distortion).

    The process to undistort a 2D-image point is as follows:

    1. The ``image_points`` (:math:`x_I`) are normalized by multiplying by the inverse of the intrinsic matrix K to obtain the ``distorted_points`` (:math:`x_D`).
    2. The ``distorted_points`` (:math:`x_D`) are undistorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``normalized_points`` (:math:`x_N`).
    3. A rectification matrix R and a new projection matrix P can be applied to the ``normalized_points`` to return the ``undistorted_points`` in the space required by the user.

    .. warning::

        Iterative non-linear optimization is used to find the undistorted points.

    The given points ``image_points`` are assumed to be in the image coordinate system and expressed in 2D coordinates with shape (..., 2).
    If the user gives an identity matrix K, it equivalent to give directly the normalized points.

    This method not compute the jacobians of the undistortion process, but it can be done by using the ``dx`` and ``dp`` flags in the intrinsic and distortion models.

    Parameters
    ----------
    image_points : numpy.ndarray
        The 2D image points in the camera coordinate system. Shape (..., 2)
        
    K : Optional[numpy.ndarray]
        The intrinsic camera matrix (or vector). Shape (3, 3) or (4,).
        If None, the identity intrinsic matrix is used.

    distortion : Optional[Distortion]
        The distortion model to be applied to the normalized points.
        If None, a zero distortion is applied.

    R : Optional[numpy.ndarray], optional
        The rotation matrix (or vector) of the camera. Shape (3,) or (3, 3).
        If None, the identity rotation is used. Default is None.

    P : Optional[numpy.ndarray], optional
        The new projection matrix (or vector). Shape (3, 3) or (4,).
        If None, the identity projection matrix is used. Default is None.

    transpose : bool, optional
        If True, the input points are assumed to be in the shape (2, ...) instead of (..., 2). Default is False.
        The output points will be in the same shape as the input points.
    
    kwargs : optional
        Additional arguments to be passed to the distortion model "undistort" method.
        This is useful for some distortion models that require additional parameters.

    Returns
    -------
    numpy.ndarray
        The undistorted 2D image points in the camera coordinate system. Shape (..., 2). If no ``P`` is given, the ``normalized_points`` are returned instead of the ``undistorted_points``.

    Example
    --------
    The following example shows how to undistort 2D image points using the intrinsic camera matrix and a distortion model.

    .. code-block:: python

        import numpy
        from pydistort import cv2_undistort_points, Cv2Distortion
        from py3dframe import Frame

        # Define the 2D image points in the camera coordinate system
        image_points = numpy.array([[320.0, 240.0],
                                    [420.0, 440.0],
                                    [520.0, 540.0],
                                    [620.0, 640.0],
                                    [720.0, 740.0]]) # shape (5, 2)

        # Define the intrinsic camera matrix
        K = numpy.array([[1000.0, 0.0, 320.0],
                        [0.0, 1000.0, 240.0],
                        [0.0, 0.0, 1.0]])

        # Define the distortion model (optional)
        distortion = Cv2Distortion([0.1, 0.2, 0.3, 0.4, 0.5])

        # Undistort the 2D image points
        normalized_points = cv2_undistort_points(image_points, K=K, distortion=distortion)

    To return the undistorted points in the image coordinate system, you can provide a projection matrix P equal to the intrinsic matrix K:

    .. code-block:: python

        undistorted_points = cv2_undistort_points(image_points, K=K, distortion=distortion, P=K)
    
    """
    # Set the default values if None
    if K is None:
        K = numpy.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]], dtype=numpy.float64)
    if R is None:
        R = numpy.zeros((3,), dtype=numpy.float64)
    if P is None:
        P = numpy.eye(3, dtype=numpy.float64)
    if distortion is None:
        distortion = NoDistortion()

    # Create the intrinsic and distortion objects
    intrinsic = Cv2Intrinsic()
    K = numpy.asarray(K, dtype=numpy.float64)
    if K.size == 4:
        intrinsic.intrinsic_vector = K
    elif K.size == 9:
        intrinsic.intrinsic_matrix = K
    else:
        raise ValueError("K must be of shape (4,) or (3, 3)")
    
    intrinsic_projection = Cv2Intrinsic()
    P = numpy.asarray(P, dtype=numpy.float64)
    if P.size == 4:
        intrinsic_projection.intrinsic_vector = P
    elif P.size == 9:
        intrinsic_projection.intrinsic_matrix = P
    else:
        raise ValueError("P must be of shape (4,) or (3, 3)")
    
    rectification = Cv2Extrinsic()
    rectification.translation_vector = numpy.zeros((3,), dtype=numpy.float64)
    R = numpy.asarray(R, dtype=numpy.float64)
    if R.size == 3:
        rectification.rotation_vector = R
    elif R.size == 9:
        rectification.rotation_matrix = R
    else:
        raise ValueError("R must be of shape (3,) or (3, 3)")
    
    if not isinstance(distortion, Distortion):
        raise ValueError("distortion must be an instance of the Distortion class")
    if not intrinsic.is_set():
        raise ValueError("The intrinsic matrix K must be set")
    if not intrinsic_projection.is_set():
        raise ValueError("The projection matrix P must be set")
    if not rectification.is_set():
        raise ValueError("The rectification matrix R must be set")
    if not distortion.is_set():
        raise ValueError("The distortion coefficients must be set")
    
    # Create the array of points
    points = numpy.asarray(image_points, dtype=numpy.float64)

    # Transpose the points if needed
    if transpose:
        points = numpy.moveaxis(points, 0, -1) # (2, ...) -> (..., 2)

    # Extract the original shape
    shape = points.shape # (..., 2)

    # Flatten the points along the last axis
    points_flat = points.reshape(-1, shape[-1]) # shape (..., 2) -> shape (Npoints, 2)
    shape_flat = points_flat.shape # (Npoints, 2)
    Npoints = shape_flat[0] # Npoints

    # Check the shape of the points
    if points_flat.ndim !=2 or points_flat.shape[1] != 2:
        raise ValueError(f"The points must be in the shape (Npoints, 2) or (2, Npoints) if ``transpose`` is True. Got {points_flat.shape} instead and transpose is {transpose}.")
    
    # Realize the transformation:
    distorted_points, _, _ = intrinsic._inverse_transform(points_flat, dx=False, dp=False) # shape (Npoints, 2) -> shape (Npoints, 2)
    undistorted_points, _, _ = distortion._inverse_transform(distorted_points, dx=False, dp=False, **kwargs) # shape (Npoints, 2) -> shape (Npoints, 2)
    
    if not numpy.allclose(rectification.rotation_vector, numpy.zeros((3,), dtype=numpy.float64)):
        undistorted_points, _, _ = rectification._transform(numpy.concatenate((undistorted_points, numpy.ones((Npoints, 1))), axis=1), dx=False, dp=False) # shape (Npoints, 3) -> shape (Npoints, 2)

    if not numpy.allclose(intrinsic_projection.intrinsic_matrix, numpy.eye(3, dtype=numpy.float64)):
        undistorted_points, _, _ = intrinsic_projection._transform(undistorted_points, dx=False, dp=False) # shape (Npoints, 2) -> shape (Npoints, 2)

    # Reshape the normalized points back to the original shape (Warning shape is (..., 2) and not (..., 3))
    undistorted_points = undistorted_points.reshape(shape) # shape (Npoints, 2) -> (..., 2)

    # Transpose the points back to the original shape if needed
    if transpose:
        undistorted_points = numpy.moveaxis(undistorted_points, -1, 0) # (..., 2) -> (2, ...)
    
    return undistorted_points

