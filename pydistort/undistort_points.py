from typing import Optional
import numpy


from .core.distortion import Distortion
from .core.intrinsic import Intrinsic
from .core.extrinsic import Extrinsic

from .distortion_objects.no_distortion import NoDistortion
from .intrinsic_objects.no_intrinsic import NoIntrinsic
from .extrinsic_objects.no_extrinsic import NoExtrinsic




def undistort_points(
        image_points: numpy.ndarray,
        intrinsic: Optional[Intrinsic],
        distortion: Optional[Distortion],
        R: Optional[Extrinsic] = None,
        P: Optional[Intrinsic] = None,
        *,
        transpose: bool = False,
        **kwargs
    ) -> numpy.ndarray:
    r"""
    Undistort 2D image points using the camera intrinsic, distortion and extrinsic transformations.

    .. seealso::

        To use a method usage-like OpenCV, use the :func:`pydistort.cv2_undistort_points` function.

    The process to undistort a 2D-image point is as follows:

    1. The ``image_points`` (:math:`x_I`) are normalized by applying the inverse intrinsic application to obtain the ``distorted_points`` (:math:`x_D`).
    2. The ``distorted_points`` (:math:`x_D`) are undistorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``normalized_points`` (:math:`x_N`).
    3. A rectification extrinsic operation R and a new projection intrinsic projection P can be applied to the ``normalized_points`` to return the ``undistorted_points`` in the space required by the user.

    .. note::

        The ``P = intrinsic`` to return the undistorted points in the image coordinate system.

    .. warning::

        Iterative non-linear optimization is used to find the undistorted points.

    The given points ``image_points`` are assumed to be in the image coordinate system and expressed in 2D coordinates with shape (..., 2).
    If the user gives not give a intrinsic transformation, it equivalent to give directly the normalized points.

    This method not compute the jacobians of the undistortion process, but it can be done by using the ``dx`` and ``dp`` flags in the intrinsic and distortion models if they can compute the jacobians.

    Parameters
    ----------
    image_points : numpy.ndarray
        The 2D image points in the camera coordinate system. Shape (..., 2)
        
    intrinsic : Optional[Intrinsic]
        The intrinsic transformation to be applied to the image points.
        If None, a zero intrinsic is applied (i.e., identity transformation).

    distortion : Optional[Distortion]
        The distortion model to be applied to the normalized points.
        If None, a zero distortion is applied (i.e., identity transformation).

    R : Optional[Extrinsic], optional
        The rectification extrinsic transformation (rotation and translation) to be applied to the normalized points.
        If None, a zero extrinsic is applied (i.e., identity transformation). Default is None.

    P : Optional[Intrinsic], optional
        The projection intrinsic transformation to be applied to the normalized points.
        If None, the intrinsic matrix is assumed to be the identity matrix (i.e., no projection transformation).
        This is useful to return the undistorted points in the image coordinate system.

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
        from pydistort import undistort_points, Cv2Distortion, Cv2Intrinsic
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
    
        # Create the intrinsic object
        intrinsic = Cv2Intrinsic(intrinsic_matrix=K)

        # Define the distortion model (optional)
        distortion = Cv2Distortion([0.1, 0.2, 0.3, 0.4, 0.5])

        # Undistort the 2D image points
        normalized_points = undistort_points(image_points, intrinsic=intrinsic, distortion=distortion)

    To return the undistorted points in the image coordinate system, you can provide a projection P equal to the intrinsic K:

    .. code-block:: python

        undistorted_points = undistort_points(image_points, intrinsic=intrinsic, distortion=distortion, P=K)
    
    """
    # Set the default values if None
    if intrinsic is None:
        intrinsic = NoIntrinsic()
    if distortion is None:
        distortion = NoDistortion()
    if R is None:
        R = NoExtrinsic()
    if P is None:
        P = NoIntrinsic()

    # Check the types of the parameters
    if not isinstance(intrinsic, Intrinsic):
        raise ValueError("intrinsic must be an instance of the Intrinsic class")
    if not intrinsic.is_set():
        raise ValueError("The intrinsic object must be ready to transform the points, check is_set() method.")
    if not isinstance(distortion, Distortion):
        raise ValueError("distortion must be an instance of the Distortion class.")
    if not distortion.is_set():
        raise ValueError("The distortion object must be ready to transform the points, check is_set() method.")
    if not isinstance(R, Extrinsic):
        raise ValueError("R must be an instance of the Extrinsic class")
    if not R.is_set():
        raise ValueError("The rectification extrinsic object must be ready to transform the points, check is_set() method.")
    if not isinstance(P, Intrinsic):
        raise ValueError("P must be an instance of the Intrinsic class")
    if not P.is_set():
        raise ValueError("The projection intrinsic object must be ready to transform the points, check is_set() method.")

    if not isinstance(transpose, bool):
        raise ValueError("transpose must be a boolean value")
    
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

    if not isinstance(R, NoExtrinsic):
        undistorted_points, _, _ = R._transform(numpy.concatenate((undistorted_points, numpy.ones((Npoints, 1))), axis=1), dx=False, dp=False) # shape (Npoints, 2) -> shape (Npoints, 3)

    if not isinstance(P, NoIntrinsic):
        undistorted_points, _, _ = P._transform(undistorted_points, dx=False, dp=False) # shape (Npoints, 3) -> shape (Npoints, 2)
    
    # Reshape the normalized points back to the original shape (Warning shape is (..., 2) and not (..., 3))
    undistorted_points = undistorted_points.reshape(shape) # shape (Npoints, 2) -> (..., 2)

    # Transpose the points back to the original shape if needed
    if transpose:
        undistorted_points = numpy.moveaxis(undistorted_points, -1, 0) # (..., 2) -> (2, ...)
    
    return undistorted_points

