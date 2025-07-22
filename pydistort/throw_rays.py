from typing import Optional
import numpy
import cv2
import scipy

from .core.distortion import Distortion
from .core.intrinsic import Intrinsic
from .core.extrinsic import Extrinsic

from .distortion_objects.no_distortion import NoDistortion
from .intrinsic_objects.no_intrinsic import NoIntrinsic
from .extrinsic_objects.no_extrinsic import NoExtrinsic

def throw_rays(
    image_points: numpy.ndarray,
    intrinsic: Optional[Intrinsic],
    distortion: Optional[Distortion],
    extrinsic: Optional[Extrinsic],
    *,
    transpose: bool = False,
    **kwargs
    ) -> numpy.ndarray:
    r"""

    Compute the rays emitted from the camera to the scene based on the given image points, intrinsic parameters, distortion model, and extrinsic parameters.

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

    The inverse process to compute the rays is as follows:

    1. The ``image_points`` (:math:`x_I`) are normalized by multiplying by the inverse of the intrinsic matrix K to obtain the ``distorted_points`` (:math:`x_D`).
    2. The ``distorted_points`` (:math:`x_D`) are undistorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``normalized_points`` (:math:`x_N`).
    3. The ``normalized_points`` (:math:`x_N`) are used to compute the rays in the world coordinate system using the extrinsic parameters (rotation and translation) to obtain the ``rays``.

    The ray structure is as follows:

    - The first 3 elements are the origin of the ray in the world coordinate system.
    - The last 3 elements are the direction of the ray in the world coordinate system. The direction vector is normalized.

    .. note::

        The expected image points can be extracted from the pixels coordinates in the image by swaping the axes :

        .. code-block:: python

            import numpy
            import cv2

            image = cv2.imread('image.jpg')
            image_height, image_width = image.shape[:2]

            pixel_points = numpy.indices((image_height, image_width), dtype=numpy.float64) # shape (2, H, W)
            pixel_points = pixel_points.reshape(2, -1).T  # shape (H*W, 2) WARNING: [H, W -> Y, X]
            
            image_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format 

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

    extrinsic : Optional[Extrinsic]
        The extrinsic transformation (rotation and translation) to be applied to the normalized points.
        If None, a zero extrinsic is applied (i.e., identity transformation).

    transpose : bool, optional
        If True, the input image points are transposed before processing, the input shape is expected to be (2, ...) instead of (..., 2) and the output shape will be (6, ...).
        Default is False.

    **kwargs : dict
        Additional keyword arguments for distortion models ``undistort`` method.

    Returns
    -------
    numpy.ndarray
        The rays in the world coordinate system. Shape (..., 6)
    
    Example
    -------

    Create a simple example to construct the rays from an image to the scene:

    .. code-block:: python

        import numpy
        import cv2
        from pydistort import throw_rays, Cv2Extrinsic, Cv2Intrinsic, Cv2Distortion

        # Read the image : 
        image = cv2.imread('image.jpg')
        image_height, image_width = image.shape[:2]

        # Construct the intrinsic transformation :
        intrinsic = Cv2Intrinsic(intrinsic_matrix=numpy.array([[1000, 0, image_width / 2],
                                                              [0, 1000, image_height / 2],
                                                              [0, 0, 1]], dtype=numpy.float64))

        # Construct the distortion transformation:
        distortion = Cv2Distortion(parameters=numpy.array([0.1, -0.05, 0, 0, 0], dtype=numpy.float64))

        # Construct the extrinsic transformation:
        extrinsic = Cv2Extrinsic(rotation_vector=[0.1, 0.2, 0.3], translation_vector=[0, 0, 5])

        # Define the image points (e.g., pixels in the image):
        pixel_points = numpy.indices((image_height, image_width), dtype=numpy.float64) # shape (2, H, W)
        pixel_points = pixel_points.reshape(2, -1).T  # shape (H*W, 2) WARNING: [H, W -> Y, X]
        image_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format

        # Throw rays from the image points to the scene:
        rays = throw_rays(image_points, intrinsic, distortion, extrinsic, transpose=False)

        # Here `rays` will contain the origin and direction of the rays in the world coordinate system with shape (..., 6).
        # rays[:,i] = [origin_x, origin_y, origin_z, direction_x, direction_y, direction_z]

    """
    # Set the default values if None
    if intrinsic is None:
        intrinsic = NoIntrinsic()
    if distortion is None:
        distortion = NoDistortion()
    if extrinsic is None:
        extrinsic = NoExtrinsic()

    # Check the types of the parameters
    if not isinstance(intrinsic, Intrinsic):
        raise ValueError("intrinsic must be an instance of the Intrinsic class")
    if not intrinsic.is_set():
        raise ValueError("The intrinsic object must be ready to transform the points, check is_set() method.")
    if not isinstance(distortion, Distortion):
        raise ValueError("distortion must be an instance of the Distortion class.")
    if not distortion.is_set():
        raise ValueError("The distortion object must be ready to transform the points, check is_set() method.")
    if not isinstance(extrinsic, Extrinsic):
        raise ValueError("extrinsic must be an instance of the Extrinsic class")
    if not extrinsic.is_set():
        raise ValueError("The extrinsic object must be ready to transform the points, check is_set() method.")
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
    rays = extrinsic._compute_rays(undistorted_points) # shape (Npoints, 2) -> shape (Npoints, 6)
    
    # Reshape the rays  back to the original shape
    rays = rays.reshape((*shape[:-1], 6)) # shape (Npoints, 6) -> (..., 6)

    # Transpose the rays back to the original shape if needed
    if transpose:
        rays = numpy.moveaxis(rays, -1, 0) # (..., 6) -> (6, ...)

    return rays