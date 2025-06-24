from typing import Optional
import numpy
import cv2

from .distortion import Distortion
from .no_distortion import NoDistortion
from .intrinsic import Intrinsic


def undistort_image(
        src: numpy.ndarray,
        K: Optional[numpy.ndarray],
        distortion: Optional[Distortion],
        **kwargs
    ) -> numpy.ndarray:
    r"""
    Undistort an image using the camera intrinsic and distortion coefficients.

    The process to undistort an image is as follows:

    1. The output pixels are converted to a normalized coordinate system using the intrinsic matrix K.
    2. The normalized points are distorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}`.
    3. The distorted points are projected back to the input image coordinate system using the intrinsic matrix K.
    4. The undistorted image is obtained by mapping the pixels from the original image to the undistorted points.

    The given image ``src`` is assumed to be in the image coordinate system and expressed in 2D coordinates with shape (H, W, [C], [D]).
    If the user gives an identity matrix K, it is equivalent to giving directly the normalized points.

    The mapping is performed using OpenCV's `cv2.remap` function, which requires the source image and the mapping of pixel coordinates.

    .. note::

        - For an image the X dimension corresponds to the width and the Y dimension corresponds to the height.
        - Pixel [0, 1] is at XY = [1, 0] in the image coordinate system.
    
    Parameters
    ----------
    src : numpy.ndarray
        The input image to be undistorted. Shape (H, W, ...) where H is the height, W is the width.

    K : Optional[numpy.ndarray]
        The intrinsic camera matrix (or vector). Shape (3, 3) or (4,).
        If None, the identity intrinsic matrix is used.

    distortion : Optional[Distortion]
        The distortion model to be applied. If None, no distortion is applied.

    kwargs : dict
        Additional arguments to be passed to the distortion model "distort" method.
    
    Returns
    -------
    numpy.ndarray
        The undistorted image. Shape (H, W, ...) where H is the height, W is the width.
    
    Example
    -------

    .. code-block:: python

        import numpy
        from pydistort import undistort_points, Cv2Distortion

        # Define the intrinsic camera matrix
        K = numpy.array([[1000.0, 0.0, 320.0],
                        [0.0, 1000.0, 240.0],
                        [0.0, 0.0, 1.0]])

        # Define the distortion model (optional)
        distortion = Cv2Distortion([0.1, 0.2, 0.3, 0.4, 0.5])

        # Load the image to be undistorted
        src = cv2.imread('image.jpg')

        # Undistort the image
        undistorted_image = undistort_image(src, K, distortion)

    """
    # Set the default values if None
    if K is None:
        K = numpy.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]], dtype=numpy.float64)
    if distortion is None:
        distortion = NoDistortion()
    
    # Create the intrinsic and distortion objects
    intrinsic = Intrinsic()
    K = numpy.asarray(K, dtype=numpy.float64)
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
    if not distortion.is_set():
        raise ValueError("The distortion coefficients must be set")
    
    # Check if the input image is a valid numpy array
    if not isinstance(src, numpy.ndarray):
        raise ValueError("src must be a numpy array")
    
    if src.ndim < 2 or src.ndim > 4:
        raise ValueError("src must have 2 to 4 dimensions (H, W, [C], [D])")
    
    # Construct the pixel points in the image coordinate system
    height, width = src.shape[:2]
    pixel_points = numpy.indices((height, width), dtype=numpy.float64) # shape (2, H, W)
    
    image_points = pixel_points.copy() # shape (2, H, W) [2, Y, X]
    image_points = image_points.reshape(2, -1).T  # shape (2, H, W) [2, Y, X] -> shape (Npoints, 2) [Y, X]
    image_points = image_points[:, [1, 0]]  # Switch to [X, Y] format, shape (Npoints, 2) [Y, X] -> shape (Npoints, 2) [X, Y]

    # Distort the pixel points using the distortion model
    normalized_points, _, _ = intrinsic._inverse_transform(image_points, dx=False, dp=False) # shape (Npoints, 2) [X, Y] -> shape (Npoints, 2) [X/Z, Y/Z]
    distorted_points, _, _ = distortion._transform(normalized_points, dx=False, dp=False, **kwargs) # shape (Npoints, 2) [X/Z, Y/Z] -> shape (Npoints, 2) [X'/Z', Y'/Z']
    distorted_image_points, _, _ = intrinsic._transform(distorted_points, dx=False, dp=False) # shape (Npoints, 2) [X'/Z', Y'/Z'] -> shape (Npoints, 2) [X', Y']

    # Reshape the distorted image points for cv2.remap
    distorted_image_points = distorted_image_points[:, [1, 0]]  # Switch to [Y, X] format, shape (Npoints, 2) [X', Y'] -> shape (Npoints, 2) [Y', X']
    distorted_pixel_points = distorted_image_points.T.reshape(2, height, width) # shape (Npoints, 2) [Y', X'] -> shape (2, H, W) [Y', X']

    # Create the map for cv2.remap
    # dst(x, y) = src(map_x(x, y), map_y(x, y))

    map_x = distorted_pixel_points[1, :, :]  # X' coordinates, shape (H, W)
    map_y = distorted_pixel_points[0, :, :]  # Y' coordinates, shape (H, W)

    # Remap the image using OpenCV
    undistorted_image = cv2.remap(src, map_x.astype(numpy.float32), map_y.astype(numpy.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return undistorted_image





    

    
