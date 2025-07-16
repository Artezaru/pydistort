from typing import Optional
import numpy
import cv2
import scipy

from .core.distortion import Distortion
from .core.intrinsic import Intrinsic

from .distortion_objects.no_distortion import NoDistortion
from .intrinsic_objects.no_intrinsic import NoIntrinsic

def distort_image(
        src: numpy.ndarray,
        intrinsic: Optional[Intrinsic],
        distortion: Optional[Distortion],
        method: str = "undistort",
        interpolation: str = "linear",
        **kwargs
    ) -> numpy.ndarray:
    r"""
    Distort an image using the camera intrinsic and distortion coefficients.

    .. seealso::

        To use a method usage-like OpenCV, use the :func:`pydistort.cv2_distort_image` function.

    The process to undistort an image is as follows:

    1. The output pixels are converted to a normalized coordinate system using the inverse intrinsic transformation.
    2. The normalized points are undistorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}`.
    3. The undistorted points are projected back to the input image coordinate system using the same intrinsic transformation.
    4. The distorted image is obtained by mapping the pixels from the original image to the distorted points.

    The given image ``src`` is assumed to be in the image coordinate system and expressed in 2D coordinates with shape (H, W, [C], [D]).
    If the user gives an identity matrix K, it is equivalent to giving directly the normalized points.

    .. note::

        - For an image the X dimension corresponds to the width and the Y dimension corresponds to the height.
        - Pixel [0, 1] is at XY = [1, 0] in the image coordinate system.

    Linear interpolation is used for each method to compute the pixel values in the distorted image.

    Fill Values for the output image are set to 0.0.

    METHOD 1 : Undistort
    ------------------------------
    
    The mapping is performed using OpenCV's `cv2.remap` function (or ``scipy.interpolate``), which requires the source image and the mapping of pixel coordinates.
    The mapping of pixel coordinates is performed using the ``undistort`` method of the distortion model, which applies the inverse distortion to the normalized points.

    In this case, the output pixels (``distorted_image``) are projected back to the input image coordinate system using the intrinsic matrix K.
    Then an interpolation is performed in the input image where the known points are a perfect regular grid of pixels !

    .. warning::

        Iterative non-linear optimization is used to find the undistorted points.

    Different interpolation methods can be used, such as "linear", "nearest", etc. The default is "linear".
    The table below shows the available interpolation methods for METHOD 1:

    +----------------+----------------------------------------------------------------------------------------------------------------+
    | Interpolation  | Description                                                                                                    |
    +================+================================================================================================================+
    | "linear"       | Linear interpolation (default). Use cv2.INTER_LINEAR.                                                          |
    +----------------+----------------------------------------------------------------------------------------------------------------+
    | "nearest"      | Nearest neighbor interpolation. Use cv2.INTER_NEAREST.                                                         |
    +----------------+----------------------------------------------------------------------------------------------------------------+
    | "cubic"        | Bicubic interpolation. Use cv2.INTER_CUBIC.                                                                    |
    +----------------+----------------------------------------------------------------------------------------------------------------+
    | "area"         | Resampling using pixel area relation. Use cv2.INTER_AREA.                                                      |
    +----------------+----------------------------------------------------------------------------------------------------------------+
    | "lanczos4"     | Lanczos interpolation over 8x8 pixel neighborhood. Use cv2.INTER_LANCZOS4.                                     |
    +----------------+----------------------------------------------------------------------------------------------------------------+
    | "spline3"      | Spline interpolation. Use scipy.interpolate.RectBivariateSpline for kx=ky=3                                    |
    +----------------+----------------------------------------------------------------------------------------------------------------+

    METHOD 2 : Distort
    ------------------------------

    The mapping is performed using scipy `scipy.interpolate` function, which requires the source image and the mapping of pixel coordinates.
    The mapping of pixel coordinates is performed using the ``distort`` method of the distortion model, which applies the distortion to the normalized points.

    In this case, the input pixels (``src``) are projected to the output image coordinate system using the intrinsic matrix K.
    Then an interpolation is performed in the output image for all the output pixels where the known points are a irregular cloud of points.

    .. warning::

        - Time computation is higher than the first method.
        - Output values are not integer values (even if the input image is integer), so the user must apply (numpy.round) to the output image to get integer values.

    Different interpolation methods can be used, such as "linear", etc. The default is "linear".
    The table below shows the available interpolation methods for METHOD 2:

    +----------------+----------------------------------------------------------------------------------------------------------------+
    | Interpolation  | Description                                                                                                    |
    +================+================================================================================================================+
    | "linear"       | Linear interpolation (default). Use scipy.interpolate.LinearNDInterpolator                                     |
    +----------------+----------------------------------------------------------------------------------------------------------------+
    | "nearest"      | Nearest neighbor interpolation. Use scipy.interpolate.NearestNDInterpolator                                    |
    +----------------+----------------------------------------------------------------------------------------------------------------+
    | "clough"       | Piecewise cubic interpolation. Use scipy.interpolate.CloughTocher2DInterpolator.                               |
    +----------------+----------------------------------------------------------------------------------------------------------------+

    Parameters
    ----------
    src : numpy.ndarray
        The input image to be undistorted. Shape (H, W, ...) where H is the height, W is the width.

    intrinsic : Optional[Intrinsic]
        The intrinsic transformation to be applied to the image points.
        If None, a zero intrinsic is applied (i.e., identity transformation).

    distortion : Optional[Distortion]
        The distortion model to be applied. If None, no distortion is applied.

    method : str
        The method to use for undistortion. The value can be "undistort" or "distort".

        - "undistort": Uses OpenCV's `cv2.remap` and the ``undistort`` method of the distortion model.
        - "distort": Uses `scipy.interpolate` and the ``distort`` method of the distortion model.

    interpolation : str, optional
        The interpolation method to be used for remapping the pixels. Default is "linear".

    kwargs : dict
        Additional arguments to be passed to the distortion model "undistort" method.
    
    Returns
    -------
    numpy.ndarray
        The undistorted image. Shape (H, W, ...) where H is the height.
    
    Example
    -------

    .. code-block:: python

        import numpy
        from pydistort import distort_image, Cv2Distortion, Cv2Intrinsic

        # Define the intrinsic camera matrix
        K = numpy.array([[1000.0, 0.0, 320.0],
                        [0.0, 1000.0, 240.0],
                        [0.0, 0.0, 1.0]])

        # Create the intrinsic object
        intrinsic = Cv2Intrinsic(intrinsic_matrix=K)

        # Define the distortion model (optional)
        distortion = Cv2Distortion([0.1, 0.2, 0.3, 0.4, 0.5])

        # Load the image to be distorted
        src = cv2.imread('image.jpg')

        # Distort the image
        distorted_image = distort_image(src, intrinsic=intrinsic, distortion=distortion)

    """
    # Check if the method is valid
    if method not in ["undistort", "distort"]:
        raise ValueError(f"Invalid method: {method}. Use 'undistort' or 'distort'.")
    
    # Set the default values if None
    if intrinsic is None:
        intrinsic = NoIntrinsic()
    if distortion is None:
        distortion = NoDistortion()

    # Check the types of the parameters
    if not isinstance(intrinsic, Intrinsic):
        raise ValueError("intrinsic must be an instance of the Intrinsic class")
    if not intrinsic.is_set():
        raise ValueError("The intrinsic object must be ready to transform the points, check is_set() method.")
    if not isinstance(distortion, Distortion):
        raise ValueError("distortion must be an instance of the Distortion class.")
    if not distortion.is_set():
        raise ValueError("The distortion object must be ready to transform the points, check is_set() method.")
    
    # Check if the input image is a valid numpy array
    if not isinstance(src, numpy.ndarray):
        raise ValueError("src must be a numpy array")
    
    if src.ndim < 2 or src.ndim > 4:
        raise ValueError("src must have 2 to 4 dimensions (H, W, [C], [D])")
    
    # Get the interpolation method
    use_remap = False
    use_bivariate_spline = False
    if method == "undistort":
        if interpolation == "linear":
            interpolation_method = cv2.INTER_LINEAR
            use_remap = True
        elif interpolation == "nearest":
            interpolation_method = cv2.INTER_NEAREST
            use_remap = True
        elif interpolation == "cubic":
            interpolation_method = cv2.INTER_CUBIC
            use_remap = True
        elif interpolation == "area":
            interpolation_method = cv2.INTER_AREA
            use_remap = True
        elif interpolation == "lanczos4":
            interpolation_method = cv2.INTER_LANCZOS4
            use_remap = True
        elif interpolation == "spline3":
            interpolation_method = scipy.interpolate.RectBivariateSpline
            use_bivariate_spline = True
        else:
            raise ValueError(f"Invalid interpolation method for METHOD 1: {interpolation}. Available methods: 'linear', 'nearest', 'cubic', 'area', 'lanczos4', 'spline3'.")
    elif method == "distort":
        if interpolation == "linear":
            interpolation_method = scipy.interpolate.LinearNDInterpolator
        elif interpolation == "nearest":
            interpolation_method = scipy.interpolate.NearestNDInterpolator
        elif interpolation == "clough":
            interpolation_method = scipy.interpolate.CloughTocher2DInterpolator
        else:
            raise ValueError(f"Invalid interpolation method for METHOD 2: {interpolation}. Available methods: 'linear', 'nearest', 'clough'.")
    
    
    # Construct the pixel points in the image coordinate system
    height, width = src.shape[:2]
    pixel_points = numpy.indices((height, width), dtype=numpy.float64) # shape (2, H, W)
    
    image_points = pixel_points.copy() # shape (2, H, W) [2, Y, X]
    image_points = image_points.reshape(2, -1).T  # shape (2, H, W) [2, Y, X] -> shape (Npoints, 2) [Y, X]
    image_points = image_points[:, [1, 0]]  # Switch to [X, Y] format, shape (Npoints, 2) [Y, X] -> shape (Npoints, 2) [X, Y]
    

    # ======================================================================================
    # ================================ WITH METHOD 1: UNDISTORT ============================
    # ======================================================================================
    if method == "undistort":

        # Undistort the pixel points using the distortion model
        normalized_points, _, _ = intrinsic._inverse_transform(image_points, dx=False, dp=False) # shape (Npoints, 2) [X', Y'] -> shape (Npoints, 2) [X'/Z, Y'/Z]
        undistorted_points, _, _ = distortion._inverse_transform(normalized_points, dx=False, dp=False, **kwargs) # shape (Npoints, 2) [X'/Z, Y'/Z] -> shape (Npoints, 2) [X/Z, Y/Z]
        undistorted_image_points, _, _ = intrinsic._transform(undistorted_points, dx=False, dp=False) # shape (Npoints, 2) [X/Z, Y/Z] -> shape (Npoints, 2) [X, Y]

        # Reshape the undistorted image points for cv2.remap
        undistorted_image_points = undistorted_image_points[:, [1, 0]]  # Switch to [Y, X] format, shape (Npoints, 2) [X', Y'] -> shape (Npoints, 2) [Y', X']
        undistorted_pixel_points = undistorted_image_points.T.reshape(2, height, width) # shape (Npoints, 2) [Y', X'] -> shape (2, H, W) [Y', X']

        if use_remap:
            # Create the map for cv2.remap
            # dst(x, y) = src(map_x(x, y), map_y(x, y))
            map_x = undistorted_pixel_points[1, :, :]  # X coordinates, shape (H, W)
            map_y = undistorted_pixel_points[0, :, :]  # Y coordinates, shape (H, W)

            # Remap the image using OpenCV
            distorted_image = cv2.remap(src, map_x.astype(numpy.float32), map_y.astype(numpy.float32), interpolation=interpolation_method, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            return distorted_image
        
        elif use_bivariate_spline:
            # Create the values and the image (H, W, 1 * [C] * [D])
            values = src.reshape(height, width, -1).astype(numpy.float64) # shape (H, W, 1 * [C] * [D])

            # Initialize the distorted image
            distorted_image = numpy.zeros_like(values, dtype=numpy.float64) # shape (H, W, 1 * [C] * [D])

            # For all image data dimensions, interpolate the undistorted pixel points in the src image
            for i in range(values.shape[-1]):
                # Create the interpolator for the undistorted pixel points
                interp = scipy.interpolate.RectBivariateSpline(
                    numpy.arange(height),
                    numpy.arange(width),
                    values[:, :, i],
                    kx=3, ky=3, s=0 # Spline interpolation with kx=ky=3
                )

                # Create the mask for points that are within the image bounds and finite
                mask = numpy.isfinite(undistorted_pixel_points[0, :, :]) & numpy.isfinite(undistorted_pixel_points[1, :, :]) & (0.0 <= undistorted_pixel_points[0, :, :]) & (undistorted_pixel_points[0, :, :] <= height-1.0) & (0.0 <= undistorted_pixel_points[1, :, :]) & (undistorted_pixel_points[1, :, :] <= width-1.0)

                # Interpolate the pixel points in the distorted image
                result = interp.ev(undistorted_pixel_points[0, mask], undistorted_pixel_points[1, mask])
                distorted_image[mask, i] = result

            # Reshape the distorted image to the original shape
            distorted_image = distorted_image.reshape(height, width, *src.shape[2:]) # (H, W, 1 * [C] * [D]) -> (H, W, [C], [D])
            return distorted_image


    









    # ======================================================================================
    # ================================ WITH METHOD 2: DISTORT ==============================
    # ======================================================================================
    elif method == "distort":

        # Distort the pixel points using the distortion model
        normalized_points, _, _ = intrinsic._inverse_transform(image_points, dx=False, dp=False) # shape (Npoints, 2) [X, Y] -> shape (Npoints, 2) [X/Z, Y/Z]
        distorted_points, _, _ = distortion._transform(normalized_points, dx=False, dp=False, **kwargs) # shape (Npoints, 2) [X/Z, Y/Z] -> shape (Npoints, 2) [X'/Z, Y'/Z]
        distorted_image_points, _, _ = intrinsic._transform(distorted_points, dx=False, dp=False) # shape (Npoints, 2) [X'/Z, Y'/Z] -> shape (Npoints, 2) [X', Y']

        # Reshape the distorted image points for scipy.interpolate.LinearNDInterpolator
        distorted_image_points = distorted_image_points[:, [1, 0]] # Switch to [Y, X] format, shape (Npoints, 2) [X', Y'] -> shape (Npoints, 2) [Y', X']

        # Create the values and the image (H, W, 1 * [C] * [D])
        values = src.reshape(height, width, -1).astype(numpy.float64) # shape (H, W, 1 * [C] * [D])
        values = values.reshape(-1, values.shape[-1])  # shape (H, W, 1 * [C] * [D]) -> shape (Npoints, 1 * [C] * [D])

        # Create the mapping for scipy.interpolate.LinearNDInterpolator
        distorted_image = numpy.zeros_like(values, dtype=numpy.float64)
        image_points = pixel_points.reshape(2, -1).T # shape (2, H, W) [2, Y, X] -> shape (Npoints, 2) [Y', X']

        # For all image data dimensions, interpolate the pixel_points in the cloud of points (distorted_image_points ; values)
        for i in range(values.shape[-1]):
            # Remove the NaN values from the distorted image points and values
            mask = numpy.isfinite(distorted_image_points[:, 0]) & numpy.isfinite(distorted_image_points[:, 1]) & numpy.isfinite(values[:, i])
            distorted_image_points_filtered = distorted_image_points[mask, :]
            values_filtered = values[mask, i]

            # Create the interpolator for the distorted pixel points
            interp = scipy.interpolate.LinearNDInterpolator(distorted_image_points_filtered, values_filtered, fill_value=0.0)

            # Interpolate the pixel points in the distorted image
            result = interp(image_points)
            distorted_image[:, i] = result

        # Reshape the distorted image to the original shape
        distorted_image = distorted_image.reshape(height, width, -1) # (H, W, 1 * [C] * [D])
        distorted_image = distorted_image.reshape(height, width, *src.shape[2:]) # (H, W, 1 * [C] * [D]) -> (H, W, [C], [D])

        return distorted_image
    








    

    