API Reference
==============

Transformations
-----------------

The package `pydistort` provides a set of tools for camera distortion modeling and correction, primarily using OpenCV.
It includes classes and functions to handle various types of distortion parameters.

The process to correspond a 3D-world point to a 2D-image point is as follows:

1. The ``world_3dpoints`` (:math:`X_W`) are expressed in the camera coordinate system using the rotation and translation matrices to obtain the ``camera_3dpoints`` (:math:`X_C`).
2. The ``camera_3dpoints`` (:math:`X_C`) are normalized by dividing by the third coordinate to obtain the ``normalized_points`` (:math:`x_N`).
3. The ``normalized_points`` (:math:`x_N`) are distorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``distorted_points`` (:math:`x_D`).
4. The ``distorted_points`` (:math:`x_D`) are projected onto the image plane using the intrinsic matrix K to obtain the ``image_points`` (:math:`x_I`).

To processing is performed by the following abstract classes stored in the `pydistort.core` module:

.. toctree::
   :maxdepth: 1
   :caption: Transformation Classes:

   ./api_doc/transform.rst
   ./api_doc/transform_inversion.rst
   ./api_doc/transform_composition.rst
   ./api_doc/extrinsic.rst
   ./api_doc/intrinsic.rst
   ./api_doc/distortion.rst

These classes are designed to handle the transformation of points between different coordinate systems, including intrinsic and extrinsic parameters, as well as distortion models.

Some Intrinsic classes are provided in the package, such as:

.. toctree::
   :maxdepth: 1
   :caption: Intrinsic Models:

   ./api_doc/no_intrinsic.rst
   ./api_doc/cv2_intrinsic.rst
   ./api_doc/skew_intrinsic.rst

Some distortion models are provided in the package, such as:

.. toctree::
   :maxdepth: 1
   :caption: Distortion Models:

   ./api_doc/no_distortion.rst
   ./api_doc/cv2_distortion.rst
   ./api_doc/zernike_distortion.rst

Some extrinsic models are provided in the package, such as:

.. toctree::
   :maxdepth: 1
   :caption: Extrinsic Models:

   ./api_doc/no_extrinsic.rst
   ./api_doc/cv2_extrinsic.rst


Global Functions
-----------------

The package also provides global functions for common tasks related to camera distortion and transformation.
These functions are designed to simplify the process of applying transformations and corrections to camera images.

.. toctree::
   :maxdepth: 1
   :caption: Pydistort Functions:

   ./api_doc/project_points.rst
   ./api_doc/undistort_points.rst
   ./api_doc/undistort_image.rst
   ./api_doc/distort_image.rst
   ./api_doc/throw_rays.rst

Some functions are inspired by the OpenCV functions `cv2.projectPoints` and `cv2.undistortPoints`, but they are designed to work with the `pydistort` package's classes and methods.

.. warning::

   DEPRECATED: The following functions are deprecated and will be removed in a future version. Use the corresponding `pydistort` functions instead.

.. toctree::
   :maxdepth: 1
   :caption: (as OPENCV - deprecated):

   ./api_doc/cv2_project_points.rst
   ./api_doc/cv2_undistort_points.rst
   ./api_doc/cv2_undistort_image.rst
   ./api_doc/cv2_distort_image.rst
 
