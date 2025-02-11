Welcome to pydistort's documentation!
=====================================

``pydistort`` is a package for the analysis of stereo images using the DIC (Digital Image Correlation) method.

In the pinhole camera model, the distortion is represented by a set of coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}`.
The process to correspond a 3D-world point to a 2D-image point is as follows:

1. The 3D-world point is expressed in the camera coordinate system.
2. The 3D-world point is normalized by dividing by the third coordinate.
3. The normalized point is distorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}`.
4. The distorted point is projected onto the image plane using the intrinsic matrix K.

To clarify the various stereo-step, we name the points as follows:

- ``world_point`` is the 3D-world point expressed in the world coordinate system.
- ``camera_point`` is the 3D-world point expressed in the camera coordinate system.
- ``normalized_point`` is the 2D-image point obtained by normalizing the ``camera_point`` by dividing by the third coordinate.
- ``distorted_point`` is the 2D-image point obtained by distorting the ``normalized_point`` using the distortion model.
- ``image_point`` is the 2D-image point obtained by projecting the ``distorted_point`` onto the image plane using the intrinsic matrix K.

This module focuses on the distortion of the ``normalized_point`` to the ``distorted_point``.

The documentation is divided into two main sections:
- The API of the package is described in the :doc:`./api` section.
- The usage of the package is described in the :doc:`./usage` section.

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   ./api
   ./usage

Author
------

The package ``pydistort`` was created by the following author:

- **Name**: Artezaru
- **Email**: artezaru.github@proton.me
- **GitHub**: [Artezaru](https://github.com/Artezaru/pydistort.git)

License
-------

Please cite and refer to the package as mentioned in the License.

``pydistort`` is an open-source project, and contributions are welcome! If you encounter any issues or have feature requests, please submit them on GitHub.

