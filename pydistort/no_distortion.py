from typing import Optional, Tuple
import numpy

from .objects.distortion import Distortion

class NoDistortion(Distortion):
    r"""
    Class to represent the empty distortion model where the ``distorted_points`` are equal to the ``normalized_points``.

    The NoDistortionmodel is used when there is no distortion in the camera.
    """
    def __init__(self):
        super().__init__()

    @property
    def Nparams(self) -> int:
        r"""
        Returns the number of parameters for the no distortion model, which is always 0.
        """
        return 0
    
    @property
    def parameters(self) -> None:
        r"""
        Here always returns None since there are no parameters for the no distortion model.
        It cannot be set, as there are no parameters to set.
        """
        return None

    def is_set(self) -> bool:
        r"""
        Returns True, indicating that the no distortion model is always set.
        """
        return True
    
    def _transform(self, normalized_points: numpy.ndarray, *, dx = True, dp = True) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        """
        Where no distortion is applied, the normalized points are returned as distorted points.

        The jacobians for the distortion parameters is an empty array, as there are no parameters to compute the jacobian for.
        The jacobian for the normalized points is set to the identity matrix, as the distorted points are equal to the normalized points.
        
        .. warning::

            This method is not designed to be used directly for the transformation of points.
            No checks are performed on the input points, so it is the user's responsibility to ensure that the input points are valid.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The points in normalized coordinates to be transformed. Shape (Npoints, 2).

        dx : bool, optional
            If True, the jacobian with respect to the normalized points is computed. Default is True

        dp : bool, optional
            If True, the jacobian with respect to the distortion parameters is computed. Default is True

        Returns
        -------
        distorted_points : numpy.ndarray
            The points in distorted coordinates, which are equal to the normalized points. Shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the distorted points with respect to the normalized points. Shape (Npoints, 2, 2) if dx is True, otherwise None.

        jacobian_dp : Optional[numpy.ndarray]
            The jacobian of the distorted points with respect to the distortion parameters if dp is True, otherwise None. Shape (Npoints, 2, Nparams) if dp is True, otherwise None.
        """
        distorted_points = normalized_points.copy() # shape (Npoints, 2)
        jacobian_dx = None # shape (Npoints, 2, 2)
        jacobian_dp = None # shape (Npoints, 2, Nparams)
        if dx:
            jacobian_dx = numpy.zeros((normalized_points.shape[0], 2, 2), dtype=numpy.float64) # shape (Npoints, 2, 2)
            jacobian_dx[:, 0, 0] = 1.0
            jacobian_dx[:, 1, 1] = 1.0
        if dp:
            jacobian_dp = numpy.empty((normalized_points.shape[0], 2, 0), dtype=numpy.float64) # shape (Npoints, 2, 0)
        return distorted_points, jacobian_dx, jacobian_dp
    
    
    def _inverse_transform(self, distorted_points: numpy.ndarray, *, dx = True, dp = True) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        """
        The inverse transform for the no distortion model is the same as the transform, since the distorted points are equal to the normalized points.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The points in distorted coordinates to be transformed. Shape (Npoints, 2).

        dx : bool, optional
            If True, the jacobian with respect to the normalized points is computed. Default is True

        dp : bool, optional
            If True, the jacobian with respect to the distortion parameters is computed. Default is True

        Returns
        -------
        normalized_points : numpy.ndarray
            The points in normalized coordinates, which are equal to the distorted points. Shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the normalized points with respect to the distorted points. Shape (Npoints, 2, 2) if dx is True, otherwise None.

        jacobian_dp : Optional[numpy.ndarray]
            The jacobian of the normalized points with respect to the distortion parameters if dp is True, otherwise None. Shape (Npoints, 2, Nparams) if dp is True, otherwise None.
        """
        return self._transform(distorted_points, dx=dx, dp=dp)