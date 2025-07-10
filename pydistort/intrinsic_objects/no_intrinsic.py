from typing import Optional, Tuple, Dict
from numbers import Number
import numpy

from ..core import Intrinsic




class NoIntrinsic(Intrinsic):
    r"""
    Subclass of :class:`pydistort.core.Intrinsic` to represent an empty intrinsic model where the image plane is assumed to be the same as the normalized plane.
    """
    def __init__(self):
        super().__init__()

    @property
    def Nparams(self) -> int:
        r"""
        Returns the number of parameters for the no intrinsic model, which is always 0.
        """
        return 0
    
    @property
    def parameters(self) -> None:
        r"""
        Here always returns None since there are no parameters for the no intrinsic model.
        It cannot be set, as there are no parameters to set.
        """
        return None

    def is_set(self) -> bool:
        r"""
        Returns True, indicating that the no intrinsic model is always set.
        """
        return True
    
    def _transform(self, distorted_points: numpy.ndarray, *, dx = False, dp = False, **kwargs) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        """
        Where no intrinsic is applied, the distorted points are returned as image points.

        The jacobians for the intrinsic parameters is an empty array, as there are no parameters to compute the jacobian for.
        The jacobian for the distorted points is set to the identity matrix, as the image points are equal to the distorted points.
        
        .. warning::

            This method is not designed to be used directly for the transformation of points.
            No checks are performed on the input points, so it is the user's responsibility to ensure that the input points are valid.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted points in normalized coordinates to be transformed. Shape (Npoints, 2).

        dx : bool, optional
            If True, the jacobian with respect to the distorted points is computed. Default is False

        dp : bool, optional
            If True, the jacobian with respect to the intrinsic parameters is computed. Default is False

        Returns
        -------
        image_points : numpy.ndarray
            The image points in image coordinates, which are equal to the distorted points. Shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the image points with respect to the distorted points. Shape (Npoints, 2, 2) if dx is True, otherwise None.

        jacobian_dp : Optional[numpy.ndarray]
            The jacobian of the image points with respect to the intrinsic parameters if dp is True, otherwise None. Shape (Npoints, 2, Nparams) if dp is True, otherwise None.
        """
        image_points = distorted_points.copy() # shape (Npoints, 2)
        jacobian_dx = None # shape (Npoints, 2, 2)
        jacobian_dp = None # shape (Npoints, 2, Nparams)
        if dx:
            jacobian_dx = numpy.zeros((image_points.shape[0], 2, 2), dtype=numpy.float64) # shape (Npoints, 2, 2)
            jacobian_dx[:, 0, 0] = 1.0
            jacobian_dx[:, 1, 1] = 1.0
        if dp:
            jacobian_dp = numpy.empty((image_points.shape[0], 2, 0), dtype=numpy.float64) # shape (Npoints, 2, 0)
        return image_points, jacobian_dx, jacobian_dp
    
    
    def _inverse_transform(self, image_points: numpy.ndarray, *, dx = False, dp = False, **kwargs) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        """
        The inverse transform for the no intrinsic model is the same as the transform, since the image points are equal to the distorted points.

        Parameters
        ----------
        image_points : numpy.ndarray
            The image points in image coordinates to be transformed. Shape (Npoints, 2).

        dx : bool, optional
            If True, the jacobian with respect to the image points is computed. Default is False

        dp : bool, optional
            If True, the jacobian with respect to the intrinsic parameters is computed. Default is False

        Returns
        -------
        distorted_points : numpy.ndarray
            The distorted points in normalized coordinates, which are equal to the image points. Shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the distorted points with respect to the image points. Shape (Npoints, 2, 2) if dx is True, otherwise None.

        jacobian_dp : Optional[numpy.ndarray]
            The jacobian of the distorted points with respect to the intrinsic parameters if dp is True, otherwise None. Shape (Npoints, 2, Nparams) if dp is True, otherwise None.
        """
        return self._transform(image_points, dx=dx, dp=dp)