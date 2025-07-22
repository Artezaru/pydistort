from typing import Optional, Tuple, Dict
from numbers import Number
import numpy

from ..core import Extrinsic




class NoExtrinsic(Extrinsic):
    r"""
    Subclass of :class:`pydistort.core.Extrinsic` to represent an empty extrinsic model where the 3D points are assumed to be the same plane as the normalized plane.
    """
    def __init__(self):
        super().__init__()

    @property
    def Nparams(self) -> int:
        r"""
        Returns the number of parameters for the no extrinsic model, which is always 0.
        """
        return 0
    
    @property
    def parameters(self) -> None:
        r"""
        Here always returns None since there are no parameters for the no extrinsic model.
        It cannot be set, as there are no parameters to set.
        """
        return None

    def is_set(self) -> bool:
        r"""
        Returns True, indicating that the no extrinsic model is always set.
        """
        return True
    
    def _transform(self, world_3dpoints: numpy.ndarray, *, dx = False, dp = False, **kwargs) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        """
        Where no extrinsic is applied, the world 3D points points are returned as normalized points (only the x and y coordinates are used).

        The jacobians for the extrinsic parameters is an empty array, as there are no parameters to compute the jacobian for.
        The jacobian for the world points is set to the identity matrix (for x and y) and zero for z, as the world points are equal to the normalized points.
        
        .. warning::

            This method is not designed to be used directly for the transformation of points.
            No checks are performed on the input points, so it is the user's responsibility to ensure that the input points are valid.

        Parameters
        ----------
        world_3dpoints : numpy.ndarray
            The world 3D points in global coordinates to be transformed. Shape (Npoints, 3).

        dx : bool, optional
            If True, the jacobian with respect to the world 3D points is computed. Default is False

        dp : bool, optional
            If True, the jacobian with respect to the extrinsic parameters is computed. Default is False

        Returns
        -------
        normalized_points : numpy.ndarray
            The normalized points in normalized coordinates, which are equal to the x and y componants of the world 3D points. Shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the normalized points with respect to the world 3D points. Shape (Npoints, 2, 3) if dx is True, otherwise None.

        jacobian_dp : Optional[numpy.ndarray]
            The jacobian of the normalized points with respect to the extrinsic parameters if dp is True, otherwise None. Shape (Npoints, 2, Nparams) if dp is True, otherwise None.
        """
        normalized_points = world_3dpoints[:, :2].copy() # shape (Npoints, 2)
        jacobian_dx = None # shape (Npoints, 2, 2)
        jacobian_dp = None # shape (Npoints, 2, Nparams)
        if dx:
            jacobian_dx = numpy.zeros((normalized_points.shape[0], 2, 3), dtype=numpy.float64) # shape (Npoints, 2, 3)
            jacobian_dx[:, 0, 0] = 1.0
            jacobian_dx[:, 1, 1] = 1.0
        if dp:
            jacobian_dp = numpy.empty((normalized_points.shape[0], 2, 0), dtype=numpy.float64) # shape (Npoints, 2, 0)
        return normalized_points, jacobian_dx, jacobian_dp
    
    
    def _inverse_transform(self, normalized_points: numpy.ndarray, *, dx = False, dp = False, **kwargs) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        """
        The inverse transform for the no extrinsic model is the same as the transform, since the normalized_points points are equal to the world 3D points (with z=1 for normalization plane).

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points in normalized coordinates in image coordinates to be transformed. Shape (Npoints, 2).

        dx : bool, optional
            If True, the jacobian with respect to the normalized points is computed. Default is False

        dp : bool, optional
            If True, the jacobian with respect to the extrinsic parameters is computed. Default is False

        Returns
        -------
        world_3dpoints : numpy.ndarray
            The world 3D points in global coordinates, which are equal to the normalized points with z=1. Shape (Npoints, 3).

        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the world 3D points with respect to the normalized points. Shape (Npoints, 3, 2) if dx is True, otherwise None.

        jacobian_dp : Optional[numpy.ndarray]
            The jacobian of the world 3D points with respect to the extrinsic parameters if dp is True, otherwise None. Shape (Npoints, 2, Nparams) if dp is True, otherwise None.
        """
        world_3dpoints = numpy.empty((normalized_points.shape[0], 3), dtype=numpy.float64) # shape (Npoints, 3)
        world_3dpoints[:, :2] = normalized_points.copy() # copy x and y coordinates
        world_3dpoints[:, 2] = 1.0 # set z coordinate
        jacobian_dx = None # shape (Npoints, 2, 2)
        jacobian_dp = None # shape (Npoints, 2, Nparams)
        if dx:
            jacobian_dx = numpy.zeros((normalized_points.shape[0], 3, 2), dtype=numpy.float64) # shape (Npoints, 3, 2)
            jacobian_dx[:, 0, 0] = 1.0
            jacobian_dx[:, 1, 1] = 1.0
        if dp:
            jacobian_dp = numpy.empty((normalized_points.shape[0], 2, 0), dtype=numpy.float64) # shape (Npoints, 2, 0)
        return normalized_points, jacobian_dx, jacobian_dp


    def _compute_rays(self, normalized_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        Computes the rays in the world coordinate system for the given normalized points.

        A ray is the concatenation of the normalized points with a z-coordinate of 1.0 representing the origin of the ray in the world coordinate system and a direction vector of (0, 0, 1) representing the direction of the ray in the world coordinate system.

        The ray structure is as follows:

        - The first 3 elements are the origin of the ray in the world coordinate system (the normalized points with z=1).
        - The last 3 elements are the direction of the ray in the world coordinate system, which is always (0, 0, 1) for the no extrinsic model. The direction vector is normalized.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points in the camera coordinate system. Shape (Npoints, 2).

        Returns
        -------
        rays : numpy.ndarray
            The rays in the world coordinate system. Shape (Npoints, 6).
        """
        rays = numpy.empty((normalized_points.shape[0], 6), dtype=numpy.float64)
        rays[:, :2] = normalized_points.copy()  # copy x and y coordinates
        rays[:, 2] = 1.0  # set z coordinate to 1
        rays[:, 3] = 0.0  # direction x
        rays[:, 4] = 0.0  # direction y
        rays[:, 5] = 1.0  # direction z
        return rays