from typing import Optional, Callable, Any, Dict, Union
from rich import print
import tqdm
import numpy

class Distortion(object):
    r"""
    Represents a distortion model for camera calibration.

    ``Distortion`` is a parent class for different types of distortion models, such as :class:`pydistort.ZernikeDistortion`.

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

    This class deals with the normalized points and the distorted points.

    .. important::

        The subclasses of `Distortion` must implement the following methods:

        - ``_distort``: Distorts a set of normalized image points. It must return the distorted image points.
        - ``_undistort``: Undistorts a set of distorted image points. It must return the normalized image points.
        - ``_jacobian_distort``: Computes the Jacobian matrix of the distortion model for a set of normalized image points. It must return the Jacobian matrix.
        - ``_jacobian_undistort``: Computes the Jacobian matrix of the undistortion model for a set of distorted image points. It must return the Jacobian matrix.
        - ``_normalized_domain_mask``: Computes a mask to remove the points outside the domain of the distortion model. It must return a boolean mask.
        - ``_distorted_domain_mask``: Computes a mask to remove the points outside the domain of the undistortion model. It must return a boolean mask.

    Attributes
    ----------
    coefficients : numpy.ndarray, optional
        The coefficients of the distortion model. Default is `None`.

    Methods
    -------
    distort(normalized_points: numpy.ndarray) -> numpy.ndarray
        Distorts a set of normalized image points.

    undistort(distorted_points: numpy.ndarray) -> numpy.ndarray
        Undistorts a set of distorted image points.
    
    jacobian_distort(normalized_points: numpy.ndarray) -> numpy.ndarray
        Computes the Jacobian matrix of the distortion model for a set of normalized image points.
    
    jacobian_undistort(distorted_points: numpy.ndarray) -> numpy.ndarray
        Computes the Jacobian matrix of the undistortion model for a set of distorted image points.

    """

    def __init__(self, coefficients: Optional[numpy.ndarray] = None, **kwargs) -> None:
        super().__init__()
        self.coefficients = coefficients
        self._kwargs = {}
        # Add the default keyword arguments
        self._add_kwargs(
            undistort_Newton_epsilon=1e-8,
            undistort_Newton_max_iter=100,
            undistort_Newton_gamma_initial=1.0,
            undistort_Newton_Armijo=True,
            undistort_Newton_gamma_divisor=2.0,
            undistort_Newton_gamma_min=1e-4,
            undistort_Newton_Armijo_coeff=numpy.array([1e-4, 0.9]),
            jacobian_distort_Differential_epsilon=1e-8,
            verbose=False,
        )
        # Add the custom keyword arguments
        self._set_kwargs(**kwargs)



    @property
    def coefficients(self) -> numpy.ndarray:
        r"""
        Getter for the coefficients attribute.

        Returns
        -------
        numpy.ndarray
            The coefficients of the distortion model.
        """
        return self._coefficients
    


    @coefficients.setter
    def coefficients(self, coefficients: Optional[numpy.ndarray]) -> None:
        r"""
        Setter for the coefficients attribute.

        Parameters
        ----------
        coefficients : numpy.ndarray
            The coefficients of the distortion model.
        """
        if coefficients is not None and not isinstance(coefficients, numpy.ndarray):
            raise ValueError("The coefficients must be a numpy.ndarray.")
        self._coefficients = coefficients



    # ----------------------------------



    @property
    def kwargs(self) -> Dict[str, Any]:
        return self._kwargs



    def _add_kwargs(self, **kwargs) -> None:
        self._kwargs.update(kwargs)


    
    def _set_kwargs(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key not in self._kwargs.keys():
                raise ValueError(f"Invalid key: {key}.")
            else:
                self._kwargs[key] = value



    def local_kwargs(self, **kwargs) -> Dict[str, Any]:
        r"""
        Returns a dictionary with the local keyword arguments.
        The local keyword arguments are used to store temporary values during the computation.

        Returns
        -------
        Dict[str, Any]
            The local keyword arguments.
        """
        # Copy the kwargs
        current_kwargs = self.kwargs.copy()

        # Use the setter to check the inputs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Copy the local kwargs
        local_kwargs = self.kwargs.copy()

        # Restore the original kwargs
        self._kwargs = current_kwargs

        return local_kwargs



    # ----------------------------------
    


    @property
    def verbose(self) -> bool:
        return self.kwargs["verbose"]

    @verbose.setter
    def verbose(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("The verbose parameter must be a boolean.")
        self._set_kwargs(verbose=value)

    

    @property
    def undistort_Newton_epsilon(self) -> float:
        return self.kwargs["undistort_Newton_epsilon"]

    @undistort_Newton_epsilon.setter
    def undistort_Newton_epsilon(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("The undistort_Newton_epsilon parameter must be a number.")
        if value <= 0:
            raise ValueError("The undistort_Newton_epsilon parameter must be positive.")
        self._set_kwargs(undistort_Newton_epsilon=value)
    


    @property
    def undistort_Newton_max_iter(self) -> int:
        return self.kwargs["undistort_Newton_max_iter"]
    
    @undistort_Newton_max_iter.setter
    def undistort_Newton_max_iter(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError("The undistort_Newton_max_iter parameter must be an integer.")
        if value <= 0:
            raise ValueError("The undistort_Newton_max_iter parameter must be positive.")
        self._set_kwargs(undistort_Newton_max_iter=value)
    


    @property
    def undistort_Newton_gamma_initial(self) -> float:
        return self.kwargs["undistort_Newton_gamma_initial"]

    @undistort_Newton_gamma_initial.setter
    def undistort_Newton_gamma_initial(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("The undistort_Newton_gamma_initial parameter must be a number.")
        if value <= 0:
            raise ValueError("The undistort_Newton_gamma_initial parameter must be positive.")
        self._set_kwargs(undistort_Newton_gamma_initial=value)


    
    @property
    def undistort_Newton_Armijo(self) -> bool:
        return self.kwargs["undistort_Newton_Armijo"]

    @undistort_Newton_Armijo.setter
    def undistort_Newton_Armijo(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("The undistort_Newton_Armijo parameter must be a boolean.")
        self._set_kwargs(undistort_Newton_Armijo=value)

    

    @property
    def undistort_Newton_gamma_divisor(self) -> float:
        return self.kwargs["undistort_Newton_gamma_divisor"]
    
    @undistort_Newton_gamma_divisor.setter
    def undistort_Newton_gamma_divisor(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("The undistort_Newton_gamma_divisor parameter must be a number.")
        if value <= 1:
            raise ValueError("The undistort_Newton_gamma_divisor parameter must be greater than 1.")
        self._set_kwargs(undistort_Newton_gamma_divisor=value)
    


    @property
    def undistort_Newton_gamma_min(self) -> float:
        return self.kwargs["undistort_Newton_gamma_min"]

    @undistort_Newton_gamma_min.setter
    def undistort_Newton_gamma_min(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("The undistort_Newton_gamma_min parameter must be a number.")
        if value <= 0:
            raise ValueError("The undistort_Newton_gamma_min parameter must be positive.")
        self._set_kwargs(undistort_Newton_gamma_min=value)



    @property
    def undistort_Newton_Armijo_coeff(self) -> numpy.ndarray:
        return self.kwargs["undistort_Newton_Armijo_coeff"]

    @undistort_Newton_Armijo_coeff.setter
    def undistort_Newton_Armijo_coeff(self, value: Union[list, tuple, numpy.ndarray]) -> None:
        if not isinstance(value, (list, tuple, numpy.ndarray)):
            raise ValueError("The undistort_Newton_Armijo_coeff parameter must be a list, tuple, or numpy.ndarray.")
        if len(value) != 2:
            raise ValueError("The undistort_Newton_Armijo_coeff parameter must have two elements.")
        if not all(isinstance(v, (int, float)) for v in value):
            raise ValueError("The undistort_Newton_Armijo_coeff parameter must contain numbers.")
        if not 0 < value[0] < value[1] < 1:
            raise ValueError("The undistort_Newton_Armijo_coeff parameter must respect 0 < c1 < c2 < 1.")
        self._set_kwargs(undistort_Newton_Armijo_coeff=numpy.array(value))

    

    @property
    def jacobian_distort_Differential_epsilon(self) -> float:
        return self.kwargs["jacobian_distort_Differential_epsilon"]
    
    @jacobian_distort_Differential_epsilon.setter
    def jacobian_distort_Differential_epsilon(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("The jacobian_distort_Differential_epsilon parameter must be a number.")
        if value <= 0:
            raise ValueError("The jacobian_distort_Differential_epsilon parameter must be positive.")
        self._set_kwargs(jacobian_distort_Differential_epsilon=value)
    


    # ----------------------------------



    def check_array_2D_Npoints(self, array: numpy.ndarray) -> None:
        r"""
        Checks if the input array is a numpy.ndarray of shape (2, Npoints).

        Parameters
        ----------
        array : numpy.ndarray
            The array to check.

        Raises
        ------
        TypeError
            If the array is not a numpy.ndarray.
        ValueError
            If the array is not a numpy.ndarray of shape (2, Npoints).
        """
        if not isinstance(array, numpy.ndarray):
            raise TypeError("The array must be a numpy.ndarray.")
        if array.ndim != 2 or array.shape[0] != 2:
            raise ValueError("The array must be a numpy.ndarray of shape (2, Npoints).")



    def nan_1D_mask(self, array: numpy.ndarray, axis: int = 0) -> numpy.ndarray:
        r"""
        Creates a mask for nan values in the array along a specified axis.

        The array is in the form (N0, N1, ..., Nk) where k >= 2.
        If the given axis is n, the output mask is a 1D numpy array with shape (Nn,).

        If mask[i] is True, the i-th orthogonal slice along the axis contains nan values.

        Parameters
        ----------
        array : numpy.ndarray
            The input array.
        
        axis : Optional[int], optional
            The axis along which the mask is created. The default is 0.
        
        Returns
        -------
        numpy.ndarray
            The mask.

        Raises
        ------
        ValueError
            If the input array is not a 2D numpy array.
        """
        # Check the input parameters
        if not isinstance(array, numpy.ndarray):
            raise TypeError("Array must be a numpy array.")
        if not isinstance(axis, int):
            raise TypeError("Axis must be an integer.")
        if not axis in range(array.ndim):
            raise ValueError("Invalid axis.")

        # Create the mask
        return numpy.any(numpy.isnan(array), axis=tuple(i for i in range(array.ndim) if i != axis))



    def _private_execute(self, function: Callable, points: numpy.ndarray, compute_type: str = "point", **kwargs) -> numpy.ndarray:
        r"""
        Generic method to synthesise the following methods: distort, undistort, jacobian_distort, jacobian_undistort.

        This method is used to avoid code duplication in the distort, undistort, jacobian_distort, jacobian_undistort methods.

        According the shape of the input points, the method will return the following:
        - If the input points are None, the method will return None.
        - If the input points are a numpy.ndarray of shape (2, ), the method will return the processed points with shape (2, ) or the processed jacboian with shape (2, 2).
        - If the input points are a numpy.ndarray of shape (2, ``Npoints``), the method will return the processed points with shape (2, ``Npoints``) or the processed jacboian with shape (2, 2, ``Npoints``).

        To deal with homogeneous coordinates, the input points can be a numpy.ndarray of shape (``Ndim``, ``Npoints``) where ``Ndim`` >= 2 is the number of dimensions.
        In this case, the processed points will be a numpy.ndarray of shape (``Ndim``, ``Npoints``) or the processed jacboian will have the shape (``Ndim``, ``Ndim``, ``Npoints``).
        Only the first two dimensions are processed.

        Parameters
        ----------
        function : Callable
            The function to execute.

        points : numpy.ndarray
            The points to process.

        compute_type : str
            The type of computation to perform. It can be "point" or "jacobian".
        
        kwargs : dict
            Additional keyword arguments for the function.

        Returns
        -------
        numpy.ndarray
            The processed points.

        Raises
        ------
        ValueError
            If the points are not a numpy.ndarray with a correct shape.
        """
        # Check the input points
        if not (points is None or isinstance(points, numpy.ndarray)):
            raise ValueError("The points must be a numpy.ndarray.")
        if not isinstance(compute_type, str) or not compute_type in ["point", "jacobian"]:
            raise ValueError("The compute_type must be a string and must be 'point' or 'jacobian'.")
        
        # Case 0. The input points are None
        if points is None:
            return None

        # Case 1. The input points are a single point
        if points.ndim == 1:
            Ndim = points.shape[0]

            # Case A. The shape is not correct
            if Ndim < 2:
                raise ValueError("The shape of the points must be (2,) or (Ndim,) where Ndim >= 2.")
            
            # Case B. The shape is exactly (2,)
            if Ndim == 2:
                return function(points.reshape((2, 1)), **kwargs).reshape((2,))

            # Case C. The shape is (Ndim,) where Ndim > 2
            if Ndim > 2:
                # Case a. The computation is for a point
                if compute_type == "point":
                    processed_first_two = function(points[:2].reshape((2, 1)), **kwargs).reshape((2,))
                    return numpy.vstack((processed_first_two, points[2:]))

                # Case b. The computation is for a jacobian
                if compute_type == "jacobian":
                    jacobian_first_two = function(points[:2].reshape((2, 1)), **kwargs).reshape((2, 2))
                    jacobian = numpy.eye(Ndim)
                    jacobian[:2, :2] = jacobian_first_two
                    return jacobian
            
        # Case 2. The input points are a set of points
        if points.ndim == 2:
            Ndim, Npoints = points.shape

            # Case A. The shape is not correct
            if Ndim < 2:
                raise ValueError("The shape of the points must be (2, Npoints) or (Ndim, Npoints) where Ndim >= 2.")
            if Npoints < 1:
                raise ValueError("The number of points Npoints must be at least 1.")
            
            # Case B. The shape is (2, Npoints)
            if Ndim == 2:
                return function(points, **kwargs)

            # Case C. The shape is (Ndim, Npoints) where Ndim > 2
            if Ndim > 2:
                # Case a. The computation is for a point
                if compute_type == "point":
                    processed_first_two = function(points[:2, :].reshape((2, 1)), **kwargs).reshape((2,))
                    return numpy.vstack((processed_first_two, points[2:, :]))

                # Case b. The computation is for a jacobian
                if compute_type == "jacobian":
                    jacobian_first_two = function(points[:2, :].reshape((2, 1)), **kwargs).reshape((2, 2))
                    jacobian = numpy.eye(Ndim)[:, :, numpy.newaxis]
                    jacobian = numpy.tile(jacobian, (1, 1, Npoints))
                    jacobian[:2, :2, :] = jacobian_first_two
                    return jacobian
        
        # Case 3. The shape of the points is not correct
        raise ValueError("The shape of the points must be (2,) or (2, Npoints) or (Ndim, Npoints) where Ndim >= 2.")



    def distort(self, normalized_points: Optional[numpy.ndarray] = None, **kwargs) -> numpy.ndarray:
        r"""
        Distorts a set of normalized image points.

        According the shape of the input points, the method will return the following:
        - If the input normalized points are None, the method will return None.
        - If the input normalized points are a numpy.ndarray of shape (2, ), the method will return the distorted image points with shape (2, ).
        - If the input normalized points are a numpy.ndarray of shape (2, ``Npoints``), the method will return the distorted image points with shape (2, ``Npoints``).

        To deal with homogeneous coordinates, the normalized points can be a numpy.ndarray of shape (``Ndim``, ``Npoints``) where ``Ndim`` >= 2 is the number of dimensions.
        In this case, the distorted points will be a numpy.ndarray of shape (``Ndim``, ``Npoints``).
        Only the first two dimensions are distorted by the distortion model.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized image points to distort.
        
        kwargs : dict
            Additional keyword arguments for the distortion model.

        Returns
        -------
        numpy.ndarray
            The distorted image points.

        Raises
        ------
        ValueError
            If the points are not a numpy.ndarray with a correct shape.
        """
        return self._private_execute(self._distort, normalized_points, compute_type="point", **kwargs)



    def _distort(self, normalized_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        Distorts a set of normalized image points.

        This method must be implemented by the child classes.

        The input normalized points are an array of shape (2, ``Npoints``) where ``Npoints`` is the number of points.
        The output distorted points must have the same shape (2, ``Npoints``).

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized image points to distort.

        args : tuple
            Additional arguments for the distortion model.
        
        kwargs : dict
            Additional keyword arguments for the distortion model.

        Returns
        -------
        numpy.ndarray
            The distorted image points.
        """
        raise NotImplementedError("The _distort method must be implemented by the child classes.")



    def undistort(self, distorted_points: Optional[numpy.ndarray] = None, **kwargs) -> numpy.ndarray:
        r"""
        Undistorts a set of distorted image points.

        According the shape of the input points, the method will return the following:
        - If the input distorted points are None, the method will return None.
        - If the input distorted points are a numpy.ndarray of shape (2, ), the method will return the normalized image points with shape (2, ).
        - If the input distorted points are a numpy.ndarray of shape (2, ``Npoints``), the method will return the normalized image points with shape (2, ``Npoints``).

        To deal with homogeneous coordinates, the distorted points can be a numpy.ndarray of shape (``Ndim``, ``Npoints``) where ``Ndim`` >= 2 is the number of dimensions.
        In this case, the normalized points will be a numpy.ndarray of shape (``Ndim``, ``Npoints``).
        Only the first two dimensions are undistorted by the distortion model.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted image points to undistort.
        
        kwargs : dict
            Additional keyword arguments for the distortion model.

        Returns
        -------
        numpy.ndarray
            The normalized image points.

        Raises
        ------
        ValueError
            If the points are not a numpy.ndarray with a correct shape.
        """
        return self._private_execute(self._undistort, distorted_points, compute_type="point", **kwargs)


    
    def _undistort(self, distorted_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        Undistorts a set of distorted image points.

        This method must be implemented by the child classes.

        The input distorted points are an array of shape (2, ``Npoints``) where ``Npoints`` is the number of points.
        The output normalized points must have the same shape (2, ``Npoints``).

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted image points to undistort.

        kwargs : dict
            Additional keyword arguments for the distortion model.

        Returns
        -------
        numpy.ndarray
            The normalized image points.
        """
        raise NotImplementedError("The _undistort method must be implemented by the child classes.")


    
    def jacobian_distort(self, normalized_points: Optional[numpy.ndarray] = None, **kwargs) -> numpy.ndarray:
        r"""
        Computes the Jacobian matrix of the distortion model for a set of normalized image points.

        The Jacobian matrix is the matrix of the partial derivatives of the distortion model with respect to the normalized image points.

        According the shape of the input points, the method will return the following:
        - If the input normalized points are None, the method will return None.
        - If the input normalized points are a numpy.ndarray of shape (2, ), the method will return the Jacobian matrix with shape (2, 2).
        - If the input normalized points are a numpy.ndarray of shape (2, ``Npoints``), the method will return the Jacobian matrix with shape (2, 2, ``Npoints``).

        To deal with homogeneous coordinates, the normalized points can be a numpy.ndarray of shape (``Ndim``, ``Npoints``) where ``Ndim`` >= 2 is the number of dimensions.
        In this case, the Jacobian matrix will have the shape (``Ndim``, ``Ndim``, ``Npoints``).
        Only the coordinates [0, 0], [0, 1], [1, 0], [1, 1] of the Jacobian matrix are computed.
        The other coordinates are set to 1 in the diagonal and 0 in the off-diagonal.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized image points for which to compute the Jacobian matrix.

        kwargs : dict
            Additional keyword arguments for the distortion model.

        Returns
        -------
        numpy.ndarray
            The Jacobian matrix of the distortion model.

        Raises
        ------
        ValueError
            If the points are not a numpy.ndarray with a correct shape.
        """
        return self._private_execute(self._jacobian_distort, normalized_points, compute_type="jacobian", **kwargs)



    def _jacobian_distort(self, normalized_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        Computes the Jacobian matrix of the distortion model for a set of normalized image points.

        This method must be implemented by the child classes.

        The input normalized points are an array of shape (2, ``Npoints``) where ``Npoints`` is the number of points.
        The output Jacobian matrix must have the shape (2, 2, ``Npoints``).

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized image points for which to compute the Jacobian matrix.
        
        kwargs : dict
            Additional keyword arguments for the distortion model.

        Returns
        -------
        numpy.ndarray
            The Jacobian matrix of the distortion model.
        """
        raise NotImplementedError("The _jacobian_distort method must be implemented by the child classes.")

    

    def jacobian_undistort(self, distorted_points: Optional[numpy.ndarray] = None, **kwargs) -> numpy.ndarray:
        r"""
        Computes the Jacobian matrix of the undistortion model for a set of distorted image points.

        The Jacobian matrix is the matrix of the partial derivatives of the undistortion model with respect to the distorted image points.

        According the shape of the input points, the method will return the following:
        - If the input distorted points are None, the method will return None.
        - If the input distorted points are a numpy.ndarray of shape (2, ), the method will return the Jacobian matrix with shape (2, 2).
        - If the input distorted points are a numpy.ndarray of shape (2, ``Npoints``), the method will return the Jacobian matrix with shape (2, 2, ``Npoints``).

        To deal with homogeneous coordinates, the distorted points can be a numpy.ndarray of shape (``Ndim``, ``Npoints``) where ``Ndim`` >= 2 is the number of dimensions.
        In this case, the Jacobian matrix will have the shape (``Ndim``, ``Ndim``, ``Npoints``).
        Only the coordinates [0, 0], [0, 1], [1, 0], [1, 1] of the Jacobian matrix are computed.
        The other coordinates are set to 1 in the diagonal and 0 in the off-diagonal.

        The jacobian_undistort can be processed using the following formula:

        .. math::

            J_{f^{-1}} = \left( J_{f} \circ f^{-1} \right)^{-1}

        where :math:`J_{f}` is the Jacobian matrix of the distortion model and :math:`f^{-1}` is the undistortion model.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted image points for which to compute the Jacobian matrix.

        kwargs : dict
            Additional keyword arguments for the distortion model

        Returns
        -------
        numpy.ndarray
            The Jacobian matrix of the undistortion model.

        Raises
        ------
        ValueError
            If the points are not a numpy.ndarray with a correct shape.
        """
        return self._private_execute(self._jacobian_undistort, distorted_points, compute_type="jacobian", **kwargs)


    
    def _jacobian_undistort(self, distorted_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        Computes the Jacobian matrix of the undistortion model for a set of distorted image points.

        The input distorted points are an array of shape (2, ``Npoints``) where ``Npoints`` is the number of points.
        The output Jacobian matrix must have the shape (2, 2, ``Npoints``).

        The jacobian_undistort can be processed using the following formula:

        .. math::

            J_{f^{-1}} = \left( J_{f} \circ f^{-1} \right)^{-1}

        where :math:`J_{f}` is the Jacobian matrix of the distortion model and :math:`f^{-1}` is the undistortion model.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted image points for which to compute the Jacobian matrix.

        kwargs : dict
            Additional keyword arguments for the distortion model.

        Returns
        -------
        numpy.ndarray
            The Jacobian matrix of the undistortion model.
        """
        raise NotImplementedError("The _jacobian_undistort method must be implemented by the child classes.")



    def _normalized_domain_mask(self, normalized_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        Computes a mask to remove the points outside the domain of the distortion model.

        The input normalized points are an array of shape (2, ``Npoints``) where ``Npoints`` is the number of points.
        The output mask is a boolean array of shape (``Npoints``, ) where the points outside the domain are set to False.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized image points for which to compute the mask.

        kwargs : dict
            Additional keyword arguments for the distortion model

        Returns
        -------
        numpy.ndarray
            The mask to remove the points outside the domain of the distortion model.
        """
        raise NotImplementedError("The _normalized_domain_mask method must be implemented by the child classes.")



    def _distorted_domain_mask(self, distorted_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        Computes a mask to remove the points outside the domain of the undistortion model.

        The input distorted points are an array of shape (2, ``Npoints``) where ``Npoints`` is the number of points.
        The output mask is a boolean array of shape (``Npoints``, ) where the points outside the domain are set to False.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted image points for which to compute the mask.
        
        kwargs : dict
            Additional keyword arguments for the distortion model

        Returns
        -------
        numpy.ndarray
            The mask to remove the points outside the domain of the undistortion model.
        """
        raise NotImplementedError("The _distorted_domain_mask method must be implemented by the child classes.")



    # ----------------------------------


    
    def undistort_by_Newton(self, distorted_points: numpy.ndarray, undistorted_guess: Optional[numpy.ndarray] = None, **kwargs) -> numpy.ndarray:
        r"""
        
        .. seealso::

            - :func:`pysdic.distortion.Distortion._undistort`
        
        The algorithm minimizes the residual:

        .. math::

            \text{residual}(x_{\text{nor}}) = \Pi{x_{\text{nor}}} = || x_{\text{dis}} - \text{distort}(x_{\text{nor}}) ||^2

        Where :math:`x_{\text{nor}}` are the normalized points, :math:`x_{\text{dis}}` are the distorted points, and :math:`\Pi{x_{\text{nor}}}` is the residual.

        To minimize the residual, we proceed iteratively using Newton's method:

        .. math::

            x_{\text{nor}}^{(k+1)} = x_{\text{nor}}^{(k)} - \gamma^{(k)} H_{\Pi}^{-1} \nabla_{\Pi} 

        Where :math:`H_{\Pi}` is the Hessian matrix of the residual and :math:`\nabla_{\Pi}` is the gradient of the residual and :math:`\gamma^{(k)}` is the step size.
        The direction of the step is given by :math:`d^{(k)} = - H_{\Pi}^{-1} \nabla_{\Pi}`.

        The gradient of the residual can be expressed using the Jacobian matrix of the distortion function:

        .. math::

            \nabla_{\Pi} = - 2 J_{\text{distort}}^T (x_{\text{nor}}) (x_{\text{dis}} - \text{distort}(x_{\text{nor}}))

        The Hessian matrix of the residual can be expressed using the Jacobian matrix of the distortion function:
        
        .. math:: 

            H_{\Pi} = 2 J_{\text{distort}}^T (x_{\text{nor}}) J_{\text{distort}} (x_{\text{nor}}) - 2 \frac{\partial J_{\text{distort}}}{\partial x_{\text{nor}}} (x_{\text{nor}}) (x_{\text{dis}} - \text{distort}(x_{\text{nor}}))

        However, the math:`(x_{\text{dis}} - \text{distort}(x_{\text{nor}}))` is close to zero when the iteration converges, so we can approximate the Hessian matrix as:

        .. math::

            H_{\Pi} \approx 2 J_{\text{distort}}^T (x_{\text{nor}}) J_{\text{distort}} (x_{\text{nor}})

        To find the step size, we start from gamma = ``undistort_Newton_gamma_initial`` and we divide by ``undistort_Newton_gamma_divisor`` until the residual decreases.
        The step size is limited by ``undistort_Newton_gamma_min``.
        The Armijo condition is used to find the step size.

        .. math::
        
            \Pi(x_{\text{nor}}^{(k)} + \gamma^{(k)} d^{(k)}) \leq \Pi(x_{\text{nor}}^{(k)}) + c_1 \gamma^{(k)} {d^{(k)}}^T \nabla_{\Pi(x_{\text{nor}}^{(k)})}

        .. math::

            c_2 {d^{(k)}}^T \nabla_{\Pi(x_{\text{nor}}^{(k)})} \geq {d^{(k)}}^T \nabla_{\Pi(x_{\text{nor}}^{(k)} + \gamma^{(k)} d^{(k)})}

        where :math:`c_1` and :math:`c_2` are the Armijo constants such that :math:`0 < c_1 < c_2 < 1`.

        We proceed with the iteration until the residual is below a certain threshold or until the maximum number of iterations is reached.
        The threshold is set by the ``undistort_Newton_epsilon`` parameter and the maximum number of iterations is set by the ``undistort_Newton_max_iter`` parameter.

        If a distorted point is outside distortion domain, the normalized point is set to numpy.nan.

        If a point comes out of the normalized domain, the normalized point is set to numpy.nan.
        If a point is blocked by singularities (e.g., the Hessian matrix is not invertible), the normalized point is set to numpy.nan.

        .. seealso::

            - :func:`pysdic.distortion.Distortion._normalized_domain_mask`
            - :func:`pysdic.distortion.Distortion._distorted_domain_mask`

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted points (shape: `(2, Npoints)`).

        undistorted_guess : numpy.ndarray, optional
            The initial guess for the undistorted points. Default is `None`.
            If given, it must have the same shape as the distorted points.

        kwargs : dict
            Additional keyword arguments for the distortion model
        
        Returns
        -------
        numpy.ndarray
            The undistorted (normalized) points (shape: `(2, Npoints)`).

        Raises
        ------
        TypeError
            If the distorted points are not a numpy array.
        ValueError
            If the shape of the distorted points is not `(2, Npoints)` or if the points are outside the unit circle or if the distorted points contain NaN values.
        """
        # To clear the understanding of the algorithm, we will use the following notations [WARNING]:
        #
        # - targeted_points: the distorted points (x_{\text{dis}}) given as input
        # - undistorted_points: the undistorted points (x_{\text{nor}}) to find
        # - distorted_points: the distorted points (x_{\text{dis}}) computed from the undistorted points

        targeted_points = distorted_points

        # Check the input parameters
        self.check_array_2D_Npoints(targeted_points)
        if undistorted_guess is not None:
            self.check_array_2D_Npoints(undistorted_guess)
            if undistorted_guess.shape != targeted_points.shape:
                raise ValueError("The initial guess must have the same shape as the input distorted points.")

        # Get the kwargs parameters
        kwargs = self.local_kwargs(**kwargs)

        # Extract the number of points
        Npoints = targeted_points.shape[1]

        # Not considering points outside the distorted domain
        distorted_domain_mask = self._distorted_domain_mask(targeted_points, **kwargs)
        nan_mask = self.nan_1D_mask(targeted_points, axis=1)
        valid_mask = numpy.logical_and(distorted_domain_mask, numpy.logical_not(nan_mask))

        if not numpy.all(valid_mask):
            # Print a warning message
            if kwargs["verbose"]:
                Nout = Npoints - numpy.sum(valid_mask)
                print(f"[bold red]Warning:[/bold red]: {Nout} distorted points are outside the distortion domain.")

            # Recursively call the function for the valid points
            undistorted_points = numpy.full((2, Npoints), numpy.nan)
            if undistorted_guess is not None:
                undistorted_points[:, valid_mask] = self.undistort_by_Newton(targeted_points[:, valid_mask], undistorted_guess[:, valid_mask], **kwargs)
            else:
                undistorted_points[:, valid_mask] = self.undistort_by_Newton(targeted_points[:, valid_mask], **kwargs)
            return undistorted_points

        # ----------------------------------

        def compute_residual(distorted_points, targeted_points):
            return numpy.linalg.norm(targeted_points - distorted_points, axis=0)**2

        def create_is_converged_mask(points, targeted_points, epsilon, use_distorted=False, **kwargs):
            if use_distorted: 
                residual = compute_residual(points, targeted_points)
            else:
                residual = compute_residual(self._distort(points, **kwargs), targeted_points)
            return residual < epsilon

        def create_is_singular_mask(hessian):
            return numpy.isclose(numpy.linalg.det(hessian.transpose(2, 0, 1)), 0, atol=1e-8)

        def create_in_course_mask(is_converged_mask, is_nan_mask, is_singular_mask, is_in_domain_mask):
            return numpy.logical_and(numpy.logical_and(numpy.logical_and(numpy.logical_not(is_converged_mask), numpy.logical_not(is_nan_mask)), numpy.logical_not(is_singular_mask)), is_in_domain_mask)

        # ----------------------------------

        # Initialize the normalized points
        if undistorted_guess is not None:
            undistorted_points = undistorted_guess.copy()
        else:
            undistorted_points = targeted_points.copy()

        # Initialize the mask
        is_converged_mask = create_is_converged_mask(undistorted_points, targeted_points, kwargs["undistort_Newton_epsilon"], use_distorted=False, **kwargs)
        is_nan_mask = self.nan_1D_mask(undistorted_points, axis=1)
        is_singular_mask = numpy.full(Npoints, False)
        is_in_domain_mask = self._normalized_domain_mask(undistorted_points, **kwargs)
        in_course_mask = create_in_course_mask(is_converged_mask, is_nan_mask, is_singular_mask, is_in_domain_mask)

        Newton_iter = 0
        with tqdm.tqdm(
            total = kwargs["undistort_Newton_max_iter"],
            desc = f"Undistort Newton's method - (max iteration: {kwargs['undistort_Newton_max_iter']})",
            disable = not kwargs["verbose"]
            ) as pbar:

            while numpy.any(in_course_mask) and Newton_iter < kwargs['undistort_Newton_max_iter']:
                # Compute the number of points in course for the Newton's method
                NIC = numpy.sum(in_course_mask)

                # Extract the points in course (IC) with shape (2, NIC)
                undistorted_IC = undistorted_points[:, in_course_mask]
                targueted_IC = targeted_points[:, in_course_mask]
                distorted_IC = self._distort(undistorted_IC, **kwargs)

                # Compute the residual (shape: (NIC,))
                residual_IC = compute_residual(distorted_IC, targueted_IC)

                # Compute the jacobian matrix of the distortion model (shape: (2, 2, NIC))
                jacobian = self._jacobian_distort(undistorted_IC, **kwargs)

                # Compute the gradient of the residual (shape: (2, NIC))
                gradient = - 2 * numpy.einsum("jik, jk->ik", jacobian, targueted_IC - distorted_IC)

                # Compute the Hessian matrix of the residual (shape: (2, 2, NIC))
                hessian = 2 * numpy.einsum("jik, jlk->ilk", jacobian, jacobian)

                # Update the singular matrix mask
                is_singular_mask[in_course_mask] = create_is_singular_mask(hessian)
                continue_mask = numpy.logical_not(is_singular_mask[in_course_mask]) 
                NCONT = numpy.sum(continue_mask)

                # Reducing the size of the array for consistency. Notted (CONT)
                undistorted_CONT = undistorted_IC[:, continue_mask] # shape: (2, NCONT)
                targueted_CONT = targueted_IC[:, continue_mask] # shape: (2, NCONT)
                distorted_CONT = distorted_IC[:, continue_mask] # shape: (2, NCONT)
                residual_CONT = residual_IC[continue_mask] # shape: (NCONT,)
                jacobian_CONT = jacobian[:, :, continue_mask] # shape: (2, 2, NCONT)
                gradient_CONT = gradient[:, continue_mask] # shape: (2, NCONT)
                hessian_CONT = hessian[:, :, continue_mask] # shape: (2, 2, NCONT)

                # Invert the Hessian matrix of the residual (shape: (2, 2, NCONT))
                hessian_inv_CONT = numpy.linalg.inv(hessian_CONT.transpose(2, 0, 1)).transpose(1, 2, 0)

                # Compute the direction of the step (shape: (2, NCONT))
                direction = - numpy.einsum("ijk, jk->ik", hessian_inv_CONT, gradient_CONT)

                # Initialize the update of the undistorted points (shape: (2, NCONT))
                update_CONT = numpy.zeros((2, NCONT))

                # Compute the step size with the Armijo condition
                if kwargs["undistort_Newton_Armijo"]:
                    gamma = numpy.full((NCONT,), kwargs["undistort_Newton_gamma_initial"]*kwargs["undistort_Newton_gamma_divisor"]) # multiply by the divisor in order than gamma/divisor = gamma_initial
                    Armijo_mask = numpy.full((NCONT,), True)
                    c1, c2 = kwargs["undistort_Newton_Armijo_coeff"][0], kwargs["undistort_Newton_Armijo_coeff"][1]
                    while numpy.any(Armijo_mask):
                        # Number of points in course for the Armijo condition
                        NAC = numpy.sum(numpy.logical_not(Armijo_mask))

                        # update the step size
                        gamma[Armijo_mask] /= kwargs["undistort_Newton_gamma_divisor"]

                        # Compute the update
                        update_CONT[:, Armijo_mask] = gamma[Armijo_mask] * direction[:, Armijo_mask]

                        # Compute the new undistorted points and the new distorted points (shape: (2, NAC))
                        new_undistorted = undistorted_CONT[:, Armijo_mask] + update_CONT[:, Armijo_mask]
                        new_distorted = self._distort(new_undistorted, **kwargs)

                        # Compute the new residual (shape: (NAC,))
                        new_residual = compute_residual(new_undistorted, targueted_CONT[:, Armijo_mask]) 

                        # Compute the jacobian matrix of the distortion model for the new undistorted points (shape: (2, 2, NAC))
                        new_jacobian = self._jacobian_distort(new_undistorted, **kwargs)

                        # Compute the new gradient of the residual (shape: (2, NAC))
                        new_gradient = - 2 * numpy.einsum("jik, jk->ik", new_jacobian, targueted_CONT[:, Armijo_mask] - new_distorted)

                        # Compute the Armijo condition (shape: (NAC,))
                        Armijo_cond1 = new_residual <= residual_CONT[Armijo_mask] + c1 * gamma[Armijo_mask] * numpy.einsum("ik, ik->k", update_CONT[:, Armijo_mask], gradient_CONT[:, Armijo_mask])
                        Armijo_cond2 = c2 * numpy.einsum("ik, ik->k", update_CONT[:, Armijo_mask], gradient_CONT[:, Armijo_mask]) >= numpy.einsum("ik, ik->k", update_CONT[:, Armijo_mask], new_gradient)
                        gamma_mask = gamma[Armijo_mask] > kwargs["undistort_Newton_gamma_min"]

                        # Update the Armijo mask
                        Armijo_mask[Armijo_mask] = numpy.logical_and(numpy.logical_not(Armijo_cond1), numpy.logical_not(Armijo_cond2), gamma_mask)

                # Or use a fixed step size
                else:
                    gamma = kwargs["undistort_Newton_gamma_initial"]
                    update_CONT = gamma * direction

                # Update the undistorted points
                undistorted_CONT = undistorted_CONT + update_CONT
                undistorted_IC[:, continue_mask] = undistorted_CONT
                undistorted_points[:, in_course_mask] = undistorted_IC

                # Update the masks
                is_converged_mask[in_course_mask] = create_is_converged_mask(undistorted_IC, targueted_IC, kwargs["undistort_Newton_epsilon"], use_distorted=False, **kwargs)
                is_nan_mask[in_course_mask] = self.nan_1D_mask(undistorted_IC, axis=1)
                is_in_domain_mask[in_course_mask] = self._normalized_domain_mask(undistorted_IC, **kwargs)
                in_course_mask = create_in_course_mask(is_converged_mask, is_nan_mask, is_singular_mask, is_in_domain_mask)

                # Next iteration
                Newton_iter += 1
                pbar.update(1)
        
        # ----------------------------------

        if Newton_iter == kwargs["undistort_Newton_max_iter"] and kwargs["verbose"]:
            Nopt = numpy.sum(in_course_mask)
            print(f"[bold red]Warning:[/bold red] Maximum number of iterations reached ({kwargs['undistort_Newton_max_iter']}). {Nopt} points are not converged completely.")
        
        undistorted_points[:, numpy.logical_not(is_in_domain_mask)] = numpy.nan
        undistorted_points[:, is_singular_mask] = numpy.nan
        undistorted_points[:, is_nan_mask] = numpy.nan

        if kwargs["verbose"]:
            N_out_domain = numpy.sum(numpy.logical_not(is_in_domain_mask))
            N_singular = numpy.sum(is_singular_mask)
            N_nan = numpy.sum(is_nan_mask)
            if N_out_domain + N_singular + N_nan > 0:
                print(f"[bold red]Warning:[/bold red] {N_out_domain} points come out of the normalized domain, set to NaN.")
                print(f"[bold red]Warning:[/bold red] {N_singular} points are blocked by singularities, set to NaN.")
                print(f"[bold red]Warning:[/bold red] {N_nan} points contain NaN values, set to NaN.")
        
        return undistorted_points



    def jacobian_distort_by_differential(self, normalized_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        Computes the Jacobian matrix of the distortion model for a set of normalized image points.

        The Jacobian matrix is computed using the differential method.

        The input normalized points are an array of shape (2, ``Npoints``) where ``Npoints`` is the number of points.
        The output Jacobian matrix must have the shape (2, 2, ``Npoints``).

        The differential method is based on the following formula:

        .. math::

            \frac{\partial f}{\partial x} = \frac{f(x + \epsilon) - f(x)}{\epsilon}

        where :math:`f` is the distortion model and :math:`x` is the normalized image points.

        If a distorted point is outside the distortion domain, the jacobian matrix is set to numpy.nan.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized image points for which to compute the Jacobian matrix.
        
        kwargs : dict
            Additional keyword arguments for the distortion model

        Returns
        -------
        numpy.ndarray
            The Jacobian matrix of the distortion model.
        
        Raises
        ------
        TypeError
            If the normalized points are not a numpy array.
        ValueError
            If the shape of the normalized points is not `(2, Npoints)`.
        """
        # Check the input parameters
        self.check_array_2D_Npoints(normalized_points)

        # Get the kwargs parameters
        kwargs = self.local_kwargs(**kwargs)
        jacobian_distort_Differential_epsilon = kwargs["jacobian_distort_Differential_epsilon"]

        # Extract the number of points
        Npoints = normalized_points.shape[1]

        # Case 0. points contains nan values or is not in the unit circle
        normalized_domain_mask = self._normalized_domain_mask(normalized_points, **kwargs)
        nan_mask = self.nan_1D_mask(normalized_points, axis=1)
        valid_mask = numpy.logical_and(normalized_domain_mask, numpy.logical_not(nan_mask))

        if not numpy.all(valid_mask):
            # Recursively call the function for the valid points
            jacobian = numpy.full((2, 2, Npoints), numpy.nan)
            jacobian[:, :, valid_mask] = self.jacobian_distort_by_differential(normalized_points[:, valid_mask], **kwargs)
            return jacobian

        # Case 1. Compute the Jacobian matrix using the differential method
        jacobian = numpy.zeros((2, 2, Npoints))
        epsilon = numpy.full((Npoints,), jacobian_distort_Differential_epsilon)
        delta_x = numpy.vstack((epsilon, numpy.zeros(Npoints)))
        delta_y = numpy.vstack((numpy.zeros(Npoints), epsilon))

        jacobian[:, 0, :] = (self._distort(normalized_points + delta_x, **kwargs) - self._distort(normalized_points, **kwargs)) / epsilon
        jacobian[:, 1, :] = (self._distort(normalized_points + delta_y, **kwargs) - self._distort(normalized_points, **kwargs)) / epsilon

        return jacobian



    def jacobian_undistort_by_inversion(self, distorted_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        Computes the Jacobian matrix of the undistortion model for a set of distorted image points.

        The input distorted points are an array of shape (2, ``Npoints``) where ``Npoints`` is the number of points.
        The output Jacobian matrix must have the shape (2, 2, ``Npoints``).

        The jacobian_undistort can be processed using the following formula:

        .. math::

            J_{f^{-1}} = \left( J_{f} \circ f^{-1} \right)^{-1}

        where :math:`J_{f}` is the Jacobian matrix of the distortion model and :math:`f^{-1}` is the undistortion model.

        If a distorted point is outside the distortion domain, the jacobian matrix is set to numpy.nan.
        If a jacobian matrix is not invertible, the jacobian matrix is set to numpy.nan.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted image points for which to compute the Jacobian matrix.

        kwargs : dict
            Additional keyword arguments for the distortion model
        
        Returns
        -------
        numpy.ndarray
            The Jacobian matrix of the undistortion model.
        
        Raises
        ------
        TypeError
            If the distorted points are not a numpy array.
        ValueError
            If the shape of the distorted points is not `(2, Npoints)`.
        """
        # Check the input parameters
        self.check_array_2D_Npoints(distorted_points)

        # Compute the normalized points
        normalized_points = self._undistort(distorted_points, **kwargs)

        # Compute the Jacobian matrix of the distortion model
        jacobian_distort = self._jacobian_distort(normalized_points, **kwargs)

        # Initialize the Jacobian matrix of the undistortion model
        jacobian_undistort = numpy.full((2, 2, distorted_points.shape[1]), numpy.nan)

        # Compute the Jacobian matrix of the undistortion model for invertible jacobian matrix
        invertible_mask = numpy.isclose(numpy.linalg.det(jacobian_distort.transpose(2, 0, 1)), 0, atol=1e-10)

        # Compute the Jacobian matrix of the undistortion model
        jacobian_undistort[:, :, invertible_mask] = numpy.linalg.inv(jacobian_distort[:, :, invertible_mask].transpose(2, 0, 1)).transpose(1, 2, 0)

        return jacobian_undistort
