from .distortion import Distortion
from typing import Optional, List, Tuple, Union, Any
import numpy
import scipy
from scipy.spatial import Delaunay
import tqdm
from rich import print
from rich.console import Console
from rich.table import Table


class ZernikeDistortion(Distortion):
    r"""
    ``ZernikeDistortion`` class is a subclass of ``Distortion`` class. It is used to apply Zernike distortion to an image.
    The distorted points can be calculated using the normalized points and the Zernike coefficients as follows:

    .. math::

        x_{\text{dis}} = x_{\text{nor}} + \sum_{m=0}^{\infty} \sum_{n=m}^{\infty} \left( a_{m,n} \cos(m \theta) + b_{m,n} \sin(m \theta) \right) R_{n}^{m}(\rho)
        
    .. math::

        y_{\text{dis}} = y_{\text{nor}} + \sum_{m=0}^{\infty} \sum_{n=m}^{\infty} \left( c_{m,n} \cos(m \theta) + d_{m,n} \sin(m \theta) \right) R_{n}^{m}(\rho)

    The :math:`\rho` and :math:`\theta` are 0the polar coordinates of the normalized point in the unit circle.

    The coefficients :math:`a_{m,n}`, :math:`b_{m,n}`, :math:`c_{m,n}`, and :math:`d_{m,n}` are stored in the coefficients attribute.
    If we denote ``Mzer`` and ``Nzer`` as the maximum orders of the Zernike polynomials, the coefficients attribute is a 2D numpy array with shape (Mzer + 1, Nzer + 1, 4).

    To access the coefficients, the user can use the following properties of the class:
    
    - ``cosine_x`` or ``a``: The Zernike coefficients for the cosine terms in the x-direction.
    - ``sine_x`` or ``b``: The Zernike coefficients for the sine terms in the x-direction.
    - ``cosine_y`` or ``c``: The Zernike coefficients for the cosine terms in the y-direction.
    - ``sine_y`` or ``d``: The Zernike coefficients for the sine terms in the y-direction.

    The coefficients attribute is a triangular matrix as n > m.
    Futhermore, if n-m is odd, the coefficients are zero.

    .. math::

        A = \begin{bmatrix}
            a_{0,0} & 0 & a_{0,2} & 0 & \cdots \\
            0 & a_{1,1} & 0 & a_{1,3} & \cdots \\
            0 & 0 & a_{2,2} & 0 & \cdots \\
            0 & 0 & 0 & a_{3,3} & \cdots \\
            \vdots & \vdots & \vdots & \vdots & \vdots 
        \end{bmatrix}

    To access the coefficients, the user can use the a and b properties of the class.

    .. code-block:: python

        zernike = ZernikeDistortion()
        zernike.a[1, 1] = 1.0
        zernike.sine_y[1, 1] = 0.0

    The unit circle for the polar coordinates is the equivalent radius of the image.
    The center of the circle is the center coordinate of the image (in general, the center of the image) and the radius is the maximum distance from the center to the corner of the image.

    Parameters
    ----------
    coefficients : numpy.ndarray, optional
        The Zernike coefficients. The default is None.

    Mzer : int, optional
        The maximum order of the Zernike polynomials. The default is None. Used if the coefficients is None.
    
    Nzer : int, optional
        The maximum order of the Zernike polynomials. The default is None. Used if the coefficients is None.

    Attributes
    ----------
    a : numpy.ndarray
        The Zernike coefficients for the cosine terms in the x-direction.

    b : numpy.ndarray
        The Zernike coefficients for the sine terms in the x-direction.

    c : numpy.ndarray
        The Zernike coefficients for the cosine terms in the y-direction.

    d : numpy.ndarray
        The Zernike coefficients for the sine terms in the y-direction.
    """


    # ==================================================
    # ========= Internal Helper Methods ================
    # ==================================================
    class _CoefficientAccessor:
        """ Provides access to Zernike coefficients with automatic enforcement of validity constraints. """
        def __init__(self, parent, index):
            self._parent = parent  # Reference to the main instance
            self._index = index  # 0 for `a`, 1 for `b`, 2 for `c`, 3 for `d`


        def _validate_key(self, key):
            """ Validates the key format for accessing coefficients. """
            # Check the key format
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError("Invalid indexing. Use zernike.a[m, n] or zernike.a[m_start:m_end, n_start:n_end].")
            if not all(isinstance(i, (int, slice)) for i in key):
                raise TypeError("Indices must be integers or slices.")


        def __getitem__(self, key):
            """ Retrieves individual or sliced coefficient values. """
            self._validate_key(key)
            return self._parent._coefficients[key[0], key[1], self._index]


        def __setitem__(self, key, value):
            """ Sets individual or sliced coefficient values, enforcing Zernike constraints. """
            self._validate_key(key)
            m_slice, n_slice = key
            m_indices, n_indices = numpy.meshgrid(numpy.arange(self._parent.Mzer + 1)[m_slice], 
                                               numpy.arange(self._parent.Nzer + 1)[n_slice], indexing='ij')

            # Create validity mask (n >= m and (n - m) even)
            valid_mask = (n_indices >= m_indices) & ((n_indices - m_indices) % 2 == 0)
            value_array = numpy.array(value, dtype=self._parent._coefficients.dtype)
            if value_array.shape == ():
                value_array = numpy.full(valid_mask.shape, value, dtype=self._parent._coefficients.dtype) # scalar to array

            # Ensure value array shape matches the valid mask
            if value_array.shape != valid_mask.shape:
                raise ValueError(f"Shape mismatch: expected {valid_mask.shape}, got {value_array.shape}")

            # Apply mask: invalid values remain unchanged
            self._parent._coefficients[m_slice, n_slice, self._index] = valid_mask * value_array


        def __array__(self):
            """ Returns the underlying numpy array representation of `a` or `b`. """
            return self._parent._coefficients[:, :, self._index]


        def __repr__(self):
            """ Displays the coefficient matrix. """
            return repr(self.__array__())
        
        def copy(self):
            """ Returns a copy of the coefficient matrix. """
            return self.__array__().copy()



    @property
    def a(self):
        return self._a_accessor

    @property
    def cosine_x(self):
        return self.a

    @a.setter
    def a(self, new_array):
        """ Allows assigning an entire numpy array to `zernike.a`. """
        # Check the input parameters
        if not isinstance(new_array, numpy.ndarray):
            raise TypeError("New coefficients must be a numpy array.")
        if new_array.shape != (self.Mzer + 1, self.Nzer + 1):
            raise ValueError(f"Array shape must be {(self.Mzer + 1, self.Nzer + 1)}")

        # Create a mask to enforce Zernike constraints
        m_indices, n_indices = numpy.meshgrid(numpy.arange(self.Mzer + 1), numpy.arange(self.Nzer + 1), indexing='ij')
        valid_mask = (n_indices >= m_indices) & ((n_indices - m_indices) % 2 == 0)
        self._coefficients[:, :, 0] = valid_mask * new_array
    
    @cosine_x.setter
    def cosine_x(self, new_array):
        self.a = new_array



    @property
    def b(self):
        return self._b_accessor

    @property
    def sine_x(self):
        return self.b

    @b.setter
    def b(self, new_array):
        """ Allows assigning an entire numpy array to `zernike.b`. """
        # Check the input parameters
        if not isinstance(new_array, numpy.ndarray):
            raise TypeError("New coefficients must be a numpy array.")
        if new_array.shape != (self.Mzer + 1, self.Nzer + 1):
            raise ValueError(f"Array shape must be {(self.Mzer + 1, self.Nzer + 1)}")

        # Create a mask to enforce Zernike constraints
        m_indices, n_indices = numpy.meshgrid(numpy.arange(self.Mzer + 1), numpy.arange(self.Nzer + 1), indexing='ij')
        valid_mask = (n_indices >= m_indices) & ((n_indices - m_indices) % 2 == 0)
        self._coefficients[:, :, 1] = valid_mask * new_array

    @sine_x.setter
    def sine_x(self, new_array):
        self.b = new_array



    @property
    def c(self):
        return self._c_accessor
    
    @property
    def cosine_y(self):
        return self.c
    
    @c.setter
    def c(self, new_array):
        """ Allows assigning an entire numpy array to `zernike.c`. """
        # Check the input parameters
        if not isinstance(new_array, numpy.ndarray):
            raise TypeError("New coefficients must be a numpy array.")
        if new_array.shape != (self.Mzer + 1, self.Nzer + 1):
            raise ValueError(f"Array shape must be {(self.Mzer + 1, self.Nzer + 1)}")

        # Create a mask to enforce Zernike constraints
        m_indices, n_indices = numpy.meshgrid(numpy.arange(self.Mzer + 1), numpy.arange(self.Nzer + 1), indexing='ij')
        valid_mask = (n_indices >= m_indices) & ((n_indices - m_indices) % 2 == 0)
        self._coefficients[:, :, 2] = valid_mask * new_array
    
    @cosine_y.setter
    def cosine_y(self, new_array):
        self.c = new_array
    


    @property
    def d(self):
        return self._d_accessor
    
    @property
    def sine_y(self):
        return self.d

    @d.setter
    def d(self, new_array):
        """ Allows assigning an entire numpy array to `zernike.d`. """
        # Check the input parameters
        if not isinstance(new_array, numpy.ndarray):
            raise TypeError("New coefficients must be a numpy array.")
        if new_array.shape != (self.Mzer + 1, self.Nzer + 1):
            raise ValueError(f"Array shape must be {(self.Mzer + 1, self.Nzer + 1)}")
        
        # Create a mask to enforce Zernike constraints
        m_indices, n_indices = numpy.meshgrid(numpy.arange(self.Mzer + 1), numpy.arange(self.Nzer + 1), indexing='ij')
        valid_mask = (n_indices >= m_indices) & ((n_indices - m_indices) % 2 == 0)
        self._coefficients[:, :, 3] = valid_mask * new_array
    
    @sine_y.setter
    def sine_y(self, new_array):
        self.d = new_array



    @property
    def coefficients(self) -> numpy.ndarray:
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        # Check the input parameters
        if not isinstance(coefficients, numpy.ndarray):
            raise TypeError("Coefficients must be a numpy array.")
        if coefficients.shape != self._coefficients.shape:
            raise ValueError("Coefficients must have the same shape as the current ones.")

        # Update the coefficients
        self._coefficients = coefficients
    


    @property
    def Mzer(self) -> int:
        return self._coefficients.shape[0] - 1

    @property
    def Nzer(self) -> int:
        return self._coefficients.shape[1] - 1



    # --------------------------------------------------



    def __init__(self, coefficients: Optional[numpy.ndarray] = None, Mzer: int = 10, Nzer: int = 10, **kwargs) -> None:

        # Check the input parameters
        if coefficients is None:
            if Mzer is None or Nzer is None:
                raise ValueError("If coefficients are not provided, Mzer and Nzer must be specified.")
            if not all(isinstance(i, int) for i in (Mzer, Nzer)):
                raise TypeError("Mzer and Nzer must be integers.")
            if Mzer < 1 or Nzer < 1:
                raise ValueError("Mzer and Nzer must be positive integers.")
            coefficients = numpy.zeros((Mzer + 1, Nzer + 1, 4))
        if not isinstance(coefficients, numpy.ndarray):
            raise TypeError("Coefficients must be a numpy array.")
        if coefficients.ndim != 3 or coefficients.shape[2] != 4:
            raise ValueError("Coefficients must be a 3D array with shape (Mzer, Nzer, 4).")

        # Initialize the parent class
        self._coefficients = coefficients
        super().__init__(coefficients)
        self._active_tables = {}
        self.active_repr(["a", "b", "c", "d"])

        # Add the new attributes
        self._add_kwargs(
            center=numpy.array([0.0, 0.0]),
            radius=1.0,
        )

        # Set the attributes
        self._set_kwargs(**kwargs)

        # Create coefficient accessors
        self._a_accessor = self._CoefficientAccessor(self, index=0)
        self._b_accessor = self._CoefficientAccessor(self, index=1)
        self._c_accessor = self._CoefficientAccessor(self, index=2)
        self._d_accessor = self._CoefficientAccessor(self, index=3)


    
    # --------------------------------------------------



    @property
    def center(self) -> numpy.ndarray:
        return self.kwargs["center"]

    @center.setter
    def center(self, center: Union[numpy.ndarray, list, tuple]):
        if not isinstance(center, (numpy.ndarray, list, tuple)):
            raise TypeError("Center must be a numpy array, list, or tuple.")
        if len(center) != 2:
            raise ValueError("Center must have two elements.")
        if not all(isinstance(i, (int, float)) for i in center):
            raise TypeError("Center elements must be numbers.")
        self._set_kwargs(center=numpy.array(center, dtype=float))

    

    @property
    def radius(self) -> float:
        return self.kwargs["radius"]

    @radius.setter
    def radius(self, radius: Union[int, float]):
        if not isinstance(radius, (int, float)):
            raise TypeError("Radius must be a number.")
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self._set_kwargs(radius=float(radius))



    # --------------------------------------------------



    def cartesian_to_polar(self, normalized_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        Converts the normalized points to polar coordinates.

        The normalized points are in the form of a 2D numpy array with shape (2, Npoints).
        The polar coordinates are in the form of a 2D numpy array with shape (2, Npoints).

        In this transformation, the image's circular region is mapped to a unit circle, 
       
        The unit circle is defined by the ``center`` and ``radius`` attributes.

        The conversion is done as follows:
        (x, y) -> (rho, theta)

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points.

        kwargs : dict
            Additional keyword arguments for the distortion model

        Returns
        -------
        numpy.ndarray
            The polar coordinates.

        Raises
        ------
        ValueError
            If the shape of the input array is not (2, Npoints).
        """
        # Check the input parameters
        self.check_array_2D_Npoints(normalized_points)

        # Get the kwargs parameters
        kwargs = self.local_kwargs(**kwargs)
        center = kwargs["center"]
        radius = kwargs["radius"]

        # Compute the polar coordinates
        rho = numpy.linalg.norm(normalized_points - center.reshape(-1, 1), axis=0)/radius
        theta = numpy.arctan2(normalized_points[1, :] - center[1], normalized_points[0, :] - center[0])
        return numpy.vstack((rho, theta))



    def polar_to_cartesian(self, polar_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        Converts the polar coordinates to normalized points.

        The polar points are in the form of a 2D numpy array with shape (2, Npoints).
        The normalized points are in the form of a 2D numpy array with shape (2, Npoints).

        In this transformation, the unit circle is mapped to the image's circular region, 
        
        The unit circle is defined by the ``center`` and ``radius`` attributes.

        The conversion is done as follows:
        (rho, theta) -> (x, y)

        Parameters
        ----------
        polar_points : numpy.ndarray
            The polar points.

        kwargs : dict
            Additional keyword arguments for the distortion model

        Returns
        -------
        numpy.ndarray
            The normalized points.

        Raises
        ------
        ValueError
            If the shape of the input array is not (2, Npoints).
        """
        # Check the input parameters
        self.check_array_2D_Npoints(polar_points)

        # Get the kwargs parameters
        kwargs = self.local_kwargs(**kwargs)
        center = kwargs["center"]
        radius = kwargs["radius"]

        # Compute the normalized points
        x = radius * polar_points[0, :] * numpy.cos(polar_points[1, :]) + center[0]
        y = radius * polar_points[0, :] * numpy.sin(polar_points[1, :]) + center[1]
        return numpy.vstack((x, y))


    
    def unit_circle_mask(self, normalized_points: numpy.ndarray, use_polar: bool = False, **kwargs) -> numpy.ndarray:
        """
        Checks if the normalized points are inside the unit circle.

        The normalized points are in the form of a 2D numpy array with shape (2, Npoints).
        The output is a 1D numpy array with shape (Npoints,).

        The unit circle is defined by the ``center`` and ``radius`` attributes.

        use_polar parameter is used to determine if the polar coordinates are used directly.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points.

        use_polar : bool, optional
            If True, the polar coordinates are used. The default is False.
        
        Returns
        -------
        numpy.ndarray
            The mask.

        Raises
        ------
        TypeError
            If use_polar is not a boolean.
        ValueError
            If the shape of the input array is not (2, Npoints).
        """
        # Check the input parameters
        self.check_array_2D_Npoints(normalized_points)
        if not isinstance(use_polar, bool):
            raise TypeError("Use polar must be a boolean.")

        # Get the kwargs parameters
        kwargs = self.local_kwargs(**kwargs)
        center = kwargs["center"]
        radius = kwargs["radius"]

        # Compute the mask
        if use_polar:
            return normalized_points[0, :] <= 1
        else:
            return numpy.linalg.norm(normalized_points - center.reshape(-1, 1), axis=0) <= radius   



    def global_radial_polynomials(self, rho: numpy.ndarray, n: int, m: int) -> numpy.ndarray:
        r"""
        Computes the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` for :math:`\rho \leq 1`.

        The rho values are in the form of a 1D numpy array with shape (Npoints,).
        The radial Zernike polynomial is in the form of a 1D numpy array with shape (Npoints,).

        The radial Zernike polynomial is defined as follows:

        .. math::

            R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} \rho^{n-2k}

        if :math:`n < 0`, :math:`m < 0`, :math:`n < m`, or :math:`(n - m)` is odd, the output is a zero array.

        if :math:`\rho ` is not in :math:`0 \leq \rho \leq 1` or :math`\rho` is numpy.nan, the output is numpy.nan.

        Parameters
        ----------
        rho : numpy.ndarray
            The rho values.
        
        n : int
            The order of the Zernike polynomial.

        m : int
            The degree of the Zernike polynomial.
        
        Returns
        -------
        numpy.ndarray
            The radial Zernike polynomial.
        
        Raises
        ------
        TypeError
            If the rho values are not a numpy array or if n and m are not integers.
        ValueError
            If the shape of the rho values is not (Npoints,).
        """
        # Check the input parameters
        if not isinstance(rho, numpy.ndarray):
            raise TypeError("Rho values must be a numpy array.")
        if rho.ndim != 1:
            raise ValueError("Rho values must be a 1D array.")
        if not all(isinstance(i, int) for i in (n, m)):
            raise TypeError("n and m must be integers.")

        # Extract the number of points
        Npoints = rho.shape[0]

        # Case 0. rho contains nan values or is not in 0 <= rho <= 1
        unit_circle_mask = numpy.logical_and(0 <= rho, rho <= 1)
        nan_mask = self.nan_1D_mask(rho, axis=0)
        valid_mask = numpy.logical_and(unit_circle_mask, numpy.logical_not(nan_mask))

        if not numpy.all(valid_mask):
            result = numpy.full((Npoints,), numpy.nan)
            result[valid_mask] = self.global_radial_polynomials(rho[valid_mask], n, m) # Recursive call for valid rho values 
            return result

        # --------------------------------------

        # Case 1. Special cases
        if n < 0 or m < 0 or n < m or (n - m) % 2 == 1:
            return numpy.zeros((Npoints,))

        # Case 2. Compute the radial polynomial
        s = (n - m) // 2
        result = numpy.zeros_like(rho)
        for k in range(0, s + 1):
            result += (-1)**k * factorial(n - k) / (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)) * rho**(n - 2*k)
        return result



    def global_derivative_radial_polynomials(self, rho: numpy.ndarray, n: int, m: int) -> numpy.ndarray:
        r"""
        Computes the derivative of the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` for :math:`\rho \leq 1`.

        The rho values are in the form of a 1D numpy array with shape (Npoints,).
        The derivative of the radial Zernike polynomial is in the form of a 1D numpy array with shape (Npoints,).

        The derivative of the radial Zernike polynomial is defined as follows:

        .. math::

            \frac{dR_{n}^{m}(\rho)}{d\rho} = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} (n-2k) \rho^{n-2k-1}

        if :math:`n < 0`, :math:`m < 0`, :math:`n < m`, or :math:`(n - m)` is odd, the output is a zero array.

        if :math:`\rho ` is not in :math:`0 \leq \rho \leq 1` or :math`\rho` is numpy.nan, the output is numpy.nan.

        Parameters
        ----------
        rho : numpy.ndarray
            The rho values.
        
        n : int
            The order of the Zernike polynomial.

        m : int
            The degree of the Zernike polynomial.
        
        Returns
        -------
        numpy.ndarray
            The derivative of the radial Zernike polynomial.
        
        Raises
        ------
        TypeError
            If the rho values are not a numpy array or if n and m are not integers.
        ValueError
            If the shape of the rho values is not (Npoints,).
        """
        # Check the input parameters
        if not isinstance(rho, numpy.ndarray):
            raise TypeError("Rho values must be a numpy array.")
        if rho.ndim != 1:
            raise ValueError("Rho values must be a 1D array.")
        if not all(isinstance(i, int) for i in (n, m)):
            raise TypeError("n and m must be integers.")

        # Extract the number of points
        Npoints = rho.shape[0]

        # Case 0. rho contains nan values or is not in 0 <= rho <= 1
        unit_circle_mask = numpy.logical_and(0 <= rho, rho <= 1)
        nan_mask = self.nan_1D_mask(rho, axis=0)
        valid_mask = numpy.logical_and(unit_circle_mask, numpy.logical_not(nan_mask))

        if not numpy.all(valid_mask):
            result = numpy.full((Npoints,), numpy.nan)
            result[valid_mask] = self.global_derivative_radial_polynomials(rho[valid_mask], n, m) # Recursive call for valid rho values
            return result
        
        # --------------------------------------

        # Case 1. Special cases
        if n < 0 or m < 0 or n < m or (n - m) % 2 == 1:
            return numpy.zeros((Npoints,))

        # Case 2. Compute the derivative of the radial polynomial
        s = (n - m) // 2
        result = numpy.zeros_like(rho)
        for k in range(0, s + 1):
            result += (-1)**k * factorial(n - k) / (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)) * (n - 2*k) * rho**(n - 2*k - 1)
        return result
    


    def radial_polynomials(self, rho: numpy.ndarray, n: int, m: int) -> numpy.ndarray:
        r"""
        Computes the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` for :math:`\rho \leq 1`.

        The rho values are in the form of a 1D numpy array with shape (Npoints,).
        The radial Zernike polynomial is in the form of a 1D numpy array with shape (Npoints,).

        The radial Zernike polynomial is defined as follows:

        .. math::

            R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} \rho^{n-2k}

        if :math:`n < 0`, :math:`m < 0`, :math:`n < m`, or :math:`(n - m)` is odd, the output is a zero array.

        if :math:`\rho ` is not in :math:`0 \leq \rho \leq 1` or :math`\rho` is numpy.nan, the output is numpy.nan.

        Parameters
        ----------
        rho : numpy.ndarray
            The rho values.
        
        n : int
            The order of the Zernike polynomial.

        m : int
            The degree of the Zernike polynomial.
        
        Returns
        -------
        numpy.ndarray
            The radial Zernike polynomial.
        
        Raises
        ------
        TypeError
            If the rho values are not a numpy array or if n and m are not integers.
        ValueError
            If the shape of the rho values is not (Npoints,).
        """
        # Check the input parameters
        if not isinstance(rho, numpy.ndarray):
            raise TypeError("Rho values must be a numpy array.")
        if rho.ndim != 1:
            raise ValueError("Rho values must be a 1D array.")
        if not all(isinstance(i, int) for i in (n, m)):
            raise TypeError("n and m must be integers.")

        # Extract the number of points
        Npoints = rho.shape[0]

        # Case 0. rho contains nan values or is not in 0 <= rho <= 1
        unit_circle_mask = numpy.logical_and(0 <= rho, rho <= 1)
        nan_mask = self.nan_1D_mask(rho, axis=0)
        valid_mask = numpy.logical_and(unit_circle_mask, numpy.logical_not(nan_mask))

        if not numpy.all(valid_mask):
            result = numpy.full((Npoints,), numpy.nan)
            result[valid_mask] = self.radial_polynomials(rho[valid_mask], n, m) # Recursive call for valid rho values
            return result

        # --------------------------------------

        # Case 1. Special cases
        if n < 0 or m < 0 or n < m or (n - m) % 2 == 1:
            return numpy.zeros((Npoints,))

        # Case 2. Small n and m
        if n == 0:
            if m == 0:
                return(numpy.ones((Npoints,)))
        elif n == 1:
            if m == 1:
                return(rho)
        elif n == 2:
            if m == 0:
                return(2*rho**2 - 1)
            elif m == 2:
                return(rho**2)
        elif n == 3:
            if m == 1:
                return(3*rho**3 - 2*rho)
            elif m == 3:
                return(rho**3)
        elif n == 4:
            if m == 0:
                return(6*rho**4 - 6*rho**2 + 1)
            elif m == 2:
                return(4*rho**4 - 3*rho**2)
            elif m == 4:
                return(rho**4)
        elif n == 5:
            if m == 1:
                return(10*rho**5 - 12*rho**3 + 3*rho)
            elif m == 3:
                return(5*rho**5 - 4*rho**3)
            elif m == 5:
                return(rho**5)
        elif n == 6:
            if m == 0:
                return(20*rho**6 - 30*rho**4 + 12*rho**2 - 1)
            elif m == 2:
                return(15*rho**6 - 20*rho**4 + 6*rho**2)
            elif m == 4:
                return(6*rho**6 - 5*rho**4)
            elif m == 6:
                return(rho**6)
        elif n == 7:
            if m == 1:
                return(35*rho**7 - 60*rho**5 + 30*rho**3 - 4*rho)
            elif m == 3:
                return(21*rho**7 - 30*rho**5 + 10*rho**3)
            elif m == 5:
                return(7*rho**7 - 6*rho**5)
            elif m == 7:
                return(rho**7)

        # Case 3. Large n and m
        else:
            return(self.global_radial_polynomials(rho, n, m))



    def derivative_radial_polynomials(self, rho: numpy.ndarray, n: int, m: int) -> numpy.ndarray:
        r"""
        Computes the derivative of the radial Zernike polynomial :math:`\frac{d}{d\rho} R_{n}^{m}(\rho)` for :math:`\rho \leq 1`.

        The rho values are in the form of a 1D numpy array with shape (Npoints,).
        The derivative of the radial Zernike polynomial is in the form of a 1D numpy array with shape (Npoints,).

        The derivative of the radial Zernike polynomial is defined as follows:

        .. math::

            \frac{d}{d\rho} R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} (n-2k) \rho^{n-2k-1}

        if :math:`n < 0`, :math:`m < 0`, :math:`n < m`, or :math:`(n - m)` is odd, the output is a zero array.

        if :math:`\rho ` is not in :math:`0 \leq \rho \leq 1` or :math`\rho` is numpy.nan, the output is numpy.nan.

        Parameters
        ----------
        rho : numpy.ndarray
            The rho values.

        n : int
            The order of the Zernike polynomial.

        m : int
            The degree of the Zernike polynomial.
        
        Returns
        -------
        numpy.ndarray
            The derivative of the radial Zernike polynomial.

        Raises
        ------
        TypeError
            If the rho values are not a numpy array or if n and m are not integers.
        ValueError
            If the shape of the rho values is not (Npoints,).
        """
        # Check the input parameters
        if not isinstance(rho, numpy.ndarray):
            raise TypeError("Rho values must be a numpy array.")
        if rho.ndim != 1:
            raise ValueError("Rho values must be a 1D array.")
        if not all(isinstance(i, int) for i in (n, m)):
            raise TypeError("n and m must be integers.")
        
        # Extract the number of points
        Npoints = rho.shape[0]

        # Case 0. rho contains nan values or is not in 0 <= rho <= 1
        unit_circle_mask = numpy.logical_and(0 <= rho, rho <= 1)
        nan_mask = self.nan_1D_mask(rho, axis=0)
        valid_mask = numpy.logical_and(unit_circle_mask, numpy.logical_not(nan_mask))

        if not numpy.all(valid_mask):
            result = numpy.full((Npoints,), numpy.nan)
            result[valid_mask] = self.derivative_radial_polynomials(rho[valid_mask], n, m) # Recursive call for valid rho values
            return result

        # --------------------------------------

        # Case 1. Special cases
        if n < 0 or m < 0 or n < m or (n - m) % 2 == 1:
            return numpy.zeros((Npoints,))
        
        # Case 2. Small n and m
        if n == 0:
            if m == 0:
                return(numpy.zeros((Npoints,)))
        elif n == 1:
            if m == 1:
                return(numpy.ones((Npoints,)))
        elif n == 2:
            if m == 0:
                return(4*rho)
            elif m == 2:
                return(2*rho)
        elif n == 3:
            if m == 1:
                return(9*rho**2 - 2)
            elif m == 3:
                return(3*rho**2)
        elif n == 4:
            if m == 0:
                return(24*rho**3 - 12*rho)
            elif m == 2:
                return(16*rho**3 - 6*rho)
            elif m == 4:
                return(4*rho**3)
        elif n == 5:
            if m == 1:
                return(45*rho**4 - 40*rho**2)
            elif m == 3:
                return(25*rho**4 - 18*rho**2)
            elif m == 5:
                return(5*rho**4)
        elif n == 6:
            if m == 0:
                return(120*rho**5 - 160*rho**3 + 60*rho)
            elif m == 2:
                return(90*rho**5 - 90*rho**3)
            elif m == 4:
                return(36*rho**5 - 24*rho**3)
            elif m == 6:
                return(6*rho**5)
        elif n == 7:
            if m == 1:
                return(189*rho**6 - 315*rho**4 + 140*rho**2)
            elif m == 3:
                return(126*rho**6 - 180*rho**4)
            elif m == 5:
                return(49*rho**6 - 36*rho**4)
            elif m == 7:
                return(7*rho**6)
        
        # Case 3. Large n and m
        else:
            return(self.global_derivative_radial_polynomials(rho, n, m))

        

    def active_repr(self, coefficient_name: Union[str, List[str]], active: Union[bool, List[bool]] = True):
        """
        Enables or disables the display of a specific coefficient matrix in `__repr__`.

        Parameters
        ----------
        coefficient_name : Union[str, List[str]]
            The coefficient table to enable/disable. Must be one of: "a", "b", "c", or "d".
        active : Union[bool, List[bool]], optional
            The active status of the coefficient table. Default is True.

        Raises
        ------
        ValueError
            If an invalid coefficient name is provided.
        """
        # Check the input parameters
        if isinstance(coefficient_name, str): # Convert to list for consistency
            coefficient_name = [coefficient_name]
        if not isinstance(coefficient_name, list):
            raise TypeError("Coefficient name must be a string or a list of strings.")
        if not all(isinstance(name, str) for name in coefficient_name): # Check if all elements are strings
            raise TypeError("Coefficient name must be a string or a list of strings.")
        if not all(name in ["a", "b", "c", "d"] for name in coefficient_name): # Check if all elements are valid
            raise ValueError("Invalid coefficient name. Choose from 'a', 'b', 'c', or 'd'.")
        if len(coefficient_name) != len(set(coefficient_name)):
            raise ValueError("Coefficient name must be unique.")
        
        if isinstance(active, bool): # Convert to list for consistency
            active = [active] * len(coefficient_name)
        if not isinstance(active, list):
            raise TypeError("Active status must be a boolean or a list of booleans.")
        if not all(isinstance(status, bool) for status in active): # Check if all elements are booleans
            raise TypeError("Active status must be a boolean or a list of booleans.")
        if len(active) != len(coefficient_name): # Check if the length matches
            raise ValueError("Coefficient name and active status must have the same length.")            

        # Update the active status
        for index in range(len(coefficient_name)):
            self._active_tables[coefficient_name[index]] = active[index]



    def __repr__(self) -> str:
        """ 
        Prints a formatted representation of the ZernikeDistortion instance 
        using Rich tables. Displays selected coefficient matrices (a, b, c, d) 
        with a cross indicating invalid coefficients.

        The matrices are formatted as follows:
        - Each **row** represents the index **m** (degree of azimuthal variation).
        - Each **column** represents the index **n** (radial order).
        - Invalid coefficients (where n < m or (n - m) is odd) are marked with a red cross.
        - Valid coefficients are displayed with two decimal precision.
        - The displayed tables depend on the active status set via `active_repr()`.

        This method prints directly to the console for a better visual representation 
        but returns an empty string to avoid unnecessary text output when used in 
        interactive environments.

        Returns
        -------
        str
            An empty string, as the output is printed directly to the console.
        """
        console = Console()

        def format_table(title, coefficients):
            table = Table(title=title, show_lines=True)

            # Add column headers (n values)
            table.add_column("m \\ n", justify="center", style="bold")
            for n in range(self.Nzer):
                table.add_column(str(n), justify="center")

            # Add rows (m values)
            for m in range(self.Mzer):
                row = [f"[bold]{m}[/bold]"]
                for n in range(self.Nzer):
                    if n < m or (n - m) % 2 == 1:
                        row.append("[red]❌[/red]")
                    else:
                        row.append(f"[green]{coefficients[m, n]:.2f}[/green]")
                table.add_row(*row)

            console.print(table)

        console.print("\n[bold cyan] Zernike Distortion Coefficients[/bold cyan]\n")

        # Display only the tables that are enabled
        if self._active_tables["a"]:
            format_table("Cosine X Coefficients (a)", self.a)
        if self._active_tables["b"]:
            format_table("Sine X Coefficients (b)", self.b)
        if self._active_tables["c"]:
            format_table("Cosine Y Coefficients (c)", self.c)
        if self._active_tables["d"]:
            format_table("Sine Y Coefficients (d)", self.d)

        return ""  # Ensures no unnecessary text output



    # --------------------------------------------------



    def _normalized_domain_mask(self, normalized_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""

        .. seealso::
        
            - :func:`pydistort.Distortion._normalized_domain_mask` 

        The normalized points shape is `(2, Npoints)`.
        The output mask shape is `(Npoints,)`.

        The normalized domain is defined by the unit circle.

        .. seealso::

            - :func:`pydistort.ZernikeDistortion.unit_circle_mask`

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points.

        kwargs : dict
            Additional keyword arguments for the distortion model.

        Returns
        -------
        numpy.ndarray
            The mask of the normalized domain.

        Raises
        ------
        TypeError
            If the normalized points are not a numpy array.
        ValueError
            If the shape of the normalized points is not (2, Npoints).
        """
        return self.unit_circle_mask(normalized_points, use_polar=False, **kwargs)



    def _distorted_domain_mask(self, distorted_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""

        .. seealso::
        
            - :func:`pydistort.Distortion._distorted_domain_mask` 

        The distorted points shape is `(2, Npoints)`.
        The output mask shape is `(Npoints,)`.

        The distorted domain is defined by the unit circle.

        .. seealso::

            - :func:`pydistort.ZernikeDistortion.unit_circle_mask`

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted points.

        kwargs : dict
            Additional keyword arguments for the distortion model.

        Returns
        -------
        numpy.ndarray
            The mask of the distorted domain.

        Raises
        ------
        TypeError
            If the distorted points are not a numpy array.
        ValueError
            If the shape of the distorted points is not (2, Npoints).
        """
        return self.unit_circle_mask(distorted_points, use_polar=False, **kwargs)



    def _distort(self, normalized_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        .. seealso::

            - :func:`pydistort.Distortion._distort`

        The input normalized points shape is `(2, Npoints)`.

        The distorted points are calculated using the Zernike coefficients as follows:

        .. math::

            x_{\text{dis}} = x_{\text{nor}} + \sum_{m=0}^{\infty} \sum_{n=m}^{\infty} \left( a_{m,n} \cos(m \theta) + b_{m,n} \sin(m \theta) \right) R_{n}^{m}(\rho)

        .. math::

            y_{\text{dis}} = y_{\text{nor}} + \sum_{m=0}^{\infty} \sum_{n=m}^{\infty} \left( c_{m,n} \cos(m \theta) + d_{m,n} \sin(m \theta) \right) R_{n}^{m}(\rho)

        where :math:`\rho` and :math:`\theta` are the polar coordinates of the normalized point in the unit circle.

        If a point is outside the unit circle or contains numpy.nan, the distorted point is set to numpy.nan.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points.

        kwargs : dict
            Additional keyword arguments for the distortion model

        Returns
        -------
        numpy.ndarray
            The distorted points.

        Raises
        ------
        TypeError
            If the normalized points are not a numpy array.
        ValueError
            If the shape of the normalized points is not (2, Npoints).
        """
        # Check the input parameters
        self.check_array_2D_Npoints(normalized_points)

        # Get the kwargs parameters
        kwargs = self.local_kwargs(**kwargs)

        # Extract the number of points
        Npoints = normalized_points.shape[1]

        # Case 0. points contains nan values or is not in the unit circle
        normalized_domain_mask = self._normalized_domain_mask(normalized_points, **kwargs)
        nan_mask = self.nan_1D_mask(normalized_points, axis=1)
        valid_mask = numpy.logical_and(normalized_domain_mask, numpy.logical_not(nan_mask))

        if not numpy.all(valid_mask):
            # Recursively call the function for the valid points
            distorted_points = numpy.full((2, Npoints), numpy.nan)
            distorted_points[:, valid_mask] = self._distort(normalized_points[:, valid_mask], **kwargs)
            return distorted_points

        # --------------------------------------

        # Compute the polar coordinates
        polar_points = self.cartesian_to_polar(normalized_points, **kwargs)
        rho = polar_points[0, :]
        theta = polar_points[1, :]

        # Compute the distorted points using the Zernike coefficients
        cos_theta = numpy.cos(theta)
        sin_theta = numpy.sin(theta)
        distorted_delta = numpy.zeros((2, Npoints))

        # Loop over the Zernike coefficients
        for m in range(self.Mzer):
            for n in range(m, self.Nzer): # n >= m
                if (n - m) % 2 == 0 and (self.a[m, n] != 0 or self.b[m, n] != 0 or self.c[m, n] != 0 or self.d[m, n] != 0):
                    # Compute the radial polynomial
                    radial_polynomial = self.radial_polynomials(rho, n, m)

                    # Add the cosine and sine terms
                    distorted_delta[0, :] += (self.a[m, n] * cos_theta + self.b[m, n] * sin_theta) * radial_polynomial
                    distorted_delta[1, :] += (self.c[m, n] * cos_theta + self.d[m, n] * sin_theta) * radial_polynomial

        # Assign the distorted points
        distorted_points = normalized_points + distorted_delta
        return distorted_points

        

    def _undistort(self, distorted_points: numpy.ndarray, undistorted_guess: Optional[numpy.ndarray] = None, **kwargs) -> numpy.ndarray:
        r"""
        
        .. seealso::

            - :func:`pydistort.Distortion._undistort`
            - :func:`pydistort.Distortion.undistort_by_Newton`

        The input distorted points shape is `(2, Npoints)`.
        The output undistorted points shape is `(2, Npoints)`.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted points.

        undistorted_guess : Optional[numpy.ndarray], optional
            The initial guess for the undistorted points. Default is None.
        
        kwargs : dict
            Additional keyword arguments for the distortion model.

        Returns
        -------
        numpy.ndarray
            The undistorted points.

        Raises
        ------
        TypeError
            If the distorted points are not a numpy array.
        ValueError
            If the shape of the distorted points is not (2, Npoints).        
        """
        return self.undistort_by_Newton(distorted_points, undistorted_guess=undistorted_guess, **kwargs)



    def _jacobian_distort(self, normalized_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        .. seealso::

            - :func:`pydistort.Distortion._jacobian_distort`

        The input normalized points shape is `(2, Npoints)`.
        The output Jacobian matrix shape is `(2, 2, Npoints)`.

        Lets :math:`f(x,y) = g(\theta(x,y), \rho(x,y))`. The first derivative of :math:`f` with respect to :math:`x` is:

        .. math::

            \frac{\partial f}{\partial x} = \frac{\partial g}{\partial \theta} \frac{\partial \theta}{\partial x} + \frac{\partial g}{\partial \rho} \frac{\partial \rho}{\partial x}

        where :

        .. math::

            \frac{\partial \theta}{\partial x} = -\frac{y}{\rho^2} \quad \text{and} \quad \frac{\partial \rho}{\partial x} = \frac{x}{\rho}
        
        A similar expression can be derived for the derivative with respect to :math:`y`.

        In our case, the derivative of the first coordinate of the distorted points along the x-direction is given by:

        .. math::

            \frac{\partial x_{\text{dis}}}{\partial x_{\text{nor}}} = 1 + \frac{-y_{\text{nor}}}{\rho^2} \sum_{m=0}^{\infty} \sum_{n=m}^{\infty} \left( a_{m,n} \cos(m \theta) + b_{m,n} \sin(m \theta) \right) \frac{\partial R_{n}^{m}(\rho)}{\partial \rho} + \frac{x_{\text{nor}}}{\rho} \sum_{m=0}^{\infty} \sum_{n=m}^{\infty} \left( b_{m,n} \cos(m \theta) - a_{m,n} \sin(m \theta) \right) R_{n}^{m}(\rho)
        
        A similar expression can be derived for the derivative of the first coordinate of the distorted points along the y-direction and for the second coordinate of the distorted points.

        If a point is outside the unit circle or contains numpy.nan, the jacobian matrix is set to numpy.nan.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points.

        kwargs : dict
            Additional keyword arguments for the distortion model.

        Returns
        -------
        numpy.ndarray
            The Jacobian matrix of the distorted points with shape `(2, 2, Npoints)`. 

        Raises
        ------
        TypeError
            If the normalized points are not a numpy array.
        ValueError
            If the shape of the normalized points is not `(2, Npoints)`.
        """
        return self.jacobian_distort_by_differential(normalized_points, **kwargs)

        # Check the input parameters
        self.check_array_2D_Npoints(normalized_points)

        # Get the kwargs parameters
        kwargs = self.local_kwargs(**kwargs)

        # Extract the number of points
        Npoints = normalized_points.shape[1]

        # Case 0. points contains nan values or is not in the unit circle
        normalized_domain_mask = self._normalized_domain_mask(normalized_points, **kwargs)
        nan_mask = self.nan_1D_mask(normalized_points, axis=1)
        valid_mask = numpy.logical_and(normalized_domain_mask, numpy.logical_not(nan_mask))

        if not numpy.all(valid_mask):
            # Recursively call the function for the valid points
            jacobian = numpy.full((2, 2, Npoints), numpy.nan)
            jacobian[:, :, valid_mask] = self._jacobian_distort(normalized_points[:, valid_mask], **kwargs)
            return jacobian

        # --------------------------------------

        # Compute the polar coordinates
        polar_points = self.cartesian_to_polar(normalized_points, **kwargs)
        rho = polar_points[0, :]
        theta = polar_points[1, :]
        x = normalized_points[0, :]
        y = normalized_points[1, :]

        # To reduce the number of computations, we compute the following terms first :
        #
        # 1. sum_term_xdis_drho :
        #
        # .. math::
        #
        #     \sum_{m=0}^{\infty} \sum_{n=m}^{\infty} \left( a_{m,n} \cos(m \theta) + b_{m,n} \sin(m \theta) \right) \frac{\partial R_{n}^{m}(\rho)}{\partial \rho}
        #
        # 2. sum_term_xdis_dtheta :
        #
        # .. math::
        #
        #     \sum_{m=0}^{\infty} \sum_{n=m}^{\infty} \left( b_{m,n} \cos(m \theta) - a_{m,n} \sin(m \theta) \right) R_{n}^{m}(\rho)
        #
        # 3. sum_term_ydis_drho :
        #
        # .. math::
        #
        #     \sum_{m=0}^{\infty} \sum_{n=m}^{\infty} \left( c_{m,n} \cos(m \theta) + d_{m,n} \sin(m \theta) \right) \frac{\partial R_{n}^{m}(\rho)}{\partial \rho}
        #
        # 4. sum_term_ydis_dtheta :
        #
        # .. math::
        #
        #     \sum_{m=0}^{\infty} \sum_{n=m}^{\infty} \left( d_{m,n} \cos(m \theta) - c_{m,n} \sin(m \theta) \right) R_{n}^{m}(\rho)
        #

        # Compute the cosine and sine terms
        cos_theta = numpy.cos(theta)
        sin_theta = numpy.sin(theta)

        # Create the 4 terms for the Jacobian matrix
        sum_term_xdis_drho = numpy.zeros((Npoints,))
        sum_term_xdis_dtheta = numpy.zeros((Npoints,))
        sum_term_ydis_drho = numpy.zeros((Npoints,))
        sum_term_ydis_dtheta = numpy.zeros((Npoints,))

        # Loop over the Zernike coefficients
        for m in range(self.Mzer):
            for n in range(m, self.Nzer):
                if (n - m) % 2 == 0 and (self.a[m, n] != 0 or self.b[m, n] != 0 or self.c[m, n] != 0 or self.d[m, n] != 0):
                    # Compute the radial polynomial and its derivative
                    radial_polynomial = self.radial_polynomials(rho, n, m)
                    derivative_radial_polynomial = self.derivative_radial_polynomials(rho, n, m)

                    # Construct the terms for the Jacobian matrix
                    sum_term_xdis_drho += (self.a[m, n] * cos_theta + self.b[m, n] * sin_theta) * derivative_radial_polynomial
                    sum_term_xdis_dtheta += (self.b[m, n] * cos_theta - self.a[m, n] * sin_theta) * radial_polynomial
                    sum_term_ydis_drho += (self.c[m, n] * cos_theta + self.d[m, n] * sin_theta) * derivative_radial_polynomial
                    sum_term_ydis_dtheta += (self.d[m, n] * cos_theta - self.c[m, n] * sin_theta) * radial_polynomial
        
        # Construct the derivative of rho and theta with respect to x and y
        drho_dx = x / rho
        drho_dy = y / rho
        dtheta_dx = -y / rho**2
        dtheta_dy = x / rho**2

        # Construct the Jacobian matrix
        jacobian = numpy.zeros((2, 2, Npoints))
        jacobian[0, 0, :] = 1 + drho_dx * sum_term_xdis_drho + dtheta_dx * sum_term_xdis_dtheta
        jacobian[0, 1, :] = drho_dy * sum_term_xdis_drho + dtheta_dy * sum_term_xdis_dtheta
        jacobian[1, 0, :] = drho_dx * sum_term_ydis_drho + dtheta_dx * sum_term_ydis_dtheta
        jacobian[1, 1, :] = 1 + drho_dy * sum_term_ydis_drho + dtheta_dy * sum_term_ydis_dtheta

        return jacobian



    def _jacobian_undistort(self, distorted_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""

        .. seealso::

            - :func:`pydistort.Distortion._jacobian_undistort`
            - :func:`pydistort.Distortion.jacobian_undistort_by_inversion`

        Computes the Jacobian matrix of the undistortion model for a set of distorted image points.

        The input distorted points are an array of shape (2, ``Npoints``) where ``Npoints`` is the number of points.
        The output Jacobian matrix must have the shape (2, 2, ``Npoints``).

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted image points for which to compute the Jacobian matrix.

        kwargs : dict
            Additional keyword arguments for the distortion model.

        Returns
        -------
        numpy.ndarray
            The Jacobian matrix of the undistortion model with shape (2, 2, ``Npoints``).
        """
        return self.jacobian_undistort_by_inversion(distorted_points, **kwargs)

