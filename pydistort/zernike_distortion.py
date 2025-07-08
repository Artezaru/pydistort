from typing import Optional, Tuple
from numbers import Integral, Number
import numpy
from pyzernike import xy_zernike_polynomial_up_to_order, zernike_order_to_index

from .core.distortion import Distortion


class ZernikeDistortion(Distortion):
    r"""
    Class to apply distortion with a Zernike polynomial model.

    Distort the given ``normalized_points`` using the distortion model to obtain the ``distorted_points``.

    .. math::

        x_D = \text{distort}(x_N, \lambda_1, \lambda_2, \lambda_3, \ldots)

    The distorted points can be calculated using the normalized points and the Zernike coefficients as follows:

    .. math::

        x_{D} = x_{N} + \sum_{n=0}^{N_{zer}} \sum_{m=-n}^{n} C^{x}_{n,m} Z_{nm}(\rho, \theta)

    .. math::

        y_{D} = y_{N} + \sum_{n=0}^{N_{zer}} \sum_{m=-n}^{n} C^{y}_{n,m} Z_{nm}(\rho, \theta)

    where :math:`Z_{nm}(\rho, \theta)` are the Zernike polynomials, :math:`C^{x}_{n,m}` and :math:`C^{y}_{n,m}` are the
    Zernike coefficients for the x and y coordinates, respectively, and :math:`\rho` and :math:`\theta` are the
    polar coordinates of the normalized points in the defined unit ellipse with radius :math:`R_x, R_y` and center :math:`(x_0, y_0)`:

    .. math::

        \rho = \sqrt{\left(\frac{x_{N} - x_{0}}{R_x}\right)^2 + \left(\frac{y_{N} - y_{0}}{R_y}\right)^2}

    .. math::

        \theta = \arctan2(y_{N} - y_{0}, x_{N} - x_{0})

    .. note::

        For more informations about Zernike polynomials, see the package pyzernike (https://github.com/Artezaru/pyzernike).

    Only coeffcients for :math:`n \leq N_{zer}` and :math:`m \in [-n, n]` and :math:`n-m \equiv 0 \mod 2` are stored.

    The coefficients are storred in a ``parameters`` 1D-array with the OSA/ANSI standard indices but with a x/y separation:

    - C^{x}_{0,0}, parameters[0] for the x coordinate :math:`n=0, m=0`
    - C^{y}_{0,0}, parameters[1] for the y coordinate :math:`n=0, m=0`
    - C^{x}_{1,-1}, parameters[2] for the x coordinate :math:`n=1, m=-1`
    - C^{y}_{1,-1}, parameters[3] for the y coordinate :math:`n=1, m=-1`
    - C^{x}_{1,1}, parameters[4] for the x coordinate :math:`n=1, m=1`
    - C^{y}_{1,1}, parameters[5] for the y coordinate :math:`n=1, m=1`
    - C^{x}_{2,-2}, parameters[6] for the x coordinate :math:`n=2, m=-2`
    - C^{y}_{2,-2}, parameters[7] for the y coordinate :math:`n=2, m=-2`
    - C^{x}_{2,0}, parameters[8] for the x coordinate :math:`n=2, m=0`
    - C^{y}_{2,0}, parameters[9] for the y coordinate :math:`n=2, m=0`
    - C^{x}_{2,2}, parameters[10] for the x coordinate :math:`n=2, m=2`
    - C^{y}_{2,2}, parameters[11] for the y coordinate :math:`n=2, m=2`
    - ...

    If the number of input parameters is not equal to the number of parameters required by the model, the other parameters are set to 0.

    The number of parameters is given by the formula:

    .. math::

        N_{params} = (N_{zer}+1)(N_{zer}+2)

    +---------------------------+---------------------------------+-------------------------------------+
    | Ordre of Zernike ``Nzer`` | Nparameters for X or Y          | Nparameters in model ``Nparams``    |
    +===========================+=================================+=====================================+
    | None                      | 0                               | 0                                   |
    +---------------------------+---------------------------------+-------------------------------------+
    | 0                         | 1                               | 2                                   |
    +---------------------------+---------------------------------+-------------------------------------+
    | 1                         | 3                               | 6                                   |
    +---------------------------+---------------------------------+-------------------------------------+
    | 2                         | 6                               | 12                                  |
    +---------------------------+---------------------------------+-------------------------------------+
    | 3                         | 10                              | 20                                  |
    +---------------------------+---------------------------------+-------------------------------------+
    | 4                         | 15                              | 30                                  |
    +---------------------------+---------------------------------+-------------------------------------+

    .. warning::

        If the ordre of the zernike polynomials ``Nzer`` is given during instantiation, the given parameters are truncated or extended to the given number of parameters. Same for the number of parameters ``Nparams``.

    To compute the Distortion, the user must define the unit circle in which the normalized points are defined.
    The unit circle is defined by the radius :math:`R` and the center :math:`(x_{0}, y_{0})`.

    Parameters
    ----------
    parameters : numpy.ndarray, optional
        The parameters of the distortion model. If None, no distortion is applied. The default is None.

    Nzer : int, optional
        The order of the Zernike polynomials. If None, the order is set according to the number of parameters. The default is None.
        Only use ``Nzer`` or ``Nparams``, not both.

    Nparams : int, optional
        The number of parameters of the distortion model. If None, the number of parameters is set according to the order of the Zernike polynomials. The default is None.
        Only use ``Nzer`` or ``Nparams``, not both.

    radius : float, optional
        The radius of the unit circle in which the normalized points are defined. If None, the radius is set to 1.0. The default is None.
        This radius is used to normalize the points in the unit circle. If given, ``radius_x`` and ``radius_y`` must be None.

    radius_x : float, optional
        The radius of the unit ellipse in which the normalized points are defined along x-axis. If None, the radius is set to 1.0. The default is None.

    radius_y : float, optional
        The radius of the unit ellipse in which the normalized points are defined along y-axis. If None, the radius is set to 1.0. The default is None.

    center : Tuple[float, float], optional
        The center of the unit circle in which the normalized points are defined. If None, the center is set to (0.0, 0.0). The default is None.

    Examples
    --------
    Create an distortion object with a given model:

    .. code-block:: python

        import numpy as np
        from pydistort import ZernikeDistortion

        # Create a distortion object with 8 parameters
        distortion = ZernikeDistortion(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])) # Model with Nzer=1, -> Nparams=6

    Then you can use the distortion object to distort ``normalized_points``:

    .. code-block:: python

        normalized_points = np.array([[0.1, 0.2],
                                       [0.3, 0.4],
                                       [0.5, 0.6]]) # shape (3, 2)

        result = distortion.transform(normalized_points): #alias distort is also available
        distorted_points = result.distorted_points # shape (3, 2) -> distorted points in (normalized) image coordinates
        print(distorted_points)

    You can also access to the jacobian of the distortion transformation:

    .. code-block:: python

        result = distortion.transform(distorted_points, dx=True, dp=True)
        distorted_points_dx = result.jacobian_dx  # Jacobian of the image points with respect to the distorted points # Shape (..., 2, 3)
        distorted_points_dp = result.jacobian_dp  # Jacobian of the image points with respect to the intrinsic parameters # Shape (..., 2, Nparams)

    The parameters are ordered as given in the model description above.
    """
    def __init__(self, parameters: Optional[numpy.ndarray] = None, Nzer: Optional[Integral] = None, Nparams: Optional[Integral] = None, radius: Optional[float] = None, radius_x: Optional[float] = None, radius_y: Optional[float] = None, center: Optional[Tuple[float, float]] = None) -> None:
        super().__init__()
        self.parameters = parameters
        if Nparams is not None and Nzer is not None:
            raise ValueError("Only use Nzer or Nparams, not both.")
        if Nparams is not None:
            self.Nparams = Nparams
        if Nzer is not None:
            self.Nzer = Nzer
        
        if radius is not None and (radius_x is not None or radius_y is not None):
            raise ValueError("If radius is given, radius_x and radius_y must be None.")
        
        if radius is not None:
            radius_x = radius
            radius_y = radius

        self.radius_x = radius_x if radius_x is not None else 1.0
        self.radius_y = radius_y if radius_y is not None else 1.0
        self.center = center if center is not None else numpy.array([0.0, 0.0], dtype=numpy.float64)

    # =================================================================
    # Parameters
    # =================================================================
    @property
    def parameters(self) -> Optional[numpy.ndarray]:
        r"""
        Get the parameters of the distortion model.

        Returns
        -------
        numpy.ndarray
            The parameters of the distortion model.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: Optional[numpy.ndarray]) -> None:
        r"""
        Set the parameters of the distortion model.
        If None, no distortion is applied.

        The number of parameters should be a 1D numpy array with a size in [None, 2, 6, 12, 20, 30, ...].

        .. math::

            N_{params} = (N_{zer}+1)(N_{zer}+2)

        If the number of input parameters is not equal to the number of parameters required by the model, the other parameters are set to 0.

        The parameters are set in the following order:

        - N = 0 parameters : similar than None
        - N = 2 parameters : :math:`C^{x}_{0,0}, C^{y}_{0,0}` : Zernike coefficients for the x and y coordinates with order 0
        - N = 6 parameters : :math:`C^{x}_{0,0}, C^{y}_{0,0}, C^{x}_{1,-1}, C^{y}_{1,-1}, C^{x}_{1,1}, C^{y}_{1,1}` : Zernike coefficients for the x and y coordinates with order 0 and 1
        - N = 12 parameters : :math:`C^{x}_{0,0}, C^{y}_{0,0}, C^{x}_{1,-1}, C^{y}_{1,-1}, C^{x}_{1,1}, C^{y}_{1,1}, C^{x}_{2,-2}, C^{y}_{2,-2}, C^{x}_{2,0}, C^{y}_{2,0}, C^{x}_{2,2}, C^{y}_{2,2}` : Zernike coefficients for the x and y coordinates with order 0, 1 and 2
        - N = 20 parameters : ...

        To easily use the parameters, you can use the methods ``get_Cx``, ``set_Cx``, ``get_Cy`` and ``set_Cy`` to get and set the Zernike coefficients for the x and y coordinates.

        Parameters
        ----------
        parameters : numpy.ndarray, optional
            The parameters of the distortion model. If None, no distortion is applied. The default is None.

        Raises
        -------
        ValueError
            If the parameters is not a 1D numpy array.
        """
        if parameters is not None:
            parameters = numpy.asarray(parameters, dtype=numpy.float64)
            if parameters.ndim != 1:
                raise ValueError("The parameters should be a 1D numpy array.")
            # Extend the number of parameters to a valid number
            Nzer = 0
            while (Nzer + 1) * (Nzer + 2) < parameters.size:
                Nzer += 1
            Nparams = (Nzer + 1) * (Nzer + 2)
            # Extend the parameters to the next valid size
            if Nparams > parameters.size:
                parameters = numpy.concatenate((parameters, numpy.zeros(Nparams - parameters.size)))
            # Set to None if the number of parameters is 0
            if parameters.size == 0:
                parameters = None
        self._parameters = parameters

    @property
    def Nparams(self) -> int:
        r"""
        Get the number of parameters of the distortion model.

        Returns
        -------
        int
            The number of parameters of the distortion model.
        """
        if self.parameters is None:
            return 0
        else:
            return self.parameters.size
        
    @Nparams.setter
    def Nparams(self, value: Integral) -> None:
        r"""
        Set the number of parameters of the distortion model.
        
        The given number of parameters must be in [0, 2, 6, 12, 20, 30, ...].

        .. math::

            N_{params} = (N_{zer}+1)(N_{zer}+2)

        If the given number of parameters is less than the current number of parameters, the parameters are truncated.
        If the given number of parameters is greater than the current number of parameters, the parameters are extended with zeros.

        Parameters
        ----------
        value : int
            The number of parameters of the distortion model.
        """
        if not isinstance(value, Integral):
            raise TypeError("The number of parameters should be an integer.")
        if value < 0:
            raise ValueError("The number of parameters should be a non-negative integer.")
        if value == 0:
            self.parameters = None
            return
        
        Nzer = 0
        while (Nzer + 1) * (Nzer + 2) < value:
            Nzer += 1

        # Check if the number of parameters is valid
        if (Nzer + 1) * (Nzer + 2) != value:
            raise ValueError("The number of parameters should be in [0, 2, 6, 12, 20, 30, ...].")
        
        # If parameters is None, create a new array of zeros
        if self.parameters is None:
            self.parameters = numpy.zeros(value)
            return
        
        if value < self.Nparams:
            self.parameters = self.parameters[:value]
        elif value > self.Nparams:
            self.parameters = numpy.concatenate((self.parameters, numpy.zeros(value - self.Nparams)))

    @property
    def Nzer(self) -> Optional[Integral]:
        r"""
        Get the order of the Zernike polynomials.

        Returns
        -------
        Optional[Integral]
            The order of the Zernike polynomials. If no distortion is applied, returns None.
        """
        if self.parameters is None:
            return None
        
        Nparams = self.Nparams
        Nzer = 0
        while (Nzer + 1) * (Nzer + 2) < Nparams:
            Nzer += 1
        return Nzer
    
    @Nzer.setter
    def Nzer(self, value: Integral) -> None:
        r"""
        Set the order of the Zernike polynomials.

        The given order must be in [0, 1, 2, 3, 4, ...].

        If the given order is less than the current order, the parameters are truncated.
        If the given order is greater than the current order, the parameters are extended with zeros.

        Parameters
        ----------
        value : int
            The order of the Zernike polynomials.
        """
        if not isinstance(value, Integral):
            raise TypeError("The order of the Zernike polynomials should be an integer.")
        if value < 0:
            raise ValueError("The order of the Zernike polynomials should be a non-negative integer.")
        
        value = int(value)
        self.Nparams = (value + 1) * (value + 2)

    @property
    def radius(self) -> float:
        r"""
        Get the radius of the unit circle in which the normalized points are defined.

        Returns
        -------
        float
            The radius of the unit circle.
        """
        if abs(self._radius_x - self._radius_y) > 1e-6:
            raise ValueError("The radius_x and radius_y should be the same for the unit circle. Please set the same value for both or use radius_x and radius_y separately for ellipses.")
        return self._radius_x
    
    @radius.setter
    def radius(self, value: float) -> None:
        r"""
        Set the radius of the unit circle in which the normalized points are defined.

        The value will be propagated to both `radius_x` and `radius_y`.

        Parameters
        ----------
        value : float
            The radius of the unit circle.
        """
        if not isinstance(value, Number):
            raise TypeError("The radius should be a float or an integer.")
        if value <= 0:
            raise ValueError("The radius should be a positive number.")
        self._radius_x = float(value)
        self._radius_y = float(value)

    @property
    def radius_x(self) -> float:
        r"""
        Get the radius of the unit ellipse in which the normalized points are defined along x-axis.

        Returns
        -------
        float
            The radius of the unit ellipse along x-axis.
        """
        return self._radius_x
    
    @radius_x.setter
    def radius_x(self, value: float) -> None:
        r"""
        Set the radius of the unit ellipse in which the normalized points are defined along x-axis.

        Parameters
        ----------
        value : float
            The radius of the unit ellipse along x-axis.
        """
        if not isinstance(value, Number):
            raise TypeError("The radius_x should be a float or an integer.")
        if value <= 0:
            raise ValueError("The radius_x should be a positive number.")
        self._radius_x = float(value)

    @property
    def radius_y(self) -> float:
        r"""
        Get the radius of the unit ellipse in which the normalized points are defined along y-axis.

        Returns
        -------
        float
            The radius of the unit ellipse along y-axis.
        """
        return self._radius_y
    
    @radius_y.setter
    def radius_y(self, value: float) -> None:
        r"""
        Set the radius of the unit ellipse in which the normalized points are defined along y-axis.

        Parameters
        ----------
        value : float
            The radius of the unit ellipse along y-axis.
        """
        if not isinstance(value, Number):
            raise TypeError("The radius_y should be a float or an integer.")
        if value <= 0:
            raise ValueError("The radius_y should be a positive number.")
        self._radius_y = float(value)

    @property
    def center(self) -> numpy.ndarray:
        r"""
        Get the center of the unit circle in which the normalized points are defined.

        Returns
        -------
        numpy.ndarray
            The center of the unit circle as a 1D array with two elements (x, y).
        """
        return numpy.asarray(self._center, dtype=numpy.float64)
    
    @center.setter
    def center(self, value: numpy.ndarray) -> None:
        r"""
        Set the center of the unit circle in which the normalized points are defined.

        Parameters
        ----------
        value : numpy.ndarray
            The center of the unit circle as a 1D array with two elements (x, y).
            The array should be of shape (2,) and of type float or int.
        """
        value = numpy.asarray(value, dtype=numpy.float64)
        if value.ndim != 1 or value.size != 2:
            raise ValueError("The center should be a 1D array with two elements (x, y).")
        if not numpy.isfinite(value).all():
            raise ValueError("The center coordinates should be finite numbers.")
        self._center = value

    @property
    def x0(self) -> float:
        r"""
        Get the x coordinate of the center of the unit circle in which the normalized points are defined.

        Returns
        -------
        float
            The x coordinate of the center of the unit circle.
        """
        return self._center[0]
    
    @x0.setter
    def x0(self, value: float) -> None:
        r"""
        Set the x coordinate of the center of the unit circle in which the normalized points are defined.

        Parameters
        ----------
        value : float
            The x coordinate of the center of the unit circle.
        """
        if not isinstance(value, Number):
            raise TypeError("The x0 should be a float or an integer.")
        self._center[0] = float(value)

    @property
    def y0(self) -> float:
        r"""
        Get the y coordinate of the center of the unit circle in which the normalized points are defined.

        Returns
        -------
        float
            The y coordinate of the center of the unit circle.
        """
        return self._center[1]
    
    @y0.setter
    def y0(self, value: float) -> None:
        r"""
        Set the y coordinate of the center of the unit circle in which the normalized points are defined.

        Parameters
        ----------
        value : float
            The y coordinate of the center of the unit circle.
        """
        if not isinstance(value, Number):
            raise TypeError("The y0 should be a float or an integer.")
        self._center[1] = float(value)

    # =================================================================
    # Distortion Model Coefficients
    # =================================================================
    @property
    def parameters_x(self) -> Optional[numpy.ndarray]:
        r"""
        Get the Zernike coefficients for the x coordinate.

        Returns
        -------
        Optional[numpy.ndarray]
            The Zernike coefficients for the x coordinate. If no distortion is applied, returns None.
        """
        if self.parameters is None:
            return None
        return self.parameters[0::2]
    
    @parameters_x.setter
    def parameters_x(self, value: numpy.ndarray) -> None:
        r"""
        Set the Zernike coefficients for the x coordinate.

        .. warning::

            The value must be a 1D numpy array with the same number of elements as the number of parameters requested by the model.
            Use ``parameters``, ``Nzer`` or ``Nparams`` to change the model !

        Parameters
        ----------
        value : numpy.ndarray
            The Zernike coefficients for the x coordinate.
        """
        if self.parameters is None:
            raise ValueError("No distortion model is defined. Set the parameters first.")
        value = numpy.asarray(value, dtype=numpy.float64)
        if not value.ndim == 1:
            raise ValueError("The Zernike coefficients for the x coordinate should be a 1D numpy array.")
        if not value.size == self.Nparams // 2:
            raise ValueError(f"The number of Zernike coefficients for the x coordinate should be {self.Nparams // 2}.")
        if not numpy.all(numpy.isfinite(value)):
            raise ValueError("The Zernike coefficients for the x coordinate should be finite numbers.")
        self.parameters[0::2] = value

    @property
    def parameters_y(self) -> Optional[numpy.ndarray]:
        r"""
        Get the Zernike coefficients for the y coordinate.

        Returns
        -------
        Optional[numpy.ndarray]
            The Zernike coefficients for the y coordinate. If no distortion is applied, returns None.
        """
        if self.parameters is None:
            return None
        return self.parameters[1::2]
    
    @parameters_y.setter
    def parameters_y(self, value: numpy.ndarray) -> None:
        r"""
        Set the Zernike coefficients for the y coordinate.

        .. warning::

            The value must be a 1D numpy array with the same number of elements as the number of parameters requested by the model.
            Use ``parameters``, ``Nzer`` or ``Nparams`` to change the model !

        Parameters
        ----------
        value : numpy.ndarray
            The Zernike coefficients for the y coordinate.
        """
        if self.parameters is None:
            raise ValueError("No distortion model is defined. Set the parameters first.")
        value = numpy.asarray(value, dtype=numpy.float64)
        if not value.ndim == 1:
            raise ValueError("The Zernike coefficients for the y coordinate should be a 1D numpy array.")
        if not value.size == self.Nparams // 2:
            raise ValueError(f"The number of Zernike coefficients for the y coordinate should be {self.Nparams // 2}.")
        if not numpy.all(numpy.isfinite(value)):
            raise ValueError("The Zernike coefficients for the y coordinate should be finite numbers.")
        self.parameters[1::2] = value

    # =================================================================
    # Methods to set and get the coefficients
    # =================================================================
    def get_index(self, n: Integral, m: Integral, coord: str) -> int:
        r"""
        Get the index of the Zernike coefficient for the given order and azimuthal frequency.

        .. math::

            j = n(n+2) + m + (0 \text{ if } coord = 'x' \text{ else } 1)

        Parameters
        ----------
        n : int
            The order of the Zernike polynomial.
        m : int
            The azimuthal frequency of the Zernike polynomial.
        coord : str
            The coordinate ('x' or 'y') for which to get the index.

        Returns
        -------
        int
            The index of the Zernike coefficient in the parameters array.
        """
        if coord not in ['x', 'y']:
            raise ValueError("The coordinate should be 'x' or 'y'.")
        if not isinstance(n, Integral) or not isinstance(m, Integral):
            raise TypeError("The order and azimuthal frequency should be integers.")
        if n < 0 or abs(m) > n or (n - m) % 2 != 0:
            raise ValueError("Invalid order or azimuthal frequency for Zernike polynomial.")
        if self.parameters is None:
            raise ValueError("No distortion model is defined.")
        if n > self.Nzer:
            raise ValueError(f"The order of the Zernike polynomial {n} is greater than the defined order {self.Nzer}.")
        
        index = n * (n + 2) + m
        if coord == 'y':
            index += 1
        return index
        
    def set_Cx(self, n: Integral, m: Integral, value: Number) -> None:
        r"""
        Set the Zernike coefficient for the x coordinate.

        Parameters
        ----------
        n : int
            The order of the Zernike polynomial.
        m : int
            The azimuthal frequency of the Zernike polynomial.
        value : float
            The value of the Zernike coefficient.
        """
        index = self.get_index(n, m, 'x')
        self.parameters[index] = value

    def get_Cx(self, n: Integral, m: Integral) -> Number:
        r"""
        Get the Zernike coefficient for the x coordinate.

        Parameters
        ----------
        n : int
            The order of the Zernike polynomial.
        m : int
            The azimuthal frequency of the Zernike polynomial.

        Returns
        -------
        float
            The value of the Zernike coefficient.
        """
        index = self.get_index(n, m, 'x')
        return self.parameters[index]
        
    def set_Cy(self, n: Integral, m: Integral, value: Number) -> None:
        r"""
        Set the Zernike coefficient for the y coordinate.

        Parameters
        ----------
        n : int
            The order of the Zernike polynomial.
        m : int
            The azimuthal frequency of the Zernike polynomial.
        value : float
            The value of the Zernike coefficient.
        """
        index = self.get_index(n, m, 'y')
        self.parameters[index] = value

    def get_Cy(self, n: Integral, m: Integral) -> Number:
        r"""
        Get the Zernike coefficient for the y coordinate.

        Parameters
        ----------
        n : int
            The order of the Zernike polynomial.
        m : int
            The azimuthal frequency of the Zernike polynomial.

        Returns
        -------
        float
            The value of the Zernike coefficient.
        """
        index = self.get_index(n, m, 'y')
        return self.parameters[index]

    # =================================================================
    # Display the distortion model
    # =================================================================
    def __repr__(self) -> str:
        r"""
        Return a string representation of the distortion model.

        Returns
        -------
        str
            The string representation of the distortion model.
        """
        if self.parameters is None:
            return "ZernikeDistortion: No distortion model"
        
        Nzer = self.Nzer
        Nparams = self.Nparams
        parameters_str = f"ZernikeDistortion: {Nparams} parameters (Nzer={Nzer})\n"
        for n in range(Nzer + 1):
            for m in range(-n, n + 1, 2):
                parameters_str += f"  Cx[n={n}, m={m}] = {self.parameters[self.get_index(n, m, 'x')]:.6f}\n"
                parameters_str += f"  Cy[n={n}, m={m}] = {self.parameters[self.get_index(n, m, 'y')]:.6f}\n"
        return parameters_str
            

    # =================================================================
    # Implementation of the abstract methods
    # =================================================================
    def is_set(self) -> bool:
        r"""
        Check if the distortion model is set.

        Returns
        -------
        bool
            True if the distortion model is set, False otherwise.
        """
        # The distortion model is always set (Nparams = 0 just means no distortion)
        return True 
    

    def _transform(self, normalized_points: numpy.ndarray, *, dx: bool = False, dp: bool = False) -> tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        This method is called by the :meth:`pydistort.core.Transform.transform` method to perform the distortion transformation.
        This method allows to transform the ``normalized_points`` to the ``distorted_points`` using the distortion model.

        .. note::

            For ``_transform`` the input must have shape (Npoints, 2) with float64 type.
            The output has shape (Npoints, 2) for the image points and (Npoints, 2, 2) for the jacobian with respect to the normalized points and (Npoints, 2, 4) for the jacobian with respect to the distortion parameters.

        The equation used for the transformation is given in the main documentation of the class.

        We have (only for the x coordinate):

        .. math::

            x_{D} = x_{N} + \sum_{n=0}^{N_{zer}} \sum_{m=-n}^{n} C^{x}_{n,m} Z_{nm}(\rho, \theta)

        .. math::

            x_{D} = x_{N} + \sum_{n=0}^{N_{zer}} \sum_{m=-n}^{n} C^{x}_{n,m} Z_{nm}(\sqrt{\left(\frac{x_{N} - x_{0}}{R_x}\right)^2 + \left(\frac{y_{N} - y_{0}}{R_y}\right)^2}, \arctan2(\frac{y_{N} - y_{0}}{R_y}, \frac{x_{N} - x_{0}}{R_x})))

        The derivative of the distorted points with respect to the normalized points is given by:

        .. math::

            \frac{\partial x_{D}}{\partial x_{N}} = 1 + \sum_{n=0}^{N_{zer}} \sum_{m=-n}^{n} C^{x}_{n,m} \frac{\partial Z_{nm}}{\partial x_{N}}

        .. math::

            \frac{\partial x_{D}}{\partial y_{N}} = \sum_{n=0}^{N_{zer}} \sum_{m=-n}^{n} C^{x}_{n,m} \frac{\partial Z_{nm}}{\partial y_{N}}

        Where:

        .. math::

            \frac{\partial Z_{nm}}{\partial x_{N}} = \frac{\partial Z_{nm}}{\partial \rho} \cdot \frac{\partial \rho}{\partial x_{N}} + \frac{\partial Z_{nm}}{\partial \theta} \cdot \frac{\partial \theta}{\partial x_{N}}

        .. math::

            \frac{\partial Z_{nm}}{\partial y_{N}} = \frac{\partial Z_{nm}}{\partial \rho} \cdot \frac{\partial \rho}{\partial y_{N}} + \frac{\partial Z_{nm}}{\partial \theta} \cdot \frac{\partial \theta}{\partial y_{N}}

        .. seealso::

            Package ``pyzernike`` (https://github.com/Artezaru/pyzernike) for the implementation of the Zernike polynomials and their derivatives.

        .. warning::

            This method is not designed to be used directly for the transformation of points.
            No checks are performed on the input points, so it is the user's responsibility to ensure that the input points are valid.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            Array of normalized points to be transformed with shape (Npoints, 2).

        dx : bool, optional
            If True, the Jacobian of the distorted points with respect to the normalized points is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 2, 2).

        dp : bool, optional
            If True, the Jacobian of the distorted points with respect to the distortion parameters is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 2, Nparams).

        Returns
        -------
        distorted_points : numpy.ndarray
            The transformed distorted points in normalized coordinates. It will be a 2D array of shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The Jacobian of the distorted points with respect to the normalized points if ``dx`` is True. Otherwise None.
            It will be a 2D array of shape (Npoints, 2, 2).

        jacobian_dp : Optional[numpy.ndarray]
            The Jacobian of the distorted points with respect to the distortion parameters if ``dp`` is True. Otherwise None.
            It will be a 2D array of shape (Npoints, 2, Nparams).
        """
        # Prepare the inputs data for distortion
        x_N = normalized_points[:, 0] # shape (Npoints,)
        y_N = normalized_points[:, 1] # shape (Npoints,)
        Nparams = self.Nparams

        # Prepare the output jacobian arrays
        if dx:
            jacobian_dx = numpy.tile(numpy.eye(2, dtype=numpy.float64), (x_N.size, 1, 1))  # shape (Npoints, 2, 2)
        else:
            jacobian_dx = None

        if dp:
            jacobian_dp = numpy.empty((x_N.size, 2, Nparams), dtype=numpy.float64)
        else:
            jacobian_dp = None

        # If no distortion model is defined, return the normalized points
        if self.parameters is None:
            return normalized_points.copy(), jacobian_dx, jacobian_dp
        
        # Initialize the distorted points
        x_D = x_N.copy()
        y_D = y_N.copy()

        # Construct the derivatives to compute the Jacobian
        if dx:
            list_dx = [0, 1, 0]
            list_dy = [0, 0, 1]
        else:
            list_dx = [0]
            list_dy = [0]
                
        # Construct the zernike polynomial values and their derivatives
        zernike_results = xy_zernike_polynomial_up_to_order(x_N, y_N, order=self.Nzer, Rx=self.radius_x, Ry=self.radius_y, x0=self.center[0], y0=self.center[1], x_derivative=list_dx, y_derivative=list_dy)

        # Initialize the distorted points and jacobians
        for n in range(self.Nzer + 1):
            for m in range(-n, n + 1, 2):
                zernike_index = zernike_order_to_index(n=[n], m=[m])[0]
                # Get the Zernike polynomial value
                Z_nm = zernike_results[0][zernike_index]
                
                # Get the dérivatives of the Zernike polynomial if requested
                if dx:
                    Z_nm_dx = zernike_results[1][zernike_index]
                    Z_nm_dy = zernike_results[2][zernike_index]

                # Extract the coefficients for the x and y coordinates
                index_x = self.get_index(n, m, 'x')
                Cx = self.parameters[index_x]
                index_y = self.get_index(n, m, 'y')
                Cy = self.parameters[index_y]

                # Update the distorted points
                x_D += Cx * Z_nm
                y_D += Cy * Z_nm

                if dx:
                    # Compute the Jacobian with respect to the normalized points
                    jacobian_dx[:, 0, 0] += Cx * Z_nm_dx
                    jacobian_dx[:, 0, 1] += Cx * Z_nm_dy
                    jacobian_dx[:, 1, 0] += Cy * Z_nm_dx
                    jacobian_dx[:, 1, 1] += Cy * Z_nm_dy

                if dp:
                    # Compute the Jacobian with respect to the distortion parameters
                    jacobian_dp[:, 0, index_x] = Z_nm
                    jacobian_dp[:, 1, index_y] = Z_nm

        # Convert the distorted points back to the original coordinates
        distorted_points = numpy.empty_like(normalized_points, dtype=numpy.float64)
        distorted_points[:, 0] = x_D
        distorted_points[:, 1] = y_D

        # Return the distorted points and the jacobians if requested
        return distorted_points, jacobian_dx, jacobian_dp
    

    
    def _inverse_transform(
        self,
        distorted_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        **kwargs
    ) -> tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        This method is called by the :meth:`pydistort.core.Transform.inverse_transform` method to perform the inverse distortion transformation.
        This method allows to transform the ``distorted_points`` back to the ``normalized_points`` using the distortion model.

        .. note::

            For ``_inverse_transform`` the input must have shape (Npoints, 2) with float64 type.
            The output has shape (Npoints, 2) for the normalized points and the jacobian are always None.

        See the :meth:`pydistort.core.objetcs.Transform.optimize_input_points` method for more details.
        The initial guess is setted to :math:`\mathbf{x}_{N} = \mathbf{x}_{D} - U(\mathbf{x}_{D})``, where :math:`U(\mathbf{x}_{D})` is the distortion filed applied to the distorted points.

        .. warning::

            This method is not designed to be used directly for the transformation of points.
            No checks are performed on the input points, so it is the user's responsibility to ensure that the input points are valid.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            Array of distorted points to be transformed back to normalized points with shape (Npoints, 2).

        dx : bool, optional
            [Always False]
        
        dp : bool, optional
            [Always False]

        **kwargs : dict, optional
            Additional keyword arguments to be passed to the optimization method.

        Returns
        -------
        normalized_points : numpy.ndarray
            The transformed normalized points in normalized coordinates. It will be a 2D array of shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            [Always None]

        jacobian_dp : Optional[numpy.ndarray]
            [Always None]

        """
        if dx or dp:
            print("\n[WARNING]: Undistortion with dx=True or dp=True. The jacobians cannot be computed with this method. They are always None.\n")

        normalized_points = self.optimize_input_points(
            distorted_points,
            guess = 2 * distorted_points - self._transform(distorted_points, dx=False, dp=False)[0],
            _skip = True,  # Skip the checks on the input points
            **kwargs
        )

        return normalized_points, None, None