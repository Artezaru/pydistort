from typing import Optional, Tuple
from numbers import Integral, Number
import numpy
import cv2

from .distortion import Distortion, DistortionResult, UndistortResult

class Cv2Distortion(Distortion):
    r"""
    Class to apply distortion with the OpenCV distortion model.

    Distort the given ``normalized_points`` using the distortion model to obtain the ``distorted_points``.

    .. math::

        x_D = \text{distort}(x_N, \lambda_1, \lambda_2, \lambda_3, \ldots)

    The model of OpenCV is the following one:

    .. math::

        \begin{bmatrix}
        x_D \\
        y_D
        \end{bmatrix}
        =
        \begin{bmatrix}
        x_N \frac{1+k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + 2p_1 x_N y_N + p_2 (r^2 + 2x_N^2) + s_1 r^2 + s_2 r^4 \\
        y_N \frac{1+k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + p_1 (r^2 + 2y_N^2) + 2p_2 x_N y_N + s_3 r^2 + s_4 r^4
        \end{bmatrix}
    
    where :math:`r^2 = x_N^2 + y_N^2` and :math:`k_i` are the radial distortion coefficients, :math:`p_i` are the tangential distortion coefficients and :math:`s_i` are the thin prism distortion coefficients.

    Then a perspective transformation is applied using :math:`\tau_x` and :math:`\tau_y` to obtain the final distorted points.

    .. math::

        \begin{bmatrix}
        x_D \\
        y_D \\
        1
        \end{bmatrix}
        =
        \begin{bmatrix}
        R_{33}(\tau) & 0 & -R_{13}(\tau) \\
        0 & R_{33}(\tau) & -R_{23}(\tau) \\
        0 & 0 & 1
        \end{bmatrix}
        R(\tau)
        \begin{bmatrix}
        x_D \\
        y_D \\
        1
        \end{bmatrix}
    
    where :

    .. math::

        R(\tau) = \begin{bmatrix}
        cos(\tau_y) & sin(\tau_x)sin(\tau_y) & -cos(\tau_x)sin(\tau_y) \\
        0 & cos(\tau_x) & sin(\tau_x) \\
        sin(\tau_y) & -sin(\tau_x)cos(\tau_y) & cos(\tau_x)cos(\tau_y)
        \end{bmatrix}
    
    and :math:`R_{ij}(\tau)` are the elements of the rotation matrix.

    .. seealso::

        - https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html for the OpenCV documentation

    OpenCV can use various models for distortion,

    - N = 4 parameters : :math:`(k_1, k_2, p_1, p_2)` : radial and tangential distortion
    - N = 5 parameters : :math:`(k_1, k_2, p_1, p_2, k_3)` : radial and tangential distortion with third order radial distortion
    - N = 8 parameters : :math:`(k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6)` : radial and tangential distortion with fractional radial distortion
    - N = 12 parameters : :math:`(k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6, s_1, s_2, s_3, s_4)` : radial and tangential distortion with thin prism distortion
    - N = 14 parameters : :math:`(k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6, s_1, s_2, s_3, s_4, \tau_x, \tau_y)` : radial and tangential distortion with thin prism distortion and perspective transformation

    If the number of input parameters is not equal to the number of parameters required by the model, the other parameters are set to 0.

    .. warning::

        If the number of parameters ``Nparams`` is given during instantiation, the given parameters are truncated or extended to the given number of parameters.

    Parameters
    ----------
    parameters : numpy.ndarray, optional
        The parameters of the distortion model. If None, no distortion is applied. The default is None.

    Nparams : int, optional
        The number of parameters of the distortion model. If None, the number of parameters is set to the number of parameters of the model. The default is None.
        Must be in [0, 4, 5, 8, 12, 14] if given.
        
    Examples
    --------

    Create an distortion object with a given model:

    .. code-block:: python

        import numpy as np
        from pydistort import Cv2Distortion

        # Create a distortion object with 8 parameters
        distortion = Cv2Distortion(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])) # Model with 8 parameters

    Then you can use the distortion object to distort ``normalized_points``:

    .. code-block:: python

        normalized_points = np.array([[0.1, 0.2],
                                       [0.3, 0.4],
                                       [0.5, 0.6]]) # shape (3, 2)

        result = distortion.distort(normalized_points) #alias transform is also available
        result.distorted_points # shape (3, 2) -> distorted points in pixel coordinates
        result.jacobian_dx # shape (3, 2, 3) -> jacobian of the distorted points with respect to the normalized points
        result.jacobian_dp # shape (3, 2, Nparams) (here Nparams = 8) -> jacobian of the distorted points with respect to the distortion parameters.

    The distortion parameters are ordered as given below.
    """
    def __init__(self, parameters: Optional[numpy.ndarray] = None, Nparams: Optional[Integral] = None) -> None:
        super().__init__()
        self.parameters = parameters
        if Nparams is not None:
            self.Nparams = Nparams

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

        The number of parameters should be 4, 5, 8, 12 or 14.
        If the number of input parameters is not equal to the number of parameters required by the model, the other parameters are set to 0.

        The parameters are set in the following order:

        - N = 0 parameters : similar than None
        - N = 4 parameters : :math:`(k_1, k_2, p_1, p_2)` : radial and tangential distortion
        - N = 5 parameters : :math:`(k_1, k_2, p_1, p_2, k_3)` : radial and tangential distortion with third order radial distortion
        - N = 8 parameters : :math:`(k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6)` : radial and tangential distortion with fractional radial distortion
        - N = 12 parameters : :math:`(k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6, s_1, s_2, s_3, s_4)` : radial and tangential distortion with thin prism distortion
        - N = 14 parameters : :math:`(k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6, s_1, s_2, s_3, s_4, \tau_x, \tau_y)` : radial and tangential distortion with thin prism distortion and perspective transformation

        Parameters
        ----------
        parameters : numpy.ndarray, optional
            The parameters of the distortion model. If None, no distortion is applied. The default is None.

        Raises
        -------
        ValueError
            If the parameters is not a 1D numpy array.
            If more than 14 parameters are given.
        """
        if parameters is not None:
            parameters = numpy.asarray(parameters, dtype=numpy.float64)
            if parameters.ndim != 1:
                raise ValueError("The parameters should be a 1D numpy array.")
            if parameters.size > 14:
                raise ValueError("The number of parameters of CV2 distortion should be less than or equal to 14.")
            # Extend the number of parameters to a valid number
            valid_sizes = [0, 4, 5, 8, 12, 14]
            index = 0
            while valid_sizes[index] < parameters.size:
                index += 1
            Nparams = valid_sizes[index]
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
        
        The given number of parameters must be in [0, 4, 5, 8, 12, 14].

        If the given number of parameters is less than the current number of parameters, the parameters are truncated.
        If the given number of parameters is greater than the current number of parameters, the parameters are extended with zeros.

        Parameters
        ----------
        value : int
            The number of parameters of the distortion model.
        """
        if not isinstance(value, Integral):
            raise TypeError("The number of parameters should be an integer.")
        if value not in [0, 4, 5, 8, 12, 14]:
            raise ValueError("The number of parameters should be in [0, 4, 5, 8, 12, 14].")
        
        # If parameters is None, create a new array of zeros
        if self.parameters is None:
            self.parameters = numpy.zeros(value)
            return
        
        # Update the number of parameters instead of creating a new array
        if value == 0:
            self.parameters = None
        elif value < self.Nparams:
            self.parameters = self.parameters[:value]
        elif value > self.Nparams:
            self.parameters = numpy.concatenate((self.parameters, numpy.zeros(value - self.Nparams)))
    


    @property
    def k1(self) -> float:
        r"""
        Get the first radial distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 4.
            Set the number of parameters to 4 or more before getting the value.

        Returns
        -------
        float
            The first radial distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 4.
        """
        if self.Nparams < 4: 
            raise ValueError("The number of parameters is less than 4. Set the number of parameters to 4 or more before getting the k1 value.")
        return self.parameters[0]
    
    @k1.setter
    def k1(self, value: float) -> None:
        r"""
        Set the first radial distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 4.
            Set the number of parameters to 4 or more before setting the value.

        Parameters
        ----------
        value : float
            The first radial distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 4.
            If the value is not a number.
        """
        if not isinstance(value, Number):
            raise TypeError("The value should be a number.")
        if self.Nparams < 4:
            raise ValueError("The number of parameters is less than 4. Set the number of parameters to 4 or more before setting the k1 value.")
        self.parameters[0] = float(value)


    @property
    def k2(self) -> float:
        r"""
        Get the second radial distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 4.
            Set the number of parameters to 4 or more before getting the value.

        Returns
        -------
        float
            The second radial distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 4.
        """
        if self.Nparams < 4: 
            raise ValueError("The number of parameters is less than 4. Set the number of parameters to 4 or more before getting the k2 value.")
        return self.parameters[1]
    
    @k2.setter
    def k2(self, value: float) -> None:
        r"""
        Set the second radial distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 4.
            Set the number of parameters to 4 or more before setting the value.

        Parameters
        ----------
        value : float
            The second radial distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 4.
            If the value is not a number.
        """
        if not isinstance(value, Number):
            raise TypeError("The value should be a number.")
        if self.Nparams < 4:
            raise ValueError("The number of parameters is less than 4. Set the number of parameters to 4 or more before setting the k2 value.")
        self.parameters[1] = float(value)

    
    @property
    def p1(self) -> float:
        r"""
        Get the first tangential distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 4.
            Set the number of parameters to 4 or more before getting the value.

        Returns
        -------
        float
            The first tangential distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 4.
        """
        if self.Nparams < 4: 
            raise ValueError("The number of parameters is less than 4. Set the number of parameters to 4 or more before getting the p1 value.")
        return self.parameters[2]
    
    @p1.setter
    def p1(self, value: float) -> None:
        r"""
        Set the first tangential distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 4.
            Set the number of parameters to 4 or more before setting the value.

        Parameters
        ----------
        value : float
            The first tangential distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 4.
            If the value is not a number.
        """
        if not isinstance(value, Number):
            raise TypeError("The value should be a number.")
        if self.Nparams < 4:
            raise ValueError("The number of parameters is less than 4. Set the number of parameters to 4 or more before setting the p1 value.")
        self.parameters[2] = float(value)

    
    @property
    def p2(self) -> float:
        r"""
        Get the second tangential distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 4.
            Set the number of parameters to 4 or more before getting the value.

        Returns
        -------
        float
            The second tangential distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 4.
        """
        if self.Nparams < 4: 
            raise ValueError("The number of parameters is less than 4. Set the number of parameters to 4 or more before getting the p2 value.")
        return self.parameters[3]
    
    @p2.setter
    def p2(self, value: float) -> None:
        r"""
        Set the second tangential distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 4.
            Set the number of parameters to 4 or more before setting the value.

        Parameters
        ----------
        value : float
            The second tangential distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 4.
            If the value is not a number.
        """
        if not isinstance(value, Number):
            raise TypeError("The value should be a number.")
        if self.Nparams < 4:
            raise ValueError("The number of parameters is less than 4. Set the number of parameters to 4 or more before setting the p2 value.")
        self.parameters[3] = float(value)

    
    @property
    def k3(self) -> Optional[float]:
        r"""
        Get the third radial distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 5.
            Set the number of parameters to 5 or more before getting the value.

        Returns
        -------
        float
            The third radial distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 5.
        """
        if self.Nparams < 5: 
            raise ValueError("The number of parameters is less than 5. Set the number of parameters to 5 or more before getting the k3 value.")
        return self.parameters[4]
    
    @k3.setter
    def k3(self, value: float) -> None:
        r"""
        Set the third radial distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 5.
            Set the number of parameters to 5 or more before setting the value.

        Parameters
        ----------
        value : float
            The third radial distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 5.
            If the value is not a number.
        """
        if not isinstance(value, Number):
            raise TypeError("The value should be a number.")
        if self.Nparams < 5:
            raise ValueError("The number of parameters is less than 5. Set the number of parameters to 5 or more before setting the k3 value.")
        self.parameters[4] = float(value)

    
    @property
    def k4(self) -> Optional[float]:
        r"""
        Get the first fractional radial distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 8.
            Set the number of parameters to 8 or more before getting the value.

        Returns
        -------
        float
            The first fractional radial distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 8.
        """
        if self.Nparams < 8: 
            raise ValueError("The number of parameters is less than 8. Set the number of parameters to 8 or more before getting the k4 value.")
        return self.parameters[5]

    @k4.setter
    def k4(self, value: float) -> None:
        r"""
        Set the first fractional radial distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 8.
            Set the number of parameters to 8 or more before setting the value.

        Parameters
        ----------
        value : float
            The first fractional radial distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 8.
            If the value is not a number.
        """
        if not isinstance(value, Number):
            raise TypeError("The value should be a number.")
        if self.Nparams < 8:
            raise ValueError("The number of parameters is less than 8. Set the number of parameters to 8 or more before setting the k4 value.")
        self.parameters[5] = float(value)
    

    @property
    def k5(self) -> Optional[float]:
        r"""
        Get the second fractional radial distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 8.
            Set the number of parameters to 8 or more before getting the value.

        Returns
        -------
        float
            The second fractional radial distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 8.
        """
        if self.Nparams < 8: 
            raise ValueError("The number of parameters is less than 8. Set the number of parameters to 8 or more before getting the k5 value.")
        return self.parameters[6]
    
    @k5.setter
    def k5(self, value: float) -> None:
        r"""
        Set the second fractional radial distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 8.
            Set the number of parameters to 8 or more before setting the value.

        Parameters
        ----------
        value : float
            The second fractional radial distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 8.
            If the value is not a number.
        """
        if not isinstance(value, Number):
            raise TypeError("The value should be a number.")
        if self.Nparams < 8:
            raise ValueError("The number of parameters is less than 8. Set the number of parameters to 8 or more before setting the k5 value.")
        self.parameters[6] = float(value)
    

    @property
    def k6(self) -> Optional[float]:
        r"""
        Get the third fractional radial distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 8.
            Set the number of parameters to 8 or more before getting the value.

        Returns
        -------
        float
            The third fractional radial distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 8.
        """
        if self.Nparams < 8: 
            raise ValueError("The number of parameters is less than 8. Set the number of parameters to 8 or more before getting the k6 value.")
        return self.parameters[7]

    @k6.setter
    def k6(self, value: float) -> None:
        r"""
        Set the third fractional radial distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 8.
            Set the number of parameters to 8 or more before setting the value.

        Parameters
        ----------
        value : float
            The third fractional radial distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 8.
            If the value is not a number.
        """
        if not isinstance(value, Number):
            raise TypeError("The value should be a number.")
        if self.Nparams < 8:
            raise ValueError("The number of parameters is less than 8. Set the number of parameters to 8 or more before setting the k6 value.")
        self.parameters[7] = float(value)

    
    @property
    def s1(self) -> Optional[float]:
        r"""
        Get the first thin prism distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 12.
            Set the number of parameters to 12 or more before getting the value.

        Returns
        -------
        float
            The first thin prism distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 12.
        """
        if self.Nparams < 12: 
            raise ValueError("The number of parameters is less than 12. Set the number of parameters to 12 or more before getting the s1 value.")
        return self.parameters[8]
    
    @s1.setter
    def s1(self, value: float) -> None:
        r"""
        Set the first thin prism distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 12.
            Set the number of parameters to 12 or more before setting the value.

        Parameters
        ----------
        value : float
            The first thin prism distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 12.
            If the value is not a number.
        """
        if not isinstance(value, Number):
            raise TypeError("The value should be a number.")
        if self.Nparams < 12:
            raise ValueError("The number of parameters is less than 12. Set the number of parameters to 12 or more before setting the s1 value.")
        self.parameters[8] = float(value)

    
    @property
    def s2(self) -> Optional[float]:
        r"""
        Get the second thin prism distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 12.
            Set the number of parameters to 12 or more before getting the value.

        Returns
        -------
        float
            The second thin prism distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 12.
        """
        if self.Nparams < 12: 
            raise ValueError("The number of parameters is less than 12. Set the number of parameters to 12 or more before getting the s2 value.")
        return self.parameters[9]
    
    @s2.setter
    def s2(self, value: float) -> None:
        r"""
        Set the second thin prism distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 12.
            Set the number of parameters to 12 or more before setting the value.

        Parameters
        ----------
        value : float
            The second thin prism distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 12.
            If the value is not a number.
        """
        if not isinstance(value, Number):
            raise TypeError("The value should be a number.")
        if self.Nparams < 12:
            raise ValueError("The number of parameters is less than 12. Set the number of parameters to 12 or more before setting the s2 value.")
        self.parameters[9] = float(value)


    @property
    def s3(self) -> Optional[float]:
        r"""
        Get the third thin prism distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 12.
            Set the number of parameters to 12 or more before getting the value.

        Returns
        -------
        float
            The third thin prism distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 12.
        """
        if self.Nparams < 12: 
            raise ValueError("The number of parameters is less than 12. Set the number of parameters to 12 or more before getting the s3 value.")
        return self.parameters[10]
    
    @s3.setter
    def s3(self, value: float) -> None:
        r"""
        Set the third thin prism distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 12.
            Set the number of parameters to 12 or more before setting the value.

        Parameters
        ----------
        value : float
            The third thin prism distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 12.
            If the value is not a number.
        """
        if not isinstance(value, Number):
            raise TypeError("The value should be a number.")
        if self.Nparams < 12:
            raise ValueError("The number of parameters is less than 12. Set the number of parameters to 12 or more before setting the s3 value.")
        self.parameters[10] = float(value)

    
    @property
    def s4(self) -> Optional[float]:
        r"""
        Get the fourth thin prism distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 12.
            Set the number of parameters to 12 or more before getting the value.

        Returns
        -------
        float
            The fourth thin prism distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 12.
        """
        if self.Nparams < 12: 
            raise ValueError("The number of parameters is less than 12. Set the number of parameters to 12 or more before getting the s4 value.")
        return self.parameters[11]
    
    @s4.setter
    def s4(self, value: float) -> None:
        r"""
        Set the fourth thin prism distortion coefficient.

        .. warning::

            An error is raised if the number of parameters is less than 12.
            Set the number of parameters to 12 or more before setting the value.

        Parameters
        ----------
        value : float
            The fourth thin prism distortion coefficient.

        Raises
        -------
        ValueError
            If the number of parameters is less than 12.
            If the value is not a number.
        """
        if not isinstance(value, Number):
            raise TypeError("The value should be a number.")
        if self.Nparams < 12:
            raise ValueError("The number of parameters is less than 12. Set the number of parameters to 12 or more before setting the s4 value.")
        self.parameters[11] = float(value)
    

    @property
    def tau_x(self) -> Optional[float]:
        r"""
        Get the x component of the perspective transformation.

        .. warning::

            An error is raised if the number of parameters is less than 14.
            Set the number of parameters to 14 or more before getting the value.

        Returns
        -------
        float
            The x component of the perspective transformation.

        Raises
        -------
        ValueError
            If the number of parameters is less than 14.
        """
        if self.Nparams < 14: 
            raise ValueError("The number of parameters is less than 14. Set the number of parameters to 14 or more before getting the tau_x value.")
        return self.parameters[12]
    
    @tau_x.setter
    def tau_x(self, value: float) -> None:
        r"""
        Set the x component of the perspective transformation.

        .. warning::

            An error is raised if the number of parameters is less than 14.
            Set the number of parameters to 14 or more before setting the value.

        Parameters
        ----------
        value : float
            The x component of the perspective transformation.

        Raises
        -------
        ValueError
            If the number of parameters is less than 14.
            If the value is not a number.
        """
        if not isinstance(value, Number):
            raise TypeError("The value should be a number.")
        if self.Nparams < 14:
            raise ValueError("The number of parameters is less than 14. Set the number of parameters to 14 or more before setting the tau_x value.")
        self.parameters[12] = float(value)

    
    @property
    def tau_y(self) -> Optional[float]:
        r"""
        Get the y component of the perspective transformation.

        .. warning::

            An error is raised if the number of parameters is less than 14.
            Set the number of parameters to 14 or more before getting the value.

        Returns
        -------
        float
            The y component of the perspective transformation.

        Raises
        -------
        ValueError
            If the number of parameters is less than 14.
        """
        if self.Nparams < 14: 
            raise ValueError("The number of parameters is less than 14. Set the number of parameters to 14 or more before getting the tau_y value.")
        return self.parameters[13]
    
    @tau_y.setter
    def tau_y(self, value: float) -> None:
        r"""
        Set the y component of the perspective transformation.

        .. warning::

            An error is raised if the number of parameters is less than 14.
            Set the number of parameters to 14 or more before setting the value.

        Parameters
        ----------
        value : float
            The y component of the perspective transformation.

        Raises
        -------
        ValueError
            If the number of parameters is less than 14.
            If the value is not a number.
        """
        if not isinstance(value, Number):
            raise TypeError("The value should be a number.")
        if self.Nparams < 14:
            raise ValueError("The number of parameters is less than 14. Set the number of parameters to 14 or more before setting the tau_y value.")
        self.parameters[13] = float(value)
       

    def make_empty(self) -> None:
        r"""
        Set to zero the parameters of the distortion model.
        """
        self.parameters = numpy.zeros((self.Nparams, ), dtype=numpy.float64)


    # =================================================================
    # Internal methods to compute the distortion
    # =================================================================
    def _compute_tilt_matrix(self, dp: bool = True, inv: bool = True) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the tilt matrix for the perspective transformation for N = 14 (only).

        The tilt matrix is computed using the following equation:

        .. math::

            R_{\text{tilt}}{\tau} = R_Z[R_Y R_X] R_Y R_X

        where :math:`R_X` and :math:`R_Y` are the rotation matrices along X and Y respectively, and :math:`R_Z` is the rotation matrix along Z.

        .. math::
            R_X = \begin{pmatrix}
                1 & 0 & 0 \\
                0 & \cos(\tau_x) & \sin(\tau_x) \\
                0 & -\sin(\tau_x) & \cos(\tau_x)
            \end{pmatrix}

        .. math::
            R_Y = \begin{pmatrix}
                \cos(\tau_y) & 0 & -\sin(\tau_y) \\
                0 & 1 & 0 \\
                \sin(\tau_y) & 0 & \cos(\tau_y)
            \end{pmatrix}

        and we note that the rotation matrix along Z is given by:

        .. math::

            R_z[R] = \begin{pmatrix}
                R_{33} & 0 & -R_{13} \\
                0 & R_{33} & -R_{23} \\
                0 & 0 & 1
            \end{pmatrix}

        The derivatives of the tilt matrix with respect to :math:`\tau_x` and :math:`\tau_y` are also computed.
        The derivatives are computed using the following equations:

        .. math::

            \frac{\partial R_{\text{tilt}}}{\partial \tau_x} = R_Z [R_Y \frac{\partial R_X}{\partial \tau_x}, 0] R_Y R_X + R_Z [R_Y R_X, 1] R_Y \frac{\partial R_X}{\partial \tau_x}
        
        .. math::

            \frac{\partial R_{\text{tilt}}}{\partial \tau_y} = R_Z [\frac{\partial R_Y}{\partial \tau_y} R_X, 0] R_Y R_X + R_Z [R_Y R_X, 1] \frac{\partial R_Y}{\partial \tau_y} R_X    

        Finnally, the inverse of the tilt matrix is computed using the following equation:

        .. math::

            R_{\text{tilt}}^{-1} = (Ry Rx).T @ invRz[Ry Rx]

        Where :math:`invRz` is the inverse of the rotation matrix along Z given by:

        .. math::

            (R_z[R])^{-1} = \begin{pmatrix}
                1/R_{33} & 0 & R_{13}/R_{33} \\
                0 & 1/R_{33} & R_{23}/R_{33} \\
                0 & 0 & 1
            \end{pmatrix}

        .. note:: 

            If the model is not set to 14 parameters, the method returns a identity matrix and the derivatives are set to zero. 

        Parameters
        ----------
        dp : bool, optional
            If True, the derivatives of the tilt matrix are computed. The default is True.
            If False, the derivatives are set to None.

        inv : bool, optional
            If True, the inverse of the tilt matrix is computed. The default is True.
            If False, the inverse of the tilt matrix is set to None.

        Returns
        -------
        numpy.ndarray
            The tilt matrix.

        numpy.ndarray
            The derivative of the tilt matrix with respect to :math:`\tau_x` if ``dp`` is True, else None.

        numpy.ndarray
            The derivative of the tilt matrix with respect to :math:`\tau_y` if ``dp`` is True, else None.
        
        numpy.ndarray
            The inverse of the tilt matrix if ``inv`` is True, else None.
        """
        if not isinstance(dp, bool):
            raise TypeError("The dp parameter should be a boolean.")
        
        # Initialize the rotation matrices
        R = None
        Rdtx = None
        Rdty = None
        invR = None
    
        # If the number of parameters is not 14, return identity matrix and zero derivatives
        if self.Nparams != 14:
            R = numpy.eye(3, dtype=numpy.float64)
            if dp:
                Rdtx = numpy.zeros((3, 3), dtype=numpy.float64)
                Rdty = numpy.zeros((3, 3), dtype=numpy.float64)
            if inv:
                invR = numpy.eye(3, dtype=numpy.float64)
            return R, Rdtx, Rdty, invR

        # Prepare the cosinus and sinus of the angles
        ctx = numpy.cos(self.tau_x)
        cty = numpy.cos(self.tau_y)
        stx = numpy.sin(self.tau_x)
        sty = numpy.sin(self.tau_y)

        # Prepare the rotation matrix along X and Y
        Rx = numpy.array([
            [1, 0, 0],
            [0, ctx, stx],
            [0, -stx, ctx]
        ], dtype=numpy.float64)

        Ry = numpy.array([
            [cty, 0, -sty],
            [0, 1, 0],
            [sty, 0, cty]
        ], dtype=numpy.float64)

        if dp:
            Rxdtx = numpy.array([
                [0, 0, 0],
                [0, -stx, ctx],
                [0, -ctx, -stx]
            ], dtype=numpy.float64)

            Rydty = numpy.array([
                [-sty, 0, -cty],
                [0, 0, 0],
                [cty, 0, -sty]
            ], dtype=numpy.float64)

        # Compute the products of the rotation matrices
        Rxy = numpy.dot(Ry, Rx)

        if dp:
            Rxydtx = numpy.dot(Ry, Rxdtx)
            Rxydty = numpy.dot(Rydty, Rx)

        if inv:
            invRxy = Rxy.T
            
        # Compute the rotation along Z
        Rz = numpy.array([
            [Rxy[2, 2], 0, -Rxy[0, 2]],
            [0, Rxy[2, 2], -Rxy[1, 2]],
            [0, 0, 1]
        ], dtype=numpy.float64)

        if dp:
            Rzdtx = numpy.array([
                [Rxydtx[2, 2], 0, -Rxydtx[0, 2]],
                [0, Rxydtx[2, 2], -Rxydtx[1, 2]],
                [0, 0, 0]
            ], dtype=numpy.float64)

            Rzdty = numpy.array([
                [Rxydty[2, 2], 0, -Rxydty[0, 2]],
                [0, Rxydty[2, 2], -Rxydty[1, 2]],
                [0, 0, 0]
            ], dtype=numpy.float64)
        
        if inv:
            invRz = numpy.array([
                [1/Rxy[2, 2], 0, Rxy[0, 2]/Rxy[2, 2]],
                [0, 1/Rxy[2, 2], Rxy[1, 2]/Rxy[2, 2]],
                [0, 0, 1]
            ], dtype=numpy.float64)

        # Compute the tilt matrix and the derivatives
        R = numpy.dot(Rz, Rxy)

        if dp:
            Rdtx = numpy.dot(Rz, Rxydtx) + numpy.dot(Rzdtx, Rxy)
            Rdty = numpy.dot(Rz, Rxydty) + numpy.dot(Rzdty, Rxy)
        
        if inv:
            invR = numpy.dot(Rxy.T, invRz)

        # Return the tilt matrix and the derivatives
        return R, Rdtx, Rdty, invR
    

    # =================================================================
    # Distortion methods with OpenCV
    # =================================================================
    def distort_opencv(self, normalized_points: numpy.ndarray, transpose: bool = False, **kwargs) -> DistortionResult:
        r"""
        This method achieves the distortion of the given ``normalized_points`` using the OpenCV function ``projectPoints``.

        .. seealso::

            - :meth:pydistort.Distortion.distort` to achieve the distortion using the internal method.

        .. warnign::

            The output of this method is similar than the output of the :meth:pydistort.Distortion.distort` method, but the jacobian with respect to the normalized points is not computed and 
            the jacobian with respect to the distortion parameters is always computed.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            Array of normalized points to be transformed with shape (..., 2).

        transpose : bool, optional
            If True, the input points are assume to have shape (2, ...).
            In this case, the output points will have shape (2, ...) as well and the jacobian matrices will have shape (2, ..., 2) and (2, ..., Nparams) respectively.
            Default is False.

        kwargs : dict, optional
            Additional keyword arguments to be passed to the distortion model.

        Returns
        -------
        distortion_result : DistortionResult

            The result of the distortion transformation containing the image points and the jacobian matrices.
            This object has the following attributes:

            image_points : numpy.ndarray
                The transformed image points in pixels. It will be a 2D array of shape (..., 2) if ``transpose`` is False.

            jacobian_dx : None
                Always None. Use the :meth:pydistort.Distortion.distort` method to compute the jacobian with respect to the normalized points.

            jacobian_dp : Optional[numpy.ndarray]
                The Jacobian of the image points with respect to the distortion parameters.
                It will be a 2D array of shape (..., 2, Nparams) if ``transpose`` is False.                     
        """
        # Check the boolean parameters
        if not isinstance(transpose, bool):
            raise ValueError("The transpose parameter must be a boolean.")
        
        # Check if the distortion parameters are set
        if not self.is_set():
            raise ValueError("The distortion parameters is not set. Please set the distortion parameters before using this method.")
        
        # Create the array of points
        points = numpy.asarray(normalized_points, dtype=numpy.float64)

        # Transpose the points if needed
        if transpose:
            points = numpy.moveaxis(points, 0, -1) # (2, ...) -> (..., 2)

        # Extract the original shape
        shape = points.shape # (..., 2)

        # Flatten the points along the last axis
        points_flat = points.reshape(-1, shape[-1]) # shape (..., 2) -> shape (Npoints, 2)
        Npoints = points_flat.shape[0] # Npoints 

        # Check the shape of the points
        if points_flat.ndim !=2 or points_flat.shape[1] != 2:
            raise ValueError(f"The points must be in the shape (Npoints, 2) or (2, Npoints) if ``transpose`` is True. Got {points_flat.shape} instead and transpose is {transpose}.")
        
        # Create the contiguous array of shape (Npoints, 1, 3) for cv2 compatibility
        object_points = numpy.concatenate((points_flat, numpy.ones((points_flat.shape[0], 1))), axis=1)
        object_points = numpy.ascontiguousarray(object_points.reshape(-1, 1, 3), dtype=numpy.float64)

        # Apply the OpenCV distortion removing rvec, tvec and intrinsic matrix
        rvec = numpy.zeros((3, 1), dtype=numpy.float64)
        tvec = numpy.zeros((3, 1), dtype=numpy.float64)
        intrinsic_matrix = numpy.eye(3, dtype=numpy.float64)
        image_points, jacobian = cv2.projectPoints(object_points, rvec, tvec, intrinsic_matrix, self.parameters) # shape (Npoints, 1, 2)

        # Reshape the image points to (2, Npoints)
        distorted_points_flat = numpy.asarray(image_points[:,0,:], dtype=numpy.float64)
        jacobian = numpy.asarray(jacobian, dtype=numpy.float64)[:, -self.Nparams:] # shape (2 * Npoints, Nparams)
        jacobian_flat_dp = numpy.zeros((Npoints, 2, self.Nparams), dtype=numpy.float64)
        jacobian_flat_dp[:, 0, :] = jacobian[0::2, :]
        jacobian_flat_dp[:, 1, :] = jacobian[1::2, :]

        # Reshape the image points back to the original shape
        distorted_points = distorted_points_flat.reshape(shape) # (Npoints, 2) -> (..., 2)
        jacobian_dp = jacobian_flat_dp.reshape((*shape, self.Nparams)) # (Npoints, 2, Nparams) -> (..., 2, Nparams)

        # Transpose the points back to the original shape if needed
        if transpose:
            distorted_points = numpy.moveaxis(distorted_points, -1, 0) # (..., 2) -> (2, ...)
            jacobian_dp = numpy.moveaxis(jacobian_dp, -2, 0) # (..., 2, Nparams) -> (2, ..., Nparams)

        # Return the image points and the jacobian matrices
        result = DistortionResult(distorted_points, None, jacobian_dp)
        return result
    
    def undistort_opencv(self, distorted_points: numpy.ndarray, transpose: bool = False, **kwargs) -> UndistortResult:
        r"""
        This method achieves the undistortion of the given ``distorted_points`` using the OpenCV function ``undistortPoints``.

        .. seealso::

            - :meth:pydistort.Distortion.undistort` to achieve the undistortion using the internal method.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted points to be undistorted with shape (..., 2).

        transpose : bool, optional
            If True, the input points are assume to have shape (2, ...).
            In this case, the output points will have shape (2, ...) as well.
            Default is False.

        kwargs : dict, optional
            Additional keyword arguments to be passed to the distortion model.

        Returns
        -------
        undistort_result : UndistortResult

            The result of the undistortion transformation containing the normalized points.
            This object has the following attributes:

            normalized_points : numpy.ndarray
                The transformed normalized points in pixels. It will be a 2D array of shape (..., 2) if ``transpose`` is False.
        """
        # Check the boolean parameters
        if not isinstance(transpose, bool):
            raise ValueError("The transpose parameter must be a boolean.")
        
        # Check if the distortion parameters are set
        if not self.is_set():
            raise ValueError("The distortion parameters is not set. Please set the distortion parameters before using this method.")
        
        # Create the array of points
        points = numpy.asarray(distorted_points, dtype=numpy.float64)

        # Transpose the points if needed
        if transpose:
            points = numpy.moveaxis(points, 0, -1) # (2, ...) -> (..., 2)

        # Extract the original shape
        shape = points.shape

        # Flatten the points along the last axis
        points_flat = points.reshape(-1, shape[-1]) # shape (..., 2) -> shape (Npoints, 2)
        Npoints = points_flat.shape[0]

        # Check the shape of the points
        if points_flat.ndim != 2 or points_flat.shape[1] != 2:
            raise ValueError(f"The points must be in the shape (Npoints, 2) or (2, Npoints) if ``transpose`` is True. Got {points_flat.shape} instead and transpose is {transpose}.")
        
        # Create the contiguous array of shape (Npoints, 1, 2) for cv2 compatibility
        distorted_points = numpy.ascontiguousarray(points_flat.reshape(-1, 1, 2), dtype=numpy.float64)

        # Apply the OpenCV undistortion removing rvec, tvec and intrinsic matrix
        Rmat = numpy.eye(3, dtype=numpy.float64)
        Pmat = numpy.eye(3, dtype=numpy.float64)
        intrinsic_matrix = numpy.eye(3, dtype=numpy.float64)
        normalized_points = cv2.undistortPoints(distorted_points, intrinsic_matrix, self.parameters, Rmat, Pmat) # shape (Npoints, 1, 2)

        # Reshape the normalized points to (Npoints, 2)
        normalized_points_flat = numpy.asarray(normalized_points[:,0,:], dtype=numpy.float64)

        # Reshape the normalized points back to the original shape
        normalized_points = normalized_points_flat.reshape(shape) # (Npoints, 2) -> (..., 2)

        # Transpose the points back to the original shape if needed
        if transpose:
            normalized_points = numpy.moveaxis(normalized_points, -1, 0)

        # Return the normalized points
        result = UndistortResult(normalized_points)
        return result
    
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
        if self.Nparams == 0:
            return "Cv2Distortion: No distortion model"
        elif self.Nparams == 4:
            return f"Cv2Distortion: k1 = {self.k1}, k2 = {self.k2}, p1 = {self.p1}, p2 = {self.p2}"
        elif self.Nparams == 8:
            return f"Cv2Distortion: k1 = {self.k1}, k2 = {self.k2}, p1 = {self.p1}, p2 = {self.p2}, k3 = {self.k3}, k4 = {self.k4}, k5 = {self.k5}, k6 = {self.k6}"
        elif self.Nparams == 12:
            return f"Cv2Distortion: k1 = {self.k1}, k2 = {self.k2}, p1 = {self.p1}, p2 = {self.p2}, k3 = {self.k3}, k4 = {self.k4}, k5 = {self.k5}, k6 = {self.k6}, s1 = {self.s1}, s2 = {self.s2}, s3 = {self.s3}, s4 = {self.s4}"
        elif self.Nparams == 14:
            return f"Cv2Distortion: k1 = {self.k1}, k2 = {self.k2}, p1 = {self.p1}, p2 = {self.p2}, k3 = {self.k3}, k4 = {self.k4}, k5 = {self.k5}, k6 = {self.k6}, s1 = {self.s1}, s2 = {self.s2}, s3 = {self.s3}, s4 = {self.s4}, tau_x = {self.tau_x}, tau_y = {self.tau_y}"
        else:
            raise ValueError(f"Unknown distortion model with {self.Nparams} parameters.") 

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
    
    
    def _distort(self, normalized_points: numpy.ndarray, dx: bool = False, dp: bool = False, **kwargs) -> tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        From the abstract method to apply distortion to a set of points.

        .. note::

            For ``_distort`` the output is always in the shape (Npoints, 2).
            The output must be (Npoints, 2) for the distorted points and (Npoints, 2, 2) for the jacobian with respect to the normalized points and (Npoints, 2, Nparams) for the jacobian with respect to the distortion parameters.

        .. seealso::

            - :meth:pydistort.Cv2Distortion.distort_opencv` to achieve the distortion using OpenCV.

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
        x_N = x_N[:, numpy.newaxis] # shape (Npoints, 1)
        y_N = y_N[:, numpy.newaxis] # shape (Npoints, 1)
        Npoints = normalized_points.shape[0]
        Nparams = self.Nparams

        zero = lambda dim: numpy.zeros((Npoints, dim), dtype=numpy.float64) # shape (Npoints, dim)
        ccat = lambda tup: numpy.concatenate(tup, axis=1) # Concatenate along the second axis
        dDdxy = None # The derivative of the distortion with respect to the normalized points
        dDdp = None # The derivative of the distortion with respect to the distortion parameters

        # Prepare some variables for the distortion
        xN_yN = x_N * y_N # shape (Npoints, 1)
        xN2 = x_N ** 2 # shape (Npoints, 1)
        yN2 = y_N ** 2 # shape (Npoints, 1)
        
        # Return a identity Jacobian and an empty jacobian if no parameters in the model
        if Nparams == 0:
            distorted_points = numpy.copy(normalized_points) # shape (Npoints, 2)
            if dx: 
                dDdxy = numpy.zeros((Npoints, 2, 2), dtype=numpy.float64) # shape (Npoints, 2, 2)
                dDdxy[:, 0, 0] = 1.0
                dDdxy[:, 1, 1] = 1.0
            if dp:
                dDdp = numpy.empty((Npoints, 2, 0), dtype=numpy.float64) # shape (Npoints, 2, 0)
            return distorted_points, dDdxy, dDdp
                
        # Prepare the powers of the norm (r) [only if needed] with shape (Npoints, 1)
        r2 = xN2 + yN2 # shape (Npoints, 1)
        r4 = r2 ** 2 # shape (Npoints, 1)

        if Nparams >= 5:
            r6 = r2 * r4 # shape (Npoints, 1)

        # Prepare the radial distortion coefficients [only if needed] with shape (Npoints, 1)
        if Nparams == 4:
            K = (1 + self.k1 * r2 + self.k2 * r4) # shape (Npoints, 1)

        elif Nparams == 5:
            K = (1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6) # shape (Npoints, 1)
        
        else: # Nparams >= 8
            Kup = (1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6) # shape (Npoints, 1)
            Kdown = (1 + self.k4 * r2 + self.k5 * r4 + self.k6 * r6) # shape (Npoints, 1)
            iKdown = 1 / Kdown # shape (Npoints, 1)
            i2Kdown = iKdown ** 2 # shape (Npoints, 1)
            K = Kup * iKdown # shape (Npoints, 1)
                            
        x_radial = x_N * K # shape (Npoints, 1)
        y_radial = y_N * K # shape (Npoints, 1)

        # Prepare the tangential distortion coefficients [only if needed] with shape (Npoints, 1)
        axp1 = ayp2 = 2 * xN_yN # shape (Npoints, 1)
        axp2 = r2 + 2 * xN2 # shape (Npoints, 1)
        ayp1 = r2 + 2 * yN2 # shape (Npoints, 1)
        x_tangential = self.p1 * axp1 + self.p2 * axp2 # shape (Npoints, 1)
        y_tangential = self.p1 * ayp1 + self.p2 * ayp2 # shape (Npoints, 1)

        # Prepare the prism distortion coefficients [only if needed] with shape (Npoints, 1)
        if Nparams < 12:
            x_prism = zero(1) # shape (Npoints, 1)
            y_prism = zero(1) # shape (Npoints, 1)
            
        else: # Nparams >= 12
            x_prism = self.s1 * r2 + self.s2 * r4 # shape (Npoints, 1)
            y_prism = self.s3 * r2 + self.s4 * r4 # shape (Npoints, 1)
            
        # Compute the distorted points
        x_D = x_radial + x_tangential + x_prism # shape (Npoints, 1)
        y_D = y_radial + y_tangential + y_prism # shape (Npoints, 1)

        # Prepare some variables for the jacobians
        if (dx and Nparams >= 12) or dp:
            xN_r2 = x_N * r2 # shape (Npoints, 1)
            yN_r2 = y_N * r2 # shape (Npoints, 1)

        # Compute the Jacobians with respect to the normalized points with shape (Npoints, 2)
        if dx:
            x_Ddxy = numpy.empty((Npoints, 2), dtype=numpy.float64) # shape (Npoints, 2)
            y_Ddxy = numpy.empty((Npoints, 2), dtype=numpy.float64) # shape (Npoints, 2)

            if Nparams == 4:
                dK_r2 = (2 * self.k1 + 4 * self.k2 * r2) # shape (Npoints, 1)
            elif Nparams == 5:
                dK_r2 = (2 * self.k1 + 4 * self.k2 * r2 + 6 * self.k3 * r4) # shape (Npoints, 1)
            else:
                dK_r2 = (2 * self.k1 + 4 * self.k2 * r2 + 6 * self.k3 * r4) * Kdown - Kup * (2 * self.k4 + 4 * self.k5 * r2 + 6 * self.k6 * r4) # shape (Npoints, 1)
                dK_r2 = dK_r2 * i2Kdown

            x_radial_dx = K + dK_r2 * xN2 # shape (Npoints, 1)
            x_radial_dy = y_radial_dx = dK_r2 * xN_yN # shape (Npoints, 1)
            y_radial_dy = K + dK_r2 * yN2 # shape (Npoints, 1)

            x_tangential_dx = (2 * self.p1) * y_N + (6 * self.p2) * x_N
            x_tangential_dy = (2 * self.p1) * x_N + (2 * self.p2) * y_N
            y_tangential_dx = (2 * self.p2) * y_N + (2 * self.p1) * x_N
            y_tangential_dy = (2 * self.p2) * x_N + (6 * self.p1) * y_N

            x_Ddxy[:, 0] = (x_radial_dx + x_tangential_dx).ravel()
            x_Ddxy[:, 1] = (x_radial_dy + x_tangential_dy).ravel()
            y_Ddxy[:, 0] = (y_radial_dx + y_tangential_dx).ravel()
            y_Ddxy[:, 1] = (y_radial_dy + y_tangential_dy).ravel()

            if Nparams >= 12:
                x_Ddxy[:, 0] += ((2 * self.s1) * x_N + (4 * self.s2) * xN_r2).ravel()
                x_Ddxy[:, 1] += ((2 * self.s1) * y_N + (4 * self.s2) * yN_r2).ravel()
                y_Ddxy[:, 0] += ((2 * self.s3) * x_N + (4 * self.s4) * xN_r2).ravel()
                y_Ddxy[:, 1] += ((2 * self.s3) * y_N + (4 * self.s4) * yN_r2).ravel()

        if dp:
            x_Ddp = numpy.empty((Npoints, Nparams), dtype=numpy.float64) # shape (Npoints, Nparams)
            y_Ddp = numpy.empty((Npoints, Nparams), dtype=numpy.float64) # shape (Npoints, Nparams)

            x_Ddp[:, 0] = xN_r2.ravel() if Nparams <= 5 else (xN_r2 * iKdown).ravel()
            x_Ddp[:, 1] = (r4 * x_N).ravel() if Nparams <= 5 else (r4 * x_N * iKdown).ravel()
            x_Ddp[:, 2] = axp1.ravel()
            x_Ddp[:, 3] = axp2.ravel()
            y_Ddp[: ,0] = yN_r2.ravel() if Nparams <= 5 else (yN_r2 * iKdown).ravel()
            y_Ddp[:, 1] = (r4 * y_N).ravel() if Nparams <= 5 else (r4 * y_N * iKdown).ravel()
            y_Ddp[:, 2] = ayp1.ravel()
            y_Ddp[:, 3] = ayp2.ravel()

            if Nparams >= 5:
                x_Ddp[:, 4] = (r6 * x_N).ravel() if Nparams <= 5 else (r6 * x_N * iKdown).ravel()
                y_Ddp[:, 4] = (r6 * y_N).ravel() if Nparams <= 5 else (r6 * y_N * iKdown).ravel()

            if Nparams >= 8:
                m_Kup_i2Kdown = -Kup * i2Kdown
                m_Kup_i2Kdown_xN = m_Kup_i2Kdown * x_N
                m_Kup_i2Kdown_yN = m_Kup_i2Kdown * y_N
                x_Ddp[:, 5] = (m_Kup_i2Kdown_xN * r2).ravel()
                x_Ddp[:, 6] = (m_Kup_i2Kdown_xN * r4).ravel() 
                x_Ddp[:, 7] = (m_Kup_i2Kdown_xN * r6).ravel()
                y_Ddp[:, 5] = (m_Kup_i2Kdown_yN * r2).ravel()
                y_Ddp[:, 6] = (m_Kup_i2Kdown_yN * r4).ravel() 
                y_Ddp[:, 7] = (m_Kup_i2Kdown_yN * r6).ravel()

            if Nparams >= 12:
                x_Ddp[:, 8] = r2.ravel()
                x_Ddp[:, 9] = r4.ravel()
                x_Ddp[:, 10:12] = 0.0
                y_Ddp[:, 8:10] = 0.0
                y_Ddp[:, 10] = r2.ravel()
                y_Ddp[:, 11] = r4.ravel()

            if Nparams >= 14:
                x_Ddp[:, 12:14] = 0.0
                y_Ddp[:, 12:14] = 0.0


        # Apply the perspective transformation [only if needed]
        # Also compute the finals derivatives with respect to the normalized points and to the parameters
        if self.Nparams >= 14:
            # Get the tilt matrix
            R, Rdtx, Rdty, _ = self._compute_tilt_matrix(dp = dp, inv=False) # shape (3, 3) ; shape (3, 3) ; shape (3, 3)

            # Apply the perspective transformation
            x_perspectD = R[0, 0] * x_D + R[0, 1] * y_D + R[0, 2] # shape (Npoints, 1)
            y_perspectD = R[1, 0] * x_D + R[1, 1] * y_D + R[1, 2] # shape (Npoints, 1)
            z_perspectD = R[2, 0] * x_D + R[2, 1] * y_D + R[2, 2] # shape (Npoints, 1)
            iz_perspectD = 1 / z_perspectD # shape (Npoints, 1)
            i2z_perspectD = iz_perspectD ** 2 # shape (Npoints, 1)
            if dx:
                x_perspectDdxy = (R[0, 0] * x_Ddxy + R[0, 1] * y_Ddxy) # shape (Npoints, 2)
                y_perspectDdxy = (R[1, 0] * x_Ddxy + R[1, 1] * y_Ddxy) # shape (Npoints, 2)
                z_perspectDdxy = (R[2, 0] * x_Ddxy + R[2, 1] * y_Ddxy) # shape (Npoints, 2)
            if dp:
                x_perspectDdp = numpy.empty((Npoints, Nparams), dtype=numpy.float64) # shape (Npoints, Nparams)
                x_perspectDdp[:, :12] = (R[0, 0] * x_Ddp[:, :12] + R[0, 1] * y_Ddp[:, :12]) # shape (Npoints, 12)
                x_perspectDdp[:, 12] = (Rdtx[0, 0] * x_D + Rdtx[0, 1] * y_D + Rdtx[0, 2]).ravel() # shape (Npoints, 1)
                x_perspectDdp[:, 13] = (Rdty[0, 0] * x_D + Rdty[0, 1] * y_D + Rdty[0, 2]).ravel() # shape (Npoints, 1)

                y_perspectDdp = numpy.empty((Npoints, Nparams), dtype=numpy.float64) # shape (Npoints, Nparams)
                y_perspectDdp[:, :12] = (R[1, 0] * x_Ddp[:, :12] + R[1, 1] * y_Ddp[:, :12]) # shape (Npoints, 12)
                y_perspectDdp[:, 12] = (Rdtx[1, 0] * x_D + Rdtx[1, 1] * y_D + Rdtx[1, 2]).ravel() # shape (Npoints, 1)
                y_perspectDdp[:, 13] = (Rdty[1, 0] * x_D + Rdty[1, 1] * y_D + Rdty[1, 2]).ravel() # shape (Npoints, 1)

                z_perspectDdp = numpy.empty((Npoints, Nparams), dtype=numpy.float64) # shape (Npoints, Nparams)
                z_perspectDdp[:, :12] = (R[2, 0] * x_Ddp[:, :12] + R[2, 1] * y_Ddp[:, :12]) # shape (Npoints, 12)
                z_perspectDdp[:, 12] = (Rdtx[2, 0] * x_D + Rdtx[2, 1] * y_D + Rdtx[2, 2]).ravel() # shape (Npoints, 1)
                z_perspectDdp[:, 13] = (Rdty[2, 0] * x_D + Rdty[2, 1] * y_D + Rdty[2, 2]).ravel() # shape (Npoints, 1)
            
            # Normalize the points by the perspective transformation
            x_D = x_perspectD * iz_perspectD # shape (Npoints, 1)
            y_D = y_perspectD * iz_perspectD # shape (Npoints, 1)
            if dx:
                x_Ddxy = (x_perspectDdxy * numpy.broadcast_to(z_perspectD, (Npoints, 2)) - numpy.broadcast_to(x_perspectD, (Npoints, 2)) * z_perspectDdxy) * i2z_perspectD # shape (Npoints, 2)
                y_Ddxy = (y_perspectDdxy * numpy.broadcast_to(z_perspectD, (Npoints, 2)) - numpy.broadcast_to(y_perspectD, (Npoints, 2)) * z_perspectDdxy) * i2z_perspectD # shape (Npoints, 2)
            if dp:
                x_Ddp = (x_perspectDdp * numpy.broadcast_to(z_perspectD, (Npoints, Nparams)) - numpy.broadcast_to(x_perspectD, (Npoints, Nparams)) * z_perspectDdp) * i2z_perspectD # shape (Npoints, Nparams)
                y_Ddp = (y_perspectDdp * numpy.broadcast_to(z_perspectD, (Npoints, Nparams)) - numpy.broadcast_to(y_perspectD, (Npoints, Nparams)) * z_perspectDdp) * i2z_perspectD # shape (Npoints, Nparams)
            
        # Construct the final outputs
        distorted_points = ccat((x_D, y_D)) # shape (Npoints, 2)
        if dx:
            dDdxy = numpy.zeros((Npoints, 2, 2), dtype=numpy.float64) # shape (Npoints, 2, 2)
            dDdxy[:, 0, :] = x_Ddxy # shape (Npoints, 2)
            dDdxy[:, 1, :] = y_Ddxy # shape (Npoints, 2)
        if dp:
            dDdp = numpy.zeros((Npoints, 2, Nparams), dtype=numpy.float64) # shape (Npoints, 2, Nparams)
            dDdp[:, 0, :] = x_Ddp # shape (Npoints, Nparams)
            dDdp[:, 1, :] = y_Ddp # shape (Npoints, Nparams)
        
        # Return the distorted points and the derivatives
        return distorted_points, dDdxy, dDdp
    

    def _undistort(self, distorted_points: numpy.ndarray, max_iter: int = 10, eps: float = 1e-8, **kwargs) -> numpy.ndarray:
        r"""
        From the abstract method to remove distortion from a set of points.

        To achieve the undistortion, an iterative algorithm is used to find the normalized points that correspond to the distorted points.
        The algorithm is based on the following equations:

        .. math::
            \begin{bmatrix}
            x_N [\text{it }k+1]\\
            y_N [\text{it }k+1]
            \end{bmatrix}
            = 
            \begin{bmatrix}
            (x_D - \Delta x [\text{it }k]) / \text{Rad} [\text{it }k] \\
            (y_D - \Delta y [\text{it }k]) / \text{Rad}[\text{it }k]
            \end{bmatrix}

        Where :math:`\Delta x [\text{it }k]` and :math:\Delta y [\text{it }k] are the tangential and prism distortion contributions to the distorted points computed at iteration :math:k.
        And :math:`\text{Rad} [\text{it }k]` is the radial distortion contribution to the distorted points computed at iteration :math:k.

        .. math::
            \begin{bmatrix}
            \Delta x
            \Delta y
            \end{bmatrix}
            =
            \begin{bmatrix}
            2 p_1 x_N y_N + p_2 (r^2 + 2x_N^2) + s_1 r^2 + s_2 r^4 \\
            2 p_2 x_N y_N + p_1 (r^2 + 2y_N^2) + s_3 r^2 + s_4 r^4
            \end{bmatrix}

        .. math::
            \begin{bmatrix}
            \text{Rad}_x \\
            \text{Rad}_y
            \end{bmatrix}
            =
            \begin{bmatrix}
            \frac{1+k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} \\
            \frac{1+k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}
            \end{bmatrix}

        .. note::

            For ``_undistort`` the input is always in the shape (Npoints, 2) with float64 type.

        .. seealso::

            - :meth:pydistort.Cv2Distortion.undistort_opencv` to achieve the undistortion using OpenCV.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            Array of distorted points to be transformed with shape (Npoints, 2).

        max_iter : int, optional
            The maximum number of iterations for the optimization algorithm. Default is 100.

        eps : float, optional
            The tolerance for the optimization algorithm. Default is 1e-6.

        Returns
        -------
        normalized_points : numpy.ndarray
            The transformed normalized points in normalized coordinates. It will be a 2D array of shape (Npoints, 2).
        """
        # Prepare the inputs data for undistortion
        x_D = distorted_points[:, 0] # shape (Npoints,)
        y_D = distorted_points[:, 1] # shape (Npoints,)
        x_D = x_D[:, numpy.newaxis] # shape (Npoints, 1)
        y_D = y_D[:, numpy.newaxis] # shape (Npoints, 1)
        Npoints = distorted_points.shape[0]
        Nparams = self.Nparams

        # Case of no parameters in the model
        if Nparams == 0:
            normalized_points = numpy.copy(distorted_points)
            return normalized_points

        # Get the tilt matrix [only if needed]
        if self.Nparams >= 14:
            R, _, _, invR = self._compute_tilt_matrix(dp=False, inv=True)

        # Prepare the output array:
        normalized_points = numpy.empty((Npoints, 2), dtype=numpy.float64) # shape (Npoints, 2)

        # Create the mask for the points in computation
        mask = numpy.ones((Npoints,), dtype=numpy.bool) # shape (Npoints,)

        # Remove the perspective transformation [only if needed]
        if self.Nparams >= 14:
            x_0 = numpy.dot(invR[0, 0], x_D) + numpy.dot(invR[0, 1], y_D) + invR[0, 2] # shape (Npoints, 1)
            y_0 = numpy.dot(invR[1, 0], x_D) + numpy.dot(invR[1, 1], y_D) + invR[1, 2] # shape (Npoints, 1)
            z_0 = numpy.dot(invR[2, 0], x_D) + numpy.dot(invR[2, 1], y_D) + invR[2, 2] # shape (Npoints, 1)
            x_0 = x_0 / z_0 # shape (Npoints, 1)
            y_0 = y_0 / z_0 # shape (Npoints, 1)
        else:
            x_0 = x_D # shape (Npoints, 1)
            y_0 = y_D # shape (Npoints, 1)

        # Initialize the guess for the normalized points
        x_N = x_0.copy() # shape (Npoints, 1)
        y_N = y_0.copy() # shape (Npoints, 1)
        Nopt = Npoints # Number of points in computation

        # Run the iterative algorithm
        for it in range(max_iter):

            # Prepare the powers of the norm (r) [only if needed] with shape (Nopt, 1)
            r2 = x_N ** 2 + y_N ** 2 # shape (Nopt, 1)
            r4 = r2 ** 2 # shape (Nopt, 1)
            if Nparams >= 5:
                r6 = r2 * r4 # shape (Nopt, 1)

            # Prepare the radial distortion coefficients [only if needed] with shape (Nopt, 1)
            if Nparams == 4:
                invK = 1/(1 + self.k1 * r2 + self.k2 * r4) # shape (Nopt, 1)

            elif Nparams == 5:
                invK = 1/(1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6) # shape (Nopt, 1)
            
            else: # Nparams >= 8
                Kup = (1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6) # shape (Nopt, 1)
                Kdown = (1 + self.k4 * r2 + self.k5 * r4 + self.k6 * r6) # shape (Nopt, 1)
                invK = Kdown / Kup # shape (Nopt, 1)

            # Prepare the tangential distortion coefficients [only if needed] with shape (Nopt, 1)
            axp1 = ayp2 = 2 * x_N * y_N # shape (Nopt, 1)
            axp2 = r2 + 2 * x_N ** 2 # shape (Nopt, 1)
            ayp1 = r2 + 2 * y_N ** 2 # shape (Nopt, 1)
            x_tangential = self.p1 * axp1 + self.p2 * axp2 # shape (Nopt, 1)
            y_tangential = self.p1 * ayp1 + self.p2 * ayp2 # shape (Nopt, 1)

            # Prepare the prism distortion coefficients [only if needed] with shape (Nopt, 1)
            x_prism = numpy.zeros((Nopt, 1), dtype=numpy.float64) # shape (Nopt, 1)
            y_prism = numpy.zeros((Nopt, 1), dtype=numpy.float64) # shape (Nopt, 1)
            if Nparams >= 12:
                x_prism = self.s1 * r2 + self.s2 * r4 # shape (Nopt, 1)
                y_prism = self.s3 * r2 + self.s4 * r4 # shape (Nopt, 1)

            # Update the normalized points
            x_N = (x_0[mask, :] - x_tangential - x_prism) * invK # shape (Nopt, 1)
            y_N = (y_0[mask, :] - y_tangential - y_prism) * invK # shape (Nopt, 1)

            # Update the normalized points
            normalized_points[mask, 0] = x_N.ravel() # shape (Nopt,)
            normalized_points[mask, 1] = y_N.ravel() # shape (Nopt,)

            # Distortion convergence check
            distorted_points_optimized, _, _ = self._distort(numpy.concatenate((x_N, y_N), axis=1), dx=False, dp=False) # shape (Nopt, 2)

            # Compute the norm of the difference
            diff = numpy.linalg.norm(distorted_points_optimized - distorted_points[mask, :], axis=1) # shape (Nopt,)
            eps_mask = diff > eps # shape (Nopt,)
            mask[mask] = numpy.logical_and(mask[mask], eps_mask)

            # Crop the X_N and Y_N arrays
            Nopt = numpy.sum(mask)
            if Nopt == 0:
                break
            x_N = x_N[eps_mask] # shape (NewNopt, 1)
            y_N = y_N[eps_mask] # shape (NewNopt, 1)

        # Return the normalized points
        return normalized_points










  