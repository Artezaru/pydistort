import numpy
import numbers
from scipy.special import factorial
from .zernike_polynomial import zernike_polynomial

def xy_zernike_polynomial(x: numpy.ndarray, y: numpy.ndarray, n: int, m: int, x_derivative: int = 0, y_derivative: int = 0, default: float = numpy.nan, Rx: float = 1.0, Ry: float = 1.0, x0: float = 0.0, y0: float = 0.0, alpha: float = 0.0, h: float = 0.0, theta1: float = 0.0, theta2: float = 2 * numpy.pi) -> numpy.ndarray:
    r"""
    Computes the Extended Zernike polynomial :math:`Z_{n}^{m}(x, y)` on a extended domain.

    .. seealso::

        For the standard Zernike polynomial, see :func:`pyzernike.zernike_polynomial` taking :math:`rho` and :math:`\theta` as inputs.

    .. seealso::

        For the mathematical development of the method, see the paper `Generalization of Zernike polynomials for regular portions of circles and ellipses` by Rafael Navarro, José L. López, José Rx. Díaz, and Ester Pérez Sinusía.
        The associated paper is available in the resources folder of the package.

        Download the PDF : :download:`PDF <../../../pyzernike/resources/Navarro and al. Generalization of Zernike polynomials for regular portions of circles and ellipses.pdf>`

    The user must provide the x and y coordinates of the points where the polynomial is evaluated and the parameters of the extended domain G:

    - :math:`R_x` and :math:`R_y` are the lenght of the semi-axis of the ellipse (outer boundary noted A and B on the figure).
    - :math:`x_0` and :math:`y_0` are the coordinates of the center of the ellipse.
    - :math:`\alpha` is the rotation angle of the ellipse in radians.
    - :math:`h=\frac{a}{R_x}=\frac{b}{R_y}` defining the inner boundary of the ellipse.
    - :math:`\theta_1` and :math:`\theta_2` are the angles defining the sector of the ellipse where the polynomial is described.

    .. figure:: ../../../pyzernike/resources/extended_parameters.png
        :width: 400px
        :align: center

        The parameters to define the extended domain of the Zernike polynomial. ``A`` and ``B`` are the semi-major and semi-minor axes of the ellipse, respectively, they correspond to :math:`R_x` and :math:`R_y`.

    The applied mapping is as follows:

    .. math::

        Zxy_{n}^{m}(x, y) = Z_{n}^{m}\left(\frac{r - h}{1 - h}, \frac{2 \pi (\theta - \theta_1)}{\theta_2 - \theta_1}\right)

    Where:

    .. math::

        r = \sqrt{\left(\frac{X}{R_x}\right)^{2} + \left(\frac{Y}{R_y}\right)^{2}}

    .. math::

        \theta = \text{atan2} (\frac{Y}{R_y}, \frac{X}{R_x})

    .. math::

        X = \cos(\alpha) (x - x_0) + \sin(\alpha) (y - y_0)
    
    .. math::

        Y = -\sin(\alpha) (x - x_0) + \cos(\alpha) (y - y_0)

    ..  note::

        An alias for this method is available as ``Zxy``.

    By default, the function computes the Zernike polynomial on a full circle with a radius of 1.0, i.e., the unit circle.
    This method is différent from the standard Zernike polynomial function becuase it uses the cartesian coordinates (x, y) instead of the polar coordinates (rho, theta).

    The derivative of order (derivative (a)) of the Zernike polynomial with respect to x and order (derivative (b)) with respect to y is defined as follows :

    .. math::

        \frac{\partial^{a}\partial^{b}Zxy_{n}^{m}(x, y)}{\partial x^{a} \partial y^{b}} = \frac{\partial^{a}\partial^{b}Zxy_{n}^{m}(\rho_{eq}(x,y), \theta_{eq}(x,y))}{\partial x^{a} \partial y^{b}}

    .. warning::

        Due to complications for the complication of the derivatives, the function supports the computation of the derivatives with respect to x and y for the extended Zernike polynomial only for (x_derivative, y_derivative) = (0, 0), (1, 0) and (0, 1).
        For more complex derivatives, it is recommended to use the standard Zernike polynomial function with the polar coordinates (rho, theta) as inputs and compute the assembled derivatives using the chain rule.

    .. math::

        \frac{\partial Zxy_{n}^{m}(x, y)}{\partial x} = \frac{\partial Zxy_{n}^{m}}{\partial \rho_{eq}} \frac{\partial \rho_{eq}}{\partial x} + \frac{\partial Zxy_{n}^{m}}{\partial \theta_{eq}} \frac{\partial \theta_{eq}}{\partial x}

    With : 

    .. math::

        \frac{\partial \rho_{eq}}{\partial z} = \frac{1}{1 - h} \frac{1}{r} \left(\frac{X}{R_x^2} \frac{\partial X}{\partial z} + \frac{Y}{R_y^2} \frac{\partial Y}{\partial z}\right)

    .. math::

        \frac{\partial \theta_{eq}}{\partial z} = \frac{2 \pi}{\theta_2 - \theta_1} \frac{1}{R_x R_y r^2} \left(X \frac{\partial Y}{\partial z} - Y \frac{\partial X}{\partial z}\right)

    Parameters
    ----------
    x : numpy.ndarray
        The x values.

    y : numpy.ndarray
        The y values.

    n : int
        The order of the Zernike polynomial.

    m : int
        The degree of the Zernike polynomial.

    x_derivative : int, optional
        The order of the derivative with respect to x. The default is 0.

    y_derivative : int, optional
        The order of the derivative with respect to y. The default is 0.

    default : float, optional
        The default value for invalid rho values. The default is numpy.nan.

    Rx : float, optional
        The length of the semi-major axis of the ellipse (outer boundary). The default is 1.0. Must be greater than 0.

    Ry : float, optional
        The length of the semi-minor axis of the ellipse (outer boundary). The default is 1.0. Must be greater than 0.
    
    x0 : float, optional
        The x-coordinate of the center of the ellipse. The default is 0.0.
    
    y0 : float, optional
        The y-coordinate of the center of the ellipse. The default is 0.0.

    alpha : float, optional
        The rotation angle of the ellipse in radians. The default is 0.0.

    h : float, optional
        The ratio of the inner semi-axis to the outer semi-axis. The default is 0. Must be in the range [0, 1[.
    
    theta1 : float, optional
        The starting angle of the sector in radians. The default is 0.0.

    theta2 : float, optional
        The ending angle of the sector in radians. The default is 2 * pi. Must be greater than theta1 and less than or equal to theta1 + 2 * pi.
    
    Returns
    -------
    numpy.ndarray
        The radial Zernike polynomial.
    
    Raises
    ------
    TypeError
        If the x or y values are not a numpy array or if n and m are not integers.
        If the derivative is not an integer.
        If Rx, Ry, alpha, h, theta1, or theta2 are not valid types.
    ValueError
        If the derivative is negative.
        If x and y do not have the same shape.
        If Rx, Ry, alpha, h, theta1, or theta2 are not in the valid ranges.

    Examples
    --------
    
    Let's consider a full circle with a radius of 10 centered at the origin (0, 0).
    The value of the zernike polynomial :math:`Z_{2}^{0}` at the point (x, y) is given by:

    .. code-block:: python

        import numpy
        from pyzernike import xy_zernike_polynomial # or Zxy
        x = numpy.linspace(-10, 10, 100)
        y = numpy.linspace(-10, 10, 100)
        X, Y = numpy.meshgrid(x, y)

        zernike = xy_zernike_polynomial(X, Y, 2, 0, Rx=10, Ry=10, x0=0.0, y0=0.0) # Shape similar to X and Y

    The first derivative with respect to y is given by:

    .. code-block:: python

        import numpy
        from pyzernike import xy_zernike_polynomial # or Zxy

        x = numpy.linspace(-10, 10, 100)
        y = numpy.linspace(-10, 10, 100)
        X, Y = numpy.meshgrid(x, y)
        zernike_derivative_y = xy_zernike_polynomial(X, Y, 2, 0, y_derivative=1, Rx=10, Ry=10, x0=0.0, y0=0.0) # Shape similar to X and Y


    For a full circle with a radius of R and centered at (x_0, y_0), the value of the zernike polynomial :math:`Z_{2}^{0}` at the point (x, y) is given by:

    .. code-block:: python

        Zxy(x, y, n, m, Rx=R, Ry=R, x0=x_0, y0=y_0)

    For an ellipse with semi-major axis Rx and semi-minor axis Ry, centered at (x_0, y_0) and rotated by an angle alpha, the value of the zernike polynomial :math:`Z_{2}^{0}` at the point (x, y) is given by:

    .. code-block:: python

        Zxy(x, y, n, m, Rx=Rx, Ry=Ry, x0=x_0, y0=y_0, alpha=alpha)

    For a annular sector of the ellipse with semi-major axis Rx and semi-minor axis Ry, centered at (x_0, y_0), rotated by an angle alpha, and an aspect ratio h, the value of the zernike polynomial :math:`Z_{2}^{0}` at the point (x, y) is given by:

    .. code-block:: python

        Zxy(x, y, n, m, Rx=Rx, Ry=Ry, x0=x_0, y0=y_0, alpha=alpha, h=h)

    """
    # Check the input parameters
    if not isinstance(x, numpy.ndarray):
        raise TypeError("X values must be a numpy array.")
    if not isinstance(y, numpy.ndarray):
        raise TypeError("Y values must be a numpy array.")
    if not x.shape == y.shape:
        raise ValueError("X and Y must have the same shape.")
    if not isinstance(n, numbers.Integral) or not isinstance(m, numbers.Integral):
        raise TypeError("n and m must be integers.")
    if not isinstance(x_derivative, numbers.Integral) or x_derivative < 0:
        raise ValueError("The order of the derivative with respect to x must be a positive integer.")
    if not isinstance(y_derivative, numbers.Integral) or y_derivative < 0:
        raise ValueError("The order of the derivative with respect to y must be a positive integer.")
    if not isinstance(default, numbers.Real):
        raise TypeError("The default value must be a real number.")
    
    # Check the parameters for the extended domain
    if not isinstance(Rx, numbers.Real) or Rx <= 0:
        raise TypeError("Rx must be a positive real number.")
    if not isinstance(Ry, numbers.Real) or Ry <= 0:
        raise TypeError("Ry must be a positive real number.")
    if not isinstance(x0, numbers.Real):
        raise TypeError("x0 must be a real number.")
    if not isinstance(y0, numbers.Real):
        raise TypeError("y0 must be a real number.")
    if not isinstance(alpha, numbers.Real):
        raise TypeError("Alpha must be a real number.")
    if not isinstance(h, numbers.Real) or not (0 <= h < 1):
        raise TypeError("h must be a real number in the range [0, 1[.")
    if not isinstance(theta1, numbers.Real):
        raise TypeError("Theta1 must be a real number.")
    if not isinstance(theta2, numbers.Real):
        raise TypeError("Theta2 must be a real number.")
    if abs(theta2 - theta1) > 2 * numpy.pi:
        raise ValueError("The angle between theta1 and theta2 must be less than or equal to 2 * pi.")
    if theta1 >= theta2:
        raise ValueError("Theta1 must be less than Theta2.")
    
    # Complete circle case
    if abs(theta2 - theta1 - 2 * numpy.pi) < 1e-10:
        closed_circle = True
    else:
        closed_circle = False
    
    # Warning on derivatives
    if (x_derivative, y_derivative) not in [(0, 0), (1, 0), (0, 1)]:
        raise ValueError("The function supports only the derivatives (0, 0), (1, 0) and (0, 1). For more complex derivatives, use the standard Zernike polynomial function with polar coordinates.")
   
    # flatten the rho values to handle multiple dimensions
    shape = x.shape
    x_flat = x.flatten()
    y_flat = y.flatten()
    x_centered = x_flat - x0
    y_centered = y_flat - y0
    X = numpy.cos(alpha) * x_centered + numpy.sin(alpha) * y_centered
    Y = - numpy.sin(alpha) * x_centered + numpy.cos(alpha) * y_centered

    # Compute the equivalent polar coordinates
    r_prim = numpy.sqrt((X / Rx) ** 2 + (Y / Ry) ** 2)
    theta_prim = numpy.arctan2(Y / Ry, X / Rx)

    # Angular convertion in 0 to 2*pi range
    theta_prim_2pi = theta_prim % (2 * numpy.pi)
    theta1_2pi = theta1 % (2 * numpy.pi)
    theta2_2pi = theta2 % (2 * numpy.pi)
    
    # Compute the equivalent rho values
    rho_eq = (r_prim - h) / (1 - h)

    # Compute the equivalent theta values
    if closed_circle:
        # theta_1 = theta_2 , theta_eq = 0 for theta_1
        t = theta_prim_2pi
        t1 = t2 = theta1_2pi
        theta_eq = t - t1
    elif theta1_2pi < theta2_2pi and not closed_circle:
        # 0 -------[t1, t2]------- 2*pi
        t = theta_prim_2pi
        t1 = theta1_2pi
        t2 = theta2_2pi
        theta_eq = 2 * numpy.pi * (t - t1) / (t2 - t1)
    elif theta1_2pi > theta2_2pi and not closed_circle:
        # [0, t2]-------[t1, 2*pi]
        # Transform theta_prim_2pi to be in [theta1, theta1 + 2 * pi]
        t = theta_prim_2pi.copy()
        t[t < theta1_2pi] += 2 * numpy.pi # Define know in [theta1, theta1 + 2 * pi]
        t1 = theta1_2pi
        t2 = theta2_2pi + 2 * numpy.pi
        theta_eq = 2 * numpy.pi * (t - t1) / (t2 - t1)
    else:
        raise ValueError("Invalid theta1 and theta2 values. They must define a valid sector.")

    # Prepare the output array
    output_flat = numpy.full_like(rho_eq, default, dtype=numpy.float64)

    # Mask for valid rho values
    rho_valid_mask = numpy.logical_and(0 <= rho_eq, rho_eq <= 1)

    # Mask for valid theta values
    if closed_circle:
        # All theta values are valid in a closed circle
        theta_valid_mask = numpy.ones_like(theta_eq, dtype=bool)
    else:
        # Only consider theta values within the defined sector
        theta_valid_mask = numpy.logical_and(0 <= theta_eq, theta_eq <= 2 * numpy.pi)

    # Combine the masks
    valid_mask = numpy.logical_and(rho_valid_mask, theta_valid_mask)

    # According to the expected derivatives, compute the Zernike polynomial
    if (x_derivative, y_derivative) == (0, 0):
        output_flat[valid_mask] = zernike_polynomial(rho_eq[valid_mask], theta_eq[valid_mask], n, m, rho_derivative=0, theta_derivative=0, default=default)

    elif (x_derivative, y_derivative) == (1, 0):
        rho_derivative = (1 / (1 - h)) * (1 / r_prim[valid_mask]) * ( (X[valid_mask] / Rx**2) * numpy.cos(alpha) - (Y[valid_mask] / Ry**2) * numpy.sin(alpha))
        theta_derivative = (2 * numpy.pi / (theta2 - theta1)) * (1 / (Rx * Ry * r_prim[valid_mask]**2)) * ( - X[valid_mask] * numpy.sin(alpha) - Y[valid_mask] * numpy.cos(alpha))
        zernike_rho_derivative = zernike_polynomial(rho_eq[valid_mask], theta_eq[valid_mask], n, m, rho_derivative=1, theta_derivative=0, default=default)
        zernike_theta_derivative = zernike_polynomial(rho_eq[valid_mask], theta_eq[valid_mask], n, m, rho_derivative=0, theta_derivative=1, default=default)
        output_flat[valid_mask] = zernike_rho_derivative * rho_derivative + zernike_theta_derivative * theta_derivative

    elif (x_derivative, y_derivative) == (0, 1):
        rho_derivative = (1 / (1 - h)) * (1 / r_prim[valid_mask]) * ( (X[valid_mask] / Rx**2) * numpy.sin(alpha) + (Y[valid_mask] / Ry**2) * numpy.cos(alpha))
        theta_derivative = (2 * numpy.pi / (theta2 - theta1)) * (1 / (Rx * Ry * r_prim[valid_mask]**2)) * ( X[valid_mask] * numpy.cos(alpha) - Y[valid_mask] * numpy.sin(alpha))
        zernike_rho_derivative = zernike_polynomial(rho_eq[valid_mask], theta_eq[valid_mask], n, m, rho_derivative=1, theta_derivative=0, default=default)
        zernike_theta_derivative = zernike_polynomial(rho_eq[valid_mask], theta_eq[valid_mask], n, m, rho_derivative=0, theta_derivative=1, default=default)
        output_flat[valid_mask] = zernike_rho_derivative * rho_derivative + zernike_theta_derivative * theta_derivative

    else:
        raise ValueError("The function supports only the derivatives (0, 0), (1, 0) and (0, 1). For more complex derivatives, use the standard Zernike polynomial function with polar coordinates.")

    # Reshape the output array
    output = output_flat.reshape(shape)
    
    return output
    