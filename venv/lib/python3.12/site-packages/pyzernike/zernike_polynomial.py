import numpy
import numbers
from scipy.special import factorial
from .radial_polynomial import radial_polynomial

def zernike_polynomial(rho: numpy.ndarray, theta: numpy.ndarray, n: int, m: int, rho_derivative: int = 0, theta_derivative: int = 0, default: float = numpy.nan) -> numpy.ndarray:
    r"""
    Computes the Zernike polynomial :math:`Z_{n}^{m}(\rho, \theta)` for :math:`\rho \leq 1`.

    The Zernike polynomial is defined as follows:

    .. math::

        Z_{n}^{m}(\rho, \theta) = R_{n}^{m}(\rho) \cos(m \theta) \quad \text{if} \quad m > 0
    
    .. math::

        Z_{n}^{m}(\rho, \theta) = R_{n}^{-m}(\rho) \sin(-m \theta) \quad \text{if} \quad m < 0

    The derivative of order (derivative (a)) of the Zernike polynomial with respect to rho and order (derivative (b)) with respect to theta is defined as follows :

    .. math::

        \frac{\partial^{a}\partial^{b}Z_{n}^{m}(\rho, \theta)}{\partial \rho^{a} \partial \theta^{b}} = \frac{\partial^{a}R_{n}^{m}(\rho)}{\partial \rho^{a}} \frac{\partial^{b}\cos(m \theta)}{\partial \theta^{b}} \quad \text{if} \quad m > 0

    If :math:`|m| > n` or :math:`n < 0`, or :math:`(n - m)` is odd, the output is a zeros array with the same shape as :math:`\rho`.

    if :math:`\rho` is not in :math:`0 \leq \rho \leq 1` or :math:`\rho` is numpy.nan, the output is set to the default value (numpy.nan by default).

    .. note::

        The alias ``Z`` is available for this function.

        .. code-block:: python

            from pyzernike import Z

    .. seealso::

        For :math:`n > 10`, the function :func:`pyzernike.global_radial_polynomial` is used.

    Parameters
    ----------
    rho : numpy.ndarray
        The rho values.
    
    theta : numpy.ndarray
        The theta values.
    
    n : int
        The order of the Zernike polynomial.

    m : int
        The degree of the Zernike polynomial.

    rho_derivative : int, optional
        The order of the derivative with respect to rho. The default is 0.
    
    theta_derivative : int, optional
        The order of the derivative with respect to theta. The default is 0.

    default : float, optional
        The default value for invalid rho values. The default is numpy.nan.
    
    Returns
    -------
    numpy.ndarray
        The radial Zernike polynomial.
    
    Raises
    ------
    TypeError
        If the rho values are not a numpy array or if n and m are not integers.
        If the derivative is not an integer.
    ValueError
        If the derivative is negative.
        If rho and theta do not have the same shape.

    Examples
    --------
    
    .. code-block:: python

        import numpy
        from pyzernike import zernike_polynomial # or from pyzernike import Z
        rho = numpy.linspace(0, 1, 100)
        theta = numpy.linspace(0, 2*numpy.pi, 100)
        zernike_polynomial(rho, theta, 2, 0)

    returns the radial Zernike polynomial :math:`Z_{2}^{0}(\rho, \theta)` for :math:`\rho \leq 1`.

    .. code-block:: python

        import numpy
        from pyzernike import zernike_polynomial # or from pyzernike import Z
        rho = numpy.linspace(0, 1, 100)
        theta = numpy.linspace(0, 2*numpy.pi, 100)
        zernike_polynomial(rho, theta, 2, 0, rho_derivative=1)

    returns the derivative of the radial Zernike polynomial :math:`Z_{2}^{0}(\rho, \theta)` for :math:`\rho \leq 1` with respect to rho.
    """
    # Check the input parameters
    if not isinstance(rho, numpy.ndarray):
        raise TypeError("Rho values must be a numpy array.")
    if not isinstance(theta, numpy.ndarray):
        raise TypeError("Theta values must be a numpy array.")
    if not rho.shape == theta.shape:
        raise ValueError("Rho and theta must have the same shape.")
    if not isinstance(n, numbers.Integral) or not isinstance(m, numbers.Integral):
        raise TypeError("n and m must be integers.")
    if not isinstance(rho_derivative, numbers.Integral) or rho_derivative < 0:
        raise ValueError("The order of the derivative with respect to rho must be a positive integer.")
    if not isinstance(theta_derivative, numbers.Integral) or theta_derivative < 0:
        raise ValueError("The order of the derivative with respect to theta must be a positive integer.")
    if not isinstance(default, numbers.Real):
        raise TypeError("The default value must be a real number.")

    # Case of n < 0, (n - m) is odd or |m| > n
    if n < 0 or (n - m) % 2 != 0 or abs(m) > n:
        output = numpy.zeros_like(rho)
        return output
    
    # flatten the rho values to handle multiple dimensions
    shape = rho.shape
    rho_flat = rho.flatten()
    theta_flat = theta.flatten()
    
    # Create the mask for valid rho values
    unit_circle_mask = numpy.logical_and(0 <= rho_flat, rho_flat <= 1)
    nan_mask = numpy.isnan(rho_flat)
    valid_mask = numpy.logical_and(unit_circle_mask, ~nan_mask)

    # Initialize the output array
    output_flat = numpy.full_like(rho_flat, default)

    # Compute for valid rho values
    rho_valid = rho_flat[valid_mask]
    theta_valid = theta_flat[valid_mask]

    # Compute the radial polynomial
    radial_valid = radial_polynomial(rho_valid, n, abs(m), rho_derivative=rho_derivative)

    # Compute the cosine or sine term
    if m == 0:
        if theta_derivative == 0:
            cosine_valid = 1.0
        else:
            cosine_valid = 0.0
        
    if m > 0:
        if theta_derivative == 0:
            cosine_valid = numpy.cos(m * theta_valid)
        else:
            phase_shift = numpy.pi/2 * (theta_derivative%4)
            cosine_valid = (m ** theta_derivative) * numpy.cos(m * theta_valid + phase_shift)
        
    if m < 0:
        m = abs(m)
        if theta_derivative == 0:
            cosine_valid = numpy.sin(m * theta_valid)
        else:
            phase_shift = numpy.pi/2 * (theta_derivative%4)
            cosine_valid = (m ** theta_derivative) * numpy.sin(m * theta_valid + phase_shift)

    # Compute the output
    output_flat[valid_mask] = cosine_valid * radial_valid

    # Reshape the output array
    output = output_flat.reshape(shape)
    return output
    