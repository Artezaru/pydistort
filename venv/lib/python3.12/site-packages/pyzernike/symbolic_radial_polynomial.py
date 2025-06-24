import numpy
import numbers
import sympy
from scipy.special import factorial
from .global_radial_polynomial import global_radial_polynomial

def symbolic_radial_polynomial(rho: numpy.ndarray, n: int, m: int, rho_derivative: int = 0, default: float = numpy.nan) -> numpy.ndarray:
    r"""
    Computes the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` for :math:`\rho \leq 1`.

    The radial Zernike polynomial is defined as follows:

    .. math::

        R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} \rho^{n-2k}

    if :math:`n < 0`, :math:`m < 0`, :math:`n < m`, or :math:`(n - m)` is odd, the output is a zeros array with the same shape as :math:`\rho`.

    The derivative of order (derivative (a)) of the radial Zernike polynomial is defined as follows :

    .. math::

        \frac{d^{a}R_{n}^{m}(\rho)}{d\rho^{a}} = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} (n-2k) (n-2k-1) \ldots (n-2k-a+1) \rho^{n-2k-a}

    if :math:`\rho` is not in :math:`0 \leq \rho \leq 1` or :math:`\rho` is numpy.nan, the output is set to the default value (numpy.nan by default).

    This function use sympy symbolic computation for the litteral expression of the :math:`n \leq 10` radial Zernike polynomial.
    
    .. note::

        If :math:`n > 10`, the function :func:`pyzernike.global_radial_polynomial` is used.

    .. seealso::

        the function :func:`pyzernike.radial_polynomial` is preferred for better performance.

    Parameters
    ----------
    rho : numpy.ndarray
        The rho values.
    
    n : int
        The order of the Zernike polynomial.

    m : int
        The degree of the Zernike polynomial.
    
    rho_derivative : int, optional
        The order of the rho_derivative. The default is 0.

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
        If the rho_derivative is not an integer.
    ValueError
        If the order of the rho_derivative is negative.
    """
    # Check the input parameters
    if not isinstance(rho, numpy.ndarray):
        raise TypeError("Rho values must be a numpy array.")
    if not isinstance(n, numbers.Integral) or not isinstance(m, numbers.Integral):
        raise TypeError("n and m must be integers.")
    if not isinstance(rho_derivative, numbers.Integral) or rho_derivative < 0:
        raise TypeError("The order of the rho_derivative must be a positive integer.")
    if not isinstance(default, numbers.Real):
        raise TypeError("The default value must be a real number.")

    # Case of n < 0, m < 0, n < m, or (n - m) is odd
    if n < 0 or m < 0 or n < m or (n - m) % 2 != 0:
        output = numpy.zeros_like(rho)
        return output

    # flatten the rho values to handle multiple dimensions
    shape = rho.shape
    rho_flat = rho.flatten()
    
    # Create the mask for valid rho values
    unit_circle_mask = numpy.logical_and(0 <= rho_flat, rho_flat <= 1)
    nan_mask = numpy.isnan(rho_flat)
    valid_mask = numpy.logical_and(unit_circle_mask, ~nan_mask)

    # Initialize the output array
    output_flat = numpy.full_like(rho_flat, default)

    # Compute for valid rho values
    rho_valid = rho_flat[valid_mask]

    # =================================================================================
    # ============================ Litteral computation ===============================
    # =================================================================================

    x = sympy.symbols('x')
    use_global = False

    if n == 0 and m == 0:
        expression = 1

    elif n == 1:
        if m == 1:
            expression = x

    elif n == 2:
        if m == 0:
            expression = 2 * x**2 - 1
        elif m == 2:
            expression = x**2

    elif n == 3:
        if m == 1:
            expression = 3 * x**3 - 2 * x
        elif m == 3:
            expression = x**3

    elif n == 4:
        if m == 0:
            expression = 6 * x**4 - 6 * x**2 + 1
        elif m == 2:
            expression = 4 * x**4 - 3 * x**2
        elif m == 4:
            expression = x**4

    elif n == 5:
        if m == 1:
            expression = 10 * x**5 - 12 * x**3 + 3 * x
        elif m == 3:
            expression = 5 * x**5 - 4 * x**3
        elif m == 5:
            expression = x**5

    elif n == 6:
        if m == 0:
            expression = 20 * x**6 - 30 * x**4 + 12 * x**2 - 1
        elif m == 2:
            expression = 15 * x**6 - 20 * x**4 + 6 * x**2
        elif m == 4:
            expression = 6 * x**6 - 5 * x**4
        elif m == 6:
            expression = x**6

    elif n == 7:
        if m == 1:
            expression = 35 * x**7 - 60 * x**5 + 30 * x**3 - 4 * x
        elif m == 3:
            expression = 21 * x**7 - 30 * x**5 + 10 * x**3
        elif m == 5:
            expression = 7 * x**7 - 6 * x**5
        elif m == 7:
            expression = x**7

    elif n == 8:
        if m == 0:
            expression = 70 * x**8 - 140 * x**6 + 90 * x**4 - 20 * x**2 + 1
        elif m == 2:
            expression = 56 * x**8 - 105 * x**6 + 60 * x**4 - 10 * x**2
        elif m == 4:
            expression = 28 * x**8 - 42 * x**6 + 15 * x**4
        elif m == 6:
            expression = 8 * x**8 - 7 * x**6
        elif m == 8:
            expression = x**8

    elif n == 9:
        if m == 1:
            expression = 126 * x**9 - 280 * x**7 + 210 * x**5 - 60 * x**3 + 5 * x
        elif m == 3:
            expression = 84 * x**9 - 168 * x**7 + 105 * x**5 - 20 * x**3
        elif m == 5:
            expression = 36 * x**9 - 56 * x**7 + 21 * x**5
        elif m == 7:
            expression = 9 * x**9 - 8 * x**7
        elif m == 9:
            expression = x**9

    else:
        use_global = True

    
    # Compute the output
    if use_global:
        output_flat[valid_mask] = global_radial_polynomial(rho_valid, n, m, rho_derivative=rho_derivative)
    else:
        # Compute the rho_derivative
        if rho_derivative > 0:
            expression = sympy.diff(expression, x, rho_derivative)

        # Convert the sympy expression to a numpy function
        func = sympy.lambdify(x, expression, 'numpy')
        output_flat[valid_mask] = func(rho_valid)

    # Reshape the output
    output = output_flat.reshape(shape)
    return output
