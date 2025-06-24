import numpy
import numbers
import sympy
from scipy.special import factorial

def litteral_radial_polynomial(rho: numpy.ndarray, n: int, m: int, rho_derivative: int = 0, default: float = numpy.nan) -> numpy.ndarray:
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

    This function only include litteral computation of the polynomials for :math:`n \leq 5`.

    .. note::

        If :math:`n > 5`, the function :func:`pyzernike.global_radial_polynomial` is used.

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

    # Rho power

    rho_pow = lambda power: numpy.power(rho_valid, power)

    # =================================================================================
    # ============================ Litteral computation ===============================
    # =================================================================================

    use_global = False

    # ---------------------------------------------------------------------------------
    # --------------------------- rho_derivative = 0 --------------------------------------
    # ---------------------------------------------------------------------------------

    if rho_derivative == 0:

        if n == 0 and m == 0:
            output_flat[valid_mask] = 1.0
        elif n == 1:
            if m == 1:
                output_flat[valid_mask] = rho_pow(1)
        elif n == 2:
            if m == 0:
                output_flat[valid_mask] = 2 * rho_pow(2) - 1
            elif m == 2:
                output_flat[valid_mask] = rho_pow(2)
        elif n == 3:
            if m == 1:
                output_flat[valid_mask] = 3 * rho_pow(3) - 2 * rho_pow(1)
            elif m == 3:
                output_flat[valid_mask] = rho_pow(3)
        elif n == 4:
            if m == 0:
                output_flat[valid_mask] = 6 * rho_pow(4) - 6 * rho_pow(2) + 1
            elif m == 2:
                output_flat[valid_mask] = 4 * rho_pow(4) - 3 * rho_pow(2)
            elif m == 4:
                output_flat[valid_mask] = rho_pow(4)
        elif n == 5:
            if m == 1:
                output_flat[valid_mask] = 10 * rho_pow(5) - 12 * rho_pow(3) + 3 * rho_pow(1)
            elif m == 3:
                output_flat[valid_mask] = 5 * rho_pow(5) - 4 * rho_pow(3)
            elif m == 5:
                output_flat[valid_mask] = rho_pow(5)
        else:
            use_global = True
        


    # ---------------------------------------------------------------------------------
    # --------------------------- rho_derivative = 1 --------------------------------------
    # ---------------------------------------------------------------------------------

    elif rho_derivative == 1:

        if n <= 0:
            output_flat[valid_mask] = 0.0
        elif n == 1:
            if m == 1:
                output_flat[valid_mask] = 1.0
        elif n == 2:
            if m == 0:
                output_flat[valid_mask] = 4 * rho_pow(1)
            elif m == 2:
                output_flat[valid_mask] = 2 * rho_pow(1)
        elif n == 3:
            if m == 1:
                output_flat[valid_mask] = 9 * rho_pow(2) - 2
            elif m == 3:
                output_flat[valid_mask] = 3 * rho_pow(2)
        elif n == 4:
            if m == 0:
                output_flat[valid_mask] = 24 * rho_pow(3) - 12 * rho_pow(1)
            elif m == 2:
                output_flat[valid_mask] = 16 * rho_pow(3) - 6 * rho_pow(1)
            elif m == 4:
                output_flat[valid_mask] = 4 * rho_pow(3)
        elif n == 5:
            if m == 1:
                output_flat[valid_mask] = 50 * rho_pow(4) - 36 * rho_pow(2) + 3
            elif m == 3:
                output_flat[valid_mask] = 25 * rho_pow(4) - 12 * rho_pow(2)
            elif m == 5:
                output_flat[valid_mask] = 5 * rho_pow(4)
        else:
            use_global = True

    # ---------------------------------------------------------------------------------
    # --------------------------- rho_derivative = 2 --------------------------------------
    # ---------------------------------------------------------------------------------

    elif rho_derivative == 2:

        if n <= 1:
            output_flat[valid_mask] = 0.0
        elif n == 2:
            if m == 0:
                output_flat[valid_mask] = 4.0
            elif m == 2:
                output_flat[valid_mask] = 2.0
        elif n == 3:
            if m == 1:
                output_flat[valid_mask] = 18 * rho_pow(1)
            elif m == 3:
                output_flat[valid_mask] = 6 * rho_pow(1)
        elif n == 4:
            if m == 0:
                output_flat[valid_mask] = 72 * rho_pow(2) - 12
            elif m == 2:
                output_flat[valid_mask] = 48 * rho_pow(2) - 6
            elif m == 4:
                output_flat[valid_mask] = 12 * rho_pow(2)
        elif n == 5:
            if m == 1:
                output_flat[valid_mask] = 200 * rho_pow(3) - 72 * rho_pow(1)
            elif m == 3:
                output_flat[valid_mask] = 100 * rho_pow(3) - 24 * rho_pow(1)
            elif m == 5:
                output_flat[valid_mask] = 20 * rho_pow(3)
        else:
            use_global = True

    # ---------------------------------------------------------------------------------
    # --------------------------- rho_derivative = 3 --------------------------------------
    # ---------------------------------------------------------------------------------

    elif rho_derivative == 3:

        if n <= 2:
            output_flat[valid_mask] = 0.0
        elif n == 3:
            if m == 1:
                output_flat[valid_mask] = 18.0
            elif m == 3:
                output_flat[valid_mask] = 6.0
        elif n == 4:
            if m == 0:
                output_flat[valid_mask] = 144 * rho_pow(1)
            elif m == 2:
                output_flat[valid_mask] = 96 * rho_pow(1)
            elif m == 4:
                output_flat[valid_mask] = 24 * rho_pow(1)
        elif n == 5:
            if m == 1:
                output_flat[valid_mask] = 600 * rho_pow(2) - 72
            elif m == 3:
                output_flat[valid_mask] = 300 * rho_pow(2) - 24
            elif m == 5:
                output_flat[valid_mask] = 60 * rho_pow(2)
        else:
            use_global = True

    # ---------------------------------------------------------------------------------
    # --------------------------- rho_derivative = 4 --------------------------------------
    # ---------------------------------------------------------------------------------

    elif rho_derivative == 4:

        if n <= 3:
            output_flat[valid_mask] = 0.0
        elif n == 4:
            if m == 0:
                output_flat[valid_mask] = 144.0
            elif m == 2:
                output_flat[valid_mask] = 96.0
            elif m == 4:
                output_flat[valid_mask] = 24.0
        elif n == 5:
            if m == 1:
                output_flat[valid_mask] = 1200 * rho_pow(1)
            elif m == 3:
                output_flat[valid_mask] = 600 * rho_pow(1)
            elif m == 5:
                output_flat[valid_mask] = 120 * rho_pow(1)
        else:
            use_global = True

    # ---------------------------------------------------------------------------------
    # --------------------------- rho_derivative = 5 --------------------------------------
    # ---------------------------------------------------------------------------------

    elif rho_derivative == 5:

        if n <= 4:
            output_flat[valid_mask] = 0.0
        elif n == 5:
            if m == 1:
                output_flat[valid_mask] = 1200.0
            elif m == 3:
                output_flat[valid_mask] = 600.0
            elif m == 5:
                output_flat[valid_mask] = 120.0
        else:
            use_global = True

    # ---------------------------------------------------------------------------------
    # --------------------------- rho_derivative > 5 --------------------------------------
    # --------------------------------------------------------------------------------- 
    
    else:
        use_global = True

    # Compute the output
    if use_global:
        output_flat[valid_mask] = global_radial_polynomial(rho_valid, n, m, rho_derivative=rho_derivative, default=default)
    output = output_flat.reshape(shape)
    return output
