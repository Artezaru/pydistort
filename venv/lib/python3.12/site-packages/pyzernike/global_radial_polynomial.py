import numpy
import numbers
from scipy.special import gammaln

def global_radial_polynomial(rho: numpy.ndarray, n: int, m: int, rho_derivative: int = 0, default: float = numpy.nan) -> numpy.ndarray:
    r"""
    Computes the radial Zernike polynomial :math:`R_{n}^{m}(\rho)` for :math:`\rho \leq 1`.

    The radial Zernike polynomial is defined as follows:

    .. math::

        R_{n}^{m}(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} \rho^{n-2k}

    if :math:`n < 0`, :math:`m < 0`, :math:`n < m`, or :math:`(n - m)` is odd, the output is a zeros array with the same shape as :math:`\rho`.

    The derivative of order (derivative (a)) of the radial Zernike polynomial is defined as follows :

    .. math::

        \frac{d^{a}R_{n}^{m}(\rho)}{d\rho^{a}} = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} (n-2k) (n-2k-1) \ldots (n-2k-a+1) \rho^{n-2k-a}

    The computation of the factorial is done using the function :func:`scipy.special.gammaln` for better performance and stability.

    .. math::

        \text{log}(n!) = \text{gammaln}(n+1)

    So the coefficient of the radial polynomial is computed as follows:

    .. math::

        \frac{(n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} = \text{exp} (\text{gammaln}(n-k+1) - \text{gammaln}(k+1) - \text{gammaln}((n+m)/2 - k + 1) - \text{gammaln}((n-m)/2 - k + 1))
    
    if :math:`\rho` is not in :math:`0 \leq \rho \leq 1` or :math:`\rho` is numpy.nan, the output is set to the default value (numpy.nan by default).

    .. seealso::

        For n and m smaller than 10, the function :func:`pyzernike.radial_polynomial` is faster.

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
        If the rho_derivative is negative.

    Examples
    --------
    
    .. code-block:: python

        import numpy
        from pyzernike import global_radial_polynomial
        rho = numpy.linspace(0, 1, 100)
        global_radial_polynomial(rho, 2, 0)

    returns the radial Zernike polynomial :math:`R_{2}^{0}(\rho)` for :math:`\rho \leq 1`.

    .. code-block:: python

        import numpy
        from pyzernike import global_radial_polynomial
        rho = numpy.linspace(0, 1, 100)
        global_radial_polynomial(rho, 2, 0, rho_derivative=1)
    
    returns the first rho_derivative of the radial Zernike polynomial :math:`R_{2}^{0}(\rho)` for :math:`\rho \leq 1`.
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

    # Compute the number of terms
    s = (n-m)//2
    k = numpy.arange(0, s+1)

    # Vectorize coefficients
    log_k_coef = gammaln(n - k + 1) - gammaln(k + 1) - gammaln((n + m) // 2 - k + 1) - gammaln((n - m) // 2 - k + 1)
    sign = 1 - 2 * (k % 2)
    k_coef = sign * numpy.exp(log_k_coef)

    if rho_derivative == 0:
        coef = k_coef
    else:
        # Create a 2D array for (n - 2k - i)
        i = numpy.arange(rho_derivative)
        term_matrix = (n - 2 * k)[:, None] - i

        # Compute the product over the rho_derivative axis
        rho_derivative_coef = numpy.prod(term_matrix, axis=1)
        coef = k_coef * rho_derivative_coef

    # Compute the rho power
    exponent = n - 2 * k - rho_derivative
    exponent_positive_mask = exponent > 0
    exponent_0_mask = exponent == 0
    exponent_negative_mask = exponent < 0

    rho_powers = numpy.zeros((len(rho_valid), len(k)))
    rho_powers[:, exponent_positive_mask] = numpy.power(rho_valid[:, None], exponent[exponent_positive_mask])
    rho_powers[:, exponent_0_mask] = 1.0
    rho_powers[:, exponent_negative_mask] = 0.0

    # Compute the result
    result = numpy.dot(rho_powers, coef)

    # Assign the result to the output
    output_flat[valid_mask] = result
    
    # Reshape the output to the original shape
    output = output_flat.reshape(shape)
    return output
    