import numpy
import numbers
from scipy.special import factorial
from .global_radial_polynomial import global_radial_polynomial
from .litteral_radial_polynomial import litteral_radial_polynomial
from .symbolic_radial_polynomial import symbolic_radial_polynomial

def radial_polynomial(rho: numpy.ndarray, n: int, m: int, rho_derivative: int = 0, default: float = numpy.nan) -> numpy.ndarray:
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

    According to the value of :math:`n`, the function will call the appropriate function to compute the radial Zernike polynomial.

    .. seealso::

        - For :math:`n \leq 5`, the function :func:`pyzernike.litteral_radial_polynomial` is used.
        - For :math:`5 < n \leq 10`, the function :func:`pyzernike.symbolic_radial_polynomial` is used.
        - For :math:`n > 10`, the function :func:`pyzernike.global_radial_polynomial` is used.

    .. note::

        The alias ``R`` is available for this function.

        .. code-block:: python

            from pyzernike import R

    The output array as the same shape as the input array :math:`\rho`.

    Parameters
    ----------
    rho : numpy.ndarray
        The rho values.
    
    n : int
        The order of the Zernike polynomial

    m : int
        The degree of the Zernike polynomial.

    rho_derivative : int, optional
        The order of the rho_derivative. The default is 0.

    default : float, optional
        The default value for invalid rho values. The default is numpy.nan.

    Npool : int, optional
        The number of pools for the parallel computation. The default is 4.
    
    Returns
    -------
    numpy.ndarray
        The radial Zernike polynomial with the same shape as :math:`\rho`.

    Raises
    ------
    TypeError
        If the rho values are not a numpy array.
        If the rho_derivative is not an integer.
    ValueError
        If the rho_derivative is negative.

    Examples
    --------
    
    .. code-block:: python

        import numpy
        from pyzernike import radial_polynomial # or from pyzernike import R
        rho = numpy.linspace(0, 1, 100)
        radial_polynomial(rho, 2, 0)

    returns the radial Zernike polynomial :math:`R_{2}^{0}(\rho)` for :math:`\rho \leq 1`.

    .. code-block:: python

        import numpy
        from pyzernike import radial_polynomial # or from pyzernike import R
        rho = numpy.linspace(0, 1, 100)
        radial_polynomial(rho, 2, 0, rho_derivative=1)
    
    returns the first rho_derivative of the radial Zernike polynomial :math:`R_{2}^{0}(\rho)` for :math:`\rho \leq 1`.
    """
    # Check the input parameters
    if not isinstance(rho, numpy.ndarray):
        raise TypeError("Rho values must be a numpy array.")
    if not isinstance(n, numbers.Integral) or not isinstance(m, numbers.Integral):
        raise TypeError("n and m must be integers.")
    if not isinstance(rho_derivative, numbers.Integral) or rho_derivative < 0:
        raise TypeError("The rho_derivative must be a non-negative integer.")
    if not isinstance(default, numbers.Real):
        raise TypeError("The default value must be a real number.")

    if n < 0 or m < 0 or n < m or (n - m) % 2 != 0:
        output = numpy.zeros_like(rho)
        return output
    
    if n <= 5:
        output = litteral_radial_polynomial(rho, n, m, rho_derivative, default)
    elif n <= 10:
        output = symbolic_radial_polynomial(rho, n, m, rho_derivative, default)
    else:
        output = global_radial_polynomial(rho, n, m, rho_derivative, default)
    
    return output