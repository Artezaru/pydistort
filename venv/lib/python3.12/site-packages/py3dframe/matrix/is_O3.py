import numpy

def is_O3(matrix: numpy.ndarray, tolerance: float = 1e-6) -> bool:
    r"""
    Check if the given matrix is orthonormal (include on the orthogonal group :math:`O(3)`).

    A matrix is orthonormal if its transpose is equal to its inverse.

    .. math::

        M^T = M^{-1}

    .. seealso::

        - function :func:`py3dframe.matrix.is_SO3` for the check of special orthogonality.
        - function :func:`py3dframe.matrix.O3_project` for the projection of a matrix to the orthogonal group :math:`O(3)`.

    Parameters
    ----------
    matrix : array_like
        The matrix with shape (3, 3).

    tolerance : float, optional
        The tolerance for the comparison of the matrix with the identity matrix. Default is 1e-6.
    
    Returns
    -------
    bool
        True if the matrix is in the orthogonal group, False otherwise.

    Raises
    ------
    ValueError
        If the matrix is not 3x3.

    Examples
    --------

    >>> import numpy
    >>> from py3dframe.matrix import is_O3
    >>> e1 = numpy.array([1, 1, 0])
    >>> e2 = numpy.array([-1, 1, 0])
    >>> e3 = numpy.array([0, 0, 1])
    >>> matrix = numpy.column_stack((e1, e2, e3))
    >>> print(is_O3(matrix))
    False

    >>> import numpy
    >>> from py3dframe.matrix import is_O3
    >>> e1 = numpy.array([1, 1, 0]) / numpy.sqrt(2)
    >>> e2 = numpy.array([-1, 1, 0]) / numpy.sqrt(2)
    >>> e3 = numpy.array([0, 0, 1])
    >>> matrix = numpy.column_stack((e1, e2, e3))
    >>> print(is_O3(matrix))
    True

    >>> import numpy
    >>> from py3dframe.matrix import is_O3
    >>> e1 = numpy.array([-1, 1, 0]) / numpy.sqrt(2)
    >>> e2 = numpy.array([1, 1, 0]) / numpy.sqrt(2)
    >>> e3 = numpy.array([0, 0, 1])
    >>> matrix = numpy.column_stack((e1, e2, e3))
    >>> print(is_O3(matrix))
    True

    >>> import numpy
    >>> from py3dframe.matrix import is_O3
    >>> e1 = numpy.array([1, 1, 1]) / numpy.sqrt(3)
    >>> e2 = numpy.array([-1, 1, 0]) / numpy.sqrt(2)
    >>> e3 = numpy.array([0, 0, 1])
    >>> matrix = numpy.column_stack((e1, e2, e3))
    >>> print(is_O3(matrix))
    False

    """
    matrix = numpy.array(matrix).astype(numpy.float64)

    if matrix.shape != (3, 3):
        raise ValueError("The matrix must be 3x3.")
    
    dot_products = matrix.T @ matrix
    return numpy.allclose(dot_products, numpy.eye(3), atol=tolerance)