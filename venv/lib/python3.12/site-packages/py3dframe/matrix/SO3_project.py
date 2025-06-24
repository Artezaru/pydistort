import numpy
from .is_SO3 import is_SO3

def SO3_project(matrix: numpy.ndarray) -> numpy.ndarray:
    r"""
    Project a matrix to the special orthogonal group :math:`SO(3)` using SVD and minimisation of the frobenius norm.

    The orthogonal group `SO(3)` is the set of 3x3 orthonormal matrices with determinant equal to 1.

    To project a matrix to `SO(3)`, the SVD is computed and the orthogonal matrix is obtained by:

    .. math::

        \mathbf{O} = \mathbf{U} \mathbf{V}^T

    If the determinant of the orthogonal matrix is -1, the last column of the component :math:`\mathbf{U}` is multiplied by -1.
    
    where :math:`\mathbf{U}` and :math:`\mathbf{V}` are the left and right singular vectors of the matrix such as:

    .. math::

        \mathbf{M} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T

    .. seealso::

        - function :func:`py3dframe.matrix.is_SO3` for the check of special orthogonality.
        - function :func:`py3dframe.matrix.O3_project` for the projection of a matrix to the orthogonal group :math:`O(3)`.

    Parameters
    ----------
    matrix : array_like
        A 3x3 matrix to be projected.

    Returns
    -------
    numpy.ndarray
        The `SO(3)` projection of the matrix.

    Raises
    ------
    ValueError
        If the matrix is not 3x3.

    Examples
    --------

    >>> import numpy
    >>> from py3dframe import SO3_project
    >>> e1 = numpy.array([1, 1, 0])
    >>> e2 = numpy.array([-1, 1, 0])
    >>> e3 = numpy.array([0, 0, 1])
    >>> matrix = numpy.column_stack((e1, e2, e3))
    >>> print(SO3_project(matrix))
    [[ 0.70710678  -0.70710678  0.        ]
     [ 0.70710678  0.70710678  0.        ]
     [ 0.          0.          1.        ]]
    """
    matrix = numpy.array(matrix).astype(numpy.float64)

    if matrix.shape != (3, 3):
        raise ValueError("The matrix must be 3x3.")
    
    U, _, Vt = numpy.linalg.svd(matrix)
    orthogonal_matrix = U @ Vt

    if not is_SO3(orthogonal_matrix):
        U[:, 2] *= -1.0
        orthogonal_matrix = U @ Vt
        
    return orthogonal_matrix
