from __future__ import annotations
import scipy
import numpy
from .rotation import Rotation
from typing import Union, Tuple

def switch_RT_convention(
        rotation: Rotation,
        translation: numpy.ndarray,
        input_convention: Union[int, str] = 0,
        output_convention: Union[int, str] = 0,
    ) -> Tuple[Rotation, numpy.ndarray]:  
    r"""
    Lets consider two frames of reference :math:`E` and :math:`F`.
    Lets consider a point :math:`X` whose coordinates in the frame :math:`E` are :math:`X_E` and in the frame :math:`F` are :math:`X_F`.
    The transformation from the frame :math:`E` to the frame :math:`F` is defined by a rotation matrix :math:`R` and a translation vector :math:`T`.

    It exists 8 principal conventions to define the transformation between the two frames of reference.

    +---------------------+----------------------------------------------------------------+
    | Index               | Formula                                                        |
    +=====================+================================================================+
    | 0                   | :math:`\mathbf{X}_E = \mathbf{R} \mathbf{X}_F + \mathbf{T}`    |
    +---------------------+----------------------------------------------------------------+
    | 1                   | :math:`\mathbf{X}_E = \mathbf{R} \mathbf{X}_F - \mathbf{T}`    |
    +---------------------+----------------------------------------------------------------+
    | 2                   | :math:`\mathbf{X}_E = \mathbf{R} (\mathbf{X}_F + \mathbf{T})`  |
    +---------------------+----------------------------------------------------------------+
    | 3                   | :math:`\mathbf{X}_E = \mathbf{R} (\mathbf{X}_F - \mathbf{T})`  |
    +---------------------+----------------------------------------------------------------+
    | 4                   | :math:`\mathbf{X}_F = \mathbf{R} \mathbf{X}_E + \mathbf{T}`    |
    +---------------------+----------------------------------------------------------------+
    | 5                   | :math:`\mathbf{X}_F = \mathbf{R} \mathbf{X}_E - \mathbf{T}`    |
    +---------------------+----------------------------------------------------------------+
    | 6                   | :math:`\mathbf{X}_F = \mathbf{R} (\mathbf{X}_E + \mathbf{T})`  |
    +---------------------+----------------------------------------------------------------+
    | 7                   | :math:`\mathbf{X}_F = \mathbf{R} (\mathbf{X}_E - \mathbf{T})`  |
    +---------------------+----------------------------------------------------------------+

    This function allows to switch between the 8 conventions.

    .. note::

        In the convention 0, the columns of the rotation matrix are the coordinates of the frame :math:`F` in the frame :math:`E` and 
        the translation vector is the origin of the frame :math:`F` in the frame :math:`E` coordinates.

        If the axes of the frame :math:`F` are noted :math:`\vec{i}`, :math:`\vec{j}` and :math:`\vec{k}`.
        In the convention 0, the rotation matrix is :math:`\begin{bmatrix} \vec{i} & \vec{j} & \vec{k} \end{bmatrix}`.

        .. code-block:: python
            
            import numpy
            from py3dframe import Rotation
            vec_i = numpy.array([1, 1, 0])
            vec_j = numpy.array([-1, 1, 0])
            vec_k = numpy.array([0, 0, 1])
            matrix = numpy.column_stack((vec_i, vec_j, vec_k))
            R = Rotation.from_matrix(matrix)

    .. warning::        

        The points :math:`X_E` and :math:`X_F` are 3 elements vectors with shape (3, 1) such as :math:`X_E = R X_F + T` is valid.
        To use this point out of the package, the operation can be processed using as follows:

        .. code-block:: python

            X_E = R.as_matrix() @ X_F + T # The shape are compatible
            X_E = R.apply(X_F.T).T + T # Use the transpose to get the shape (1, 3) numpy-array convention.


    
    Parameters
    ----------
    rotation :Rotation
        The rotation of the transformation. It must be a scipy.spatial.transform.Rotation.

    translation : numpy.ndarray
        The translation vector of the transformation. It must be a 3 elements vector with shape (3, 1).

    input_convention : Union[int, str], optional
        The convention of the input transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is 0.

    output_convention : Union[int, str], optional
        The convention of the output transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is 0.
    
    Returns
    -------
    R_out : Rotation
        The rotation of the output transformation.
    
    T_out : numpy.ndarray
        The translation vector of the output transformation with shape (3, 1).
    
    Raises
    ------
    TypeError
        If the input_convention or the output_convention is not an integer.
        If the rotation is not a scipy.spatial.transform.Rotation.
        If the translation vector is not a 3 elements vector.
    ValueError
        If the input_convention or the output_convention is not between 0 and 7.
        If the translation vector is not a 3 elements vector.

    Examples
    --------

    >>> import numpy as np
    >>> from py3dframe import Rotation, switch_RT_convention
    >>> R = numpy.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]])
    >>> R = R / numpy.linalg.norm(R, axis=0) # Normalize the columns to get an orthonormal matrix
    >>> R = Rotation.from_matrix(R)
    >>> T = [3, 2, 1]
    >>> R_out, T_out = switch_RT_convention(R, T, 0, 2)
    >>> R_out.as_matrix()
    array([[0.70710678, 0.70710678, 0.        ],
           [-0.70710678, 0.70710678, 0.        ],
           [0.        , 0.        , 1.        ]])
    >>> T_out
    array([[3.53553391],
           [-0.70710678],
           [1.        ]])

    """
    # Check the input_convention and the output_convention
    if not isinstance(input_convention, int):
        raise TypeError("The input_convention must be an integer.")
    if not isinstance(output_convention, int):
        raise TypeError("The output_convention must be an integer.")
    if not isinstance(rotation, Rotation):
        raise TypeError("The rotation must be a scipy.spatial.transform.Rotation.")
    
    # Check the input_convention and the output_convention
    if not 0 <= input_convention <= 7:
        raise ValueError("The input_convention must be between 0 and 7.")
    if not 0 <= output_convention <= 7:
        raise ValueError("The output_convention must be between 0 and 7.")

    # Get the translation vector
    translation = numpy.array(translation).reshape((3, 1)).astype(numpy.float64)

    # Case 0. input_convention = output_convention
    if input_convention == output_convention:
        R_out = rotation
        T_out = translation
        return R_out, T_out
    
    # Create the tuple of conventions
    switch = (input_convention, output_convention)

    # Apply the transformation
    if switch in [(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)]:
        R_out = rotation
        T_out = - translation
    elif switch in [(0, 2), (1, 3), (4, 6), (5, 7)]:
        R_out = rotation
        T_out = rotation.inv().apply(translation.T).T
    elif switch in [(0, 3), (1, 2), (4, 7), (5, 6)]:
        R_out = rotation
        T_out = - rotation.inv().apply(translation.T).T
    elif switch in [(2, 0), (3, 1), (6, 4), (7, 5)]:
        R_out = rotation
        T_out = rotation.apply(translation.T).T
    elif switch in [(2, 1), (3, 0), (6, 5), (7, 4)]:
        R_out = rotation
        T_out = - rotation.apply(translation.T).T
    elif switch in [(0, 4), (1, 5), (4, 0), (5, 1)]:
        R_out = rotation.inv()
        T_out = - rotation.inv().apply(translation.T).T
    elif switch in [(0, 5), (1, 4), (4, 1), (5, 0)]:
        R_out = rotation.inv()
        T_out = rotation.inv().apply(translation.T).T
    elif switch in [(0, 6), (1, 7), (2, 4), (3, 5), (4, 2), (5, 3), (6, 0), (7, 1)]:
        R_out = rotation.inv()
        T_out = - translation
    elif switch in [(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)]:
        R_out = rotation.inv()
        T_out = translation
    elif switch in [(2, 6), (3, 7), (6, 2), (7, 3)]:
        R_out = rotation.inv()
        T_out = - rotation.apply(translation.T).T
    elif switch in [(2, 7), (3, 6), (6, 3), (7, 2)]:
        R_out = rotation.inv()
        T_out = rotation.apply(translation.T).T
    else:
        raise ValueError("The switch is not defined.")
    return R_out, T_out




