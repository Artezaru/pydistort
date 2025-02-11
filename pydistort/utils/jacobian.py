import numpy
from typing import Callable, Iterable, Optional

class Jacobian(object):
    r"""
    A class to compute the Jacobian of a function.

    Lets consider a function f :math:`\mathbb{R}^n \to \mathbb{R}^m` and a point :math:`x \in \mathbb{R}^n`.

    The Jacobian of f at x is the matrix of partial derivatives of f at x.
    The Jacobian of f at x is a matrix of size :math:`m \times n` where the element :math:`J_{ij}` is the partial derivative of the :math:`i`-th component of f with respect to the :math:`j`-th variable.


    Properties
    ----------
    epsilon : float
        The epsilon value used to compute the Jacobian.

    Methods
    -------
    jacobian(f: Callable, x: numpy.ndarray, args, kwargs) -> numpy.ndarray
        Compute the Jacobian of the function f at the point x.

    """
    
    def __init__(self, epsilon: Optional[float] = 1e-6) -> None:
        """
        Parameters
        ----------
        epsilon : float, optional
            The epsilon value used to compute the Jacobian, by default 1e-6
        """
        self.epsilon = epsilon



    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon: float) -> None:
        if not isinstance(epsilon, (int, float)):
            raise TypeError("epsilon must be a number")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._epsilon = epsilon



    def jacobian(self, function: Callable, input_vector: numpy.ndarray, *args, **kwargs) -> numpy.ndarray:
        r"""
        Compute the Jacobian of the function ``f`` at the point ``x``.

        The function ``f`` must be a function that first argument is a numpy.ndarray and first returns a numpy.ndarray.

        .. code-block:: python

            def f(x: numpy.ndarray, *args, **kwargs) -> numpy.ndarray, ... :
                ...

        The input vector ``x`` and the output vector ``y`` must be 1-D numpy.ndarray.

        The output jacobian is a 2-D numpy.ndarray of size :math:`m \times n` where the element :math:`J_{ij}` is the partial derivative of the :math:`i`-th component of f with respect to the :math:`j`-th variable.

        .. math::

            J_{ij} = \frac{\partial f_i}{\partial x_j}

        The derivative is computed using the central difference method:

        .. math::

            \frac{\partial f_i}{\partial x_j} \approx \frac{f_i(x + \epsilon e_j) - f_i(x - \epsilon e_j)}{2 \epsilon}

        where :math:`e_j` is the unit vector along the :math:`j`-th axis and :math:`\epsilon` is the epsilon value.
        The :math:`\epsilon` value is set by the ``epsilon`` attribute of the class.

        In case of N-D numpy.ndarray, the coordinates of each input vector must be along the first axis.
        The higher dimensions of the input and output vectors must be the same.
        The jacobian will be a (N+1)-D numpy.ndarray where two first axis is the jacobian matrix for the first input vector.

        If a input vector contains a nan value, the jacobian will contain nan values.

        Parameters
        ----------
        function : Callable
            The function to compute the Jacobian. 
        
        input_vector : numpy.ndarray
            The input vector x where the Jacobian will be computed with shape ``(n,)`` or ``(n, ...)``.

        \*args
            Additional arguments to pass to the function.
        
        \*\*kwargs
            Additional keyword arguments to pass to the function.

        Returns
        -------
        numpy.ndarray
            The Jacobian matrix of the function at the point x with shape ``(m, n)`` or ``(m, n, ...)``.

        Raises
        ------
        TypeError
            If the function is not callable.
        ValueError
            If the input vector is not a non-empty numpy.ndarray.

        Examples
        --------

        .. code-block:: python
            
            import numpy as np
            from pysdic.utils.jacobian import Jacobian

            def function(input_vector: np.ndarray) -> np.ndarray:
                x = input_vector[0]
                y = input_vector[1]
                
                return np.array([x**2, x*y, y])

            jacobian = Jacobian(epsilon=1e-6)
            x = np.array([1, 2])
            print(jacobian.jacobian(f, x))

        .. code-block:: console

            array([[2., 0.],
                   [2., 1.],
                   [0., 1.]])

        With a N-D numpy.ndarray and multiple arguments:

        .. code-block:: python

            import numpy as np
            from pysdic.utils.jacobian import Jacobian

            def function(input_vector: np.ndarray, a, b=2) -> np.ndarray:
                # Each column is a different input vector
                x = input_vector[0,:]
                y = input_vector[1,:]
                
                return np.array([a*x**2, b*x*y, y])

            jacobian = Jacobian(epsilon=1e-6)
            x = np.array([[1, 2], [3, 4]])
            print(jacobian.jacobian(f, x, 2, b=3))

        .. code-block:: console

            array([[[ 4.,  0.],
                    [ 6.,  3.],
                    [ 0.,  1.]],

                   [[12.,  0.],
                    [12.,  9.],
                    [ 0.,  1.]]])
        """
        # Function to extract the first output of the function
        def get_output(function: Callable, *args, **kwargs) -> numpy.ndarray:
            # Call the function 
            output = function(*args, **kwargs)

            # Check if several outputs are returned
            if isinstance(output, tuple):
                return output[0]
            return output
            

        # Check the parameters
        if not callable(function):
            raise TypeError("The function must be callable.")
        if not isinstance(input_vector, numpy.ndarray):
            raise ValueError("The input vector must be a numpy.ndarray.")
        if input_vector.size == 0:
            raise ValueError("The input vector must be non-empty.")
        
        # Extract the dimension of the input vector
        N = input_vector.shape[0]

        # Extract the dimension of the output vector and compute the value at the input vector
        output_vector = get_output(function(input_vector, *args, **kwargs))

        # Check the output vector
        if not isinstance(output_vector, numpy.ndarray):
            raise ValueError("The output vector must be a numpy.ndarray.")
        if output_vector.size == 0:
            raise ValueError("The output vector must be non-empty.")
        if input_vector.shape[1:] != output_vector.shape[1:]:
            raise ValueError("The input and output vectors must have the same dimensions.")
        
        # Extract the dimension of the output vector
        M = output_vector.shape[0]

        # Initialize the Jacobian matrix
        jacobian = numpy.full((M, N, *input_vector.shape[1:]), numpy.nan)

        # Loop over the dimensions of the input vector
        for index_n in range(N):
            # Create the epsilon vector along the n-th axis
            delta = numpy.full_like(input_vector, 0)
            delta[index_n] = self.epsilon

            # Compute the partial derivative
            partial_derivative = (get_output(function(input_vector + delta, *args, **kwargs)) - get_output(function(input_vector - delta, *args, **kwargs))) / (2 * self.epsilon)
        
            # Assign the partial derivative to the Jacobian matrix
            jacobian[:, index_n, ...] = partial_derivative
        
        return jacobian
        
        




    