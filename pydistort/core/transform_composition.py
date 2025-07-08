from typing import Sequence, Tuple, Optional
import numpy

from .transform import Transform



class TransformComposition(Transform):
    r"""
    A class to represent the composition of multiple transformations.

    It inherits from the :class:`pydistort.core.Transform` class and overrides the `_transform` and `_inverse_transform` methods to apply the composition of transformations.

    If the transformations to represent is :math:`T_n (T_{n-1}(...(T_1(X))))`, the transformations must be given in the order of application, i.e. from the first to the last transformation.

    .. code-block:: python

        transformations = [Transform1(), Transform2(), ..., TransformN()]

    Parameters
    ----------
    transformations : Sequence[Transform]
        A list of transformations to be composed. Each transformation must be an instance of the `Transform` class or its subclasses.

    """
    def __init__(self, transformations: Sequence[Transform]):
        super().__init__()
        
        # Check if the transformations are a sequence of Transform instances
        if not isinstance(transformations, Sequence):
            raise TypeError(f"transformations must be a sequence, got {type(transformations)}")
        if len(transformations) == 0:
            raise ValueError("transformations must contain at least one transformation.")
        for t in transformations:
            if not isinstance(t, Transform):
                raise TypeError(f"Each transformation must be an instance of Transform, got {type(t)}")
        
        # Check the chain of dimensions
        for i in range(len(transformations) - 1):
            if transformations[i].output_dim != transformations[i + 1].input_dim:
                raise ValueError(f"Output dimension of transformation {i} ({transformations[i].output_dim}) does not match input dimension of transformation {i + 1} ({transformations[i + 1].input_dim}).")
            
        # Set the transformations and their parameters
        self.transformations = transformations


    @property
    def input_dim(self) -> int:
        r"""
        The input dimension of the first transformation in the composition.
        """
        return self.transformations[0].input_dim
    

    @property
    def output_dim(self) -> int:
        r"""
        The output dimension of the last transformation in the composition.
        """
        return self.transformations[-1].output_dim
    

    @property
    def Nparams(self) -> int:
        r"""
        The total number of parameters in the composition of transformations.
        This is the sum of the number of parameters of each transformation in the composition.
        """
        return sum(t.Nparams for t in self.transformations)
    

    @property
    def parameters(self) -> numpy.ndarray:
        r"""
        The parameters of the composition of transformations.
        This is a concatenation of the parameters of each transformation in the composition.

        If a transformation does not have parameters, it is represented by a zero array of its Nparams.
        The resulting array has shape (Nparams,).
        """
        return numpy.concatenate([t.parameters if t.parameters is not None else numpy.zeros(t.Nparams, dtype=numpy.float64) for t in self.transformations], axis=0)
    

    @parameters.setter
    def parameters(self, value: numpy.ndarray):
        r"""
        Set the parameters of the composition of transformations.

        The input value must be a 1D numpy array with shape (Nparams,).
        The parameters are set to the corresponding transformations in the composition.
        If a transformation does not have parameters, it is skipped.
        
        Parameters
        ----------
        value : numpy.ndarray
            The parameters to set. Must be a 1D numpy array with shape (Nparams,).
        """
        if not isinstance(value, numpy.ndarray) or value.ndim != 1:
            raise TypeError(f"Parameters must be a 1D numpy array, got {type(value)} with {value.ndim} dimensions.")
        
        if value.shape[0] != self.Nparams:
            raise ValueError(f"Parameters must have {self.Nparams} elements, got {value.shape[0]} elements.")
        
        # Set the parameters for each transformation
        start = 0
        for t in self.transformations:
            end = start + t.Nparams
            t.parameters = value[start:end]
            start = end


    def is_set(self) -> bool:
        r"""
        Check if the parameters of all transformations in the composition are set.

        Returns
        -------
        bool
            True if all transformations have their parameters set, False otherwise.
        """
        return all(t.is_set() for t in self.transformations)
    

    def _transform(
        self,
        points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        **kwargs
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the composition of transformations to the given points.

        This method applies each transformation in the composition sequentially to the input points.

        The jacobian matrices with respect to the input points and the parameters are computed using the chain rule.

        Parameters
        ----------
        points : numpy.ndarray
            The input points to be transformed. Shape (Npoints, input_dim).

        dx : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is False.

        **kwargs
            Additional keyword arguments for the transformations.

        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]
            A tuple containing:

            - `transformed_points`: The transformed points of shape (Npoints, output_dim).
            - `jacobian_dx`: The Jacobian matrix with respect to the input points of shape (Npoints, output_dim, input_dim) if `dx` is True, otherwise None.
            - `jacobian_dp`: The Jacobian matrix with respect to the parameters of the transformation of shape (Npoints, output_dim, Nparams) if `dp` is True, otherwise None.
        """
        Npoints = points.shape[0]  # Number of points in computation
        jacobian_dx_list = []
        jacobian_dp_list = []

        # Iterate over each transformation in the composition
        for index, t in enumerate(self.transformations):
            # Apply the transformation to the points
            transformed_points, jacobian_dx, jacobian_dp = t._transform(points, dx=dx, dp=dp or (dx and index != len(self.transformations)), **kwargs)  # (Npoints, output_dim_t), (Npoints, output_dim_t, input_dim_t), (Npoints, output_dim_t, Nparams_t)
            
            # Append the transformed points and Jacobians to the lists
            jacobian_dx_list.append(jacobian_dx)
            jacobian_dp_list.append(jacobian_dp)

            # Update the points for the next transformation
            points = transformed_points

        # Apply the chain rules to compute the Jacobians with respect to the parameters
        if dp and all(jacobian_dx_list[i] is not None for i in range(len(jacobian_dx_list) - 1)) and all(jacobian_dp_list[i] is not None for i in range(len(jacobian_dp_list))):
            jacobian_dp = numpy.empty((Npoints, self.output_dim, self.Nparams), dtype=numpy.float64)
            start = 0
            # Boucle over the transformations to compute the Jacobian with respect to the parameters
            for index, t in enumerate(self.transformations):
                end = start + t.Nparams
                jacobian_dp_t = jacobian_dp_list[index]  # (Npoints, output_dim_t, Nparams_t)
                for index_2 in range(index + 1, len(self.transformations)):
                    # Compute the chain rule for the Jacobian with respect to the parameters including the dx for the next transformation
                    numpy.matmul(jacobian_dx_list[index_2], jacobian_dp_t, out=jacobian_dp_t)  # (Npoints, output_dim_t2, input_dim_t2) * (Npoints, output_dim_t, Nparams_t) -> (Npoints, output_dim_t2, Nparams_t)
                jacobian_dp[:, :, start:end] = jacobian_dp_t  # (Npoints, output_dim_t, Nparams_t)
                start = end
        else:
            jacobian_dp = None
        
        # Apply the chain rules to compute the Jacobians with respect to the input points
        if dx and all(jacobian_dx_list[i] is not None for i in range(len(jacobian_dx_list))):
            # Compute the Jacobian with respect to the input points
            jacobian_dx = jacobian_dx_list[0] # (Npoints, output_dim_0, input_dim_0)
            for index, t in enumerate(self.transformations[1:]):
                # Apply the chain rule for the Jacobian with respect to the input points
                jacobian_dx = numpy.matmul(jacobian_dx_list[index + 1], jacobian_dx)  # (Npoints, input_dim_t, output_dim_t)
        else:
            jacobian_dx = None

        return (
            points,  # (Npoints, output_dim)
            jacobian_dx,  # (Npoints, output_dim, input_dim) if dx is True, otherwise None
            jacobian_dp  # (Npoints, output_dim, Nparams) if dp is True, otherwise None
        )
    


    def _inverse_transform(
        self,
        points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        **kwargs
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the inverse transformation to the given points.

        This method applies the inverse of each transformation in the composition sequentially to the input points.

        The jacobian matrices with respect to the input points and the parameters are computed using the chain rule.

        Parameters
        ----------
        points : numpy.ndarray
            The input points to be transformed. Shape (Npoints, output_dim).

        dx : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is False.

        **kwargs
            Additional keyword arguments for the transformations.

        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]
            A tuple containing:

            - `transformed_points`: The transformed points of shape (Npoints, input_dim).
            - `jacobian_dx`: The Jacobian matrix with respect to the input points of shape (Npoints, input_dim, output_dim) if `dx` is True, otherwise None.
            - `jacobian_dp`: The Jacobian matrix with respect to the parameters of the transformation of shape (Npoints, input_dim, Nparams) if `dp` is True, otherwise None.
        """
        Npoints = points.shape[0]  # Number of points in computation
        jacobian_dx_list = []
        jacobian_dp_list = []

        # Iterate over each transformation in the composition in reverse order !
        for index, t in enumerate(reversed(self.transformations)):
            # Apply the inverse transformation to the points
            transformed_points, jacobian_dx, jacobian_dp = t._inverse_transform(points, dx=dx, dp=dp or (dx and index != len(self.transformations)), **kwargs) # (Npoints, input_dim_t), (Npoints, input_dim_t, output_dim_t), (Npoints, input_dim_t, Nparams_t)
            # Append the transformed points and Jacobians to the lists
            jacobian_dx_list.append(jacobian_dx)
            jacobian_dp_list.append(jacobian_dp)

            # Update the points for the next transformation
            points = transformed_points

        # Apply the chain rules to compute the Jacobians with respect to the parameters
        if dp and all(jacobian_dx_list[i] is not None for i in range(len(jacobian_dx_list) - 1)) and all(jacobian_dp_list[i] is not None for i in range(len(jacobian_dp_list))):
            jacobian_dp = numpy.empty((Npoints, self.input_dim, self.Nparams), dtype=numpy.float64)

            start = 0
            # Boucle over the transformations to compute the Jacobian with respect to the parameters
            for index, t in enumerate(reversed(self.transformations)):
                end = start + t.Nparams
                jacobian_dp_t = jacobian_dp_list[index]  # (Npoints, input_dim_t, Nparams_t)
                for index_2 in range(index + 1, len(reversed(self.transformations))):
                    # Compute the chain rule for the Jacobian with respect to the parameters
                    numpy.matmul(jacobian_dx_list[index_2], jacobian_dp_t, out=jacobian_dp_t)  # (Npoints, input_dim_t2, output_dim_t2) * (Npoints, input_dim_t, Nparams_t) -> (Npoints, input_dim_t2, Nparams_t)
                jacobian_dp[:, :, start:end] = jacobian_dp_t  # (Npoints, input_dim_t, Nparams_t)
                start = end
        else:
            jacobian_dp = None

        # Apply the chain rules to compute the Jacobians with respect to the input points
        if dx and all(jacobian_dx_list[i] is not None for i in range(len(jacobian_dx_list))):
            # Compute the Jacobian with respect to the input points
            jacobian_dx = jacobian_dx_list[0] # (Npoints, input_dim_-1, output_dim_-1)
            for index, t in enumerate(reversed(self.transformations[:-1])):
                # Apply the chain rule for the Jacobian with respect to the input points
                jacobian_dx = numpy.matmul(jacobian_dx_list[index + 1], jacobian_dx)  # (Npoints, input_dim_t, output_dim_t)
        else:
            jacobian_dx = None

        return (
            points,  # (Npoints, input_dim)
            jacobian_dx,  # (Npoints, input_dim, output_dim) if dx is True, otherwise None
            jacobian_dp  # (Npoints, input_dim, Nparams) if dp is True, otherwise None
        )

              







