from abc import ABC, abstractmethod
from typing import Optional, Tuple, Sequence
from dataclasses import dataclass
import numpy
import copy


@dataclass
class TransformResult:
    r"""
    A class to represent the result of a transformation.

    This class is used to store the results of a transformation, including the transformed points and the Jacobian matrices.

    .. seealso::

        - :class:`pydistort.objects.Transform` for the base class of all transformations.
        - :meth:`pydistort.objects.Transform.transform` for applying the transformation to points.
        - :meth:`pydistort.objects.Transform.inverse_transform` for applying the inverse transformation to points (`output_dim` and `input_dim` are swapped).

    For a transformation from :math:`\mathbb{R}^{input\_dim}` to :math:`\mathbb{R}^{output\_dim}`, the input points are assumed to have shape (..., input_dim) and the output points will have shape (..., output_dim).
    
    .. note::

        If ``transpose`` is set to True during the transformation, the output points will have shape (output_dim, ...) instead of (..., output_dim), same for the Jacobian matrices.

    Attributes
    ----------
    transformed_points : numpy.ndarray
        The transformed points after applying the transformation.
        Shape (..., output_dim).

    jacobian_dx : Optional[numpy.ndarray]
        The Jacobian matrix with respect to the input points.
        Shape (..., output_dim, input_dim).

    jacobian_dp : Optional[numpy.ndarray]
        The Jacobian matrix with respect to the parameters of the transformation.
        Shape (..., output_dim, Nparams).
    """
    transformed_points: numpy.ndarray
    jacobian_dx: Optional[numpy.ndarray] = None
    jacobian_dp: Optional[numpy.ndarray] = None












class Transform(ABC):
    r"""
    Transform is the base class to manage transformations from :math:`\mathbb{R}^{input\_dim}` to :math:`\mathbb{R}^{output\_dim}`.

    In this package, the transformations wiil be used to project 3D-world points onto a 2D-image plane in the stenopic camera model.

    The process to correspond a 3D-world point to a 2D-image point in the stenopic camera model is as follows:

    1. The ``world_3dpoints`` (:math:`X_W`) are expressed in the camera coordinate system using the rotation and translation matrices to obtain the ``camera_3dpoints`` (:math:`X_C`).
    2. The ``camera_3dpoints`` (:math:`X_C`) are normalized by dividing by the third coordinate to obtain the ``normalized_points`` (:math:`x_N`).
    3. The ``normalized_points`` (:math:`x_N`) are distorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``distorted_points`` (:math:`x_D`).
    4. The ``distorted_points`` (:math:`x_D`) are projected onto the image plane using the intrinsic matrix K to obtain the ``image_points`` (:math:`x_I`).

    This tranformation can be decomposed into 3 main steps:

    1. **Extrinsic**: Transform the ``world 3dpoints`` to ``normalized_points`` using the extrinsic parameters (rotation and translation).
    2. **Distortion**: Transform the ``normalized_points`` to ``distorted_points`` using the distortion model.
    3. **Intrinsic**: Transform the ``distorted_points`` to ``image_points`` using the intrinsic matrix K.

    This class provides the base for all transformations in the stenopic camera model. It defines the interface for extrinsic, distortion, and intrinsic transformations.

    Each sub-classes must implement the following methods and properties:

    - `input_dim`: (attribute) The dimension of the input points (should be 2 for 2D points).
    - `output_dim`: (attribute) The dimension of the output points (should be 2 for
    - `parameters` (attribute) The parameters of the transformation, if applicable.
    - `Nparams`: (attribute) The number of parameters for the transformation, if applicable.
    - `_transform`: (method) Apply the transformation to the given points.
    - `is_set`: (method) Check if the transformation is set (i.e., if the parameters are initialized).
    - `_inverse_transform`: (method) Apply the inverse transformation to the given points.


    More details on the transformation methods are provided in the `transform` and `inverse_transform` methods. 

    .. seealso::

        - :meth:`pydistort.objects.Transform.transform` for applying the transformation to points.
        - :meth:`pydistort.objects.Transform.inverse_transform` for applying the inverse transformation to points.
        - :class:`pydistort.objects.TransformResult` for the result of the transformation.

    .. note::

        ``...`` in the shape of the attributes indicates that the shape can have any number of leading dimensions, which is useful for batch processing of points.

    If given, subclasses of ``TransformResult`` should be used to return the results of the transformation and inverse transformation.
    The attributes `result_class` and `inverse_result_class` can be overridden to specify the result classes for the transformation and inverse transformation, respectively.    
    """

    @property
    def result_class(self) -> type:
        r"""
        Property to return the class used for the result of the transformation.
        
        The default is `TransformResult`, but subclasses can override this to return a different class.
        
        Returns
        -------
        type
            The class used for the result of the transformation.
        """
        return TransformResult
    

    @property
    def inverse_result_class(self) -> type:
        r"""
        Property to return the class used for the result of the inverse transformation.
        
        The default is `TransformResult`, but subclasses can override this to return a different class.
        
        Returns
        -------
        type
            The class used for the result of the inverse transformation.
        """
        return TransformResult


    @property
    @abstractmethod
    def input_dim(self) -> int:
        r"""
        Property to return the input dimension of the transformation.
        
        The input dimension must be a positive integer representing the number of dimensions of the input points.
        
        Returns
        -------
        int
            The number of dimensions of the input points.
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        r"""
        Property to return the output dimension of the transformation.
        
        The output dimension must be a positive integer representing the number of dimensions of the output points.
        
        Returns
        -------
        int
            The number of dimensions of the output points.
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> Optional[numpy.ndarray]:
        r"""
        Property to return the parameters of the transformation.
        
        The parameters must be a numpy array of shape (Nparams,) where Nparams is the number of parameters of the transformation.

        If the transformation does not have parameters or they are not set, this property should return None.
        
        Returns
        -------
        Optional[numpy.ndarray]
            The parameters of the transformation.
        """
        pass

    @property
    @abstractmethod
    def Nparams(self) -> int:
        r"""
        Property to return the number of parameters of the transformation.
        
        The number of parameters must be a non-negative integer representing the number of parameters of the transformation.
        
        Returns
        -------
        int
            The number of parameters of the transformation.
        """
        pass

    @abstractmethod
    def is_set(self) -> bool:
        r"""
        Method to check if the transformation parameters are set.
        
        This method should return True if the transformation parameters are initialized, otherwise False.
        
        Returns
        -------
        bool
            True if the transformation parameters are set, otherwise False.
        """
        pass

    def transform(
        self,
        points: numpy.ndarray,
        *,
        transpose: bool = False,
        dx: bool = False,
        dp: bool = False,
        _skip: bool = False,
        **kwargs
    ) -> numpy.ndarray:
        r"""
        The given points ``points`` are assumed to be with shape (..., input_dim) or (input_dim, ...), depending on the value of ``transpose``.

        The output ``transformed_points`` will have shape (..., output_dim) if ``transpose`` is False, or (output_dim, ...) if ``transpose`` is True.

        .. warning::

            The points are converting to float64 before applying the transformation.

        The method also computes 2 Jacobian matrices if requested:

        - ``dx``: Jacobian of the transformed points with respect to the input points.
        - ``dp``: Jacobian of the transformed points with respect to the parameters of the transformation.

        The jacobian matrice with respect to the input points is a (..., output_dim, input_dim) matrix where:

        .. code-block:: python

            jacobian_dx[..., 0, 0]  # ∂x_o/∂x_i -> Jacobian of the coordinates x_o with respect to the coordinates x_i.
            jacobian_dx[..., 0, 1]  # ∂x_o/∂y_i
            ...
            
            jacobian_dx[..., 1, 0]  # ∂y_o/∂x_i -> Jacobian of the coordinates y_o with respect to the coordinates x_i.
            jacobian_dx[..., 1, 1]  # ∂y_o/∂y_i
            ...

        The Jacobian matrice with respect to the parameters is a (..., output_dim, Nparams) matrix where:

        .. code-block:: python

            jacobian_dp[..., 0, 0]  # ∂x_o/∂λ_1 -> Jacobian of the coordinates x_o with respect to the first parameter λ_1.
            jacobian_dp[..., 0, 1]  # ∂x_o/∂λ_2
            ...

            jacobian_dp[..., 1, 0]  # ∂y_o/∂λ_1 -> Jacobian of the coordinates y_o with respect to the first parameter λ_1.
            jacobian_dp[..., 1, 1]  # ∂y_o/∂λ_2
            ...

        The Jacobian matrices are computed only if ``dx`` or ``dp`` are set to True, respectively.

        The output will be a `TransformResult` object containing the transformed points and the Jacobian matrices if requested.

        .. note::

            The _skip parameter is used to skip the checks for the transformation parameters and assume the points are given in the (Npoints, input_dim) float64 format.
            Please use this parameter with caution, as it may lead to unexpected results if the transformation parameters are not set correctly.
        
        Parameters
        ----------
        points : numpy.ndarray
            The input points to be transformed. Shape (..., input_dim) (or (input_dim, ...) if `transpose` is True).

        transpose : bool, optional
            If True, the input points are transposed to shape (input_dim, ...). Default is False.

        dx : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is False.

        _skip : bool, optional
            If True, skip the checks for the transformation parameters and assume the points are given in the (Npoints, input_dim) float64 format.
            `transpose` is ignored if this parameter is set to True.

        **kwargs
            Additional keyword arguments for the transformation.

        Returns
        -------
        TransformResult
            An object containing the transformed points and the Jacobian matrices if requested.

        Developer Notes
        ~~~~~~~~~~~~~~~
        
        The subclasses must implement the `_transform` method to apply the transformation to the input points.
        
        The `_transform` method should:

        - take the input points as a numpy array of shape (Npoints, input_dim)
        - return 3 numpy arrays:

            - `transformed_points`: The transformed points of shape (Npoints, output_dim).
            - `jacobian_dx`: The Jacobian matrix with respect to the input points of shape (Npoints, output_dim, input_dim) if `dx` is True, otherwise None.
            - `jacobian_dp`: The Jacobian matrix with respect to the parameters of the transformation of shape (Npoints, output_dim, Nparams) if `dp` is True, otherwise None.

        """
        if not _skip:
            # Check the boolean flags
            if not isinstance(dx, bool):
                raise TypeError(f"dx must be a boolean, got {type(dx)}")
            if not isinstance(dp, bool):
                raise TypeError(f"dp must be a boolean, got {type(dp)}")
            if not isinstance(transpose, bool):
                raise TypeError(f"transpose must be a boolean, got {type(transpose)}")
            
            # Check if the transformation is set
            if not self.is_set():
                raise ValueError("Transformation parameters are not set. Please set the parameters before transforming points.")
        
            # Convert input points to float64
            points = numpy.asarray(points, dtype=numpy.float64)

            # Check the shape of the input points
            if points.ndim < 2:
                raise ValueError(f"Input points must have at least 2 dimensions, got {points.ndim} dimensions.")

            # Transpose the input points if requested
            if transpose:
                points = numpy.moveaxis(points, 0, -1)  # (input_dim, ...) -> (..., input_dim)

            # Save the shape of the input points
            shape = points.shape # (..., input_dim)

            # Check the last dimension of the input points
            if shape[-1] != self.input_dim:
                raise ValueError(f"Input points must have {self.input_dim} dimensions, got {shape[-1]} dimensions.")

            # Flatten the input points to 2D for processing
            points = points.reshape(-1, self.input_dim) # (..., input_dim) -> (Npoints, input_dim)

        # Apply the transformation
        transformed_points, jacobian_dx, jacobian_dp = self._transform(points, dx=dx, dp=dp, **kwargs) # (Npoints, output_dim), (Npoints, output_dim, input_dim), (Npoints, output_dim, Nparams)

        if not _skip:
            # Reshape the transformed points to the original shape
            transformed_points = transformed_points.reshape(*shape[:-1], self.output_dim)  # (Npoints, output_dim) -> (..., output_dim)
            jacobian_dx = jacobian_dx.reshape(*shape[:-1], self.output_dim, self.input_dim) if jacobian_dx is not None else None  # (Npoints, output_dim, input_dim) -> (..., output_dim, input_dim)
            jacobian_dp = jacobian_dp.reshape(*shape[:-1], self.output_dim, self.Nparams) if jacobian_dp is not None else None # (Npoints, output_dim, Nparams) -> (..., output_dim, Nparams)

            # Transpose the transformed points if requested
            if transpose:
                transformed_points = numpy.moveaxis(transformed_points, -1, 0) # (..., output_dim) -> (output_dim, ...)
                jacobian_dx = numpy.moveaxis(jacobian_dx, -2, 0) if jacobian_dx is not None else None # (..., output_dim, input_dim) -> (output_dim, ..., input_dim)
                jacobian_dp = numpy.moveaxis(jacobian_dp, -2, 0) if jacobian_dp is not None else None # (..., output_dim, Nparams) -> (output_dim, ..., Nparams)

        # Return the result as a TransformResult object
        return self.result_class(
            transformed_points=transformed_points,
            jacobian_dx=jacobian_dx,
            jacobian_dp=jacobian_dp
        )
    


    def inverse_transform(
        self,
        points: numpy.ndarray,
        *,
        transpose: bool = False,
        dx: bool = False,
        dp: bool = False,
        _skip: bool = False,
        **kwargs
    ) -> numpy.ndarray:
        r"""
        The given points ``points`` are assumed to be with shape (..., output_dim) or (output_dim, ...), depending on the value of ``transpose``.

        The output ``transformed_points`` will have shape (..., input_dim) if ``transpose`` is False, or (input_dim, ...) if ``transpose`` is True.

        .. warning::

            The points are converting to float64 before applying the inverse transformation.

        The method also computes 2 Jacobian matrices if requested:

        - ``dx``: Jacobian of the transformed points with respect to the input points.
        - ``dp``: Jacobian of the transformed points with respect to the parameters of the transformation.

        The jacobian matrice with respect to the input points is a (..., input_dim, output_dim) matrix where:

        .. code-block:: python

            jacobian_dx[..., 0, 0]  # ∂x_i/∂x_o -> Jacobian of the coordinates x_i with respect to the coordinates x_o.
            jacobian_dx[..., 0, 1]  # ∂x_i/∂y_o
            ...
            
            jacobian_dx[..., 1, 0]  # ∂y_i/∂x_o -> Jacobian of the coordinates y_i with respect to the coordinates x_o.
            jacobian_dx[..., 1, 1]  # ∂y_i/∂y_o
            ...

        The Jacobian matrice with respect to the parameters is a (..., input_dim, Nparams) matrix where:

        .. code-block:: python

            jacobian_dp[..., 0, 0]  # ∂x_i/∂λ_1 -> Jacobian of the coordinates x_i with respect to the first parameter λ_1.
            jacobian_dp[..., 0, 1]  # ∂x_i/∂λ_2
            ...

            jacobian_dp[..., 1, 0]  # ∂y_i/∂λ_1 -> Jacobian of the coordinates y_i with respect to the first parameter λ_1.
            jacobian_dp[..., 1, 1]  # ∂y_i/∂λ_2
            ...

        The Jacobian matrices are computed only if ``dx`` or ``dp`` are set to True, respectively.

        The output will be a `TransformResult` object containing the transformed points and the Jacobian matrices if requested.

        .. note::

            The _skip parameter is used to skip the checks for the transformation parameters and assume the points are given in the (Npoints, output_dim) float64 format.
            Please use this parameter with caution, as it may lead to unexpected results if the transformation parameters are not set correctly.

        Parameters
        ----------
        points : numpy.ndarray
            The input points to be transformed. Shape (..., output_dim) (or (output_dim, ...) if `transpose` is True).

        transpose : bool, optional
            If True, the input points are transposed to shape (output_dim, ...). Default is False.

        dx : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is False.

        _skip : bool, optional
            If True, skip the checks for the transformation parameters and assume the points are given in the (Npoints, output_dim) float64 format.
            `transpose` is ignored if this parameter is set to True.

        **kwargs
            Additional keyword arguments for the transformation.

        Returns
        -------
        TransformResult
            An object containing the transformed points and the Jacobian matrices if requested.

            
        Developer Notes
        ~~~~~~~~~~~~~~~

        The subclasses must implement the `_inverse_transform` method to apply the inverse transformation to the input points.

        The `_inverse_transform` method should:

        - take the input points as a numpy array of shape (Npoints, output_dim)
        - return 3 numpy arrays:

            - `transformed_points`: The transformed points of shape (Npoints, input_dim).
            - `jacobian_dx`: The Jacobian matrix with respect to the input points of shape (Npoints, input_dim, output_dim) if `dx` is True, otherwise None.
            - `jacobian_dp`: The Jacobian matrix with respect to the parameters of the transformation of shape (Npoints, input_dim, Nparams) if `dp` is True, otherwise None.
        """
        if not _skip:
            # Check the boolean flags
            if not isinstance(dx, bool):
                raise TypeError(f"dx must be a boolean, got {type(dx)}")
            if not isinstance(dp, bool):
                raise TypeError(f"dp must be a boolean, got {type(dp)}")
            if not isinstance(transpose, bool):
                raise TypeError(f"transpose must be a boolean, got {type(transpose)}")
            
            # Check if the transformation is set
            if not self.is_set():
                raise ValueError("Transformation parameters are not set. Please set the parameters before transforming points.")
            
            # Convert input points to float64
            points = numpy.asarray(points, dtype=numpy.float64)

            # Check the shape of the input points
            if points.ndim < 2:
                raise ValueError(f"Input points must have at least 2 dimensions, got {points.ndim} dimensions.")
            
            # Transpose the input points if requested
            if transpose:
                points = numpy.moveaxis(points, 0, -1) # (output_dim, ...) -> (..., output_dim)
            
            # Save the shape of the input points
            shape = points.shape # (..., output_dim)

            # Check the last dimension of the input points
            if shape[-1] != self.output_dim:
                raise ValueError(f"Input points must have {self.output_dim} dimensions, got {shape[-1]} dimensions.")
            
            # Flatten the input points to 2D for processing
            points = points.reshape(-1, self.output_dim) # (..., output_dim)

        # Apply the inverse transformation
        transformed_points, jacobian_dx, jacobian_dp = self._inverse_transform(points, dx=dx, dp=dp, **kwargs) # (Npoints, input_dim), (Npoints, input_dim, output_dim), (Npoints, input_dim, Nparams)

        if not _skip:
            # Reshape the transformed points to the original shape
            transformed_points = transformed_points.reshape(*shape[:-1], self.input_dim)  # (Npoints, input_dim) -> (..., input_dim)
            jacobian_dx = jacobian_dx.reshape(*shape[:-1], self.input_dim, self.output_dim) if jacobian_dx is not None else None  # (..., input_dim, output_dim)
            jacobian_dp = jacobian_dp.reshape(*shape[:-1], self.input_dim, self.Nparams) if jacobian_dp is not None else None # (..., input_dim, Nparams)

            # Transpose the transformed points if requested
            if transpose:
                transformed_points = numpy.moveaxis(transformed_points, -1, 0) # (..., input_dim) -> (input_dim, ...)
                jacobian_dx = numpy.moveaxis(jacobian_dx, -2, 0) if jacobian_dx is not None else None # (..., input_dim, output_dim) -> (input_dim, ..., output_dim)
                jacobian_dp = numpy.moveaxis(jacobian_dp, -2, 0) if jacobian_dp is not None else None # (..., input_dim, Nparams) -> (input_dim, ..., Nparams)

        # Return the result as a InverseTransformResult object
        return self.inverse_result_class(
            transformed_points=transformed_points,
            jacobian_dx=jacobian_dx,
            jacobian_dp=jacobian_dp
        )
    
    @abstractmethod
    def _transform(
        self,
        points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        **kwargs
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the transformation to the given points.

        This method must be implemented by subclasses to apply the transformation to the input points.

        Parameters
        ----------
        points : numpy.ndarray
            The input points to be transformed. Shape (Npoints, input_dim).

        dx : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is False.

        **kwargs
            Additional keyword arguments for the transformation.

        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]
            A tuple containing:

            - `transformed_points`: The transformed points of shape (Npoints, output_dim).
            - `jacobian_dx`: The Jacobian matrix with respect to the input points of shape (Npoints, output_dim, input_dim) if `dx` is True, otherwise None.
            - `jacobian_dp`: The Jacobian matrix with respect to the parameters of the transformation of shape (Npoints, output_dim, Nparams) if `dp` is True, otherwise None.
        """
        raise NotImplementedError("Subclasses must implement the _transform method.")
    

    @abstractmethod
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

        This method must be implemented by subclasses to apply the inverse transformation to the input points.

        Parameters
        ----------
        points : numpy.ndarray
            The input points to be transformed. Shape (Npoints, output_dim).

        dx : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is False.

        **kwargs
            Additional keyword arguments for the transformation.

        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]
            A tuple containing:

            - `transformed_points`: The transformed points of shape (Npoints, input_dim).
            - `jacobian_dx`: The Jacobian matrix with respect to the input points of shape (Npoints, input_dim, output_dim) if `dx` is True, otherwise None.
            - `jacobian_dp`: The Jacobian matrix with respect to the parameters of the transformation of shape (Npoints, input_dim, Nparams) if `dp` is True, otherwise None.
        """
        raise NotImplementedError("Subclasses must implement the _inverse_transform method.")
    

    def optimize_parameters(
        self,
        input_points: numpy.ndarray,
        output_points: numpy.ndarray,
        guess: Optional[numpy.ndarray] = None,
        *,
        transpose: bool = False,
        max_iter: int = 10,
        eps: float = 1e-8,
        verbose: bool = False,
        cond_cutoff: float = 1e5,
        reg_factor: float = 0.0,
        dim_eig_reduce_crit: float = 0.0,
        _skip: bool = False
    ) -> numpy.ndarray:
        r"""
        Optimize the parameters of the transformation using the given input and output points.

        Estimate the optimized parameters of the transformation such that the transformed input points match the output points.

        Lets consider a set of input points :math:`X_I` with shape (..., input_dim) and a set of output points :math:`X_O` with shape (..., output_dim).
        We search :math:`\lambda = \lambda_0 + \delta \lambda` such that:

        .. math::

            X_O = \text{Transform}(X_I, \lambda) = T(X_I, \lambda_0 + \delta \lambda)

        .. note::

            The current parameters of the transformation are not directly modified.
        
        We have:

        .. math::

            \nabla_{\lambda} T (X_I, \lambda_0) \delta \lambda = X_O - T(X_I, \lambda_0)

        The corrections are computed using the following equations:

        .. math::

            J^{T} J \delta \lambda = J^{T} R

        Where :math:`J = \nabla_{\lambda} T (X_I, \lambda_0)` is the Jacobian matrix of the transformation with respect to the parameters, and :math:`R = X_O - T(X_I, \lambda_0)` is the residual vector.

        :math:`\lambda_0` is the initial guess for the parameters, if None, the current parameters of the transformation are used.

        .. note::

            This method can be used to optimize the parameters of any transformation that implements the `_transform` method.

        .. note::

            The ``_skip`` parameter is used to skip the checks for the transformation parameters and assume the input and output points are given in the (Npoints, input_dim) and (Npoints, output_dim) float64 format, respectively.
            Please use this parameter with caution, as it may lead to unexpected results if the transformation parameters are not set correctly.

        For conditioning, the following steps are applied:

        - First, a regularization term is added to the Jacobian matrix to improve stability: :math:`J^{T} J + \text{regfactor} I`.
        - Second, a dimensionality reduction is applied to the Jacobian matrix based on the eigenvalues. Lower eigenvalues :math:`k_i` are removed if :math:`\frac{k_i}{k_0} < \text{dim\_eig\_reduce\_crit}` where :math:`k_0` is the largest eigenvalue.
        
        The `cond_cutoff` parameter is used to detect ill-conditioned problems. If the condition number of the Jacobian matrix is greater than this value, a warning is raised and the optimization returns NaN array.

        Parameters
        ----------
        input_points : numpy.ndarray
            The input points to be transformed. Shape (..., input_dim) (or (input_dim, ...) if `transpose` is True).
        
        output_points : numpy.ndarray
            The output points to be matched. Shape (..., output_dim) (or (output_dim, ...) if `transpose` is True).

        guess : Optional[numpy.ndarray], optional
            The initial guess for the parameters of the transformation with shape (Nparams,). If None, the current parameters of the transformation are used. Default is None.

        transpose : bool, optional
            If True, the input and output points are transposed to shape (input_dim, ...) and (output_dim, ...), respectively. Default is False.

        max_iter : int, optional
            The maximum number of iterations for the optimization. Default is 10.

        eps : float, optional
            The convergence threshold for the optimization. Default is 1e-8.

        verbose : bool, optional
            If True, print the optimization progress and diagnostics. Default is False.

        cond_cutoff : float, optional
            The cutoff value for the condition number of the Jacobian matrix. If the condition number is greater than this value, the optimization will be considered unstable and will raise a warning and return NaN array. This is used to detect ill-conditioned problems. Default is 1e5.

        reg_factor : float, optional
            The regularization factor for the optimization. If greater than 0, it adds a tikhonov regularization term to the optimization problem to improve stability :math:`J^{T} J + \text{regfactor} I`. Default is 0.0.

        dim_eig_reduce_crit : float, optional
            The criterion for reducing the dimensionality of the Jacobian matrix based on the eigenvalues. lower eigenvalues :math:`k_i` are removed if :math:`\frac{k_i}{k_0} < \text{dim_eig_reduce_crit}` where :math:`k_0` is the largest eigenvalue. 
            Default is 0.0, which means no dimensionality reduction is applied. Must be between 0.0 and 1.0.

        _skip : bool, optional
            If True, skip the checks for the transformation parameters and assume the input and output points are given in the (Npoints, input_dim) and (Npoints, output_dim) float64 format, respectively.
            The guess must be given in the (Nparams,) float64 format.
            `transpose` is ignored if this parameter is set to True.

        Returns
        -------
        numpy.ndarray
            The optimized parameters of the transformation with shape (Nparams,).

        Raises
        ------
        ValueError
            If the input and output points do not have the same number of points, or if the input and output dimensions do not match the transformation's input and output dimensions.

        TypeError
            If the input and output points are not numpy arrays, or if the guess is not a numpy array.

        Developer Notes
        ~~~~~~~~~~~~~~~

        The subclasses must implement the `_transform` method to apply the transformation to the input points.
        The `_transform` method should return the transformed points and the Jacobian matrix with respect to the parameters of the transformation.
        """
        if not _skip:
            # Check the boolean flags
            if not isinstance(transpose, bool):
                raise TypeError(f"transpose must be a boolean, got {type(transpose)}")
            if not isinstance(max_iter, int) or max_iter <= 0:
                raise TypeError(f"max_iter must be an integer greater than 0, got {max_iter}")
            if not isinstance(eps, float) or eps <= 0:
                raise TypeError(f"eps must be a positive float, got {eps}")
            if not isinstance(verbose, bool):
                raise TypeError(f"verbose must be a boolean, got {type(verbose)}")
            if not isinstance(cond_cutoff, float) or cond_cutoff <= 0:
                raise TypeError(f"cond_cutoff must be a positive float, got {cond_cutoff}")
            if not isinstance(reg_factor, float) or reg_factor < 0:
                raise TypeError(f"reg_factor must be a non-negative float, got {reg_factor}")
            if not isinstance(dim_eig_reduce_crit, float) or not (0.0 <= dim_eig_reduce_crit <= 1.0):
                raise TypeError(f"dim_eig_reduce_crit must be a float between 0.0 and 1.0, got {dim_eig_reduce_crit}")

            # Check if the transformation is set
            if not self.is_set():
                raise ValueError("Transformation parameters are not set. Please set the parameters before optimizing.")

            # Convert input and output points to float64
            input_points = numpy.asarray(input_points, dtype=numpy.float64)
            output_points = numpy.asarray(output_points, dtype=numpy.float64)

            # Check the shape of the input and output points
            if input_points.ndim < 2 or output_points.ndim < 2:
                raise ValueError(f"Input and output points must have at least 2 dimensions, got {input_points.ndim} and {output_points.ndim} dimensions respectively.")
            
            # Transpose the input and output points if requested
            if transpose:
                input_points = numpy.moveaxis(input_points, 0, -1) # (input_dim, ...) -> (..., input_dim)
                output_points = numpy.moveaxis(output_points, 0, -1) # (output_dim, ...) -> (..., output_dim)

            # Flatten the input and output points to 2D for processing
            input_points = input_points.reshape(-1, self.input_dim)  # (..., input_dim) -> (Npoints, input_dim)
            output_points = output_points.reshape(-1, self.output_dim)  # (..., output_dim) -> (Npoints, output_dim)

            # Check the number of points
            if input_points.shape[0] != output_points.shape[0]:
                raise ValueError(f"Input and output points must have the same number of points, got {input_points.shape[0]} and {output_points.shape[0]} points respectively.")
            
            if input_points.shape[0] == 0:
                raise ValueError("Input and output points must have at least one point.")
            
            # Check the last dimension of the input and output points
            if input_points.shape[-1] != self.input_dim:
                raise ValueError(f"Input points must have {self.input_dim} dimensions, got {input_points.shape[-1]} dimensions.")
            if output_points.shape[-1] != self.output_dim:
                raise ValueError(f"Output points must have {self.output_dim} dimensions, got {output_points.shape[-1]} dimensions.")
            
            # Check the guess
            if guess is not None:
                guess = numpy.asarray(guess, dtype=numpy.float64)
                if guess.ndim != 1:
                    raise ValueError(f"Guess must be a 1D array, got {guess.ndim} dimensions.")
                if guess.shape[0] != self.Nparams:
                    raise ValueError(f"Guess must have {self.Nparams} parameters, got {guess.shape[0]} parameters.")
            
            else:
                # Use the current parameters as the guess
                guess = self.parameters if self.is_set() else numpy.zeros(self.Nparams, dtype=numpy.float64)

        
        # Return empty arrays if Nparams is 0
        if self.Nparams == 0:
            return numpy.zeros(0, dtype=numpy.float64)
        
        # Create a perfect copy of the current class to avoid modifying the original one
        object_class = copy.deepcopy(self)
        Npoints = input_points.shape[0]  # Number of points in computation

        # Set the parameters of the object class to the guess
        object_class.parameters = guess
        delta_itk = numpy.zeros_like(object_class.parameters, dtype=numpy.float64)

        # Run the iterative algorithm
        for it in range(max_iter):

            #============================================
            # Transformation and Jacobian computation
            #============================================

            # Compute the transformed points and the Jacobian with respect to the parameters
            transformed_points_itk, _, jacobian_dp = object_class._transform(input_points, dx=False, dp=True)  # shape (Npoints, output_dim), None, (Npoints, output_dim, Nparams)

            # Check if the jacobian_dp is None$
            if jacobian_dp is None:
                raise ValueError("Jacobian with respect to the parameters is not available. Please implement the _transform method to return the Jacobian with respect to the parameters.")

            # Check the convergence of the optimization
            diff = numpy.linalg.norm(transformed_points_itk - output_points, axis=1)  # shape (Npoints,)
            if verbose:
                print(f"Iteration {it}: |X_O - X_I| - Max difference: {numpy.nanmax(diff)}, Mean difference: {numpy.nanmean(diff)}")

            if numpy.all(diff[~numpy.isnan(diff)] < eps):
                if verbose:
                    print(f"Optimization converged in {it} iterations.")
                break
            
            #===================================================
            # Create the residual vector and Jacobian matrix
            #===================================================

            if verbose:
                print("\n#=====================================================")
                print(f"STARTING ITERATION {it+1} OF THE OPTIMIZATION PROCESS")
                print("#=====================================================\n")

            # Construct the residual vector R and the Jacobian J
            R = output_points - transformed_points_itk  # shape (Npoints, output_dim)
            J = jacobian_dp  # shape (Npoints, output_dim, Nparams)

            # Create masks to filter out invalid points
            mask_R = numpy.isfinite(R).all(axis=1)  # Create a mask for finite values in R
            mask_J = numpy.isfinite(J).all(axis=(1, 2))  # Create a mask for finite values in each row of J
            mask = mask_R & mask_J  # Combine the masks to filter out invalid points

            if verbose:
                print(f"Iteration {it+1}: {numpy.sum(mask)} valid points out of {Npoints}.")

            # Apply the masks to R_flat and J_flat
            R = R[mask, :]  # shape (Nvalid_points, output_dim)
            J = J[mask, :, :]  # shape (Nvalid_points, output_dim, Nparams)

            # Flatten the residual vector and Jacobian matrix
            R_flat = R.flatten()  # Flatten the residual vector to shape (Npoints * output_dim,)
            J_flat = J.reshape(Npoints * self.output_dim, -1)  # Flatten the Jacobian to shape (Npoints * output_dim, Nparams)

            # Compute the delta using the normal equations: J^T J delta = J^T R
            JTJ = numpy.dot(J_flat.T, J_flat)  # shape (Nparams, Nparams)
            JTR = numpy.dot(J_flat.T, R_flat)  # shape (Nparams,)


            #===================================================
            # Regularization and Dimensionality Reduction
            #===================================================

            # Add regularization if requested
            if reg_factor > 0.0:
                JTJ += reg_factor * numpy.eye(self.Nparams, dtype=numpy.float64)

            # Reducing the dimensionality of the Jacobian based on eigenvalues
            if dim_eig_reduce_crit > 0.0 or verbose:
                # Compute the eigenvalues and eigenvectors of JTJ
                eigvals, eigvecs = numpy.linalg.eig(JTJ)

                if verbose:
                    print(f"Iteration {it+1}: Eigenvalues before reduction:\n{eigvals}\n")

                if dim_eig_reduce_crit > 0.0:
                    # Reduce the dimensionality based on the criterion
                    eigmask = numpy.abs(eigvals / numpy.max(eigvals)) >= dim_eig_reduce_crit  # Keep eigenvalues that satisfy the criterion

                    if verbose:
                        print(f"Iteration {it+1}: Number of eigenvalues kept: {numpy.sum(eigmask)} out of {self.Nparams}")

                    # Apply the mask to the eigenvalues (set the small eigenvalues to zero)
                    eigvals[~eigmask] = 0.0

                    # Reconstruct the reduced JTJ matrix
                    JTJ = eigvecs @ numpy.diag(eigvals) @ eigvecs.T  # shape (Nparams, Nparams)


            # ===================================================
            # Condition number check
            # ===================================================

            # Condition number check
            cond_number = numpy.linalg.cond(JTJ)

            if verbose:
                print(f"Iteration {it+1}: Condition number of JTJ: {cond_number}")

            if cond_number > cond_cutoff:
                print(f"Warning: Condition number {cond_number} exceeds cutoff {cond_cutoff}. Optimization may be unstable. skipping iteration {it+1} and returning NaN array.")
                return numpy.full(self.Nparams, numpy.nan, dtype=numpy.float64)
            
            # ====================================================
            # Solve the linear system to find the delta
            # ====================================================

            # Solve the linear system to find the delta
            delta_itk = numpy.linalg.solve(JTJ, JTR) # shape (Nparams,)

            if verbose:
                print(f"Iteration {it+1}: Delta parameters:\n{delta_itk}\n")
                
            # Update the parameters of the object class
            object_class.parameters = object_class.parameters + delta_itk  # shape (Nparams,)

            if verbose:
                print(f"Iteration {it+1}: Updated parameters:\n{object_class.parameters}\n")
        
        return object_class.parameters  # shape (Nparams,)
    



    def optimize_input_points(
        self,
        output_points: numpy.ndarray,
        guess: Optional[numpy.ndarray] = None,
        *,
        transpose: bool = False,
        max_iter: int = 10,
        eps: float = 1e-8,
        verbose: bool = False,
        _skip: bool = False
    ) -> numpy.ndarray:
        r"""
        Optimize the input points of the transformation using the given output points.

        Estimate the optimized input points of the transformation such that the transformed input points match the output points.

        .. warning::

            This method can only be used if the dimensions are the same, i.e. input_dim == dim.

        Lets consider a set of output points :math:`X_O` with shape (..., dim) and a set of input points :math:`X_I` with shape (..., input_dim).
        We search :math:`X_I = X_{I0} + \delta X_I` such that:

        .. math::

            X_O = \text{Transform}(X_I, \lambda) = T(X_I + \delta X_I, \lambda)
        
        We have:

        .. math::

            \nabla_{X} T (X_I, \lambda_0) \delta \lambda = X_O - T(X_I, \lambda_0)

        The corrections are computed using the following equations:

        .. math::

            J \delta X_I = R

        Where :math:`J = \nabla_{X} T (X_I, \lambda_0)` is the Jacobian matrix of the transformation with respect to the input points, and :math:`R = X_O - T(X_I, \lambda_0)` is the residual vector.

        :math:`X_{I0}` is the initial guess for the input points, if None, it use the output points as the initial guess.

        .. note::

            The ``_skip`` parameter is used to skip the checks for the transformation parameters and assume the output points are given in the (Npoints, dim) float64 format.
            Please use this parameter with caution, as it may lead to unexpected results if the transformation parameters are not set correctly.

        Parameters
        ----------
        output_points : numpy.ndarray
            The output points to be matched. Shape (..., dim) (or (dim, ...) if `transpose` is True).

        guess : Optional[numpy.ndarray], optional
            The initial guess for the input points of the transformation with shape (..., dim). If None, the output points are used as the initial guess. Default is None.

        transpose : bool, optional
            If True, the output points are transposed to shape (dim, ...). Default is False.

        max_iter : int, optional
            The maximum number of iterations for the optimization. Default is 10.

        eps : float, optional
            The convergence threshold for the optimization. Default is 1e-8.

        verbose : bool, optional
            If True, print the optimization progress and diagnostics. Default is False.

        _skip : bool, optional
            If True, skip the checks for the transformation parameters and assume the output points are given in the (Npoints, dim) float64 format.
            The guess must be given in the (Npoints, dim) float64 format.
            `transpose` is ignored if this parameter is set to True.

        Returns
        -------
        numpy.ndarray
            The optimized input points of the transformation with shape (..., dim).

        Raises
        ------
        ValueError
            If the output points do not have the expected shape, or if the input and output dimensions do not match the transformation's input and output dimensions.

        TypeError
            If the output points or guess are not numpy arrays, or if the guess is not a numpy array.

        Developer Notes
        ~~~~~~~~~~~~~~~

        The subclasses must implement the `_transform` method to apply the transformation to the input points.
        The `_transform` method should return the transformed points and the Jacobian matrix with respect to the input points.

        """
        if self.input_dim != self.output_dim:
            raise ValueError(f"Input dimension ({self.input_dim}) must be equal to output dimension ({self.output_dim}) for this method to work.")
        dim = self.input_dim  # Since input_dim == output_dim

        if not _skip:
            # Check the boolean flags
            if not isinstance(transpose, bool):
                raise TypeError(f"transpose must be a boolean, got {type(transpose)}")
            if not isinstance(max_iter, int) or max_iter <= 0:
                raise TypeError(f"max_iter must be an integer greater than 0, got {max_iter}")
            if not isinstance(eps, float) or eps <= 0:
                raise TypeError(f"eps must be a positive float, got {eps}")
            if not isinstance(verbose, bool):
                raise TypeError(f"verbose must be a boolean, got {type(verbose)}")

            # Check if the transformation is set
            if not self.is_set():
                raise ValueError("Transformation parameters are not set. Please set the parameters before optimizing.")

            # Convert output points to float64
            output_points = numpy.asarray(output_points, dtype=numpy.float64)

            # Check the guess
            if guess is not None:
                guess = numpy.asarray(guess, dtype=numpy.float64)
            else:
                # Use the output points as the initial guess
                guess = numpy.zeros((output_points.shape[0], dim), dtype=numpy.float64)

            # Check the shape of the output points
            if output_points.ndim < 2:
                raise ValueError(f"Output points must have at least 2 dimensions, got {output_points.ndim} dimensions.")
            if guess.ndim < 2:
                    raise ValueError(f"Guess must have at least 2 dimensions, got {guess.ndim} dimensions.")
            
            # Transpose the output points if requested
            if transpose:
                output_points = numpy.moveaxis(output_points, 0, -1) # (dim, ...) -> (..., dim)
                guess = numpy.moveaxis(guess, 0, -1) # (dim, ...) -> (..., dim)

            # Flatten the output points to 2D for processing
            shape = output_points.shape  # (..., dim)
            output_points = output_points.reshape(-1, dim)  # (..., dim) -> (Npoints, dim)
            guess = guess.reshape(-1, dim)  # (..., dim) -> (Npoints, dim)
            
            # Check the number of points
            if output_points.shape[0] != guess.shape[0]:
                raise ValueError(f"Output points and guess must have the same number of points, got {output_points.shape[0]} and {guess.shape[0]} points respectively.")
            if output_points.shape[0] == 0:
                raise ValueError("Output points and guess must have at least one point.")

            if output_points.shape[-1] != dim:
                raise ValueError(f"Output points must have {dim} dimensions, got {output_points.shape[-1]} dimensions.")
            if guess.shape[-1] != dim:
                raise ValueError(f"Guess must have {dim} dimensions, got {guess.shape[-1]} dimensions.")
            
        # Initialize the guess for the input points
        Npoints = output_points.shape[0]
        delta_itk = numpy.zeros_like(guess, dtype=numpy.float64) # shape (Npoints, dim) (Delta for the next iteration)
        Nopt = Npoints # Number of points in computation

        # Prepare the output array:
        input_points = guess

        # Create the mask for the points in computation
        mask = numpy.logical_and(numpy.isfinite(output_points).all(axis=1), numpy.isfinite(input_points).all(axis=1))  # shape (Npoints,)

        # Run the iterative algorithm
        for it in range(max_iter):
            # Compute the transformation of the input points and the Jacobian with respect to the input points
            output_points_itk, jacobian_dx, _ = self._transform(input_points[mask, :], dx=True, dp=False) # shape (Nopt, dim), (Nopt, dim, dim), None

            # Check if the jacobian_dx is None
            if jacobian_dx is None:
                raise ValueError("Jacobian with respect to the input points is not available. Please implement the _transform method to return the Jacobian with respect to the input points.")
            
            # Check the convergence of the optimization
            diff = numpy.linalg.norm(output_points_itk - output_points[mask, :], axis=1)  # shape (Nopt,)
            eps_mask = diff > eps # shape (Nopt,)
            mask[mask] = numpy.logical_and(mask[mask], eps_mask)

            if numpy.sum(mask) == 0:
                if verbose:
                    print(f"Optimization converged in {it} iterations.")
                break

            Nopt = numpy.sum(mask)  # Update the number of points in computation

            output_points_itk = output_points_itk[eps_mask, :]  # shape (NewNopt, dim)
            jacobian_dx = jacobian_dx[eps_mask, :, :]  # shape (NewNopt, dim, dim)

            # Construct the residual vector R and the Jacobian J
            R = output_points[mask, :] - output_points_itk # shape (Nopt, dim)
            J = jacobian_dx # shape (Nopt, dim, dim)

            # Solve the linear system to find the delta
            delta_itk = numpy.array([numpy.linalg.solve(J[i], R[i]) for i in range(Nopt)], dtype=numpy.float64) # shape (Nopt, dim)

            # Update the input points
            input_points[mask, :] = input_points[mask, :] + delta_itk
            if verbose:
                print(f"Iteration {it+1}: {numpy.sum(mask)} valid points out of {Npoints}. Max delta: {numpy.max(numpy.abs(delta_itk))}")

        # Return the optimized input points
        if not _skip:
            input_points = input_points.reshape(*shape[:-1], dim)  # (Npoints, dim) -> (..., dim)

            if transpose:
                input_points = numpy.moveaxis(input_points, -1, 0) # (..., dim) -> (dim, ...)

        return input_points


            



















class TransformComposition(Transform):
    r"""
    A class to represent the composition of multiple transformations.

    This class allows to compose multiple transformations and apply them sequentially.
    It inherits from the `Transform` class and overrides the `_transform` and `_inverse_transform` methods to apply the composition of transformations.

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
                for index_2, t_2 in enumerate(self.transformations[index + 1:]):
                    # Compute the chain rule for the Jacobian with respect to the parameters
                    jacobian_dp_t = numpy.einsum("nij, njk -> nik", jacobian_dx_list[index_2], jacobian_dp_t)  # (Npoints, output_dim_t, Nparams_t)
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
                jacobian_dx = numpy.einsum("nij, njk -> nik", jacobian_dx_list[index + 1], jacobian_dx)  # (Npoints, output_dim_t, input_dim_t)
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
                for index_2, t_2 in enumerate(reversed(self.transformations[index + 1:])):
                    # Compute the chain rule for the Jacobian with respect to the parameters
                    jacobian_dp_t = numpy.einsum("nij, njk -> nik", jacobian_dx_list[index_2], jacobian_dp_t)
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
                jacobian_dx = numpy.einsum("nij, njk -> nik", jacobian_dx_list[index], jacobian_dx)
        else:
            jacobian_dx = None

        return (
            points,  # (Npoints, input_dim)
            jacobian_dx,  # (Npoints, input_dim, output_dim) if dx is True, otherwise None
            jacobian_dp  # (Npoints, input_dim, Nparams) if dp is True, otherwise None
        )

              