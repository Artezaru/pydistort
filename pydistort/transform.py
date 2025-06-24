from abc import ABC, abstractmethod
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy
import cv2



@dataclass
class TransformResult:
    r"""
    A class to represent the result of a transformation.

    This class is used to store the results of a transformation, including the transformed points and the Jacobian matrices.

    .. note::

        ``...`` in the shape of the attributes indicates that the shape can have any number of leading dimensions, which is useful for batch processing of points.

    .. warning::

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


@dataclass
class InverseTransformResult:
    r"""
    A class to represent the result of an inverse transformation.

    This class is used to store the results of an inverse transformation, including the transformed points and the Jacobian matrices.

    .. note::

        ``...`` in the shape of the attributes indicates that the shape can have any number of leading dimensions, which is useful for batch processing of points.

    .. warning::

        If ``transpose`` is set to True during the inverse transformation, the output points will have shape (input_dim, ...) instead of (..., input_dim), same for the Jacobian matrices.

    Attributes
    ----------
    transformed_points : numpy.ndarray
        The transformed points after applying the inverse transformation.
        Shape (..., input_dim).

    jacobian_dx : Optional[numpy.ndarray]
        The Jacobian matrix with respect to the input points.
        Shape (..., input_dim, output_dim).

    jacobian_dp : Optional[numpy.ndarray]
        The Jacobian matrix with respect to the parameters of the transformation.
        Shape (..., input_dim, Nparams).
    """
    transformed_points: numpy.ndarray
    jacobian_dx: Optional[numpy.ndarray] = None
    jacobian_dp: Optional[numpy.ndarray] = None


class Transform(ABC):
    r"""
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
    - `Nparams`: (attribute) The number of parameters for the transformation, if applicable.
    - `_transform`: (method) Apply the transformation to the given points.
    - `_inverse_transform`: (method) Apply the inverse transformation to the given points.
    - `is_set`: (method) Check if the transformation is set (i.e., if the parameters are initialized).

    More details on the transformation methods are provided in the `transform` and `inverse_transform` methods. 

    .. seealso::

        - :meth:`pydistort.Transform.transform` for applying the transformation to points.
        - :meth:`pydistort.Transform.inverse_transform` for applying the inverse transformation to points.
        - :class:`pydistort.TransformResult` for the result of the transformation.
        - :class:`pydistort.InverseTransformResult` for the result of the inverse transformation.

    .. note::

        ``...`` in the shape of the attributes indicates that the shape can have any number of leading dimensions, which is useful for batch processing of points.

    If given, subclasses of ``TransformResult`` and ``InverseTransformResult`` should be used to return the results of the transformation and inverse transformation, respectively.
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
        
        The default is `InverseTransformResult`, but subclasses can override this to return a different class.
        
        Returns
        -------
        type
            The class used for the result of the inverse transformation.
        """
        return InverseTransformResult


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
        points_flat = points.reshape(-1, self.input_dim) # (..., input_dim) -> (Npoints, input_dim)

        # Apply the transformation
        transformed_points, jacobian_dx, jacobian_dp = self._transform(points_flat, dx=dx, dp=dp, **kwargs) # (Npoints, output_dim), (Npoints, output_dim, input_dim), (Npoints, output_dim, Nparams)

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

        The output will be a `InverseTransformResult` object containing the transformed points and the Jacobian matrices if requested.

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

        **kwargs
            Additional keyword arguments for the transformation.

        Returns
        -------
        InverseTransformResult
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
        points_flat = points.reshape(-1, self.output_dim) # (..., output_dim)

        # Apply the inverse transformation
        transformed_points, jacobian_dx, jacobian_dp = self._inverse_transform(points_flat, dx=dx, dp=dp, **kwargs) # (Npoints, input_dim), (Npoints, input_dim, output_dim), (Npoints, input_dim, Nparams)

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
        dx: bool = True,
        dp: bool = True,
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
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is True.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is True.

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
        dx: bool = True,
        dp: bool = True,
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
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is True.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is True.

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
    



