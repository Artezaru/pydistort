from .transform import Transform
from typing import Tuple, Optional
import numpy

from .transform import Transform

class TransformInversion(Transform):
    r"""
    A class to represent the inversion of a transformation.

    It inherits from the :class:`pydistort.core.Transform` class and overrides the `_transform` and `_inverse_transform` methods to apply the inverse transformation.

    Lets consider a transformation `T` that maps points from an input space :math:`X_I` to an output space :math:`X_O`:

    .. math::

        X_{O} = T(X_{I}) \quad X_{I} = T^{-1}(X_{O})

    The inverse transformation `G = T^{-1}` maps points from the output space :math:`X_O` back to the input space :math:`X_I`:

    .. math::

        X_{I} = G(X_{O}) \quad X_{O} = G^{-1}(X_{I}) = T(X_{I})

    This class is used to invert a given transformation, allowing the user to apply the inverse transformation to points in the output space to retrieve their corresponding points in the input space.
    
    Parameters
    ----------
    transform : Transform
        The transformation to be inverted. Must be an instance of the `Transform` class or its subclasses.

    """
    def __init__(self, transform: Transform):
        super().__init__()
        if not isinstance(transform, Transform):
            raise TypeError(f"transform must be an instance of Transform, got {type(transform)}")
        self.transform = transform

    @property
    def input_dim(self) -> int:
        r"""
        The input dimension is the output dimension of the transformation to be inverted.
        """
        return self.transform.output_dim
    
    @property
    def output_dim(self) -> int:
        r"""
        The output dimension is the input dimension of the transformation to be inverted.
        """
        return self.transform.input_dim
    
    @property
    def Nparams(self) -> int:
        r"""
        The number of parameters is the same as the number of parameters of the transformation to be inverted.
        """
        return self.transform.Nparams
    
    @property
    def parameters(self) -> numpy.ndarray:
        r"""
        The parameters are the same as the parameters of the transformation to be inverted.
        """
        return self.transform.parameters
    
    @parameters.setter
    def parameters(self, value: numpy.ndarray):
        self.transform.parameters = value

    def is_set(self) -> bool:
        r"""
        Check if the parameters of the transformation to be inverted are set.
        """
        return self.transform.is_set()
    
    def _transform(
            self,
            points: numpy.ndarray,
            *,
            dx: bool = False,
            dp: bool = False,
            **kwargs
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        The transform method is the inverse of the transformation to be inverted.
        """
        return self.transform._inverse_transform(points, dx=dx, dp=dp, **kwargs)
    

    def _inverse_transform(
            self,
            points: numpy.ndarray,
            *,
            dx: bool = False,
            dp: bool = False,
            **kwargs
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        The inverse transform method is the direct transformation of the transformation to be inverted.
        """
        return self.transform._transform(points, dx=dx, dp=dp, **kwargs)