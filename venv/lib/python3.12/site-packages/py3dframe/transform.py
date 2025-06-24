import numpy 
import scipy
from .frame import Frame
from typing import Union, Optional

from .switch_RT_convention import switch_RT_convention
from .rotation import Rotation
from .matrix.is_SO3 import is_SO3

class Transform:
    r"""
    Lets consider two orthonormal reference frames :math:`E` and :math:`F` of :math:`\mathbb{R}^3` (see :class:`py3dframe.Frame`).
    The transformation from the frame E (input frame) to the frame F (output frame) can be stored in a Transform object.

    Lets consider a point :math:`X` whose coordinates in the frame E are :math:`\mathbf{X}_i` and in the frame F are :math:`\mathbf{X}_o`.
    There exist 8 principal conventions to express the transformation between the frame E and the frame F.

    The 8 conventions are summarized as follows:

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

    Because the frames are orthonormal, the matrix :math:`\mathbf{R}` is an orthogonal matrix, i.e. :math:`\mathbf{R}^T = \mathbf{R}^{-1}`.

    .. seealso::

        - function :func:`py3dframe.switch_RT_convention` to convert the transformation between the frames E and F from one convention to another.

    Parameters
    ----------
    input_frame : Optional[Frame], optional
        The input frame of the transformation. Default is None - the global frame.
    
    output_frame : Optional[Frame], optional
        The output frame of the transformation. Default is None - the global frame.
    
    dynamic : bool, optional
        If True, the transformation will be affected by the changes in the input frame or the output frame. Default is True.
    
    convention : Union[int, str], optional
        The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is 0.

    Raises  
    ------
    TypeError
        If the input_frame or the output_frame is not a Frame object (or None).
        If the rotation_matrix is not an array_like object (or None).
        If the translation is not an array_like object (or None).
        If the dynamic is not a boolean.
        If the convention is not an integer or a string.
    ValueError
        If the rotation_matrix is not a 3x3 matrix.
        If the translation_vector is not a 3-element vector.
        If the convention is not between 0 and 7 or a valid string.
        If the user provides both the rotation_matrix and the translation_vector and the input_frame and the output_frame.
    

    Examples
    --------

    To create a Transform object, the user must provide two Frame objects.

    .. code-block:: python

        from py3dframe import Frame, Transform

        frame_E = Frame() # Global frame
        frame_F = Frame(origin=[1, 2, 3], x=[1, 0, 0], y=[0, 1, 0], z=[0, 0, 1]) # Local frame
        transform = Transform(input_frame=frame_E, output_frame=frame_F, dynamic=True, convention=0)

    - If the ``dynamic`` parameter is set to ``True``, the Transform object will be affected by the changes in the input frame or the output frame.
    - If the ``dynamic`` parameter is set to ``False``, the Transform object will correspond to the transformation between the input frame and the output frame at the time of the creation of the Transform object.
    - If the ``dynamic`` parameter is set to ``True`` and then changed to ``False``, the Transform object will correspond to the transformation between the input frame and the output frame at the time of the change of the ``dynamic`` parameter.
    
    The user can access the rotation matrix and the translation vector of the transformation as follows:

    .. code-block:: python

        R = transform.rotation_matrix
        T = transform.translation

    The user can also access the input frame and the output frame of the transformation as follows:

    .. code-block:: python

        frame_E = transform.input_frame
        frame_F = transform.output_frame

    The Transform object can be used to transform points or vectors from the input frame to the output frame and vice versa.

    .. code-block:: python

        X_i = [1, 2, 3]
        X_o = transform.transform(point=X_i)
        X_i = transform.inverse_transform(point=X_o)

    For vectors, the translation vector is not taken into account.

    .. code-block:: python

        X_i = [1, 2, 3]
        X_o = transform.transform(point=X_i) # X_i = R X_o + T
        V_i = [1, 2, 3]
        V_o = transform.transform(vector=V_i) # V_i = R V_o
    """
    def __init__(
            self,
            *,
            input_frame: Optional[Frame] = None,
            output_frame: Optional[Frame] = None,
            dynamic: bool = True,
            convention: int = 0,
        ):
        if input_frame is None:
            input_frame = Frame()
        if output_frame is None:
            output_frame = Frame()
            
        self.input_frame = input_frame
        self.output_frame = output_frame
        self.convention = convention
        self.dynamic = dynamic

    
    # ====================================================================================================================================
    # Developer methods
    # ====================================================================================================================================

    @property
    def _R_dev(self) -> scipy.spatial.transform.Rotation:
        """
        Getter and setter for the rotation object between the parent frame and the frame in the convention 0.

        The rotation is a scipy.spatial.transform.Rotation object. 

        Returns
        -------
        scipy.spatial.transform.Rotation
            The rotation between the parent frame and the frame in the convention 0.
        """
        if self.dynamic:
            # ====================================================================================================================================
            # Lets note : 
            # Xg : the coordinates of a point in the global frame
            # X1 : the coordinates of the same point in the frame 1
            # X2 : the coordinates of the same point in the frame 2
            # R1 : the rotation matrix from the global frame to the frame 1
            # R2 : the rotation matrix from the global frame to the frame 2
            # T1 : the translation vector from the global frame to the frame 1
            # T2 : the translation vector from the global frame to the frame 2
            # R : the rotation matrix from the frame 1 to the frame 2
            # T : the translation vector from the frame 1 to the frame 2
            # 
            # We have :
            # Xg = R1 * X1 + T1
            # Xg = R2 * X2 + T2
            # X1 = R * X2 + T
            #
            # We search R:
            # X1 = R1.inv() * (Xg - T1)
            # X1 = R1.inv() * (R2 * X2 + T2 - T1)
            # X1 = R1.inv() * R2 * X2 + R1.inv() * (T2 - T1)
            # So R = R1.inv() * R2
            # ====================================================================================================================================
            R_input = self.input_frame.get_global_rotation(convention=0)
            R_output = self.output_frame.get_global_rotation(convention=0)
            return R_input.inv() * R_output
        
        return self.__R_dev
    
    @_R_dev.setter
    def _R_dev(self, R: scipy.spatial.transform.Rotation) -> None:
        if not isinstance(R, scipy.spatial.transform.Rotation):
            raise TypeError("The rotation must be a scipy.spatial.transform.Rotation object.")
        self.__R_dev = R
    
    @property
    def _T_dev(self) -> numpy.ndarray:
        """
        Getter and setter for the translation vector between the parent frame and the frame in the convention 0.

        The translation vector is a 3-element vector.

        .. warning::

            The T_dev attribute is flags.writeable = False. To change the translation vector, use the setter.

        Returns
        -------
        numpy.ndarray
            The translation vector between the parent frame and the frame in the convention 0 with shape (3, 1).
        """
        if self.dynamic:
            # ====================================================================================================================================
            # Lets note : 
            # Xg : the coordinates of a point in the global frame
            # X1 : the coordinates of the same point in the frame 1
            # X2 : the coordinates of the same point in the frame 2
            # R1 : the rotation matrix from the global frame to the frame 1
            # R2 : the rotation matrix from the global frame to the frame 2
            # T1 : the translation vector from the global frame to the frame 1
            # T2 : the translation vector from the global frame to the frame 2
            # R : the rotation matrix from the frame 1 to the frame 2
            # T : the translation vector from the frame 1 to the frame 2
            # 
            # We have :
            # Xg = R1 * X1 + T1
            # Xg = R2 * X2 + T2
            # X1 = R * X2 + T
            #
            # We search T:
            # X1 = R1.inv() * (Xg - T1)
            # X1 = R1.inv() * (R2 * X2 + T2 - T1)
            # X1 = R1.inv() * R2 * X2 + R1.inv() * (T2 - T1)
            # So T = R1.inv() * (T2 - T1)
            # ====================================================================================================================================
            R_input = self.input_frame.get_global_rotation(convention=0)
            T_input = self.input_frame.get_global_translation(convention=0)
            T_output = self.output_frame.get_global_translation(convention=0)
            return R_input.inv().apply((T_output - T_input).T).T
        
        T_dev = self.__T_dev.copy()
        T_dev.flags.writeable = True
        return T_dev
    
    @_T_dev.setter
    def _T_dev(self, T: numpy.ndarray) -> None:
        T = numpy.array(T).reshape((3,1)).astype(numpy.float64)
        self.__T_dev = T
        self.__T_dev.flags.writeable = False



    # ====================================================================================================================================
    # Public methods
    # ====================================================================================================================================

    @property
    def input_frame(self) -> Frame:
        """
        Getter and setter for the input_frame attribute.

        The input_frame attribute is a Frame object.

        .. warning::

            dynamic attribute must be set to True to take into account the changes in the input frame.

        Returns
        -------
        Frame
            The input frame of the transformation.
        """
        return self._input_frame
    
    @input_frame.setter
    def input_frame(self, input_frame: Optional[Frame]) -> None:
        if input_frame is None:
            input_frame = Frame()
        if not isinstance(input_frame, Frame):
            raise TypeError("The input_frame must be a Frame object.")
        self._input_frame = input_frame



    @property
    def output_frame(self) -> Frame:
        """
        Getter and setter for the output_frame attribute.

        The output_frame attribute is a Frame object.

        .. warning::

            dynamic attribute must be set to True to take into account the changes in the output frame.

        Returns
        -------
        Frame
            The output frame of the transformation.
        """
        return self._output_frame
    
    @output_frame.setter
    def output_frame(self, output_frame: Optional[Frame]) -> None:
        if output_frame is None:
            output_frame = Frame()
        if not isinstance(output_frame, Frame):
            raise TypeError("The output_frame must be a Frame object.")
        self._output_frame = output_frame



    @property
    def dynamic(self) -> bool:
        """
        Getter and setter for the dynamic attribute.

        The dynamic attribute is a boolean. If True, the transformation will be affected by the changes in the input frame or the output frame.

        Returns
        -------
        bool
            If True, the transformation will be affected by the changes in the input frame or the output frame.
        """
        return self._dynamic

    @dynamic.setter
    def dynamic(self, dynamic: bool) -> None:
        if not isinstance(dynamic, bool):
            raise TypeError("The dynamic must be a boolean.")
        if dynamic:
            self.__R_dev = None
            self.__T_dev = None
        else:
            self.__R_dev = self._R_dev
            self.__T_dev = self._T_dev
        self._dynamic = dynamic



    @property
    def convention(self) -> int:
        """
        Getter and setter for the convention parameter.

        Returns
        -------
        int
            The convention parameter.
        """
        return self._convention
    
    @convention.setter
    def convention(self, convention: int) -> None:
        if not isinstance(convention, int):
            raise TypeError("The convention parameter must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        self._convention = convention


    
    def get_rotation(self, *, convention: Optional[int] = None) -> scipy.spatial.transform.Rotation:
        """
        Get the rotation between the input frame and the output frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        Returns
        -------
        Rotation
            The rotation between the input frame and the output frame in the given convention.
        """
        if convention is None:
            convention = self._convention
        if not isinstance(convention, int):
            raise TypeError("The convention must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        R, _ = switch_RT_convention(self._R_dev, self._T_dev, 0, convention)
        return R

    @property
    def rotation(self) -> scipy.spatial.transform.Rotation:
        """
        Getter for the rotation between the input frame and the output frame in the convention of the frame.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_rotation` to get the rotation in a specific convention.

        Returns
        -------
        Rotation
            The rotation between the input frame and the output frame in the convention of the frame.
        """
        return self.get_rotation(convention=self._convention)
    


    def get_translation(self, *, convention: Optional[int] = None) -> numpy.ndarray:
        """
        Get the translation vector between the input frame and the output frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        Returns
        -------
        numpy.ndarray
            The translation vector between the input frame and the output frame in the given convention with shape (3, 1).
        """
        if convention is None:
            convention = self._convention
        if not isinstance(convention, int):
            raise TypeError("The convention must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        _, T = switch_RT_convention(self._R_dev, self._T_dev, 0, convention)
        return T

    @property
    def translation(self) -> numpy.ndarray:
        """
        Getter for the translation vector between the input frame and the output frame in the convention of the frame.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_translation` to get the translation vector in a specific convention.

        Returns
        -------
        numpy.ndarray
            The translation vector between the input frame and the output frame in the convention of the frame with shape (3, 1).
        """
        return self.get_translation(convention=self._convention)
    


    def get_rotation_matrix(self, *, convention: Optional[int] = None) -> numpy.ndarray:
        """
        Get the rotation matrix between the input frame and the output frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        Returns
        -------
        numpy.ndarray
            The rotation matrix between the input frame and the output frame in the given convention with shape (3, 3).
        """
        return self.get_rotation(convention=convention).as_matrix()

    @property
    def rotation_matrix(self) -> numpy.ndarray:
        """
        Getter for the rotation matrix between the input frame and the output frame in the convention of the frame.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_rotation_matrix` to get the rotation matrix in a specific convention.

        Returns
        -------
        numpy.ndarray
            The rotation matrix between the input frame and the output frame in the convention of the frame with shape (3, 3).
        """
        return self.get_rotation_matrix(convention=self._convention)
    


    def get_quaternion(self, *, convention: Optional[int] = None, scalar_first: bool = True) -> numpy.ndarray:
        """
        Get the quaternion between the input frame and the output frame in the given convention.
        
        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        scalar_first : bool, optional
            If True, the quaternion will be in the scalar-first convention. Default is True.

        Returns
        -------
        numpy.ndarray
            The quaternion between the input frame and the output frame in the given convention with shape (4,).
        """
        if not isinstance(scalar_first, bool):
            raise TypeError("The scalar_first must be a boolean.")
        return self.get_rotation(convention=convention).as_quat(scalar_first=scalar_first)
    
    @property
    def quaternion(self) -> numpy.ndarray:
        """
        Getter for the quaternion between the input frame and the output frame in the convention of the frame.
        The quaternion is in the scalar-first convention.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_quaternion` to get the quaternion in a specific convention.

        Returns
        -------
        numpy.ndarray
            The quaternion between the input frame and the output frame in the convention of the frame with shape (4,) in the scalar-first convention.
        """
        return self.get_quaternion(convention=self._convention, scalar_first=True)



    def get_euler_angles(self, *, convention: Optional[int] = None, seq: str = 'xyz', degrees: bool = False) -> numpy.ndarray:
        """
        Get the Euler angles between the input frame and the output frame in the given convention.
        
        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        seq : str, optional
            The sequence of the Euler angles. Default is 'xyz'.

        degrees : bool, optional
            If True, the Euler angles will be in degrees. Default is False.

        Returns
        -------
        numpy.ndarray
            The Euler angles between the input frame and the output frame in the given convention with shape (3,).
        """
        if not isinstance(seq, str):
            raise TypeError("The seq must be a string.")
        if not isinstance(degrees, bool):
            raise TypeError("The degrees must be a boolean.")
        if not len(seq) == 3:
            raise ValueError("The seq must be a string of length 3.")
        if not all([s in 'XYZxyz' for s in seq]):
            raise ValueError("The seq must contain only the characters 'X', 'Y', 'Z', 'x', 'y', 'z'.") 
        return self.get_rotation(convention=convention).as_euler(seq, degrees=degrees)
    
    @property
    def euler_angles(self) -> numpy.ndarray:
        """
        Getter for the Euler angles between the input frame and the output frame in the convention of the frame.
        The Euler angles are in the 'xyz' sequence and in radians.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_euler_angles` to get the Euler angles in a specific convention.

        Returns
        -------
        numpy.ndarray
            The Euler angles between the input frame and the output frame in the convention of the frame with shape (3,) in radians.
        """
        return self.get_euler_angles(convention=self._convention, seq='xyz', degrees=False)
    


    def get_rotation_vector(self, *, convention: Optional[int] = None, degrees: bool = False) -> numpy.ndarray:
        """
        Get the rotation vector between the input frame and the output frame in the given convention.
        
        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        degrees : bool, optional
            If True, the rotation vector will be in degrees. Default is False.

        Returns
        -------
        numpy.ndarray
            The rotation vector between the input frame and the output frame in the given convention with shape (3,).
        """
        if not isinstance(degrees, bool):
            raise TypeError("The degrees must be a boolean.")
        return self.get_rotation(convention=convention).as_rotvec(degrees=degrees)

    @property
    def rotation_vector(self) -> numpy.ndarray:
        """
        Getter for the rotation vector between the input frame and the output frame in the convention of the frame.
        The rotation vector is in radians.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_rotation_vector` to get the rotation vector in a specific convention.

        Returns
        -------
        numpy.ndarray
            The rotation vector between the input frame and the output frame in the convention of the frame with shape (3,) in radians.
        """
        return self.get_rotation_vector(convention=self._convention, degrees=False)

    

    def transform(self, *, point: Optional[numpy.ndarray] = None, vector: Optional[numpy.ndarray] = None) -> numpy.ndarray:
        r"""
        Transform a point or a vector from the input frame to the output frame.

        If the point is provided, the method will return the coordinates of the point in the output frame.
        If the vector is provided, the method will return the coordinates of the vector in the output frame.

        Several points / vectors can be transformed at the same time by providing a 2D numpy array with shape (3, N).

        If both the point and the vector are provided, the method will raise a ValueError.
        If neither the point nor the vector is provided, the method will return None.

        In the convention 0:

        .. math::

            X_{\text{output_frame}} = R^{-1} * (X_{\text{input_frame}} - T)

        .. math::

            V_{\text{output_frame}} = R^{-1} * V_{\text{input_frame}}

        Parameters
        ----------
        point : Optional[array_like], optional
            The coordinates of the point in the input frame with shape (3, 1). Default is None.
        
        vector : Optional[array_like], optional
            The coordinates of the vector in the input frame with shape (3, 1). Default is None.

        Returns
        -------
        numpy.ndarray
            The coordinates of the point or the vector in the output frame with shape (3, 1).

        Raises
        ------
        ValueError
            If the point or the vector is not provided.
        """
        if point is not None and vector is not None:
            raise ValueError("Only one of 'point' or 'vector' can be provided.")
        if point is None and vector is None:
            return None
        
        input_data = point if point is not None else vector
        input_data = numpy.array(input_data).reshape((3, -1)).astype(numpy.float64)

        # Convert the point to vector
        if point is not None:
            input_data = input_data - self._T_dev
        
        # Convert the input data to the output frame
        output_data = self._R_dev.inv().apply(input_data.T).T

        return output_data
    


    def inverse_transform(self, *, point: Optional[numpy.ndarray] = None, vector: Optional[numpy.ndarray] = None) -> numpy.ndarray:
        r"""
        Transform a point or a vector from the output frame to the input frame.

        If the point is provided, the method will return the coordinates of the point in the input frame.
        If the vector is provided, the method will return the coordinates of the vector in the input frame.

        Several points / vectors can be transformed at the same time by providing a 2D numpy array with shape (3, N).

        If both the point and the vector are provided, the method will raise a ValueError.
        If neither the point nor the vector is provided, the method will return None.

        In the convention 0:

        .. math::

            X_{\text{input_frame}} = R * X_{\text{output_frame}} + T

        .. math::

            V_{\text{input_frame}} = R * V_{\text{output_frame}}

        Parameters
        ----------
        point : Optional[array_like], optional
            The coordinates of the point in the output frame with shape (3, 1). Default is None.
        
        vector : Optional[array_like], optional
            The coordinates of the vector in the output frame with shape (3, 1). Default is None.

        Returns
        -------
        numpy.ndarray
            The coordinates of the point or the vector in the input frame with shape (3, 1).

        Raises
        ------
        ValueError
            If the point or the vector is not provided.
        """
        if point is not None and vector is not None:
            raise ValueError("Only one of 'point' or 'vector' can be provided.")
        if point is None and vector is None:
            return None
        
        output_data = point if point is not None else vector
        output_data = numpy.array(output_data).reshape((3, -1)).astype(numpy.float64)

        # Convert the output data to vector input
        input_data = self._R_dev.apply(output_data.T).T

        # Convert the vector to point
        if point is not None:
            input_data = input_data + self._T_dev

        return input_data