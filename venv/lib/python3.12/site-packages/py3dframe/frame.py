from __future__ import annotations
import numpy 
import scipy
import matplotlib.pyplot
import matplotlib.axes
from typing import Optional, Tuple, Dict, Any, Union, Sequence
import json

from .switch_RT_convention import switch_RT_convention
from .matrix import is_SO3
from .rotation import Rotation

class Frame:
    r"""
    A Frame object represents a frame of reference of :math:`\mathbb{R}^3` (only orthogonal and right-handed frames are supported).

    A frame of reference is defined by its origin and its basis vectors in the global frame coordinates.
    The origin is a 3-element vector. 
    The basis vectors can be given in 3 different 3-element vectors or in a 3x3 matrix with the basis vectors as columns.

    .. code-block:: python

        import numpy
        from py3dframe import Frame
        e1 = [1, 0, 0]
        e2 = [0, 1, 0]
        e3 = [0, 0, 1]
        origin = [1, 2, 3]
        frame = Frame(origin=origin, x_axis=e1, y_axis=e2, z_axis=e3)
        frame = Frame(origin=origin, axes=numpy.column_stack((e1, e2, e3)))

    Sometimes the user wants to define a frame relatively to another frame.
    For example, if we consider a person in a train, the person is defined in the train frame and the train frame is defined in the global frame.
    When the parent frame changes, the child frame changes too but the transformation between the frames remains the same.
    In this case, the user must provide the parent frame (``parent`` parameter) and the origin and the basis vectors of the frame in the parent frame coordinates.

    An other way to define a frame is to use the transformation between the parent frame and the frame.
    Lets note :math:`E` the parent frame (or the global frame) and :math:`F` the frame to define.
    The transformation between the frame :math:`E` and the frame :math:`F` is defined by a rotation matrix :math:`R` and a translation vector :math:`T`.
    Several conventions can be used to define the transformation between the frames.

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

    .. note::

        If a parent is given to the frame, the frame will be defined relative to the parent frame.
        A change in the parent frame will affect the frame but the transformation between the frames will remain the same.
        If you want use the parent frame only to define the frame and not to link the frames, set the parameter ``setup_only`` to True (the parent attribute will be seted to None after the frame is created).

    .. note::

        If the axes are provided, the class will normalize the basis vectors to get an orthonormal matrix.    

    .. seealso::

        - class :class:`py3dframe.Transform` to represent a transformation between two frames of reference and convert points between the frames.
        - function :func:`py3dframe.matrix.O3_project` to project a 3x3 matrix to the orthogonal group O(3) before using it as a rotation matrix.

    Parameters
    ----------
    origin : array_like, optional
        The coordinates of the origin of the frame in the parent frame coordinates. Default is the zero vector.
    
    axes : array_like, optional
        The basis vectors of the frame in the parent frame coordinates. Default is the identity matrix.

    x_axis : array_like, optional
        The x-axis of the frame in the parent frame coordinates. Default is the x-axis of the parent frame.
    
    y_axis : array_like, optional
        The y-axis of the frame in the parent frame coordinates. Default is the y-axis of the parent frame.
    
    z_axis : array_like, optional
        The z-axis of the frame in the parent frame coordinates. Default is the z-axis of the parent frame.

    translation : array_like, optional
        The translation vector in the convention used to define the frame. Default is the zero vector.
        The translation vector is a 3-element vector.

    rotation: scipy.spatial.transform.Rotation, optional
        The rotation in the convention used to define the frame. Default is the identity rotation.

    rotation_matrix : array_like, optional
        The rotation matrix in the convention used to define the frame. Default is the identity matrix.
        The rotation matrix is a 3x3 matrix.

    quaternion : array_like, optional
        The quaternion in the convention used to define the frame. Default is the identity quaternion.
        The quaternion is a 4-element vector [w, x, y, z] (scalar first convention).

    euler_angles : array_like, optional
        The euler angles in the convention used to define the frame. Default is the zero vector.
        The euler angles are a 3-element vector with xyz convention in radians.

    rotation_vector : array_like, optional
        The rotation vector in the convention used to define the frame. Default is the zero vector.
        The rotation vector is a 3-element vector with the angle in radians.

    parent : Optional[Frame], optional
        The parent frame of the frame. Default is None - the global frame.

    setup_only : bool, optional
        If True, the parent frame will be used only to define the frame and not to link the frames. Default is False.

    convention : int, optional
        The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is 0.
        This parameter is used only if the axes are not defined by the ``axes`` parameter.
    
    Raises
    ------
    TypeError
        If any of the parameters is wrong type.
    ValueError
        If the origin is not a 3-element vector.
        If the axes is not a 3x3 matrix.
        If the parent is not a Frame object.
        If axes is not an orthogonal matrix.
        If several parameters are provided to define the frame.
    
    Examples
    --------

    Lets :math:`E = (O_E, \mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3)` be a frame of reference of :math:`\mathbb{R}^3`.
    To create the frame :math:`E`, the user must provide the origin and the basis vectors of the frame.

    .. code-block:: python

        from py3dframe import Frame

        origin = [1, 2, 3]
        x_axis = [1, 1, 0]
        y_axis = [1, -1, 0]
        z_axis = [0, 0, 1]
        frame_E = Frame(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, parent=None)

    A frame can also be defined in another frame of reference (named the parent frame).

    Lets consider a frame :math:`F = (O_F, \mathbf{f}_1, \mathbf{f}_2, \mathbf{f}_3)` defined in the frame :math:`E`.
    The user must provide the origin and the basis vectors of the frame :math:`F` in the frame :math:`E`.

    .. code-block:: python

        from py3dframe import Frame

        origin = [1, -2, 3]
        x_axis = [1, 0, 1]
        y_axis = [0, 1, 0]
        z_axis = [-1, 0, 1]
        frame_F = Frame(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, parent=frame_E)
    
    In this case the frame :math:`F` is defined relative to the frame :math:`E`.
    A change in the frame :math:`E` will affect the frame :math:`F` but the transformation between the frames will remain the same.

    The user can access the origin and the basis vectors of the frame as follows:

    .. code-block:: python

        # In the global frame
        frame_F.global_origin 
        frame_F.global_x_axis
        frame_F.global_y_axis
        frame_F.gloabl_z_axis
    
        # In the parent frame coordinates and relative to the parent frame
        frame_F.origin
        frame_F.x_axis
        frame_F.y_axis
        frame_F.z_axis

    To finish, the user can define a frame using the transformation between the parent frame and the frame.
    Using the convention 0, the rotation matrix and the translation vector will exactly be the basis vectors and the origin of the frame.

    .. code-block:: python

        from py3dframe import Frame

        translation = [1, 2, 3]
        rotation_matrix = np.array([[1, 1, 0], [1, -1, 0], [0, 0, 1]]).T # Equivalent to the column_stack((x_axis, y_axis, z_axis))
        frame_E = Frame(translation=translation, rotation_matrix=rotation_matrix, parent=None, convention=0)

    """

    def __init__(
            self,
            *,
            origin: Optional[numpy.ndarray] = None,
            axes: Optional[numpy.ndarray] = None,
            x_axis: Optional[numpy.ndarray] = None,
            y_axis: Optional[numpy.ndarray] = None,
            z_axis: Optional[numpy.ndarray] = None,
            translation: Optional[numpy.ndarray] = None,
            rotation: Optional[scipy.spatial.transform.Rotation] = None,
            rotation_matrix: Optional[numpy.ndarray] = None,
            quaternion: Optional[numpy.ndarray] = None,
            euler_angles: Optional[numpy.ndarray] = None,
            rotation_vector: Optional[numpy.ndarray] = None,
            parent: Optional[Frame] = None,
            setup_only: bool = False,
            convention: Optional[int] = 0
        ) -> None:

        # ==============================================================================================================
        # First we check the parameters and the consistency of the user input.
        # The user must use only one way to define the frame.
        # ==============================================================================================================

        # Check the parameters
        if rotation is not None and not isinstance(rotation, Rotation):
            raise TypeError("The rotation must be a Rotation object.")
        if parent is not None and not isinstance(parent, Frame):
            raise TypeError("The parent must be a Frame object.")
        if not isinstance(setup_only, bool):
            raise TypeError("The setup_only parameter must be a boolean.")
        if not isinstance(convention, int):
            raise TypeError("The convention parameter must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")

        # Get the method to define the frame.
        define_by_origin_and_axes = origin is not None or x_axis is not None or y_axis is not None or z_axis is not None or axes is not None
        define_by_translation_and_rotation = translation is not None or rotation is not None or rotation_matrix is not None or quaternion is not None or euler_angles is not None or rotation_vector is not None
        
        # Check if the parameters are consistent
        if define_by_origin_and_axes and define_by_translation_and_rotation:
            raise ValueError("If the origin or axes is provided, the other parameters mustn't be provided. The frame must be defined by [the origin and the axes] or by [the translation vector and the rotation].")
        
        if define_by_origin_and_axes:
            # If axes is provided, x-axis, y-axis, and z-axis mustn't be provided.
            if axes is not None and (x_axis is not None or y_axis is not None or z_axis is not None):
                raise ValueError("If the axes is provided, the x_axis, y_axis, and z_axis mustn't be provided.")
            
        if define_by_translation_and_rotation:
            # Only one of the rotation parameters must be provided.
            if sum([rotation is not None, rotation_matrix is not None, quaternion is not None, euler_angles is not None, rotation_vector is not None]) > 1:
                raise ValueError("Only one of the rotation parameters must be provided.")

        # Default values if no orientation is provided.
        if not define_by_origin_and_axes and not define_by_translation_and_rotation:
            define_by_origin_and_axes = True

        # ==============================================================================================================
        # The frame will only conserve the transformation between the parent frame and the frame in the convention 0.
        # First we define the translation and the rotation according to the user input.
        # ==============================================================================================================

        # Prepare the translation and rotation parameters if define_by_translation_and_rotation is True.
        if define_by_origin_and_axes:
            # Define the translation vector.
            if origin is None:
                origin = [0, 0, 0]
            translation = numpy.array(origin).reshape((3,)).astype(numpy.float64)

            # Define the rotation matrix.
            if axes is None:
                if x_axis is None:
                    x_axis = [1, 0, 0]
                x_axis = numpy.array(x_axis).reshape((3,1)).astype(numpy.float64)
                if y_axis is None:
                    y_axis = [0, 1, 0]
                y_axis = numpy.array(y_axis).reshape((3,1)).astype(numpy.float64)
                if z_axis is None:
                    z_axis = [0, 0, 1]
                z_axis = numpy.array(z_axis).reshape((3,1)).astype(numpy.float64)
                axes = numpy.column_stack((x_axis, y_axis, z_axis)).astype(numpy.float64)
            rotation_matrix = numpy.array(axes).reshape((3,3)).astype(numpy.float64)  
            rotation_matrix = rotation_matrix / numpy.linalg.norm(rotation_matrix, axis=0) # Normalize the columns to get an orthonormal matrix
            if not is_SO3(rotation_matrix):
                raise ValueError("The axes must be a special orthogonal matrix.")

            # Create the rotation object.
            rotation = Rotation.from_matrix(rotation_matrix)
            convention = 0 # The convention is 0 because the rotation matrix is the basis vectors of the frame.
        

        # Prepare the translation and rotation parameters if define_by_translation_and_rotation is True.          
        if define_by_translation_and_rotation:
            if translation is None:
                translation = [0, 0, 0]
            translation = numpy.array(translation).reshape((3,1)).astype(numpy.float64)

            if rotation_matrix is not None:
                rotation_matrix = numpy.array(rotation_matrix).reshape((3,3)).astype(numpy.float64)
                norm = numpy.linalg.norm(rotation_matrix, axis=0)
                if numpy.any(norm == 0):
                    raise ValueError("The basis vectors can't be 0.")
                rotation_matrix = rotation_matrix / norm
                if not is_SO3(rotation_matrix):
                    raise ValueError("The rotation matrix must be a special orthogonal matrix.")
                rotation = Rotation.from_matrix(rotation_matrix)
            
            if quaternion is not None:
                quaternion = numpy.array(quaternion).reshape((4,)).astype(numpy.float64)
                norm = numpy.linalg.norm(quaternion)
                if norm == 0:
                    raise ValueError("The quaternion can't be 0.")
                quaternion = quaternion / norm
                rotation = Rotation.from_quat(quaternion, scalar_first=True)
            
            if euler_angles is not None:
                euler_angles = numpy.array(euler_angles).reshape((3,)).astype(numpy.float64)
                rotation = Rotation.from_euler("XYZ", euler_angles, degrees=False)

            if rotation_vector is not None:
                rotation_vector = numpy.array(rotation_vector).reshape((3,)).astype(numpy.float64)
                rotation = Rotation.from_rotvec(rotation_vector, degrees=False)

            if rotation is None:
                rotation = Rotation.from_matrix(numpy.eye(3))

        # ====================================================================================================================================
        # At this point, the translation vector and the rotation object are defined and convention parameters are seted.
        # Lets convert the translation and the rotation to the convention 0 (default convention for the developer).
        # ====================================================================================================================================
        rotation, translation = switch_RT_convention(rotation, translation, convention, 0)

        # ====================================================================================================================================
        # In the user give a parent and the setup_only parameter is True, 
        # we must construct the composite transformation between the global frame and the parent frame.
        # ====================================================================================================================================
        if parent is not None and setup_only:
            # Construct the composite transformation between the global frame and the parent frame.
            R_parent = parent.get_global_rotation(convention=0)
            T_parent = parent.get_global_translation(convention=0)

            # ====================================================================================================================================
            # Lets note : 
            # Xg : the coordinates of a point in the global frame
            # Xp : the coordinates of the same point in the parent frame
            # Xf : the coordinates of the same point in the frame
            # Rg : the rotation matrix between the global frame and the parent frame
            # Rp : the rotation matrix between the parent frame and the frame
            # Tg : the translation vector between the global frame and the parent frame
            # Tp : the translation vector between the parent frame and the frame
            # R : the rotation matrix between the global frame and the frame
            # T : the translation vector between the global frame and the frame
            # 
            # We have :
            # Xg = Rg * Xp + Tg
            # Xp = Rp * Xf + Tp
            # Xg = R * Xf + T
            #
            # We search R and T:
            # Xg = Rg * (Rp * Xf + Tp) + Tg
            # Xg = Rg * Rp * Xf + Rg * Tp + Tg
            #
            # So the composite transformation is R = Rg * Rp and T = Rg * Tp + Tg
            #
            # However, the rotation is right-handed or left-handed depending on the determinant of the rotation matrix.
            # We have R = r * k where k is the matrix [[1, 0, 0], [0, 1, 0], [0, 0, det(R)]] 
            # and r is the equivalent direct rotation (corresponding to the equivalent direct frame with the Z-axis inverted).
            #
            # So T = rg * (kg * Tp) + Tg
            #
            # And R = rg * kp * rp * kp
            # Can we have R = r * k ?
            # Not directly if det(Rg) = - 1 -> So we need to use the rotation matrix of the Rotation object. :( Thats sad ...
            # For now , lets remove the right-handed parameter from the class.
            # ====================================================================================================================================
            rotation = R_parent * rotation
            translation = R_parent.apply(translation.T).T + T_parent
            parent = None

        # ====================================================================================================================================
        # At this point we can store the useful parameters of the frame.
        # ====================================================================================================================================
        self._R_dev = rotation
        self._T_dev = translation
        self._parent = parent
        self._convention = convention


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
        T_dev = self.__T_dev.copy()
        T_dev.flags.writeable = True
        return T_dev
    
    @_T_dev.setter
    def _T_dev(self, T: numpy.ndarray) -> None:
        T = numpy.array(T).reshape((3,1)).astype(numpy.float64)
        self.__T_dev = T
        self.__T_dev.flags.writeable = False



    # ====================================================================================================================================
    # User methods
    # ====================================================================================================================================

    @property
    def parent(self) -> Optional[Frame]:
        """
        Getter and setter for the parent frame of the frame.

        Returns
        -------
        Optional[Frame]
            The parent frame of the frame.
        """
        return self._parent
    
    @parent.setter
    def parent(self, parent: Optional[Frame]) -> None:
        if parent is not None and not isinstance(parent, Frame):
            raise TypeError("The parent must be a Frame object.")
        self._parent = parent
    


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
    


    @property
    def origin(self) -> numpy.ndarray:
        """
        Getter and setter for the origin of the frame in the parent frame coordinates.

        Returns
        -------
        numpy.ndarray
            The origin of the frame in the parent frame coordinates with shape (3, 1).
        """
        return self.get_translation(convention=0)
    
    @origin.setter
    def origin(self, origin: numpy.ndarray) -> None:
        self.set_translation(origin, convention=0)
    


    @property
    def axes(self) -> numpy.ndarray:
        """
        Getter and setter for the basis vectors of the frame in the parent frame coordinates.

        Returns
        -------
        numpy.ndarray
            The basis vectors of the frame in the parent frame coordinates with shape (3, 3).
        """
        return self.get_rotation_matrix(convention=0)
    
    @axes.setter
    def axes(self, axes: numpy.ndarray) -> None:
        axes = numpy.array(axes).reshape((3,3)).astype(numpy.float64)
        norm = numpy.linalg.norm(axes, axis=0)
        if numpy.any(norm == 0):
            raise ValueError("The basis vectors must be linearly independent.")
        axes = axes / norm
        self.set_rotation_matrix(axes, convention=0)
    


    @property
    def x_axis(self) -> numpy.ndarray:
        """
        Getter for the x-axis of the frame in the parent frame coordinates.

        .. warning::

            The x_axis attributes can't be changed. To change the x-axis, use the axes attribute.

        Returns
        -------
        numpy.ndarray
            The x-axis of the frame in the parent frame coordinates with shape (3, 1).
        """
        x_axis = self.axes[:,0].reshape((3,1))
        return x_axis
    


    @property
    def y_axis(self) -> numpy.ndarray:
        """
        Getter for the y-axis of the frame in the parent frame coordinates.

        .. warning::

            The y_axis attributes can't be changed. To change the y-axis, use the axes attribute.

        Returns
        -------
        numpy.ndarray
            The y-axis of the frame in the parent frame coordinates with shape (3, 1).
        """
        y_axis = self.axes[:,1].reshape((3,1))
        return y_axis
    


    @property
    def z_axis(self) -> numpy.ndarray:
        """
        Getter for the z-axis of the frame in the parent frame coordinates.

        .. warning::

            The z_axis attributes can't be changed. To change the z-axis, use the axes attribute.

        Returns
        -------
        numpy.ndarray
            The z-axis of the frame in the parent frame coordinates with shape (3, 1).
        """
        z_axis = self.axes[:,2].reshape((3,1))
        return z_axis



    def get_rotation(self, *, convention: Optional[int] = None) -> scipy.spatial.transform.Rotation:
        """
        Get the rotation between the parent frame and the frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        Returns
        -------
        Rotation
            The rotation between the parent frame and the frame in the given convention.
        """
        if convention is None:
            convention = self._convention
        if not isinstance(convention, int):
            raise TypeError("The convention must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        R, _ = switch_RT_convention(self._R_dev, self._T_dev, 0, convention)
        return R
    
    def set_rotation(self, rotation: Rotation, *, convention: Optional[int] = None) -> None:
        """
        Set the rotation between the parent frame and the frame in the given convention.

        Parameters
        ----------
        rotation : Rotation
            The rotation between the parent frame and the frame in the given convention.
        
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.
        """
        if convention is None:
            convention = self._convention
        if not isinstance(convention, int):
            raise TypeError("The convention must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        if not isinstance(rotation, Rotation):
            raise TypeError("The rotation must be a Rotation object.")
        _, current_T = switch_RT_convention(self._R_dev, self._T_dev, 0, convention)
        self._R_dev, self._T_dev = switch_RT_convention(rotation, current_T, convention, 0)

    @property
    def rotation(self) -> scipy.spatial.transform.Rotation:
        """
        Getter and setter for the rotation between the parent frame and the frame in the convention of the frame.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_rotation` to get the rotation in a specific convention.

        Returns
        -------
        Rotation
            The rotation between the parent frame and the frame in the convention of the frame.
        """
        return self.get_rotation(convention=self._convention)
    
    @rotation.setter
    def rotation(self, rotation: scipy.spatial.transform.Rotation) -> None:
        self.set_rotation(rotation, convention=self._convention)
    


    def get_translation(self, *, convention: Optional[int] = None) -> numpy.ndarray:
        """
        Get the translation vector between the parent frame and the frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        Returns
        -------
        numpy.ndarray
            The translation vector between the parent frame and the frame in the given convention with shape (3, 1).
        """
        if convention is None:
            convention = self._convention
        if not isinstance(convention, int):
            raise TypeError("The convention must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        _, T = switch_RT_convention(self._R_dev, self._T_dev, 0, convention)
        return T
    
    def set_translation(self, translation: numpy.ndarray, *, convention: Optional[int] = None) -> None:
        """
        Set the translation vector between the parent frame and the frame in the given convention.

        Parameters
        ----------
        translation : numpy.ndarray
            The translation vector between the parent frame and the frame in the given convention with shape (3, 1).
        
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.
        """
        if convention is None:
            convention = self._convention
        if not isinstance(convention, int):
            raise TypeError("The convention must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        translation = numpy.array(translation).reshape((3,1)).astype(numpy.float64)
        current_R, _ = switch_RT_convention(self._R_dev, self._T_dev, 0, convention)
        self._R_dev, self._T_dev = switch_RT_convention(current_R, translation, convention, 0)

    @property
    def translation(self) -> numpy.ndarray:
        """
        Getter and setter for the translation vector between the parent frame and the frame in the convention of the frame.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_translation` to get the translation vector in a specific convention.

        Returns
        -------
        numpy.ndarray
            The translation vector between the parent frame and the frame in the convention of the frame with shape (3, 1).
        """
        return self.get_translation(convention=self._convention)
    
    @translation.setter
    def translation(self, translation: numpy.ndarray) -> None:
        self.set_translation(translation, convention=self._convention)
    


    def get_rotation_matrix(self, *, convention: Optional[int] = None) -> numpy.ndarray:
        """
        Get the rotation matrix between the parent frame and the frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        Returns
        -------
        numpy.ndarray
            The rotation matrix between the parent frame and the frame in the given convention with shape (3, 3).
        """
        return self.get_rotation(convention=convention).as_matrix()
    
    def set_rotation_matrix(self, rotation_matrix: numpy.ndarray, *, convention: Optional[int] = None) -> None:
        """
        Set the rotation matrix between the parent frame and the frame in the given convention.

        Parameters
        ----------
        rotation_matrix : numpy.ndarray
            The rotation matrix between the parent frame and the frame in the given convention with shape (3, 3).
        
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.
        """
        if convention is None:
            convention = self._convention
        if not isinstance(convention, int):
            raise TypeError("The convention must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        rotation_matrix = numpy.array(rotation_matrix).reshape((3,3)).astype(numpy.float64)
        if not is_SO3(rotation_matrix):
            raise ValueError("The rotation matrix must be a special orthogonal matrix.")
        R = Rotation.from_matrix(rotation_matrix)
        self.set_rotation(R, convention=convention)

    @property
    def rotation_matrix(self) -> numpy.ndarray:
        """
        Getter and setter for the rotation matrix between the parent frame and the frame in the convention of the frame.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_rotation_matrix` to get the rotation matrix in a specific convention.

        Returns
        -------
        numpy.ndarray
            The rotation matrix between the parent frame and the frame in the convention of the frame with shape (3, 3).
        """
        return self.get_rotation_matrix(convention=self._convention)

    @rotation_matrix.setter
    def rotation_matrix(self, rotation_matrix: numpy.ndarray) -> None:
        self.set_rotation_matrix(rotation_matrix, convention=self._convention)



    def get_quaternion(self, *, convention: Optional[int] = None, scalar_first: bool = True) -> numpy.ndarray:
        """
        Get the quaternion between the parent frame and the frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        scalar_first : bool, optional
            If True, the quaternion is returned in the scalar first convention. Default is True.
        
        Returns
        -------
        numpy.ndarray
            The quaternion between the parent frame and the frame in the given convention with shape (4,).
        """
        return self.get_rotation(convention=convention).as_quat(scalar_first=scalar_first)
    
    def set_quaternion(self, quaternion: numpy.ndarray, *, convention: Optional[int] = None, scalar_first: bool = True) -> None:
        """
        Set the quaternion between the parent frame and the frame in the given convention.

        Parameters
        ----------
        quaternion : numpy.ndarray
            The quaternion between the parent frame and the frame in the given convention with shape (4,).
        
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.
        
        scalar_first : bool, optional
            If True, the quaternion is in the scalar first convention. Default is True.
        """
        if convention is None:
            convention = self._convention
        if not isinstance(convention, int):
            raise TypeError("The convention must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        if not isinstance(scalar_first, bool):
            raise TypeError("The scalar_first parameter must be a boolean.")
        quaternion = numpy.array(quaternion).reshape((4,)).astype(numpy.float64)
        norm = numpy.linalg.norm(quaternion)
        if norm == 0:
            raise ValueError("The quaternion can't be 0.")
        quaternion = quaternion / norm
        R = Rotation.from_quat(quaternion, scalar_first=scalar_first)
        self.set_rotation(R, convention=convention)
    
    @property
    def quaternion(self) -> numpy.ndarray:
        """
        Getter and setter for the quaternion between the parent frame and the frame in the convention of the frame.
        The quaternion is in the scalar first convention.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_quaternion` to get the quaternion in a specific convention.

        Returns
        -------
        numpy.ndarray
            The quaternion between the parent frame and the frame in the convention of the frame with shape (4,).
        """
        return self.get_quaternion(convention=self._convention, scalar_first=True)
    
    @quaternion.setter
    def quaternion(self, quaternion: numpy.ndarray) -> None:
        self.set_quaternion(quaternion, convention=self._convention, scalar_first=True)

    

    def get_euler_angles(self, *, convention: Optional[int] = None, degrees: bool = False, seq: str = "xyz") -> numpy.ndarray:
        """
        Get the Euler angles between the parent frame and the frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        degrees : bool, optional
            If True, the Euler angles are returned in degrees. Default is False.

        seq : str, optional
            The axes of the Euler angles. Default is "xyz".

        Returns
        -------
        numpy.ndarray
            The Euler angles between the parent frame and the frame in the given convention with shape (3,).
        """
        if not isinstance(degrees, bool):
            raise TypeError("The degrees parameter must be a boolean.")
        if not isinstance(seq, str):
            raise TypeError("The seq parameter must be a string.")
        if not len(seq) == 3:
            raise ValueError("The seq parameter must have 3 characters.")
        if not all([s in 'XYZxyz' for s in seq]):
            raise ValueError("The seq must contain only the characters 'X', 'Y', 'Z', 'x', 'y', 'z'.") 
        return self.get_rotation(convention=convention).as_euler(seq, degrees=degrees)
    
    def set_euler_angles(self, euler_angles: numpy.ndarray, *, convention: Optional[int] = None, degrees: bool = False, seq: str = "xyz") -> None:
        """
        Set the Euler angles between the parent frame and the frame in the given convention.

        Parameters
        ----------
        euler_angles : numpy.ndarray
            The Euler angles between the parent frame and the frame in the given convention with shape (3,).
        
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        degrees : bool, optional
            If True, the Euler angles are in degrees. Default is False.

        seq : str, optional
            The axes of the Euler angles. Default is "xyz".
        """
        if convention is None:
            convention = self._convention
        if not isinstance(convention, int):
            raise TypeError("The convention must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        if not isinstance(degrees, bool):
            raise TypeError("The degrees parameter must be a boolean.")
        if not isinstance(seq, str):
            raise TypeError("The seq parameter must be a string.")
        if not len(seq) == 3:
            raise ValueError("The seq parameter must have 3 characters.")
        if not all([s in 'XYZxyz' for s in seq]):
            raise ValueError("The seq must contain only the characters 'X', 'Y', 'Z', 'x', 'y', 'z'.") 
        euler_angles = numpy.array(euler_angles).reshape((3,)).astype(numpy.float64)
        R = Rotation.from_euler(seq, euler_angles, degrees=degrees)
        self.set_rotation(R, convention=convention)

    @property
    def euler_angles(self) -> numpy.ndarray:
        """
        Getter and setter for the Euler angles between the parent frame and the frame in the convention of the frame.
        The Euler angles are in radians and the axes are "xyz".

        .. seealso::

            - method :meth:`py3dframe.Frame.get_euler_angles` to get the Euler angles in a specific convention.

        Returns
        -------
        numpy.ndarray
            The Euler angles between the parent frame and the frame in the convention of the frame with shape (3,).
        """
        return self.get_euler_angles(convention=self._convention, degrees=False, seq="xyz")

    @euler_angles.setter
    def euler_angles(self, euler_angles: numpy.ndarray) -> None:
        self.set_euler_angles(euler_angles, convention=self._convention, degrees=False, seq="xyz")



    def get_rotation_vector(self, *, convention: Optional[int] = None, degrees: bool = False) -> numpy.ndarray:
        """
        Get the rotation vector between the parent frame and the frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        degrees : bool, optional
            If True, the rotation vector is returned in degrees. Default is False.

        Returns
        -------
        numpy.ndarray
            The rotation vector between the parent frame and the frame in the given convention with shape (3,).
        """
        if not isinstance(degrees, bool):
            raise TypeError("The degrees parameter must be a boolean.")
        return self.get_rotation(convention=convention).as_rotvec(degrees=degrees)

    def set_rotation_vector(self, rotation_vector: numpy.ndarray, *, convention: Optional[int] = None, degrees: bool = False) -> None:
        """
        Set the rotation vector between the parent frame and the frame in the given convention.

        Parameters
        ----------
        rotation_vector : numpy.ndarray
            The rotation vector between the parent frame and the frame in the given convention with shape (3,).
        
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        degrees : bool, optional
            If True, the rotation vector is in degrees. Default is False.
        """
        if not isinstance(degrees, bool):
            raise TypeError("The degrees parameter must be a boolean.")
        rotation_vector = numpy.array(rotation_vector).reshape((3,)).astype(numpy.float64)
        R = Rotation.from_rotvec(rotation_vector, degrees=degrees)
        self.set_rotation(R, convention=convention)
    
    @property
    def rotation_vector(self) -> numpy.ndarray:
        """
        Getter and setter for the rotation vector between the parent frame and the frame in the convention of the frame.
        The rotation vector is in radians.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_rotation_vector` to get the rotation vector in a specific convention.

        Returns
        -------
        numpy.ndarray
            The rotation vector between the parent frame and the frame in the convention of the frame with shape (3,).
        """
        return self.get_rotation_vector(convention=self._convention, degrees=False)
    
    @rotation_vector.setter
    def rotation_vector(self, rotation_vector: numpy.ndarray) -> None:
        self.set_rotation_vector(rotation_vector, convention=self._convention, degrees=False)



    def get_global_frame(self) -> Frame:
        """
        Get the Frame object between the global frame and the frame.

        .. note::

            The parent attribute of the global frame is None.
            The convention of the returned frame is set equal to the convention of the frame.
        
        .. warning::

            The Frame object is a new object. Any change in the returned object will not affect the original object.
            Furthermore, any change in the original object will not affect the returned object.
            It describes the frame at the state of the call.

        Returns
        -------
        Frame
            The global frame of the frame.
        """
        if self._parent is None:
            return self
        
        # Construct the composite transformation between the global frame and the frame.
        R_parent = self._parent.get_global_rotation(convention=0)
        T_parent = self._parent.get_global_translation(convention=0)

        # ====================================================================================================================================
        # Lets note : 
        # Xg : the coordinates of a point in the global frame
        # Xp : the coordinates of the same point in the parent frame
        # Xf : the coordinates of the same point in the frame
        # Rg : the rotation matrix between the global frame and the parent frame
        # Rp : the rotation matrix between the parent frame and the frame
        # Tg : the translation vector between the global frame and the parent frame
        # Tp : the translation vector between the parent frame and the frame
        # R : the rotation matrix between the global frame and the frame
        # T : the translation vector between the global frame and the frame
        # 
        # We have :
        # Xg = Rg * Xp + Tg
        # Xp = Rp * Xf + Tp
        # Xg = R * Xf + T
        #
        # We search R and T:
        # Xg = Rg * (Rp * Xf + Tp) + Tg
        # Xg = Rg * Rp * Xf + Rg * Tp + Tg
        #
        # So the composite transformation is R = Rg * Rp and T = Rg * Tp + Tg
        # ====================================================================================================================================
        rotation = R_parent * self._R_dev
        translation = R_parent.apply(self._T_dev.T).T + T_parent
        global_frame = Frame(translation=translation, rotation=rotation, parent=None, convention=0)
        global_frame.convention = self._convention
        return global_frame



    def get_global_rotation(self, *, convention: Optional[int] = None) -> scipy.spatial.transform.Rotation:
        """
        Get the rotation between the global frame and the frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        Returns
        -------
        scipy.spatial.transform.Rotation
            The rotation between the global frame and the frame in the given convention.
        """
        global_frame = self.get_global_frame()
        return global_frame.get_rotation(convention=convention)

    def set_global_rotation(self, rotation: scipy.spatial.transform.Rotation, *, convention: Optional[int] = None) -> None:
        """
        Set the rotation between the global frame and the frame in the given convention.

        Parameters
        ----------
        rotation : scipy.spatial.transform.Rotation
            The rotation between the global frame and the frame in the given convention.
        
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.
        """
        if self._parent is None:
            self.set_rotation(rotation, convention=convention)
            return
        
        # ====================================================================================================================================
        # Lets note : 
        # Xg : the coordinates of a point in the global frame
        # Xp : the coordinates of the same point in the parent frame
        # Xf : the coordinates of the same point in the frame
        # Rg : the rotation matrix between the global frame and the parent frame
        # Rp : the rotation matrix between the parent frame and the frame
        # Tg : the translation vector between the global frame and the parent frame
        # Tp : the translation vector between the parent frame and the frame
        # R : the rotation matrix between the global frame and the frame
        # T : the translation vector between the global frame and the frame
        # 
        # We have :
        # Xg = Rg * Xp + Tg
        # Xp = Rp * Xf + Tp
        # Xg = R * Xf + T
        #
        # We search Rp:
        # Xg = Rg * (Rp * Xf + Tp) + Tg
        # Xg = Rg * Rp * Xf + Rg * Tp + Tg
        # R = Rg * Rp
        # T = Rg * Tp + Tg
        #
        # So Rp = Rg.inv() * R
        # ====================================================================================================================================

        R_parent = self._parent.get_global_rotation(convention=0)

        rotation, _ = switch_RT_convention(rotation, self._T_dev, convention, 0)
        rotation = R_parent.inv() * rotation
        self.set_rotation(rotation, convention=0)

    @property
    def global_rotation(self) -> scipy.spatial.transform.Rotation:
        """
        Getter and setter for the rotation between the global frame and the frame in the convention of the frame.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_global_rotation` to get the rotation in a specific convention.

        Returns
        -------
        scipy.spatial.transform.Rotation
            The rotation between the global frame and the frame in the convention of the frame.
        """
        return self.get_global_rotation(convention=self._convention)

    @global_rotation.setter
    def global_rotation(self, rotation: scipy.spatial.transform.Rotation) -> None:
        self.set_global_rotation(rotation, convention=self._convention)
    


    def get_global_translation(self, *, convention: Optional[int] = None) -> numpy.ndarray:
        """
        Get the translation vector between the global frame and the frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        Returns
        -------
        numpy.ndarray
            The translation vector between the global frame and the frame in the given convention with shape (3, 1).
        """
        global_frame = self.get_global_frame()
        return global_frame.get_translation(convention=convention)
    
    def set_global_translation(self, translation: numpy.ndarray, *, convention: Optional[int] = None) -> None:
        """
        Set the translation vector between the global frame and the frame in the given convention.

        Parameters
        ----------
        translation : numpy.ndarray
            The translation vector between the global frame and the frame in the given convention with shape (3, 1).
        
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.
        """
        if self._parent is None:
            self.set_translation(translation, convention=convention)
            return
        
        # ====================================================================================================================================
        # Lets note : 
        # Xg : the coordinates of a point in the global frame
        # Xp : the coordinates of the same point in the parent frame
        # Xf : the coordinates of the same point in the frame
        # Rg : the rotation matrix between the global frame and the parent frame
        # Rp : the rotation matrix between the parent frame and the frame
        # Tg : the translation vector between the global frame and the parent frame
        # Tp : the translation vector between the parent frame and the frame
        # R : the rotation matrix between the global frame and the frame
        # T : the translation vector between the global frame and the frame
        # 
        # We have :
        # Xg = Rg * Xp + Tg
        # Xp = Rp * Xf + Tp
        # Xg = R * Xf + T
        #
        # We search Tp:
        # Xg = Rg * (Rp * Xf + Tp) + Tg
        # Xg = Rg * Rp * Xf + Rg * Tp + Tg
        # R = Rg * Rp
        # T = Rg * Tp + Tg
        #
        # So Tp = Rg.inv() * (T - Tg)
        # ====================================================================================================================================

        R_parent = self._parent.get_global_rotation(convention=0)
        T_parent = self._parent.get_global_translation(convention=0)

        _, translation = switch_RT_convention(self._R_dev, translation, 0, convention)
        translation = R_parent.inv().apply((translation - T_parent).T).T
        self.set_translation(translation, convention=0)

    @property
    def global_translation(self) -> numpy.ndarray:
        """
        Getter and setter for the translation vector between the global frame and the frame in the convention of the frame.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_global_translation` to get the translation vector in a specific convention.

        Returns
        -------
        numpy.ndarray
            The translation vector between the global frame and the frame in the convention of the frame with shape (3, 1).
        """
        return self.get_global_translation(convention=self._convention)

    @global_translation.setter
    def global_translation(self, translation: numpy.ndarray) -> None:
        self.set_global_translation(translation, convention=self._convention)
    


    def get_global_rotation_matrix(self, *, convention: Optional[int] = None) -> numpy.ndarray:
        """
        Get the rotation matrix between the global frame and the frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        Returns
        -------
        numpy.ndarray
            The rotation matrix between the global frame and the frame in the given convention with shape (3, 3).
        """
        global_frame = self.get_global_frame()
        return global_frame.get_rotation_matrix(convention=convention)

    def set_global_rotation_matrix(self, rotation_matrix: numpy.ndarray, *, convention: Optional[int] = None) -> None:
        """
        Set the rotation matrix between the global frame and the frame in the given convention.

        Parameters
        ----------
        rotation_matrix : numpy.ndarray
            The rotation matrix between the global frame and the frame in the given convention with shape (3, 3).
        
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.
        """
        if convention is None:
            convention = self._convention
        if not isinstance(convention, int):
            raise TypeError("The convention must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        rotation_matrix = numpy.array(rotation_matrix).reshape((3,3)).astype(numpy.float64)
        if not is_SO3(rotation_matrix):
            raise ValueError("The rotation matrix must be a special orthogonal matrix.")
        R = Rotation.from_matrix(rotation_matrix)
        self.set_global_rotation(R, convention=convention)
    
    @property
    def global_rotation_matrix(self) -> numpy.ndarray:
        """
        Getter and setter for the rotation matrix between the global frame and the frame in the convention of the frame.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_global_rotation_matrix` to get the rotation matrix in a specific convention.

        Returns
        -------
        numpy.ndarray
            The rotation matrix between the global frame and the frame in the convention of the frame with shape (3, 3).
        """
        return self.get_global_rotation_matrix(convention=self._convention)

    @global_rotation_matrix.setter
    def global_rotation_matrix(self, rotation_matrix: numpy.ndarray) -> None:
        self.set_global_rotation_matrix(rotation_matrix, convention=self._convention)


    
    def get_global_quaternion(self, *, convention: Optional[int] = None, scalar_first: bool = True) -> numpy.ndarray:
        """
        Get the quaternion between the global frame and the frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        scalar_first : bool, optional
            If True, the quaternion is returned in the scalar first convention. Default is True.

        Returns
        -------
        numpy.ndarray
            The quaternion between the global frame and the frame in the given convention with shape (4,).
        """
        global_frame = self.get_global_frame()
        return global_frame.get_quaternion(convention=convention, scalar_first=scalar_first)

    def set_global_quaternion(self, quaternion: numpy.ndarray, *, convention: Optional[int] = None, scalar_first: bool = True) -> None:
        """
        Set the quaternion between the global frame and the frame in the given convention.

        Parameters
        ----------
        quaternion : numpy.ndarray
            The quaternion between the global frame and the frame in the given convention with shape (4,).
        
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.
        
        scalar_first : bool, optional
            If True, the quaternion is in the scalar first convention. Default is True.
        """
        global_frame = self.get_global_frame()
        global_frame.set_quaternion(quaternion, convention=convention, scalar_first=scalar_first)
    
    @property
    def global_quaternion(self) -> numpy.ndarray:
        """
        Getter and setter for the quaternion between the global frame and the frame in the convention of the frame.
        The quaternion is in the scalar first convention.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_global_quaternion` to get the quaternion in a specific convention.

        Returns
        -------
        numpy.ndarray
            The quaternion between the global frame and the frame in the convention of the frame with shape (4,).
        """
        return self.get_global_quaternion(convention=self._convention, scalar_first=True)

    @global_quaternion.setter
    def global_quaternion(self, quaternion: numpy.ndarray) -> None:
        self.set_global_quaternion(quaternion, convention=self._convention, scalar_first=True)
    


    def get_global_euler_angles(self, *, convention: Optional[int] = None, degrees: bool = False, seq: str = "xyz") -> numpy.ndarray:
        """
        Get the Euler angles between the global frame and the frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        degrees : bool, optional
            If True, the Euler angles are returned in degrees. Default is False.

        seq : str, optional
            The axes of the Euler angles. Default is "xyz".

        Returns
        -------
        numpy.ndarray
            The Euler angles between the global frame and the frame in the given convention with shape (3,).
        """
        global_frame = self.get_global_frame()
        return global_frame.get_euler_angles(convention=convention, degrees=degrees, seq=seq)

    def set_global_euler_angles(self, euler_angles: numpy.ndarray, *, convention: Optional[int] = None, degrees: bool = False, seq: str = "xyz") -> None:
        """
        Set the Euler angles between the global frame and the frame in the given convention.

        Parameters
        ----------
        euler_angles : numpy.ndarray
            The Euler angles between the global frame and the frame in the given convention with shape (3,).
        
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        degrees : bool, optional
            If True, the Euler angles are in degrees. Default is False.

        seq : str, optional
            The axes of the Euler angles. Default is "xyz".
        """
        global_frame = self.get_global_frame()
        global_frame.set_euler_angles(euler_angles, convention=convention, degrees=degrees, seq=seq)
    
    @property
    def global_euler_angles(self) -> numpy.ndarray:
        """
        Getter and setter for the Euler angles between the global frame and the frame in the convention of the frame.
        The Euler angles are in radians and the axes are "xyz".

        .. seealso::

            - method :meth:`py3dframe.Frame.get_global_euler_angles` to get the Euler angles in a specific convention.

        Returns
        -------
        numpy.ndarray
            The Euler angles between the global frame and the frame in the convention of the frame with shape (3,).
        """
        return self.get_global_euler_angles(convention=self._convention, degrees=False, seq="xyz")

    @global_euler_angles.setter
    def global_euler_angles(self, euler_angles: numpy.ndarray) -> None:
        self.set_global_euler_angles(euler_angles, convention=self._convention, degrees=False, seq="xyz")
    


    def get_global_rotation_vector(self, *, convention: Optional[int] = None, degrees: bool = False) -> numpy.ndarray:
        """
        Get the rotation vector between the global frame and the frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        degrees : bool, optional
            If True, the rotation vector is returned in degrees. Default is False.

        Returns
        -------
        numpy.ndarray
            The rotation vector between the global frame and the frame in the given convention with shape (3,).
        """
        global_frame = self.get_global_frame()
        return global_frame.get_rotation_vector(convention=convention, degrees=degrees)

    def set_global_rotation_vector(self, rotation_vector: numpy.ndarray, *, convention: Optional[int] = None, degrees: bool = False) -> None:
        """
        Set the rotation vector between the global frame and the frame in the given convention.

        Parameters
        ----------
        rotation_vector : numpy.ndarray
            The rotation vector between the global frame and the frame in the given convention with shape (3,).
        
        convention : Optional[int], optional
            The convention to express the transformation. It can be an integer between 0 and 7 or a string corresponding to the conventions. Default is the convention of the frame.

        degrees : bool, optional
            If True, the rotation vector is in degrees. Default is False.
        """
        global_frame = self.get_global_frame()
        global_frame.set_rotation_vector(rotation_vector, convention=convention, degrees=degrees)
    
    @property
    def global_rotation_vector(self) -> numpy.ndarray:
        """
        Getter and setter for the rotation vector between the global frame and the frame in the convention of the frame.
        The rotation vector is in radians.

        .. seealso::

            - method :meth:`py3dframe.Frame.get_global_rotation_vector` to get the rotation vector in a specific convention.

        Returns
        -------
        numpy.ndarray
            The rotation vector between the global frame and the frame in the convention of the frame with shape (3,).
        """
        return self.get_global_rotation_vector(convention=self._convention, degrees=False)

    @global_rotation_vector.setter
    def global_rotation_vector(self, rotation_vector: numpy.ndarray) -> None:
        self.set_global_rotation_vector(rotation_vector, convention=self._convention, degrees=False)
    


    @property
    def global_axes(self) -> numpy.ndarray:
        """
        Getter and setter for the basis vectors of the frame in the global frame coordinates.

        Returns
        -------
        numpy.ndarray
            The basis vectors of the frame in the global frame coordinates with shape (3, 3).
        """
        axes = self.get_global_rotation(convention=0).as_matrix()
        return axes

    @global_axes.setter
    def global_axes(self, axes: numpy.ndarray) -> None:
        axes = numpy.array(axes).reshape((3,3)).astype(numpy.float64)
        norm = numpy.linalg.norm(axes, axis=0)
        if numpy.any(norm == 0):
            raise ValueError("The axes must be non-zero.")
        axes = axes / norm
        self.set_global_rotation_matrix(axes, convention=0)
    


    @property
    def global_x_axis(self) -> numpy.ndarray:
        """
        Getter for the x-axis of the frame in the global frame coordinates.

        .. warning::

            The global_x_axis attributes can't be changed. To change the x-axis, use the global_axes attribute.

        Returns
        -------
        numpy.ndarray
            The x-axis of the frame in the global frame coordinates with shape (3, 1).
        """
        x_axis = self.get_global_rotation(convention=0).as_matrix()[:,0].reshape((3,1))
        return x_axis
    

    
    @property
    def global_y_axis(self) -> numpy.ndarray:
        """
        Getter for the y-axis of the frame in the global frame coordinates.

        .. warning::

            The global_y_axis attributes can't be changed. To change the y-axis, use the global_axes attribute.

        Returns
        -------
        numpy.ndarray
            The y-axis of the frame in the global frame coordinates with shape (3, 1).
        """
        y_axis = self.get_global_rotation(convention=0).as_matrix()[:,1].reshape((3,1))
        return y_axis



    @property
    def global_z_axis(self) -> numpy.ndarray:
        """
        Getter for the z-axis of the frame in the global frame coordinates.

        .. warning::

            The global_z_axis attributes can't be changed. To change the z-axis, use the global_axes attribute.

        Returns
        -------
        numpy.ndarray
            The z-axis of the frame in the global frame coordinates with shape (3, 1).
        """
        z_axis = self.get_global_rotation(convention=0).as_matrix()[:,2].reshape((3,1))
        return z_axis



    @property
    def global_origin(self) -> numpy.ndarray:
        """
        Getter for the origin of the frame in the global frame coordinates.

        .. warning::

            The global_origin attributes can't be changed. To change the origin, use the global_translation attribute.

        Returns
        -------
        numpy.ndarray
            The origin of the frame in the global frame coordinates with shape (3, 1).
        """
        origin = self.get_global_translation(convention=0)
        return origin

    # ====================================================================================================================================
    # Magic methods
    # ====================================================================================================================================
    def __repr__(self) -> str:
        """
        Return the string representation of the Frame object.

        Returns
        -------
        str
            The string representation of the Frame object.
        """
        return f"Frame(origin={self.global_origin}, x_axis={self.global_x_axis}, y_axis={self.global_y_axis}, z_axis={self.global_z_axis}"
    


    def __eq__(self, other: Frame) -> bool:
        """
        Return the equality of the Frame object.
        
        Two Frame objects are equal if their global coordinates are equal.

        Parameters
        ----------
        other : Frame
            The other Frame object to compare.
        
        Returns
        -------
        bool
            True if the Frame objects are equal, False otherwise.
        """
        if not isinstance(other, Frame):
            return False
        global_frame = self.get_global_frame()
        other_global_frame = other.get_global_frame()
        return numpy.allclose(global_frame.translation, other_global_frame.translation) and numpy.allclose(global_frame.rotation.as_quat(), other_global_frame.rotation.as_quat())
    


    def __ne__(self, other: Frame) -> bool:
        """
        Return the inequality of the Frame object.
        
        Two Frame objects are equal if their global coordinates are equal.

        Parameters
        ----------
        other : Frame
            The other Frame object to compare.
        
        Returns
        -------
        bool
            True if the Frame objects are not equal, False otherwise.
        """
        return not self.__eq__(other)
    


    # ====================================================================================================================================
    # draw methods
    # ====================================================================================================================================
    def draw(self, ax: matplotlib.axes.Axes, xcolor: str = 'r', ycolor: str = 'g', zcolor: str = 'b', scale: float = 1, **kwargs) -> Tuple[matplotlib.lines.Line2D, matplotlib.lines.Line2D, matplotlib.lines.Line2D]:
        """
        Draw the frame in the matplotlib axes (projection 3D).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axes to draw the frame.
        
        xcolor : str, optional
            The color of the x-axis. Default is 'r'.
        
        ycolor : str, optional
            The color of the y-axis. Default is 'g'.
        
        zcolor : str, optional
            The color of the z-axis. Default is 'b'.
        
        scale : float, optional
            The scale of the frame. Default is 1.
        
        **kwargs
            Additional arguments to pass to the matplotlib.lines.Line2D constructor.
        
        Returns
        -------
        Tuple[matplotlib.lines.Line2D, matplotlib.lines.Line2D, matplotlib.lines.Line2D]
            The x-axis, y-axis and z-axis of the frame.
        """
        origin = self.global_origin
        x_axis = self.global_x_axis
        y_axis = self.global_y_axis
        z_axis = self.global_z_axis

        x = numpy.hstack((origin, origin + scale * x_axis))
        y = numpy.hstack((origin, origin + scale * y_axis))
        z = numpy.hstack((origin, origin + scale * z_axis))

        x_line = ax.plot(x[0,:], x[1,:], x[2,:], color=xcolor, **kwargs)[0]
        y_line = ax.plot(y[0,:], y[1,:], y[2,:], color=ycolor, **kwargs)[0]
        z_line = ax.plot(z[0,:], z[1,:], z[2,:], color=zcolor, **kwargs)[0]
        return x_line, y_line, z_line



    # ====================================================================================================================================
    # Load and Save method
    # ====================================================================================================================================
    def save_to_dict(self, method: Union[str, Sequence[str]] = ["quaternion", "rotation_vector", "rotation_matrix"]) -> Dict[str, Any]:
        r"""
        Save the Frame object to a dictionary.

        The dictionary has the following structure:

        .. code-block:: python

            {
                "translation": [float, float, float],
                "quaternion": [float, float, float, float],
                "rotation_vector": [float, float, float],
                "rotation_matrix": [[float, float, float], [float, float, float], [float, float, float]],
                "euler_angles": [float, float, float],
                "convention": int
                "parent": None
            }

        - The quaternion is given in WXYZ format (scalar first).
        - The rotation vector is given in radians.
        - The Euler angles are given in radians and the axes are "xyz".
        - The rotation is given in the convention of the frame.
        - The translation vector is given in the convention of the frame.

        .. seealso::

            - method :meth:`py3dframe.Frame.load_from_dict` to load the Frame object from a dictionary.

            For the reader, only one of the rotation keys is needed to reconstruct the frame. The other keys are provided for convenience and user experience.
            The reader chooses the key to use in the following order of preference if several are given:

            - quaternion
            - rotation_vector
            - rotation_matrix
            - euler_angles

        .. warning::

            ``euler_angles`` can raise a this warning : 
            
            - UserWarning: Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.

            I recommand to not use it.

        .. note::

            For retrocompatibility, the default method "quaternion" must be used.

        Parameters
        ----------
        method : Union[str, Sequence[str]], optional
            The method to use to save the rotation. It can be one of the following : "quaternion", "rotation_vector", "rotation_matrix" or "euler_angles".
            Several methods can be used at the same time. Default is ["quaternion", "rotation_vector", "rotation_matrix"].

        Returns
        -------
        Dict[str, Any]
            The dictionary containing the Frame object.
        """
        # Check if the method is a string or a list of strings
        if isinstance(method, str):
            method = [method]
        if not isinstance(method, Sequence):
            raise TypeError("The method must be a string or a list of strings.")
        if not all(isinstance(m, str) and m in ["quaternion", "rotation_vector", "rotation_matrix", "euler_angles"] for m in method):
            raise ValueError("The method must be one of the following : 'quaternion', 'rotation_vector', 'rotation_matrix' or 'euler_angles'.")

        data = {
            "translation": self.translation.flatten().tolist(),
            "convention": self._convention,
        }

        # Add the rotation method to the dictionary
        for m in method:
            if m == "quaternion":
                data["quaternion"] = self.get_quaternion(convention=None, scalar_first=True).tolist()
            elif m == "rotation_vector":
                data["rotation_vector"] = self.get_rotation_vector(convention=None, degrees=False).tolist()
            elif m == "rotation_matrix":
                data["rotation_matrix"] = self.get_rotation_matrix(convention=None).tolist()
            elif m == "euler_angles":
                data["euler_angles"] = self.get_euler_angles(convention=None, degrees=False, seq="xyz").tolist()

        # Add the parent frame to the dictionary
        if self._parent is None:
            data["parent"] = None
        else:
            data["parent"] = self._parent.save_to_dict(method=method)
        return data



    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> Frame:
        r"""
        Load the Frame object from a dictionary.

        The dictionary has the following structure:

        .. code-block:: python

            {
                "translation": [float, float, float],
                "quaternion": [float, float, float, float],
                "rotation_vector": [float, float, float],
                "rotation_matrix": [[float, float, float], [float, float, float], [float, float, float]],
                "euler_angles": [float, float, float],
                "convention": int
                "parent": None
            }

        - The quaternion is given in WXYZ format (scalar first).
        - The rotation vector is given in radians.
        - The Euler angles are given in radians and the axes are "xyz".
        - The rotation is given in the convention of the frame.
        - The translation vector is given in the convention of the frame.

        .. seealso::

            - method :meth:`py3dframe.Frame.save_to_dict` to save the Frame object to a dictionary.

        .. note::

            Only one of the rotation keys is needed to reconstruct the frame. 
            The reader chooses the key to use in the following order of preference if several are given:

            - quaternion
            - rotation_vector
            - rotation_matrix
            - euler_angles

        .. warning::

            ``euler_angles`` can raise a this warning : 
            
            - UserWarning: Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.

            I recommand to not use it.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary containing the Frame object.

        Returns
        -------
        Frame
            The Frame object.
        """
        # Check if the data is a dictionary and contains the required keys
        if not isinstance(data, dict):
            raise TypeError("The data must be a dictionary.")
        if not "translation" in data:
            raise ValueError("The dictionary must contain the 'translation' key.")
        if not "quaternion" in data and not "rotation_vector" in data and not "rotation_matrix" in data and not "euler_angles" in data:
            raise ValueError("The dictionary must contain at least one of the 'quaternion', 'rotation_vector', 'rotation_matrix' or 'euler_angles' keys.")
        if not "convention" in data:
            raise ValueError("The dictionary must contain the 'convention' key.")
        if not "parent" in data:
            raise ValueError("The dictionary must contain the 'parent' key.")
        # Convert the data to the correct types
        translation = numpy.array(data["translation"]).reshape((3,1)).astype(numpy.float64)
        convention = data["convention"]
        parent = data["parent"]
        if parent is None:
            parent_frame = None
        else:
            parent_frame = cls.load_from_dict(parent)
        # According to the order of preference, create the rotation object
        if "quaternion" in data:
            quaternion = numpy.array(data["quaternion"]).reshape((4,)).astype(numpy.float64)
            rotation = Rotation.from_quat(quaternion, scalar_first=True)
        elif "rotation_vector" in data:
            rotation_vector = numpy.array(data["rotation_vector"]).reshape((3,)).astype(numpy.float64)
            rotation = Rotation.from_rotvec(rotation_vector, degrees=False)
        elif "rotation_matrix" in data:
            rotation_matrix = numpy.array(data["rotation_matrix"]).reshape((3,3)).astype(numpy.float64)
            rotation = Rotation.from_matrix(rotation_matrix)
        elif "euler_angles" in data:
            euler_angles = numpy.array(data["euler_angles"]).reshape((3,)).astype(numpy.float64)
            rotation = Rotation.from_euler("xyz", euler_angles, degrees=False)
        else:
            raise ValueError("The dictionary must contain at least one of the 'quaternion', 'rotation_vector', 'rotation_matrix' or 'euler_angles' keys.")
        # Create the Frame object
        frame = cls(translation=translation, rotation=rotation, parent=parent_frame, convention=convention)
        return frame



    def save_to_json(self, filename: str) -> None:
        """
        Save the Frame object to a JSON file.

        .. seealso::

            - method :meth:`py3dframe.Frame.save_to_dict` to save the Frame object to a dictionary.

        Parameters
        ----------
        filename : str
            The name of the JSON file.
        """
        data = self.save_to_dict()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    


    @classmethod
    def load_from_json(cls, filename: str) -> Frame:
        """
        Load the Frame object from a JSON file.

        .. seealso::

            - method :meth:`py3dframe.Frame.load_from_dict` to load the Frame object from a dictionary.

        Parameters
        ----------
        filename : str
            The name of the JSON file.
        
        Returns
        -------
        Frame
            The Frame object.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        frame = cls.load_from_dict(data)
        return frame