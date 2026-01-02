#!/usr/bin/env python3
"""Lightweight wrapper for accessing USD prim transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple
from isaacsim.core.prims import SingleXFormPrim
from pxr import Usd

from .array_backend import mathops as mo
from .grasp_config import PosePq

if TYPE_CHECKING:
    from .grasp_config import ApproachFrameConfig, GraspDetectionConfig


__all__ = ["TargetObject"]




# ---------------------------------------------------------------------------
# Helper functions for coordinate frame transformation
# ---------------------------------------------------------------------------

def _axis_string_to_vector(axis: str) -> Any:
    """Convert axis string (e.g., "+x", "-y") to unit vector."""
    axis_vectors = {
        "+x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "+y": (0.0, 1.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "+z": (0.0, 0.0, 1.0),
        "-z": (0.0, 0.0, -1.0),
    }
    return mo.asarray(axis_vectors[axis])


def _compute_approach_frame_rotation(approach_axis: str, grasp_axis: str) -> Any:
    """
    Compute the rotation matrix from object frame to approach frame.

    The approach frame has:
    - X axis = approach direction (from approach_axis)
    - Y axis = grasp direction (from grasp_axis)
    - Z axis = X × Y (right-hand rule)

    Args:
        approach_axis: Object axis that maps to approach frame +X.
        grasp_axis: Object axis that maps to approach frame +Y.

    Returns:
        3x3 rotation matrix R such that: orientation_approach = orientation_object @ R.T
    """
    # Get the object-frame vectors
    obj_approach = _axis_string_to_vector(approach_axis)  # Maps to +X
    obj_grasp = _axis_string_to_vector(grasp_axis)        # Maps to +Y
    
    # Compute up axis using right-hand rule: Z = X × Y
    obj_up = mo.cross(obj_approach, obj_grasp)
    obj_up = obj_up / mo.norm(obj_up)  # Normalize

    # Build rotation matrix
    # The rows of R are where each approach-frame axis comes from in object frame
    # R transforms object coords to approach coords:
    # Approach X (forward) <- obj_approach
    # Approach Y (grasp)   <- obj_grasp
    # Approach Z (up)      <- obj_up
    R = mo.vstack([obj_approach, obj_grasp, obj_up])
    return R


class TargetObject:
    """A simple wrapper around a USD prim for pose access.

    This class wraps an existing USD prim (specified by path) with a
    ``SingleXFormPrim`` and exposes methods for reading/writing the world pose
    and obtaining the 4x4 homogeneous transformation matrix.

    Optionally, a GraspDetectionConfig can be provided to configure:
    1. Approach frame transformation (via target_frame inside the config)
    2. Handle position computation for objects with graspable handles (e.g., fans)

    Args:
        prim_path: The USD prim path of an existing object in the scene.
        name: Optional friendly name. Defaults to the last segment of the path.
        grasp_config: Optional GraspDetectionConfig containing:
                      - target_frame: ApproachFrameConfig for coordinate transformation
                      - handle_y_offset, handle_x_offset: for handle position computation
    
    Example:
        >>> from grasp_config import GraspDetectionConfig
        >>> config = GraspDetectionConfig(handle_y_offset=0.025)
        >>> fan = TargetObject(
        ...     prim_path="/World/Fan",
        ...     grasp_config=config,
        ... )
        >>> # Get handle poses
        >>> left_pose, right_pose = fan.get_handle_poses()
        >>> print(f"Left handle position: {left_pose.p}")
    """

    def __init__(
        self, 
        prim_path: str, 
        name: Optional[str] = None,
        grasp_config: Optional["GraspDetectionConfig"] = None,
    ) -> None:
        self._prim_path = prim_path
        self._name = name or prim_path.rsplit("/", 1)[-1]
        self._xform = SingleXFormPrim(prim_path=prim_path, name=self._name)
        self._grasp_config = grasp_config
        
        # Pre-compute rotation matrix if approach frame is set (via grasp_config.target_frame)
        self._approach_rotation: Optional[Any] = None
        if grasp_config is not None and grasp_config.target_frame is not None:
            self._approach_rotation = _compute_approach_frame_rotation(
                grasp_config.target_frame.approach_axis,
                grasp_config.target_frame.grasp_axis,
            )

    @property
    def name(self) -> str:
        """The friendly name of this object."""
        return self._name

    @property
    def prim(self) -> Usd.Prim:
        """The underlying USD prim."""
        return self._xform.prim

    @property
    def prim_path(self) -> str:
        """The USD prim path."""
        return self._prim_path

    @property
    def grasp_config(self) -> Optional["GraspDetectionConfig"]:
        """The grasp detection configuration, if set."""
        return self._grasp_config

    @property
    def target_frame(self) -> Optional["ApproachFrameConfig"]:
        """The approach frame configuration (from grasp_config.target_frame), if set."""
        if self._grasp_config is None:
            return None
        return self._grasp_config.target_frame

    def get_raw_world_pose(self) -> Tuple[Any, Any]:
        """Get the object's world pose in its native frame (no transformation).

        Returns:
            A tuple (position, orientation) where position is a 3D vector
            and orientation is a quaternion in wxyz format.
            Backend depends on SimulationContext (numpy ndarray or torch Tensor).
        """
        return self._xform.get_world_pose()

    def get_world_pose(self) -> Tuple[Any, Any]:
        """Get the object's world pose (with approach frame transformation if set).

        If an approach_frame is configured, the returned orientation is
        transformed so that:
        - The object's approach axis aligns with +X (forward)
        - The object's grasp axis aligns with +Y (lateral)
        - The up axis (Z) is derived via right-hand rule

        Returns:
            A tuple (position, orientation) where position is a 3D vector
            and orientation is a quaternion in wxyz format.
            Backend depends on SimulationContext (numpy ndarray or torch Tensor).
        """
        position, orientation = self._xform.get_world_pose()
        
        if self._approach_rotation is None:
            return position, orientation
        
        # Apply approach frame transformation to orientation
        # R_world_object = rotation from object frame to world frame
        # R_approach = rotation from object frame to approach frame
        # R_world_approach = R_world_object @ R_approach^T
        R_world_object = mo.quat_to_rot_matrix(orientation)
        R_world_approach = R_world_object @ self._approach_rotation.T
        
        new_orientation = mo.rot_matrix_to_quat(R_world_approach)
        
        return position, new_orientation

    def set_world_pose(
        self,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
    ) -> None:
        """Set the object's world pose.

        Note: This sets the raw pose without any transformation.

        Args:
            position: The 3D position (x, y, z) in world coordinates.
            orientation: The orientation quaternion in wxyz format.
        """
        self._xform.set_world_pose(position, orientation)

    def get_transform(self) -> Any:
        """Get the object's world pose as a 4x4 homogeneous transformation matrix.

        If an approach frame is configured, the returned matrix uses the
        transformed orientation.

        Returns:
            A 4x4 array/tensor representing the homogeneous transformation matrix.
        """
        position, orientation = self.get_world_pose()
        pose = PosePq(position, orientation)
        return pose.to_T()

    # -----------------------------------------------------------------------
    # Handle pose computation
    # -----------------------------------------------------------------------

    def get_handle_poses(self) -> Tuple[PosePq, PosePq]:
        """
        Get virtual left and right handle poses in world frame.

        The handle positions are computed from the object center using
        offsets from the grasp_config. The offsets are applied in the
        approach frame coordinate system:
        - X axis: approach direction (EE forward)
        - Y axis: grasp direction (+Y = left handle, -Y = right handle)
        - Z axis: up direction

        Both handles share the same orientation as the object (with approach
        frame transformation applied if configured).

        Returns:
            Tuple of (left_handle_pose, right_handle_pose) in world frame.
            Each is a PosePq with:
            - p: 3D position vector
            - q: quaternion in wxyz format

        Raises:
            ValueError: If grasp_config is not set.

        Example:
            >>> from grasp_config import GraspDetectionConfig
            >>> config = GraspDetectionConfig(handle_y_offset=0.025)
            >>> fan = TargetObject("/World/Fan", grasp_config=config)
            >>> left_pose, right_pose = fan.get_handle_poses()
            >>> print(f"Left handle position: {left_pose.p}")
            >>> print(f"Left handle orientation: {left_pose.q}")
        """
        if self._grasp_config is None:
            raise ValueError(
                "grasp_config is not set. Provide GraspDetectionConfig at initialization."
            )
        
        position, orientation = self.get_world_pose()
        R = mo.quat_to_rot_matrix(orientation)

        # Left handle: +Y offset in approach frame
        left_offset_local = mo.asarray([
            self._grasp_config.handle_x_offset,
            self._grasp_config.handle_y_offset,
            0.0,
        ])
        # Right handle: -Y offset in approach frame
        right_offset_local = mo.asarray([
            self._grasp_config.handle_x_offset,
            -self._grasp_config.handle_y_offset,
            0.0,
        ])

        left_handle_pos = position + R @ left_offset_local
        right_handle_pos = position + R @ right_offset_local

        # Both handles share the same orientation as the object
        left_pose = PosePq(left_handle_pos, orientation)
        right_pose = PosePq(right_handle_pos, orientation)

        return left_pose, right_pose
