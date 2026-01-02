#!/usr/bin/env python3
"""
Articulation observer for monitoring robot arm and gripper state.

This module provides a composition-based observer class that wraps a
SingleArticulation to expose joint positions, velocities, end-effector pose,
and gripper state without any motion control functionality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from isaacsim.core.prims import SingleArticulation, SingleXFormPrim

from .array_backend import mathops as mo
from .grasp_config import PosePq


__all__ = ["ArticulationObserver", "RobotJointConfig"]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class RobotJointConfig:
    """
    Configuration for robot joint names and gripper parameters.

    Attributes:
        arm_joint_names: Names of the arm joints (in order).
        gripper_joint_names: Names of the gripper joints.
        end_effector_prim_suffix: Rail prim path suffix from robot root.
        end_effector_offset: End-effector offset from rail in rail local frame.
        left_finger_y_offset: Left finger offset from EE along local Y-axis.
        right_finger_y_offset: Right finger offset from EE along local Y-axis.
    """

    arm_joint_names: Sequence[str] = field(default_factory=lambda: (
        "Revolute7",
        "Revolute6",
        "Revolute5",
        "Revolute4",
        "Revolute3",
        "Revolute2",
        "Revolute1",
    ))
    gripper_joint_names: Sequence[str] = field(default_factory=lambda: (
        "Slider9",
        "Slider10",
    ))
    end_effector_prim_suffix: str = "TF_1/gripper_base_2/rail_2"
    end_effector_offset: Sequence[float] = field(default_factory=lambda: (0.06, 0.0, 0.0))
    left_finger_y_offset: float = 0.075
    right_finger_y_offset: float = -0.075


# ---------------------------------------------------------------------------
# ArticulationObserver
# ---------------------------------------------------------------------------

class ArticulationObserver:
    """
    Observer for monitoring robot articulation state.

    This class wraps a SingleArticulation to provide read-only access to:
    - Joint positions and velocities (full, arm subset, gripper subset)
    - End-effector pose via a rail prim in world space
    - Gripper finger positions and width

    No motion control or commanding functionality is included.

    Args:
        prim_path: The USD prim path of the robot articulation.
        joint_config: Robot joint configuration. If None, uses default RobotJointConfig.
        name: Optional friendly name for logging.

    Example:
        >>> observer = ArticulationObserver(
        ...     prim_path="/World/Robot",
        ... )
        >>> observer.initialize()
        >>> ee_pose = observer.get_end_effector_pose()
        >>> print(f"EE position: {ee_pose.p}, orientation: {ee_pose.q}")
    """

    def __init__(
        self,
        prim_path: str,
        joint_config: Optional[RobotJointConfig] = None,
        name: Optional[str] = None,
    ) -> None:
        self._prim_path = prim_path
        self._name = name or prim_path.rsplit("/", 1)[-1]

        # Use provided config or defaults
        self._config = joint_config or RobotJointConfig()

        # Create the articulation wrapper
        self._articulation = SingleArticulation(
            prim_path=prim_path,
            name=self._name,
        )

        # End-effector rail prim (initialized later)
        self._ee_rail_prim_path: Optional[str] = None
        self._ee_rail_xform: Optional[SingleXFormPrim] = None

        # Joint index caches (populated after initialize)
        self._arm_joint_indices: Optional[List[int]] = None
        self._gripper_joint_indices: Optional[List[int]] = None
        self._initialized = False

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The friendly name of this observer."""
        return self._name

    @property
    def prim_path(self) -> str:
        """The USD prim path of the articulation."""
        return self._prim_path

    @property
    def articulation(self) -> SingleArticulation:
        """The underlying SingleArticulation object."""
        return self._articulation

    @property
    def config(self) -> RobotJointConfig:
        """The robot joint configuration."""
        return self._config

    @property
    def num_dof(self) -> int:
        """Total number of degrees of freedom."""
        return self._articulation.num_dof

    @property
    def dof_names(self) -> List[str]:
        """List of all DOF names."""
        return list(self._articulation.dof_names)

    @property
    def arm_joint_names(self) -> List[str]:
        """Names of the arm joints."""
        return list(self._config.arm_joint_names)

    @property
    def gripper_joint_names(self) -> List[str]:
        """Names of the gripper joints."""
        return list(self._config.gripper_joint_names)

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def initialize(self, physics_sim_view=None) -> None:
        """
        Initialize the observer.

        This must be called after the simulation has started and the
        articulation is valid. It sets up:
        - The underlying articulation
        - The end-effector rail prim wrapper
        - Joint index caches for arm and gripper subsets

        Args:
            physics_sim_view: Optional physics simulation view.
        """
        if self._initialized:
            return

        # Initialize articulation
        self._articulation.initialize(physics_sim_view)

        # Build joint index caches
        dof_names = list(self._articulation.dof_names)
        print(f"[{self._name}] Available DOFs: {dof_names}")

        self._arm_joint_indices = []
        for jname in self._config.arm_joint_names:
            if jname in dof_names:
                self._arm_joint_indices.append(dof_names.index(jname))
            else:
                print(f"[{self._name}] Warning: Arm joint '{jname}' not found in articulation")

        self._gripper_joint_indices = []
        for jname in self._config.gripper_joint_names:
            if jname in dof_names:
                self._gripper_joint_indices.append(dof_names.index(jname))
            else:
                print(f"[{self._name}] Warning: Gripper joint '{jname}' not found in articulation")

        print(f"[{self._name}] Arm joint indices: {self._arm_joint_indices}")
        print(f"[{self._name}] Gripper joint indices: {self._gripper_joint_indices}")

        ee_suffix = str(self._config.end_effector_prim_suffix).strip("/")
        if not ee_suffix:
            raise ValueError("end_effector_prim_suffix cannot be empty.")
        root_path = self._prim_path.rstrip("/")
        self._ee_rail_prim_path = f"{root_path}/{ee_suffix}"
        self._ee_rail_xform = SingleXFormPrim(
            prim_path=self._ee_rail_prim_path,
            name=f"{self._name}_ee_rail",
        )

        self._initialized = True
        print(f"[{self._name}] ArticulationObserver initialized successfully")


    # -----------------------------------------------------------------------
    # Joint state access
    # -----------------------------------------------------------------------

    def get_joint_positions(self) -> Any:
        """Get all joint positions."""
        return self._articulation.get_joint_positions()

    def get_joint_velocities(self) -> Any:
        """Get all joint velocities."""
        return self._articulation.get_joint_velocities()

    def get_arm_joint_positions(self) -> Any:
        """Get arm joint positions (7-DOF)."""
        if self._arm_joint_indices is None:
            raise RuntimeError("ArticulationObserver not initialized. Call initialize() first.")
        all_positions = self._articulation.get_joint_positions()
        return all_positions[self._arm_joint_indices]

    def get_arm_joint_velocities(self) -> Any:
        """Get arm joint velocities (7-DOF)."""
        if self._arm_joint_indices is None:
            raise RuntimeError("ArticulationObserver not initialized. Call initialize() first.")
        all_velocities = self._articulation.get_joint_velocities()
        return all_velocities[self._arm_joint_indices]

    def get_gripper_joint_positions(self) -> Any:
        """
        Get gripper joint positions.

        Returns:
            Array of [Slider9, Slider10] positions.
            - Slider9: range [0, 0.05], positive = open
            - Slider10: range [-0.05, 0], negative = open
        """
        if self._gripper_joint_indices is None:
            raise RuntimeError("ArticulationObserver not initialized. Call initialize() first.")
        all_positions = self._articulation.get_joint_positions()
        return all_positions[self._gripper_joint_indices]

    def get_gripper_joint_velocities(self) -> Any:
        """Get gripper joint velocities."""
        if self._gripper_joint_indices is None:
            raise RuntimeError("ArticulationObserver not initialized. Call initialize() first.")
        all_velocities = self._articulation.get_joint_velocities()
        return all_velocities[self._gripper_joint_indices]

    # -----------------------------------------------------------------------
    # End-effector pose
    # -----------------------------------------------------------------------

    def get_end_effector_pose(
        self,
        config: Optional[Any] = None,
    ) -> PosePq:
        """
        Get the end-effector pose from the rail prim in world space.

        Args:
            config: Unused. Present for API compatibility.

        Returns:
            PosePq with:
            - p: 3D position in world frame
            - q: quaternion in wxyz format
        """
        if self._ee_rail_xform is None:
            raise RuntimeError("ArticulationObserver not initialized. Call initialize() first.")

        rail_pos, rail_quat = self._ee_rail_xform.get_world_pose()
        rail_rot = mo.quat_to_rot_matrix(rail_quat)

        offset = mo.asarray(self._config.end_effector_offset)
        ee_pos = rail_pos + rail_rot @ offset
        ee_quat = rail_quat
        return PosePq(ee_pos, ee_quat)

    def get_finger_poses(
        self,
        config: Optional[Any] = None,
    ) -> Tuple[PosePq, PosePq]:
        """
        Get both finger poses as PosePq objects.

        Since the gripper sliders are not part of the rail prim pose, we
        compute the end-effector pose and apply the finger offsets along
        the local Y-axis.

        Args:
            config: Unused. Present for API compatibility.

        Returns:
            Tuple of (left_finger_pose, right_finger_pose) in world frame.
            Each is a PosePq with:
            - p: 3D position vector
            - q: quaternion in wxyz format

        Example:
            >>> left_pose, right_pose = observer.get_finger_poses()
            >>> print(f"Left finger position: {left_pose.p}")
            >>> print(f"Left finger orientation: {left_pose.q}")
        """
        ee_pose = self.get_end_effector_pose(config)
        ee_pos = ee_pose.p
        ee_rot = mo.quat_to_rot_matrix(ee_pose.q)

        # Get current slider positions
        gripper_pos = self.get_gripper_joint_positions()
        left_slider_pos = gripper_pos[0] if len(gripper_pos) > 0 else 0.0
        right_slider_pos = gripper_pos[1] if len(gripper_pos) > 1 else 0.0

        # Left finger offset in local frame: Y = base_offset - slider_position
        # (slider moves in -Y direction according to URDF axis)
        y_axis = mo.asarray([0.0, 1.0, 0.0])
        left_local_offset = (
            mo.asarray([0.0, self._config.left_finger_y_offset, 0.0])
            - y_axis * left_slider_pos
        )
        left_world_offset = ee_rot @ left_local_offset
        left_finger_pos = ee_pos + left_world_offset

        # Right finger offset in local frame
        right_local_offset = (
            mo.asarray([0.0, self._config.right_finger_y_offset, 0.0])
            - y_axis * right_slider_pos
        )
        right_world_offset = ee_rot @ right_local_offset
        right_finger_pos = ee_pos + right_world_offset

        # Convert rotation matrix to quaternion (wxyz format)
        ee_quat = mo.rot_matrix_to_quat(ee_rot)

        left_pose = PosePq(left_finger_pos, ee_quat)
        right_pose = PosePq(right_finger_pos, ee_quat)

        return left_pose, right_pose

    # -----------------------------------------------------------------------
    # World pose access
    # -----------------------------------------------------------------------

    def get_world_pose(self) -> Tuple[Any, Any]:
        """
        Get the robot base world pose.

        Returns:
            Tuple of (position, orientation) where orientation is quaternion (wxyz).
            Backend depends on SimulationContext (numpy ndarray or torch Tensor).
        """
        return self._articulation.get_world_pose()
