# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from tkinter.font import names
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform


from .robotic_env_cfg import RoboticEnvCfg

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

from gym import spaces
import numpy as np

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from .pose_monitor import PoseMonitor
from .grasp_config import GraspDetectionConfig

class RoboticEnv(DirectRLEnv):
    cfg: RoboticEnvCfg

    def __init__(self, cfg: RoboticEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        self.arm_dof_ids, _ = self.robot.find_joints([f"Revolute{i}" for i in range(1, 8)])
        self.grip_dof_ids, _ = self.robot.find_joints(["Slider9", "Slider10"])

        # EE delta position (meters per step) + EE delta rotation + gripper command
        ee_step = 0.003   # 3 mm / step（很穩，之後可調）

        low  = np.array([-ee_step, -ee_step, -ee_step, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([ ee_step,  ee_step,  ee_step,  1.0,  1.0,  1.0,  1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        names = self.robot.body_names
        print("Robot bodies:", names)
        self._finger_idx_pair = (names.index("grasp_3"), names.index("grasp_4"))
        print(f"[EE] using fingers midpoint: grasp_3, grasp_4")
        
        self._init_tensors_once()

        fan_world_pos = self.fan.data.root_pos_w.clone()            # [num_envs, 3]
        fan_world_quat = self.fan.data.root_quat_w.clone()          # [num_envs, 4]
        self._fan_spawn_local_pos  = fan_world_pos  - self.scene.env_origins  # local
        self._fan_spawn_local_quat = fan_world_quat.clone()                     # 世界四元數 = local（根是 env_x 原點）

        self.ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",          
            use_relative_mode=True,        
            ik_method="dls",                  # damped least squares
        )

        self.ik_controller = DifferentialIKController(
            cfg=self.ik_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

        self.touch_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.episode_touch_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

        self.total_episodes = 0
        self.total_touches = 0

        # ===== Grasp (holding) detection buffers =====
        self.grasp_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._grasp_frame_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

        # ===== Grasp thresholds (mirror GraspDetectionConfig defaults) =====
        # Slider9 ~ +0.02, Slider10 ~ -0.02 when holding
        self.grip_position_min = 0.018
        self.grip_position_max = 0.022

        # grasp zone by EE/finger-to-fan distance (meters)
        self.grasp_zone_min_m = 0.00
        self.grasp_zone_max_m = 0.03

        # consecutive frames to confirm holding
        self.grasp_confirm_frames = 3
        
        self.ee_offset_tf1 = torch.tensor([0.118, 0.0, -0.003], device=self.device, dtype=torch.float32)

        approach_axis = "-y"   # EE 往 fan 的方向（approach / forward）
        grasp_axis    = "+x"   # 夾爪張開方向（finger axis）

        self.R_obj_to_approach = compute_obj_to_approach_R(
            approach_axis=approach_axis,
            grasp_axis=grasp_axis,
            device=self.device,
            dtype=torch.float32,
        )

        # test
        print("dt =", self.cfg.sim.dt, "decimation =", self.cfg.decimation)
        print("episode_length_s =", self.cfg.episode_length_s)
        print("max_episode_length (expected) ≈",
            int(self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)))
        print("actual max_episode_length (env) =", int(self.max_episode_length))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # add objects
        fan_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/fan",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.cfg.fan_usd,
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, linear_damping=0.01, angular_damping=0.01,
                    max_depenetration_velocity=2.0
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.15),  # 依模型調
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=self.cfg.fan_spawn_base, rot=(1,0,0,0),
            ),
        )
        
        self.fan = RigidObject(fan_cfg)

        # plate = sim_utils.UsdFileCfg(usd_path=self.cfg.plate_usd)
        # plate.func("/World/envs/env_.*/plate", plate, translation=self.cfg.plate_spawn_base)

        # rack = sim_utils.UsdFileCfg(usd_path=self.cfg.rack_usd)
        # rack.func("/World/envs/env_.*/rack", rack, translation=self.cfg.rack_spawn_base)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["fan"]   = self.fan
        # self.scene.
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.jacobians = None

        # monitor
        # self.monitor = []
        # for i in range(self.num_envs):
        #     mon = PoseMonitor.create_default(
        #         robot_prim_path=f"/World/envs/env_{i}/Robot/RS_M90E7A_Left",
        #         fan_prim_path=f"/World/envs/env_{i}/fan",
        #         ground_truth_prim_path=f"/World/envs/env_{i}/rack",
        #     )
        #     self.monitor.append(mon)
        
        # self.monitor_initized = False

    def _init_tensors_once(self):
        self.prev_actions = torch.zeros((self.num_envs, self.action_space.shape[0]), device=self.device)
        self.prev_xy_dist = torch.zeros((self.num_envs,), device=self.device)
        self.ee_pos  = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.fan_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.fan_quat= torch.zeros((self.num_envs, 4), device=self.device)
    
    def _compute_intermediate(self):
        itf = self.robot.body_names.index("TF_1")
        # --- raw TF_1 pose (env-local position, world quat) ---
        tf1_pos_local = self.robot.data.body_pos_w[:, itf] - self.scene.env_origins   # (N,3)
        tf1_quat_wxyz = self.robot.data.body_quat_w[:, itf]                           # (N,4) assumed wxyz

        # --- apply TF1->grasp_center offset in TF_1 local frame ---
        R_tf1 = quat_to_rot_wxyz(tf1_quat_wxyz)                                       # (N,3,3)

        # ensure offset dtype/device matches (in case mixed precision)
        ee_offset = self.ee_offset_tf1.to(device=tf1_pos_local.device, dtype=tf1_pos_local.dtype)

        ee_pos_local = tf1_pos_local + torch.matmul(R_tf1, ee_offset.view(1, 3, 1)).squeeze(-1)

        # --- publish "EE pose" consistent with PoseMonitor ---
        self.ee_pos = ee_pos_local
        self.ee_quat = tf1_quat_wxyz  # IMPORTANT: monitor keeps TF_1 orientation (no rotation offset)

        # print("TF_1 EE位置:", self.ee_pos)
        # print("手指中點位置:", self.finger_pos)

        ## use monitor to get more accurate EE pos
        # ee_p_list = []
        # ee_q_list = []

        # for mon in self.monitor:
        #     pose = mon.get_end_effector_pose()   # PosePq
        #     # pose.p: (3,)  pose.q: (4,)  (usually numpy or list)
        #     ee_p_list.append(torch.as_tensor(pose.p, device=self.device, dtype=torch.float32))
        #     ee_q_list.append(torch.as_tensor(pose.q, device=self.device, dtype=torch.float32))

        # ee_p = torch.stack(ee_p_list, dim=0)   # (N,3)
        # ee_q = torch.stack(ee_q_list, dim=0)   # (N,4)

        # # 如果 monitor 回來的是 world pose，你這裡照你原本做法轉成 env-local
        # self.ee_pos  = ee_p - self.scene.env_origins
        # self.ee_quat = ee_q
        # print("monitor EE位置:", self.ee_pos)

        self.fan_pos  = self.fan.data.root_pos_w - self.scene.env_origins
        self.fan_quat = self.fan.data.root_quat_w

        idx10 = self.cfg.dof_names.index("Slider10")
        idx09 = self.cfg.dof_names.index("Slider9")
        jpos = self.robot.data.joint_pos
        self.gripper_gap = (jpos[:, idx10] - jpos[:, idx09]).abs()

    def _check_touch(self) -> torch.Tensor:
        touch = (torch.linalg.norm(self.ee_pos - self.fan_pos, dim=-1) < 0.15)

        return touch
    
    def _check_grasped(self) -> torch.Tensor:
        # --- (1) gripper closed check (symmetric range) ---
        idx10 = self.cfg.dof_names.index("Slider10")
        idx09 = self.cfg.dof_names.index("Slider9")
        jpos = self.robot.data.joint_pos  # (N, dof)

        slider9  = jpos[:, idx09]
        slider10 = jpos[:, idx10]

        slider9_ok  = (slider9  >= self.grip_position_min) & (slider9  <= self.grip_position_max)
        slider10_ok = (slider10 >= -self.grip_position_max) & (slider10 <= -self.grip_position_min)
        is_closed = slider9_ok & slider10_ok

        # --- (2) grasp zone check (distance to fan) ---
        # PoseMonitor uses EE-to-fan error distance; in your env you already use finger midpoint to reach,
        # so we'll use finger_pos to fan_pos for grasp-zone.
        ee_dist = torch.linalg.norm(self.ee_pos - self.fan_pos, dim=-1)
        is_in_zone = (ee_dist >= self.grasp_zone_min_m) & (ee_dist <= self.grasp_zone_max_m)

        # --- (3) frame-based confirmation ---
        candidate = is_closed & is_in_zone
        self._grasp_frame_count = torch.where(
            candidate,
            self._grasp_frame_count + 1,
            torch.zeros_like(self._grasp_frame_count),
        )
        confirmed = self._grasp_frame_count >= int(self.grasp_confirm_frames)

        self.grasp_buf = confirmed
        return self.grasp_buf

        # grasped_list = []
        # for mon in self.monitor:
        #     grasped = mon.is_holding_fan()
        #     grasped_list.append(grasped)
        # grasped_tensor = torch.as_tensor(grasped_list, device=self.device, dtype=torch.bool)
        # return grasped_tensor
    
    def _update_jacobian(self):
        # 1) EE body index (cache once is fine)
        if not hasattr(self, "_ee_body_idx"):
            self._ee_body_idx = self.robot.body_names.index("TF_1")

        # 2) extract EE Jacobian for arm joints
        # shape: (N, 6, 7)
        J_all = self.robot.root_physx_view.get_jacobians()  # (N, num_bodies, 6, num_dof)
        J = J_all[:, self._ee_body_idx, :, self.arm_dof_ids]  # (N, 6, 7)
        self.jacobians = J

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        actions = self.actions
        if actions.dim() == 3:
            actions = actions[:, 0, :]
        assert actions.shape[-1] == 7, f"Expected action dim=7 (Δx,Δy,Δz,Δrx,Δry,Δrz,g), got {actions.shape}"

        # ---- 2) split ----
        ee_cmd = actions[:, :6]     # (N,3)
        g_cmd    = actions[:, 6]      # (N,)

        # ---- 4) update current EE pose ----
        self._compute_intermediate()
        self._update_jacobian()
        ee_pos  = self.ee_pos
        ee_quat = self.ee_quat

        # ---- 5) IK command (relative) ----
        # Requires cfg.command_type="position" and cfg.use_relative_mode=True
        self.ik_controller.set_command(ee_cmd, ee_pos=ee_pos, ee_quat=ee_quat)

        # ---- 6) compute arm joint velocities ----
        q_arm = self.robot.data.joint_pos[:, self.arm_dof_ids]   # (N,7)
        J = self.jacobians                         # (N,6,7) for the SAME EE link as ee_pos/quat

        q_des = self.ik_controller.compute(
            jacobian=J,
            joint_pos=q_arm,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
        )  # -> (N,7)

        # ---- 7) assemble full joint velocity target ----
        joint_vels = torch.zeros(
            (self.num_envs, len(self.arm_dof_ids) + len(self.grip_dof_ids)),
            device=self.device,
            dtype=torch.float32,
        )

        # arm
        dt = self.cfg.sim.dt * self.cfg.decimation
        qd_arm = (q_des - q_arm) / dt
        joint_vels[:, self.arm_dof_ids] = qd_arm

        # gripper: map g_cmd (-1..1) -> slider speed (m/s)
        # IMPORTANT: keep small to avoid contact blow-ups
        g_speed = torch.clamp(g_cmd, -1.0, 1.0) * 0.05  # 0.05 m/s safe start

        joint_vels[:, self.grip_dof_ids[0]] =  g_speed   # Slider9
        joint_vels[:, self.grip_dof_ids[1]] = -g_speed   # Slider10

        # ---- 8) apply ----
        self.robot.set_joint_velocity_target(joint_vels)
    
    def _get_observations(self) -> dict:
        self._compute_intermediate()

        rel_pos = self.ee_pos - self.fan_pos
        obs_list = [
            self.ee_pos,                # 3
            self.ee_quat,               # 4
            self.fan_pos,               # 3
            self.fan_quat,              # 4
            self.gripper_gap.unsqueeze(-1),  # 1
            self.prev_actions           # 7
        ]
        obs = torch.cat(obs_list, dim=-1)  # 維度 = 3+4+4+3+1+4 = 19
        self.cfg.observation_space = obs.shape[-1]  # 動態校正

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate()

        # === 1) 距離 ===
        rel = self.ee_pos - self.fan_pos
        r_reach = -torch.linalg.norm(rel, dim=-1)
        # print("距離獎勵:", r_reach)

        # === 角度 ===
        R_ee  = quat_to_rot_wxyz(self.ee_quat)     # world<-ee(TF_1)
        R_fan = quat_to_rot_wxyz(self.fan_quat)    # world<-fan(object)

        # world<-fan_approach = world<-fan_object @ object<-approach
        R_fan_app = R_fan @ self.R_obj_to_approach.T

        # angle between EE and fan_approach
        trace = R_ee.transpose(-1, -2) @ R_fan_app
        tr = trace[..., 0, 0] + trace[..., 1, 1] + trace[..., 2, 2]
        c = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
        ang = torch.acos(c)

        r_angle = -0.5 * ang
        # print("角度獎勵:", r_angle)

        ## use monitor EE directly
        # r_reach_list = []
        # for mon in self.monitor:
        #     error = mon.get_ee_to_fan_error()
        #     r_reach_list.append(error.distance)
        #     print("Monitor error distance:", error.distance)
        #     # print(f"Monitor error position: {error.position_error}")
        # r_reach_tensor = torch.as_tensor(r_reach_list, device=self.device, dtype=torch.float32)
        # r_reach = -r_reach_tensor
        # print("monitor-距離獎勵:", r_reach)

        # === 2) 接觸獎勵 ===
        touch = self._check_touch()
        newly_touch = touch & (~self.touch_buf)
        r_touch = newly_touch.float() * 5.0
        self.touch_buf |= touch
        
        # === 2) 抓取獎勵 ===
        grasped = self._check_grasped()
        self.grasp_success_buf = grasped.clone()
        r_grasp = grasped.float() * 200.0

        rew = r_reach + r_grasp + r_angle
        # === 狀態記錄 ===
        self.prev_actions = self.actions.clone()

        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._compute_intermediate()
        grasped = self._check_grasped()
        # touching = self._check_touch()
        touching = False

        success = grasped | touching

        done = time_out | success
        return done, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # initialize monitors
        # if not self.monitor_initized:
        #     for mon in self.monitor:
        #         mon.initialize()

        #     print("PoseMonitor initialized.")
        #     self.monitor_initized = True
        
        # # reset montitors
        # for i in env_ids:
        #     self.monitor[i].reset_holding_confirmation()

        # set the root state for the reset envs
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        # joints: 打開夾爪，手臂回到 default（或你自定 reset 姿）
        jpos = self.robot.data.default_joint_pos[env_ids].clone()
        # 讓夾爪張開一點（0.02 m 之類）
        jpos[:, 7] =  0.02
        jpos[:, 8] = -0.02
        jvel = torch.zeros_like(jpos)
        self.robot.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)
        self.robot.reset()

        # fan randomize
        fan_state = self.fan.data.default_root_state[env_ids].clone()
        # 位置＝(落位 local) + env_origin
        fan_state[:, 0:3] = self._fan_spawn_local_pos[env_ids] + self.scene.env_origins[env_ids]

        # 角度：沿用當時落位的方向（或保留你自己的 yaw 隨機）
        fan_state[:, 3:7] = self._fan_spawn_local_quat[env_ids]

        # 速度清零，避免解穿插把它彈走
        fan_state[:, 7:] = 0.0

        self.fan.write_root_state_to_sim(fan_state, env_ids)
        self.fan.reset()

        # 清空暫存
        self.prev_actions[env_ids] =  torch.zeros_like(self.prev_actions[env_ids])
        self._compute_intermediate()
        rel = (self.ee_pos - self.fan_pos)
        self.prev_xy_dist[env_ids] = torch.linalg.norm(rel[env_ids, :2], dim=-1)

        ep_touch = self.touch_buf[env_ids].int()
        self.episode_touch_count[env_ids] = ep_touch

        self._grasp_frame_count[env_ids] = 0
        self.grasp_buf[env_ids] = False
        if hasattr(self, "grasp_success_buf"):
            self.grasp_success_buf[env_ids] = False

        # 更新全域統計（Python int）
        self.total_episodes += int(len(env_ids))
        self.total_touches += int(ep_touch.sum().item())

        # reset success flag
        self.touch_buf[env_ids] = False

        if "log" not in self.extras:
            self.extras["log"] = {}

        # (A) per-episode success for the envs that just ended (mean over reset envs)
        self.extras["log"]["touch_success_mean"] = ep_touch.float().mean()

        # (B) your global running success rate
        success_rate = 0.0 if self.total_episodes == 0 else (self.total_touches / self.total_episodes)
        self.extras["log"]["touch_rate"] = torch.tensor(success_rate, device=self.device, dtype=torch.float32)

def axis_string_to_vec(axis: str, device=None, dtype=torch.float32):
    table = {
        "+x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "+y": (0.0, 1.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "+z": (0.0, 0.0, 1.0),
        "-z": (0.0, 0.0, -1.0),
    }
    v = torch.tensor(table[axis], device=device, dtype=dtype)
    return v

def compute_obj_to_approach_R(approach_axis: str, grasp_axis: str, device=None, dtype=torch.float32):
    """
    回傳 R_obj_to_approach (3x3)，使得：
      v_approach = R_obj_to_approach @ v_object

    定義：
      approach frame +X 對齊 object 的 approach_axis
      approach frame +Y 對齊 object 的 grasp_axis
      approach frame +Z = +X × +Y (right-hand)
    """
    x_obj = axis_string_to_vec(approach_axis, device=device, dtype=dtype)  # object frame 中的向量
    y_obj = axis_string_to_vec(grasp_axis,   device=device, dtype=dtype)

    z_obj = torch.cross(x_obj, y_obj)
    z_obj = z_obj / (torch.norm(z_obj) + 1e-9)

    # columns = approach axes in object coords
    # A = [x_obj y_obj z_obj]  (3x3)
    A = torch.stack([x_obj, y_obj, z_obj], dim=1)

    # 若 columns 是 approach axes in object coords，
    # 則把 object vector 轉到 approach： v_a = A^T v_o
    R_obj_to_approach = A.T
    return R_obj_to_approach

def quat_to_rot_wxyz(q: torch.Tensor) -> torch.Tensor:
    # q: (...,4) wxyz
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = torch.stack([
        ww+xx-yy-zz, 2*(xy-wz),   2*(xz+wy),
        2*(xy+wz),   ww-xx+yy-zz, 2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),   ww-xx-yy+zz
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))
    return R
