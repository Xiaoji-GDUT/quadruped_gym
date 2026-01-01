import numpy as np
import torch

from isaacgym.torch_utils import quat_apply, quat_rotate_inverse, torch_rand_float
from isaacgym import gymtorch
import math

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.go2.go2_allterrain_config import Go2AllTerrainCfg

def wrap_to_pi(angles):
    """
    将角度张量限制到 [-pi, pi]
    支持 torch Tensor
    """
    return (angles + math.pi) % (2 * math.pi) - math.pi

class Go2AllTerrainRobot(LeggedRobot):
    """GO2 all-terrain locomotion env.

    Design goals (sim2real friendly):
    - Actor obs: only IMU + joint proprioception (+ short history).
    - Critic obs: privileged terrain heights + (optional) friction + curriculum level.
    - Terrain: mixed tile-like flat/slope + stairs + rough (grass-like).
    - Commands: speed curriculum + terrain-type-dependent speed limits (stairs only walk).
    - Gait: diagonal trot encouraged, but not hard-gated for stairs robustness.
    """

    cfg: Go2AllTerrainCfg

    # ---- Core loop overrides -------------------------------------------------
    def post_physics_step(self):
        # refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # base state in body frame
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute rewards, resets, observations
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        # update history
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """Add tilt-based termination to speed up learning on stairs/rough terrain."""
        super().check_termination()
        # tilt: projected_gravity z-component encodes cos(theta); -z≈1 when upright
        tilt = torch.acos(torch.clamp(-self.projected_gravity[:, 2], -1.0, 1.0))
        self.reset_buf |= tilt > float(getattr(self.cfg.rewards, "termination_tilt", 1.0))

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # terrain curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # command curriculum (global, evaluated at episode boundaries)
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # logging
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"][f"rew_{key}"] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0

        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = float(self.command_ranges["lin_vel_x"][1])
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    # ---- Observations --------------------------------------------------------
    def compute_observations(self):
        # 59D actor obs (sim2real friendly): imu + joint + short action history + gait phase (clock)
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        # gait phase clock (same idea as go2_trot_env.py)
        phase = self._get_phase()
        sin_pos = torch.sin(2 * math.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * math.pi * phase).unsqueeze(1)

        obs = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.projected_gravity,  # 3
                sin_pos,  # 1
                cos_pos,  # 1
                self.commands[:, :3] * self.commands_scale,  # 3
                q,  # 12
                dq,  # 12
                self.actions,  # 12 (t-1)
                self.last_actions,  # 12 (t-2)
            ),
            dim=-1,
        )
        self.obs_buf = obs

        # 251D privileged obs: actor obs + base lin vel + local height samples + (friction, terrain_level)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1.0, 1.0)
            heights = heights * self.obs_scales.height_measurements
        else:
            heights = torch.zeros(self.num_envs, self.num_height_points, device=self.device, dtype=torch.float)

        friction = self._get_friction_tensor()
        terrain_level = self._get_terrain_level_tensor()

        self.privileged_obs_buf = torch.cat(
            (
                self.obs_buf,  # 59
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                heights,  # 187
                friction,  # 1
                terrain_level,  # 1
            ),
            dim=-1,
        )

        if self.add_noise:
            self.obs_buf = self.obs_buf + (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        # Must match compute_observations() layout (59D)
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # base_ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level  # projected_gravity
        noise_vec[6:8] = 0.0  # sin/cos (clock)
        noise_vec[8:11] = 0.0  # commands
        noise_vec[11:23] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos  # dof_pos
        noise_vec[23:35] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # dof_vel
        noise_vec[35:47] = 0.0  # actions
        noise_vec[47:59] = 0.0  # last_actions
        return noise_vec

    def _get_friction_tensor(self) -> torch.Tensor:
        # Stored on CPU in the base implementation; keep a cached GPU copy for obs
        if hasattr(self, "friction_coeffs_tensor"):
            friction = self.friction_coeffs_tensor
        elif hasattr(self, "friction_coeffs"):
            friction = self.friction_coeffs.to(self.device)
        else:
            friction = torch.full((self.num_envs, 1), float(self.cfg.terrain.static_friction), device=self.device)
        return friction.view(self.num_envs, 1)

    def _get_terrain_level_tensor(self) -> torch.Tensor:
        if hasattr(self, "terrain_levels") and hasattr(self, "max_terrain_level") and self.max_terrain_level > 0:
            return (self.terrain_levels.float() / float(self.max_terrain_level)).view(self.num_envs, 1)
        return torch.zeros(self.num_envs, 1, device=self.device)

    # ---- Terrain helpers -----------------------------------------------------
    def _get_env_origins(self):
        super()._get_env_origins()
        self._update_terrain_category_cache()

    def _update_terrain_category_cache(self):
        """Precompute terrain category from terrain_types (column index).

        Categories:
        0: smooth (tile/flat/slope)
        1: rough (grass-like)
        2: stairs (up/down)
        3: discrete obstacles/others
        """
        self.terrain_category = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            return

        props = np.cumsum(self.cfg.terrain.terrain_proportions).tolist()
        if len(props) < 4:
            return

        choice = self.terrain_types.float() / float(self.cfg.terrain.num_cols) + 0.001
        p0, p1, p2, p3 = props[0], props[1], props[2], props[3]
        self.terrain_category[:] = 3  # default
        self.terrain_category[choice < p0] = 0
        self.terrain_category[(choice >= p0) & (choice < p1)] = 1
        self.terrain_category[(choice >= p1) & (choice < p3)] = 2  # stairs band

    # ---- Command logic (speed curriculum + per-terrain limits) ---------------
    def _post_physics_step_callback(self):
        # resample commands
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        # optional heading command
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        # measure terrain heights (privileged)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # random pushes
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            # avoid pushes on stairs to stabilize early stair learning
            if hasattr(self, "terrain_category"):
                stair_mask = self.terrain_category == 2
                allow_stairs = (self.common_step_counter * self.dt) > float(self.cfg.domain_rand.stairs_push_after_s)
                target_envs = torch.arange(self.num_envs, device=self.device)[~stair_mask | allow_stairs]
                if len(target_envs) > 0:
                    max_vel = self.cfg.domain_rand.max_push_vel_xy
                    self.root_states[target_envs, 7:9] = torch_rand_float(-max_vel, max_vel, (len(target_envs), 2), device=self.device)
                    self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            else:
                self._push_robots()

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        # sample from global curriculum ranges
        cmd_x = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        cmd_y = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        cmd_yaw = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # terrain-type dependent speed scaling (stairs only walk)
        if hasattr(self, "terrain_category"):
            cat = self.terrain_category[env_ids]
            scale_xy = torch.ones(len(env_ids), device=self.device)
            scale_yaw = torch.ones(len(env_ids), device=self.device)

            stairs = cat == 2
            rough = cat == 1
            slope = cat == 0
            discrete = cat == 3

            scale_xy[stairs] = float(self.cfg.commands.stairs_speed_scale)
            scale_yaw[stairs] = float(self.cfg.commands.stairs_yaw_scale)

            scale_xy[rough] = float(self.cfg.commands.rough_speed_scale)
            scale_yaw[rough] = float(self.cfg.commands.rough_yaw_scale)

            scale_xy[slope] = float(self.cfg.commands.slope_speed_scale)
            scale_yaw[slope] = float(self.cfg.commands.slope_yaw_scale)

            scale_xy[discrete] = float(self.cfg.commands.discrete_speed_scale)
            scale_yaw[discrete] = float(self.cfg.commands.discrete_yaw_scale)

            cmd_x = cmd_x * scale_xy
            cmd_y = cmd_y * scale_xy
            cmd_yaw = cmd_yaw * scale_yaw

            # hard caps for stairs (no sprinting)
            cmd_x[stairs] = torch.clamp(cmd_x[stairs], float(self.cfg.commands.stairs_lin_vel_x_clip[0]), float(self.cfg.commands.stairs_lin_vel_x_clip[1]))
            cmd_y[stairs] = torch.clamp(cmd_y[stairs], float(self.cfg.commands.stairs_lin_vel_y_clip[0]), float(self.cfg.commands.stairs_lin_vel_y_clip[1]))
            cmd_yaw[stairs] = torch.clamp(cmd_yaw[stairs], float(self.cfg.commands.stairs_ang_vel_yaw_clip[0]), float(self.cfg.commands.stairs_ang_vel_yaw_clip[1]))

            # simplify stair commands to ease learning (go straight, no side-stepping/spinning)
            if bool(getattr(self.cfg.commands, "stairs_zero_lateral", True)):
                cmd_y[stairs] = 0.0
            if bool(getattr(self.cfg.commands, "stairs_zero_yaw", True)):
                cmd_yaw[stairs] = 0.0
            if bool(getattr(self.cfg.commands, "stairs_only_forward", True)):
                cmd_x[stairs] = torch.clamp(cmd_x[stairs], min=0.0)

        self.commands[env_ids, 0] = cmd_x
        self.commands[env_ids, 1] = cmd_y
        self.commands[env_ids, 2] = cmd_yaw

        # set small commands to zero (stand-still)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def update_command_curriculum(self, env_ids):
        # More conservative curriculum than the default (helps stairs stability).
        if "tracking_lin_vel" not in self.episode_sums or "tracking_lin_vel" not in self.reward_scales:
            return

        tracking = torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
        if tracking < float(self.cfg.commands.curriculum_success_thresh) * self.reward_scales["tracking_lin_vel"]:
            return

        # expand forward range
        step_x = float(self.cfg.commands.curriculum_step_lin_vel_x)
        self.command_ranges["lin_vel_x"][0] = float(
            np.clip(self.command_ranges["lin_vel_x"][0] - step_x, -self.cfg.commands.max_curriculum_lin_vel_x, 0.0)
        )
        self.command_ranges["lin_vel_x"][1] = float(
            np.clip(self.command_ranges["lin_vel_x"][1] + step_x, 0.0, self.cfg.commands.max_curriculum_lin_vel_x)
        )

        # expand lateral and yaw more slowly
        step_y = float(self.cfg.commands.curriculum_step_lin_vel_y)
        self.command_ranges["lin_vel_y"][0] = float(
            np.clip(self.command_ranges["lin_vel_y"][0] - step_y, -self.cfg.commands.max_curriculum_lin_vel_y, 0.0)
        )
        self.command_ranges["lin_vel_y"][1] = float(
            np.clip(self.command_ranges["lin_vel_y"][1] + step_y, 0.0, self.cfg.commands.max_curriculum_lin_vel_y)
        )

        step_yaw = float(self.cfg.commands.curriculum_step_ang_vel_yaw)
        self.command_ranges["ang_vel_yaw"][0] = float(
            np.clip(self.command_ranges["ang_vel_yaw"][0] - step_yaw, -self.cfg.commands.max_curriculum_ang_vel_yaw, 0.0)
        )
        self.command_ranges["ang_vel_yaw"][1] = float(
            np.clip(self.command_ranges["ang_vel_yaw"][1] + step_yaw, 0.0, self.cfg.commands.max_curriculum_ang_vel_yaw)
        )

    # ---- Domain randomization hooks -----------------------------------------
    def _process_rigid_shape_props(self, props, env_id):
        # keep base friction randomization, but cache a GPU copy for privileged obs
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1), device="cpu")
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device="cpu")
                self.friction_coeffs = friction_buckets[bucket_ids]
                self.friction_coeffs_tensor = self.friction_coeffs.to(self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    # ---- Gait utilities + rewards -------------------------------------------
    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        return (self.episode_length_buf * self.dt) % cycle_time / cycle_time

    def _get_gait_phase(self):
        # diagonal trot: (FL, RR) vs (FR, RL)
        phase = self._get_phase()
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = phase < 0.5
        stance_mask[:, 1] = phase >= 0.5
        return stance_mask

    def _compute_trot_agreement(self):
        """Soft trot agreement in [0, 1].

        Uses the same clocked diagonal trot pattern as go2_trot_env.py, but returns a dense signal
        (fraction of feet matching expected stance) to be more robust on mixed terrains.
        """
        # Need to filter contacts because PhysX contact reporting is unreliable on meshes.
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.contact_force_threshold
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        stance_mask = self._get_gait_phase().to(torch.bool)

        # assumes feet order: [FL, FR, RL, RR]
        desired_diag0 = stance_mask[:, 0]
        desired_diag1 = stance_mask[:, 1]

        agree = (
            (contact_filt[:, 0] == desired_diag0).to(torch.float)
            + (contact_filt[:, 3] == desired_diag0).to(torch.float)
            + (contact_filt[:, 1] == desired_diag1).to(torch.float)
            + (contact_filt[:, 2] == desired_diag1).to(torch.float)
        ) / 4.0

        moving = torch.norm(self.commands[:, :2], dim=1) > 0.1
        return agree * moving.to(torch.float)

    # ---- Reward plumbing (ordered, easier to read/tune) ---------------------
    def _prepare_reward_function(self):
        """Order-sensitive reward list (task -> stability -> style -> costs)."""
        # remove zero scales + scale by dt (same behavior as base)
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        preferred_order = [
            "tracking_lin_vel",
            "tracking_ang_vel",
            "trot",
            "feet_clearance",
            "orientation",
            "base_height",
            "lin_vel_z",
            "ang_vel_xy",
            "collision",
            "stumble",
            "slip",
            "stand_still",
            "feet_contact_forces",
            "torques",
            "dof_vel",
            "dof_acc",
            "action_rate",
            "default_hip_pos",
            "default_pos",
        ]
        ordered = preferred_order + [n for n in self.reward_scales.keys() if n not in preferred_order]

        self.reward_functions = []
        self.reward_names = []
        for name in ordered:
            if name == "termination" or name not in self.reward_scales:
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, f"_reward_{name}"))

        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def _reward_trot(self):
        # Strong constraint on flat/rough, weak on stairs
        agree = self._compute_trot_agreement()
        if hasattr(self, "terrain_category"):
            stairs = self.terrain_category == 2
            agree = torch.where(stairs, agree * float(self.cfg.rewards.trot_stairs_scale), agree)
        return agree

    def _trot_tracking_gate(self):
        """Gate tracking rewards by trot quality on non-stairs terrains.

        - Flat/rough/discrete: tracking reward is reduced unless trot agreement is high.
        - Stairs: no gating (prioritize climbing robustness over aesthetic gait).
        """
        if not hasattr(self, "terrain_category"):
            return torch.ones(self.num_envs, device=self.device, dtype=torch.float)

        stairs = self.terrain_category == 2
        moving = torch.norm(self.commands[:, :2], dim=1) > 0.1
        apply_gate = (~stairs) & moving
        if not torch.any(apply_gate):
            return torch.ones(self.num_envs, device=self.device, dtype=torch.float)

        agree = self._compute_trot_agreement()
        min_gate = float(self.cfg.rewards.tracking_trot_gate_min)

        gate = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        power = float(getattr(self.cfg.rewards, "tracking_trot_gate_power", 1.0))
        gate[apply_gate] = min_gate + (1.0 - min_gate) * torch.pow(agree[apply_gate], power)
        return gate

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        rew = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return rew * self._trot_tracking_gate()

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
        return rew * self._trot_tracking_gate()

    def _reward_slip(self):
        """Penalize foot tangential speed while in contact (reduces skating on tiles/stairs edges)."""
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.contact_force_threshold
        if not torch.any(contact):
            return torch.zeros(self.num_envs, device=self.device)

        # rigid_state[..., 7:10] holds linear velocity in world frame
        foot_lin_vel = self.rigid_state[:, self.feet_indices, 7:10]
        tangential = torch.linalg.norm(foot_lin_vel[:, :, :2], dim=2)
        slip = tangential * contact

        # average over feet; zero out swing legs
        return torch.sum(slip, dim=1) / (torch.sum(contact, dim=1) + 1e-3)

    def _reward_lin_vel_z(self):
        """Penalize vertical velocity; relaxed on stairs/obstacles to avoid over-constraining climbs."""
        penalty = torch.square(self.base_lin_vel[:, 2])
        if hasattr(self, "terrain_category"):
            scale = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
            scale[self.terrain_category == 2] = 0.5  # stairs need more vertical motion
            scale[self.terrain_category == 3] = 0.7  # discrete obstacles
            penalty = penalty * scale
        return penalty

    def _reward_orientation(self):
        # Penalize non-flat base orientation (stronger on smooth, weaker on stairs).
        pitch_w = float(getattr(self.cfg.rewards, "orientation_pitch_weight", 1.0))
        roll_w = float(getattr(self.cfg.rewards, "orientation_roll_weight", 1.0))
        err = pitch_w * torch.square(self.projected_gravity[:, 0]) + roll_w * torch.square(self.projected_gravity[:, 1])

        if not hasattr(self, "terrain_category"):
            return err

        scale = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        scale[self.terrain_category == 0] = float(getattr(self.cfg.rewards, "orientation_scale_smooth", 1.0))
        scale[self.terrain_category == 1] = float(getattr(self.cfg.rewards, "orientation_scale_rough", 1.0))
        scale[self.terrain_category == 2] = float(getattr(self.cfg.rewards, "orientation_scale_stairs", 1.0))
        scale[self.terrain_category == 3] = float(getattr(self.cfg.rewards, "orientation_scale_discrete", 1.0))
        return err * scale

    def _get_heights_at_points(self, points_world: torch.Tensor, mode: str = "min") -> torch.Tensor:
        """Sample terrain height at arbitrary world XY points.

        Args:
            points_world: (num_envs, P, 3) world positions.
            mode: "min" (conservative for stability) or "max" (conservative for obstacle clearance).
        Returns:
            heights: (num_envs, P) in meters.
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(points_world.shape[0], points_world.shape[1], device=self.device, requires_grad=False)
        if self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = points_world.clone()
        points[..., :2] += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()

        px = points[..., 0].view(-1)
        py = points[..., 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        if mode == "min":
            heights = torch.min(torch.min(heights1, heights2), heights3)
        elif mode == "max":
            heights = torch.max(torch.max(heights1, heights2), heights3)
        else:
            raise ValueError(f"Unknown height sampling mode: {mode}")
        heights = heights.view(points_world.shape[0], -1) * self.terrain.cfg.vertical_scale
        return heights

    def _reward_feet_clearance(self):
        """Terrain-aware swing height shaping.

        Goals:
        - Smooth/rough: keep steps low and symmetric (nice/economical gait).
        - Stairs/obstacles: encourage higher swing clearance (avoid catching the riser/edge).
        """
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            return torch.zeros(self.num_envs, device=self.device)

        moving = torch.norm(self.commands[:, :2], dim=1) > 0.1
        if not torch.any(moving):
            return torch.zeros(self.num_envs, device=self.device)

        # foot world positions (num_envs, 4, 3)
        feet_pos = self.rigid_state[:, self.feet_indices, 0:3]
        terrain_h_min = self._get_heights_at_points(feet_pos, mode="min")
        terrain_h_max = self._get_heights_at_points(feet_pos, mode="max")

        # Use a more conservative (higher) terrain estimate for stairs/obstacles to encourage enough clearance.
        use_max = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if hasattr(self, "terrain_category"):
            use_max = (self.terrain_category == 2) | (self.terrain_category == 3)
        terrain_h = torch.where(use_max.unsqueeze(1), terrain_h_max, terrain_h_min)

        clearance = feet_pos[:, :, 2] - terrain_h - float(self.cfg.rewards.foot_radius)

        # diagonal pairs: diag0=(FL, RR), diag1=(FR, RL) (same as go2_trot_env.py)
        diag0 = clearance[:, [0, 3]]  # (N,2)
        diag1 = clearance[:, [1, 2]]  # (N,2)

        stance_mask = self._get_gait_phase()  # (N,2) in {0,1}
        swing_mask = 1.0 - stance_mask

        # swing height clock (0 at liftoff/touchdown, 1 at mid-swing)
        phase = self._get_phase()
        swing_profile = torch.abs(torch.sin(2 * math.pi * phase))  # (N,)

        # per-terrain target swing clearance (peak value, meters)
        target = torch.full((self.num_envs,), float(self.cfg.rewards.target_foot_clearance_smooth), device=self.device)
        if hasattr(self, "terrain_category"):
            target[self.terrain_category == 1] = float(self.cfg.rewards.target_foot_clearance_rough)
            target[self.terrain_category == 2] = float(self.cfg.rewards.target_foot_clearance_stairs)
            target[self.terrain_category == 3] = float(self.cfg.rewards.target_foot_clearance_discrete)

            # difficulty-adaptive boost (helps when terrain_level increases)
            level_scale = float(getattr(self.cfg.rewards, "foot_clearance_level_scale", 0.0))
            if level_scale > 0.0 and hasattr(self, "max_terrain_level") and self.max_terrain_level > 0:
                level_norm = (self.terrain_levels.float() / float(self.max_terrain_level)).clamp(0.0, 1.0)
                target = target + level_scale * level_norm

        target_traj = (swing_profile * target).unsqueeze(1)  # (N,1)

        # terrain-dependent asymmetric penalty:
        # - always penalize being below target during swing
        # - penalize being above target only on smooth/rough (keep gait low & tidy)
        under_w = torch.full((self.num_envs,), float(self.cfg.rewards.foot_clearance_under_weight_smooth), device=self.device)
        over_w = torch.full((self.num_envs,), float(self.cfg.rewards.foot_clearance_over_weight_smooth), device=self.device)
        if hasattr(self, "terrain_category"):
            under_w[self.terrain_category == 1] = float(self.cfg.rewards.foot_clearance_under_weight_rough)
            under_w[self.terrain_category == 2] = float(self.cfg.rewards.foot_clearance_under_weight_stairs)
            under_w[self.terrain_category == 3] = float(self.cfg.rewards.foot_clearance_under_weight_discrete)

            over_w[self.terrain_category == 1] = float(self.cfg.rewards.foot_clearance_over_weight_rough)
            over_w[self.terrain_category == 2] = float(self.cfg.rewards.foot_clearance_over_weight_stairs)
            over_w[self.terrain_category == 3] = float(self.cfg.rewards.foot_clearance_over_weight_discrete)

        err0_under = (target_traj - diag0).clamp(min=0.0)
        err1_under = (target_traj - diag1).clamp(min=0.0)
        err0_over = (diag0 - target_traj).clamp(min=0.0)
        err1_over = (diag1 - target_traj).clamp(min=0.0)

        alpha = float(getattr(self.cfg.rewards, "foot_clearance_exp_scale", 10.0))
        pen0 = torch.sum(under_w.unsqueeze(1) * err0_under + over_w.unsqueeze(1) * err0_over, dim=1)
        pen1 = torch.sum(under_w.unsqueeze(1) * err1_under + over_w.unsqueeze(1) * err1_over, dim=1)

        rew0 = torch.exp(-alpha * pen0) * swing_mask[:, 0]
        rew1 = torch.exp(-alpha * pen1) * swing_mask[:, 1]
        rew = rew0 + rew1
        return rew * moving.to(torch.float)

    def _reward_default_hip_pos(self):
        # keep hip joints close to neutral for nicer posture
        hip_ids = torch.tensor([0, 3, 6, 9], device=self.device, dtype=torch.long)
        return torch.sum(torch.abs(self.dof_pos[:, hip_ids]), dim=1)

    def _reward_default_pos(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    # ---- Buffer init extension ----------------------------------------------
    def _init_buffers(self):
        super()._init_buffers()

        # rigid body state tensor for feet clearance reward
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)

        # cache gpu friction tensor (if base friction randomization was used)
        if hasattr(self, "friction_coeffs") and not hasattr(self, "friction_coeffs_tensor"):
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device)

    def _parse_cfg(self, cfg):
        # keep base parsing
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
