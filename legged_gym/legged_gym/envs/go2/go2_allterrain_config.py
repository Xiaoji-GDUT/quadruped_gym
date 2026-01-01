from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2AllTerrainCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # Notes: this task adds gait phase (sin/cos) to make trot constraints learnable.
        num_envs = 1024
        num_observations = 59
        num_privileged_obs = 251
        num_actions = 12
        episode_length_s = 24
        env_spacing = 3.0
        send_timeouts = True

    class terrain(LeggedRobotCfg.terrain):
        # 使用 trimesh 以获得更精确的接触检测
        mesh_type = "trimesh"
        curriculum = True
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]

        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

        # privileged terrain sensing (critic only)
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 1   # start easier to let stair policy emerge
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # difficulty levels
        num_cols = 20  # terrain types (columns)

        # 【平滑地形，粗糙地形，楼梯上坡，楼梯下坡，离散障碍物】
        terrain_proportions = [0.18, 0.22, 0.42, 0.13, 0.05]  # more stair exposure early
        slope_treshold = 0.75

    class commands(LeggedRobotCfg.commands):
        # global command curriculum (expanded by env.update_command_curriculum)
        curriculum = True   # 开启速度课程学习
        max_curriculum = 3.0
        num_commands = 4
        resampling_time = 10.0  # 多久重新随机一次命令
        heading_command = False

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.8, 0.8]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-0.6, 0.6]
            heading = [-3.14, 3.14]

        # curriculum maxima (used by Go2AllTerrainRobot.update_command_curriculum)
        max_curriculum_lin_vel_x = 2.0
        max_curriculum_lin_vel_y = 0.6
        max_curriculum_ang_vel_yaw = 1.5

        # curriculum update speed (used by Go2AllTerrainRobot.update_command_curriculum)
        curriculum_success_thresh = 0.9
        curriculum_step_lin_vel_x = 0.15
        curriculum_step_lin_vel_y = 0.10
        curriculum_step_ang_vel_yaw = 0.10

        # 在不同地形上的速度缩放
        # 楼梯限幅
        stairs_speed_scale = 0.45
        stairs_yaw_scale = 0.6
        # 斜坡
        slope_speed_scale = 0.85
        slope_yaw_scale = 0.9
        # 粗糙地形
        rough_speed_scale = 1.0
        rough_yaw_scale = 1.0
        # 离散地形
        discrete_speed_scale = 0.7
        discrete_yaw_scale = 0.8

        # 楼梯上的硬性速度限制，防止机器人飞出楼梯
        stairs_lin_vel_x_clip = [0.0, 0.6]
        stairs_lin_vel_y_clip = [-0.05, 0.05]
        stairs_ang_vel_yaw_clip = [-0.2, 0.2]

        # 训练阶段默认让楼梯指令更简单（更容易学会爬楼梯）
        stairs_only_forward = True
        stairs_zero_lateral = True
        stairs_zero_yaw = True

    class init_state:
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control:
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]    # 惩罚
        terminate_after_contacts_on = ["base"]              # 终止
        disable_gravity = False
        collapse_fixed_joints = False
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.25, 1.25]

        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]

        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6
        # allow pushes on stairs only after some training time to avoid early instability
        stairs_push_after_s = 4000  # policy steps * dt (approx seconds of experience)

        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = True
        added_base_com_range = [-0.03, 0.03]

    class rewards(LeggedRobotCfg.rewards):
        # gait clock
        cycle_time = 0.5
        # feet/contacts
        contact_force_threshold = 5.0
        foot_radius = 0.02
        min_foot_clearance = 0.04
        # gait shaping (flat strong, stairs weak)
        tracking_trot_gate_min = 0.3
        tracking_trot_gate_power = 2.0
        trot_stairs_scale = 0.2

        # posture shaping (use env-side terrain-dependent scaling)
        orientation_pitch_weight = 2.0
        orientation_roll_weight = 1.0
        orientation_scale_smooth = 5.0
        orientation_scale_rough = 3.0
        orientation_scale_stairs = 0.6
        orientation_scale_discrete = 4.0

        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.32
        max_contact_force = 100.0

        # swing-foot clearance shaping (terrain-aware)
        foot_clearance_exp_scale = 18.0
        foot_clearance_level_scale = 0.02

        # 不同地形的目标足部高度
        target_foot_clearance_smooth = 0.05 # 平滑地面
        target_foot_clearance_rough = 0.09  # 粗糙地面
        target_foot_clearance_stairs = 0.20 # 楼梯
        target_foot_clearance_discrete = 0.11 # 离散障碍物

        foot_clearance_under_weight_smooth = 1.0
        foot_clearance_under_weight_rough = 1.2
        foot_clearance_under_weight_stairs = 3.0
        foot_clearance_under_weight_discrete = 2.0

        foot_clearance_over_weight_smooth = 1.0
        foot_clearance_over_weight_rough = 0.4
        foot_clearance_over_weight_stairs = 0.0
        foot_clearance_over_weight_discrete = 0.0

        termination_tilt = 0.9  # [rad] terminate early when roll/pitch too large

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -0.0

            # task
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.5

            # stability
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.2      # 姿态旋转
            base_height = -3.0      # 约束高度

            # smoothness / energy
            torques = -0.0001
            dof_vel = -1.0e-4
            dof_acc = -2.5e-7
            action_rate = -0.01

            # contacts / safety
            collision = -1.0
            stumble = -0.6
            slip = -0.5
            feet_contact_forces = -0.0
            stand_still = -1.0

            # style / gait
            trot = 0.6
            feet_clearance = 0.2
            default_hip_pos = -0.2
            default_pos = -0.1
            feet_air_time = 0.0

    class normalization(LeggedRobotCfg.normalization):
        clip_observations = 100.0
        clip_actions = 100.0

        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11.0, 5, 3.0]


class Go2AllTerrainCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.7
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # stable defaults for all-terrain
        entropy_coef = 0.005
        learning_rate = 1.0e-3
        num_learning_epochs = 5
        num_mini_batches = 4
        desired_kl = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 50000

        save_interval = 100
        experiment_name = "go2_allterrain"
        run_name = ""
