# # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: BSD-3-Clause
# #
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # 1. Redistributions of source code must retain the above copyright notice, this
# # list of conditions and the following disclaimer.
# #
# # 2. Redistributions in binary form must reproduce the above copyright notice,
# # this list of conditions and the following disclaimer in the documentation
# # and/or other materials provided with the distribution.
# #
# # 3. Neither the name of the copyright holder nor the names of its
# # contributors may be used to endorse or promote products derived from
# # this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# #
# # Copyright (c) 2021 ETH Zurich, Nikita Rudin

# from legged_gym import LEGGED_GYM_ROOT_DIR
# import os

# import isaacgym
# from legged_gym.envs import *
# from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, export_policy_as_onnx

# import numpy as np
# import torch


# def play(args):
#     env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
#     # override some parameters for testing
#     env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
#     env_cfg.terrain.num_rows = 5
#     env_cfg.terrain.num_cols = 5
#     env_cfg.terrain.curriculum = False
#     env_cfg.noise.add_noise = False
#     env_cfg.domain_rand.randomize_friction = False
#     env_cfg.domain_rand.push_robots = False

#     # prepare environment
#     env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
#     print(env.dof_names)
#     obs = env.get_observations()
#     # load policy
#     train_cfg.runner.resume = True
#     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
#     policy = ppo_runner.get_inference_policy(device=env.device)

#     # export policy as a jit module (used to run it from C++)
#     if EXPORT_POLICY:
#         path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
#         export_policy_as_jit(ppo_runner.alg.actor_critic, path)
#         # export_policy_as_onnx(ppo_runner.alg.actor_critic, path)
#         print('Exported policy as jit script to: ', path)

#     logger = Logger(env.dt)
#     robot_index = 0 # which robot is used for logging
#     joint_index = 1 # which joint is used for logging
#     stop_state_log = 100 # number of steps before plotting states
#     stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
#     camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
#     camera_vel = np.array([1., 1., 0.])
#     camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
#     img_idx = 0

#     for i in range(10*int(env.max_episode_length)):
#         actions = policy(obs.detach())
#         obs, _, rews, dones, infos = env.step(actions.detach())
#         if RECORD_FRAMES:
#             if i % 2:
#                 filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
#                 env.gym.write_viewer_image_to_file(env.viewer, filename)
#                 img_idx += 1
#         if MOVE_CAMERA:
#             camera_position += camera_vel * env.dt
#             env.set_camera(camera_position, camera_position + camera_direction)

#         if i < stop_state_log:
#             logger.log_states(
#                 {
#                     'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
#                     'dof_pos': env.dof_pos[robot_index, joint_index].item(),
#                     'dof_vel': env.dof_vel[robot_index, joint_index].item(),
#                     'dof_torque': env.torques[robot_index, joint_index].item(),
#                     'command_x': env.commands[robot_index, 0].item(),
#                     'command_y': env.commands[robot_index, 1].item(),
#                     'command_yaw': env.commands[robot_index, 2].item(),
#                     'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
#                     'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
#                     'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
#                     'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
#                     'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
#                 }
#             )
#         elif i==stop_state_log:
#             logger.plot_states()
#         if  0 < i < stop_rew_log:
#             if infos["episode"]:
#                 num_episodes = torch.sum(env.reset_buf).item()
#                 if num_episodes>0:
#                     logger.log_rewards(infos["episode"], num_episodes)
#         elif i==stop_rew_log:
#             logger.print_rewards()

# if __name__ == '__main__':
#     EXPORT_POLICY = True
#     RECORD_FRAMES = False
#     MOVE_CAMERA = False
#     args = get_args()
#     play(args)

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from isaacgym import gymapi
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import torch

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # 覆盖测试参数
    # 这里保持和命令行一致，或者手动限制
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # 1. 准备环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # 2. 注册按键订阅
    if env.viewer is not None:
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_W, "forward")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_S, "backward")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_A, "left")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_D, "right")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_Q, "turn_left")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_E, "turn_right")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_L, "stop")

    # 3. 加载策略
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # --- 修复点：动态获取 env.commands 的维度 (可能是 3 也有可能是 4) ---
    num_commands = env.commands.shape[1]
    my_commands = torch.zeros(env.num_envs, num_commands, device=env.device)

    lin_speed_step = 0.1
    ang_speed_step = 0.2

    # --- 设置初始全局视角 ---
    if env.viewer is not None:
        cam_pos = gymapi.Vec3(10.0, 10.0, 10.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)

    print("\n" + "="*50)
    print(f"全局指令模式 [维度: {num_commands}] 已启动:")
    print("  W / S : 前进 / 后退")
    print("  A / D : 左移 / 右移")
    print("  Q / E : 左转 / 右转")
    print("  L     : 停止 (清零)")
    print("-" * 50)
    print("提示：已自动适配指令维度，环境随机重置已被屏蔽。")
    print("="*50 + "\n")

    for i in range(10*int(env.max_episode_length)):

        # 4. 获取并处理 Action 事件
        if env.viewer is not None:
            actions_evts = env.gym.query_viewer_action_events(env.viewer)
            for evt in actions_evts:
                if evt.value > 0: # 按键按下
                    # 只操作前三个维度 (x, y, yaw)
                    if evt.action == "forward":   my_commands[:, 0] += lin_speed_step
                    if evt.action == "backward":  my_commands[:, 0] -= lin_speed_step
                    if evt.action == "left":      my_commands[:, 1] += lin_speed_step
                    if evt.action == "right":     my_commands[:, 1] -= lin_speed_step
                    if evt.action == "turn_left": my_commands[:, 2] += ang_speed_step
                    if evt.action == "turn_right":my_commands[:, 2] -= ang_speed_step
                    if evt.action == "stop":       my_commands[:, :] = 0.0

                    # 实时显示
                    print(f"目标指令: x={my_commands[0,0]:.2f}, y={my_commands[0,1]:.2f}, yaw={my_commands[0,2]:.2f}      ", end='\r')

        # 5. 【核心修复】强制覆盖指令，确保形状完全一致
        env.commands[:, :] = my_commands[:, :]

        # 6. 执行推理与物理仿真
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    args = get_args()
    play(args)
