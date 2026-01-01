# GO2 All-Terrain（全地形：瓷砖/斜坡/楼梯/水泥/草地）训练指南

面向：强化学习小白（但有 C++/ROS2 基础），目标是能跑通训练、理解代码结构、并能按真实场景逐步调参做到可上实机。

对应代码：
- 环境：`quadruped_gym/legged_gym/legged_gym/envs/go2/go2_allterrain_env.py`
- 参数：`quadruped_gym/legged_gym/legged_gym/envs/go2/go2_allterrain_config.py`
- 任务名：`go2_allterrain`（已在 `legged_gym/envs/__init__.py` 注册）

---

## 0. 设计目标（你提出的需求逐条落地）
1) **全地形**：训练场景混合“平地/斜坡/楼梯/粗糙地面（类草地）/少量离散障碍”。  
2) **显存约束**：默认 `num_observations=59`、`num_privileged_obs=251`（保留 gait phase clock，去掉重复速度历史降低输入维度）。  
3) **实机可部署**：Actor（策略网络）观测只用 IMU + 关节本体（含短历史），不依赖雷达/高度图。  
4) **交叉步（trot）美观**：奖励显式鼓励对角小跑；但不对楼梯做“硬门控”，避免因为严格步态约束导致爬楼梯学不起来。  
5) **速度分层**：平地/草地允许更快；楼梯默认只走不跑（通过指令采样的“地形类型限速”实现）。  
6) **课程学习**：  
   - 地形难度 curriculum（能走远就升难度、走不远就降难度）  
   - 指令速度 curriculum（跟踪表现好就逐步放宽速度范围）  

---

## 1. 如何运行
训练（headless 推荐）：
```bash
cd quadruped_gym/legged_gym
python legged_gym/scripts/train.py --task go2_allterrain --run_name demo --headless
```

可视化播放：
```bash
cd quadruped_gym/legged_gym
python legged_gym/scripts/play.py --task go2_allterrain --run_name demo
```

常用参数：
- `--num_envs 1024`：显存不够就先降到 512/256 让策略先学会走。
- `--seed 1`：可重复性。
- `--max_iterations`：可覆盖 config 里的 `Go2AllTerrainCfgPPO.runner.max_iterations`。

---

## 2. 频率与时序（你实机要用的“低频策略 + 高频控制”）
关键参数在 `go2_allterrain_config.py`：
- 物理步长：`sim.dt = 0.005 s`（200 Hz）
- 控制降采样：`control.decimation = 4`
- 策略频率：`policy_dt = sim.dt * decimation = 0.02 s`（50 Hz）
- Episode 时长：`episode_length_s = 24 s`
- 指令重采样：`commands.resampling_time = 10 s`

实机建议：
- 策略 50 Hz 输出关节目标（或关节增量）。
- 底层控制器 500–1000 Hz（内插 + 关节限位/速度限位/力矩限位 + 安全状态机）。

---

## 3. 观测设计（最重要：为什么是 59 / 251）
### 3.1 Actor 观测（59 维，实机可得）
在 `go2_allterrain_env.py::compute_observations()`：
```
59 = 3(角速度) + 3(重力投影) + 2(sin/cos相位) + 3(速度指令) +
     12(关节角偏差) + 12(关节速度) +
     12(a[t-1]) + 12(a[t-2])
```
含义：
- IMU：`base_ang_vel(3)` + `projected_gravity(3)`  
  - `projected_gravity` 等价于“姿态信息”，比直接用欧拉角更稳定（无万向节问题）。
- 指令：`v_x, v_y, ω_yaw`（3）
- 本体：关节角（相对默认）12、关节速度12
- 短历史：上一时刻动作、上上时刻动作  
  - 这是为了让策略在 **不看地形高度** 的前提下，仍能从“触地冲击/阻力变化/速度变化”里推断是否在爬楼梯/上坡。

### 3.2 Critic 特权观测（251 维，训练用，上帝视角）
```
251 = 59(actor_obs) + 3(base_lin_vel) + 187(高度点) + 1(摩擦系数) + 1(地形难度level)
```
目的：
- Critic 更“懂地形”，能给 Actor 更稳定的 value 估计（Asymmetric Actor-Critic）。
- Actor 不依赖高度点，实机依然可用。

---

## 4. 地形设计（如何覆盖你的真实场景）
在 `go2_allterrain_config.py::terrain`：
- `mesh_type = "trimesh"`：楼梯接触更真实（强烈建议）。
- `terrain_proportions = [0.18, 0.22, 0.42, 0.13, 0.05]`  
  对应 Terrain 生成器中的：
  - smooth slope（可覆盖：瓷砖平地/缓坡）
  - rough slope（可覆盖：草地/不平整水泥）
  - stairs（上/下楼梯：两列分别对应 step_height 正/负，比例更大以强化楼梯学习）
  - discrete obstacles（少量离散小障碍，增强鲁棒性）

楼梯尺寸说明：
- 当前仓库的 `legged_gym/utils/terrain.py` 里，楼梯 `step_width`、`step_height` 是代码写死的组合（与 difficulty 相关）。
- 如果你要严格贴合写字楼楼梯（比如 0.28m 宽、0.17m 高），建议后续把 `Terrain.make_terrain()` 里的楼梯参数改成与你真实楼梯一致（这一步对实机成功率很关键）。

---

## 5. 速度难度学习：为什么楼梯不跑、平地可冲刺
### 5.1 全局速度课程（Command Curriculum）
在 `go2_allterrain_env.py::update_command_curriculum()`：
- 当上一批 episode 的 `tracking_lin_vel` 表现足够好（> 0.8），就逐步放宽：
  - `lin_vel_x` 最大到 `commands.max_curriculum_lin_vel_x`（默认 3.0）
  - `lin_vel_y` 最大到 1.0
  - `ang_vel_yaw` 最大到 2.0

### 5.2 “按地形限速”（关键）
在 `go2_allterrain_env.py:: _resample_commands()`：
- 先从“全局速度范围”采样
- 再根据 `terrain_category` 做缩放/裁剪：
  - 楼梯：`stairs_speed_scale` + `stairs_*_clip`（默认硬上限 `v_x ≤ 0.6`，基本只走）
  - 平地/草地：缩放接近 1.0（允许更快）
  - 额外：楼梯默认 `only_forward + zero_lateral + zero_yaw`（更容易先把“上得去”学出来）

这满足你的需求：
- 平地/草地：可走/小跑/冲刺（靠 curriculum 放开上限）
- 楼梯：只需正常步行（靠限速实现）

---

## 6. 奖励函数（怎么理解 + 怎么调）
总奖励形式（重要）：  
`R = Σ (scale_i * dt * r_i)`（除 termination 外），`dt≈0.02`。

在 `go2_allterrain_config.py::rewards.scales`：
- 主任务：
  - `tracking_lin_vel`、`tracking_ang_vel`：跟踪速度指令（是你“想快就快/想慢就慢”的来源）
- 稳定性与安全：
  - `orientation`、`ang_vel_xy`、`lin_vel_z`：抑制翻滚/跳跃
  - `collision`、`stumble`、`slip`：防打腿/撞到垂直面、减少脚底打滑（瓷砖/楼梯棱边常见）
  - `base_height`：维持合理机身高度（在楼梯上也更稳定）
- 平滑与能耗：
  - `action_rate`、`dof_acc`、`torques`、`dof_vel`：抑制抖动、降低能耗（“优雅省电”的核心）
- 步态美观：
  - `trot`：鼓励对角交叉步
  - `default_hip_pos`、`default_pos`：把关节拉回中性姿态，外观更“正”
- 楼梯抬脚（但不鼓励无意义踢高）：
  - `feet_clearance`：**按地形设定摆动腿目标高度**：平地低抬腿更省电；楼梯/障碍要求更高 clearance

### 6.1 调参顺序（强烈建议照这个走）
1) **先让它在混合地形上站稳并走起来**：  
   - 降低 `max_push_vel_xy` 或关掉 `push_robots`
   - 降低速度范围（把 `max_curriculum_*` 先设小）
2) **再把步态变好看（trot 更稳定）**：  
   - 增大 `trot`（例如 0.6 → 1.0）
   - 增大 `default_hip_pos`/`default_pos`（更收敛姿态，但太大会变僵）
3) **再提升楼梯通过率**：  
   - 增大楼梯比例：`terrain_proportions` 里 stairs 的占比
   - 增大 `stumble`/`collision` 惩罚（能逼它抬脚/调整落脚）
   - 适当提高 `min_foot_clearance`（0.03 → 0.04/0.05），并把 `target_foot_clearance_stairs` 提高到 ~0.20 m
4) **最后再追求平地冲刺**：  
   - 放开 `max_curriculum_lin_vel_x`
   - 适当减小 `stairs_speed_scale` 保持“楼梯不跑”

### 6.2 奖励量级（“参数量纲/量级分析”怎么做）
你要关注的是：每项 **单步 raw r_i ∈ [0,1] 或平方项**，再乘 `scale_i*dt` 后对总奖励的贡献谁更大。
实践建议：
- 把 `extras["episode"]["rew_*"]` 打到 tensorboard（legged_gym 默认会记录），看：
  - `rew_tracking_lin_vel` 是否明显为正并主导
  - `rew_collision`、`rew_stumble` 是否在楼梯上过大（说明经常踢台阶/打腿）
  - `rew_torques`、`rew_action_rate` 是否过大（抖动/能耗）
  - `terrain_level` 是否逐步升高（地形课程在起作用）

---

## 7. 代码结构（你后续改哪里最有效）
`go2_allterrain_env.py` 里最关键的函数：
- `compute_observations()`：决定 Actor/Critic 看见什么（Sim2Real 核心）。
- `_update_terrain_category_cache()`：把 terrain column 映射成“平地/粗糙/楼梯/障碍”类别。
- `_resample_commands()`：实现“楼梯限速 + 平地可冲刺”。
- `update_command_curriculum()`：速度课程学习。
- `_reward_trot()`：交叉步约束。
- `_reward_feet_clearance()`：摆动腿不擦地（楼梯更友好）。

`go2_allterrain_config.py` 里最常改的参数块：
- `terrain.terrain_proportions`
- `commands.max_curriculum_*` 与 `stairs_*` 限速/缩放
- `rewards.scales.*`
- `domain_rand.friction_range`、`noise.noise_scales`

---

## 8. Sim2Real / ROS2 工程化建议（落地经验）
1) **观测对齐**：  
   - IMU 角速度：用 body frame（与仿真一致）
   - `projected_gravity`：由姿态四元数得到（或用 IMU AHRS）
   - 关节角/关节速度：与 URDF 顺序一致（很关键）
2) **动作对齐**：  
   - 训练中 `action_scale=0.25` 表示“目标角偏移（rad）”  
   - 实机要做：`q_target = q_default + action * action_scale`，再由底层 PD 跟踪
3) **安全与实时**：  
   - 在 ROS2 节点里做：关节限位、速度限位、姿态过倾保护、急停
   - 策略输出频率建议 50 Hz；底层插值到 500–1000 Hz
4) **域随机化**：  
   - 先用 `friction_range` + `added_mass_range` 跑通  
   - 后续想更强：可以继续加入电机延迟/观测延迟、Kp/Kd 随机、关节零偏等（你开源 stairs 代码里有相关思路）

---

## 9. 常见问题与快速定位
- **楼梯上总是踢台阶/上不去**：  
  - 增大 `stumble`/`collision` 惩罚
  - 提高 `min_foot_clearance`
  - 降低楼梯速度上限（`stairs_lin_vel_x_clip`）
  - 增大楼梯占比（`terrain_proportions`）
- **平地抬腿太高、不优雅**：  
  - 降低 `feet_clearance` 奖励
  - 增大 `torques` / `action_rate` / `default_pos` 惩罚（让它更省电更收敛）
- **走路抖动大**：  
  - 增大 `action_rate` 和 `dof_acc` 惩罚
  - 降低 `action_scale`（动作幅度）
- **训练很慢/不收敛**：  
  - 先关 `push_robots`
  - 降低地形难度（`max_init_terrain_level`）
  - 先把速度范围缩小（`max_curriculum_*` 设小）

---

如果你希望我下一步继续帮你做“更贴合写字楼楼梯尺寸的 Terrain 生成器”（把楼梯 step width/height 变成可配置），或者把你开源 stairs 里的“观测/动作延迟随机化”移植到这个 allterrain 环境里，我可以直接在仓库里改好并给出建议的实机部署参数。
