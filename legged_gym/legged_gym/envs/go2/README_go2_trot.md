# GO2 Trot 训练学习指南

面向：第一次接触强化学习的同学，结合 `go2_trot` 代码快速理解并能举一反三到其它 GO2 任务。
参考代码版本：`quadruped_gym/legged_gym/legged_gym/envs/go2/go2_trot_env.py` 与 `go2_trot_config.py`。

---

## 1. 四个 GO2 任务各训练什么
- `go2_flat`：在平地行走，观察量更丰富（步态参数、历史动作、地形高度），用于通用平地奔跑。
- `go2_rough`：粗糙/起伏地形，启用高度感知和更强的随机化，考验足端定位与稳定性。
- `go2_stairs`：三角网格楼梯/坡面，主攻上台阶与坡面稳定。
- `go2_trot`（本文主角）：平地上学习 **固定节奏的对角小跑（trot gait）**，观察与奖励都显式编码了步态时钟与落脚相位。

---

## 2. 如何运行
```bash
cd quadruped_gym/legged_gym
python legged_gym/scripts/train.py --task go2_trot --run_name demo --headless
```
- 常用调参：`--num_envs`（默认 4096），`--seed`（默认 1），`--max_iterations` 在 `Go2TrotCfgPPO.runner.max_iterations=50000`。

---

## 3. 频率与时序
- 物理步长 `sim.dt = 0.005s`，动作保持 `decimation = 4` → **策略频率 50 Hz** (`dt = 0.02s`)。见 `go2_trot_config.py:54-63` 与 `go2_trot_env.py:643-651`。
- 一集时长 24 s，对应 1200 策略步，超时仅重置不额外惩罚。
- 指令重采样每 5 s；推扰动每 4 s；步态周期 `cycle_time = 0.5 s`，半周期 0.25 s。

---

## 4. 观察与动作
- 观察维度 59（`go2_trot_config.py:4-11`，`go2_trot_env.py:149-165`）：
  - `base_ang_vel`(3)、步态时钟 `sin/cos`(2)、指令 `v_x,v_y,ω_yaw`(3)
  - 关节相对默认角度(12) & 关节速度(12)
  - 基座欧拉角(3)、当前/上一动作各12
- 特权观察 77（用于 critic），额外包含线速度、绝对关节角、期望支撑相位与接触状态等，便于不完全观测下的稳定学习（`go2_trot_env.py:167-179`）。
- 动作 12 维：各关节的 **目标角偏移**，经缩放 `action_scale=0.25 rad` 加到默认角度，再由 PD 控制器执行（`go2_trot_env.py:264-274`）。刚度/阻尼 `Kp=20, Kd=0.5`。
- 引入步态相位信息 `phase = self._get_phase()` 以训练狗子对角小跑。

---

## 5. 奖励设计（数学 + 直觉）
总奖励：
`R = Σ_i (scale_i * dt * r_i)`（除 termination 外），`dt=0.02`。负号代表惩罚。尺度见 `go2_trot_config.py:114-130`，实现见 `go2_trot_env.py:656-754`。

| 目标 | 公式/判定 | 权重 | 作用 |
| --- | --- | --- | --- |
| 对角小跑相位对齐 `trot` | 双前/后脚接触模式与时钟掩码一致则为1，否则0；仅在 | `+0.8` | 强制学会对角成对落地/摆动。 |
|  | `‖v_cmd‖>0.1` 时生效（`go2_trot_env.py:656-666`） |  |  |
| 足端摆动高度 `feet_clearance` | 摆动期足高逼近 `|sin(2π phase)|*0.06`，误差越小越好（`go2_trot_env.py:667-682`） | `+0.1` | 防拖地、节奏感。 |
| 线速度跟踪 `tracking_lin_vel` | `exp(-‖v_xy - v_cmd‖²/σ)`，但需 `trot_mean>0.78` 才计入（`go2_trot_env.py:734-737`） | `+2.0` | 只有走出稳定小跑才给速度奖励，避免乱跑刷分。 |
| 角速度跟踪 `tracking_ang_vel` | `exp(-(ω_z-ω_cmd)²/σ)` 同上（`go2_trot_env.py:738-742`） | `+2.0` | 旋转指令跟踪。 |
| 竖直速度惩罚 `lin_vel_z` | `v_z²`（`go2_trot_env.py:689-691`） | `-2.0` | 防弹跳。 |
| 横滚/俯仰角速度 `ang_vel_xy` | `‖ω_x,ω_y‖²`（`go2_trot_env.py:693-695`） | `-0.05` | 抑制摇晃。 |
| 姿态水平性 `orientation` | `‖proj_gravity_xy‖²`（`go2_trot_env.py:697-699`） | `-2.0` | 维持机身水平。 |
| 机身高度 `base_height` | `(z-0.29)²`（`go2_trot_env.py:701-704`） | `-5.0` | 过高/过低都惩罚，帮助找合适支撑。 |
| 力矩平方 `torques` | `‖τ‖²`（`go2_trot_env.py:706-708`） | `-1e-4` | 平滑能耗。 |
| 关节加速度 `dof_acc` | `‖(ẋ_prev-ẋ)/dt‖²`（`go2_trot_env.py:714-716`） | `-2.5e-7` | 抑制抖动。 |
| 动作跃变 `action_rate` | `‖a_prev-a‖²`（`go2_trot_env.py:718-720`） | `-0.01` | 平滑控制输入。 |
| 自碰撞 `collision` | 大于0.1N 的大腿/小腿碰撞次数（`go2_trot_env.py:721-723`） | `-1.0` | 防打腿。 |
| 静止惩罚 `stand_still` | 无指令时关节偏离默认角度之和（`go2_trot_env.py:744-746`） | `-1.0` | 没指令时别乱动。 |
| 髋角偏移 `default_hip_pos` | 四个髋关节绝对值之和（`go2_trot_env.py:684-687`） | `-0.2` | 保持髋在中立附近。 |
| 全关节偏移 `default_pos` | 所有关节偏离默认角度之和（`go2_trot_env.py:752-754`） | `-0.1` | 额外收紧姿态。 |
| 终止 `termination` | 撞击机身或超时 → `-0`（这里为0，仅用于复位） | 0 | 控制 episode 结束。 |

> 小技巧：想看单项曲线，可在 RL logger 里记录 `extras["episode"]["rew_*"]`，它们在 `reset_idx` 时被归一化为每秒回报（`go2_trot_env.py:110-118`）。

---

## 6. 步态时钟与接触判定
- 相位 `phase = (t mod 0.5)/0.5`（`go2_trot_env.py:186-190`），`sin/cos` 写入观测，便于策略感知时序。
- 期望支撑掩码 `stance_mask = [phase<0.5, phase>0.5]`（`go2_trot_env.py:192-200`），对应对角两腿成对交替。
- 实际接触：脚底竖直力 > 5 N 视为着地（`go2_trot_env.py:154-155`、`656-665`）。与期望掩码比对得到 `trot` 奖励，同时 gating 到速度跟踪上，使策略必须按节奏落脚才能拿到主要正奖励。

---

## 7. 课程学习与随机化
- **指令课程**：若上一集线速度跟踪得分超过最大可得的 80%，则将 `v_x` 范围向两端扩 0.5 m/s，最高到 ±2 m/s（`go2_trot_env.py:365-374`）。意味着先学慢走，再加速。
- **域随机化**（`go2_trot_config.py:86-103` + `go2_trot_env.py:255-279`）：
  - 摩擦系数 0.4~0.8（在 `_process_rigid_shape_props` 里，基类实现）。
  - 基座质量 ±1 kg、各连杆质量乘以 0.9~1.1、质心平移 ±2 cm。
  - 外推扰：每 4 s 随机设定线/角速度，相当于推机器人（`go2_trot_env.py:336-340`）。
 这些随机化帮助策略抵抗建模误差，便于后续 sim2real。

---

## 8. PPO 训练超参（关键摘录）
- Actor/Critic 网络：`[512, 256, 128]`，ELU 激活（`go2_trot_config.py:158-163`）。
- 每次迭代样本量：`num_envs * num_steps_per_env = 4096 * 24 ≈ 98k` transition。
- `γ=0.99`, `λ=0.95`, `entropy_coef=0.01`, 自适应学习率调度（基础在 `LeggedRobotCfgPPO`）。
- 模型每 100 iter 保存一次，默认 50k iter 可在 1×A100 上数小时训练完；如显存不足，降低 `num_envs` 并相应调大 `num_mini_batches` 以保持 batch 大小。

---

## 9. 举一反三到其它 GO2 任务
- **观察差异**：`go2_flat/rough` 额外提供步态命令、历史动作、地形高度；`go2_stairs` 的特权观察包含大量高度点（247 维），需要高度感知来踩台阶。
- **奖励差异**：`flat/rough` 使用更通用的足空中时间、轨迹跟踪奖励；`stairs` 增加对爬坡/楼梯的特定高度与碰撞惩罚。查看各 `*_config.py` 下 `rewards.scales` 即可快速理解设计意图。
- **地形**：`go2_trot` 只在平地；`rough/stairs` 将 `terrain.mesh_type` 设为 `trimesh/heightfield` 并开启 `measure_heights`，需要额外的高度点传感。
- 学习路径建议：先跑 `go2_trot` 体会步态时钟 → `go2_flat` 体验更丰富观测 → `go2_rough`/`go2_stairs` 练习感知与域随机化。

---

## 10. 实战与 ROS2/sim2real 提示
- 保持同样的低层控制：仿真中动作为关节目标角，实机上可直接投喂到电机 PD（注意实机 Kp/Kd 不同，需做尺度匹配）。
- 传感器对齐：实机 IMU 角速度/姿态的噪声与延迟可在仿真中通过 `noise_scales`、`push_interval` 做“延迟+扰动”训练。
- 触地阈值：仿真用 5 N，实机足端力/电流阈值可能不同，部署前可把接触逻辑放在实机侧单独实现，而策略只用时钟条件也能走。
- ROS2 接入：建议封装两个节点
  1) `go2_policy_node`：订阅 `/imu`, `/joint_states`，输出关节目标。
  2) `low_level_controller`：内插到 500–1000 Hz，做安全裁剪（力矩/关节限位）并发给驱动板。
- 观察规范化：复用训练时的均值方差或直接用原始尺度（本代码未做归一化中心化，只做线性缩放）；保持一致即可。

---

## 11. 常见调试项
- 走不起来：先看 `rew_trot` 是否为 0，若是，可能时钟不同步或接触阈值过高；减小阈值或把 `cycle_time` 调长。
- 抖动大：适当增大 `action_rate` 或 `dof_acc` 惩罚；或减小 `action_scale`。
- 摔倒多：增大 `base_height` 惩罚或降低 `max_push_vel_xy`；也可暂时关掉推扰动验证训练是否收敛。

---

祝训练顺利，理解好这套“步态时钟 + 速度跟踪 + 随机化”的组合后，再看其它任务的差异就会非常容易。
