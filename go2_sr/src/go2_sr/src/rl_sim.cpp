#include "rl_sim.hpp"

RL_Sim::RL_Sim(int argc, char **argv){
    this->ang_vel_axis = "world";  // Set to "world" frame for simulation
    ros::NodeHandle nh;
    nh.param<std::string>("ros_namespace", this->ros_namespace, "");
    nh.param<std::string>("robot_name", this->robot_name, "");
    // read params from yaml
    this->ReadYaml(this->robot_name, "base.yaml");
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name)){
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr){
            this->fsm = *fsm_ptr;
        }
    }
    else{
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }
    // init robot
    this->joint_publishers_commands.resize(this->params.Get<int>("num_of_dofs"));
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();
    auto joint_controller_names_vec = this->params.Get<std::vector<std::string>>("joint_controller_names");  // avoid dangling reference
    this->StartJointController(this->ros_namespace, joint_controller_names_vec);
    // publisher
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i){
        const std::string &joint_controller_name = joint_controller_names_vec[i];
        const std::string topic_name = this->ros_namespace + joint_controller_name + "/command";
        this->joint_publishers[joint_controller_name] =
            nh.advertise<robot_msgs::MotorCommand>(topic_name, 10);
    }
    // subscriber
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Sim::CmdvelCallback, this);
    this->model_state_subscriber = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 10, &RL_Sim::ModelStatesCallback, this);
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i){
        const std::string &joint_controller_name = joint_controller_names_vec[i];
        const std::string topic_name = this->ros_namespace + joint_controller_name + "/state";
        this->joint_subscribers[joint_controller_name] =
            nh.subscribe<robot_msgs::MotorState>(topic_name, 10,
                [this, joint_controller_name](const robot_msgs::MotorState::ConstPtr &msg)
                {
                    this->JointStatesCallback(msg, joint_controller_name);
                }
            );
        this->joint_positions[joint_controller_name] = 0.0f;
        this->joint_velocities[joint_controller_name] = 0.0f;
        this->joint_efforts[joint_controller_name] = 0.0f;
    }
    // service
    nh.param<std::string>("gazebo_model_name", this->gazebo_model_name, "");
    this->gazebo_pause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
    this->gazebo_unpause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
    this->gazebo_reset_world_client = nh.serviceClient<std_srvs::Empty>("/gazebo/reset_world");


    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;
}

RL_Sim::~RL_Sim(){
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}
// Start joint controllers in Gazebo simulation
void RL_Sim::StartJointController(const std::string& ros_namespace, const std::vector<std::string>& name){
    pid_t pid0 = fork();
    if (pid0 == 0){
        std::string cmd = "rosrun controller_manager spawner joint_state_controller ";
        for (const auto& name : name){
            cmd += name + " ";
        }
        cmd += "__ns:=" + ros_namespace;
        // cmd += " > /dev/null 2>&1";  // Comment this line to see the output
        execlp("sh", "sh", "-c", cmd.c_str(), nullptr);
        exit(1);
    }
}

//Get the current state of the robot from Gazebo simulation
void RL_Sim::GetState(RobotState<float> *state){
    const auto &orientation = this->pose.orientation;
    const auto &angular_velocity = this->vel.angular;

    state->imu.quaternion[0] = orientation.w;
    state->imu.quaternion[1] = orientation.x;
    state->imu.quaternion[2] = orientation.y;
    state->imu.quaternion[3] = orientation.z;

    state->imu.gyroscope[0] = angular_velocity.x;
    state->imu.gyroscope[1] = angular_velocity.y;
    state->imu.gyroscope[2] = angular_velocity.z;

    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i){
        state->motor_state.q[i] = this->joint_positions[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
        state->motor_state.dq[i] = this->joint_velocities[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
        state->motor_state.tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
    }
}

void RL_Sim::SetCommand(const RobotCommand<float> *command){
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i){
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].q = command->motor_command.q[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq = command->motor_command.dq[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp = command->motor_command.kp[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd = command->motor_command.kd[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau = command->motor_command.tau[i];
    }
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i){
        this->joint_publishers[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]].publish(this->joint_publishers_commands[i]);
    }
}

void RL_Sim::RobotControl(){
    // update robot state
    this->GetState(&this->robot_state);
    // state controller
    this->StateController(&this->robot_state, &this->robot_command);
    // reset simulation
    if (this->control.current_key == Input::Keyboard::R){
        std_srvs::Empty empty;
        this->gazebo_reset_world_client.call(empty);
        this->control.current_key = this->control.last_key;
    }
    // pause/unpause simulation
    if (this->control.current_key == Input::Keyboard::Enter){
        if (simulation_running){
            std_srvs::Empty empty;
            this->gazebo_pause_physics_client.call(empty);
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else{
            std_srvs::Empty empty;
            this->gazebo_unpause_physics_client.call(empty);
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
        this->control.current_key = this->control.last_key;
    }
    // clear input
    this->control.ClearInput();
    // send command to robot
    this->SetCommand(&this->robot_command);
}
// Callback function to receive model states from Gazebo
void RL_Sim::ModelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr &msg){
    this->vel = msg->twist[2];
    this->pose = msg->pose[2];
}
// Callback function to receive cmd_vel messages
void RL_Sim::CmdvelCallback(const geometry_msgs::Twist::ConstPtr &msg){
    this->cmd_vel = *msg;
}
// Callback function to receive joint states messages
void RL_Sim::JointStatesCallback(const robot_msgs::MotorState::ConstPtr &msg, const std::string &joint_controller_name){
    this->joint_positions[joint_controller_name] = msg->q;
    this->joint_velocities[joint_controller_name] = msg->dq;
    this->joint_efforts[joint_controller_name] = msg->tau_est;
}
// Run the RL model to compute actions and update outputs
void RL_Sim::RunModel(){
    if (this->rl_init_done && simulation_running){
        this->episode_length_buf += 1;
        
        // 更新基础观测
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;
        this->obs.last_actions = this->obs.actions;
        
        // 计算projected_gravity
        this->obs.projected_gravity = QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec);
        
        // 更新命令
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        if (this->control.navigation_mode){
            this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};
        }
        
        // 更新步态参数
        this->obs.body_height_cmd = this->params.Get<float>("body_height");
        this->obs.gait_freq_cmd = this->params.Get<float>("gait_freq");
        this->obs.gait_phase_cmd = this->params.Get<float>("gait_phase");
        this->obs.gait_offset_cmd = this->params.Get<float>("gait_offset");
        this->obs.gait_bound_cmd = this->params.Get<float>("gait_bound");
        this->obs.gait_duration_cmd = this->params.Get<float>("gait_duration");
        this->obs.footswing_height_cmd = this->params.Get<float>("swing_height");
        this->obs.body_pitch_cmd = this->params.Get<float>("body_pitch");
        this->obs.body_roll_cmd = this->params.Get<float>("body_roll");
        
        // 计算步态索引和时钟输入
        float dt = this->params.Get<float>("dt") * this->params.Get<int>("decimation");
        float gait_freq = this->obs.gait_freq_cmd;
        float gait_phase = this->obs.gait_phase_cmd;
        float gait_offset = this->obs.gait_offset_cmd;
        float gait_bound = this->obs.gait_bound_cmd;
        float gait_duration = this->obs.gait_duration_cmd;
        
        this->obs.gait_indices = std::fmod(this->obs.gait_indices + gait_freq * dt, 1.0f);
        
        // 计算四足的相位
        std::vector<float> foot_indices(4);
        foot_indices[0] = std::fmod(this->obs.gait_indices + gait_phase + gait_offset + gait_bound, 1.0f);  // FL
        foot_indices[1] = std::fmod(this->obs.gait_indices + gait_offset, 1.0f);  // RL
        foot_indices[2] = std::fmod(this->obs.gait_indices + gait_bound, 1.0f);  // FR
        foot_indices[3] = std::fmod(this->obs.gait_indices + gait_phase, 1.0f);  // RR
        
        // 转换相位到时钟输入
        std::vector<float> transformed_indices(4);
        for (int i = 0; i < 4; ++i){
            float phase = foot_indices[i];
            if (phase < gait_duration){
                transformed_indices[i] = phase * (0.5f / gait_duration);
            } else {
                transformed_indices[i] = 0.5f + (phase - gait_duration) * (0.5f / (1.0f - gait_duration));
            }
        }
        
        // 计算时钟输入（sin值）
        this->obs.clock_inputs.resize(4);
        for (int i = 0; i < 4; ++i){
            this->obs.clock_inputs[i] = std::sin(2.0f * M_PI * transformed_indices[i]);
        }
        
        // 执行推理
        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);
        
        if (!this->output_dof_pos.empty()){
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (!this->output_dof_vel.empty()){
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (!this->output_dof_tau.empty()){
            output_dof_tau_queue.push(this->output_dof_tau);
        }
    }
}
// Compute actions using the RL model based on current observations
std::vector<float> RL_Sim::Forward(){
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);
    if (!lock.owns_lock()){
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (this->params.Get<std::vector<int>>("observations_history").size() != 0){
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({this->history_obs});
    }
    else{
        actions = this->model->forward({clamped_obs});
    }
    // clamp actions
    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty()){
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    else{
        return actions;
    }
}
// shutdown handler
void signalHandler(int signum){
    ros::shutdown();
    exit(0);
}
// main
int main(int argc, char **argv){
    signal(SIGINT, signalHandler);
    ros::init(argc, argv, "go2_sr");
    RL_Sim rl_sar(argc, argv);
    ros::spin();
    return 0;
}