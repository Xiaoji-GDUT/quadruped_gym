#include "obs_buffer.hpp"
#include <stdexcept>
#include <algorithm>

ObservationBuffer::ObservationBuffer() {}

ObservationBuffer::ObservationBuffer(int num_envs,
                                     const std::vector<int>& obs_dims,
                                     int history_length,
                                     const std::string& priority)
    : num_envs(num_envs),
      obs_dims(obs_dims),
      history_length(history_length),
      priority(priority){
    if (num_envs <= 0 || history_length <= 0){
        throw std::invalid_argument("num_envs and history_length must be positive integers.");
    }
    for (int dim : obs_dims){
        if (dim <= 0){
            throw std::invalid_argument("All observation dimensions must be positive");
        }
        num_obs += dim;
    }
    num_obs_total = num_obs;
    if (num_obs_total <= 0){
        throw std::runtime_error("Invalid total observation dimension");
    }

    obs_buf.resize(num_envs);
    for (int env_idx = 0; env_idx < num_envs; ++env_idx){
        obs_buf[env_idx].resize(history_length);
        for (int t = 0; t < history_length; ++t){
            obs_buf[env_idx][t].resize(num_obs_total, 0.0f);
        }
    }
}

void ObservationBuffer::reset(std::vector<int> reset_idxs, const std::vector<float>& new_obs){
    if (obs_buf.empty()){
        return;
    }
    for (int env_idx : reset_idxs){
        if (env_idx >= 0 && env_idx < num_envs){
            for (int t = 0; t < history_length; ++t){
                for (int i = 0; i < num_obs_total && i < static_cast<int>(new_obs.size()); ++i){
                    obs_buf[env_idx][t][i] = new_obs[i];
                }
            }
        }
    }
}

void ObservationBuffer::insert(const std::vector<float>& new_obs){
    if (obs_buf.empty() || new_obs.size() != static_cast<size_t>(num_obs_total)){
        return;
    }
    for (int env_idx = 0; env_idx < num_envs; ++env_idx){
        for (int t = history_length - 1; t > 0; --t){
            obs_buf[env_idx][t] = obs_buf[env_idx][t - 1];
        }
        obs_buf[env_idx][0] = new_obs;
    }
}

std::vector<float> ObservationBuffer::get_obs_vec(std::vector<int> obs_ids){
    if (obs_buf.empty() || obs_ids.empty()){
        return std::vector<float>();
    }
    int output_size = 0;
    for (int obs_id : obs_ids){
        if (obs_id >= 0 && obs_id < static_cast<int>(obs_dims.size())){
            output_size += obs_dims[obs_id];
        }
    }
    if (output_size == 0){
        return std::vector<float>();
    }

    std::vector<float> output;
    output.reserve(num_envs * history_length * output_size);
    if (this->priority == "time"){
        for (int env_idx = 0; env_idx < num_envs; ++env_idx){
            for (int obs_id : obs_ids){
                if (obs_id >= 0 && obs_id < history_length){
                    int slice_idx = obs_id;
                    for (int i = 0; i < num_obs_total; ++i){
                        output.push_back(obs_buf[env_idx][slice_idx][i]);
                    }
                }
            }
        }
    }
    else if (this->priority == "term"){
        for (int env_idx = 0; env_idx < num_envs; ++env_idx){
            int obs_offset = 0;
            for (size_t i = 0; i < obs_dims.size(); ++i){
                int dim = obs_dims[i];
                for (int step : obs_ids){
                    if (step >= 0 && step < history_length){
                        int time_offset = step;
                        for (int j = 0; j < dim; ++j){
                            output.push_back(obs_buf[env_idx][time_offset][obs_offset + j]);
                        }
                    }
                }
                obs_offset += dim;
            }
        }
    }
    return output;
}