#ifndef OBS_BUFFER_HPP
#define OBS_BUFFER_HPP

#include <vector>
#include <string>

class ObservationBuffer{
public:
    ObservationBuffer(int num_envs, const std::vector<int>& obs_dims, int history_length, const std::string& priority);
    ObservationBuffer();

    void reset(std::vector<int> reset_idxs, const std::vector<float>& new_obs);
    void insert(const std::vector<float>& new_obs);
    std::vector<float> get_obs_vec(std::vector<int> obs_ids);

private:
    int num_envs;
    std::vector<int> obs_dims;
    std::string priority;
    int num_obs = 0;
    int history_length = 0;
    int num_obs_total = 0;
    std::vector<std::vector<std::vector<float>>> obs_buf;
};

#endif
