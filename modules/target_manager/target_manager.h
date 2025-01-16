#ifndef TARGET_MANAGER_H
#define TARGET_MANAGER_H

#include <unordered_map>
#include <memory>
#include <vector> 
#include "../feature_store/feature_store.h"

class TargetManager {
private:
    std::unordered_map<int, std::unique_ptr<Feature_Store>> target_stores;
    double deltaT;
    int based_window;
    int cache_length;
    
public:
    TargetManager(
        double deltaT,
        int based_window,
        int cache_length,
        int max_sequence_length
    );
    
    // 添加新目标
    void add_target(int target_id);
    
    // 移除目标
    void remove_target(int target_id);
    
    // 检查目标是否存在
    bool has_target(int target_id) const;
    
    // 获取特定目标的Feature Store
    Feature_Store* get_feature_store(int target_id);
    
    // 更新目标航迹数据
    void update_target_trace(
        int target_id,
        double obs_x, double obs_y, double obs_z,
        double filter_p_x, double filter_p_y, double filter_p_z,
        double filter_v_x, double filter_v_y, double filter_v_z,
        double filter_a_x, double filter_a_y, double filter_a_z
    );
    
    // 更新目标图像数据
    void update_target_image(int target_id, const std::vector<unsigned char>& image_data);
    
    // 添加检查目标初始化状态的函数
    bool is_target_track_initialized(int target_id) const;
    bool is_target_image_initialized(int target_id) const;
    bool is_target_fully_initialized(int target_id) const;
};

#endif 