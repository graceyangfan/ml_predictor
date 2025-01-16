#include "target_manager.h"
#include <stdexcept>

TargetManager::TargetManager(
    double deltaT,
    int based_window,
    int cache_length,
    int max_sequence_length
) : deltaT(deltaT),
    based_window(based_window),
    cache_length(cache_length)
{
}

void TargetManager::add_target(int target_id) {
    if (target_stores.find(target_id) != target_stores.end()) {
        //add log 
        return; // 目标已存在
    }
    
    target_stores[target_id] = std::make_unique<Feature_Store>(
        deltaT,
        based_window,
        cache_length,
        max_sequence_length
    );
}

void TargetManager::remove_target(int target_id) {
    target_stores.erase(target_id);
}

bool TargetManager::has_target(int target_id) const {
    return target_stores.find(target_id) != target_stores.end();
}

Feature_Store* TargetManager::get_feature_store(int target_id) {
    auto it = target_stores.find(target_id);
    if (it == target_stores.end()) {
        return nullptr;
    }
    return it->second.get();
}

void TargetManager::update_target_trace(
    int target_id,
    double obs_x, double obs_y, double obs_z,
    double filter_p_x, double filter_p_y, double filter_p_z,
    double filter_v_x, double filter_v_y, double filter_v_z,
    double filter_a_x, double filter_a_y, double filter_a_z
) {
    auto feature_store = get_feature_store(target_id);
    if (!feature_store) {
        throw std::runtime_error("Target ID not found: " + std::to_string(target_id));
    }
    
    feature_store->update(
        obs_x, obs_y, obs_z,
        filter_p_x, filter_p_y, filter_p_z,
        filter_v_x, filter_v_y, filter_v_z,
        filter_a_x, filter_a_y, filter_a_z
    );
}

void TargetManager::update_target_image(int target_id, const std::vector<unsigned char >& image_data) {
    auto feature_store = get_feature_store(target_id);
    if (!feature_store) {
        throw std::runtime_error("Target ID not found: " + std::to_string(target_id));
    }
    
    feature_store->update_image(image_data);
}

bool TargetManager::is_target_track_initialized(int target_id) const {
    auto it = target_stores.find(target_id);
    if (it == target_stores.end()) {
        return false;
    }
    return it->second->is_track_initialized();
}

bool TargetManager::is_target_image_initialized(int target_id) const {
    auto it = target_stores.find(target_id);
    if (it == target_stores.end()) {
        return false;
    }
    return it->second->is_image_initialized();
}

bool TargetManager::is_target_fully_initialized(int target_id) const {
    auto it = target_stores.find(target_id);
    if (it == target_stores.end()) {
        return false;
    }
    return it->second->is_fully_initialized();
}
