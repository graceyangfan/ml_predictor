#include "prediction_system.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>

PredictionSystem::PredictionSystem(
    const std::string& target_recognition_model_figure_path,
    const std::string& target_recognition_model_trace_path,
    const std::string& trace_mean_file,
    const std::string& trace_scale_file,
    int trace_smooth_window,
    double target_delta_t,
    int target_based_window,
    int target_cache_length,
    DeviceType device_type
) : target_manager(target_delta_t, target_based_window, target_cache_length),
    target_recognition_model_figure(ModelType::CLASSIFICATION, device_type),
    target_recognition_model_trace(ModelType::CLASSIFICATION, device_type),
    image_preprocessor(256, 224),
    trace_preprocessor(),
    trace_smooth_window(trace_smooth_window)
{
    if(!target_recognition_model_figure.load_model(target_recognition_model_figure_path)) {
        throw std::runtime_error("Failed to load target_recognition_model_figure from: " + 
                               target_recognition_model_figure_path);
    }
    
    if(!target_recognition_model_trace.load_model(target_recognition_model_trace_path)) {
        throw std::runtime_error("Failed to load target_recognition_model_trace from: " + 
                               target_recognition_model_trace_path);
    }
    
    if(!trace_preprocessor.load_params(trace_mean_file, trace_scale_file)) {
        throw std::runtime_error("Failed to load trace preprocessor parameters from: " + 
                               trace_mean_file + " and " + trace_scale_file);
    }
}

bool PredictionSystem::update_info_for_target_trace(
    int target_id,
    double obs_x, 
    double obs_y, 
    double obs_z,
    double filter_p_x, 
    double filter_p_y, 
    double filter_p_z,
    double filter_v_x, 
    double filter_v_y, 
    double filter_v_z,
    double filter_a_x, 
    double filter_a_y, 
    double filter_a_z
) 
{
    //check if has target
    if(!target_manager.has_target(target_id)) {
        //try to add target
        target_manager.add_target(target_id);
    }
    //update target
    target_manager.update_target_trace(
        target_id, 
        obs_x, 
        obs_y, 
        obs_z, 
        filter_p_x, 
        filter_p_y, 
        filter_p_z, 
        filter_v_x, 
        filter_v_y, 
        filter_v_z, 
        filter_a_x, 
        filter_a_y, 
        filter_a_z
    );
    return true;
}

bool PredictionSystem::update_info_for_target_figure(
    int target_id, 
    const std::vector<unsigned char >& image_data
) 
{
    //check if has target
    if(!target_manager.has_target(target_id)) {
        //try to add target
        target_manager.add_target(target_id);
    }
    //update target
    target_manager.update_target_image(target_id, image_data);
    return true;
}


void PredictionSystem::trace_model_recognition(
    int target_id,
    std::vector<float>& trace_probs
)
{
    auto feature_store = target_manager.get_feature_store(target_id);
    if (!feature_store) {
        throw std::runtime_error("Target not found");
    }
    
    if (!feature_store->is_track_initialized()) {
        return;
    }

    // Get current features
    std::vector<double> features = feature_store->get_trace_features(this->trace_smooth_window);
    
    // Preprocess features and get predictions
    torch::Tensor normalized_features = trace_preprocessor.transform(features);
    torch::Tensor probs = target_recognition_model_trace.predict_proba(normalized_features);
    
    // Convert tensor to vector - fixed accessor usage
    torch::Tensor squeezed_probs = probs.squeeze();
    auto probs_accessor = squeezed_probs.accessor<float,1>();
    trace_probs.resize(probs_accessor.size(0));
    for(int i = 0; i < probs_accessor.size(0); i++) {
        trace_probs[i] = probs_accessor[i];
    }
}

void PredictionSystem::figure_model_recognition(
    int target_id,
    std::vector<float>& figure_probs
)
{
    auto feature_store = target_manager.get_feature_store(target_id);
    if (!feature_store) {
        throw std::runtime_error("Target not found");
    }
    
    if (!feature_store->is_image_initialized()) {
        return;
    }

    // Get and preprocess image
    const std::vector<unsigned char>& image_data = feature_store->get_image_data();
    torch::Tensor normalized_image = image_preprocessor.preprocess(image_data);
    
    // Get predictions
    torch::Tensor probs = target_recognition_model_figure.predict_proba(normalized_image);
    
    // Convert tensor to vector - fixed accessor usage
    torch::Tensor squeezed_probs = probs.squeeze();
    auto probs_accessor = squeezed_probs.accessor<float,1>();
    figure_probs.resize(probs_accessor.size(0));
    for(int i = 0; i < probs_accessor.size(0); i++) {
        figure_probs[i] = probs_accessor[i];
    }
}

int PredictionSystem::get_fusion_target_recognition(
    int target_id,
    std::vector<float> & trace_probs,
    std::vector<float> & figure_probs
)
{
    //fuse trace and figure probs
    std::vector<float> fused_probs = fuse_recognition_results(trace_probs, figure_probs);
    std::cout << "Fused Probabilities: [";
    for (size_t i = 0; i < fused_probs.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << fused_probs[i];
        if (i < fused_probs.size() - 1) {
            std::cout << ", ";
        }
    }
    //get class with maximum probability
    auto max_it = std::max_element(fused_probs.begin(), fused_probs.end());
    return std::distance(fused_probs.begin(), max_it);
}

std::vector<float> PredictionSystem::fuse_recognition_results(
    const std::vector<float>& figure_probs,
    const std::vector<float>& trace_probs
) 
{
    // Prepare evidence matrix
    std::vector<std::vector<double>> evidence(figure_probs.size(), std::vector<double>(2));
    for (size_t i = 0; i < figure_probs.size(); ++i) {
        evidence[i][0] = figure_probs[i];
        evidence[i][1] = trace_probs[i];
    }

    // Rescale evidence and compute fusion
    auto rescaled_evidence = rescaleEvidence(evidence);
    auto fusion_result = computeFusion(rescaled_evidence);

    // Convert to float vector
    return std::vector<float>(fusion_result.begin(), fusion_result.end());
}

std::vector<std::vector<double>> PredictionSystem::rescaleEvidence(
    std::vector<std::vector<double>>& Evidence
) 
{
    int num_class = Evidence.size();
    int num_models = Evidence[0].size();

    std::vector<std::vector<double>> evidence_simi(num_models, std::vector<double>(num_models, 0.0));

    for (int i = 0; i < num_models; ++i) {
        for (int j = 0; j < num_models; ++j) {
            if (i == j) {
                evidence_simi[i][j] = 0.0;
                continue;
            }
            double sii = 0.0, sij = 0.0, sjj = 0.0;
            for (int k = 0; k < num_class; ++k) {
                sii += Evidence[k][i] * Evidence[k][i];
                sij += Evidence[k][i] * Evidence[k][j];
                sjj += Evidence[k][j] * Evidence[k][j];
            }
            evidence_simi[i][j] = 1.0 / std::sqrt(0.5 * (sii + sjj - 2.0 * sij));
        }
    }

    std::vector<double> alpha(num_models, 0.0);
    for (int i = 0; i < num_models; ++i) {
        for (int j = 0; j < num_models; ++j) {
            alpha[i] += evidence_simi[i][j];
        }
    }

    double max_alpha = *std::max_element(alpha.begin(), alpha.end());
    std::vector<double> beta(num_models, 0.0);
    for (int i = 0; i < num_models; ++i) {
        beta[i] = alpha[i] / max_alpha;
    }

    for (int t = 0; t < num_class; ++t) {
        for (int i = 0; i < num_models; ++i) {
            Evidence[t][i] *= beta[i];
        }
    }

    return Evidence;
}

std::vector<double> PredictionSystem::computeFusion(
    const std::vector<std::vector<double>>& Evidence
) 
{
    int num_class = Evidence.size();
    int num_models = Evidence[0].size();

    std::vector<double> fusion_prob(num_class, 0.0);

    for (int t = 0; t < num_class; ++t) {
        double not_reject = 0.0;
        for (int j = 0; j < num_models - 1; ++j) {
            for (int k = j + 1; k < num_models; ++k) {
                not_reject += Evidence[t][j] * Evidence[t][k];
            }
        }
        fusion_prob[t] = not_reject;
    }

    double sum_prob = std::accumulate(
        fusion_prob.begin(), 
        fusion_prob.end(), 
        0.0
    );
    for (double& prob : fusion_prob) {
        prob /= sum_prob;
    }

    return fusion_prob;
}


void PredictionSystem::add_target(int target_id) {
    target_manager.add_target(target_id);
}

void PredictionSystem::remove_target(int target_id) {
    target_manager.remove_target(target_id);
}

bool PredictionSystem::is_ready() const {
    return target_recognition_model_figure.is_model_loaded() && 
           target_recognition_model_trace.is_model_loaded();
} 