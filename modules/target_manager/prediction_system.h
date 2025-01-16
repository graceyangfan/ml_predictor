#ifndef PREDICTION_SYSTEM_H
#define PREDICTION_SYSTEM_H

#include <string>
#include <vector>
#include <torch/torch.h>
#include "target_manager.h"
#include "model_wrapper.h"  
#include "../preprocessor/data_preprocessor.h" 

class PredictionSystem {
private:
    TargetManager target_manager;
    ModelWrapper target_recognition_model_figure;
    ModelWrapper target_recognition_model_trace;
    ImagePreprocessor image_preprocessor;
    TracePreprocessor trace_preprocessor;
    int trace_smooth_window;
    int sequence_length;
    int sequence_stride;
    bool allow_incomplete_sequence;

    // 私有辅助函数
    std::vector<std::vector<double>> rescaleEvidence(
        std::vector<std::vector<double>>& Evidence
    );

    std::vector<double> computeFusion(
        const std::vector<std::vector<double>>& Evidence
    );

    std::vector<float> fuse_recognition_results(
        const std::vector<float>& figure_probs,
        const std::vector<float>& trace_probs
    );

public:
    /**
     * @brief 构造函数
     * @param target_recognition_model_figure_path 图像识别模型路径
     * @param target_recognition_model_trace_path 轨迹识别模型路径
     * @param trace_mean_file 轨迹特征均值文件路径
     * @param trace_scale_file 轨迹特征缩放文件路径
     * @param trace_smooth_window 轨迹平滑窗口大小
     * @param target_delta_t 目标时间间隔
     * @param target_based_window 目标基准窗口大小
     * @param target_cache_length 目标缓存长度
     * @param device_type 设备类型（CPU/GPU）
     * @throws std::runtime_error 如果模型或参数加载失败
     */
    PredictionSystem(
        const std::string& target_recognition_model_figure_path,
        const std::string& target_recognition_model_trace_path,
        const std::string& trace_mean_file,
        const std::string& trace_scale_file,
        int trace_smooth_window,
        double target_delta_t,
        int target_based_window,
        int target_cache_length,
        DeviceType device_type,
        int sequence_length = 10,
        int sequence_stride = 1,
        bool allow_incomplete = false
    );

    /**
     * @brief 更新目标轨迹信息
     * @return 更新是否成功
     */
    bool update_info_for_target_trace(
        int target_id,
        double obs_x, double obs_y, double obs_z,
        double filter_p_x, double filter_p_y, double filter_p_z,
        double filter_v_x, double filter_v_y, double filter_v_z,
        double filter_a_x, double filter_a_y, double filter_a_z
    );

    /**
     * @brief 更新目标图像信息
     * @return 更新是否成功
     */
    bool update_info_for_target_figure(
        int target_id,
        const std::vector<unsigned char>& image_data
    );

    /**
     * @brief 使用图像模型进行目标识别
     * @param[out] figure_probs 输出的图像识别概率
     * @throws std::runtime_error 如果目标不存在或未初始化
     */
    void figure_model_recognition(
        int target_id,
        std::vector<float>& figure_probs
    );

    /**
     * @brief 获取目标预测结果
     * @param target_id 目标ID
     * @param[out] predicted_class 预测的类别
     * @param[out] is_fusion 是否是融合预测结果
     * @return 是否成功进行预测
     * 
     * 预测逻辑：
     * 1. 如果图像未准备好，返回false
     * 2. 如果图像和轨迹都准备好，返回融合预测结果
     * 3. 如果只有图像准备好，返回图像预测结果
     */
    bool get_fusion_target_recognition(
        int target_id,
        int& predicted_class,
        bool& is_fusion
    );

    /**
     * 使用序列数据进行轨迹识别
     * @param target_id 目标ID
     * @param trace_probs 输出的识别概率
     */
    void trace_model_sequence_recognition(
        int target_id,
        std::vector<float>& trace_probs
    );

    // 目标管理函数
    void add_target(int target_id);
    void remove_target(int target_id);
    
    /**
     * @brief 检查系统是否准备就绪
     * @return 如果所有模型都已加载则返回true
     */
    bool is_ready() const;
};

#endif // PREDICTION_SYSTEM_H 