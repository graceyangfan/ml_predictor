#ifndef MODEL_WRAPPER_H
#define MODEL_WRAPPER_H

#include <torch/script.h>
#include <vector>
#include <string>

// 添加条件编译宏
#if defined(USE_CUDA) && defined(TORCH_CUDA_AVAILABLE)
#define HAS_CUDA 1
#else
#define HAS_CUDA 0
#endif

enum class DeviceType {
    CPU,
    CUDA
};

enum class ModelType {
    REGRESSION,
    CLASSIFICATION
};

class ModelWrapper {
private:
    torch::jit::script::Module model;
    bool is_initialized;
    torch::Device device;
    ModelType model_type;
    
public:
    ModelWrapper(ModelType type, DeviceType device_type = DeviceType::CPU);
    
    bool load_model(const std::string& model_path);
    
    // 直接返回模型输出tensor
    torch::Tensor predict(const torch::Tensor& input);
    torch::Tensor predict_batch(const torch::Tensor& batch_input);
    
    // 分类模型的概率输出
    torch::Tensor predict_proba(const torch::Tensor& input);
    torch::Tensor predict_batch_proba(const torch::Tensor& batch_input);
    
    bool is_model_loaded() const;
    DeviceType get_device_type() const;
    ModelType get_model_type() const;
    bool switch_device(DeviceType new_device_type);

private:
    torch::Tensor forward(const torch::Tensor& input);
};
#endif 