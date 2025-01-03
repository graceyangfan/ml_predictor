#include "model_wrapper.h"
#include <stdexcept>

ModelWrapper::ModelWrapper(
    ModelType type,
    DeviceType device_type
) : is_initialized(false),
    device(
#if HAS_CUDA
        device_type == DeviceType::CUDA ? torch::kCUDA : torch::kCPU
#else
        torch::kCPU
#endif
    ),
    model_type(type)
{
}

bool ModelWrapper::load_model(const std::string& model_path) {
    try {
        // 加载模型
        model = torch::jit::load(model_path);
        
        // 将模型移动到指定设备
        model.to(device);
        
        // 设置为评估模式
        model.eval();
        
        is_initialized = true;
        return true;
    } catch (const c10::Error& e) {
        is_initialized = false;
        return false;
    }
}

torch::Tensor ModelWrapper::predict(const torch::Tensor& input) {
    if (!is_initialized) {
        throw std::runtime_error("Model not initialized");
    }

    auto input_device = input.to(device);
    return forward(input_device);
}

torch::Tensor ModelWrapper::predict_batch(const torch::Tensor& batch_input) {
    if (!is_initialized) {
        throw std::runtime_error("Model not initialized");
    }

    auto input_device = batch_input.to(device);
    return forward(input_device);
}

torch::Tensor ModelWrapper::predict_proba(const torch::Tensor& input) {
    if (model_type != ModelType::CLASSIFICATION) {
        throw std::runtime_error("Model is not configured for classification");
    }
    
    auto logits = predict(input);
    return torch::softmax(logits, 1);
}

torch::Tensor ModelWrapper::predict_batch_proba(const torch::Tensor& batch_input) {
    if (model_type != ModelType::CLASSIFICATION) {
        throw std::runtime_error("Model is not configured for classification");
    }
    
    auto logits = predict_batch(batch_input);
    return torch::softmax(logits, 1);
}

torch::Tensor ModelWrapper::forward(const torch::Tensor& input) {
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    
    return model.forward(inputs).toTensor();
}

bool ModelWrapper::is_model_loaded() const {
    return is_initialized;
}

DeviceType ModelWrapper::get_device_type() const {
    return device.is_cuda() ? DeviceType::CUDA : DeviceType::CPU;
}

ModelType ModelWrapper::get_model_type() const {
    return model_type;
}

bool ModelWrapper::switch_device(DeviceType new_device_type) {
    if (!is_initialized) {
        return false;
    }
    
    try {
        torch::Device new_device(
#if HAS_CUDA
            new_device_type == DeviceType::CUDA ? torch::kCUDA : torch::kCPU
#else
            torch::kCPU
#endif
        );
        
        // 如果设备相同，无需切换
        if (device == new_device) {
            return true;
        }
        
        // 将模型移动到新设备
        model.to(new_device);
        device = new_device;
        return true;
    } catch (const c10::Error& e) {
        return false;
    }
} 