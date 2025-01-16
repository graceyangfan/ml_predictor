#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <cmath>
#include <chrono>
#include "../modules/target_manager/prediction_system.h"

// 测试辅助宏
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assert failed: " << message << std::endl; \
            return false; \
        } \
    } while (0)

// 浮点数比较辅助函数
bool is_close(float a, float b, float rtol = 1e-5, float atol = 1e-8) {
    return std::abs(a - b) <= (atol + rtol * std::abs(b));
}

// 辅助函数：读取二进制文件
std::vector<unsigned char> read_binary_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<unsigned char> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

// 测试系统初始化
bool test_system_initialization() {
    std::cout << "Running test: System initialization..." << std::endl;
    
    try {
        // 测试正常初始化
        PredictionSystem system(
            "models/resnet18.pt",
            "models/resnet18.pt",
            "test_data/mean.npy",
            "test_data/scale.npy",
            5, 0.04, 20, 21,
            DeviceType::CPU
        );
        TEST_ASSERT(system.is_ready(), "System should be ready after initialization");

        // 测试错误路径
        try {
            PredictionSystem bad_system(
                "nonexistent_model.pt",
                "models/resnet18.pt",
                "test_data/mean.npy",
                "test_data/scale.npy",
                5, 0.04, 20, 21,
                DeviceType::CPU
            );
            TEST_ASSERT(false, "Should throw exception for nonexistent model");
        } catch (const std::runtime_error&) {
            // Expected exception
        }

        std::cout << "System initialization test passed!" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "System initialization failed: " << e.what() << std::endl;
        return false;
    }
}

// 测试目标管理功能
bool test_target_management() {
    std::cout << "Running test: Target management..." << std::endl;
    
    try {
        PredictionSystem system(
            "models/resnet18.pt",
            "models/resnet18.pt",
            "test_data/mean.npy",
            "test_data/scale.npy",
            5, 0.04, 20, 21,
            DeviceType::CPU
        );

        // 测试添加目标
        int target_id = 1;
        system.add_target(target_id);
        
        // 测试重复添加目标
        system.add_target(target_id);  // 应该被安全处理
        
        // 测试移除目标
        system.remove_target(target_id);
        
        // 测试移除不存在的目标
        system.remove_target(999);  // 应该被安全处理
        
        std::cout << "Target management test passed!" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Target management test failed: " << e.what() << std::endl;
        return false;
    }
}

// 测试图像识别功能
bool test_figure_recognition() {
    std::cout << "Running test: Figure recognition..." << std::endl;
    
    try {
        PredictionSystem system(
            "models/resnet18.pt",
            "models/resnet18.pt",
            "test_data/mean.npy",
            "test_data/scale.npy",
            5, 0.04, 20, 21,
            DeviceType::CPU
        );
        
        int target_id = 1;
        system.add_target(target_id);

        std::vector<std::string> test_images = {
            "test_data/sample.jpg",
        };

        for (const auto& image_path : test_images) {
            std::cout << "\nTesting image: " << image_path << std::endl;
            std::vector<unsigned char> image_data = read_binary_file(image_path);
            
            bool update_success = system.update_info_for_target_figure(target_id, image_data);
            TEST_ASSERT(update_success, "Failed to update target figure");
            
            std::vector<float> figure_probs;
            system.figure_model_recognition(target_id, figure_probs);
            
            // 验证概率向量
            TEST_ASSERT(!figure_probs.empty(), "Figure probabilities should not be empty");
            
            float sum = 0.0f;
            for (float prob : figure_probs) {
                TEST_ASSERT(prob >= 0.0f && prob <= 1.0f, 
                          "Probabilities should be between 0 and 1");
                sum += prob;
            }
            TEST_ASSERT(is_close(sum, 1.0f), "Probabilities should sum to 1");

            // 打印预测结果
            std::cout << "Prediction probabilities for " << image_path << ":" << std::endl;
            for (size_t i = 0; i < figure_probs.size(); ++i) {
                std::cout << "Class " << i << ": " << figure_probs[i] << std::endl;
            }
        }

        // 测试错误情况
        try {
            std::vector<float> figure_probs;
            system.figure_model_recognition(999, figure_probs);  // 不存在的目标ID
            TEST_ASSERT(false, "Should throw exception for non-existent target");
        } catch (const std::runtime_error&) {
            // Expected exception
        }

        std::cout << "Figure recognition test passed!" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Figure recognition test failed: " << e.what() << std::endl;
        return false;
    }
}

// 测试融合功能
bool test_fusion() {
    std::cout << "Running test: Fusion functionality..." << std::endl;
    
    try {
        PredictionSystem system(
            "models/resnet18.pt",
            "models/resnet18.pt",
            "test_data/mean.npy",
            "test_data/scale.npy",
            5, 0.04, 20, 21,
            DeviceType::CPU
        );
        
        int target_id = 1;
        system.add_target(target_id);
        
        // 准备测试数据
        std::vector<float> mock_trace_probs = {0.7f, 0.2f, 0.1f};  // 模拟轨迹识别结果
        std::vector<float> mock_figure_probs = {0.6f, 0.3f, 0.1f}; // 模拟图像识别结果
        
        // 获取融合结果
        int predicted_class = system.get_fusion_target_recognition(
            target_id,
            mock_trace_probs,
            mock_figure_probs
        );
        
        // 验证结果
        TEST_ASSERT(predicted_class >= 0 && predicted_class < 3, 
                   "Predicted class should be within valid range");
        
        std::cout << "Fusion test passed!" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Fusion test failed: " << e.what() << std::endl;
        return false;
    }
}

// 测试完整的识别流程
bool test_complete_recognition_flow() {
    std::cout << "Running test: Complete recognition flow..." << std::endl;
    
    try {
        PredictionSystem system(
            "models/resnet18.pt",
            "models/resnet18.pt",
            "test_data/mean.npy",
            "test_data/scale.npy",
            5, 0.04, 20, 21,
            DeviceType::CPU
        );
        
        int target_id = 1;
        system.add_target(target_id);
        
        // 更新图像数据
        std::vector<unsigned char> image_data = read_binary_file("test_data/sample.jpg");
        system.update_info_for_target_figure(target_id, image_data);

        // 更新航迹数据
        for (int i = 0; i < 15; ++i) {  // 更新足够多次以确保初始化
            system.update_info_for_target_trace(
                target_id,
                1.0 + i, 2.0 + i, 3.0 + i,  // 观测位置
                0.1 + i, 0.2 + i, 0.3 + i,  // 滤波位置
                0.01, 0.02, 0.03,           // 滤波速度
                0.001, 0.002, 0.003         // 滤波加速度
            );
        }
        
        // 获取两个模型的预测结果
        std::vector<float> figure_probs, trace_probs;
        system.figure_model_recognition(target_id, figure_probs);
        system.trace_model_recognition(target_id, trace_probs);
        
        // 打印预测概率
        std::cout << "\nPrediction probabilities:" << std::endl;
        std::cout << "Figure model predictions:" << std::endl;
        for (size_t i = 0; i < figure_probs.size(); ++i) {
            std::cout << "Class " << i << ": " << figure_probs[i] << std::endl;
        }
        
        std::cout << "\nTrace model predictions:" << std::endl;
        for (size_t i = 0; i < trace_probs.size(); ++i) {
            std::cout << "Class " << i << ": " << trace_probs[i] << std::endl;
        }
        
        // 获取融合结果
        int predicted_class = system.get_fusion_target_recognition(
            target_id,
            trace_probs,
            figure_probs
        );
        
        std::cout << "\nFinal prediction: Class " << predicted_class << std::endl;
        
        std::cout << "Complete recognition flow test passed!" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Complete recognition flow test failed: " << e.what() << std::endl;
        return false;
    }
}

// 测试序列预测
bool test_sequence_prediction() {
    std::cout << "Running test: Sequence prediction..." << std::endl;
    
    PredictionSystem system(
        "models/resnet18.pt",
        "models/resnet18.pt",
        "test_data/mean.npy",
        "test_data/scale.npy",
        5, 0.04, 20, 21,
        DeviceType::CPU,
        10  // sequence_length
    );
    
    int target_id = 1;
    system.add_target(target_id);
    
    // 更新足够的轨迹数据
    for (int i = 0; i < 30; ++i) {
        system.update_info_for_target_trace(
            target_id,
            1.0 + i, 2.0 + i, 3.0 + i,
            0.1 + i, 0.2 + i, 0.3 + i,
            0.01, 0.02, 0.03,
            0.001, 0.002, 0.003
        );
    }
    
    // 更新图像数据
    std::vector<unsigned char> image_data = read_binary_file("test_data/sample.jpg");
    system.update_info_for_target_figure(target_id, image_data);
    
    // 测试预测
    int predicted_class;
    bool is_fusion;
    bool success = system.get_fusion_target_recognition(target_id, predicted_class, is_fusion);
    
    TEST_ASSERT(success, "Prediction should succeed");
    TEST_ASSERT(is_fusion, "Should use fusion prediction with both features ready");
    
    return true;
}

int main() {
    bool all_passed = true;
    
    try {
        std::cout << "\n=== Running Prediction System Tests ===\n" << std::endl;
        
        all_passed &= test_system_initialization();
        all_passed &= test_target_management();
        all_passed &= test_figure_recognition();
        all_passed &= test_fusion();
        //all_passed &= test_complete_recognition_flow();
        
        std::cout << "\n=== Test Summary ===\n";
        if (all_passed) {
            std::cout << "All tests passed successfully!" << std::endl;
        } else {
            std::cout << "Some tests failed. Check the output above for details." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 1;
    }
    
    return all_passed ? 0 : 1;
}
