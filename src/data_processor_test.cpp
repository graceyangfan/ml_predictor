#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cassert>
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include "../modules/preprocessor/data_preprocessor.h"

// 简单的测试辅助宏
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assert failed: " << message << std::endl; \
            return false; \
        } \
    } while (0)

// 辅助函数：读取二进制文件
std::vector<unsigned char > read_binary_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 读取文件内容
    std::vector<unsigned char > buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

class DataPreprocessorTest {
private:
    std::string image_path_;
    std::string mean_path_;
    std::string scale_path_;
    std::string python_preprocessed_path_;

public:
    DataPreprocessorTest() {
        // 设置测试数据路径
        image_path_ = "test_data/sample.jpg";
        mean_path_ = "test_data/mean.npy";
        scale_path_ = "test_data/scale.npy";
        python_preprocessed_path_ = "test_data/preprocessed_image.npy";
    }

    // 测试图像预处理器初始化
    bool test_image_preprocessor_initialization() {
        std::cout << "Running test: Image preprocessor initialization..." << std::endl;
        
        ImagePreprocessor preprocessor(256, 224);
        TEST_ASSERT(preprocessor.is_initialized(), "Image preprocessor initialization failed");
        
        std::cout << "Test passed!" << std::endl;
        return true;
    }

    // 测试图像预处理完整流程
    bool test_image_preprocessing_pipeline() {
        std::cout << "Running test: Image preprocessing pipeline..." << std::endl;
        
        try {
            // 创建预处理器实例
            ImagePreprocessor preprocessor(256, 224);
            
            // 读取测试图像
            std::vector<unsigned char > image_data = read_binary_file(image_path_);
            TEST_ASSERT(!image_data.empty(), "Failed to read image file");
            
            // 执行预处理
            torch::Tensor processed = preprocessor.preprocess(image_data);
            
            // 验证输出tensor的形状和类型
            TEST_ASSERT(processed.dim() == 4, "Wrong tensor dimension");
            TEST_ASSERT(processed.size(0) == 1, "Wrong batch size");
            TEST_ASSERT(processed.size(1) == 3, "Wrong number of channels");
            TEST_ASSERT(processed.size(2) == 224, "Wrong height");
            TEST_ASSERT(processed.size(3) == 224, "Wrong width");
            TEST_ASSERT(processed.dtype() == torch::kFloat32, "Wrong tensor type");
            
        } catch (const std::exception& e) {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            return false;
        }
        
        std::cout << "Test passed!" << std::endl;
        return true;
    }

    // 测试特征预处理器初始化
    bool test_trace_preprocessor_initialization() {
        std::cout << "Running test: Trace preprocessor initialization..." << std::endl;
        
        TracePreprocessor preprocessor;
        TEST_ASSERT(!preprocessor.is_initialized(), "Trace preprocessor should not be initialized");
        
        bool load_success = preprocessor.load_params(mean_path_, scale_path_);
        TEST_ASSERT(load_success, "Failed to load parameters");
        TEST_ASSERT(preprocessor.is_initialized(), "Trace preprocessor not initialized after loading params");
        
        std::cout << "Test passed!" << std::endl;
        return true;
    }

    // 测试特征预处理完整流程
    bool test_trace_preprocessing_pipeline() {
        std::cout << "Running test: Trace preprocessing pipeline..." << std::endl;
        
        try {
            // 创建预处理器实例
            TracePreprocessor preprocessor;
            TEST_ASSERT(preprocessor.load_params(mean_path_, scale_path_), "Failed to load parameters");
            
            // 定义测试数据（与Python端一致）
            std::vector<std::vector<double>> test_data = {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0}
            };
            
            // 定义期望的标准化结果（来自sklearn）
            std::vector<std::vector<double>> expected_result = {
                {-1.22474487, -1.22474487, -1.22474487},
                {0.0, 0.0, 0.0},
                {1.22474487, 1.22474487, 1.22474487}
            };
            
            // 打印测试数据
            std::cout << "\nTest Data:" << std::endl;
            for (const auto& row : test_data) {
                for (double val : row) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
            
            // 对每一行进行预处理并验证结果
            for (size_t i = 0; i < test_data.size(); ++i) {
                // 执行变换
                torch::Tensor processed = preprocessor.transform(test_data[i]);
                auto values = processed.squeeze();  // 移除batch维度
                
                // 验证结果
                for (size_t j = 0; j < test_data[i].size(); ++j) {
                    double expected = expected_result[i][j];
                    double actual = values[j].item<double>();
                    double abs_diff = std::abs(actual - expected);
                    
                    // 使用更严格的误差限制
                    TEST_ASSERT(abs_diff < 1e-7, 
                              "Transform result mismatch at position (" + 
                              std::to_string(i) + "," + std::to_string(j) + 
                              "). Expected: " + std::to_string(expected) + 
                              ", Got: " + std::to_string(actual) +
                              ", Diff: " + std::to_string(abs_diff));
                }
                
                // 打印每行的变换结果
                std::cout << "Row " << i << " transform result: ";
                for (int j = 0; j < values.size(0); ++j) {
                    std::cout << values[j].item<double>() << " ";
                }
                std::cout << std::endl;
            }
            
            std::cout << "All transform results match sklearn's output within tolerance 1e-7" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            return false;
        }
        
        std::cout << "Test passed!" << std::endl;
        return true;
    }

    // 测试异常情况
    bool test_error_handling() {
        std::cout << "Running test: Error handling..." << std::endl;
        
        try {
            // 测试无效图像数据
            ImagePreprocessor img_preprocessor;
            std::vector<unsigned char > invalid_image_data = {0, 1, 2, 3};
            
            bool exception_thrown = false;
            try {
                img_preprocessor.preprocess(invalid_image_data);
            } catch (const std::runtime_error&) {
                exception_thrown = true;
            }
            TEST_ASSERT(exception_thrown, "Expected exception for invalid image data");

            // 测试特征维度不匹配
            TracePreprocessor trace_preprocessor;
            TEST_ASSERT(trace_preprocessor.load_params(mean_path_, scale_path_), "Failed to load parameters");
            
            std::vector<double> invalid_features = {1.0}; // 维度太小
            exception_thrown = false;
            try {
                trace_preprocessor.transform(invalid_features);
            } catch (const std::runtime_error&) {
                exception_thrown = true;
            }
            TEST_ASSERT(exception_thrown, "Expected exception for invalid feature size");
            
        } catch (const std::exception& e) {
            std::cerr << "Unexpected exception occurred: " << e.what() << std::endl;
            return false;
        }
        
        std::cout << "Test passed!" << std::endl;
        return true;
    }

    // 测试C++预处理结果与Python预处理结果的一致性
    bool test_cpp_python_consistency() {
        std::cout << "\nRunning test: C++ vs Python preprocessing consistency..." << std::endl;
        
        try {
            // 加载Python预处理的结果
            auto python_result = xt::load_npy<float>(python_preprocessed_path_);
            std::cout << "\nPython result shape: [" 
                     << python_result.shape(0) << ", "
                     << python_result.shape(1) << ", "
                     << python_result.shape(2) << ", "
                     << python_result.shape(3) << "]" << std::endl;

            // 打印Python结果的一些示例值
            std::cout << "Python result samples:" << std::endl;
            std::cout << "Channel 0 (first few values): ";
            for (size_t i = 0; i < 5; ++i) {
                std::cout << python_result(0, 0, 0, i) << " ";
            }
            std::cout << std::endl;
            
            // 创建C++预处理器并处理相同的图像
            ImagePreprocessor preprocessor(256, 224);
            std::vector<unsigned char > image_data = read_binary_file(image_path_);
            std::cout << "\nImage data size: " << image_data.size() << " bytes" << std::endl;
            
            torch::Tensor cpp_processed = preprocessor.preprocess(image_data);
            std::cout << "\nC++ processed tensor size: [";
            for (size_t i = 0; i < cpp_processed.dim(); ++i) {
                std::cout << cpp_processed.size(i) << (i < cpp_processed.dim()-1 ? ", " : "");
            }
            std::cout << "]" << std::endl;
            
            // 将torch::Tensor转换为numpy兼容格式进行比较
            auto cpp_accessor = cpp_processed.detach().cpu();  // 不需要squeeze，保持[1,3,224,224]
            std::vector<float> cpp_data(cpp_accessor.data_ptr<float>(), 
                                      cpp_accessor.data_ptr<float>() + cpp_accessor.numel());
            
            // 打印C++结果的一些示例值
            std::cout << "C++ result samples (after conversion):" << std::endl;
            std::cout << "Channel 0 (first few values): ";
            for (size_t i = 0; i < 5; ++i) {
                std::cout << cpp_data[i] << " ";
            }
            std::cout << std::endl;
            
            // 重塑为相同的形状 [1, 3, 224, 224]
            auto cpp_result = xt::adapt(cpp_data, {1, 3, 224, 224});
            
            // 计算并打印每个通道的差异统计
            for (size_t channel = 0; channel < 3; ++channel) {
                // 使用xt::view来获取每个通道的视图
                auto python_channel = xt::view(python_result, 0, channel, xt::all(), xt::all());
                auto cpp_channel = xt::view(cpp_result, 0, channel, xt::all(), xt::all());
                
                // 计算差异
                auto channel_diff = xt::abs(python_channel - cpp_channel);
                double channel_max_diff = xt::amax(channel_diff)();
                double channel_mean_diff = xt::mean(channel_diff)();
                
                std::cout << "\nChannel " << channel << " statistics:" << std::endl;
                std::cout << "Max difference: " << channel_max_diff << std::endl;
                std::cout << "Mean difference: " << channel_mean_diff << std::endl;
            }
            
            // 计算整体最大差异
            auto diff = xt::abs(python_result - cpp_result);
            double max_diff = xt::amax(diff)();
            double mean_diff = xt::mean(diff)();
            
            std::cout << "\nOverall statistics:" << std::endl;
            std::cout << "Maximum absolute difference: " << max_diff << std::endl;
            std::cout << "Mean absolute difference: " << mean_diff << std::endl;
            
            // 验证结果是否在可接受的误差范围内
            const double tolerance = 1e-5;
            TEST_ASSERT(max_diff < tolerance, 
                       "Difference between Python and C++ results exceeds tolerance\n"
                       "Max difference: " + std::to_string(max_diff) + "\n"
                       "Mean difference: " + std::to_string(mean_diff));
            
        } catch (const std::exception& e) {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            return false;
        }
        
        std::cout << "Test passed!" << std::endl;
        return true;
    }

    // 运行所有测试
    void run_all_tests() {
        std::cout << "\n=== Running Data Preprocessor Tests ===\n" << std::endl;
        
        bool all_passed = true;
        all_passed &= test_image_preprocessor_initialization();
        all_passed &= test_image_preprocessing_pipeline();
        all_passed &= test_trace_preprocessor_initialization();
        all_passed &= test_trace_preprocessing_pipeline();
        all_passed &= test_error_handling();
        all_passed &= test_cpp_python_consistency();  // 添加新的测试
        
        std::cout << "\n=== Test Summary ===\n";
        if (all_passed) {
            std::cout << "All tests passed successfully!" << std::endl;
        } else {
            std::cout << "Some tests failed. Check the output above for details." << std::endl;
        }
    }
};

int main() {
    DataPreprocessorTest test;
    test.run_all_tests();
    return 0;
}
