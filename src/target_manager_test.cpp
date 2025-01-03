#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>
#include "../modules/target_manager/target_manager.h"
#include <fstream>

// 简单的测试辅助宏
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assert failed: " << message << std::endl; \
            return false; \
        } \
    } while (0)

// 辅助函数：读取二进制文件
std::vector<unsigned char> read_binary_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 读取文件内容
    std::vector<unsigned char> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

// 测试目标管理器基本功能
bool test_target_manager_basic() {
    std::cout << "Running test: TargetManager basic functionality..." << std::endl;
    
    TargetManager manager(0.04, 10, 11);  // 使用更小的窗口大小加快测试
    int target_id = 1;
    
    // 测试添加目标
    manager.add_target(target_id);
    TEST_ASSERT(manager.has_target(target_id), "Failed to add target");
    
    // 测试重复添加目标
    manager.add_target(target_id);  // 应该被忽略
    TEST_ASSERT(manager.has_target(target_id), "Target should still exist after duplicate add");
    
    // 测试移除目标
    manager.remove_target(target_id);
    TEST_ASSERT(!manager.has_target(target_id), "Failed to remove target");
    
    std::cout << "Basic functionality test passed!" << std::endl;
    return true;
}

// 测试目标数据更新和初始化状态
bool test_target_data_updates() {
    std::cout << "Running test: Target data updates..." << std::endl;
    
    TargetManager manager(0.04, 10,11);
    int target_id = 1;
    manager.add_target(target_id);
    
    // 初始状态检查
    TEST_ASSERT(!manager.is_target_track_initialized(target_id), "Track should not be initialized initially");
    TEST_ASSERT(!manager.is_target_image_initialized(target_id), "Image should not be initialized initially");
    TEST_ASSERT(!manager.is_target_fully_initialized(target_id), "Target should not be fully initialized initially");
    
    // 更新航迹数据
    for (int i = 0; i < 35; ++i) {  // 更新足够多次以确保初始化
        manager.update_target_trace(
            target_id,
            1.0 + i, 2.0 + i, 3.0 + i,  // 观测位置
            1.1 + i, 1.9 + i, 3.1 + i,  // 滤波位置
            0.01, 0.02, 0.03,           // 滤波速度
            0.001, 0.002, 0.003         // 滤波加速度
        );
    }
    TEST_ASSERT(manager.is_target_track_initialized(target_id), "Track should be initialized after updates");
    
    // 更新图像数据
    std::vector<unsigned char> image_data = read_binary_file("test_data/sample.jpg");
    manager.update_target_image(target_id, image_data);
    TEST_ASSERT(manager.is_target_image_initialized(target_id), "Image should be initialized after update");
    
    // 检查完全初始化状态
    TEST_ASSERT(manager.is_target_fully_initialized(target_id), "Target should be fully initialized");
    
    std::cout << "Data updates test passed!" << std::endl;
    return true;
}

// 测试错误处理
bool test_error_handling() {
    std::cout << "Running test: Error handling..." << std::endl;
    
    TargetManager manager(0.08, 10, 20);
    int target_id = 1;
    
    // 测试对不存在目标的操作
    try {
        manager.update_target_trace(target_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        TEST_ASSERT(false, "Should throw exception when updating non-existent target");
    } catch (const std::runtime_error&) {
        // 预期的异常
    }
    
    try {
        manager.update_target_image(target_id, std::vector<unsigned char>());
        TEST_ASSERT(false, "Should throw exception when updating image for non-existent target");
    } catch (const std::runtime_error&) {
        // 预期的异常
    }
    
    std::cout << "Error handling test passed!" << std::endl;
    return true;
}

int main() {
    bool all_passed = true;
    
    try {
        all_passed &= test_target_manager_basic();
        all_passed &= test_target_data_updates();
        all_passed &= test_error_handling();
        
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