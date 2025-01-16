#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>
#include "../modules/feature_store/feature_store.h"
#include <fstream>
#include <cmath>

// Test assertion macro
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assert failed: " << message << std::endl; \
            return false; \
        } \
    } while (0)

// Helper function to read binary file
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

// Test basic initialization and state
bool test_feature_store_initialization() {
    std::cout << "Running test: Feature Store Initialization..." << std::endl;
    
    double deltaT = 0.04;
    int based_window = 5;
    int cache_length = 6;
    
    Feature_Store store(deltaT, based_window, cache_length);
    
    // Test initial state
    TEST_ASSERT(!store.is_track_initialized(), "Track should not be initialized initially");
    TEST_ASSERT(!store.is_image_initialized(), "Image should not be initialized initially");
    TEST_ASSERT(!store.is_fully_initialized(), "Store should not be fully initialized initially");
    
    std::cout << "Initialization test passed!" << std::endl;
    return true;
}

// Test track data updates and feature computation
bool test_feature_store_track_updates() {
    std::cout << "Running test: Feature Store Track Updates..." << std::endl;
    
    double deltaT = 0.04;
    int based_window = 5;
    int cache_length = 6;
    
    Feature_Store store(deltaT, based_window, cache_length);
    
    // Update track data multiple times
    for (int i = 0; i < 2*cache_length; ++i) {
        store.update(
            1.0 + i, 2.0 + i, 3.0 + i,  // Observe
            0.1 + i, 0.2 + i, 0.3 + i,  // Filter_P
            0.01, 0.02, 0.03,  // Filter_V
            0.001, 0.002, 0.003  // Filter_a
        );
    }
    
    // Test track initialization
    TEST_ASSERT(store.is_track_initialized(), "Track should be initialized after updates");
    
    // Test feature computation
    std::vector<double> features = store.get_trace_features(5);
    TEST_ASSERT(!features.empty(), "Features should not be empty");
    TEST_ASSERT(features.size() == 37, "Should have 37 features as specified");
    
    std::cout << "Track updates test passed!" << std::endl;
    return true;
}

// Test image data handling
bool test_feature_store_image_handling() {
    std::cout << "Running test: Feature Store Image Handling..." << std::endl;
    
    double deltaT = 0.04;
    int based_window = 5;
    int cache_length = 6;
    
    Feature_Store store(deltaT, based_window, cache_length);
    
    // Test image update
    std::vector<unsigned char> image_data = read_binary_file("test_data/sample.jpg");
    store.update_image(image_data);
    
    TEST_ASSERT(store.is_image_initialized(), "Image should be initialized after update");
    
    // Test image data retrieval
    const std::vector<unsigned char>& retrieved_data = store.get_image_data();
    TEST_ASSERT(retrieved_data.size() == image_data.size(), "Retrieved image data size should match original");
    TEST_ASSERT(retrieved_data == image_data, "Retrieved image data should match original");
    
    std::cout << "Image handling test passed!" << std::endl;
    return true;
}

// Test vector operations
bool test_feature_store_vector_operations() {
    std::cout << "Running test: Feature Store Vector Operations..." << std::endl;
    
    double deltaT = 0.04;
    int based_window = 5;
    int cache_length = 6;
    
    Feature_Store store(deltaT, based_window, cache_length);
    
    // Test vector operations
    xt::xarray<double> vec1 = {1.0, 2.0, 3.0};
    xt::xarray<double> vec2 = {4.0, 5.0, 6.0};
    
    // Test Modu (magnitude)
    double mag = store.Modu(vec1);
    TEST_ASSERT(std::abs(mag - std::sqrt(14.0)) < EPSILON, "Vector magnitude calculation failed");
    
    // Test vector addition
    xt::xarray<double> sum = store.Add(vec1, vec2);
    TEST_ASSERT(sum[0] == 5.0 && sum[1] == 7.0 && sum[2] == 9.0, "Vector addition failed");
    
    // Test vector subtraction
    xt::xarray<double> diff = store.Sub(vec2, vec1);
    TEST_ASSERT(diff[0] == 3.0 && diff[1] == 3.0 && diff[2] == 3.0, "Vector subtraction failed");
    
    std::cout << "Vector operations test passed!" << std::endl;
    return true;
}

// 添加序列特征测试
bool test_sequence_features() {
    std::cout << "Running test: Sequence features..." << std::endl;
    
    Feature_Store store(0.04, 5, 6, 10);  // max_sequence_length = 10
    
    // 更新足够多的数据以填充序列
    for (int i = 0; i < 15; ++i) {
        store.update(
            1.0 + i, 2.0 + i, 3.0 + i,  // Observe
            0.1 + i, 0.2 + i, 0.3 + i,  // Filter_P
            0.01, 0.02, 0.03,           // Filter_V
            0.001, 0.002, 0.003         // Filter_a
        );
    }
    
    // 检查序列是否准备好
    TEST_ASSERT(store.is_sequence_ready(), "Sequence should be ready after sufficient updates");
    
    // 获取序列并验证
    try {
        const auto& sequence = store.get_trace_features_sequence();
        TEST_ASSERT(sequence.size() == 10, "Sequence should have exactly 10 elements");
        
        // 验证每个特征向量的维度
        for (const auto& features : sequence) {
            TEST_ASSERT(features.size() == 37, "Each feature vector should have 37 dimensions");
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to get sequence: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

int main() {
    bool all_passed = true;
    
    try {
        all_passed &= test_feature_store_initialization();
        all_passed &= test_feature_store_track_updates();
        all_passed &= test_feature_store_image_handling();
        all_passed &= test_feature_store_vector_operations();
        all_passed &= test_sequence_features();
        
        std::cout << "\n=== Test Summary ===\n";
        if (all_passed) {
            std::cout << "All tests passed successfully!" << std::endl;
        } else {
            std::cout << "Some tests failed. Check the output above for details." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return all_passed ? 0 : 1;
}
