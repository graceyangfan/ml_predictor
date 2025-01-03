#include <iostream>
#include <fstream>
#include <vector>
#include "../modules/target_manager/model_wrapper.h"
#include "../modules/preprocessor/data_preprocessor.h"

std::vector<unsigned char > read_image_file(const std::string& image_path) {
    std::ifstream file(image_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open image file: " + image_path);
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 读取文件内容
    std::vector<unsigned char > buffer(file_size);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);
    
    return buffer;
}

std::vector<std::string> read_class_labels(const std::string& label_file) {
    std::vector<std::string> labels;
    std::ifstream file(label_file);
    std::string line;
    
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    
    return labels;
}

int main() {
    try {
        // 1. 初始化模型
        ModelWrapper model(ModelType::CLASSIFICATION, DeviceType::CPU);
        
        // 2. 加载模型
        if (!model.load_model("models/resnet18.pt")) {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }
        std::cout << "Model loaded successfully" << std::endl;
        
        // 3. 初始化图像预处理器
        ImagePreprocessor preprocessor(256, 224);  // ResNet标准预处理参数
        
        // 4. 读取测试图像
        std::vector<unsigned char > image_data = read_image_file("test_data/sample.jpg");
        std::cout << "Image loaded, size: " << image_data.size() << " bytes" << std::endl;
        
        // 5. 预处理图像
        torch::Tensor input_tensor = preprocessor.preprocess(image_data);
        std::cout << "Preprocessed tensor shape: " << input_tensor.sizes() << std::endl;
        
        // 6. 进行预测
        torch::Tensor output = model.predict(input_tensor);
        
        // 7. 获取预测概率
        torch::Tensor probabilities = torch::softmax(output, 1);
        
        // 8. 获取top-5预测结果
        auto top5 = torch::topk(probabilities, 5);
        auto top5_probs = std::get<0>(top5);
        auto top5_indices = std::get<1>(top5);
        
        // 9. 打印预测结果
        std::cout << "\nTop 5 predictions:" << std::endl;
        for (int i = 0; i < 5; ++i) {
            int idx = top5_indices[0][i].item<int>();
            float prob = top5_probs[0][i].item<float>();
            std::cout << i + 1 << ". Index " << idx << ": " 
                     << std::fixed << std::setprecision(4) << prob * 100.0 << "%" 
                     << std::endl;
        }
        
        // 保存结果用于与Python对比
        std::vector<torch::Tensor> results = {
            input_tensor,
            output,
            probabilities,
            top5_indices.to(torch::kFloat32),
            top5_probs
        };
        
        torch::save(results, "test_data/cpp_results.pt");
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 