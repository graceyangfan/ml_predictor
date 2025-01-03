#include "data_preprocessor.h"
#include <xtensor/xnpy.hpp>
#include <stdexcept>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

ImagePreprocessor::ImagePreprocessor(int target_size, int crop_size) 
    : target_size_(target_size),
      crop_size_(crop_size),
      is_initialized_(true) {
    // 初始化ImageNet标准化参数
    mean_ = torch::tensor({0.485, 0.456, 0.406}).view({3, 1, 1});
    std_ = torch::tensor({0.229, 0.224, 0.225}).view({3, 1, 1});
}

void print_sample_pixels(const cv::Mat& img, const std::string& step_name) {
    std::cout << "\n" << step_name << " sample pixels:" << std::endl;
    if (img.channels() == 3) {
        cv::Vec3b top_left = img.at<cv::Vec3b>(0, 0);
        cv::Vec3b center = img.at<cv::Vec3b>(img.rows/2, img.cols/2);
        cv::Vec3b bottom_right = img.at<cv::Vec3b>(img.rows-1, img.cols-1);
        
        std::cout << "Top-left (0,0): RGB=["
                  << (int)top_left[0] << "," << (int)top_left[1] << "," << (int)top_left[2] << "]" << std::endl;
        std::cout << "Center (" << img.rows/2 << "," << img.cols/2 << "): RGB=["
                  << (int)center[0] << "," << (int)center[1] << "," << (int)center[2] << "]" << std::endl;
        std::cout << "Bottom-right (" << img.rows-1 << "," << img.cols-1 << "): RGB=["
                  << (int)bottom_right[0] << "," << (int)bottom_right[1] << "," << (int)bottom_right[2] << "]" << std::endl;
    }
}

void print_sample_pixels_float(const cv::Mat& img, const std::string& step_name) {
    std::cout << "\n" << step_name << " sample pixels:" << std::endl;
    if (img.channels() == 3) {
        cv::Vec3f top_left = img.at<cv::Vec3f>(0, 0);
        cv::Vec3f center = img.at<cv::Vec3f>(img.rows/2, img.cols/2);
        cv::Vec3f bottom_right = img.at<cv::Vec3f>(img.rows-1, img.cols-1);
        
        std::cout << "Top-left (0,0): RGB=["
                  << top_left[0] << "," << top_left[1] << "," << top_left[2] << "]" << std::endl;
        std::cout << "Center (" << img.rows/2 << "," << img.cols/2 << "): RGB=["
                  << center[0] << "," << center[1] << "," << center[2] << "]" << std::endl;
        std::cout << "Bottom-right (" << img.rows-1 << "," << img.cols-1 << "): RGB=["
                  << bottom_right[0] << "," << bottom_right[1] << "," << bottom_right[2] << "]" << std::endl;
    }
}

void print_tensor_sample_pixels(const torch::Tensor& tensor, const std::string& step_name) {
    std::cout << "\n" << step_name << " sample pixels:" << std::endl;
    auto t = tensor.squeeze();  // Remove batch dimension if present
    if (t.dim() == 3) {  // CHW format
        std::cout << "Top-left (0,0): RGB=["
                  << t[0][0][0].item<float>() << "," 
                  << t[1][0][0].item<float>() << "," 
                  << t[2][0][0].item<float>() << "]" << std::endl;
        
        std::cout << "Center (112,112): RGB=["
                  << t[0][112][112].item<float>() << "," 
                  << t[1][112][112].item<float>() << "," 
                  << t[2][112][112].item<float>() << "]" << std::endl;
        
        std::cout << "Bottom-right (223,223): RGB=["
                  << t[0][223][223].item<float>() << "," 
                  << t[1][223][223].item<float>() << "," 
                  << t[2][223][223].item<float>() << "]" << std::endl;
    }
}

cv::Mat ImagePreprocessor::decode_image(const std::vector<unsigned char >& image_data) const {
    cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to decode image data.");
    }
    std::cout << "\nC++ preprocessing steps:" << std::endl;
    std::cout << "1. Original image size: " << img.size() << ", channels: " << img.channels() 
              << ", type: " << img.type() << std::endl;
    print_sample_pixels(img, "1. Original image");
    
    // 将图像从BGR转换为RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    print_sample_pixels(img, "1b. After BGR->RGB conversion");
    return img;
}

cv::Mat ImagePreprocessor::resize_image(const cv::Mat& img) const {
    int h = img.rows;
    int w = img.cols;
    
    // 计算输出尺寸
    // 遵循PyTorch的_compute_resized_output_size逻辑
    int new_h, new_w;
    if (w <= h) {
        // 宽度是较短边
        new_w = target_size_;
        new_h = static_cast<int>(std::round(static_cast<float>(target_size_) * h / w));
    } else {
        // 高度是较短边
        new_h = target_size_;
        new_w = static_cast<int>(std::round(static_cast<float>(target_size_) * w / h));
    }

    std::cout << "\nResize calculation details:" << std::endl;
    std::cout << "Original size: [" << h << " x " << w << "]" << std::endl;
    std::cout << "Target size (shorter edge): " << target_size_ << std::endl;
    std::cout << "Calculated new size: [" << new_h << " x " << new_w << "]" << std::endl;

    // 1. 将OpenCV Mat转换为torch tensor (HWC -> NCHW)
    auto tensor_image = torch::from_blob(
        img.data,
        {1, 3, h, w},  // NCHW格式
        torch::kUInt8
    ).clone();
    
    // 2. 转换为float32并归一化到[0, 1]
    tensor_image = tensor_image.to(torch::kFloat32) / 255.0;
    
    // 3. 使用torch的interpolate进行resize
    // 注意：PyTorch的interpolate期望输入是NCHW格式，范围[0, 1]
    auto resized_tensor = torch::nn::functional::interpolate(
        tensor_image,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{new_h, new_w})
            .mode(torch::kBilinear)
            .align_corners(false)
            .antialias(true)  // 与Python端保持一致
    );
    
    // 4. 转换回uint8范围
    resized_tensor = (resized_tensor * 255.0)
                        .round()
                        .clamp(0, 255)
                        .to(torch::kUInt8);
    
    // 5. 转换回OpenCV Mat格式 (NCHW -> HWC)
    resized_tensor = resized_tensor.squeeze(0)  // 移除batch维度
                                 .permute({1, 2, 0});  // CHW -> HWC
    
    cv::Mat resized(new_h, new_w, CV_8UC3, resized_tensor.data_ptr());
    
    // 打印调试信息
    std::cout << "2. After resize: [" << resized.size() << "]" << std::endl;
    print_sample_pixels(resized, "2. After resize");
    
    return resized.clone();  // 返回副本以确保内存安全
}

cv::Mat ImagePreprocessor::center_crop(const cv::Mat& img) const {
    // 从中心裁剪指定大小的区域
    int top = (img.rows - crop_size_) / 2;
    int left = (img.cols - crop_size_) / 2;
    cv::Rect crop_region(left, top, crop_size_, crop_size_);
    cv::Mat cropped = img(crop_region).clone();
    std::cout << "3. After crop: " << cropped.size() << std::endl;
    print_sample_pixels(cropped, "3. After crop");
    return cropped;
}

torch::Tensor ImagePreprocessor::convert_to_tensor(const cv::Mat& img) const {
    // 1. 转换为float类型并归一化到[0,1]
    cv::Mat float_img;
    img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    print_sample_pixels_float(float_img, "4. After convert to float");
    
    // 获取图像的最小最大值（分通道）
    std::vector<cv::Mat> channels;
    cv::split(float_img, channels);
    std::cout << "4. After convert to float (per channel):" << std::endl;
    for (int i = 0; i < channels.size(); ++i) {
        double minVal, maxVal;
        cv::minMaxLoc(channels[i], &minVal, &maxVal);
        std::cout << "   Channel " << i << ": [" << minVal << ", " << maxVal << "]" << std::endl;
    }

    // 2. 转换为tensor并调整维度顺序为[C, H, W]
    auto tensor_image = torch::from_blob(
        float_img.data, 
        {crop_size_, crop_size_, 3}, 
        torch::kFloat32
    ).clone();
    
    // 调整为[C, H, W]格式
    tensor_image = tensor_image.permute({2, 0, 1});
    std::cout << "5. After permute: shape=" << tensor_image.sizes() 
              << ", range=[" << tensor_image.min().item<float>() << ", " 
              << tensor_image.max().item<float>() << "]" << std::endl;
    print_tensor_sample_pixels(tensor_image, "5. After permute");
    
    // 3. 使用ImageNet均值和标准差进行标准化
    tensor_image = tensor_image.unsqueeze(0);  // 添加batch维度 [1, C, H, W]
    std::cout << "6. Before normalization: shape=" << tensor_image.sizes()
              << ", range=[" << tensor_image.min().item<float>() << ", " 
              << tensor_image.max().item<float>() << "]" << std::endl;
    print_tensor_sample_pixels(tensor_image, "6. Before normalization");
              
    tensor_image = (tensor_image - mean_) / std_;
    std::cout << "7. After normalization: shape=" << tensor_image.sizes()
              << ", range=[" << tensor_image.min().item<float>() << ", " 
              << tensor_image.max().item<float>() << "]" << std::endl;
    print_tensor_sample_pixels(tensor_image, "7. After normalization");
    
    // 打印每个通道的统计信息
    for (int i = 0; i < 3; ++i) {
        auto channel = tensor_image.select(1, i);
        std::cout << "\nChannel " << i << " statistics:" << std::endl;
        std::cout << "Mean: " << channel.mean().item<float>() << std::endl;
        std::cout << "Std: " << channel.std().item<float>() << std::endl;
        std::cout << "Min: " << channel.min().item<float>() << std::endl;
        std::cout << "Max: " << channel.max().item<float>() << std::endl;
        std::cout << "First 5 values: ";
        for (int j = 0; j < 5; ++j) {
            std::cout << channel[0][0][j].item<float>() << " ";
        }
        std::cout << std::endl;
    }
    
    return tensor_image;
}

torch::Tensor ImagePreprocessor::preprocess(const std::vector<unsigned char >& image_data) const {
    if (!is_initialized_) {
        throw std::runtime_error("Image preprocessor not initialized");
    }
    
    // 1. 解码图像
    cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to decode image data.");
    }
    
    // BGR -> RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    std::cout << "\nC++ preprocessing steps:" << std::endl;
    std::cout << "1. Original image: " << img.size() << ", channels: " << img.channels() << std::endl;
    print_sample_pixels(img, "1. Original image");

    // 2. 转换为float tensor并归一化到[0,1]
    cv::Mat float_img;
    img.convertTo(float_img, CV_32FC3, 1.0/255.0);
    
    // 转换为tensor [H,W,C] -> [C,H,W]
    auto tensor = torch::from_blob(
        float_img.data,
        {float_img.rows, float_img.cols, 3},
        torch::kFloat32
    ).clone();
    tensor = tensor.permute({2, 0, 1});
    
    std::cout << "2. After to_tensor: shape=" << tensor.sizes() 
              << ", range=[" << tensor.min().item<float>() << ", " 
              << tensor.max().item<float>() << "]" << std::endl;
    print_tensor_sample_pixels(tensor, "2. After to_tensor");

    // 3. Resize
    int h = tensor.size(1);
    int w = tensor.size(2);
    int new_h, new_w;
    if (w <= h) {
        new_w = target_size_;
        new_h = static_cast<int>(std::round(static_cast<float>(target_size_) * h / w));
    } else {
        new_h = target_size_;
        new_w = static_cast<int>(std::round(static_cast<float>(target_size_) * w / h));
    }

    tensor = tensor.unsqueeze(0);  // 添加batch维度用于interpolate
    tensor = torch::nn::functional::interpolate(
        tensor,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{new_h, new_w})
            .mode(torch::kBilinear)
            .align_corners(false)
            .antialias(true)
    );
    
    std::cout << "3. After resize: shape=" << tensor.sizes()
              << ", range=[" << tensor.min().item<float>() << ", " 
              << tensor.max().item<float>() << "]" << std::endl;
    print_tensor_sample_pixels(tensor, "3. After resize");

    // 4. Center crop
    int crop_top = (new_h - crop_size_) / 2;
    int crop_left = (new_w - crop_size_) / 2;
    tensor = tensor.slice(2, crop_top, crop_top + crop_size_)
                  .slice(3, crop_left, crop_left + crop_size_);
    
    std::cout << "4. After crop: shape=" << tensor.sizes()
              << ", range=[" << tensor.min().item<float>() << ", " 
              << tensor.max().item<float>() << "]" << std::endl;
    print_tensor_sample_pixels(tensor, "4. After crop");

    // 5. Normalize
    tensor = (tensor - mean_) / std_;
    
    std::cout << "5. After normalize: shape=" << tensor.sizes()
              << ", range=[" << tensor.min().item<float>() << ", " 
              << tensor.max().item<float>() << "]" << std::endl;
    print_tensor_sample_pixels(tensor, "5. After normalize");

    // 打印每个通道的统计信息
    // 确保使用与Python相同的计算方式
    auto tensor_no_batch = tensor.squeeze(0);  // 移除batch维度
    for (int i = 0; i < 3; ++i) {
        auto channel = tensor_no_batch[i];  // 获取单个通道 [H,W]
        auto channel_flat = channel.flatten();  // 展平为1D tensor以计算统计信息
        
        std::cout << "\nChannel " << i << " statistics:" << std::endl;
        std::cout << "Mean: " << channel_flat.mean().item<float>() << std::endl;
        std::cout << "Std: " << channel_flat.std(0).item<float>() << std::endl;  // 使用unbiased=False以匹配Python
        std::cout << "Min: " << channel_flat.min().item<float>() << std::endl;
        std::cout << "Max: " << channel_flat.max().item<float>() << std::endl;
    }
    
    return tensor;
}

TracePreprocessor::TracePreprocessor() : is_initialized_(false) {}

bool TracePreprocessor::load_params(const std::string& mean_file, const std::string& scale_file) {
    try {
        // 加载mean和scale参数
        mean_ = xt::load_npy<double>(mean_file);
        scale_ = xt::load_npy<double>(scale_file);
        
        if (mean_.shape() != scale_.shape()) {
            throw std::runtime_error("Mean and scale shapes do not match");
        }
        
        is_initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        is_initialized_ = false;
        return false;
    }
}

torch::Tensor TracePreprocessor::transform(const std::vector<double>& features) const {
    if (!is_initialized_) {
        throw std::runtime_error("Trace preprocessor not initialized");
    }
    
    if (features.size() != mean_.shape(0)) {
        throw std::runtime_error("Feature size does not match preprocessor parameters");
    }
    
    std::vector<float> transformed(features.size());
    for (size_t i = 0; i < features.size(); ++i) {
        transformed[i] = (features[i] - mean_(i)) / scale_(i);
    }
    
    // 转换为tensor并添加batch维度
    return torch::from_blob(
        transformed.data(),
        {1, static_cast<long>(features.size())},
        torch::kFloat
    ).clone();
} 