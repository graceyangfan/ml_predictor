#ifndef DATA_PREPROCESSOR_H
#define DATA_PREPROCESSOR_H

#include <torch/torch.h>
#include <xtensor/xarray.hpp>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// 图像变换基类
class Transform {
public:
    virtual torch::Tensor operator()(const torch::Tensor& tensor) const = 0;
    virtual ~Transform() = default;
};

// Resize变换
class Resize : public Transform {
private:
    int size_;
public:
    explicit Resize(int size) : size_(size) {}
    torch::Tensor operator()(const torch::Tensor& tensor) const override;
};

// CenterCrop变换
class CenterCrop : public Transform {
private:
    int size_;
public:
    explicit CenterCrop(int size) : size_(size) {}
    torch::Tensor operator()(const torch::Tensor& tensor) const override;
};

// Normalize变换
class Normalize : public Transform {
private:
    std::vector<float> mean_;
    std::vector<float> std_;
public:
    Normalize(const std::vector<float>& mean, const std::vector<float>& std)
        : mean_(mean), std_(std) {}
    torch::Tensor operator()(const torch::Tensor& tensor) const override;
};

class ImagePreprocessor {
private:
    int target_size_;    // 目标大小 (短边)
    int crop_size_;      // 裁剪大小
    torch::Tensor mean_; // ImageNet均值
    torch::Tensor std_;  // ImageNet标准差
    bool is_initialized_;

    // 私有辅助函数
    cv::Mat decode_image(const std::vector<unsigned char >& image_data) const;
    cv::Mat resize_image(const cv::Mat& img) const;
    cv::Mat center_crop(const cv::Mat& img) const;
    torch::Tensor convert_to_tensor(const cv::Mat& img) const;

public:
    /**
     * 构造函数
     * @param target_size 调整大小的目标尺寸 (短边)，默认为256
     * @param crop_size 中心裁剪的大小，默认为224
     */
    explicit ImagePreprocessor(int target_size = 256, int crop_size = 224);
    
    /**
     * 预处理图像数据并返回tensor
     * @param image_data 原始图像数据
     * @return torch::Tensor 预处理后的tensor，大小为[1, 3, crop_size_, crop_size_]
     * @throws std::runtime_error 如果预处理器未初始化或处理失败
     */
    torch::Tensor preprocess(const std::vector<unsigned char >& image_data) const;

    /**
     * 检查预处理器是否已初始化
     * @return bool 初始化状态
     */
    bool is_initialized() const { return is_initialized_; }
};

class TracePreprocessor {
private:
    xt::xarray<double> mean_;
    xt::xarray<double> scale_; // scale = 1/std
    bool is_initialized_;

public:
    TracePreprocessor();
    
    // 从文件加载参数
    bool load_params(const std::string& mean_file, const std::string& scale_file);
    
    // 标准化特征并返回tensor
    torch::Tensor transform(const std::vector<double>& features) const;
    bool is_initialized() const { return is_initialized_; }
};

#endif 