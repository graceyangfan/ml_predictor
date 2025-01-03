import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys
import os

def load_image(image_path):
    """加载并返回PIL图像"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return Image.open(image_path)

def preprocess_image(image):
    """使用标准的PyTorch预处理流程"""
    # 1. 转换为tensor (会自动归一化到[0,1])
    tensor = transforms.ToTensor()(image)
    print("\nPython preprocessing steps:")
    print(f"1. After to_tensor: shape={tensor.shape}, range=[{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    
    # 2. Resize
    resized = transforms.Resize(256)(tensor)
    print(f"2. After resize: shape={resized.shape}, range=[{resized.min().item():.4f}, {resized.max().item():.4f}]")
    
    # 3. CenterCrop
    cropped = transforms.CenterCrop(224)(resized)
    print(f"3. After crop: shape={cropped.shape}, range=[{cropped.min().item():.4f}, {cropped.max().item():.4f}]")
    
    # 4. Normalize
    normalized = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(cropped)
    print(f"4. After normalize: shape={normalized.shape}, range=[{normalized.min().item():.4f}, {normalized.max().item():.4f}]")
    
    # 打印每个通道的统计信息
    for i in range(3):
        channel = normalized[i]
        print(f"\nChannel {i} statistics:")
        print(f"Mean: {channel.mean().item():.4f}")
        print(f"Std: {channel.std().item():.4f}")
        print(f"Min: {channel.min().item():.4f}")
        print(f"Max: {channel.max().item():.4f}")
        print("First 5 values:", channel[0, 0:5].tolist())
    
    # 添加batch维度
    return normalized.unsqueeze(0)

def main():
    # 1. 加载测试图像
    image_path = "sample.jpg"
    image = load_image(image_path)
    print(f"Original image size: {image.size}")
    
    # 2. 预处理图像
    input_tensor = preprocess_image(image)
    print(f"\nFinal tensor shape: {input_tensor.shape}")
    
    # 3. 加载模型
    model = torch.jit.load("../models/resnet18.pt")
    model.eval()
    print("Model loaded successfully")
    
    # 4. 进行预测
    with torch.no_grad():
        output = model(input_tensor)
    
    # 5. 获取预测概率
    probabilities = torch.softmax(output, 1)
    
    # 6. 获取top-5预测结果
    top5_probs, top5_indices = torch.topk(probabilities, 5)
    
    # 7. 打印预测结果
    print("\nTop 5 predictions:")
    for i in range(5):
        idx = top5_indices[0][i].item()
        prob = top5_probs[0][i].item()
        print(f"{i+1}. Index {idx}: {prob*100:.4f}%")

if __name__ == "__main__":
    main()