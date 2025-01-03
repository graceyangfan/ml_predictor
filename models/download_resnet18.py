import torch
import torchvision.models as models
from pathlib import Path

def download_and_export_resnet18():
    print("Downloading pretrained ResNet18 model...")
    
    # 下载预训练的ResNet18模型
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # 转换为TorchScript格式
    example_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)
    
    # 创建保存目录
    save_path = Path(__file__).parent / "resnet18.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    traced_model.save(str(save_path))
    print(f"Model saved to: {save_path}")
    
    # 验证模型
    print("\nVerifying saved model...")
    loaded_model = torch.jit.load(str(save_path))
    loaded_model.eval()
    
    # 使用随机输入测试模型
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224)
        output = loaded_model(test_input)
        
    print(f"Model verification successful. Output shape: {output.shape}")
    print("Expected shape: [1, 1000] (1 batch, 1000 classes)")

if __name__ == "__main__":
    download_and_export_resnet18() 