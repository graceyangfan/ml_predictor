import urllib.request
from pathlib import Path

def download_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    save_path = Path(__file__).parent / "imagenet_classes.txt"
    
    print(f"Downloading ImageNet labels from {url}")
    urllib.request.urlretrieve(url, save_path)
    print(f"Labels saved to: {save_path}")
    
    # 验证文件
    with open(save_path, 'r') as f:
        labels = f.readlines()
    print(f"Successfully downloaded {len(labels)} class labels")

if __name__ == "__main__":
    download_imagenet_labels() 