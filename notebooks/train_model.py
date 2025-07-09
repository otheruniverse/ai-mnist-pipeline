import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import onnx
from tqdm import tqdm

os.environ['TORCH_HOME'] = '/tmp/torch'  # 避免缓存问题

# 1. 神经网络定义（与之前一致）
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. 训练函数
def train_model():
    # 准备数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 模型配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    model.train()
    for epoch in range(1):  # 只训练1个epoch（加速）
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/1")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # 保存模型
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("训练完成，模型已保存")
    
    # 转换为ONNX
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    torch.onnx.export(
        model, 
        dummy_input, 
        "mnist_model.onnx",
        export_params=True,
        opset_version=13,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("ONNX模型已导出")

if __name__ == "__main__":
    train_model()
