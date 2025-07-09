import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from tqdm import tqdm

# 1. 加载原始数据集
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

# 2. 加载增强的7数据集
augmented_indices = torch.load('augmented_7_indices.pth')
aug_dataset = torch.utils.data.Subset(train_dataset, augmented_indices['indices'])

# 创建组合数据集
combined_dataset = ConcatDataset([train_dataset, aug_dataset])
print(f"原始训练集大小: {len(train_dataset)}")
print(f"增强后训练集大小: {len(combined_dataset)}")

# 3. 加载预训练模型
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

model = NeuralNet()
model.load_state_dict(torch.load('mnist_model.pth'))

# 4. 微调训练配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 使用更小的学习率
train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

# 5. 微调训练循环（仅1个epoch）
def fine_tune(epochs=1):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"微调 Epoch {epoch+1}/{epochs}")
        for images, labels in loop:
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新进度条
            loop.set_postfix(loss=loss.item())

# 执行微调
fine_tune(epochs=1)

# 6. 保存微调模型
torch.save(model.state_dict(), 'mnist_model_finetuned.pth')
print("微调模型已保存为 mnist_model_finetuned.pth")
