import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. 加载测试数据集
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 加载原始模型
class NeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 10)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# 3. 收集预测结果
all_preds = []
all_labels = []
all_images = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        all_images.append(images)

# 4. 创建混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('MNIST分类混淆矩阵')
plt.savefig('confusion_matrix.png')

# 5. 分析7→2错误样本
error_indices = []
for i in range(len(all_labels)):
    if all_labels[i] == 7 and all_preds[i] == 2:
        error_indices.append(i)

print(f"\n找到 {len(error_indices)} 个7被误判为2的样本")

# 6. 可视化错误样本
plt.figure(figsize=(15, 8))
for j, idx in enumerate(error_indices[:12]):
    plt.subplot(3, 4, j+1)
    img = test_dataset[idx][0].squeeze().numpy()
    plt.imshow(img, cmap='gray')
    plt.title(f"样本 {idx} | 真实:7, 预测:2")
    plt.axis('off')
plt.tight_layout()
plt.savefig('error_samples_7_to_2.png')
