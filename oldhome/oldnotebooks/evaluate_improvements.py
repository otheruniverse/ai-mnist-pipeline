import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
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
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 加载原始模型和微调模型
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

# 原始模型
orig_model = NeuralNet()
orig_model.load_state_dict(torch.load('mnist_model.pth'))
orig_model.eval()

# 微调模型
ft_model = NeuralNet()
ft_model.load_state_dict(torch.load('mnist_model_finetuned.pth'))
ft_model.eval()

# 3. 测试函数
def evaluate_model(model, name):
    correct = 0
    total = 0
    seven_correct = 0
    seven_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            # 统计数字7的准确率
            seven_mask = (labels == 7)
            seven_total += seven_mask.sum().item()
            seven_correct += ((preds == labels) & seven_mask).sum().item()
    
    accuracy = 100 * correct / total
    seven_accuracy = 100 * seven_correct / seven_total if seven_total > 0 else 0
    
    print(f"\n{name}模型评估:")
    print(f"整体准确率: {accuracy:.2f}%")
    print(f"数字7准确率: {seven_accuracy:.2f}%")
    
    return accuracy, seven_accuracy

# 4. 评估两个模型
orig_acc, orig_7_acc = evaluate_model(orig_model, "原始")
ft_acc, ft_7_acc = evaluate_model(ft_model, "微调后")

# 5. 可视化改进效果
models = ['原始模型', '微调模型']
accuracies = [orig_acc, ft_acc]
seven_accuracies = [orig_7_acc, ft_7_acc]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, accuracies, width, label='整体准确率')
rects2 = ax.bar(x + width/2, seven_accuracies, width, label='数字7准确率')

ax.set_ylabel('准确率 (%)')
ax.set_title('微调前后模型性能对比')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(90, 100)

# 添加数据标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('model_improvement_comparison.png')
