import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import RandomApply, ColorJitter, RandomRotation

# 1. 定义增强转换
augmentation = torchvision.transforms.Compose([
    torchvision.transforms.RandomAffine(
        degrees=15,  # 随机旋转±15度
        translate=(0.1, 0.1),  # 随机平移10%
        scale=(0.9, 1.1)  # 随机缩放90%-110%
    ),
    torchvision.transforms.ColorJitter(
        contrast=0.2,  # 随机调整对比度
        brightness=0.2  # 随机调整亮度
    ),
    torchvision.transforms.RandomPerspective(
        distortion_scale=0.2,  # 随机透视变换
        p=0.5
    ),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

# 2. 应用增强到数字7样本
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True
)

# 提取所有数字7样本
seven_indices = [i for i in range(len(train_dataset)) 
                if train_dataset.targets[i] == 7]
seven_samples = [train_dataset[i][0] for i in seven_indices]

# 3. 可视化增强效果
plt.figure(figsize=(15, 10))
for i in range(15):
    # 原始样本
    plt.subplot(3, 5, i+1)
    if i < 5:
        plt.imshow(seven_samples[i], cmap='gray')
        plt.title(f"原始样本 {i}")
    # 增强样本
    else:
        aug_img = augmentation(seven_samples[i%5])
        plt.imshow(aug_img.squeeze().numpy(), cmap='gray')
        plt.title(f"增强样本 {i-5}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('data_augmentation_examples.png')

# 4. 创建增强数据集
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, transform=None, target_class=7, augment_factor=5):
        self.original_dataset = original_dataset
        self.transform = transform
        self.indices = [i for i in range(len(original_dataset)) 
                       if original_dataset.targets[i] == target_class]
        self.augment_factor = augment_factor
        
    def __len__(self):
        return len(self.indices) * (self.augment_factor + 1)
        
    def __getitem__(self, idx):
        if idx < len(self.indices):
            # 原始样本
            img, label = self.original_dataset[self.indices[idx]]
        else:
            # 增强样本
            orig_idx = idx % len(self.indices)
            img, label = self.original_dataset[self.indices[orig_idx]]
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

# 5. 保存增强数据集
aug_dataset = AugmentedDataset(train_dataset, transform=augmentation, target_class=7)
print(f"增强后7的样本数: {len(aug_dataset)} (原始: {len(seven_indices)})")

# 保存增强样本索引
torch.save({
    'indices': aug_dataset.indices,
    'augment_factor': aug_dataset.augment_factor
}, 'augmented_7_indices.pth')
