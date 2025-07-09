import onnxruntime
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载模型
ort_session = onnxruntime.InferenceSession("mnist_model_quant.onnx")

# 2. 创建测试数字7 - 使用更符合MNIST风格的绘制方式
def create_proper_7():
    """创建符合MNIST风格的数字7"""
    img = np.zeros((1, 1, 28, 28), dtype=np.float32)
    
    # 顶部横线 (更厚)
    img[0, 0, 4:7, 6:22] = 1.0
    
    # 斜线 (带弧度)
    for i in range(7, 24):
        # 增加线宽
        pos = 25 - i
        if pos >= 1 and pos < 27:  # 确保在边界内
            img[0, 0, i, pos-1:pos+2] = 1.0
    
    return img

# 3. 创建两种测试图像
test_img_simple = np.zeros((1, 1, 28, 28), dtype=np.float32)
test_img_simple[0, 0, 5, 8:20] = 1.0  # 顶部横线
for i in range(6, 22):
    test_img_simple[0, 0, i, 22-i+6] = 1.0  # 斜线

test_img_proper = create_proper_7()

# 4. 标准化处理
test_img_simple = (test_img_simple - 0.1307) / 0.3081
test_img_proper = (test_img_proper - 0.1307) / 0.3081

# 5. 可视化图像
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(test_img_simple[0, 0], cmap='gray', vmin=-1, vmax=3)
plt.title('原始测试图像')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(test_img_proper[0, 0], cmap='gray', vmin=-1, vmax=3)
plt.title('改进后测试图像')
plt.colorbar()

plt.tight_layout()
plt.savefig('test_images_comparison.png')

# 6. 运行预测并打印结果
def predict_image(img, name):
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0][0]
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    prediction = np.argmax(probabilities)
    
    print(f"\n{name}测试结果:")
    print(f"预测数字: {prediction}")
    print(f"置信度分布: {probabilities}")
    print(f"数字7的概率: {probabilities[7]:.4%}")

# 7. 对两种图像进行预测
predict_image(test_img_simple, "原始")
predict_image(test_img_proper, "改进后")

# 8. 添加真实MNIST样本测试
try:
    import torch
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    
    # 加载真实MNIST测试集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    # 找到第一个数字7的样本
    seven_idx = None
    for i in range(len(test_dataset)):
        _, label = test_dataset[i]
        if label == 7:
            seven_idx = i
            break
    
    if seven_idx is not None:
        image, label = test_dataset[seven_idx]
        real_img = image.numpy().astype(np.float32)
        
        # 可视化真实样本
        plt.figure()
        plt.imshow(image.squeeze().numpy(), cmap='gray')
        plt.title(f"真实MNIST样本 (标签:7)")
        plt.savefig('real_mnist_7.png')
        
        # 预测真实样本
        predict_image(real_img, "真实MNIST样本")
except ImportError:
    print("未安装torch，跳过真实样本测试")
