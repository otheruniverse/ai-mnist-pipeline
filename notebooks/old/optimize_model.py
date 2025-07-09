# 创建模型优化脚本 optimize_model.py
import torch
import torch.onnx
import onnx
from onnxruntime.quantization import quantize_dynamic

# 加载原始模型
class NeuralNet(torch.nn.Module):
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
model.eval()

# 转换为ONNX格式
dummy_input = torch.randn(1, 1, 28, 28)
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

# 量化模型（减小大小，提高推理速度）
quantize_dynamic(
    "mnist_model.onnx",
    "mnist_model_quant.onnx",
    weight_type=onnx.TensorProto.UINT8
)

# 比较模型大小
import os
print(f"原始模型大小: {os.path.getsize('mnist_model.pth')/1024:.2f} KB")
print(f"ONNX模型大小: {os.path.getsize('mnist_model.onnx')/1024:.2f} KB")
print(f"量化模型大小: {os.path.getsize('mnist_model_quant.onnx')/1024:.2f} KB")
