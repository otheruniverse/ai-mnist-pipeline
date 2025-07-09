import torch
import torch.onnx
import onnx
from onnxruntime.quantization import quantize_dynamic
import os

# 1. 定义神经网络结构（与训练时完全一致）
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

# 2. 加载原始模型
model = NeuralNet()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# 3. 转换为ONNX格式
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

# 4. 量化模型
quantize_dynamic(
    "mnist_model.onnx",
    "mnist_model_quant.onnx",
    weight_type=onnx.TensorProto.UINT8
)

# 5. 比较模型大小
print("\n=== 模型大小比较 ===")
print(f"原始PyTorch模型: {os.path.getsize('mnist_model.pth')/1024:.2f} KB")
print(f"ONNX模型: {os.path.getsize('mnist_model.onnx')/1024:.2f} KB")
print(f"量化ONNX模型: {os.path.getsize('mnist_model_quant.onnx')/1024:.2f} KB")
