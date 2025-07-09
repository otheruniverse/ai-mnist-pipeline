from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 1. 使用全连接模型架构（匹配现有权重）
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平图像
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载模型
model = NeuralNet()
model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
model.eval()
logger.info("全连接模型加载完成，进入评估模式")

# 2. 图像预处理函数（保持不变）
def preprocess_image(image_bytes):
    try:
        # 从字节流加载图像
        img = Image.open(io.BytesIO(image_bytes))
        
        # 转换为灰度图
        if img.mode != 'L':
            img = img.convert('L')
        
        # 调整尺寸并保持纵横比
        img = img.resize((28, 28), Image.LANCZOS)
        
        # 转换为NumPy数组
        img_array = np.array(img, dtype=np.float32)
        
        # 反转颜色：背景白->0，数字黑->255
        img_array = 255 - img_array
        
        # 归一化并应用改进的标准化参数
        img_array = img_array / 255.0
        img_array = (img_array - 0.1307) / 0.3081  # MNIST标准参数
        
        # 转换为PyTorch张量
        tensor = torch.tensor(img_array, dtype=torch.float32)
        
        # 添加批次维度（全连接模型不需要通道维度）
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    except Exception as e:
        logger.error(f"预处理失败: {str(e)}")
        raise

# 3. API端点（保持不变）
@app.route('/predict', methods=['POST'])
def predict():
    # 检查文件上传
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'empty filename'}), 400
    
    try:
        img_bytes = file.read()
        
        # 验证图像大小
        if len(img_bytes) > 2 * 1024 * 1024:  # 2MB限制
            return jsonify({'error': 'file size exceeds 2MB limit'}), 400
            
        # 预处理图像
        tensor = preprocess_image(img_bytes)
        
        # 模型预测
        with torch.no_grad():
            output = model(tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        # 记录预测结果
        logger.info(f"预测结果: {prediction.item()}, 置信度: {confidence.item():.4f}")
        
        return jsonify({
            'prediction': int(prediction.item()),
            'confidence': float(confidence.item())
        })
    
    except Exception as e:
        logger.exception("预测过程中发生错误")
        return jsonify({'error': str(e)}), 500

# 健康检查端点
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model': 'FCN'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
