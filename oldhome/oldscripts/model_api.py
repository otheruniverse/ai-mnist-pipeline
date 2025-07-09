from flask import Flask, request, jsonify
import torch
import numpy as np
from PIL import Image
import io
import torch.nn as nn

app = Flask(__name__)

# 1. 加载模型
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
model.eval()  # 切换到评估模式

# 2. 图像预处理函数
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((28, 28))
    
    # 增强对比度
    img = img.point(lambda p: p > 128 and 255)
    
    img_array = np.array(img) / 255.0
    img_array = (img_array - 0.1307) / 0.3081
    
    # 添加批次维度
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img_tensor


# 3. 定义API端点
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400
    
    try:
        img_bytes = file.read()
        tensor = preprocess_image(img_bytes)
        with torch.no_grad():
            output = model(tensor)
            _, prediction = torch.max(output, 1)
        
        return jsonify({'prediction': int(prediction.item())})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
