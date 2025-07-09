from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import onnxruntime
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import psutil
import time

# 初始化Flask应用
app = Flask(__name__)

# ===== 监控配置 =====
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests')
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'API request latency')
CPU_USAGE = Gauge('cpu_usage_percent', 'Current CPU usage')
MEM_USAGE = Gauge('memory_usage_percent', 'Current memory usage')

# ===== 加载优化模型 =====
ort_session = onnxruntime.InferenceSession("./model/mnist_model_quant.onnx")

# ===== 图像预处理函数 =====
def preprocess_image(image_bytes):
    # 转换图像为灰度图并调整大小
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((28, 28))
    
    # 归一化处理
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - 0.1307) / 0.3081  # MNIST标准化参数

    # 新增调试输出
    print("预处理后图像范围:", img_array.min(), img_array.max())
    print("预处理后图像均值:", img_array.mean())
    
    # 调整形状为 [batch, channel, height, width]
    return img_array.reshape(1, 1, 28, 28)

# ===== API端点 =====
@app.route('/predict', methods=['POST'])
def predict():
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    # 检查文件上传
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400
    
    try:
        # 读取并预处理图像
        img_bytes = file.read()
        input_data = preprocess_image(img_bytes)
        
        # ONNX推理
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # 计算置信度
        logits = ort_outs[0][0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        prediction = np.argmax(probabilities)
        confidence = np.max(probabilities)
        
        # 记录延迟和资源使用
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        CPU_USAGE.set(psutil.cpu_percent())
        MEM_USAGE.set(psutil.virtual_memory().percent)
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(confidence),
            'processing_time': round(latency, 4)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== 主程序 =====
if __name__ == '__main__':
    # 启动监控服务（端口8000）
    start_http_server(8000)
    
    # 启动API服务（端口5000）
    app.run(host='0.0.0.0', port=5000)
