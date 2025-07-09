操作系统：Ubuntu 24.04.2 LTS (noble)
当前用户：ailearner
Python版本：3.12.3
Docker版本：28.3.1
项目目录结构：
  /home/ailearner/ai_projects
  ├── ai_env/              # Python虚拟环境
  ├── Dockerfile           # Docker构建文件
  ├── notebooks/           # 训练相关
  │   ├── mnist_model.onnx
  │   ├── mnist_model.pth
  │   ├── mnist_model_quant.onnx
  │   └── train_model.py   # 训练脚本
  ├── scripts/             # 应用脚本
  │   └── model_api_optimized.py
  ├── requirements.txt     # Python依赖
  └── README.md

阿里云ACR状态（存在问题）
命名空间：ai-mnist
仓库名称：mnist-api
问题：仓库中没有镜像（应该是工作流未成功执行导致）

服务运行状态（存在问题）
没有运行的Docker容器（docker ps输出为空）
API服务未启动
