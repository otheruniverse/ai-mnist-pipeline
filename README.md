操作系统：Ubuntu 24.04.2 LTS (noble)
当前用户：ailearner
Python版本：3.12.3
Docker版本：28.3.1
项目目录结构：
  /home/ailearner/ai\_projects
  ├── ai\_env/              # Python虚拟环境
  ├── Dockerfile           # Docker构建文件
  ├── notebooks/           # 训练相关
  │   ├── mnist\_model.onnx
  │   ├── mnist\_model.pth
  │   ├── mnist\_model\_quant.onnx
  │   └── train\_model.py   # 训练脚本
  ├── scripts/             # 应用脚本
  │   └── model\_api\_optimized.py
  ├── requirements.txt     # Python依赖
  └── README.md

阿里云ACR状态
命名空间：ai-mnist
仓库名称：mnist-api

服务运行状态
没有运行的Docker
API服务启动

GitHub仓库：存储所有源代码和配置
GitHub Actions：自动化执行训练和构建
阿里云ACR：作为Docker镜像仓库，国内访问快
云服务器：从ACR拉取镜像并运行服务
Docker容器：运行AI API服务的隔离环境
