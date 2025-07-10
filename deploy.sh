#!/bin/bash

# 1. 登录ACR（个人版地址）
docker login -u 本届最靓的仔 crpi-awpzezftnb095rf0.cn-hangzhou.personal.cr.aliyuncs.com
# 输入密码: !1xiaochuan

# 2. 拉取最新镜像
docker pull crpi-awpzezftnb095rf0.cn-hangzhou.personal.cr.aliyuncs.com/ai-mnist/mnist-api:latest

# 3. 重启服务
docker stop mnist-container || true
docker rm mnist-container || true
docker run -d \
  --name mnist-container \
  -p 5000:5000 \
  -p 8000:8000 \
  crpi-awpzezftnb095rf0.cn-hangzhou.personal.cr.aliyuncs.com/ai-mnist/mnist-api:latest
