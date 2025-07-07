#!/bin/bash

# TensorFlow学习环境自动安装脚本
# 适用于Ubuntu 18.04+

echo "🚀 开始安装TensorFlow学习环境..."

# 检查是否为Ubuntu系统
if ! grep -q "ubuntu" /etc/os-release; then
    echo "❌ 此脚本仅适用于Ubuntu系统"
    exit 1
fi

# 更新系统包
echo "📦 更新系统包..."
sudo apt update && sudo apt upgrade -y

# 安装必要的系统依赖
echo "🔧 安装系统依赖..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    unzip

# 检查Python版本
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "🐍 检测到Python版本: $python_version"

if [ "$python_version" \< "3.8" ]; then
    echo "❌ Python版本过低，需要3.8+版本"
    exit 1
fi

# 创建虚拟环境
echo "🌐 创建虚拟环境..."
if [ -d "tensorflow-env" ]; then
    echo "虚拟环境已存在，跳过创建"
else
    python3 -m venv tensorflow-env
fi

# 激活虚拟环境
echo "⚡ 激活虚拟环境..."
source tensorflow-env/bin/activate

# 升级pip
echo "⬆️ 升级pip..."
pip install --upgrade pip

# 安装Python依赖
echo "📚 安装Python依赖包..."
pip install -r requirements.txt

# 验证TensorFlow安装
echo "✅ 验证TensorFlow安装..."
python3 -c "import tensorflow as tf; print('TensorFlow版本:', tf.__version__)"

echo ""
echo "🎉 安装完成！"
echo ""
echo "使用方法："
echo "1. 激活虚拟环境: source tensorflow-env/bin/activate"
echo "2. 运行HelloWorld: cd 01-hello-world && python hello_tensorflow.py"
echo ""
echo "Happy Learning! 🚀"