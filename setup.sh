#!/bin/bash

# TensorFlow学习环境自动安装脚本
# 适用于Ubuntu 20.04+ (推荐Ubuntu 24.04)

echo "🚀 开始安装TensorFlow学习环境..."

# 检查是否为Ubuntu系统
if ! grep -q "ubuntu" /etc/os-release; then
    echo "❌ 此脚本仅适用于Ubuntu系统"
    exit 1
fi

# 获取Ubuntu版本
ubuntu_version=$(lsb_release -rs)
echo "📋 检测到Ubuntu版本: $ubuntu_version"

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
    python3-full \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    pkg-config \
    libhdf5-dev \
    libssl-dev \
    libffi-dev

# 检查Python版本
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "🐍 检测到Python版本: $python_version"

# 检查Python版本兼容性
if [[ "$(printf '%s\n' "3.9" "$python_version" | sort -V | head -n1)" != "3.9" ]]; then
    echo "❌ Python版本过低，需要3.9+版本"
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

# 升级pip和setuptools
echo "⬆️ 升级pip和工具..."
pip install --upgrade pip setuptools wheel

# 安装Python依赖
echo "📚 安装Python依赖包..."
pip install -r requirements.txt

# 验证TensorFlow安装
echo "✅ 验证TensorFlow安装..."
python3 -c "
import tensorflow as tf
import numpy as np
print('✅ TensorFlow版本:', tf.__version__)
print('✅ NumPy版本:', np.__version__)
print('✅ GPU支持:', 'Yes' if len(tf.config.list_physical_devices('GPU')) > 0 else 'No (CPU only)')
"

echo ""
echo "🎉 安装完成！"
echo ""
echo "使用方法："
echo "1. 激活虚拟环境: source tensorflow-env/bin/activate"
echo "2. 运行HelloWorld: cd 01-hello-world && python hello_tensorflow.py"
echo "3. 退出虚拟环境: deactivate"
echo ""
echo "注意事项："
echo "- 在Ubuntu 24.04上，建议使用虚拟环境来管理Python包"
echo "- 如果需要GPU支持，请确保已安装NVIDIA驱动"
echo ""
echo "Happy Learning! 🚀"