#!/usr/bin/env python3
"""
TensorFlow HelloWorld示例
这是TensorFlow学习的第一个程序，验证环境安装和基本功能
"""

import tensorflow as tf
import numpy as np
import sys

def main():
    print("=" * 50)
    print("🚀 TensorFlow HelloWorld 示例")
    print("=" * 50)
    
    # 1. 输出TensorFlow版本信息
    print(f"📋 TensorFlow版本: {tf.__version__}")
    print(f"🐍 Python版本: {sys.version}")
    print(f"📊 NumPy版本: {np.__version__}")
    
    # 2. 检查GPU支持
    print("\n🔍 硬件检查:")
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"✅ 检测到 {len(gpu_devices)} 个GPU设备:")
        for i, gpu in enumerate(gpu_devices):
            print(f"   GPU {i}: {gpu}")
    else:
        print("⚠️  未检测到GPU，将使用CPU")
    
    # 3. 基本张量操作
    print("\n📐 基本张量操作:")
    
    # 创建常量张量
    hello = tf.constant("Hello, TensorFlow!")
    numbers = tf.constant([1, 2, 3, 4, 5])
    matrix = tf.constant([[1, 2], [3, 4]])
    
    print(f"字符串张量: {hello}")
    print(f"数字张量: {numbers}")
    print(f"矩阵张量:\n{matrix}")
    
    # 4. 简单的数学运算
    print("\n🧮 数学运算:")
    a = tf.constant(5)
    b = tf.constant(3)
    
    print(f"a = {a.numpy()}")
    print(f"b = {b.numpy()}")
    print(f"a + b = {tf.add(a, b).numpy()}")
    print(f"a - b = {tf.subtract(a, b).numpy()}")
    print(f"a * b = {tf.multiply(a, b).numpy()}")
    print(f"a / b = {tf.divide(a, b).numpy():.2f}")
    
    # 5. 矩阵运算
    print("\n🔢 矩阵运算:")
    matrix_a = tf.constant([[1, 2], [3, 4]])
    matrix_b = tf.constant([[5, 6], [7, 8]])
    
    print("矩阵A:")
    print(matrix_a.numpy())
    print("矩阵B:")
    print(matrix_b.numpy())
    print("A + B:")
    print(tf.add(matrix_a, matrix_b).numpy())
    print("A * B (矩阵乘法):")
    print(tf.matmul(matrix_a, matrix_b).numpy())
    
    # 6. 随机数生成
    print("\n🎲 随机数生成:")
    random_normal = tf.random.normal([2, 3], mean=0, stddev=1)
    random_uniform = tf.random.uniform([2, 3], minval=0, maxval=10)
    
    print("正态分布随机数:")
    print(random_normal.numpy())
    print("均匀分布随机数:")
    print(random_uniform.numpy())
    
    print("\n✅ HelloWorld示例运行完成!")
    print("🎉 TensorFlow环境工作正常!")

if __name__ == "__main__":
    main() 