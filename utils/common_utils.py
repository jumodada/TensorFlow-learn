#!/usr/bin/env python3
"""
TensorFlow学习项目通用工具函数
包含数据处理、可视化、模型评估等常用功能
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional

def print_section(title: str, width: int = 60):
    """打印带格式的章节标题"""
    print("=" * width)
    print(f"🔧 {title}")
    print("=" * width)

def print_tensor_info(tensor: tf.Tensor, name: str = "Tensor"):
    """打印张量的详细信息"""
    print(f"{name}信息:")
    print(f"  形状: {tensor.shape}")
    print(f"  数据类型: {tensor.dtype}")
    print(f"  大小: {tf.size(tensor).numpy()}")
    print(f"  值:\n{tensor}")

def generate_sample_data(n_samples: int = 100, 
                        n_features: int = 1, 
                        noise: float = 0.1,
                        seed: int = 42) -> Tuple[tf.Tensor, tf.Tensor]:
    """生成示例数据用于学习"""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # 生成特征
    X = tf.random.uniform([n_samples, n_features], -2.0, 2.0)
    
    # 生成标签 (线性关系 + 噪声)
    true_w = tf.constant([1.5] * n_features, dtype=tf.float32)
    true_b = tf.constant(0.5, dtype=tf.float32)
    
    y = tf.matmul(X, tf.expand_dims(true_w, 1)) + true_b
    y = y + tf.random.normal([n_samples, 1], stddev=noise)
    
    return X, tf.squeeze(y)

def plot_data_2d(X: tf.Tensor, y: tf.Tensor, 
                title: str = "数据可视化",
                xlabel: str = "X", 
                ylabel: str = "y"):
    """绘制2D数据点"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X.numpy(), y.numpy(), alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_loss_history(loss_history: List[float], 
                     title: str = "训练损失历史"):
    """绘制损失函数变化历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel("训练轮次")
    plt.ylabel("损失值")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def normalize_data(data: tf.Tensor, 
                  method: str = "zscore") -> Tuple[tf.Tensor, dict]:
    """数据标准化"""
    if method == "zscore":
        mean = tf.reduce_mean(data, axis=0)
        std = tf.math.reduce_std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)
        stats = {"mean": mean, "std": std}
    
    elif method == "minmax":
        min_val = tf.reduce_min(data, axis=0)
        max_val = tf.reduce_max(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        stats = {"min": min_val, "max": max_val}
    
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    return normalized, stats

def split_data(X: tf.Tensor, y: tf.Tensor, 
              train_ratio: float = 0.8,
              seed: int = 42) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """划分训练集和测试集"""
    tf.random.set_seed(seed)
    
    n_samples = X.shape[0]
    n_train = int(n_samples * train_ratio)
    
    # 随机打乱索引
    indices = tf.random.shuffle(tf.range(n_samples))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = tf.gather(X, train_indices)
    y_train = tf.gather(y, train_indices)
    X_test = tf.gather(X, test_indices)
    y_test = tf.gather(y, test_indices)
    
    return X_train, X_test, y_train, y_test

def calculate_metrics(y_true: tf.Tensor, y_pred: tf.Tensor) -> dict:
    """计算回归指标"""
    # 均方误差
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # 平均绝对误差
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # R²决定系数
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - ss_res / ss_tot
    
    return {
        "MSE": mse.numpy(),
        "MAE": mae.numpy(),
        "R²": r2.numpy()
    }

def print_metrics(metrics: dict, title: str = "模型评估指标"):
    """打印评估指标"""
    print(f"\n📊 {title}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

def create_batches(X: tf.Tensor, y: tf.Tensor, 
                  batch_size: int = 32) -> tf.data.Dataset:
    """创建批次数据"""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def plot_predictions(X: tf.Tensor, y_true: tf.Tensor, y_pred: tf.Tensor,
                    title: str = "预测结果对比"):
    """绘制预测结果对比"""
    if X.shape[1] == 1:  # 一维特征
        plt.figure(figsize=(10, 6))
        
        # 排序以便绘制平滑的线
        sorted_indices = tf.argsort(tf.squeeze(X))
        X_sorted = tf.gather(X, sorted_indices)
        y_true_sorted = tf.gather(y_true, sorted_indices)
        y_pred_sorted = tf.gather(y_pred, sorted_indices)
        
        plt.scatter(X.numpy(), y_true.numpy(), alpha=0.6, label="真实值")
        plt.plot(X_sorted.numpy(), y_pred_sorted.numpy(), 'r-', label="预测值")
        
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        # 多维特征时绘制真实值vs预测值散点图
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true.numpy(), y_pred.numpy(), alpha=0.6)
        
        # 绘制y=x参考线
        min_val = min(tf.reduce_min(y_true), tf.reduce_min(y_pred))
        max_val = max(tf.reduce_max(y_true), tf.reduce_max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="理想预测")
        
        plt.xlabel("真实值")
        plt.ylabel("预测值")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def save_model_summary(model, filepath: str = "model_summary.txt"):
    """保存模型摘要到文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"模型摘要已保存到: {filepath}")

def check_tensorflow_setup():
    """检查TensorFlow环境设置"""
    print("🔍 TensorFlow环境检查:")
    print(f"  TensorFlow版本: {tf.__version__}")
    print(f"  是否启用急切执行: {tf.executing_eagerly()}")
    
    # 检查GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"  检测到GPU: {len(gpu_devices)}个")
        for i, gpu in enumerate(gpu_devices):
            print(f"    GPU {i}: {gpu}")
    else:
        print("  GPU: 未检测到")
    
    # 检查内存
    print(f"  逻辑CPU数量: {len(tf.config.list_logical_devices('CPU'))}")
    
    return True

# 示例使用
if __name__ == "__main__":
    print_section("TensorFlow工具函数测试")
    
    # 检查环境
    check_tensorflow_setup()
    
    # 生成示例数据
    print("\n📊 生成示例数据:")
    X, y = generate_sample_data(n_samples=50, noise=0.2)
    print_tensor_info(X, "特征X")
    print_tensor_info(y, "标签y")
    
    # 数据标准化
    print("\n📏 数据标准化:")
    X_norm, stats = normalize_data(X)
    print(f"标准化统计: {stats}")
    
    # 数据划分
    print("\n📂 数据划分:")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    print("\n✅ 工具函数测试完成!") 