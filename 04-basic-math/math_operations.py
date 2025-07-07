#!/usr/bin/env python3
"""
TensorFlow基础数学运算
学习TensorFlow中的各种数学操作和函数
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def basic_arithmetic():
    """基础算术运算"""
    print("🧮 基础算术运算:")
    
    # 创建示例张量
    a = tf.constant([2, 4, 6, 8])
    b = tf.constant([1, 2, 3, 4])
    
    print(f"a = {a}")
    print(f"b = {b}")
    
    # 四则运算
    addition = tf.add(a, b)  # 或者 a + b
    subtraction = tf.subtract(a, b)  # 或者 a - b
    multiplication = tf.multiply(a, b)  # 或者 a * b
    division = tf.divide(a, b)  # 或者 a / b
    
    print(f"a + b = {addition}")
    print(f"a - b = {subtraction}")
    print(f"a * b = {multiplication}")
    print(f"a / b = {division}")
    
    # 其他运算
    power = tf.pow(a, 2)  # 平方
    sqrt = tf.sqrt(tf.cast(a, tf.float32))  # 平方根
    mod = tf.mod(a, 3)  # 取模
    
    print(f"a² = {power}")
    print(f"√a = {sqrt}")
    print(f"a mod 3 = {mod}")

def matrix_operations():
    """矩阵运算"""
    print("\n🔢 矩阵运算:")
    
    # 创建矩阵
    matrix_a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    matrix_b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    vector = tf.constant([1, 2], dtype=tf.float32)
    
    print(f"矩阵A:\n{matrix_a}")
    print(f"矩阵B:\n{matrix_b}")
    print(f"向量v: {vector}")
    
    # 矩阵乘法
    matmul = tf.matmul(matrix_a, matrix_b)
    print(f"\nA × B (矩阵乘法):\n{matmul}")
    
    # 矩阵-向量乘法
    matvec = tf.linalg.matvec(matrix_a, vector)
    print(f"A × v (矩阵-向量乘法): {matvec}")
    
    # 矩阵转置
    transpose = tf.transpose(matrix_a)
    print(f"A转置:\n{transpose}")
    
    # 矩阵行列式
    det = tf.linalg.det(matrix_a)
    print(f"A的行列式: {det}")
    
    # 矩阵求逆
    try:
        inv = tf.linalg.inv(matrix_a)
        print(f"A的逆矩阵:\n{inv}")
    except:
        print("矩阵不可逆")

def reduction_operations():
    """归约操作"""
    print("\n📊 归约操作:")
    
    # 创建示例数据
    data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    print(f"数据:\n{data}")
    
    # 各种归约操作
    sum_all = tf.reduce_sum(data)
    sum_axis0 = tf.reduce_sum(data, axis=0)  # 沿行求和
    sum_axis1 = tf.reduce_sum(data, axis=1)  # 沿列求和
    
    print(f"所有元素求和: {sum_all}")
    print(f"沿轴0求和 (列求和): {sum_axis0}")
    print(f"沿轴1求和 (行求和): {sum_axis1}")
    
    # 其他归约操作
    mean = tf.reduce_mean(data)
    max_val = tf.reduce_max(data)
    min_val = tf.reduce_min(data)
    std = tf.math.reduce_std(data)
    
    print(f"平均值: {mean}")
    print(f"最大值: {max_val}")
    print(f"最小值: {min_val}")
    print(f"标准差: {std}")

def trigonometric_functions():
    """三角函数"""
    print("\n📐 三角函数:")
    
    # 创建角度数据 (弧度制)
    angles = tf.constant([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
    print(f"角度 (弧度): {angles}")
    
    # 三角函数
    sin_vals = tf.sin(angles)
    cos_vals = tf.cos(angles)
    tan_vals = tf.tan(angles)
    
    print(f"sin值: {sin_vals}")
    print(f"cos值: {cos_vals}")
    print(f"tan值: {tan_vals}")
    
    # 反三角函数
    values = tf.constant([0, 0.5, 0.707, 0.866, 1.0])
    asin_vals = tf.asin(values)
    acos_vals = tf.acos(values)
    
    print(f"\n反三角函数:")
    print(f"原值: {values}")
    print(f"arcsin: {asin_vals}")
    print(f"arccos: {acos_vals}")

def logarithmic_exponential():
    """对数和指数函数"""
    print("\n📈 对数和指数函数:")
    
    # 创建数据
    x = tf.constant([1, 2, np.e, 10, 100], dtype=tf.float32)
    print(f"x = {x}")
    
    # 指数函数
    exp = tf.exp(x)
    exp2 = tf.pow(2.0, x)
    
    print(f"e^x = {exp}")
    print(f"2^x = {exp2}")
    
    # 对数函数
    ln = tf.math.log(x)
    log10 = tf.math.log(x) / tf.math.log(10.0)
    log2 = tf.math.log(x) / tf.math.log(2.0)
    
    print(f"ln(x) = {ln}")
    print(f"log₁₀(x) = {log10}")
    print(f"log₂(x) = {log2}")

def statistical_functions():
    """统计函数"""
    print("\n📈 统计函数:")
    
    # 生成随机数据
    tf.random.set_seed(42)
    data = tf.random.normal([1000], mean=5.0, stddev=2.0)
    
    # 计算统计量
    mean = tf.reduce_mean(data)
    variance = tf.math.reduce_variance(data)
    std = tf.math.reduce_std(data)
    
    print(f"数据大小: {tf.size(data)}")
    print(f"均值: {mean:.4f}")
    print(f"方差: {variance:.4f}")
    print(f"标准差: {std:.4f}")
    
    # 分位数
    percentiles = tf.constant([25.0, 50.0, 75.0])
    quantiles = tfp.stats.percentile(data, percentiles) if 'tfp' in globals() else None
    
    # 最值
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)
    
    print(f"最小值: {min_val:.4f}")
    print(f"最大值: {max_val:.4f}")

def comparison_operations():
    """比较操作"""
    print("\n⚖️ 比较操作:")
    
    a = tf.constant([1, 3, 5, 7, 9])
    b = tf.constant([2, 3, 4, 8, 9])
    
    print(f"a = {a}")
    print(f"b = {b}")
    
    # 比较操作
    equal = tf.equal(a, b)
    not_equal = tf.not_equal(a, b)
    less = tf.less(a, b)
    less_equal = tf.less_equal(a, b)
    greater = tf.greater(a, b)
    greater_equal = tf.greater_equal(a, b)
    
    print(f"a == b: {equal}")
    print(f"a != b: {not_equal}")
    print(f"a < b:  {less}")
    print(f"a <= b: {less_equal}")
    print(f"a > b:  {greater}")
    print(f"a >= b: {greater_equal}")

def logical_operations():
    """逻辑操作"""
    print("\n🔀 逻辑操作:")
    
    # 布尔张量
    a = tf.constant([True, False, True, False])
    b = tf.constant([True, True, False, False])
    
    print(f"a = {a}")
    print(f"b = {b}")
    
    # 逻辑操作
    logical_and = tf.logical_and(a, b)
    logical_or = tf.logical_or(a, b)
    logical_not = tf.logical_not(a)
    logical_xor = tf.logical_xor(a, b)
    
    print(f"a AND b: {logical_and}")
    print(f"a OR b:  {logical_or}")
    print(f"NOT a:   {logical_not}")
    print(f"a XOR b: {logical_xor}")

def conditional_operations():
    """条件操作"""
    print("\n🔀 条件操作:")
    
    # 条件选择
    condition = tf.constant([True, False, True, False])
    x = tf.constant([1, 2, 3, 4])
    y = tf.constant([10, 20, 30, 40])
    
    print(f"条件: {condition}")
    print(f"x = {x}")
    print(f"y = {y}")
    
    # where操作: 条件为True选择x，否则选择y
    result = tf.where(condition, x, y)
    print(f"结果: {result}")
    
    # 条件统计
    data = tf.constant([-2, -1, 0, 1, 2, 3])
    positive_count = tf.reduce_sum(tf.cast(data > 0, tf.int32))
    
    print(f"\n数据: {data}")
    print(f"正数个数: {positive_count}")

def practical_example():
    """实际应用示例：简单的数据标准化"""
    print("\n🎯 实际应用示例：数据标准化")
    
    # 生成示例数据
    tf.random.set_seed(42)
    raw_data = tf.random.normal([100], mean=50.0, stddev=15.0)
    
    print(f"原始数据统计:")
    print(f"  形状: {raw_data.shape}")
    print(f"  均值: {tf.reduce_mean(raw_data):.2f}")
    print(f"  标准差: {tf.math.reduce_std(raw_data):.2f}")
    print(f"  最小值: {tf.reduce_min(raw_data):.2f}")
    print(f"  最大值: {tf.reduce_max(raw_data):.2f}")
    
    # Z-score标准化
    mean = tf.reduce_mean(raw_data)
    std = tf.math.reduce_std(raw_data)
    normalized_data = (raw_data - mean) / std
    
    print(f"\n标准化后数据统计:")
    print(f"  均值: {tf.reduce_mean(normalized_data):.2f}")
    print(f"  标准差: {tf.math.reduce_std(normalized_data):.2f}")
    print(f"  最小值: {tf.reduce_min(normalized_data):.2f}")
    print(f"  最大值: {tf.reduce_max(normalized_data):.2f}")

def main():
    print("=" * 60)
    print("🧮 TensorFlow基础数学运算示例")
    print("=" * 60)
    
    basic_arithmetic()
    matrix_operations()
    reduction_operations()
    trigonometric_functions()
    logarithmic_exponential()
    statistical_functions()
    comparison_operations()
    logical_operations()
    conditional_operations()
    practical_example()
    
    print("\n✅ 基础数学运算示例完成!")
    print("💡 下一步: 学习线性回归")

if __name__ == "__main__":
    main() 