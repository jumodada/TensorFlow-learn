#!/usr/bin/env python3
"""
TensorFlow张量基本操作
学习张量的创建、形状操作、索引、切片等基础操作
"""

import tensorflow as tf
import numpy as np

def tensor_creation():
    """张量创建的各种方法"""
    print("📐 张量创建方法:")
    
    # 1. 从Python列表创建
    list_tensor = tf.constant([1, 2, 3, 4])
    print(f"从列表创建: {list_tensor}")
    
    # 2. 从NumPy数组创建
    numpy_array = np.array([[1, 2], [3, 4]])
    numpy_tensor = tf.constant(numpy_array)
    print(f"从NumPy数组创建:\n{numpy_tensor}")
    
    # 3. 创建特殊张量
    zeros = tf.zeros([2, 3])  # 全零张量
    ones = tf.ones([2, 3])    # 全一张量
    identity = tf.eye(3)      # 单位矩阵
    
    print(f"全零张量:\n{zeros}")
    print(f"全一张量:\n{ones}")
    print(f"单位矩阵:\n{identity}")
    
    # 4. 创建序列
    range_tensor = tf.range(start=0, limit=10, delta=2)
    linspace_tensor = tf.linspace(start=0.0, stop=1.0, num=5)
    
    print(f"范围张量: {range_tensor}")
    print(f"线性空间张量: {linspace_tensor}")

def tensor_properties():
    """张量属性查看"""
    print("\n🔍 张量属性:")
    
    tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    
    print(f"张量值:\n{tensor}")
    print(f"形状 (shape): {tensor.shape}")
    print(f"维度 (ndim): {tensor.ndim}")
    print(f"大小 (size): {tf.size(tensor)}")
    print(f"数据类型 (dtype): {tensor.dtype}")

def tensor_reshaping():
    """张量形状操作"""
    print("\n🔄 张量形状操作:")
    
    # 创建一个1维张量
    original = tf.constant([1, 2, 3, 4, 5, 6])
    print(f"原始张量: {original}")
    
    # 重塑为不同形状
    reshaped_2d = tf.reshape(original, [2, 3])
    reshaped_3d = tf.reshape(original, [2, 1, 3])
    
    print(f"重塑为2D [2, 3]:\n{reshaped_2d}")
    print(f"重塑为3D [2, 1, 3]:\n{reshaped_3d}")
    
    # 转置
    matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
    transposed = tf.transpose(matrix)
    
    print(f"原始矩阵:\n{matrix}")
    print(f"转置矩阵:\n{transposed}")
    
    # 扩展和压缩维度
    expanded = tf.expand_dims(original, axis=1)
    squeezed = tf.squeeze(expanded)
    
    print(f"扩展维度:\n{expanded}")
    print(f"压缩维度: {squeezed}")

def tensor_indexing():
    """张量索引和切片"""
    print("\n📑 张量索引和切片:")
    
    # 创建一个3D张量
    tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    print(f"3D张量:\n{tensor_3d}")
    
    # 基本索引
    print(f"第一个2D切片:\n{tensor_3d[0]}")
    print(f"第一行第一列元素: {tensor_3d[0, 0, 0]}")
    
    # 切片操作
    print(f"前两个切片:\n{tensor_3d[:2]}")
    print(f"每个切片的第一行:\n{tensor_3d[:, 0, :]}")
    print(f"倒序:\n{tensor_3d[::-1]}")
    
    # 条件索引
    numbers = tf.constant([1, 2, 3, 4, 5, 6])
    mask = numbers > 3
    filtered = tf.boolean_mask(numbers, mask)
    
    print(f"原始数组: {numbers}")
    print(f"过滤条件 (>3): {mask}")
    print(f"过滤结果: {filtered}")

def tensor_concatenation():
    """张量拼接和分割"""
    print("\n🔗 张量拼接和分割:")
    
    # 创建示例张量
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    
    print(f"张量A:\n{a}")
    print(f"张量B:\n{b}")
    
    # 沿不同轴拼接
    concat_axis0 = tf.concat([a, b], axis=0)  # 垂直拼接
    concat_axis1 = tf.concat([a, b], axis=1)  # 水平拼接
    
    print(f"沿轴0拼接 (垂直):\n{concat_axis0}")
    print(f"沿轴1拼接 (水平):\n{concat_axis1}")
    
    # 堆叠
    stacked = tf.stack([a, b], axis=0)
    print(f"堆叠结果:\n{stacked}")
    
    # 分割
    tensor_to_split = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
    split_result = tf.split(tensor_to_split, num_or_size_splits=2, axis=1)
    
    print(f"待分割张量:\n{tensor_to_split}")
    print(f"分割结果:")
    for i, part in enumerate(split_result):
        print(f"  部分{i+1}:\n{part}")

def data_type_operations():
    """数据类型操作"""
    print("\n🏷️  数据类型操作:")
    
    # 创建不同数据类型的张量
    int_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
    float_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    
    print(f"整数张量: {int_tensor}, 类型: {int_tensor.dtype}")
    print(f"浮点张量: {float_tensor}, 类型: {float_tensor.dtype}")
    
    # 类型转换
    int_to_float = tf.cast(int_tensor, tf.float32)
    float_to_int = tf.cast(float_tensor, tf.int32)
    
    print(f"整数转浮点: {int_to_float}, 类型: {int_to_float.dtype}")
    print(f"浮点转整数: {float_to_int}, 类型: {float_to_int.dtype}")

def main():
    print("=" * 60)
    print("🔧 TensorFlow张量基本操作示例")
    print("=" * 60)
    
    tensor_creation()
    tensor_properties()
    tensor_reshaping()
    tensor_indexing()
    tensor_concatenation()
    data_type_operations()
    
    print("\n✅ 张量基本操作示例完成!")
    print("💡 下一步: 学习变量和常量的使用")

if __name__ == "__main__":
    main() 