#!/usr/bin/env python3
"""
TensorFlow变量和常量
学习Variables和Constants的区别，以及变量的操作方法
"""

import tensorflow as tf
import numpy as np

def constants_demo():
    """常量示例"""
    print("🔒 常量 (Constants) 示例:")
    
    # 创建常量
    const_scalar = tf.constant(5.0)
    const_vector = tf.constant([1, 2, 3, 4])
    const_matrix = tf.constant([[1, 2], [3, 4]])
    
    print(f"标量常量: {const_scalar}")
    print(f"向量常量: {const_vector}")
    print(f"矩阵常量:\n{const_matrix}")
    
    # 常量是不可变的
    print(f"\n常量的特点:")
    print(f"- 值不可修改")
    print(f"- 计算图中的固定值")
    print(f"- 内存效率高")

def variables_demo():
    """变量示例"""
    print("\n🔄 变量 (Variables) 示例:")
    
    # 创建变量
    var_scalar = tf.Variable(5.0, name="scalar_var")
    var_vector = tf.Variable([1, 2, 3, 4], name="vector_var")
    var_matrix = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name="matrix_var")
    
    print(f"标量变量: {var_scalar}")
    print(f"向量变量: {var_vector}")
    print(f"矩阵变量:\n{var_matrix}")
    
    print(f"\n变量的特点:")
    print(f"- 值可以修改")
    print(f"- 用于存储模型参数")
    print(f"- 支持梯度计算")
    print(f"- 需要显式初始化")

def variable_operations():
    """变量操作"""
    print("\n⚙️ 变量操作:")
    
    # 创建变量
    weight = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name="weight")
    bias = tf.Variable([0.1, 0.2], name="bias")
    
    print(f"初始权重:\n{weight}")
    print(f"初始偏置: {bias}")
    
    # 变量赋值
    print(f"\n📝 变量赋值:")
    weight.assign([[2.0, 3.0], [4.0, 5.0]])
    bias.assign([0.5, 0.6])
    
    print(f"更新后权重:\n{weight}")
    print(f"更新后偏置: {bias}")
    
    # 变量加法赋值
    print(f"\n➕ 变量加法赋值:")
    weight.assign_add([[0.1, 0.1], [0.1, 0.1]])
    bias.assign_sub([0.05, 0.05])  # 减法赋值
    
    print(f"权重增加0.1后:\n{weight}")
    print(f"偏置减少0.05后: {bias}")

def variable_initialization():
    """变量初始化方法"""
    print("\n🎯 变量初始化方法:")
    
    # 1. 直接初始化
    var1 = tf.Variable([1, 2, 3])
    print(f"直接初始化: {var1}")
    
    # 2. 从张量初始化
    tensor = tf.constant([4, 5, 6])
    var2 = tf.Variable(tensor)
    print(f"从张量初始化: {var2}")
    
    # 3. 随机初始化
    var3 = tf.Variable(tf.random.normal([2, 3], mean=0.0, stddev=1.0))
    print(f"随机初始化:\n{var3}")
    
    # 4. 零初始化
    var4 = tf.Variable(tf.zeros([2, 2]))
    print(f"零初始化:\n{var4}")
    
    # 5. 一初始化
    var5 = tf.Variable(tf.ones([3]))
    print(f"一初始化: {var5}")

def trainable_parameters():
    """可训练参数示例"""
    print("\n🎓 可训练参数:")
    
    # 创建可训练和不可训练的变量
    trainable_var = tf.Variable([1.0, 2.0], trainable=True, name="trainable")
    non_trainable_var = tf.Variable([3.0, 4.0], trainable=False, name="non_trainable")
    
    print(f"可训练变量: {trainable_var}")
    print(f"不可训练变量: {non_trainable_var}")
    print(f"可训练变量是否可训练: {trainable_var.trainable}")
    print(f"不可训练变量是否可训练: {non_trainable_var.trainable}")
    
    # 查看所有可训练变量
    all_vars = tf.trainable_variables()
    print(f"\n所有可训练变量:")
    for var in all_vars:
        if "trainable" in var.name or "non_trainable" in var.name:
            print(f"  {var.name}: {var.shape}")

def gradient_computation():
    """梯度计算示例"""
    print("\n📊 梯度计算:")
    
    # 创建变量
    x = tf.Variable(3.0, name="x")
    
    # 计算梯度
    with tf.GradientTape() as tape:
        # 函数: y = x^2 + 2x + 1
        y = x**2 + 2*x + 1
    
    # 计算dy/dx
    gradient = tape.gradient(y, x)
    
    print(f"x = {x.numpy()}")
    print(f"y = x² + 2x + 1 = {y.numpy()}")
    print(f"dy/dx = 2x + 2 = {gradient.numpy()}")

def simple_optimization():
    """简单优化示例"""
    print("\n🎯 简单优化示例:")
    
    # 创建变量
    w = tf.Variable(0.0, name="weight")
    
    # 定义损失函数: loss = (w - 3)^2
    def loss_fn():
        return (w - 3.0) ** 2
    
    # 创建优化器
    optimizer = tf.optimizers.SGD(learning_rate=0.1)
    
    print(f"初始权重: {w.numpy():.4f}")
    print(f"目标权重: 3.0")
    print(f"训练过程:")
    
    # 训练循环
    for step in range(10):
        with tf.GradientTape() as tape:
            loss = loss_fn()
        
        gradients = tape.gradient(loss, w)
        optimizer.apply_gradients([(gradients, w)])
        
        if step % 2 == 0:
            print(f"  步骤 {step}: 权重={w.numpy():.4f}, 损失={loss.numpy():.4f}")

def constants_vs_variables():
    """常量与变量对比"""
    print("\n⚖️ 常量 vs 变量对比:")
    
    # 创建常量和变量
    const = tf.constant([1, 2, 3])
    var = tf.Variable([1, 2, 3])
    
    print("特性对比:")
    print("┌─────────────┬─────────────┬─────────────┐")
    print("│   特性      │    常量     │    变量     │")
    print("├─────────────┼─────────────┼─────────────┤")
    print("│ 值可修改    │     ❌      │     ✅      │")
    print("│ 参与训练    │     ❌      │     ✅      │")
    print("│ 内存占用    │     低      │     高      │")
    print("│ 计算效率    │     高      │     中等    │")
    print("│ 用途        │   固定数据   │  模型参数   │")
    print("└─────────────┴─────────────┴─────────────┘")

def main():
    print("=" * 60)
    print("🔄 TensorFlow变量和常量示例")
    print("=" * 60)
    
    constants_demo()
    variables_demo()
    variable_operations()
    variable_initialization()
    trainable_parameters()
    gradient_computation()
    simple_optimization()
    constants_vs_variables()
    
    print("\n✅ 变量和常量示例完成!")
    print("💡 下一步: 学习基础数学运算")

if __name__ == "__main__":
    main() 