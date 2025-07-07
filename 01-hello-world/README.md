# 01 - TensorFlow HelloWorld

## 学习目标

- 验证TensorFlow环境安装
- 了解TensorFlow基础概念
- 掌握基本张量操作
- 熟悉TensorFlow的运行方式

## 前置知识

- Python基础语法
- 基本的命令行操作

## 代码文件

- `hello_tensorflow.py` - TensorFlow的第一个程序

## 运行方式

### 1. 激活虚拟环境
```bash
source tensorflow-env/bin/activate
```

### 2. 运行HelloWorld程序
```bash
cd 01-hello-world
python hello_tensorflow.py
```

## 预期输出

程序会输出以下信息：
- TensorFlow版本信息
- GPU/CPU硬件检查结果
- 基本张量操作示例
- 数学运算演示
- 矩阵运算示例
- 随机数生成示例

## 核心概念

### 1. 张量 (Tensor)
- TensorFlow的核心数据结构
- 可以是标量、向量、矩阵或高维数组
- 不可变的多维数组

### 2. 常量张量
```python
# 创建不同类型的常量张量
scalar = tf.constant(42)          # 标量
vector = tf.constant([1, 2, 3])   # 向量  
matrix = tf.constant([[1, 2], [3, 4]])  # 矩阵
```

### 3. 基本运算
- 算术运算：`tf.add()`, `tf.subtract()`, `tf.multiply()`, `tf.divide()`
- 矩阵运算：`tf.matmul()` (矩阵乘法)
- 随机数：`tf.random.normal()`, `tf.random.uniform()`

## 常见问题

### Q: 没有检测到GPU怎么办？
A: 这是正常的，CPU也可以运行TensorFlow。如果需要GPU加速，需要安装CUDA和cuDNN。

### Q: 版本不匹配怎么办？
A: 确保使用的是TensorFlow 2.x版本，如果是1.x版本，很多API会不同。

### Q: ImportError: No module named 'tensorflow'
A: 检查是否激活了虚拟环境，并且已经安装了TensorFlow。

## 下一步

完成本章节后，继续学习：
- **02-basic-operations**: 张量的基本操作
- 学习更多张量操作和数据类型

## 练习建议

1. 修改代码中的数值，观察输出变化
2. 尝试创建不同形状的张量
3. 实验不同的数学运算
4. 查看TensorFlow官方文档了解更多API 