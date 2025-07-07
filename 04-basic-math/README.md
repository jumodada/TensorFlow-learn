# 04 - 基础数学运算

## 学习目标

- 掌握TensorFlow中的各种数学运算
- 学习矩阵运算和线性代数操作
- 了解统计函数和归约操作
- 熟悉比较和逻辑运算
- 掌握条件操作和数据标准化

## 前置知识

- 完成前三个章节的学习
- 基础的数学知识（线性代数、统计学）
- 理解张量和变量的概念

## 代码文件

- `math_operations.py` - 数学运算综合示例

## 运行方式

```bash
# 激活虚拟环境
source tensorflow-env/bin/activate

# 运行示例
cd 04-basic-math
python math_operations.py
```

## 核心概念

### 1. 基础算术运算

```python
a = tf.constant([2, 4, 6, 8])
b = tf.constant([1, 2, 3, 4])

# 四则运算
tf.add(a, b)        # 加法，等同于 a + b
tf.subtract(a, b)   # 减法，等同于 a - b
tf.multiply(a, b)   # 乘法，等同于 a * b
tf.divide(a, b)     # 除法，等同于 a / b

# 其他运算
tf.pow(a, 2)        # 平方
tf.sqrt(a)          # 平方根
tf.mod(a, 3)        # 取模
```

### 2. 矩阵运算

```python
A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
B = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

# 矩阵乘法
tf.matmul(A, B)

# 矩阵转置
tf.transpose(A)

# 行列式
tf.linalg.det(A)

# 矩阵求逆
tf.linalg.inv(A)
```

### 3. 归约操作

```python
data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

# 求和
tf.reduce_sum(data)          # 所有元素
tf.reduce_sum(data, axis=0)  # 沿轴0（列）
tf.reduce_sum(data, axis=1)  # 沿轴1（行）

# 其他归约
tf.reduce_mean(data)  # 平均值
tf.reduce_max(data)   # 最大值
tf.reduce_min(data)   # 最小值
tf.math.reduce_std(data)  # 标准差
```

### 4. 三角函数和指数对数

```python
angles = tf.constant([0, np.pi/4, np.pi/2])

# 三角函数
tf.sin(angles)
tf.cos(angles)
tf.tan(angles)

# 指数和对数
x = tf.constant([1, 2, np.e, 10])
tf.exp(x)           # e^x
tf.math.log(x)      # ln(x)
```

### 5. 比较和逻辑运算

```python
a = tf.constant([1, 3, 5])
b = tf.constant([2, 3, 4])

# 比较运算
tf.equal(a, b)      # ==
tf.greater(a, b)    # >
tf.less(a, b)       # <

# 逻辑运算
tf.logical_and(a > 2, b < 5)
tf.logical_or(a > 2, b < 5)
tf.logical_not(a > 2)
```

### 6. 条件操作

```python
condition = tf.constant([True, False, True])
x = tf.constant([1, 2, 3])
y = tf.constant([10, 20, 30])

# 条件选择
tf.where(condition, x, y)  # 条件为True选x，否则选y
```

## 实际应用示例

### 数据标准化

```python
# Z-score标准化
def standardize(data):
    mean = tf.reduce_mean(data)
    std = tf.math.reduce_std(data)
    return (data - mean) / std

# Min-Max标准化
def normalize(data):
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)
    return (data - min_val) / (max_val - min_val)
```

### 距离计算

```python
def euclidean_distance(x1, x2):
    """计算欧几里得距离"""
    return tf.sqrt(tf.reduce_sum(tf.square(x1 - x2)))

def cosine_similarity(x1, x2):
    """计算余弦相似度"""
    dot_product = tf.reduce_sum(x1 * x2)
    norm1 = tf.norm(x1)
    norm2 = tf.norm(x2)
    return dot_product / (norm1 * norm2)
```

### 激活函数

```python
def relu(x):
    """ReLU激活函数"""
    return tf.maximum(0.0, x)

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1.0 / (1.0 + tf.exp(-x))

def tanh(x):
    """Tanh激活函数"""
    return tf.tanh(x)
```

## 性能优化建议

### 1. 向量化操作
```python
# 推荐：向量化操作
result = tf.reduce_sum(x * y)

# 避免：循环操作
result = 0
for i in range(len(x)):
    result += x[i] * y[i]
```

### 2. 数据类型选择
```python
# 根据需要选择合适的数据类型
tf.float32  # 一般用途
tf.float16  # 节省内存
tf.int32    # 整数运算
```

### 3. 内存管理
```python
# 使用tf.function装饰器提升性能
@tf.function
def compute_heavy_operation(x, y):
    return tf.matmul(x, y) + tf.reduce_sum(x)
```

## 常见问题

### Q: 为什么要做数据标准化？
A: 
- 不同特征的量纲不同时，需要标准化保证公平性
- 有助于梯度下降算法的收敛
- 防止某些特征主导模型训练

### Q: 什么时候使用不同的归约操作？
A: 
- `reduce_sum`: 累加，常用于损失函数
- `reduce_mean`: 平均值，常用于评估指标
- `reduce_max/min`: 最值，常用于池化操作

### Q: 矩阵运算中的广播是什么？
A: 广播允许不同形状的张量进行运算，TensorFlow会自动扩展维度。

## 练习建议

1. **基础运算练习**
   - 实现各种数学函数
   - 对比NumPy和TensorFlow的性能差异
   - 练习不同数据类型的运算

2. **矩阵操作练习**
   - 实现线性代数基本操作
   - 练习矩阵分解
   - 理解广播机制

3. **实际应用练习**
   - 实现数据预处理函数
   - 编写常用的激活函数
   - 实现距离和相似度计算

4. **性能优化练习**
   - 对比不同实现方式的性能
   - 使用tf.function优化代码
   - 测试不同数据类型的内存占用

## 下一步

完成本章节后，你已经掌握了TensorFlow的基础操作，可以继续学习：
- **05-linear-regression**: 线性回归实战
- 开始真正的机器学习模型实现

## 总结

本章节涵盖了TensorFlow中最重要的数学运算，这些都是后续机器学习和深度学习的基础。确保你熟练掌握这些操作，它们将在所有后续项目中频繁使用。 