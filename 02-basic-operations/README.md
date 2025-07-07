# 02 - 张量基本操作

## 学习目标

- 掌握张量的多种创建方法
- 理解张量的属性和形状操作
- 学会张量的索引和切片
- 熟悉张量的拼接和分割
- 了解数据类型转换

## 前置知识

- 完成01-hello-world章节
- 了解NumPy基础操作
- Python列表和数组概念

## 代码文件

- `tensor_operations.py` - 张量基本操作示例

## 运行方式

```bash
# 激活虚拟环境
source tensorflow-env/bin/activate

# 运行示例
cd 02-basic-operations
python tensor_operations.py
```

## 核心概念

### 1. 张量创建

#### 从现有数据创建
```python
# 从Python列表
tf.constant([1, 2, 3])

# 从NumPy数组
np_array = np.array([[1, 2], [3, 4]])
tf.constant(np_array)
```

#### 创建特殊张量
```python
tf.zeros([2, 3])      # 全零张量
tf.ones([2, 3])       # 全一张量
tf.eye(3)             # 单位矩阵
tf.random.normal([2, 3])  # 随机张量
```

### 2. 张量属性

- `shape`: 张量形状
- `ndim`: 维度数量
- `dtype`: 数据类型
- `size`: 元素总数

### 3. 形状操作

- `tf.reshape()`: 改变形状
- `tf.transpose()`: 转置
- `tf.expand_dims()`: 增加维度
- `tf.squeeze()`: 压缩维度

### 4. 索引和切片

```python
tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# 基本索引
tensor[0]        # 第一个切片
tensor[0, 1, 0]  # 具体元素

# 切片
tensor[:2]       # 前两个切片
tensor[:, 0, :]  # 每个切片的第一行
```

### 5. 张量拼接

- `tf.concat()`: 沿现有轴拼接
- `tf.stack()`: 创建新轴堆叠
- `tf.split()`: 分割张量

## 实际应用

### 数据预处理
```python
# 标准化数据
data = tf.constant([[1.0, 2.0], [3.0, 4.0]])
normalized = (data - tf.reduce_mean(data)) / tf.math.reduce_std(data)
```

### 批处理操作
```python
# 批量处理图像数据
images = tf.random.normal([32, 224, 224, 3])  # [batch, height, width, channels]
resized = tf.image.resize(images, [128, 128])
```

## 常见问题

### Q: 形状不匹配错误
A: 确保操作的张量形状兼容，使用`tf.reshape()`调整形状。

### Q: 索引超出范围
A: 检查张量的shape，确保索引在有效范围内。

### Q: 数据类型不匹配
A: 使用`tf.cast()`进行类型转换。

## 练习建议

1. **张量创建练习**
   - 创建不同形状和类型的张量
   - 尝试从不同数据源创建张量

2. **形状操作练习**
   - 练习reshape操作
   - 理解转置的作用

3. **索引切片练习**
   - 提取张量的特定部分
   - 使用条件索引过滤数据

4. **实际应用练习**
   - 模拟图像数据处理
   - 实现简单的数据标准化

## 下一步

完成本章节后，继续学习：
- **03-variables-constants**: 变量和常量的区别
- 学习TensorFlow中的可训练参数 