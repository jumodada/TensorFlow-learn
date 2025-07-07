# 03 - 变量和常量

## 学习目标

- 理解TensorFlow中变量和常量的区别
- 掌握变量的创建和操作方法
- 学习可训练参数的概念
- 了解梯度计算的基础
- 实践简单的优化过程

## 前置知识

- 完成01-hello-world和02-basic-operations章节
- 理解张量的基本概念
- Python函数和类的基础知识

## 代码文件

- `variables_demo.py` - 变量和常量对比示例

## 运行方式

```bash
# 激活虚拟环境
source tensorflow-env/bin/activate

# 运行示例
cd 03-variables-constants
python variables_demo.py
```

## 核心概念

### 1. 常量 (Constants)

```python
# 创建常量
const = tf.constant([1, 2, 3])

# 特点：
# - 值不可修改
# - 内存效率高
# - 计算图中的固定值
```

### 2. 变量 (Variables)

```python
# 创建变量
var = tf.Variable([1, 2, 3], name="my_var")

# 特点：
# - 值可以修改
# - 用于存储模型参数
# - 支持梯度计算
# - 可设置是否可训练
```

### 3. 变量操作

#### 赋值操作
```python
var = tf.Variable([1.0, 2.0])

# 直接赋值
var.assign([3.0, 4.0])

# 加法赋值
var.assign_add([0.1, 0.1])

# 减法赋值
var.assign_sub([0.05, 0.05])
```

#### 初始化方法
```python
# 随机初始化
var1 = tf.Variable(tf.random.normal([2, 3]))

# 零初始化
var2 = tf.Variable(tf.zeros([2, 2]))

# 一初始化
var3 = tf.Variable(tf.ones([3]))
```

### 4. 可训练参数

```python
# 可训练变量（默认）
trainable_var = tf.Variable([1.0, 2.0], trainable=True)

# 不可训练变量
non_trainable_var = tf.Variable([3.0, 4.0], trainable=False)

# 查看所有可训练变量
all_vars = tf.trainable_variables()
```

### 5. 梯度计算

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2 + 2*x + 1

# 计算梯度
gradient = tape.gradient(y, x)
```

## 实际应用

### 线性模型参数
```python
# 定义模型参数
W = tf.Variable(tf.random.normal([input_dim, output_dim]), name="weights")
b = tf.Variable(tf.zeros([output_dim]), name="bias")

# 前向传播
def linear_model(x):
    return tf.matmul(x, W) + b
```

### 简单优化过程
```python
# 定义损失函数
def loss_fn():
    return (w - target)**2

# 创建优化器
optimizer = tf.optimizers.SGD(learning_rate=0.1)

# 训练步骤
with tf.GradientTape() as tape:
    loss = loss_fn()

gradients = tape.gradient(loss, w)
optimizer.apply_gradients([(gradients, w)])
```

## 常见问题

### Q: 什么时候使用变量，什么时候使用常量？
A: 
- **常量**: 用于固定的数据，如输入数据、超参数
- **变量**: 用于模型参数，需要在训练过程中更新的值

### Q: 为什么需要GradientTape？
A: GradientTape记录前向传播过程中的操作，用于自动微分计算梯度。

### Q: trainable=False有什么作用？
A: 设置为False的变量不会参与梯度计算和参数更新，常用于冻结某些层。

### Q: 变量初始化有什么最佳实践？
A: 
- 权重通常用随机初始化（如Xavier、He初始化）
- 偏置通常用零初始化
- 避免所有参数初始化为相同值

## 练习建议

1. **基础练习**
   - 创建不同类型的变量和常量
   - 练习各种赋值操作
   - 观察可训练和不可训练变量的区别

2. **梯度计算练习**
   - 尝试不同的数学函数
   - 计算多变量函数的梯度
   - 理解链式法则的应用

3. **优化练习**
   - 实现简单的梯度下降
   - 尝试不同的学习率
   - 观察收敛过程

4. **实际应用练习**
   - 创建简单的线性模型
   - 实现参数更新逻辑
   - 可视化训练过程

## 下一步

完成本章节后，继续学习：
- **04-basic-math**: 基础数学运算
- 为后续的机器学习模型打下基础 