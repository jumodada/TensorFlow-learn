# TensorFlow 学习项目

TensorFlow学习

## 环境要求

- **服务器端**: Ubuntu 20.04+ (推荐 Ubuntu 24.04.2 LTS)
- Python 3.9+ (已测试: 3.9, 3.10, 3.11, 3.12)
- TensorFlow 2.17+ (最新稳定版)

> 💡 **注意**: 
> - 本项目设计为在Ubuntu服务器上运行，你可以通过SSH从Mac连接到Ubuntu服务器进行学习
> - Ubuntu 24.04对Python包管理有新的安全机制，强烈建议使用虚拟环境
> - 支持CPU和GPU版本的TensorFlow

## 快速开始

### 1. SSH连接到Ubuntu服务器

推荐软件： Termius

### 2. 克隆项目到服务器

```bash
# 克隆项目（如果从Git仓库）
git clone https://github.com/jumodada/TensorFlow-learn.git
cd TensorFlow-learn

# 或直接创建项目目录
mkdir TensorFlow-learn
cd TensorFlow-learn
```

### 3. 环境安装（在Ubuntu服务器上）

```bash
# 方法1: 使用自动安装脚本
chmod +x setup.sh
./setup.sh

# 方法2: 手动安装
# 更新系统包
sudo apt update && sudo apt upgrade -y

# 安装Python和必要依赖
sudo apt install python3 python3-pip python3-venv python3-full python3-dev -y

# 创建虚拟环境
python3 -m venv tensorflow-env
source tensorflow-env/bin/activate

# 升级pip和安装依赖
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. 验证安装

```bash
# 激活虚拟环境
source tensorflow-env/bin/activate

# 运行HelloWorld
cd 01-hello-world
python hello_tensorflow.py
```

## 学习路线

### 阶段1: 基础入门 (1-2周)
- **01-hello-world**: TensorFlow基础安装和验证
- **02-basic-operations**: 基本操作和张量操作
- **03-variables-constants**: 变量和常量的使用
- **04-basic-math**: 基础数学运算

### 阶段2: 线性回归和分类 (2-3周)
- **05-linear-regression**: 线性回归实现
- **06-logistic-regression**: 逻辑回归和二分类
- **07-mnist-basic**: MNIST手写数字识别入门
- **08-data-preprocessing**: 数据预处理技巧

### 阶段3: 神经网络基础 (3-4周)
- **09-neural-network**: 基础神经网络
- **10-activation-functions**: 激活函数详解
- **11-loss-functions**: 损失函数和优化器
- **12-model-evaluation**: 模型评估和指标

### 阶段4: 卷积神经网络 (4-5周)
- **13-cnn-basics**: CNN基础概念
- **14-image-classification**: 图像分类实战
- **15-transfer-learning**: 迁移学习
- **16-data-augmentation**: 数据增强技术

### 阶段5: 循环神经网络 (5-6周)
- **17-rnn-basics**: RNN基础
- **18-lstm-gru**: LSTM和GRU详解
- **19-text-classification**: 文本分类
- **20-sequence-prediction**: 序列预测

### 阶段6: 高级主题 (6-8周)
- **21-autoencoder**: 自编码器
- **22-gan-basics**: 生成对抗网络入门
- **23-reinforcement-learning**: 强化学习基础
- **24-model-deployment**: 模型部署和服务

### 阶段7: 实战项目 (8-12周)
- **25-computer-vision-project**: 计算机视觉项目
- **26-nlp-project**: 自然语言处理项目
- **27-time-series-project**: 时间序列预测项目
- **28-end-to-end-ml**: 端到端机器学习流程

## 项目结构

```
TensorFlow-learn/
├── README.md                    # 项目说明和学习路线
├── requirements.txt             # Python依赖包
├── setup.sh                    # 环境安装脚本
├── 01-hello-world/             # HelloWorld入门
├── 02-basic-operations/         # 基本操作
├── 03-variables-constants/      # 变量和常量
├── 04-basic-math/              # 基础数学运算
├── 05-linear-regression/        # 线性回归
├── 06-logistic-regression/      # 逻辑回归
├── 07-mnist-basic/             # MNIST基础
├── 08-data-preprocessing/       # 数据预处理
├── 09-neural-network/          # 神经网络基础
├── 10-activation-functions/     # 激活函数
├── 11-loss-functions/          # 损失函数
├── 12-model-evaluation/        # 模型评估
├── 13-cnn-basics/              # CNN基础
├── 14-image-classification/     # 图像分类
├── 15-transfer-learning/        # 迁移学习
├── 16-data-augmentation/        # 数据增强
├── 17-rnn-basics/              # RNN基础
├── 18-lstm-gru/                # LSTM和GRU
├── 19-text-classification/      # 文本分类
├── 20-sequence-prediction/      # 序列预测
├── 21-autoencoder/             # 自编码器
├── 22-gan-basics/              # GAN基础
├── 23-reinforcement-learning/   # 强化学习
├── 24-model-deployment/        # 模型部署
├── 25-computer-vision-project/ # 计算机视觉项目
├── 26-nlp-project/             # NLP项目
├── 27-time-series-project/     # 时间序列项目
├── 28-end-to-end-ml/           # 端到端ML
└── utils/                      # 通用工具函数
```

## 当前完成的章节

### ✅ 已完成的学习内容

- **01-hello-world**: TensorFlow环境验证和基础操作 
  - `hello_tensorflow.py` - HelloWorld示例
  - 涵盖：版本检查、GPU检测、基本张量操作、数学运算

- **02-basic-operations**: 张量基本操作
  - `tensor_operations.py` - 张量操作大全
  - 涵盖：创建、形状操作、索引切片、拼接分割、数据类型

- **03-variables-constants**: 变量和常量
  - `variables_demo.py` - 变量与常量对比
  - 涵盖：变量操作、可训练参数、梯度计算、简单优化

- **04-basic-math**: 基础数学运算  
  - `math_operations.py` - 数学函数大全
  - 涵盖：算术、矩阵、归约、三角函数、统计、比较逻辑

- **utils**: 通用工具函数
  - `common_utils.py` - 项目通用工具
  - 涵盖：数据生成、可视化、标准化、评估指标

### 🔄 学习路径建议

1. **第1周**: 完成章节01-04，掌握TensorFlow基础
2. **每日练习**: 至少运行一个示例，修改参数观察变化  
3. **实践记录**: 在每个目录下记录学习心得
4. **问题解决**: 遇到问题先查看README，再查官方文档

## 使用方法

### 运行示例代码

```bash
# 激活环境
source tensorflow-env/bin/activate

# 进入对应章节目录
cd 01-hello-world

# 运行Python文件  
python hello_tensorflow.py

# 查看章节说明
cat README.md
```

### 学习建议

1. **循序渐进**: 按照编号顺序学习，每个阶段都有具体的代码示例
2. **动手实践**: 每个示例都要亲自运行和修改
3. **记录笔记**: 在每个目录下记录学习心得和问题
4. **扩展练习**: 尝试修改参数，观察结果变化
5. **项目实战**: 完成最后的实战项目来巩固所学

## 资源推荐

- [TensorFlow官方文档](https://www.tensorflow.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras官方指南](https://keras.io/guides/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

## 常见问题

查看各个目录下的README.md文件获取具体的问题解答和使用说明。

---

**开始你的TensorFlow学习之旅吧！** 🚀 