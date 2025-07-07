#!/usr/bin/env python3
"""
TensorFlow环境验证脚本
快速检查TensorFlow学习环境是否正确安装和配置
"""

import sys
import subprocess
import platform

def check_python_version():
    """检查Python版本"""
    print("🐍 Python版本检查:")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("   ✅ Python版本符合要求 (3.9+)")
        return True
    else:
        print("   ❌ Python版本过低，需要3.9+")
        return False

def check_tensorflow():
    """检查TensorFlow安装"""
    print("\n📊 TensorFlow安装检查:")
    try:
        import tensorflow as tf
        print(f"   ✅ TensorFlow版本: {tf.__version__}")
        
        # 检查GPU支持
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"   🚀 GPU支持: 检测到 {len(gpu_devices)} 个GPU设备")
            for i, gpu in enumerate(gpu_devices):
                print(f"      GPU {i}: {gpu.name}")
        else:
            print("   💻 GPU支持: 未检测到GPU，使用CPU模式")
        
        # 简单计算测试
        test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = tf.reduce_sum(test_tensor)
        print(f"   ✅ 基本运算测试: {result.numpy()}")
        
        return True
    except ImportError:
        print("   ❌ TensorFlow未安装")
        return False
    except Exception as e:
        print(f"   ❌ TensorFlow测试失败: {e}")
        return False

def check_dependencies():
    """检查主要依赖包"""
    print("\n📦 依赖包检查:")
    dependencies = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn')
    ]
    
    all_good = True
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', '未知版本')
            print(f"   ✅ {display_name}: {version}")
        except ImportError:
            print(f"   ❌ {display_name}: 未安装")
            all_good = False
    
    return all_good

def check_system_info():
    """显示系统信息"""
    print("\n🖥️  系统信息:")
    print(f"   操作系统: {platform.system()} {platform.release()}")
    print(f"   架构: {platform.machine()}")
    print(f"   Python路径: {sys.executable}")

def check_virtual_environment():
    """检查是否在虚拟环境中"""
    print("\n🌐 虚拟环境检查:")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   ✅ 运行在虚拟环境中")
        print(f"   环境路径: {sys.prefix}")
        return True
    else:
        print("   ⚠️  未在虚拟环境中运行")
        print("   建议: 在Ubuntu 24.04上建议使用虚拟环境")
        return False

def run_tensorflow_hello_world():
    """运行一个简单的TensorFlow示例"""
    print("\n🚀 TensorFlow HelloWorld测试:")
    try:
        import tensorflow as tf
        
        # 创建常量
        hello = tf.constant('Hello, TensorFlow!')
        try:
            hello_val = hello.numpy().decode('utf-8')
        except AttributeError:
            hello_val = str(hello.numpy())
        print(f"   字符串张量: {hello_val}")
        
        # 数学运算
        a = tf.constant(5)
        b = tf.constant(3)
        result = tf.add(a, b)
        print(f"   数学运算 (5+3): {result.numpy()}")
        
        # 矩阵运算
        matrix1 = tf.constant([[1, 2], [3, 4]])
        matrix2 = tf.constant([[5, 6], [7, 8]])
        product = tf.matmul(matrix1, matrix2)
        print(f"   矩阵乘法结果:\n{product.numpy()}")
        
        print("   ✅ TensorFlow基本功能测试通过")
        return True
    except Exception as e:
        print(f"   ❌ TensorFlow测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🔍 TensorFlow学习环境验证")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_virtual_environment(),
        check_tensorflow(),
        check_dependencies(),
        run_tensorflow_hello_world()
    ]
    
    check_system_info()
    
    print("\n" + "=" * 60)
    print("📋 验证结果总结:")
    print("=" * 60)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print("🎉 所有检查通过！环境配置正确，可以开始学习TensorFlow了！")
        print("\n💡 下一步:")
        print("   cd 01-hello-world")
        print("   python hello_tensorflow.py")
    else:
        print(f"⚠️  {total - passed} 项检查未通过，请检查安装配置")
        print("\n🔧 建议:")
        if not checks[0]:  # Python版本
            print("   - 升级Python到3.9+版本")
        if not checks[1]:  # 虚拟环境
            print("   - 创建并激活虚拟环境")
        if not checks[2]:  # TensorFlow
            print("   - 安装TensorFlow: pip install tensorflow")
        if not checks[3]:  # 依赖
            print("   - 安装依赖: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 