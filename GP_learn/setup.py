#!/usr/bin/env python
"""项目初始化脚本 - 创建必要的目录和文件"""

import os
import sys


def create_directory_structure():
    """创建项目目录结构"""
    
    # 定义目录结构
    directories = [
        'config',
        'data/raw',
        'data/processed',
        'data/results',
        'outputs/figures',
        'outputs/reports',
        'outputs/models',
        'src',
        'src/core',
        'src/visualization',
        'src/utils',
        'src/pipeline',
        'scripts',
        'tests'
    ]
    
    # 创建目录
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ 创建目录: {directory}")
    
    # 创建 __init__.py 文件
    init_files = [
        'src/__init__.py',
        'src/core/__init__.py',
        'src/visualization/__init__.py',
        'src/utils/__init__.py',
        'src/pipeline/__init__.py',
        'tests/__init__.py'
    ]
    
    # __init__.py 文件内容
    init_contents = {
        'src/__init__.py': '''"""量化因子挖掘系统"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
''',
        'src/core/__init__.py': '''"""核心功能模块"""

from .genetic_programming import GeneticProgrammingFactorMiner
from .factor_analyzer import FactorAnalyzer
from .data_processor import DataProcessor

__all__ = [
    "GeneticProgrammingFactorMiner",
    "FactorAnalyzer",
    "DataProcessor",
]
''',
        'src/visualization/__init__.py': '''"""可视化模块"""

from .factor_visualizer import FactorVisualizer

__all__ = ["FactorVisualizer"]
''',
        'src/utils/__init__.py': '''"""工具函数模块"""

from .logger import setup_logger
from .helpers import load_config, save_json, load_json

__all__ = ["setup_logger", "load_config", "save_json", "load_json"]
''',
        'src/pipeline/__init__.py': '''"""流程管理模块"""

from .factor_mining_pipeline import FactorMiningPipeline

__all__ = ["FactorMiningPipeline"]
''',
        'tests/__init__.py': ''
    }
    
    for file_path in init_files:
        content = init_contents.get(file_path, '')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 创建文件: {file_path}")
    
    # 创建 .gitkeep 文件
    gitkeep_files = [
        'data/raw/.gitkeep',
        'data/processed/.gitkeep',
        'data/results/.gitkeep',
        'outputs/figures/.gitkeep',
        'outputs/reports/.gitkeep',
        'outputs/models/.gitkeep'
    ]
    
    for file_path in gitkeep_files:
        with open(file_path, 'w') as f:
            f.write('')
        print(f"✓ 创建文件: {file_path}")


def check_imports():
    """检查是否可以正确导入模块"""
    print("\n检查模块导入...")
    
    # 添加当前目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        # 尝试导入主要模块
        from src.pipeline import FactorMiningPipeline
        print("✓ 成功导入 FactorMiningPipeline")
        
        from src.core import GeneticProgrammingFactorMiner, FactorAnalyzer, DataProcessor
        print("✓ 成功导入核心模块")
        
        from src.visualization import FactorVisualizer
        print("✓ 成功导入可视化模块")
        
        from src.utils import setup_logger, load_config
        print("✓ 成功导入工具模块")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False


def create_dummy_modules():
    """创建必要的模块文件（如果不存在）"""
    
    # 检查并创建缺失的模块文件
    module_files = {
        'src/core/genetic_programming.py': 'GeneticProgrammingFactorMiner',
        'src/core/factor_analyzer.py': 'FactorAnalyzer',
        'src/core/data_processor.py': 'DataProcessor',
        'src/visualization/factor_visualizer.py': 'FactorVisualizer',
        'src/utils/logger.py': 'setup_logger',
        'src/utils/helpers.py': 'load_config',
        'src/pipeline/factor_mining_pipeline.py': 'FactorMiningPipeline'
    }
    
    for file_path, class_name in module_files.items():
        if not os.path.exists(file_path):
            print(f"⚠️  文件不存在: {file_path}")
            print(f"   请从之前的代码中复制相应的模块内容到此文件")


def main():
    """主函数"""
    print("="*60)
    print("量化因子挖掘项目 - 初始化脚本")
    print("="*60)
    
    # 创建目录结构
    print("\n1. 创建目录结构...")
    create_directory_structure()
    
    # 检查模块文件
    print("\n2. 检查模块文件...")
    create_dummy_modules()
    
    # 检查导入
    print("\n3. 检查模块导入...")
    success = check_imports()
    
    if success:
        print("\n✅ 项目初始化完成！")
        print("\n下一步:")
        print("1. 将之前提供的各个模块代码复制到相应的文件中")
        print("2. 运行 pip install -r requirements.txt 安装依赖")
        print("3. 运行 python scripts/run_mining.py 开始因子挖掘")
    else:
        print("\n⚠️  项目初始化部分完成")
        print("请确保所有模块文件都已创建并包含正确的代码")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()