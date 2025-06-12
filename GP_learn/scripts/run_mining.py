#!/usr/bin/env python
"""运行因子挖掘主程序"""

import os
import sys
import argparse

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.pipeline import FactorMiningPipeline


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行量化因子挖掘系统')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='配置文件路径'
    )
    args = parser.parse_args()
    
    # 创建并运行流程
    pipeline = FactorMiningPipeline(config_path=args.config)
    factor_results = pipeline.run_pipeline()
    
    # 打印结果摘要
    print("\n" + "="*60)
    print("因子挖掘结果摘要")
    print("="*60)
    
    for result in factor_results:
        print(f"\n因子名称: {result['factor_name']}")
        print(f"IC均值: {result['ic_mean']:.4f}")
        print(f"IC_IR: {result['ic_ir']:.4f}")
        print(f"IC正率: {result['ic_positive_ratio']:.2%}")
        print(f"多空收益: {result['group_return_spread']:.2%}")
        print(f"平均换手率: {result['turnover_mean']:.2%}")
        print("-"*40)
    
    print(f"\n结果已保存到 outputs/ 目录")


if __name__ == "__main__":
    main()