#!/usr/bin/env python
"""生成可视化图表"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 12})


def generate_demo_plots():
    """生成演示用的可视化图表"""
    
    output_dir = 'outputs/figures/demo/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 进化过程分析图
    print("正在生成图1: 进化过程分析图...")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    generations = np.arange(0, 51)
    np.random.seed(42)
    fitness_values = []
    initial_fitness = 0.10
    
    for gen in generations:
        if gen <= 15:
            fitness = initial_fitness + (0.25 - initial_fitness) * (gen / 15) + np.random.normal(0, 0.01)
        elif gen <= 35:
            base_fitness = 0.25 + (0.30 - 0.25) * ((gen - 15) / 20)
            fitness = base_fitness + np.random.normal(0, 0.008)
        else:
            fitness = 0.31 + np.random.normal(0, 0.005)
        fitness_values.append(max(fitness, initial_fitness))
    
    ax1.plot(generations, fitness_values, 'b-', linewidth=2, label='最优适应度', marker='o', markersize=3)
    ax1.plot(generations, np.convolve(fitness_values, np.ones(5)/5, mode='same'), 'r--', linewidth=2, label='5代移动平均')
    ax1.set_xlabel('进化代数', fontsize=12)
    ax1.set_ylabel('适应度', fontsize=12)
    ax1.set_title('进化过程分析', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax1.axvspan(0, 15, alpha=0.1, color='green', label='初始阶段')
    ax1.axvspan(15, 35, alpha=0.1, color='yellow')
    ax1.axvspan(35, 50, alpha=0.1, color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_evolution_process.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. IC时序稳定性图
    print("正在生成图2: IC时序稳定性图...")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    ic_values = np.random.normal(0.0312, 0.0458, len(dates))
    ic_20d_ma = pd.Series(ic_values).rolling(window=20).mean()
    
    ax2.plot(dates, ic_values, alpha=0.6, color='lightblue', linewidth=0.8, label='日IC值')
    ax2.plot(dates, ic_20d_ma, color='navy', linewidth=2, label='20日移动平均')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='零线')
    ax2.axhline(y=0.02, color='green', linestyle='--', alpha=0.7, label='目标线(0.02)')
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('IC值', fontsize=12)
    ax2.set_title('Factor #1 IC时序稳定性', fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=11)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_ic_time_series.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 分组回测表现图
    print("正在生成图3: 分组回测表现图...")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    groups = ['G1(空)', 'G2', 'G3', 'G4', 'G5(多)']
    annual_returns = [-8.2, -2.1, 3.6, 9.8, 16.3]
    colors = ['#FF6B6B', '#FFA500', '#FFD700', '#90EE90', '#32CD32']
    
    bars = ax3.bar(groups, annual_returns, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('分组', fontsize=12)
    ax3.set_ylabel('年化收益率 (%)', fontsize=12)
    ax3.set_title('分组回测表现', fontsize=16, fontweight='bold', pad=20)
    
    for bar, value in zip(bars, annual_returns):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1.5),
                 f'{value}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_group_backtest.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 因子对比雷达图
    print("正在生成图4: 因子对比雷达图...")
    fig4, ax4 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['预测能力\n(IC均值)', '稳定性\n(IR)', '收益能力\n(日收益)', '风险调整\n(夏普比率)', '换手率控制\n(反向)']
    gp_factor = [0.0312/0.0156*100, 0.681/0.250*100, 0.124/0.045*100, 1.55/0.45*100, (1-0.235/0.4)*100]
    traditional_factor = [100, 100, 100, 100, 100]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    gp_factor += gp_factor[:1]
    traditional_factor += traditional_factor[:1]
    
    ax4.plot(angles, gp_factor, 'o-', linewidth=3, label='GP挖掘因子', color='red', markersize=8)
    ax4.fill(angles, gp_factor, alpha=0.25, color='red')
    ax4.plot(angles, traditional_factor, 'o-', linewidth=3, label='传统因子(基准)', color='blue', markersize=8)
    ax4.fill(angles, traditional_factor, alpha=0.25, color='blue')
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=11)
    ax4.set_ylim(0, 300)
    ax4.set_title('因子对比分析\n(相对传统因子提升比例)', fontsize=16, fontweight='bold', pad=30)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_factor_comparison_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 所有演示图表已保存到 '{output_dir}' 文件夹中")
    
    # 输出统计表格
    print("\n" + "="*80)
    print("因子统计特征对比表")
    print("="*80)
    
    comparison_data = {
        '指标': ['IC均值', 'IC标准差', 'IC_IR', 'IC正率', '多空收益(日)', '换手率'],
        'Factor #1': ['0.0312', '0.0458', '0.681', '58.7%', '0.124%', '23.5%'],
        'Factor #2': ['0.0287', '0.0512', '0.561', '56.2%', '0.098%', '19.8%'],
        'Factor #3': ['0.0269', '0.0487', '0.552', '55.4%', '0.087%', '21.2%'],
        '基准(MA20)': ['0.0156', '0.0623', '0.250', '51.3%', '0.045%', '15.6%']
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))


if __name__ == "__main__":
    generate_demo_plots()