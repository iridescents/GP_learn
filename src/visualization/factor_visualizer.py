"""因子可视化工具"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class FactorVisualizer:
    """因子可视化工具"""
    
    def __init__(self, figsize=(15, 10), dpi=300):
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_factor_performance_dashboard(self, factor_results: Dict, 
                                        save_path: Optional[str] = None):
        """
        绘制因子表现综合仪表板
        
        Args:
            factor_results: 因子评估结果
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. IC时间序列图
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_ic_series(factor_results['ic_series'], ax1)
        
        # 2. IC分布直方图
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_ic_distribution(factor_results['ic_series'], ax2)
        
        # 3. 分组收益累计曲线
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_group_cumulative_returns(factor_results['group_returns'], ax3)
        
        # 4. 分组收益柱状图
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_group_returns_bar(factor_results['group_returns'], ax4)
        
        # 5. 换手率时间序列
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_turnover_series(factor_results['turnover_series'], ax5)
        
        # 6. 因子统计指标表
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_factor_stats(factor_results, ax6)
        
        # 7. IC滚动统计
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_rolling_ic_stats(factor_results['ic_series'], ax7)
        
        plt.suptitle(f"因子表现综合分析 - {factor_results['factor_name']}", 
                    fontsize=16, y=0.995)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_ic_series(self, ic_series: pd.Series, ax):
        """绘制IC时间序列"""
        if len(ic_series) == 0:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=14)
            ax.set_title('IC时间序列', fontsize=14)
            return
            
        ic_series.plot(ax=ax, color=self.color_palette[0], alpha=0.7, linewidth=1)
        
        # 添加移动平均线
        if len(ic_series) >= 20:
            ic_ma20 = ic_series.rolling(20).mean()
            ic_ma20.plot(ax=ax, color=self.color_palette[1], linewidth=2, label='20日移动平均')
        
        # 添加零线
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 填充正负区域
        ax.fill_between(ic_series.index, 0, ic_series.values, 
                       where=(ic_series.values > 0), alpha=0.3, color='green', label='IC > 0')
        ax.fill_between(ic_series.index, 0, ic_series.values, 
                       where=(ic_series.values <= 0), alpha=0.3, color='red', label='IC < 0')
        
        ax.set_title('IC时间序列', fontsize=14)
        ax.set_xlabel('日期')
        ax.set_ylabel('IC值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        if len(ic_series) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_ic_distribution(self, ic_series: pd.Series, ax):
        """绘制IC分布直方图"""
        if len(ic_series) == 0:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=14)
            ax.set_title('IC分布', fontsize=14)
            return
            
        ic_series.hist(bins=50, ax=ax, color=self.color_palette[0], 
                      alpha=0.7, edgecolor='black')
        
        # 添加统计信息
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        ax.axvline(mean_ic, color='red', linestyle='--', linewidth=2, 
                  label=f'均值: {mean_ic:.4f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # 添加正态分布拟合曲线
        try:
            from scipy import stats
            x = np.linspace(ic_series.min(), ic_series.max(), 100)
            ax2 = ax.twinx()
            ax2.plot(x, stats.norm.pdf(x, mean_ic, std_ic), 'r-', 
                    linewidth=2, label='正态分布')
            ax2.set_ylabel('概率密度')
        except ImportError:
            pass
        
        ax.set_title('IC分布', fontsize=14)
        ax.set_xlabel('IC值')
        ax.set_ylabel('频数')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_group_cumulative_returns(self, group_returns: pd.DataFrame, ax):
        """绘制分组累计收益曲线"""
        if len(group_returns) == 0:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=14)
            ax.set_title('分组累计收益曲线', fontsize=14)
            return
            
        cumulative_returns = (1 + group_returns).cumprod()
        
        for i, col in enumerate(cumulative_returns.columns):
            cumulative_returns[col].plot(
                ax=ax, label=f'组{col}', 
                color=self.color_palette[i % len(self.color_palette)],
                linewidth=2
            )
        
        # 添加多空组合
        if 1 in group_returns.columns and group_returns.columns.max() in group_returns.columns:
            max_group = group_returns.columns.max()
            long_short = group_returns[max_group] - group_returns[1]
            long_short_cum = (1 + long_short).cumprod()
            long_short_cum.plot(ax=ax, label=f'多空组合(G{max_group}-G1)', 
                              color='black', linewidth=3, linestyle='--')
        
        ax.set_title('分组累计收益曲线', fontsize=14)
        ax.set_xlabel('日期')
        ax.set_ylabel('累计收益')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        if len(group_returns) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_group_returns_bar(self, group_returns: pd.DataFrame, ax):
        """绘制分组平均收益柱状图"""
        if len(group_returns) == 0:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=14)
            ax.set_title('分组平均收益', fontsize=14)
            return
            
        mean_returns = group_returns.mean()
        colors = [self.color_palette[i % len(self.color_palette)] 
                 for i in range(len(mean_returns))]
        
        bars = ax.bar(mean_returns.index, mean_returns.values, 
                      color=colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom')
        
        ax.set_title('分组平均收益', fontsize=14)
        ax.set_xlabel('组别')
        ax.set_ylabel('平均日收益率')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加趋势线
        if len(mean_returns) > 1:
            z = np.polyfit(mean_returns.index, mean_returns.values, 1)
            p = np.poly1d(z)
            ax.plot(mean_returns.index, p(mean_returns.index), 
                   "r--", alpha=0.8, linewidth=2)
    
    def _plot_turnover_series(self, turnover_series: pd.Series, ax):
        """绘制换手率时间序列"""
        if len(turnover_series) == 0:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=14)
            ax.set_title('因子换手率时间序列', fontsize=14)
            return
            
        turnover_series.plot(ax=ax, color=self.color_palette[2], 
                           alpha=0.7, linewidth=1)
        
        # 添加移动平均线
        if len(turnover_series) >= 20:
            turnover_ma20 = turnover_series.rolling(20).mean()
            turnover_ma20.plot(ax=ax, color=self.color_palette[3], 
                             linewidth=2, label='20日移动平均')
        
        # 添加平均线
        mean_turnover = turnover_series.mean()
        ax.axhline(y=mean_turnover, color='red', linestyle='--', 
                  label=f'平均换手率: {mean_turnover:.1%}')
        
        ax.set_title('因子换手率时间序列', fontsize=14)
        ax.set_xlabel('日期')
        ax.set_ylabel('换手率')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 格式化y轴为百分比
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # 格式化x轴日期
        if len(turnover_series) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_factor_stats(self, factor_results: Dict, ax):
        """绘制因子统计指标表"""
        ax.axis('off')
        
        # 准备统计数据
        stats_data = [
            ['指标', '数值'],
            ['IC均值', f"{factor_results['ic_mean']:.4f}"],
            ['IC标准差', f"{factor_results['ic_std']:.4f}"],
            ['IC_IR', f"{factor_results['ic_ir']:.4f}"],
            ['IC正率', f"{factor_results['ic_positive_ratio']:.2%}"],
            ['多空收益', f"{factor_results['group_return_spread']:.2%}"],
            ['平均换手率', f"{factor_results['turnover_mean']:.2%}"]
        ]
        
        # 创建表格
        table = ax.table(cellText=stats_data[1:], colLabels=stats_data[0],
                        cellLoc='center', loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # 设置表格样式
        for i in range(len(stats_data)):
            if i == 0:
                for j in range(2):
                    table[(i, j)].set_facecolor('#40466e')
                    table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                for j in range(2):
                    table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('因子统计指标', fontsize=14, pad=20)
    
    def _plot_rolling_ic_stats(self, ic_series: pd.Series, ax):
        """绘制IC滚动统计"""
        if len(ic_series) < 60:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', fontsize=14)
            ax.set_title('IC滚动统计', fontsize=14)
            return
            
        window = 60  # 60天滚动窗口
        
        rolling_mean = ic_series.rolling(window).mean()
        rolling_std = ic_series.rolling(window).std()
        rolling_ir = rolling_mean / (rolling_std + 1e-10)
        
        ax2 = ax.twinx()
        
        # 绘制滚动IC均值和标准差
        rolling_mean.plot(ax=ax, color=self.color_palette[0], 
                         linewidth=2, label=f'{window}日滚动IC均值')
        rolling_std.plot(ax=ax, color=self.color_palette[1], 
                        linewidth=2, label=f'{window}日滚动IC标准差')
        
        # 绘制滚动IR
        rolling_ir.plot(ax=ax2, color=self.color_palette[2], 
                       linewidth=2, label=f'{window}日滚动IR')
        
        ax.set_title(f'IC滚动统计 (窗口={window}天)', fontsize=14)
        ax.set_xlabel('日期')
        ax.set_ylabel('IC均值/标准差')
        ax2.set_ylabel('IR')
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def plot_evolution_progress(self, logbook, save_path: Optional[str] = None):
        """
        绘制进化过程图
        
        Args:
            logbook: 进化日志
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        gen = logbook.select("gen")
        fit_max = logbook.select("max")
        fit_avg = logbook.select("avg")
        fit_min = logbook.select("min")
        fit_std = logbook.select("std")
        
        # 适应度进化曲线
        ax1.plot(gen, fit_max, 'b-', label='最大适应度', linewidth=2)
        ax1.plot(gen, fit_avg, 'g-', label='平均适应度', linewidth=2)
        ax1.plot(gen, fit_min, 'r-', label='最小适应度', linewidth=2)
        ax1.fill_between(gen, np.array(fit_avg) - np.array(fit_std), 
                        np.array(fit_avg) + np.array(fit_std), 
                        alpha=0.2, color='green')
        
        ax1.set_title('适应度进化过程', fontsize=14)
        ax1.set_xlabel('代数')
        ax1.set_ylabel('适应度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 适应度提升率
        improvement_rate = np.diff(fit_max) / (np.array(fit_max[:-1]) + 1e-10)
        ax2.plot(gen[1:], improvement_rate, 'o-', 
                color=self.color_palette[3], markersize=4)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax2.set_title('适应度提升率', fontsize=14)
        ax2.set_xlabel('代数')
        ax2.set_ylabel('提升率')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_factor_comparison(self, factor_results_list: List[Dict], 
                             save_path: Optional[str] = None):
        """
        比较多个因子的表现
        
        Args:
            factor_results_list: 因子评估结果列表
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # 1. IC均值比较
        ic_means = [r['ic_mean'] for r in factor_results_list]
        factor_names = [r['factor_name'] for r in factor_results_list]
        
        bars = axes[0].bar(range(len(factor_names)), ic_means, 
                          color=self.color_palette[:len(factor_names)], alpha=0.7)
        axes[0].set_xticks(range(len(factor_names)))
        axes[0].set_xticklabels(factor_names, rotation=45, ha='right')
        axes[0].set_title('IC均值比较', fontsize=14)
        axes[0].set_ylabel('IC均值')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, value in zip(bars, ic_means):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{value:.4f}', ha='center', va='bottom')
        
        # 2. IR比较
        irs = [r['ic_ir'] for r in factor_results_list]
        bars = axes[1].bar(range(len(factor_names)), irs, 
                          color=self.color_palette[:len(factor_names)], alpha=0.7)
        axes[1].set_xticks(range(len(factor_names)))
        axes[1].set_xticklabels(factor_names, rotation=45, ha='right')
        axes[1].set_title('IR比较', fontsize=14)
        axes[1].set_ylabel('IR')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 3. 多空收益比较
        returns = [r['group_return_spread'] for r in factor_results_list]
        bars = axes[2].bar(range(len(factor_names)), returns, 
                          color=self.color_palette[:len(factor_names)], alpha=0.7)
        axes[2].set_xticks(range(len(factor_names)))
        axes[2].set_xticklabels(factor_names, rotation=45, ha='right')
        axes[2].set_title('多空收益比较', fontsize=14)
        axes[2].set_ylabel('日均收益率')
        axes[2].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # 4. 综合评分雷达图
        categories = ['IC均值', 'IR', '多空收益', 'IC正率', '低换手']
        
        # 归一化数据
        def normalize_metric(values):
            min_val, max_val = min(values), max(values)
            if max_val - min_val < 1e-10:
                return [0.5] * len(values)
            return [(v - min_val) / (max_val - min_val) for v in values]
        
        # 准备雷达图数据
        ic_means_norm = normalize_metric(ic_means)
        irs_norm = normalize_metric(irs)
        returns_norm = normalize_metric(returns)
        ic_pos_rates = [r['ic_positive_ratio'] for r in factor_results_list]
        ic_pos_norm = normalize_metric(ic_pos_rates)
        turnovers = [1 - r['turnover_mean'] for r in factor_results_list]
        turnover_norm = normalize_metric(turnovers)
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(2, 2, 4, projection='polar')
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        for i, (name, result) in enumerate(zip(factor_names, factor_results_list)):
            values = [ic_means_norm[i], irs_norm[i], returns_norm[i], 
                     ic_pos_norm[i], turnover_norm[i]]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=name, color=self.color_palette[i % len(self.color_palette)])
            ax.fill(angles, values, alpha=0.15, 
                   color=self.color_palette[i % len(self.color_palette)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('因子综合表现雷达图', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()