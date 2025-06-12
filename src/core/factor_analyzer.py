"""因子分析器"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class FactorAnalyzer:
    """因子分析器"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_ic_series(self, factor_values: pd.DataFrame, 
                           returns: pd.DataFrame, 
                           method: str = 'spearman') -> pd.Series:
        """
        计算IC序列
        
        Args:
            factor_values: 因子值DataFrame
            returns: 收益率DataFrame
            method: 相关系数计算方法
            
        Returns:
            IC序列
        """
        ic_series = []
        dates = factor_values.index
        
        for date in dates:
            if date in returns.index:
                ic = factor_values.loc[date].corr(
                    returns.loc[date], method=method)
                ic_series.append({'date': date, 'ic': ic})
        
        ic_df = pd.DataFrame(ic_series)
        if len(ic_df) > 0:
            return ic_df.set_index('date')['ic']
        else:
            return pd.Series(dtype=float)
    
    def calculate_factor_returns(self, factor_values: pd.DataFrame, 
                               returns: pd.DataFrame, 
                               n_groups: int = 5) -> pd.DataFrame:
        """
        计算分组收益
        
        Args:
            factor_values: 因子值DataFrame
            returns: 收益率DataFrame
            n_groups: 分组数量
            
        Returns:
            分组收益DataFrame
        """
        group_returns = []
        
        for date in factor_values.index:
            if date in returns.index:
                # 获取当日数据
                daily_factors = factor_values.loc[date].dropna()
                daily_returns = returns.loc[date]
                
                if len(daily_factors) >= n_groups:
                    # 分组
                    try:
                        groups = pd.qcut(daily_factors, n_groups, 
                                       labels=False, duplicates='drop')
                        
                        # 计算每组收益
                        for i in range(n_groups):
                            group_stocks = groups[groups == i].index
                            if len(group_stocks) > 0:
                                group_return = daily_returns[group_stocks].mean()
                                group_returns.append({
                                    'date': date,
                                    'group': i + 1,
                                    'return': group_return
                                })
                    except Exception as e:
                        continue
        
        if len(group_returns) > 0:
            df = pd.DataFrame(group_returns)
            return df.pivot(index='date', columns='group', values='return')
        else:
            return pd.DataFrame()
    
    def calculate_turnover(self, factor_values: pd.DataFrame, 
                          n_top: int = 50) -> pd.Series:
        """
        计算换手率
        
        Args:
            factor_values: 因子值DataFrame
            n_top: 前n只股票
            
        Returns:
            换手率序列
        """
        turnover_rates = []
        
        dates = factor_values.index
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            # 获取前后两期的top股票
            prev_factors = factor_values.loc[prev_date].dropna()
            curr_factors = factor_values.loc[curr_date].dropna()
            
            if len(prev_factors) >= n_top and len(curr_factors) >= n_top:
                prev_top = prev_factors.nlargest(n_top).index
                curr_top = curr_factors.nlargest(n_top).index
                
                # 计算换手率
                turnover = 1 - len(set(prev_top) & set(curr_top)) / n_top
                turnover_rates.append({'date': curr_date, 'turnover': turnover})
        
        if len(turnover_rates) > 0:
            return pd.DataFrame(turnover_rates).set_index('date')['turnover']
        else:
            return pd.Series(dtype=float)
    
    def evaluate_factor(self, factor_name: str, 
                       factor_values: pd.DataFrame, 
                       returns: pd.DataFrame,
                       n_groups: int = 5,
                       n_top: int = 50) -> Dict:
        """
        综合评估因子
        
        Args:
            factor_name: 因子名称
            factor_values: 因子值DataFrame
            returns: 收益率DataFrame
            n_groups: 分组数量
            n_top: 换手率计算的股票数量
            
        Returns:
            评估结果字典
        """
        # IC分析
        ic_series = self.calculate_ic_series(factor_values, returns)
        
        # 分组收益
        group_returns = self.calculate_factor_returns(
            factor_values, returns, n_groups)
        
        # 换手率
        turnover = self.calculate_turnover(factor_values, n_top)
        
        # 统计指标
        results = {
            'factor_name': factor_name,
            'ic_mean': ic_series.mean() if len(ic_series) > 0 else 0,
            'ic_std': ic_series.std() if len(ic_series) > 0 else 0,
            'ic_ir': (ic_series.mean() / (ic_series.std() + 1e-10) 
                     if len(ic_series) > 0 else 0),
            'ic_positive_ratio': ((ic_series > 0).mean() 
                                 if len(ic_series) > 0 else 0),
            'group_return_spread': 0,
            'turnover_mean': turnover.mean() if len(turnover) > 0 else 0,
            'ic_series': ic_series,
            'group_returns': group_returns,
            'turnover_series': turnover
        }
        
        # 计算多空收益
        if len(group_returns) > 0:
            if 1 in group_returns.columns and n_groups in group_returns.columns:
                results['group_return_spread'] = (
                    (group_returns[n_groups] - group_returns[1]).mean()
                )
        
        return results