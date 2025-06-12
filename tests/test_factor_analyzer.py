"""因子分析器测试"""

import unittest
import numpy as np
import pandas as pd
from src.core import FactorAnalyzer


class TestFactorAnalyzer(unittest.TestCase):
    """测试因子分析器"""
    
    def setUp(self):
        """设置测试环境"""
        self.analyzer = FactorAnalyzer()
        
        # 创建测试数据
        dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
        stocks = [f'STOCK_{i:04d}' for i in range(100)]
        
        # 因子值和收益率数据
        self.factor_values = pd.DataFrame(
            np.random.randn(len(dates), len(stocks)),
            index=dates,
            columns=stocks
        )
        
        self.returns = pd.DataFrame(
            np.random.normal(0, 0.02, (len(dates), len(stocks))),
            index=dates,
            columns=stocks
        )
    
    def test_calculate_ic_series(self):
        """测试IC序列计算"""
        ic_series = self.analyzer.calculate_ic_series(
            self.factor_values, self.returns
        )
        
        self.assertIsInstance(ic_series, pd.Series)
        self.assertTrue(len(ic_series) > 0)
        self.assertTrue(all(-1 <= ic <= 1 for ic in ic_series.dropna()))
    
    def test_calculate_factor_returns(self):
        """测试分组收益计算"""
        group_returns = self.analyzer.calculate_factor_returns(
            self.factor_values, self.returns, n_groups=5
        )
        
        self.assertIsInstance(group_returns, pd.DataFrame)
        if len(group_returns) > 0:
            self.assertEqual(len(group_returns.columns), 5)
    
    def test_calculate_turnover(self):
        """测试换手率计算"""
        turnover = self.analyzer.calculate_turnover(
            self.factor_values, n_top=20
        )
        
        self.assertIsInstance(turnover, pd.Series)
        if len(turnover) > 0:
            self.assertTrue(all(0 <= t <= 1 for t in turnover))


if __name__ == '__main__':
    unittest.main()