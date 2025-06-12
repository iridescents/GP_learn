"""数据处理器"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging


class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """
        加载数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            股票数据字典
        """
        self.logger.info(f"正在加载数据: {data_path}")
        
        try:
            # 读取parquet文件
            df = pd.read_parquet(data_path)
            
            # 按股票代码分组
            stock_data = {}
            for stock_code in df['stock_code'].unique():
                stock_df = df[df['stock_code'] == stock_code].copy()
                stock_df.set_index('date', inplace=True)
                
                # 计算额外特征
                stock_df = self._calculate_features(stock_df)
                
                # 删除缺失值
                stock_df.dropna(inplace=True)
                
                if len(stock_df) > 250:  # 至少一年数据
                    stock_data[stock_code] = stock_df
            
            self.logger.info(f"成功加载 {len(stock_data)} 只股票数据")
            return stock_data
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return {}
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算特征
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            添加特征后的DataFrame
        """
        # 计算收益率
        df['returns'] = df['close'].pct_change()
        
        # 计算波动率
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # 确保必要的列存在
        required_columns = [
            'open', 'high', 'low', 'close', 'volume', 
            'turnover_rate', 'returns', 'volatility'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'turnover_rate':
                    # 如果没有换手率，用成交量估算
                    df[col] = df['volume'] / df['volume'].rolling(20).mean()
                else:
                    df[col] = 0
        
        return df
    
    def prepare_training_data(self, stock_data: Dict[str, pd.DataFrame], 
                            train_end_date: str,
                            future_days: int = 1) -> Tuple[Dict, Dict]:
        """
        准备训练数据
        
        Args:
            stock_data: 股票数据字典
            train_end_date: 训练结束日期
            future_days: 预测未来天数
            
        Returns:
            训练数据和标签数据
        """
        train_data = {}
        returns_data = {}
        
        train_end = pd.to_datetime(train_end_date)
        
        for stock_code, df in stock_data.items():
            # 分割训练集
            train_df = df[df.index <= train_end].copy()
            
            if len(train_df) > 100:
                # 计算未来收益作为标签
                future_returns = train_df['returns'].shift(-future_days)
                
                train_data[stock_code] = train_df
                returns_data[stock_code] = future_returns
        
        self.logger.info(f"准备了 {len(train_data)} 只股票的训练数据")
        return train_data, returns_data
    
    def generate_simulation_data(self, n_stocks: int = 100, 
                               start_date: str = '2020-01-01',
                               end_date: str = '2024-01-31') -> Dict[str, pd.DataFrame]:
        """
        生成模拟数据
        
        Args:
            n_stocks: 股票数量
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            模拟股票数据字典
        """
        self.logger.info("生成模拟数据...")
        
        # 生成日期范围
        dates = pd.date_range(start_date, end_date, freq='B')  # 工作日
        
        stock_data = {}
        
        np.random.seed(42)  # 保证可重复性
        
        for i in range(n_stocks):
            stock_code = f"SIM{i:04d}"
            
            # 生成价格数据
            initial_price = np.random.uniform(10, 100)
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = initial_price * np.exp(np.cumsum(returns))
            
            # 生成OHLC数据
            df = pd.DataFrame(index=dates)
            df['close'] = prices
            df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.005, len(dates)))
            df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
            df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
            
            # 生成成交量数据
            df['volume'] = np.random.lognormal(15, 1, len(dates))
            df['amount'] = df['volume'] * df['close']
            df['turnover_rate'] = np.random.beta(2, 20, len(dates))
            
            # 计算特征
            df = self._calculate_features(df)
            
            # 删除前60行（预热期）
            df = df.iloc[60:].copy()
            df.dropna(inplace=True)
            
            df.index.name = 'date'
            stock_data[stock_code] = df
        
        self.logger.info(f"生成了 {len(stock_data)} 只模拟股票数据")
        return stock_data