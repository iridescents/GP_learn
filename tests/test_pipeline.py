"""流程管理测试"""

import unittest
import os
import tempfile
import shutil
from src.pipeline import FactorMiningPipeline


class TestFactorMiningPipeline(unittest.TestCase):
    """测试因子挖掘流程"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试配置
        self.test_config = {
            'data': {
                'input_path': 'test_data.parquet',
                'output_path': self.temp_dir,
                'train_end_date': '2022-12-31',
                'test_start_date': '2023-01-01'
            },
            'genetic_programming': {
                'population_size': 10,
                'generations': 2,
                'tournament_size': 3,
                'crossover_prob': 0.8,
                'mutation_prob': 0.1,
                'max_tree_depth': 3,
                'hall_of_fame_size': 3
            },
            'evaluation': {
                'ic_weight': 0.4,
                'ir_weight': 0.3,
                'turnover_penalty_weight': 0.2,
                'min_stocks_for_evaluation': 10,
                'evaluation_window': 20,
                'n_groups': 5,
                'top_n_for_turnover': 20
            },
            'visualization': {
                'figure_size': [10, 8],
                'dpi': 100,
                'style': 'seaborn',
                'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c']
            },
            'output': {
                'figures_dir': os.path.join(self.temp_dir, 'figures/'),
                'reports_dir': os.path.join(self.temp_dir, 'reports/'),
                'models_dir': os.path.join(self.temp_dir, 'models/'),
                'log_file': os.path.join(self.temp_dir, 'test.log')
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        }
        
        # 保存测试配置
        import yaml
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """测试流程初始化"""
        pipeline = FactorMiningPipeline(self.config_path)
        
        self.assertIsNotNone(pipeline.logger)
        self.assertIsNotNone(pipeline.data_processor)
        self.assertIsNotNone(pipeline.factor_analyzer)
        self.assertIsNotNone(pipeline.visualizer)
    
    def test_load_simulation_data(self):
        """测试加载模拟数据"""
        pipeline = FactorMiningPipeline(self.config_path)
        stock_data = pipeline.load_data()
        
        self.assertIsInstance(stock_data, dict)
        self.assertTrue(len(stock_data) > 0)


if __name__ == '__main__':
    unittest.main()