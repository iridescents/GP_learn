"""因子挖掘主流程"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Optional
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core import GeneticProgrammingFactorMiner, FactorAnalyzer, DataProcessor
from src.visualization import FactorVisualizer
from src.utils import setup_logger, load_config, save_json


class FactorMiningPipeline:
    """因子挖掘完整流程"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        初始化流程
        
        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.logger = setup_logger(
            'FactorMiningPipeline', 
            self.config['output']['log_file'],
            self.config['logging']['level']
        )
        self.results = {}
        
        # 初始化组件
        self.data_processor = DataProcessor()
        self.factor_analyzer = FactorAnalyzer()
        self.visualizer = FactorVisualizer(
            figsize=tuple(self.config['visualization']['figure_size']),
            dpi=self.config['visualization']['dpi']
        )
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """加载预处理后的数据"""
        data_path = self.config['data']['input_path']
        
        if os.path.exists(data_path):
            return self.data_processor.load_data(data_path)
        else:
            # 如果没有数据文件，生成模拟数据
            self.logger.warning("未找到数据文件，使用模拟数据")
            return self.data_processor.generate_simulation_data()
    
    def prepare_training_data(self, stock_data: Dict[str, pd.DataFrame]) -> tuple:
        """准备训练数据"""
        return self.data_processor.prepare_training_data(
            stock_data,
            self.config['data']['train_end_date']
        )
    
    def run_genetic_programming(self, train_data: Dict, 
                              returns_data: Dict) -> tuple:
        """运行遗传规划算法"""
        self.logger.info("开始运行遗传规划算法...")
        
        # 创建GP挖掘器
        gp_config = self.config['genetic_programming']
        eval_config = self.config['evaluation']
        
        gp_miner = GeneticProgrammingFactorMiner(
            population_size=gp_config['population_size'],
            generations=gp_config['generations'],
            tournament_size=gp_config['tournament_size'],
            crossover_prob=gp_config['crossover_prob'],
            mutation_prob=gp_config['mutation_prob'],
            max_tree_depth=gp_config['max_tree_depth'],
            hall_of_fame_size=gp_config['hall_of_fame_size'],
            ic_weight=eval_config['ic_weight'],
            ir_weight=eval_config['ir_weight'],
            turnover_penalty_weight=eval_config['turnover_penalty_weight']
        )
        
        # 运行进化
        halloffame, logbook = gp_miner.run_evolution(train_data, returns_data)
        
        self.logger.info(f"遗传规划完成，找到 {len(halloffame)} 个最优因子")
        
        # 保存最优因子
        self.save_best_factors(halloffame, gp_miner)
        
        return halloffame, logbook, gp_miner
    
    def save_best_factors(self, halloffame, gp_miner):
        """保存最优因子"""
        best_factors = []
        
        for i, individual in enumerate(halloffame):
            factor_info = {
                'rank': i + 1,
                'expression': gp_miner.extract_factor_expression(individual),
                'fitness': individual.fitness.values[0],
                'tree_size': len(individual),
                'tree_height': individual.height
            }
            best_factors.append(factor_info)
        
        # 保存为JSON
        output_path = os.path.join(
            self.config['output']['models_dir'], 
            'best_factors.json'
        )
        save_json(best_factors, output_path)
        
        self.logger.info(f"保存了 {len(best_factors)} 个最优因子")
    
    def evaluate_factors(self, halloffame, gp_miner, 
                        stock_data: Dict) -> List[Dict]:
        """评估最优因子"""
        self.logger.info("开始评估因子表现...")
        
        factor_results = []
        
        # 准备评估数据
        all_dates = sorted(set().union(*[df.index for df in stock_data.values()]))
        test_start = pd.to_datetime(self.config['data']['test_start_date'])
        test_dates = [d for d in all_dates if d >= test_start]
        
        # 评估前3个最优因子
        n_factors = min(3, len(halloffame))
        for i, individual in enumerate(halloffame[:n_factors]):
            self.logger.info(f"评估因子 {i+1}/{n_factors}")
            
            # 计算因子值
            factor_values_dict = {}
            for stock_code, df in stock_data.items():
                factor_values = gp_miner.calculate_factor_values(individual, df)
                factor_values_dict[stock_code] = factor_values
            
            # 转换为横截面数据
            factor_df = pd.DataFrame(factor_values_dict).T
            returns_df = pd.DataFrame({
                code: df['returns'].shift(-1) 
                for code, df in stock_data.items()
            }).T
            
            # 对齐数据
            common_dates = sorted(
                set(factor_df.columns) & set(returns_df.columns) & set(test_dates)
            )
            if len(common_dates) > 0:
                factor_df = factor_df[common_dates].T
                returns_df = returns_df[common_dates].T
                
                # 评估因子
                result = self.factor_analyzer.evaluate_factor(
                    f"GP_Factor_{i+1:03d}",
                    factor_df,
                    returns_df,
                    n_groups=self.config['evaluation']['n_groups'],
                    n_top=self.config['evaluation']['top_n_for_turnover']
                )
                
                factor_results.append(result)
        
        return factor_results
    
    def generate_report(self, factor_results: List[Dict], logbook):
        """生成分析报告"""
        self.logger.info("生成可视化报告...")
        
        # 为每个因子生成详细报告
        for i, result in enumerate(factor_results):
            self.logger.info(f"生成因子 {result['factor_name']} 的可视化报告")
            
            output_path = os.path.join(
                self.config['output']['figures_dir'],
                f"factor_performance_{result['factor_name']}.png"
            )
            self.visualizer.plot_factor_performance_dashboard(
                result, save_path=output_path
            )
        
        # 生成进化过程图
        evolution_path = os.path.join(
            self.config['output']['figures_dir'],
            'evolution_progress.png'
        )
        self.visualizer.plot_evolution_progress(logbook, save_path=evolution_path)
        
        # 生成因子对比图
        if len(factor_results) > 1:
            comparison_path = os.path.join(
                self.config['output']['figures_dir'],
                'factor_comparison.png'
            )
            self.visualizer.plot_factor_comparison(
                factor_results, save_path=comparison_path
            )
        
        self.logger.info("可视化报告生成完成")
    
    def save_results(self, factor_results: List[Dict], logbook):
        """保存分析结果"""
        # 保存因子评估结果
        results_summary = []
        for result in factor_results:
            summary = {
                'factor_name': result['factor_name'],
                'ic_mean': result['ic_mean'],
                'ic_std': result['ic_std'],
                'ic_ir': result['ic_ir'],
                'ic_positive_ratio': result['ic_positive_ratio'],
                'group_return_spread': result['group_return_spread'],
                'turnover_mean': result['turnover_mean']
            }
            results_summary.append(summary)
        
        # 保存CSV
        csv_path = os.path.join(
            self.config['output']['reports_dir'],
            'factor_evaluation_results.csv'
        )
        pd.DataFrame(results_summary).to_csv(csv_path, index=False)
        
        # 保存详细结果
        pkl_path = os.path.join(
            self.config['output']['models_dir'],
            'factor_results_detail.pkl'
        )
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'factor_results': factor_results, 
                'logbook': logbook,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        self.logger.info("结果保存完成")
    
    def run_pipeline(self) -> List[Dict]:
        """运行完整流程"""
        self.logger.info("=" * 50)
        self.logger.info("开始运行因子挖掘流程")
        self.logger.info("=" * 50)
        
        try:
            # 1. 加载数据
            stock_data = self.load_data()
            
            # 2. 准备训练数据
            train_data, returns_data = self.prepare_training_data(stock_data)
            
            # 3. 运行遗传规划
            halloffame, logbook, gp_miner = self.run_genetic_programming(
                train_data, returns_data
            )
            
            # 4. 评估因子
            factor_results = self.evaluate_factors(halloffame, gp_miner, stock_data)
            
            # 5. 生成报告
            self.generate_report(factor_results, logbook)
            
            # 6. 保存结果
            self.save_results(factor_results, logbook)
            
            self.logger.info("=" * 50)
            self.logger.info("因子挖掘流程完成！")
            self.logger.info("=" * 50)
            
            return factor_results
            
        except Exception as e:
            self.logger.error(f"流程执行失败: {e}", exc_info=True)
            raise