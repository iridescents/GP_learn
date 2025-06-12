"""遗传规划因子挖掘器"""

import numpy as np
import pandas as pd
from deap import base, creator, tools, gp, algorithms
import operator
import random
from typing import List, Dict, Tuple, Callable, Optional
import warnings
warnings.filterwarnings('ignore')


# 定义遗传规划的原语集合
def protected_div(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """安全除法，避免除零错误"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(right) > 1e-10, left / right, 1.0)


def ts_rank(x: np.ndarray, window: int = 10) -> np.ndarray:
    """时间序列排名"""
    return pd.Series(x).rolling(window).apply(
        lambda d: pd.Series(d).rank().iloc[-1] / len(d)
    ).values


def ts_max(x: np.ndarray, window: int = 10) -> np.ndarray:
    """时间序列最大值"""
    return pd.Series(x).rolling(window).max().values


def ts_min(x: np.ndarray, window: int = 10) -> np.ndarray:
    """时间序列最小值"""
    return pd.Series(x).rolling(window).min().values


def ts_mean(x: np.ndarray, window: int = 10) -> np.ndarray:
    """时间序列均值"""
    return pd.Series(x).rolling(window).mean().values


def ts_std(x: np.ndarray, window: int = 10) -> np.ndarray:
    """时间序列标准差"""
    return pd.Series(x).rolling(window).std().values


def ts_delay(x: np.ndarray, days: int = 1) -> np.ndarray:
    """时间序列延迟"""
    return pd.Series(x).shift(days).values


def sign(x: np.ndarray) -> np.ndarray:
    """符号函数"""
    return np.sign(x)


class GeneticProgrammingFactorMiner:
    """遗传规划因子挖掘器"""
    
    def __init__(self, 
                 population_size: int = 100,
                 generations: int = 50,
                 tournament_size: int = 7,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.1,
                 max_tree_depth: int = 4,
                 hall_of_fame_size: int = 10,
                 ic_weight: float = 0.4,
                 ir_weight: float = 0.3,
                 turnover_penalty_weight: float = 0.2):
        """
        初始化遗传规划因子挖掘器
        
        Args:
            population_size: 种群规模
            generations: 进化代数
            tournament_size: 锦标赛选择大小
            crossover_prob: 交叉概率
            mutation_prob: 变异概率
            max_tree_depth: 最大树深度
            hall_of_fame_size: 名人堂大小
            ic_weight: IC权重
            ir_weight: IR权重
            turnover_penalty_weight: 换手率惩罚权重
        """
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_tree_depth = max_tree_depth
        self.hall_of_fame_size = hall_of_fame_size
        self.ic_weight = ic_weight
        self.ir_weight = ir_weight
        self.turnover_penalty_weight = turnover_penalty_weight
        
        # 初始化遗传规划框架
        self._setup_gp_framework()
        
    def _setup_gp_framework(self):
        """设置遗传规划框架"""
        # 创建原语集
        self.pset = gp.PrimitiveSet("MAIN", 8)  # 8个输入特征
        
        # 基础运算符
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(protected_div, 2)
        self.pset.addPrimitive(np.abs, 1)
        self.pset.addPrimitive(sign, 1)
        self.pset.addPrimitive(np.log1p, 1)  # log(1+x)避免log(0)
        
        # 时间序列函数
        self.pset.addPrimitive(ts_rank, 1)
        self.pset.addPrimitive(ts_max, 1)
        self.pset.addPrimitive(ts_min, 1)
        self.pset.addPrimitive(ts_mean, 1)
        self.pset.addPrimitive(ts_std, 1)
        self.pset.addPrimitive(lambda x: ts_delay(x, 1), 1, name="ts_delay1")
        self.pset.addPrimitive(lambda x: ts_delay(x, 5), 1, name="ts_delay5")
        
        # 条件函数
        self.pset.addPrimitive(max, 2)
        self.pset.addPrimitive(min, 2)
        
        # 常数终端
        self.pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))
        
        # 重命名参数
        self.pset.renameArguments(
            ARG0='open', ARG1='high', ARG2='low', ARG3='close',
            ARG4='volume', ARG5='turnover', ARG6='returns', ARG7='volatility'
        )
        
        # 创建适应度和个体
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        # 工具箱
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, 
                             min_=2, max_=self.max_tree_depth)
        self.toolbox.register("individual", tools.initIterate, 
                             creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, 
                             list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
        # 遗传操作
        self.toolbox.register("select", tools.selTournament, 
                             tournsize=self.tournament_size)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, 
                             expr=self.toolbox.expr, pset=self.pset)
        
        # 装饰器限制树的深度
        self.toolbox.decorate("mate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=self.max_tree_depth))
        self.toolbox.decorate("mutate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=self.max_tree_depth))
    
    def evaluate_factor(self, individual, data: pd.DataFrame, 
                       returns: pd.Series) -> Tuple[float,]:
        """
        评估因子的适应度
        
        Args:
            individual: GP个体
            data: 输入数据
            returns: 未来收益率
            
        Returns:
            适应度分数
        """
        # 编译表达式
        func = self.toolbox.compile(expr=individual)
        
        try:
            # 计算因子值
            factor_values = func(
                data['open'].values, data['high'].values, 
                data['low'].values, data['close'].values,
                data['volume'].values, data['turnover_rate'].values, 
                data['returns'].values, data['volatility'].values
            )
            
            # 处理无效值
            factor_values = pd.Series(factor_values, index=data.index)
            factor_values = factor_values.replace([np.inf, -np.inf], np.nan)
            
            # 计算IC (Information Coefficient)
            ic = factor_values.corr(returns, method='spearman')
            
            # 计算IR (Information Ratio)
            rolling_ic = []
            for i in range(20, len(factor_values), 5):
                window_ic = factor_values.iloc[i-20:i].corr(
                    returns.iloc[i-20:i], method='spearman')
                if not np.isnan(window_ic):
                    rolling_ic.append(window_ic)
            
            if len(rolling_ic) > 0:
                ir = np.mean(rolling_ic) / (np.std(rolling_ic) + 1e-10)
            else:
                ir = 0
            
            # 计算换手率惩罚
            factor_diff = factor_values.diff().abs()
            turnover_penalty = factor_diff.mean() / (factor_values.std() + 1e-10)
            
            # 综合适应度
            fitness = (self.ic_weight * ic + 
                      self.ir_weight * ir - 
                      self.turnover_penalty_weight * turnover_penalty)
            
            return (float(fitness) if not np.isnan(fitness) else -1.0,)
            
        except Exception as e:
            return (-1.0,)
    
    def run_evolution(self, train_data: Dict[str, pd.DataFrame], 
                     returns_data: Dict[str, pd.Series]) -> Tuple[List, object]:
        """
        运行遗传进化过程
        
        Args:
            train_data: 训练数据字典 {股票代码: 特征DataFrame}
            returns_data: 收益率数据字典 {股票代码: 收益率Series}
            
        Returns:
            最优个体列表和日志
        """
        # 注册评估函数
        def eval_wrapper(individual):
            fitness_scores = []
            stock_codes = list(train_data.keys())[:50]  # 使用部分股票加速
            
            for stock_code in stock_codes:
                if stock_code in returns_data:
                    score = self.evaluate_factor(
                        individual, train_data[stock_code], 
                        returns_data[stock_code]
                    )
                    fitness_scores.append(score[0])
            
            return (np.mean(fitness_scores) if fitness_scores else -1.0,)
        
        self.toolbox.register("evaluate", eval_wrapper)
        
        # 初始化种群
        population = self.toolbox.population(n=self.population_size)
        halloffame = tools.HallOfFame(self.hall_of_fame_size)
        
        # 统计
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # 运行进化
        population, logbook = algorithms.eaSimple(
            population, self.toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.generations,
            stats=stats,
            halloffame=halloffame,
            verbose=True
        )
        
        return halloffame, logbook
    
    def extract_factor_expression(self, individual) -> str:
        """提取因子表达式"""
        return str(individual)
    
    def calculate_factor_values(self, individual, 
                              data: pd.DataFrame) -> pd.Series:
        """计算因子值"""
        func = self.toolbox.compile(expr=individual)
        
        try:
            factor_values = func(
                data['open'].values, data['high'].values, 
                data['low'].values, data['close'].values,
                data['volume'].values, data['turnover_rate'].values, 
                data['returns'].values, data['volatility'].values
            )
            
            factor_values = pd.Series(factor_values, index=data.index)
            factor_values = factor_values.replace([np.inf, -np.inf], np.nan)
            
            return factor_values
            
        except Exception as e:
            return pd.Series(index=data.index)