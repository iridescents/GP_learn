"""量化因子挖掘系统"""

__version__ = "1.0.0"
__author__ = ""
__email__ = ""

from .core import GeneticProgrammingFactorMiner, FactorAnalyzer
from .visualization import FactorVisualizer
from .pipeline import FactorMiningPipeline

__all__ = [
    "GeneticProgrammingFactorMiner",
    "FactorAnalyzer",
    "FactorVisualizer",
    "FactorMiningPipeline",
]