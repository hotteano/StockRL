"""
Equity-MARL (E-MARL): A Financial Dynamics Framework for Multi-Agent Reinforcement Learning

This framework combines:
- Shapley Value for credit assignment
- Option pricing (Black-Scholes) for dynamic valuation
- Markowitz portfolio theory for weight optimization
- Bubble detection and parameter restructuring

Author: StockRL Project
Version: 2.0
"""

from .option_pricing import (
    BlackScholesLayer,
    BinomialTreePricer,
    AsianOptionPricer,
)
from .shapley import ShapleyCalculator, MonteCarloShapley
from .valuation import ValuationEngine, StockPriceTracker
from .meta_investor import MetaInvestor, MarkowitzOptimizer
from .bubble_detector import BubbleDetector, ParameterRestructurer
from .emarl_framework import EquityMARL, EasyEquityMARL

__version__ = "2.0.0"
__all__ = [
    # Main Framework
    "EquityMARL",
    "EasyEquityMARL",
    # Option Pricing
    "BlackScholesLayer",
    "BinomialTreePricer", 
    "AsianOptionPricer",
    # Shapley Value
    "ShapleyCalculator",
    "MonteCarloShapley",
    # Valuation
    "ValuationEngine",
    "StockPriceTracker",
    # Meta-Investor
    "MetaInvestor",
    "MarkowitzOptimizer",
    # Bubble Detection
    "BubbleDetector",
    "ParameterRestructurer",
]
