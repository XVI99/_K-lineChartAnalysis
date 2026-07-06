# -*- coding: utf-8 -*-
"""
AStockQuant.llm — LLM 驱动的智能分析模块

子模块:
- news_sentiment: 新闻情绪分析（增强 layer5）
- report_analysis: 研报/板块解读（增强 layer3）
- factor_explainer: 因子解释（研究报告）
- strategy_reporter: 策略报告生成（回测输出）
"""

from AStockQuant.llm.news_sentiment import NewsSentimentAnalyzer
from AStockQuant.llm.report_analysis import ReportAnalyzer
from AStockQuant.llm.factor_explainer import FactorExplainer
from AStockQuant.llm.strategy_reporter import StrategyReporter

__all__ = [
    "NewsSentimentAnalyzer",
    "ReportAnalyzer",
    "FactorExplainer",
    "StrategyReporter",
]
