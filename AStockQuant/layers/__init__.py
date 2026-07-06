"""
AStockQuant Layers - 九层量化框架
================================

L1-L8 各层模块统一导出
"""

from .layer1_macro import MacroLayer
from .layer2_rules import RulesLayer
from .layer3_sector import SectorLayer
from .layer4_capital import CapitalLayer
from .layer5_sentiment import SentimentLayer
from .layer6_price_vol import PriceVolumeLayer, RPSCalculator
from .layer7_technical import TechnicalLayer
from .layer8_micro import BeliefLayer

__all__ = [
    'MacroLayer',           # L1 宏观层
    'RulesLayer',           # L2 制度层
    'SectorLayer',          # L3 板块层
    'CapitalLayer',         # L4 资金层
    'SentimentLayer',       # L5 情绪层
    'PriceVolumeLayer',     # L6 量价层
    'RPSCalculator',        # L6 RPS计算器
    'TechnicalLayer',       # L7 技术层
    'BeliefLayer',          # L8 贝叶斯层
]

"""
九层架构说明:
=============

L1 宏观层 (MacroLayer)
  - 大盘趋势判断 (BULL/BEAR/NEUTRAL)
  - 市场环境评分

L2 制度层 (RulesLayer)
  - ST/退市/停牌过滤
  - 涨跌停限制

L3 板块层 (SectorLayer)
  - 行业板块轮动
  - 热点板块判断
  - 龙头股识别

L4 资金层 (CapitalLayer)
  - 龙虎榜资金
  - 成交量/资金流向
  - 机构动向

L5 情绪层 (SentimentLayer)
  - 市场情绪评分
  - 涨停/连板分析
  - 追板热情

L6 量价层 (PriceVolumeLayer)
  - RPS排名 (50日/120日)
  - VCP形态
  - 量价配合

L7 技术层 (TechnicalLayer)
  - 均线系统
  - RSI/MACD/布林带
  - 技术形态评分

L8 贝叶斯层 (BeliefLayer)
  - 序贯贝叶斯更新
  - 信念后验概率
  - KL散度漂移检测
"""