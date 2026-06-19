"""
Layer6 - 量价层
=====================

功能: RPS排名、VCP形态分析

核心指标:
- RPS: 相对价格强度 (50日/120日)
- VCP: 波动收敛形态
- 量价配合分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List

class RPSCalculator:
    """RPS计算器"""
    
    @staticmethod
    def calculate_rps(price_series: pd.Series, period: int = 50) -> float:
        """
        计算RPS: 当前价格在历史区间的百分位
        
        例如: RPS=80 表示价格高于过去80%的时间
        """
        if len(price_series) < period:
            return 50.0
        
        current = price_series.iloc[-1]
        historical = price_series.iloc[-period:]
        
        rank = (historical < current).sum() / len(historical) * 100
        return rank
    
    @staticmethod
    def calculate_vcp(df: pd.DataFrame) -> float:
        """
        计算VCP质量分数
        
        VCP特征:
        - 价格波动逐渐收敛
        - 成交量递减
        - 最终向上突破
        """
        if df.empty or len(df) < 60:
            return 0.0
        
        close = df['close']
        
        # 简化VCP: 近期涨幅 + 波动率下降
        volatility_short = close.pct_change().iloc[-20:].std()
        volatility_long = close.pct_change().iloc[-60:].std()
        
        # 波动率下降表示收敛
        vol_ratio = volatility_short / volatility_long if volatility_long > 0 else 1.0
        
        # 近期涨幅
        recent_return = close.pct_change(20).iloc[-1]
        
        # VCP分数
        if vol_ratio < 0.8 and recent_return > 0:
            vcp_score = 0.8
        elif vol_ratio < 1.0 and recent_return > 0:
            vcp_score = 0.6
        else:
            vcp_score = 0.4
        
        return vcp_score

class PriceVolumeLayer:
    """
    量价层 - 价格和成交量分析
    RPS和VCP是选股的核心指标
    """
    
    def __init__(self):
        self.rps_calculator = RPSCalculator()
    
    def extract_features(self, symbol: str, df: pd.DataFrame, ctx: Dict) -> Dict:
        """提取量价层特征"""
        features = {}
        
        if df.empty or len(df) < 60:
            return features
        
        close = df['close']
        
        # RPS 50日
        rps_50 = self.rps_calculator.calculate_rps(close, 50)
        features['pv_rps_50'] = rps_50
        
        # RPS 120日
        rps_120 = self.rps_calculator.calculate_rps(close, 120)
        features['pv_rps_120'] = rps_120
        
        # RPS综合 (近期权重更高)
        features['pv_rps_combined'] = rps_50 * 0.6 + rps_120 * 0.4
        
        # VCP质量
        vcp_quality = self.rps_calculator.calculate_vcp(df)
        features['pv_vcp_quality'] = vcp_quality
        
        # 量价配合
        if 'volume' in df.columns:
            vol = df['volume']
            
            # 成交量趋势
            vol_ma5 = vol.rolling(5).mean().iloc[-1]
            vol_ma20 = vol.rolling(20).mean().iloc[-1]
            vol_trend = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1.0
            features['pv_volume_trend'] = vol_trend
            
            # 价格与成交量背离检测
            price_up = close.iloc[-1] > close.iloc[-5]
            vol_up = vol.iloc[-1] > vol.iloc[-5]
            features['pv_price_volume_align'] = 1.0 if price_up == vol_up else 0.5
            
            # 放量突破
            vol_surge = vol_trend > 1.5
            price_breakout = close.iloc[-1] > close.rolling(20).max().iloc[-2]
            features['pv_breakout_surge'] = vol_surge and price_breakout
        else:
            features['pv_volume_trend'] = 1.0
            features['pv_price_volume_align'] = 0.5
            features['pv_breakout_surge'] = False
        
        # 综合评分
        features['pv_score'] = (
            features['pv_rps_combined'] / 100 * 0.5 +
            features['pv_vcp_quality'] * 0.3 +
            features.get('pv_volume_trend', 1.0) / 2 * 0.2
        )
        
        return features