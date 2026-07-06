"""
Layer7 - 技术层
=====================

功能: 技术指标综合分析
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class TechnicalLayer:
    """技术层 - 技术指标综合分析"""
    
    def extract_features(self, symbol: str, df: pd.DataFrame, ctx: Dict, as_of_date: Optional[str] = None) -> Dict:
        """提取技术层特征

        Args:
            as_of_date: 截止日期（防未来函数）
        """
        features = {}
        
        # 时序对齐
        if as_of_date and not df.empty:
            df = df[df.index <= pd.Timestamp(as_of_date)]
        
        if df.empty or len(df) < 60:
            return features
        
        close = df['close']
        high = df.get('high', close)
        low = df.get('low', close)
        
        # 均线系统
        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else ma50
        
        cur = close.iloc[-1]
        
        features['tech_ma_bullish'] = cur > ma20 > ma50
        
        ma_score = 0.0
        if cur > ma200:
            ma_score += 0.4
        if cur > ma50:
            ma_score += 0.3
        if ma20 > ma50:
            ma_score += 0.3
        features['tech_ma_score'] = ma_score
        
        # RSI
        if len(df) >= 14:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 0.001)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            features['tech_rsi'] = float(rsi)
        else:
            features['tech_rsi'] = 50.0
        
        rsi = features['tech_rsi']
        if 40 <= rsi <= 60:
            features['tech_rsi_score'] = 1.0
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            features['tech_rsi_score'] = 0.7
        else:
            features['tech_rsi_score'] = 0.4
        
        # MACD (修复: 使用pandas Series)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd = float(macd_line.iloc[-1])
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        signal = float(signal_line.iloc[-1])
        hist = macd - signal
        
        features['tech_macd'] = macd
        features['tech_macd_hist'] = hist
        features['tech_macd_bullish'] = bool(hist > 0 and macd > 0)
        
        # 布林带
        bb_mid = close.rolling(20).mean().iloc[-1]
        bb_std = close.rolling(20).std().iloc[-1]
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        
        if bb_upper > bb_lower:
            features['tech_bb_position'] = float((cur - bb_lower) / (bb_upper - bb_lower))
        else:
            features['tech_bb_position'] = 0.5
        features['tech_bb_width'] = float((bb_upper - bb_lower) / bb_mid) if bb_mid > 0 else 0.2
        
        # 形态评分
        pattern_score = 0.0
        if features['tech_ma_bullish']:
            pattern_score += 0.3
        if features['tech_macd_bullish']:
            pattern_score += 0.2
        if 50 <= rsi <= 70:
            pattern_score += 0.2
        
        recent_low = low.iloc[-20:].min()
        if cur > recent_low * 1.05:
            pattern_score += 0.15
        
        if cur > bb_upper:
            pattern_score += 0.15
        
        features['tech_pattern_score'] = min(1.0, pattern_score)
        
        return features