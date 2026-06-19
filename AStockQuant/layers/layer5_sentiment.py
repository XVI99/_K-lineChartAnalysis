"""
Layer5 - 情绪层
=====================

功能: 市场情绪分析，涨停板、连板判断

分析:
- 市场整体情绪
- 是否涨停
- 连板天数
- 追板热情
"""

import pandas as pd
import numpy as np
from typing import Dict

class SentimentLayer:
    """
    情绪层 - 市场情绪分析
    极端情绪往往是反向指标
    """
    
    def extract_features(self, symbol: str, df: pd.DataFrame, ctx: Dict) -> Dict:
        """提取情绪层特征"""
        features = {}
        
        if df.empty or len(df) < 20:
            return features
        
        close = df['close']
        
        # 计算日收益率
        daily_returns = close.pct_change().dropna()
        
        # 市场情绪分 (0-1)
        # 计算近期收益分布
        recent_return = daily_returns.iloc[-20:].mean()
        historical_return = daily_returns.mean()
        
        if historical_return != 0:
            relative_return = recent_return / abs(historical_return)
        else:
            relative_return = 1.0
        
        # 情绪评分
        if relative_return > 2:
            features['sentiment_score'] = 0.8  # 极度乐观
        elif relative_return > 1:
            features['sentiment_score'] = 0.6  # 乐观
        elif relative_return < -1:
            features['sentiment_score'] = 0.2  # 悲观
        else:
            features['sentiment_score'] = 0.5  # 中性
        
        # 是否涨停
        if len(df) >= 2:
            prev_close = close.iloc[-2]
            today_close = close.iloc[-1]
            daily_return = (today_close / prev_close - 1) * 100
            
            features['sent_is_limit_up'] = daily_return >= 9.8
            features['sent_is_limit_down'] = daily_return <= -9.8
            features['sent_daily_return'] = daily_return
        else:
            features['sent_is_limit_up'] = False
            features['sent_is_limit_down'] = False
        
        # 计算连板天数
        consecutive_days = 0
        for i in range(min(10, len(df)-1)):
            idx = -(i+1)
            if i == 0:
                prev = df['close'].iloc[-2]
            else:
                prev = df['close'].iloc[-(i+2)]
            
            curr = df['close'].iloc[idx]
            if prev > 0 and (curr / prev - 1) >= 0.09:  # 涨停
                consecutive_days += 1
            else:
                break
        
        features['sent_consecutive_days'] = consecutive_days
        
        # 追板热情 (成交量放大且上涨)
        if 'volume' in df.columns:
            vol_now = df['volume'].iloc[-1]
            vol_avg = df['volume'].rolling(20).mean().iloc[-1]
            vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0
            features['sent_chase_ratio'] = min(2.0, vol_ratio)
        else:
            features['sent_chase_ratio'] = 1.0
        
        # 市场情绪评分 (综合)
        market_prices_df = ctx.get('market_prices_df')
        if market_prices_df is not None and not market_prices_df.empty:
            mkt_return = market_prices_df['close'].pct_change().iloc[-20:].mean()
            features['sent_market_score'] = min(1.0, max(0.0, 0.5 + mkt_return * 10))
        else:
            features['sent_market_score'] = 0.5
        
        return features