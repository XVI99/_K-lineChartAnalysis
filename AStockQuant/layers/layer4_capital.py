"""
Layer4 - 资金层
=====================

功能: 龙虎榜、机构资金流向分析

分析:
- 资金净流入/流出
- 是否上龙虎榜
- 机构持仓动向
- 大单买卖
"""

import pandas as pd
import numpy as np
from typing import Dict

class CapitalLayer:
    """
    资金层 - 资金流向分析
    追踪大资金动向是盈利的关键
    """
    
    def extract_features(self, symbol: str, df: pd.DataFrame, ctx: Dict) -> Dict:
        """提取资金层特征"""
        features = {}
        
        if df.empty or len(df) < 20:
            return features
        
        # 计算资金流向 (简化版: 用成交量变化近似)
        if 'volume' in df.columns and 'close' in df.columns:
            close = df['close']
            volume = df['volume']
            
            # 成交额 = 价格 * 成交量
            amount = close * volume
            
            # 5日平均成交额
            avg_amount_5 = amount.rolling(5).mean().iloc[-1]
            avg_amount_20 = amount.rolling(20).mean().iloc[-1]
            
            # 资金活跃度 (近期/中期)
            if avg_amount_20 > 0:
                features['capital_activity'] = avg_amount_5 / avg_amount_20
            else:
                features['capital_activity'] = 1.0
            
            # 成交量变化率
            vol_change = volume.pct_change(5).iloc[-1]
            features['volume_change'] = vol_change * 100
            
            # 资金净流入/流出评分 (0-1)
            if avg_amount_5 > avg_amount_20:
                features['capital_score'] = min(1.0, 0.5 + (avg_amount_5 / avg_amount_20 - 1) * 2)
            else:
                features['capital_score'] = max(0.0, 0.5 - (1 - avg_amount_5 / avg_amount_20) * 2)
            
            # 是否放量 (>1.5倍平均)
            features['capital_is_surge'] = avg_amount_5 / avg_amount_20 > 1.5 if avg_amount_20 > 0 else False
            
            # 龙虎榜标记 (简化版: 大幅上涨+放量视为有资金关注)
            daily_return = close.pct_change().iloc[-1]
            if daily_return > 0.05 and vol_change > 0.5:
                features['capital_lhb_on_board'] = True
            else:
                features['capital_lhb_on_board'] = False
        else:
            features['capital_score'] = 0.5
            features['capital_lhb_on_board'] = False
        
        return features