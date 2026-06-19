"""
Layer2 - 制度层
=====================

功能: ST/退市/停牌过滤，规则检查

过滤掉不适合交易的标的:
- ST股票
- 退市风险股
- 停牌股票
- 涨跌停限制
"""

import pandas as pd
from typing import Dict, Tuple

class RulesLayer:
    """
    制度层 - 交易规则检查和过滤
    通过这层的标的才能进入后续分析
    """
    
    def __init__(self, 
                 allowed_boards: tuple = ('etf_sh', 'etf_sz'),
                 exclude_risk_levels: tuple = ('st', 'delisting', 'suspended')):
        self.allowed_boards = allowed_boards
        self.exclude_risk_levels = exclude_risk_levels
        
        # ST关键词
        self.st_keywords = ['ST', '*ST', 'S*ST', 'SST']
        
    def extract_features(self, symbol: str, df: pd.DataFrame, ctx: Dict) -> Dict:
        """提取制度层特征"""
        features = {}
        
        if df.empty:
            features['rules_pass'] = False
            return features
        
        # 默认通过
        features['rules_pass'] = True
        
        # 检查涨跌停
        latest = df.iloc[-1]
        prev_close = df['close'].iloc[-2] if len(df) >= 2 else latest['close']
        
        if 'change' in latest:
            change_pct = latest['change'] / prev_close * 100
        elif 'pct_change' in df.columns:
            change_pct = latest['pct_change'] * 100
        else:
            change_pct = (latest['close'] / prev_close - 1) * 100
        
        features['daily_change_pct'] = change_pct
        
        # 检查涨停 (不能买涨停股)
        if change_pct >= 9.8:
            features['rules_pass'] = False
            features['rule_blocked'] = 'limit_up'
        
        # 检查跌停 (尽量避免)
        elif change_pct <= -9.8:
            features['rules_pass'] = False
            features['rule_blocked'] = 'limit_down'
        
        # 检查停牌 (成交量为0)
        if 'volume' in latest and latest['volume'] == 0:
            features['rules_pass'] = False
            features['rule_blocked'] = 'suspended'
        
        return features
    
    def should_exclude(self, symbol: str, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        检查是否应该排除该标的
        
        Returns:
            (should_exclude, reason)
        """
        if df.empty:
            return True, 'no_data'
        
        # 检查名称是否包含ST
        if 'name' in df.columns:
            name = df['name'].iloc[-1]
            for kw in self.st_keywords:
                if kw in str(name):
                    return True, 'st_stock'
        
        return False, ''