"""
Layer1 - 宏观层
=====================

功能: 大盘趋势判断，环境调节

判断当前市场环境:
- BULL: 牛市，多头排列，应该持仓
- BEAR: 熊市，空头排列，应该空仓
- NEUTRAL: 中性，震荡市场
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

class MacroLayer:
    """
    宏观层 - 分析大盘环境，判断市场趋势
    用于决定是否应该持仓/空仓
    """
    
    def __init__(self):
        self.index_codes = ['000300', '000001']  # 沪深300, 上证指数
    
    def extract_features(self, symbol: str, df: pd.DataFrame, ctx: Dict, as_of_date: Optional[str] = None) -> Dict:
        """提取宏观特征

        Args:
            as_of_date: 截止日期（防未来函数，只用到此日期前的数据）
        """
        features = {}
        
        # 时序对齐
        if as_of_date and not df.empty:
            df = df[df.index <= pd.Timestamp(as_of_date)]
        
        if df.empty or len(df) < 60:
            return features
        
        # 计算指数趋势
        market_prices_df = ctx.get('market_prices_df')
        
        if market_prices_df is not None and not market_prices_df.empty:
            # 时序对齐：大盘数据也截取
            if as_of_date:
                market_prices_df = market_prices_df[market_prices_df.index <= pd.Timestamp(as_of_date)]
            if market_prices_df.empty or len(market_prices_df) < 60:
                return features
            
            close = market_prices_df['close']
            
            # 均线系统
            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1]
            ma120 = close.rolling(120).mean().iloc[-1]
            cur = close.iloc[-1]
            
            # 趋势判断
            if cur > ma20 > ma60:
                features['market_regime'] = 'BULL'
                features['macro_regime_score'] = 0.8
            elif cur < ma20 < ma60:
                features['market_regime'] = 'BEAR'
                features['macro_regime_score'] = 0.2
            else:
                features['market_regime'] = 'NEUTRAL'
                features['macro_regime_score'] = 0.5
            
            # 市场位置 (距离均线的百分比)
            features['market_position'] = (cur - ma60) / ma60 * 100 if ma60 != 0 else 0
            
            # 波动率
            features['market_volatility'] = close.pct_change().rolling(20).std().iloc[-1] * 100
            
            # 趋势强度
            features['trend_strength'] = (ma20 - ma60) / ma60 * 100 if ma60 != 0 else 0

            # ==================== v2新增: 个股相对大盘强度 (有截面区分度) ====================
            # 原有宏观因子只依赖大盘数据, 对所有ETF值相同, 无选股能力
            # 新增相对强度因子: ETF收益 - 大盘收益, 正值表示跑赢大盘
            etf_close = df['close']
            if len(etf_close) >= 20:
                etf_ret_20 = etf_close.pct_change(20).iloc[-1]
                mkt_ret_20 = close.pct_change(20).iloc[-1]
                if not np.isnan(etf_ret_20) and not np.isnan(mkt_ret_20):
                    # 相对收益 (百分点)
                    features['macro_relative_strength'] = float((etf_ret_20 - mkt_ret_20) * 100)

                    # 相对强度连续评分 (sigmoid映射到0-1, 0.5=与大盘持平)
                    rel_diff = etf_ret_20 - mkt_ret_20
                    features['macro_relative_score'] = float(
                        1.0 / (1.0 + np.exp(-np.clip(rel_diff * 10, -50.0, 50.0)))
                    )

            # ETF相对大盘均线的位置 (ETF价格vs大盘MA60的偏离度差异)
            if len(etf_close) >= 60 and ma60 != 0:
                etf_ma60 = etf_close.rolling(60).mean().iloc[-1]
                if not np.isnan(etf_ma60) and etf_ma60 > 0:
                    etf_pos = (etf_close.iloc[-1] - etf_ma60) / etf_ma60 * 100
                    mkt_pos = (cur - ma60) / ma60 * 100
                    features['macro_relative_position'] = float(etf_pos - mkt_pos)

        return features
    
    def get_market_regime(self, market_df: pd.DataFrame = None) -> str:
        """获取当前市场状态"""
        if market_df is None or market_df.empty:
            return 'NEUTRAL'
        
        close = market_df['close']
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        cur = close.iloc[-1]
        
        if cur > ma20 > ma60:
            return 'BULL'
        elif cur < ma20 < ma60:
            return 'BEAR'
        return 'NEUTRAL'
    
    def should_hold(self, ctx: Dict) -> bool:
        """判断是否应该持仓"""
        market_prices_df = ctx.get('market_prices_df')
        
        if market_prices_df is None or market_prices_df.empty:
            return True  # 无数据时默认持仓
        
        close = market_prices_df['close']
        ma20 = close.rolling(20).mean().iloc[-1]
        cur = close.iloc[-1]
        
        return cur > ma20  # 价格在均线上方则持仓