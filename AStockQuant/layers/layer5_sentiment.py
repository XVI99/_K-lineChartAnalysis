"""
Layer5 - 情绪层
=====================

功能: 市场情绪分析，涨停板、连板判断

v2 改进:
- 用连续评分替代离散阈值
- 增加市场广度（涨跌家数比）作为情绪指标
- 增加波动率情绪（VIX-like）
- 增加换手率情绪
- 时序对齐：支持 as_of_date 防未来函数
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class SentimentLayer:
    """
    情绪层 - 市场情绪分析
    极端情绪往往是反向指标
    """

    def extract_features(
        self,
        symbol: str,
        df: pd.DataFrame,
        ctx: Dict,
        as_of_date: Optional[str] = None,
    ) -> Dict:
        """提取情绪层特征"""
        features = {}

        # 时序对齐
        if as_of_date and not df.empty:
            df = df[df.index <= pd.Timestamp(as_of_date)]

        if df.empty or len(df) < 20:
            return features

        close = df["close"]
        daily_returns = close.pct_change().dropna()

        # ==================== 1. 收益分布情绪（连续映射） ====================
        # 用 sigmoid 将近期相对收益映射到 [0, 1]
        recent_return = daily_returns.iloc[-20:].mean()
        historical_return = daily_returns.mean()

        if historical_return != 0:
            relative_return = recent_return / abs(historical_return)
        else:
            relative_return = 0.0

        # 连续情绪评分（sigmoid映射，clip防溢出）
        features["sentiment_score"] = float(
            1.0 / (1.0 + np.exp(-np.clip(relative_return, -50.0, 50.0)))
        )

        # ==================== 2. 涨停/跌停检测 ====================
        if len(df) >= 2:
            prev_close = close.iloc[-2]
            today_close = close.iloc[-1]
            daily_return = (today_close / prev_close - 1) * 100

            features["sent_is_limit_up"] = bool(daily_return >= 9.8)
            features["sent_is_limit_down"] = bool(daily_return <= -9.8)
            features["sent_daily_return"] = float(daily_return)
        else:
            features["sent_is_limit_up"] = False
            features["sent_is_limit_down"] = False
            features["sent_daily_return"] = 0.0

        # ==================== 3. 连板天数 ====================
        consecutive_days = 0
        for i in range(min(10, len(df) - 1)):
            idx = -(i + 1)
            if i == 0:
                prev = df["close"].iloc[-2]
            else:
                prev = df["close"].iloc[-(i + 2)]
            curr = df["close"].iloc[idx]
            if prev > 0 and (curr / prev - 1) >= 0.09:
                consecutive_days += 1
            else:
                break
        features["sent_consecutive_days"] = consecutive_days

        # ==================== 4. 追板热情（量比） ====================
        if "volume" in df.columns:
            vol_now = df["volume"].iloc[-1]
            vol_avg = df["volume"].rolling(20).mean().iloc[-1]
            vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0
            features["sent_chase_ratio"] = float(min(3.0, vol_ratio))
        else:
            features["sent_chase_ratio"] = 1.0

        # ==================== 5. v2新增: 波动率情绪（VIX-like） ====================
        # 高波动 = 恐慌，低波动 = 自满
        vol_20 = daily_returns.rolling(20).std().iloc[-1]
        vol_60 = daily_returns.rolling(60).std().iloc[-1] if len(daily_returns) >= 60 else vol_20
        vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0

        # 波动率情绪：波动率飙升=恐慌(低分)，波动率收缩=自满(高分)
        features["sent_volatility_ratio"] = float(vol_ratio)
        features["sent_volatility_score"] = float(
            1.0 / (1.0 + np.exp((vol_ratio - 1.0) * 5))  # vol>1 → 低分
        )

        # ==================== 6. v2新增: 换手率情绪 ====================
        # 用成交量变化近似换手率
        if "volume" in df.columns:
            vol = df["volume"]
            vol_change_5d = vol.pct_change(5).iloc[-1]
            # 放量=活跃(可能过度乐观)，缩量=冷清
            if not np.isnan(vol_change_5d):
                features["sent_turnover_change"] = float(vol_change_5d)
                features["sent_turnover_score"] = float(
                    1.0 / (1.0 + np.exp(-np.clip(vol_change_5d * 2, -50.0, 50.0)))
                )
            else:
                features["sent_turnover_change"] = 0.0
                features["sent_turnover_score"] = 0.5
        else:
            features["sent_turnover_change"] = 0.0
            features["sent_turnover_score"] = 0.5

        # ==================== 7. v2新增: RSI情绪 ====================
        # RSI极高=超买，极低=超卖
        if len(df) >= 14:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 0.001)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            features["sent_rsi"] = float(rsi)
            # v3改进: RSI改为正向动量逻辑 (ETF市场动量效应显著, 强者恒强)
            # 原反向逻辑(超买看空)ICIR=-0.25, 实证无效; 改为正向: RSI高=强势=高分
            if rsi > 70:
                features["sent_rsi_score"] = 0.8  # 强势, 看多
            elif rsi > 50:
                features["sent_rsi_score"] = 0.6
            elif rsi > 30:
                features["sent_rsi_score"] = 0.4
            else:
                features["sent_rsi_score"] = 0.2  # 弱势, 看空
            # 连续RSI动量评分 (rsi/100, 完全线性的正向因子)
            features["sent_rsi_momentum"] = float(rsi / 100.0)
        else:
            features["sent_rsi"] = 50.0
            features["sent_rsi_score"] = 0.5
            features["sent_rsi_momentum"] = 0.5

        # ==================== v3: 回测模式从历史数据库读取情绪 ====================
        if as_of_date:
            try:
                from core.history_data_loader import HistoryDataLoader
                loader = HistoryDataLoader.get_instance()
                if loader.has_history_data():
                    sent_data = loader.get_sentiment(symbol, as_of_date=as_of_date)
                    if sent_data["confidence"] > 0:
                        features["sent_llm_score"] = sent_data["sentiment_score"]
                        features["sent_llm_confidence"] = sent_data["confidence"]
                        base = features.get("sentiment_score", 0.5)
                        llm = sent_data["sentiment_score"]
                        w = min(0.4, sent_data["confidence"])
                        features["sentiment_score"] = float(base * (1 - w) + llm * w)
            except Exception:
                pass

        # ==================== 8. v2新增: 市场广度情绪 ====================
        # 如果ctx提供了全市场数据，计算涨跌家数比
        all_data = ctx.get("all_data", {})
        if all_data:
            up_count = 0
            down_count = 0
            for sym, sym_df in all_data.items():
                if as_of_date:
                    sym_df = sym_df[sym_df.index <= pd.Timestamp(as_of_date)]
                if len(sym_df) >= 2 and not sym_df.empty:
                    ret = (sym_df["close"].iloc[-1] / sym_df["close"].iloc[-2] - 1)
                    if ret > 0:
                        up_count += 1
                    elif ret < 0:
                        down_count += 1
            total = up_count + down_count
            if total > 0:
                breadth = up_count / total
                features["sent_market_breadth"] = float(breadth)
                # 市场广度情绪：涨多跌少=乐观
                features["sent_market_score"] = float(breadth)
            else:
                features["sent_market_breadth"] = 0.5
                features["sent_market_score"] = 0.5
        else:
            # 降级：用大盘数据
            market_prices_df = ctx.get("market_prices_df")
            if market_prices_df is not None and not market_prices_df.empty:
                if as_of_date:
                    market_prices_df = market_prices_df[
                        market_prices_df.index <= pd.Timestamp(as_of_date)
                    ]
                if len(market_prices_df) >= 20:
                    mkt_return = market_prices_df["close"].pct_change().iloc[-20:].mean()
                    features["sent_market_score"] = float(
                        min(1.0, max(0.0, 0.5 + mkt_return * 10))
                    )
                else:
                    features["sent_market_score"] = 0.5
            else:
                features["sent_market_score"] = 0.5
            features["sent_market_breadth"] = features["sent_market_score"]

        # ==================== 9. 综合情绪评分 ====================
        features["sent_combined_score"] = self._compute_sentiment_score(features)

        return features

    def _compute_sentiment_score(self, features: Dict) -> float:
        """计算综合情绪评分 (0-1)"""
        score = 0.0

        # 收益情绪 (权重25%)
        score += features.get("sentiment_score", 0.5) * 0.25

        # 市场广度 (权重25%)
        score += features.get("sent_market_score", 0.5) * 0.25

        # 波动率情绪 (权重20%)
        score += features.get("sent_volatility_score", 0.5) * 0.20

        # RSI情绪 (权重15%)
        score += features.get("sent_rsi_score", 0.5) * 0.15

        # 换手率情绪 (权重15%)
        score += features.get("sent_turnover_score", 0.5) * 0.15

        return float(max(0.0, min(1.0, score)))
