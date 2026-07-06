"""
Layer8 - 贝叶斯信念层
=====================

功能: 序贯贝叶斯更新，信念动态调整

v2 改进:
- 离散5档likelihood → 连续似然函数（多证据加权sigmoid映射）
- 多源证据融合：收益率+成交量+RPS+VCP+资金流
- 贝叶斯后验连续映射到 [0.01, 0.99]
- KL散度作为信念更新强度指标
- 时序对齐：支持 as_of_date 防未来函数
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import deque


class BeliefLayer:
    """贝叶斯信念层 - 序贯更新概率信念"""

    BASE_PRIOR = 0.50

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.belief_history: Dict[str, deque] = {}

    def _bayes_update(self, prior: float, likelihood: float) -> float:
        """标准贝叶斯更新: P(H|E) = P(E|H)*P(H) / [P(E|H)*P(H) + P(E|¬H)*P(¬H)]"""
        if prior <= 0 or prior >= 1:
            return 0.5
        if likelihood <= 0 or likelihood >= 1:
            return prior
        numerator = likelihood * prior
        denominator = likelihood * prior + (1 - likelihood) * (1 - prior)
        if denominator > 0:
            posterior = numerator / denominator
        else:
            posterior = prior
        return max(0.01, min(0.99, posterior))

    def _compute_continuous_likelihood(
        self, df: pd.DataFrame, ctx: Dict
    ) -> float:
        """
        计算连续似然值 P(E|H)

        v2: 从5档离散 → 多证据连续sigmoid加权

        证据源:
        1. 20日收益率 (权重30%)
        2. 成交量变化 (权重20%)
        3. RPS相对强度 (权重20%) - 从ctx获取
        4. VCP形态质量 (权重15%) - 从ctx获取
        5. 资金流方向 (权重15%) - 从ctx获取
        """
        if df.empty or len(df) < 20:
            return 0.50

        close = df["close"]

        # === 证据1: 20日收益率 ===
        recent_return = close.pct_change(20).iloc[-1]
        if np.isnan(recent_return):
            ret_evidence = 0.5
        else:
            # sigmoid映射: return=0→0.5, return=0.1→0.73, return=-0.1→0.27
            ret_evidence = 1.0 / (1.0 + np.exp(-np.clip(recent_return * 8, -50.0, 50.0)))

        # === 证据2: 成交量变化 ===
        if "volume" in df.columns:
            vol = df["volume"]
            vol_now = vol.iloc[-1]
            vol_ma = vol.rolling(20).mean().iloc[-1]
            if vol_ma > 0:
                vol_ratio = vol_now / vol_ma
                # 放量+上涨=看多, 放量+下跌=看空
                daily_ret = close.pct_change().iloc[-1] if len(close) >= 2 else 0
                if not np.isnan(daily_ret):
                    vol_evidence = 1.0 / (1.0 + np.exp(-np.clip((vol_ratio - 1) * daily_ret * 20, -50.0, 50.0)))
                else:
                    vol_evidence = 0.5
            else:
                vol_evidence = 0.5
        else:
            vol_evidence = 0.5

        # === 证据3: RPS (从ctx获取，跨标的相对强度) ===
        rps = ctx.get("pv_rps_combined", 50)  # 0-100
        if isinstance(rps, (int, float)) and not np.isnan(rps):
            # RPS=50→0.5, RPS=80→0.73, RPS=20→0.27
            rps_evidence = 1.0 / (1.0 + np.exp(-np.clip((rps - 50) * 0.04, -50.0, 50.0)))
        else:
            rps_evidence = 0.5

        # === 证据4: VCP形态质量 (从ctx获取) ===
        vcp_quality = ctx.get("pv_vcp_quality", 0)  # 0-1
        if isinstance(vcp_quality, (int, float)) and not np.isnan(vcp_quality):
            vcp_evidence = 0.3 + vcp_quality * 0.4  # 0.3-0.7
        else:
            vcp_evidence = 0.5

        # === 证据5: 资金流方向 (从ctx获取) ===
        capital_score = ctx.get("capital_score", 0.5)  # 0-1
        if isinstance(capital_score, (int, float)) and not np.isnan(capital_score):
            capital_evidence = capital_score
        else:
            capital_evidence = 0.5

        # === 加权融合 ===
        likelihood = (
            ret_evidence * 0.30
            + vol_evidence * 0.20
            + rps_evidence * 0.20
            + vcp_evidence * 0.15
            + capital_evidence * 0.15
        )

        return float(max(0.01, min(0.99, likelihood)))

    def extract_features(
        self,
        symbol: str,
        df: pd.DataFrame,
        ctx: Dict,
        as_of_date: Optional[str] = None,
    ) -> Dict:
        """提取贝叶斯信念层特征"""
        features = {}

        # 时序对齐
        if as_of_date and not df.empty:
            df = df[df.index <= pd.Timestamp(as_of_date)]

        if symbol not in self.belief_history:
            self.belief_history[symbol] = deque(maxlen=self.max_history)

        if df.empty or len(df) < 20:
            return features

        # 计算连续似然值
        likelihood = self._compute_continuous_likelihood(df, ctx)

        # 获取先验（上一时刻的后验）
        history = self.belief_history[symbol]
        if len(history) > 0:
            current_prior = history[-1][1]
        else:
            current_prior = self.BASE_PRIOR

        # 贝叶斯更新
        posterior = self._bayes_update(current_prior, likelihood)

        # 记录历史
        current_date = None
        if not df.empty:
            current_date = df.index[-1]
        if current_date is not None:
            history.append((str(current_date), posterior))

        features["belief_posterior"] = float(posterior)
        features["belief_prior"] = float(current_prior)
        features["belief_likelihood"] = float(likelihood)
        features["belief_base_prior"] = self.BASE_PRIOR
        features["belief_evidence_count"] = len(history)

        # 信念等级（连续映射后的离散标签）
        if posterior >= 0.75:
            level = "strongly_bullish"
        elif posterior >= 0.60:
            level = "bullish"
        elif posterior <= 0.25:
            level = "strongly_bearish"
        elif posterior <= 0.40:
            level = "bearish"
        else:
            level = "neutral"
        features["belief_level"] = level
        features["belief_signal"] = level.upper()

        # KL散度（信念偏离先验的程度）
        if 0 < posterior < 1:
            kl = posterior * np.log(posterior / self.BASE_PRIOR) + \
                 (1 - posterior) * np.log((1 - posterior) / (1 - self.BASE_PRIOR))
            features["belief_kl"] = float(max(0, kl))
        else:
            features["belief_kl"] = 0.0

        # v2新增: 信念变化量（后验-先验）
        features["belief_delta"] = float(posterior - current_prior)

        # v2新增: 信念动量（最近5次后验的变化趋势）
        if len(history) >= 5:
            recent_posteriors = [h[1] for h in list(history)[-5:]]
            belief_momentum = (recent_posteriors[-1] - recent_posteriors[0]) / 4
            features["belief_momentum"] = float(belief_momentum)
        else:
            features["belief_momentum"] = 0.0

        # 证据权重（证据累计次数/20）
        features["belief_confidence"] = float(min(1.0, len(history) / 20))

        return features

    @staticmethod
    def _kl_divergence(p: float, q: float) -> float:
        """伯努利分布 KL(p||q)，下界 0。"""
        if p <= 0 or p >= 1 or q <= 0 or q >= 1:
            return 0.0
        kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
        return float(max(0.0, kl))

    def _level_of(self, posterior: float) -> str:
        """后验 → 信念等级标签（与 extract_features 阈值一致）。"""
        if posterior >= 0.75:
            return "strongly_bullish"
        elif posterior >= 0.60:
            return "bullish"
        elif posterior <= 0.25:
            return "strongly_bearish"
        elif posterior <= 0.40:
            return "bearish"
        return "neutral"

    def get_market_summary(self) -> Dict:
        """获取市场整体信念汇总。

        返回字段供 market_scanner / main.py 使用：
        count / mean_posterior / median_posterior / std_posterior /
        bullish_count / bearish_count /
        mean_kl / max_kl / strongly_bullish_count / strongly_bearish_count /
        drift_alert_count
        """
        if not self.belief_history:
            return {}

        records: List[tuple] = []  # (symbol, posterior, kl)
        for symbol, history in self.belief_history.items():
            if len(history) > 0:
                posterior = history[-1][1]
                kl = self._kl_divergence(posterior, self.BASE_PRIOR)
                records.append((symbol, posterior, kl))

        if not records:
            return {}

        posteriors = [r[1] for r in records]
        kls = [r[2] for r in records]

        return {
            "count": len(records),
            "mean_posterior": float(np.mean(posteriors)),
            "median_posterior": float(np.median(posteriors)),
            "std_posterior": float(np.std(posteriors)),
            "bullish_count": sum(1 for p in posteriors if p > 0.6),
            "bearish_count": sum(1 for p in posteriors if p < 0.4),
            # 扩展字段
            "mean_kl": float(np.mean(kls)),
            "max_kl": float(np.max(kls)),
            "strongly_bullish_count": sum(1 for p in posteriors if p >= 0.75),
            "strongly_bearish_count": sum(1 for p in posteriors if p <= 0.25),
            "drift_alert_count": sum(1 for k in kls if k > 0.05),
        }

    def get_drift_alerts(self, min_kl: float = 0.05) -> List[Dict]:
        """返回 KL 漂移超过阈值的标的告警列表（按 KL 降序）。"""
        alerts: List[Dict] = []
        for symbol, history in self.belief_history.items():
            if not history:
                continue
            posterior = history[-1][1]
            kl = self._kl_divergence(posterior, self.BASE_PRIOR)
            if kl > min_kl:
                drift_pct = (posterior - self.BASE_PRIOR) / self.BASE_PRIOR * 100.0
                alerts.append({
                    "symbol": symbol,
                    "kl": kl,
                    "posterior": float(posterior),
                    "base_prior": float(self.BASE_PRIOR),
                    "drift_pct": float(drift_pct),
                    "level": self._level_of(posterior),
                })
        alerts.sort(key=lambda x: x["kl"], reverse=True)
        return alerts

    def get_top_beliefs(self, n: int = 10) -> pd.DataFrame:
        """返回信念最强的 top-N 标的（按 |posterior - 0.5| 降序）。"""
        rows = []
        for symbol, history in self.belief_history.items():
            if not history:
                continue
            posterior = history[-1][1]
            kl = self._kl_divergence(posterior, self.BASE_PRIOR)
            rows.append({
                "symbol": symbol,
                "posterior": float(posterior),
                "kl": float(kl),
                "level": self._level_of(posterior),
                "evidence_count": len(history),
            })
        rows.sort(key=lambda x: abs(x["posterior"] - 0.5), reverse=True)
        return pd.DataFrame(rows[:n])
