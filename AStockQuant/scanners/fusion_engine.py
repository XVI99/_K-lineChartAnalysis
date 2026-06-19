# -*- coding: utf-8 -*-
"""
fusion_engine.py — 多层信号融合器

负责将来自各层 (Layer) 的因子融合为最终的买卖信号。
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, Tuple

import numpy as np
import pandas as pd


class SignalType(Enum):
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


class MarketRegime(Enum):
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"


class FusionEngine:
    """多因子信号融合"""

    def __init__(self, ml_weight=0.35, rps_weight=0.30, vcp_weight=0.20, pattern_weight=0.15):
        self.ml_w = ml_weight
        self.rps_w = rps_weight
        self.vcp_w = vcp_weight
        self.pat_w = pattern_weight

    def fuse(
        self,
        ml_prob: float,
        rps_value: float,
        vcp_quality: float,
        pattern_score: float,
        regime: MarketRegime = MarketRegime.NEUTRAL,
    ) -> Tuple[SignalType, float, float]:
        """
        返回 (信号类型, 置信度, 综合得分)
        """
        ml_s = ml_prob
        rps_s = rps_value / 100.0
        vcp_s = vcp_quality / 100.0
        pat_s = (pattern_score + 10) / 20.0

        mw, rw, vw, pw = self.ml_w, self.rps_w, self.vcp_w, self.pat_w
        if regime == MarketRegime.BULL:
            mw += 0.05; pw += 0.05; rw -= 0.05; vw -= 0.05
        elif regime == MarketRegime.BEAR:
            rw += 0.1; mw -= 0.05; vw -= 0.05

        comp = ml_s * mw + rps_s * rw + vcp_s * vw + pat_s * pw

        sigs = [
            1 if ml_s > 0.55 else (-1 if ml_s < 0.45 else 0),
            1 if rps_s > 0.6 else (-1 if rps_s < 0.4 else 0),
            1 if vcp_s > 0.5 else 0,
            1 if pat_s > 0.55 else (-1 if pat_s < 0.45 else 0),
        ]
        agreement = sum(sigs) / len(sigs)
        conf = max(0.0, min(1.0, comp * (1 + abs(agreement) * 0.2)))

        if comp >= 0.75:
            sig = SignalType.STRONG_BUY
        elif comp >= 0.55:
            sig = SignalType.BUY
        elif comp <= 0.25:
            sig = SignalType.STRONG_SELL
        elif comp <= 0.45:
            sig = SignalType.SELL
        else:
            sig = SignalType.NEUTRAL

        return sig, round(conf, 3), round(comp, 3)

    # ==================== 贝叶斯融合方法 ====================

    def fuse_bayesian(
        self,
        belief_posterior: float,
        rps_value: float,
        vcp_quality: float,
        pattern_score: float,
        regime: MarketRegime = MarketRegime.NEUTRAL,
    ) -> Tuple[SignalType, float, float]:
        """
        贝叶斯融合：将后验概率与量价/技术信号融合。

        与传统 fuse() 的区别：
        - 输入：信念后验 P(H|E)（贝叶斯更新结果）替代 ml_prob
        - 权重：信念后验权重显著提高（0.45），ML 降权
        - 融合方式：Logit-space 乘法融合

        Args:
            belief_posterior: 贝叶斯信念后验 P(H|E)（0~1）
            rps_value: RPS 值（0~100）
            vcp_quality: VCP 质量（0~100）
            pattern_score: 形态分数（-10~+10）
            regime: 市场状态

        Returns:
            (SignalType, 置信度, 综合得分)
        """
        import math

        # 1. 信念后验 logit
        def _logit(p):
            p = max(0.001, min(0.999, p))
            return math.log(p / (1 - p))

        def _sigmoid(x):
            return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

        belief_logit = _logit(belief_posterior)

        # 2. RPS 似然注入（每超 50 的点数，转化为 log odds）
        rps_logit = (rps_value - 50) / 100.0 * 2.0  # ±1.0 logit
        # 3. VCP 似然注入
        vcp_logit = (vcp_quality / 100.0 - 0.5) * 1.0
        # 4. 形态似然注入
        pat_logit = pattern_score / 10.0 * 0.8

        # 5. 动态权重（根据市场状态调整）
        if regime == MarketRegime.BULL:
            w_belief, w_rps, w_vcp, w_pat = 0.35, 0.25, 0.20, 0.20
        elif regime == MarketRegime.BEAR:
            w_belief, w_rps, w_vcp, w_pat = 0.45, 0.25, 0.15, 0.15
        else:
            w_belief, w_rps, w_vcp, w_pat = 0.40, 0.25, 0.20, 0.15

        # 6. Logit 加权融合
        total_w = w_belief + w_rps + w_vcp + w_pat
        fused_logit = (
            belief_logit * w_belief
            + rps_logit * w_rps
            + vcp_logit * w_vcp
            + pat_logit * w_pat
        ) / total_w

        composite = _sigmoid(fused_logit)

        # 7. 计算置信度（信念稳定性和信号一致性）
        belief_confidence = 1.0 - abs(belief_posterior - 0.5) * 2.0
        signal_agreement = 1.0 - abs(belief_posterior - composite) * 2.0
        conf = max(0.0, min(1.0, belief_confidence * signal_agreement * (1 + abs(composite - 0.5))))

        # 8. 信号分类
        if composite >= 0.75:
            sig = SignalType.STRONG_BUY
        elif composite >= 0.60:
            sig = SignalType.BUY
        elif composite <= 0.25:
            sig = SignalType.STRONG_SELL
        elif composite <= 0.40:
            sig = SignalType.SELL
        else:
            sig = SignalType.NEUTRAL

        return sig, round(conf, 3), round(composite, 3)

    # ==================== 贝叶斯决策工具 ====================

    def classify_by_posterior(
        self,
        posterior: float,
        strong_buy_threshold: float = 0.75,
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.40,
        strong_sell_threshold: float = 0.25,
    ) -> SignalType:
        """基于信念后验直接分类（不依赖量价信号）"""
        if posterior >= strong_buy_threshold:
            return SignalType.STRONG_BUY
        elif posterior >= buy_threshold:
            return SignalType.BUY
        elif posterior <= strong_sell_threshold:
            return SignalType.STRONG_SELL
        elif posterior <= sell_threshold:
            return SignalType.SELL
        return SignalType.NEUTRAL

    def regime_adjustment(
        self,
        base_weights: Dict[str, float],
        regime: MarketRegime,
    ) -> Dict[str, float]:
        """
        根据市场状态动态调整融合权重。

        熊市：提高信念后验权重，降低量价依赖
        牛市：提高 RPS + 形态权重
        """
        w = base_weights.copy()
        if regime == MarketRegime.BULL:
            w["belief"] = w.get("belief", 0.40) + 0.05
            w["pattern"] = w.get("pattern", 0.15) + 0.05
            w["rps"] = w.get("rps", 0.25) - 0.05
            w["vcp"] = w.get("vcp", 0.20) - 0.05
        elif regime == MarketRegime.BEAR:
            w["belief"] = w.get("belief", 0.40) + 0.10
            w["rps"] = w.get("rps", 0.25) + 0.05
            w["vcp"] = w.get("vcp", 0.20) - 0.10
            w["pattern"] = w.get("pattern", 0.15) - 0.05
        # 中性状态：不做调整
        return w


def build_combined_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 KDJ 信号构建强做多/强做空信号。

    参数:
        df: 包含 BuySignal / SellSignal 列的 DataFrame
    返回:
        添加了 StrongLong / StrongShort 列的 DataFrame

    用法示例:
        df = calculate_kdj(df)
        df = identify_kdj_signals(df)
        df = build_combined_signals(df)
    """
    df = df.copy()
    df["StrongLong"] = df["BuySignal"].astype(bool)
    df["StrongShort"] = df["SellSignal"].astype(bool)
    return df
