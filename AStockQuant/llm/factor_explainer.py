# -*- coding: utf-8 -*-
"""
factor_explainer.py — 因子解释模块

使用 LLM 解释因子的含义和当前表现，帮助理解策略决策。
用于研究报告和策略可解释性。

输出: 自然语言解释文本
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from AStockQuant.core.llm_client import llm


SYSTEM_PROMPT = """你是一位量化投资研究员，擅长解释金融因子的含义、逻辑和表现。
你的解释需要专业但易懂，帮助投资者理解：
1. 因子衡量的是什么
2. 当前因子值意味着什么
3. 在当前市场环境下因子为什么有效或无效
4. 该因子的局限性和注意事项"""


# 因子名称中文映射
FACTOR_NAMES = {
    "pv_rps_20": "20日相对价格强度(RPS)",
    "pv_rps_combined": "综合相对价格强度",
    "tech_macd_hist": "MACD柱状图",
    "sent_combined_score": "综合情绪得分",
    "sector_momentum_short": "短期板块动量",
    "sector_breadth": "板块广度",
    "sector_combined_score": "板块综合得分",
    "belief_delta": "贝叶斯信念变化",
    "pv_vcp_volatility_contraction": "波动率收缩(VCP)",
    "pv_volume_surge_ratio": "成交量激增比",
    "pv_price_volume_divergence": "量价背离",
    "capital_flow_score": "资金流向得分",
    "sector_rotation_signal": "板块轮动信号",
}


class FactorExplainer:
    """因子解释器"""

    def __init__(self):
        self._enabled = llm.enabled and llm.config.get("tasks", {}).get("factor_explainer", True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def explain_factor(
        self,
        factor_name: str,
        factor_value: float,
        rank_pct: Optional[float] = None,
        regime: Optional[Dict] = None,
        ic_mean: Optional[float] = None,
    ) -> str:
        """
        解释单个因子

        Args:
            factor_name: 因子名（如 pv_rps_20）
            factor_value: 因子当前值
            rank_pct: 因子在截面中的百分位排名 (0-1)
            regime: 当前市场状态 {"trend": "trend_bull", "vol": "mid_vol", ...}
            ic_mean: 该因子的历史平均IC

        Returns:
            解释文本（LLM不可用时返回空字符串）
        """
        if not self._enabled:
            return ""

        cn_name = FACTOR_NAMES.get(factor_name, factor_name)
        regime_str = json.dumps(regime, ensure_ascii=False) if regime else "未知"

        prompt = f"""请解释以下量化因子的含义和当前状态：

因子名称: {factor_name}（{cn_name}）
当前值: {factor_value}
截面排名百分位: {rank_pct if rank_pct is not None else '未知'}
历史平均IC: {ic_mean if ic_mean is not None else '未知'}
当前市场状态: {regime_str}

请从以下角度解释：
1. 这个因子衡量的是什么市场信号
2. 当前值反映的市场含义
3. 在当前市场状态下该因子是否可靠
4. 需要注意什么

请用简洁的中文回答（200-300字）。"""

        return llm.chat(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=2048,
        )

    def explain_ranking(
        self,
        top_factors: List[Dict],
        bottom_factors: List[Dict],
        regime: Optional[Dict] = None,
    ) -> str:
        """
        解释因子排名的整体含义

        Args:
            top_factors: [{"name": "pv_rps_20", "value": 0.85, "rank_pct": 0.95}, ...]
            bottom_factors: [{"name": "...", ...}, ...]
            regime: 当前市场状态

        Returns:
            整体解释文本
        """
        if not self._enabled:
            return ""

        top_str = json.dumps(top_factors[:5], ensure_ascii=False, indent=2)
        bot_str = json.dumps(bottom_factors[:3], ensure_ascii=False, indent=2)
        regime_str = json.dumps(regime, ensure_ascii=False) if regime else "未知"

        prompt = f"""请分析以下ETF因子排名的整体含义：

当前市场状态: {regime_str}

排名靠前的因子:
{top_str}

排名靠后的因子:
{bot_str}

请分析：
1. 排名靠前的因子反映了什么市场特征
2. 排名靠后的因子为什么表现不佳
3. 这些因子综合来看，暗示了什么样的投资机会或风险
4. 对当前持仓有什么建议

请用简洁的中文回答（300-400字）。"""

        return llm.chat(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=2048,
        )

    def explain_regime_factors(
        self,
        regime_label: str,
        effective_factors: List[str],
        ineffective_factors: List[str],
    ) -> str:
        """
        解释在特定regime下哪些因子有效/无效

        Args:
            regime_label: 如 "trend_bull_mid_vol_theme_market"
            effective_factors: 在该regime下有效的因子列表
            ineffective_factors: 在该regime下无效的因子列表

        Returns:
            解释文本
        """
        if not self._enabled:
            return ""

        eff = ", ".join(effective_factors[:5])
        ineff = ", ".join(ineffective_factors[:5])

        prompt = f"""请分析在以下市场状态下因子的表现差异：

市场状态: {regime_label}
有效因子: {eff}
无效因子: {ineff}

请解释：
1. 为什么这些因子在当前市场状态下有效
2. 为什么另一些因子在当前市场状态下无效
3. 投资者应该重点关注哪些因子
4. 这种状态下的投资策略建议

请用简洁的中文回答（200-300字）。"""

        return llm.chat(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=2048,
        )
