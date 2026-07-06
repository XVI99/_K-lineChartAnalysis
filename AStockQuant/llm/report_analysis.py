# -*- coding: utf-8 -*-
"""
report_analysis.py — 研报/板块解读模块

使用 LLM 分析研报文本或板块行情数据，输出板块评级和驱动因素。
用于增强 layer3_sector 的板块轮动信号。

输出结构:
{
    "rating": "overweight/neutral/underweight",
    "score": 0-100,
    "drivers": ["驱动因素1", ...],
    "risks": ["风险点1", ...],
    "related_etfs": ["510300", ...],
    "time_horizon": "short/medium/long",
    "summary": "一句话总结"
}
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from AStockQuant.core.llm_client import llm


SYSTEM_PROMPT = """你是一位专业的A股行业分析师，擅长解读券商研报和板块行情数据，提取核心投资逻辑。
你需要分析研报或板块数据，返回结构化的JSON数据。
注意：
1. rating: overweight(看好)/neutral(中性)/underweight(看空)
2. score: 0-100的综合评分，50为中性
3. drivers: 列出2-4个核心驱动因素
4. risks: 列出1-3个主要风险点
5. related_etfs: 列出相关的ETF代码（6位数字）
6. time_horizon: 短期(1个月内)/中期(1-3个月)/长期(3个月以上)
7. summary: 用一句话总结投资逻辑"""


class ReportAnalyzer:
    """研报/板块分析器"""

    def __init__(self):
        self._enabled = llm.enabled and llm.config.get("tasks", {}).get("report_analysis", True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def analyze_report(
        self,
        report_text: str,
        sector: str = "",
        date: str = "",
    ) -> Optional[Dict]:
        """
        分析研报文本

        Args:
            report_text: 研报正文（截取前3000字）
            sector: 所属板块（可选）
            date: 研报日期（可选）

        Returns:
            结构化分析结果
        """
        if not self._enabled:
            return None

        text = report_text[:3000]
        header = f"板块: {sector}\n日期: {date}\n" if sector or date else ""

        prompt = f"""请分析以下A股研报，提取核心投资逻辑，返回JSON格式数据：

{header}
研报内容:
{text}

返回JSON格式如下：
{{
    "rating": "neutral",
    "score": 50,
    "drivers": [],
    "risks": [],
    "related_etfs": [],
    "time_horizon": "medium",
    "summary": ""
}}"""

        return llm.chat_json(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=2048,
        )

    def analyze_sector(
        self,
        sector_name: str,
        sector_data: Dict,
        market_context: str = "",
    ) -> Optional[Dict]:
        """
        分析板块行情数据

        Args:
            sector_name: 板块名称
            sector_data: 板块数据（如涨幅、成交额、领涨股等）
            market_context: 市场环境描述（如当前regime）

        Returns:
            结构化分析结果
        """
        if not self._enabled:
            return None

        data_str = json.dumps(sector_data, ensure_ascii=False, indent=2)
        prompt = f"""请分析以下A股板块行情数据，判断板块投资价值，返回JSON格式数据：

板块: {sector_name}
市场环境: {market_context}
板块数据:
{data_str}

返回JSON格式如下：
{{
    "rating": "neutral",
    "score": 50,
    "drivers": [],
    "risks": [],
    "related_etfs": [],
    "time_horizon": "medium",
    "summary": ""
}}"""

        return llm.chat_json(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=2048,
        )

    def compare_sectors(
        self,
        sectors: List[Dict[str, any]],
    ) -> Optional[Dict]:
        """
        对比多个板块的相对吸引力

        Args:
            sectors: [{"name": "半导体", "data": {...}}, ...]

        Returns:
            {
                "ranking": [{"sector": "半导体", "score": 75, "reason": "..."}],
                "top_pick": "半导体",
                "summary": "对比分析总结"
            }
        """
        if not self._enabled or len(sectors) < 2:
            return None

        sectors_str = json.dumps(sectors, ensure_ascii=False, indent=2)
        prompt = f"""请对比以下A股板块的相对投资吸引力，返回JSON格式数据：

板块列表:
{sectors_str}

返回JSON格式如下：
{{
    "ranking": [
        {{"sector": "板块名", "score": 75, "reason": "原因"}},
        ...
    ],
    "top_pick": "最强板块",
    "summary": "对比分析总结"
}}

请按 score 从高到低排序。"""

        return llm.chat_json(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=2048,
        )
