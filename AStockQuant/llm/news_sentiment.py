# -*- coding: utf-8 -*-
"""
news_sentiment.py — 新闻情绪分析模块

使用 LLM 分析新闻文本的情绪倾向，输出结构化情绪分数。
用于增强 layer5_sentiment 的情绪因子。

输出结构:
{
    "sentiment_score": -1.0 ~ 1.0,   # 情绪分数（负=利空，正=利好）
    "sentiment_label": "positive/negative/neutral",
    "keywords": ["关键词1", "关键词2"],
    "related_sectors": ["半导体", "新能源"],
    "impact_level": "high/medium/low",
    "summary": "一句话摘要"
}
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from AStockQuant.core.llm_client import llm


SYSTEM_PROMPT = """你是一位专业的A股市场新闻分析师，擅长从新闻文本中提取市场情绪和板块影响信息。
你需要分析新闻对A股市场的影响，并返回结构化的JSON数据。
注意：
1. sentiment_score 范围 -1.0 到 1.0，-1表示极端利空，1表示极端利好，0表示中性
2. related_sectors 用中文板块名（如：半导体、新能源、医药、消费、金融、科技等）
3. impact_level 表示该新闻对市场的影响程度
4. summary 用一句话概括新闻核心内容和对市场的影响"""


class NewsSentimentAnalyzer:
    """新闻情绪分析器"""

    def __init__(self):
        self._enabled = llm.enabled and llm.config.get("tasks", {}).get("news_sentiment", True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def analyze(
        self,
        title: str,
        content: str = "",
        source: str = "",
    ) -> Optional[Dict]:
        """
        分析单条新闻的情绪

        Args:
            title: 新闻标题
            content: 新闻正文（可选）
            source: 新闻来源（可选）

        Returns:
            结构化情绪分析结果（LLM不可用时返回None）
        """
        if not self._enabled:
            return None

        news_text = f"标题: {title}"
        if content:
            news_text += f"\n正文: {content[:2000]}"
        if source:
            news_text += f"\n来源: {source}"

        prompt = f"""请分析以下新闻对A股市场的影响，返回JSON格式数据：

{news_text}

返回JSON格式如下：
{{
    "sentiment_score": 0.0,
    "sentiment_label": "neutral",
    "keywords": [],
    "related_sectors": [],
    "impact_level": "low",
    "summary": ""
}}"""

        result = llm.chat_json(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=1024,
        )
        return result

    def analyze_batch(
        self,
        news_list: List[Dict[str, str]],
    ) -> List[Optional[Dict]]:
        """
        批量分析新闻情绪

        Args:
            news_list: [{"title": "...", "content": "...", "source": "..."}, ...]

        Returns:
            分析结果列表
        """
        if not self._enabled:
            return [None] * len(news_list)

        results = []
        for i, news in enumerate(news_list):
            r = self.analyze(
                title=news.get("title", ""),
                content=news.get("content", ""),
                source=news.get("source", ""),
            )
            results.append(r)
            if (i + 1) % 5 == 0:
                print(f"[NewsSentiment] batch {i+1}/{len(news_list)}")
        return results

    def aggregate_sentiment(self, analyses: List[Optional[Dict]]) -> Dict:
        """
        聚合多条新闻的情绪，输出综合情绪分数

        Args:
            analyses: analyze() 返回的结果列表

        Returns:
            {
                "overall_score": float,       # 加权平均情绪
                "positive_count": int,
                "negative_count": int,
                "neutral_count": int,
                "high_impact_news": List,     # 高影响新闻
                "hot_sectors": Dict[str, float],  # 热门板块及平均情绪
            }
        """
        valid = [a for a in analyses if a is not None]
        if not valid:
            return {
                "overall_score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "high_impact_news": [],
                "hot_sectors": {},
            }

        scores = [a.get("sentiment_score", 0.0) for a in valid]
        labels = [a.get("sentiment_label", "neutral") for a in valid]
        impacts = [a for a in valid if a.get("impact_level") == "high"]

        sector_scores: Dict[str, List[float]] = {}
        for a in valid:
            for s in a.get("related_sectors", []):
                sector_scores.setdefault(s, []).append(a.get("sentiment_score", 0.0))

        hot_sectors = {
            s: sum(v) / len(v)
            for s, v in sorted(sector_scores.items(), key=lambda x: abs(sum(x[1]) / len(x[1])), reverse=True)[:5]
        }

        return {
            "overall_score": sum(scores) / len(scores),
            "positive_count": sum(1 for l in labels if l == "positive"),
            "negative_count": sum(1 for l in labels if l == "negative"),
            "neutral_count": sum(1 for l in labels if l == "neutral"),
            "high_impact_news": impacts[:5],
            "hot_sectors": hot_sectors,
        }
