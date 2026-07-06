# -*- coding: utf-8 -*-
"""
history_data_loader.py — 历史数据加载器

Phase 1 优化：为 Layer3/4/5 提供历史数据回测支持。

功能:
- 从 SQLite 数据库加载历史资金流、龙虎榜、北向资金
- 从 Parquet 文件加载历史板块行情和情绪快照
- 提供 as_of_date 截止日期查询，防未来函数
- 统一的接口供各 Layer 在回测模式下调用

数据文件:
- data_cache/history/capital_flow.db   (SQLite: 资金流/龙虎榜/北向资金)
- data_cache/history/sector_history.parquet (板块行情)
- data_cache/history/sentiment_history.parquet (情绪快照)
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


class HistoryDataLoader:
    """历史数据加载器 — 回测模式下为各 Layer 提供历史面板数据"""

    _instance: Optional["HistoryDataLoader"] = None

    def __init__(self, base_dir: Optional[str] = None):
        """初始化历史数据加载器

        Args:
            base_dir: AStockQuant 根目录，默认自动检测
        """
        if base_dir is None:
            # 自动检测：当前文件位于 AStockQuant/core/ 下
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.base_dir = base_dir
        self.history_dir = os.path.join(base_dir, "data_cache", "history")
        os.makedirs(self.history_dir, exist_ok=True)

        # 数据文件路径
        self.capital_flow_db = os.path.join(self.history_dir, "capital_flow.db")
        self.sector_parquet = os.path.join(self.history_dir, "sector_history.parquet")
        self.sentiment_parquet = os.path.join(self.history_dir, "sentiment_history.parquet")

        # 内存缓存（避免反复读文件）
        self._cache: Dict[str, pd.DataFrame] = {}

    @classmethod
    def get_instance(cls, base_dir: Optional[str] = None) -> "HistoryDataLoader":
        """单例模式获取实例"""
        if cls._instance is None:
            cls._instance = cls(base_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """重置单例（测试用）"""
        cls._instance = None

    # ==================== 资金流数据 ====================

    def get_capital_flow(
        self, symbol: str, as_of_date: Optional[str] = None, lookback: int = 20
    ) -> Dict:
        """获取个股/ETF 历史资金流

        Args:
            symbol: 标的代码
            as_of_date: 截止日期 (YYYY-MM-DD)，None 则取最新
            lookback: 回看天数

        Returns:
            {net_inflow: float, flow_trend: float, flow_rank: float}
        """
        if not os.path.exists(self.capital_flow_db):
            return {"net_inflow": 0.0, "flow_trend": 0.0, "flow_rank": 50.0}

        cache_key = f"capital_flow_{symbol}"
        df = self._cache.get(cache_key)
        if df is None:
            try:
                conn = sqlite3.connect(self.capital_flow_db)
                df = pd.read_sql(
                    "SELECT * FROM etf_capital_flow WHERE symbol = ? ORDER BY date",
                    conn, params=(symbol,),
                )
                conn.close()
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                self._cache[cache_key] = df
            except Exception:
                return {"net_inflow": 0.0, "flow_trend": 0.0, "flow_rank": 50.0}

        if df.empty:
            return {"net_inflow": 0.0, "flow_trend": 0.0, "flow_rank": 50.0}

        # 时序对齐
        if as_of_date:
            df = df[df.index <= pd.Timestamp(as_of_date)]

        if len(df) == 0:
            return {"net_inflow": 0.0, "flow_trend": 0.0, "flow_rank": 50.0}

        recent = df.tail(lookback)
        net_inflow = float(recent["net_inflow"].sum())
        # 资金流趋势：近5日均值 vs 近20日均值
        if len(recent) >= 10:
            flow_5 = float(recent["net_inflow"].tail(5).mean())
            flow_20 = float(recent["net_inflow"].mean())
            flow_trend = flow_5 - flow_20
        else:
            flow_trend = 0.0

        # 排名（如果有 rank 列）
        flow_rank = float(recent["rank"].iloc[-1]) if "rank" in recent.columns else 50.0

        return {
            "net_inflow": net_inflow,
            "flow_trend": flow_trend,
            "flow_rank": flow_rank,
        }

    def get_lhb(
        self, symbol: str, as_of_date: Optional[str] = None, lookback: int = 10
    ) -> Dict:
        """获取龙虎榜数据

        Returns:
            {on_board: bool, net_buy_amount: float, reason: str}
        """
        if not os.path.exists(self.capital_flow_db):
            return {"on_board": False, "net_buy_amount": 0.0, "reason": ""}

        cache_key = "lhb_all"
        df = self._cache.get(cache_key)
        if df is None:
            try:
                conn = sqlite3.connect(self.capital_flow_db)
                df = pd.read_sql("SELECT * FROM lhb_daily ORDER BY date", conn)
                conn.close()
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                self._cache[cache_key] = df
            except Exception:
                return {"on_board": False, "net_buy_amount": 0.0, "reason": ""}

        if df.empty:
            return {"on_board": False, "net_buy_amount": 0.0, "reason": ""}

        # 筛选标的和日期
        mask = df["symbol"] == symbol
        if as_of_date:
            mask &= df["date"] <= pd.Timestamp(as_of_date)
        sub = df[mask].tail(lookback)

        if sub.empty:
            return {"on_board": False, "net_buy_amount": 0.0, "reason": ""}

        return {
            "on_board": True,
            "net_buy_amount": float(sub["net_buy_amount"].iloc[-1]),
            "reason": str(sub["reason"].iloc[-1]) if "reason" in sub.columns else "",
        }

    def get_north_flow(
        self, as_of_date: Optional[str] = None, lookback: int = 20
    ) -> Dict:
        """获取北向资金数据

        Returns:
            {net_buy: float, trend: float}
        """
        if not os.path.exists(self.capital_flow_db):
            return {"net_buy": 0.0, "trend": 0.0}

        cache_key = "north_flow"
        df = self._cache.get(cache_key)
        if df is None:
            try:
                conn = sqlite3.connect(self.capital_flow_db)
                df = pd.read_sql("SELECT * FROM north_flow_daily ORDER BY date", conn)
                conn.close()
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                self._cache[cache_key] = df
            except Exception:
                return {"net_buy": 0.0, "trend": 0.0}

        if df.empty:
            return {"net_buy": 0.0, "trend": 0.0}

        if as_of_date:
            df = df[df.index <= pd.Timestamp(as_of_date)]

        if len(df) == 0:
            return {"net_buy": 0.0, "trend": 0.0}

        recent = df.tail(lookback)
        net_buy = float(recent["net_buy_amount"].sum())
        # 趋势：近5日 vs 近20日
        if len(recent) >= 10:
            buy_5 = float(recent["net_buy_amount"].tail(5).mean())
            buy_20 = float(recent["net_buy_amount"].mean())
            trend = buy_5 - buy_20
        else:
            trend = 0.0

        return {"net_buy": net_buy, "trend": trend}

    # ==================== 板块数据 ====================

    def get_sector_data(
        self, sector: str, as_of_date: Optional[str] = None, lookback: int = 60
    ) -> Dict:
        """获取板块历史行情

        Args:
            sector: 板块名称
            as_of_date: 截止日期
            lookback: 回看天数

        Returns:
            {momentum: float, breadth: float, turnover_rank: float, phase: str}
        """
        cache_key = "sector_history"
        df = self._cache.get(cache_key)
        if df is None:
            if not os.path.exists(self.sector_parquet):
                return {"momentum": 0.0, "breadth": 0.5, "turnover_rank": 50.0, "phase": "neutral"}
            try:
                df = pd.read_parquet(self.sector_parquet)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                self._cache[cache_key] = df
            except Exception:
                return {"momentum": 0.0, "breadth": 0.5, "turnover_rank": 50.0, "phase": "neutral"}

        if df.empty:
            return {"momentum": 0.0, "breadth": 0.5, "turnover_rank": 50.0, "phase": "neutral"}

        # 筛选板块和日期
        sub = df[df["sector"] == sector].copy()
        if as_of_date:
            sub = sub[sub.index <= pd.Timestamp(as_of_date)]

        if len(sub) < 5:
            return {"momentum": 0.0, "breadth": 0.5, "turnover_rank": 50.0, "phase": "neutral"}

        recent = sub.tail(lookback)
        # 板块动量：近20日累计涨跌幅
        if "pct_change" in recent.columns and len(recent) >= 20:
            momentum = float(recent["pct_change"].tail(20).sum())
        else:
            momentum = 0.0

        # 板块宽度：上涨日占比
        if "pct_change" in recent.columns:
            breadth = float((recent["pct_change"] > 0).sum() / len(recent))
        else:
            breadth = 0.5

        # 成交额排名
        if "turnover_rank" in recent.columns:
            turnover_rank = float(recent["turnover_rank"].iloc[-1])
        else:
            turnover_rank = 50.0

        # 轮动阶段
        if "pct_change" in recent.columns and len(recent) >= 20:
            ret_5 = float(recent["pct_change"].tail(5).sum())
            ret_20 = float(recent["pct_change"].tail(20).sum())
            if ret_5 > 3 and ret_20 > 10:
                phase = "hot"
            elif ret_5 > 1 and ret_20 > 5:
                phase = "warming"
            elif ret_5 < -3:
                phase = "cooling"
            else:
                phase = "neutral"
        else:
            phase = "neutral"

        return {
            "momentum": momentum,
            "breadth": breadth,
            "turnover_rank": turnover_rank,
            "phase": phase,
        }

    def get_theme_flow(
        self, theme: str, as_of_date: Optional[str] = None, lookback: int = 20
    ) -> Dict:
        """获取主题资金流数据

        Returns:
            {net_inflow: float, breadth: float, momentum: float}
        """
        if not os.path.exists(self.capital_flow_db):
            return {"net_inflow": 0.0, "breadth": 0.5, "momentum": 0.0}

        cache_key = "theme_flow"
        df = self._cache.get(cache_key)
        if df is None:
            try:
                conn = sqlite3.connect(self.capital_flow_db)
                df = pd.read_sql("SELECT * FROM theme_flow_daily ORDER BY date", conn)
                conn.close()
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                self._cache[cache_key] = df
            except Exception:
                return {"net_inflow": 0.0, "breadth": 0.5, "momentum": 0.0}

        if df.empty:
            return {"net_inflow": 0.0, "breadth": 0.5, "momentum": 0.0}

        mask = df["theme"] == theme
        if as_of_date:
            mask &= df["date"] <= pd.Timestamp(as_of_date)
        sub = df[mask].tail(lookback)

        if sub.empty:
            return {"net_inflow": 0.0, "breadth": 0.5, "momentum": 0.0}

        net_inflow = float(sub["net_inflow"].sum()) if "net_inflow" in sub.columns else 0.0
        breadth = float(sub["breadth"].iloc[-1]) if "breadth" in sub.columns else 0.5
        momentum = float(sub["momentum"].iloc[-1]) if "momentum" in sub.columns else 0.0

        return {"net_inflow": net_inflow, "breadth": breadth, "momentum": momentum}

    # ==================== 情绪数据 ====================

    def get_sentiment(
        self, symbol: str, as_of_date: Optional[str] = None
    ) -> Dict:
        """获取历史情绪快照

        Returns:
            {sentiment_score: float, confidence: float, summary: str}
        """
        cache_key = "sentiment_history"
        df = self._cache.get(cache_key)
        if df is None:
            if not os.path.exists(self.sentiment_parquet):
                return {"sentiment_score": 0.5, "confidence": 0.0, "summary": ""}

            try:
                df = pd.read_parquet(self.sentiment_parquet)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                self._cache[cache_key] = df
            except Exception:
                return {"sentiment_score": 0.5, "confidence": 0.0, "summary": ""}

        if df.empty:
            return {"sentiment_score": 0.5, "confidence": 0.0, "summary": ""}

        # 筛选标的和日期
        mask = df["symbol"] == symbol
        if as_of_date:
            mask &= df["date"] <= pd.Timestamp(as_of_date)
        sub = df[mask].tail(1)

        if sub.empty:
            return {"sentiment_score": 0.5, "confidence": 0.0, "summary": ""}

        return {
            "sentiment_score": float(sub["sentiment_score"].iloc[0]),
            "confidence": float(sub["confidence"].iloc[0]) if "confidence" in sub.columns else 0.0,
            "summary": str(sub["summary"].iloc[0]) if "summary" in sub.columns else "",
        }

    # ==================== 工具方法 ====================

    def has_history_data(self) -> bool:
        """检查是否存在历史数据文件"""
        return (
            os.path.exists(self.capital_flow_db)
            or os.path.exists(self.sector_parquet)
            or os.path.exists(self.sentiment_parquet)
        )

    def get_data_coverage(self) -> Dict:
        """获取数据覆盖情况"""
        coverage = {
            "capital_flow": False,
            "sector_history": False,
            "sentiment_history": False,
            "capital_flow_rows": 0,
            "sector_rows": 0,
            "sentiment_rows": 0,
        }

        if os.path.exists(self.capital_flow_db):
            coverage["capital_flow"] = True
            try:
                conn = sqlite3.connect(self.capital_flow_db)
                cursor = conn.execute("SELECT COUNT(*) FROM etf_capital_flow")
                coverage["capital_flow_rows"] = cursor.fetchone()[0]
                conn.close()
            except Exception:
                pass

        if os.path.exists(self.sector_parquet):
            coverage["sector_history"] = True
            try:
                df = pd.read_parquet(self.sector_parquet)
                coverage["sector_rows"] = len(df)
            except Exception:
                pass

        if os.path.exists(self.sentiment_parquet):
            coverage["sentiment_history"] = True
            try:
                df = pd.read_parquet(self.sentiment_parquet)
                coverage["sentiment_rows"] = len(df)
            except Exception:
                pass

        return coverage

    def clear_cache(self):
        """清除内存缓存"""
        self._cache.clear()
