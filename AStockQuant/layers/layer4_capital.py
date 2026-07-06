"""
Layer4 - 资金层
=====================

功能: 真实资金流数据分析

v2 改进:
- 接入 akshare 个股资金流（主力净流入/超大单/大单/中单/小单）
- 接入 akshare 龙虎榜明细（上榜次数/净买额/成交占比）
- 接入北向资金（沪深港通）作为市场上下文
- 接入融资融券明细/汇总（融资余额变化与市场杠杆情绪）
- 保留量价近似作为 API 降级兜底
- 时序对齐：支持 as_of_date 防未来函数
- 连续评分替代布尔标记

数据源:
- akshare.stock_individual_fund_flow: 个股/ETF 资金流
- akshare.stock_lhb_detail_em: 龙虎榜明细
- akshare.stock_hsgt_fund_flow_summary_em: 北向资金汇总
- akshare.stock_margin_detail_sse / stock_margin_detail_szse: 融资融券明细
- akshare.stock_margin_sse / stock_margin_szse: 市场融资融券汇总兜底
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta

try:
    import akshare as ak
    _AK = True
except ImportError:
    _AK = False


class CapitalLayer:
    """
    资金层 - 真实资金流向分析
    追踪大资金动向是盈利的关键
    """

    # 沪深市场标识
    SH_PREFIXES = ("5", "6", "9")
    SZ_PREFIXES = ("0", "1", "2", "3")

    def __init__(self):
        # 资金流缓存（避免同一天多次请求同一标的）
        self._flow_cache: Dict[str, pd.DataFrame] = {}
        self._lhb_cache: Dict[str, pd.DataFrame] = {}
        self._margin_cache: Dict[str, pd.DataFrame] = {}
        self._north_cache: Optional[pd.DataFrame] = None
        self._cache_date: Optional[str] = None

    def _get_market(self, symbol: str) -> str:
        """获取市场标识 sh/sz"""
        code = symbol.replace(".SH", "").replace(".SZ", "").replace("sh", "").replace("sz", "")
        if code.startswith(self.SH_PREFIXES):
            return "sh"
        return "sz"

    def _fetch_fund_flow(self, symbol: str) -> pd.DataFrame:
        """获取个股/ETF资金流数据（带缓存）"""
        if symbol in self._flow_cache:
            return self._flow_cache[symbol]

        if not _AK:
            return pd.DataFrame()

        try:
            market = self._get_market(symbol)
            df = ak.stock_individual_fund_flow(stock=symbol, market=market)
            if df is None or df.empty:
                return pd.DataFrame()

            # 归一化列名（中文→英文）
            rename = {
                "日期": "date",
                "收盘价": "close",
                "涨跌幅": "pct_change",
                "主力净流入-净额": "main_inflow",
                "主力净流入-净占比": "main_inflow_pct",
                "超大单净流入-净额": "super_large_inflow",
                "超大单净流入-净占比": "super_large_pct",
                "大单净流入-净额": "large_inflow",
                "大单净流入-净占比": "large_pct",
                "中单净流入-净额": "medium_inflow",
                "中单净流入-净占比": "medium_pct",
                "小单净流入-净额": "small_inflow",
                "小单净流入-净占比": "small_pct",
            }
            df = df.rename(columns=rename)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            self._flow_cache[symbol] = df
            return df
        except Exception:
            return pd.DataFrame()

    def _fetch_lhb(self, symbol: str, as_of_date: Optional[str] = None) -> pd.DataFrame:
        """获取近30日龙虎榜明细（带缓存）"""
        if not _AK:
            return pd.DataFrame()

        end = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.today()
        start = end - pd.Timedelta(days=45)
        cache_key = f"{symbol}:{start:%Y%m%d}:{end:%Y%m%d}"
        if cache_key in self._lhb_cache:
            return self._lhb_cache[cache_key]

        try:
            api = getattr(ak, "stock_lhb_detail_em", None)
            if api is None:
                return pd.DataFrame()
            df = api(start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
            if df is None or df.empty:
                return pd.DataFrame()

            code_col = self._pick_column(df, ("代码", "股票代码", "证券代码"))
            date_col = self._pick_column(df, ("上榜日", "上榜日期", "日期"))
            if code_col is None:
                return pd.DataFrame()

            result = df[df[code_col].astype(str).str.zfill(6) == str(symbol).zfill(6)].copy()
            if date_col and not result.empty:
                result["date"] = pd.to_datetime(result[date_col], errors="coerce")
                result = result.dropna(subset=["date"]).set_index("date").sort_index()
            self._lhb_cache[cache_key] = result
            return result
        except Exception:
            return pd.DataFrame()

    def _fetch_margin_flow(self, symbol: str, as_of_date: Optional[str] = None) -> pd.DataFrame:
        """获取融资融券明细；若标的无明细则用交易所汇总作市场杠杆情绪兜底"""
        if not _AK:
            return pd.DataFrame()

        end = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.today()
        cache_key = f"{symbol}:{end:%Y%m%d}"
        if cache_key in self._margin_cache:
            return self._margin_cache[cache_key]

        code = str(symbol).zfill(6)
        is_sh = self._get_market(symbol) == "sh"

        # 交易所明细接口通常只保留近期交易日；向前回退最多10天。
        for offset in range(0, 10):
            date = (end - pd.Timedelta(days=offset)).strftime("%Y%m%d")
            try:
                if is_sh and hasattr(ak, "stock_margin_detail_sse"):
                    df = ak.stock_margin_detail_sse(date=date)
                elif (not is_sh) and hasattr(ak, "stock_margin_detail_szse"):
                    df = ak.stock_margin_detail_szse(date=date)
                else:
                    df = pd.DataFrame()

                if df is not None and not df.empty:
                    code_col = self._pick_column(df, ("证券代码", "标的证券代码", "代码"))
                    if code_col:
                        result = df[df[code_col].astype(str).str.zfill(6) == code].copy()
                        if not result.empty:
                            result["date"] = pd.to_datetime(date)
                            result = result.set_index("date")
                            self._margin_cache[cache_key] = result
                            return result
            except Exception:
                continue

        # 兜底：市场级融资融券汇总，仍然是真实交易所数据，但不是单ETF明细。
        try:
            api_name = "stock_margin_sse" if is_sh else "stock_margin_szse"
            api = getattr(ak, api_name, None)
            if api is None:
                return pd.DataFrame()
            df = api()
            if df is None or df.empty:
                return pd.DataFrame()
            date_col = self._pick_column(df, ("日期", "信用交易日期"))
            if date_col:
                df = df.copy()
                df["date"] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=["date"])
                df = df[df["date"] <= end].set_index("date").sort_index()
            self._margin_cache[cache_key] = df.tail(20)
            return self._margin_cache[cache_key]
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _pick_column(df: pd.DataFrame, candidates) -> Optional[str]:
        """从字段候选中挑选实际存在的列，兼容 akshare 中文列名变化。"""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _fetch_north_flow(self) -> pd.DataFrame:
        """获取北向资金汇总（每日缓存）"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._north_cache is not None and self._cache_date == today:
            return self._north_cache

        if not _AK:
            return pd.DataFrame()

        try:
            df = ak.stock_hsgt_fund_flow_summary_em()
            if df is None or df.empty:
                return pd.DataFrame()
            self._north_cache = df
            self._cache_date = today
            return df
        except Exception:
            return pd.DataFrame()

    def extract_features(
        self,
        symbol: str,
        df: pd.DataFrame,
        ctx: Dict,
        as_of_date: Optional[str] = None,
    ) -> Dict:
        """提取资金层特征

        Args:
            symbol: ETF代码
            df: OHLCV数据
            ctx: 上下文
            as_of_date: 截止日期（防未来函数）

        v3 改进: 回测模式优先从 HistoryDataLoader 读取历史资金流数据，
                无历史数据时回退到 akshare 实时获取或量价近似。
        """
        features = {}

        # 时序对齐
        if as_of_date and not df.empty:
            df = df[df.index <= pd.Timestamp(as_of_date)]

        if df.empty or len(df) < 20:
            return features

        # ==================== 回测模式：优先从历史数据库读取 ====================
        if as_of_date:
            history_loaded = False
            try:
                from core.history_data_loader import HistoryDataLoader
                loader = HistoryDataLoader.get_instance()
                if loader.has_history_data():
                    # 1. 历史资金流
                    flow_data = loader.get_capital_flow(symbol, as_of_date=as_of_date)
                    if flow_data["net_inflow"] != 0.0 or flow_data["flow_trend"] != 0.0:
                        features["capital_data_source"] = "history_db"
                        features["capital_main_inflow"] = flow_data["net_inflow"]
                        features["capital_main_inflow_pct"] = 0.0
                        features["capital_main_inflow_5d"] = flow_data["net_inflow"]
                        features["capital_main_inflow_20d"] = flow_data["net_inflow"]
                        features["capital_trend"] = 1.0 + flow_data["flow_trend"] / max(1, abs(flow_data["net_inflow"]))
                        features["capital_activity"] = 1.0
                        features["capital_consecutive_inflow"] = 0
                        features["capital_is_surge"] = False
                        history_loaded = True

                    # 2. 历史北向资金
                    north_data = loader.get_north_flow(as_of_date=as_of_date)
                    if north_data["net_buy"] != 0.0:
                        features["capital_north_net"] = north_data["net_buy"]
                        if north_data["net_buy"] > 5e8:
                            features["capital_north_score"] = 0.8
                        elif north_data["net_buy"] > 0:
                            features["capital_north_score"] = 0.6
                        elif north_data["net_buy"] > -5e8:
                            features["capital_north_score"] = 0.4
                        else:
                            features["capital_north_score"] = 0.2

                    # 3. 历史龙虎榜
                    lhb_data = loader.get_lhb(symbol, as_of_date=as_of_date)
                    if lhb_data["on_board"]:
                        features["capital_lhb_on_board"] = True
                        features["capital_lhb_count_30d"] = 1
                        features["capital_lhb_net_buy"] = lhb_data["net_buy_amount"]
                        features["capital_lhb_buy_amount"] = lhb_data["net_buy_amount"]
                        features["capital_lhb_sell_amount"] = 0.0
                        features["capital_lhb_turnover_pct"] = 0.0
                        score = 0.5
                        if lhb_data["net_buy_amount"] > 0:
                            score += 0.2
                        elif lhb_data["net_buy_amount"] < 0:
                            score -= 0.2
                        features["capital_lhb_score"] = float(max(0.0, min(1.0, score)))
                    else:
                        features.update(self._neutral_lhb_features())

                    # 4. 融资融券（历史库暂无，用中性值）
                    features.update(self._neutral_margin_features())

                    if history_loaded:
                        features["capital_score"] = self._compute_capital_score(features)
                        return features
            except Exception:
                pass  # 回退到原有逻辑

        # ==================== 实时模式：原有逻辑 ====================
        # 1. 真实资金流（akshare）
        flow_df = self._fetch_fund_flow(symbol)

        if as_of_date and not flow_df.empty:
            flow_df = flow_df[flow_df.index <= pd.Timestamp(as_of_date)]

        if not flow_df.empty and len(flow_df) >= 5:
            features.update(self._extract_real_flow(flow_df))
        else:
            # 降级：用量价近似
            features.update(self._extract_volume_proxy(df))

        # 2. 北向资金（市场上下文）
        north_df = self._fetch_north_flow()
        if not north_df.empty:
            features.update(self._extract_north_flow(north_df))
        else:
            features["capital_north_net"] = 0.0
            features["capital_north_score"] = 0.5

        # 3. 龙虎榜（交易异动/机构资金）
        lhb_df = self._fetch_lhb(symbol, as_of_date=as_of_date)
        if not lhb_df.empty:
            features.update(self._extract_lhb_flow(lhb_df))
        else:
            features.update(self._neutral_lhb_features())

        # 4. 融资融券（杠杆资金）
        margin_df = self._fetch_margin_flow(symbol, as_of_date=as_of_date)
        if not margin_df.empty:
            features.update(self._extract_margin_flow(margin_df))
        else:
            features.update(self._neutral_margin_features())

        # 5. 综合资金评分
        features["capital_score"] = self._compute_capital_score(features)

        return features

    def _extract_real_flow(self, flow_df: pd.DataFrame) -> Dict:
        """从真实资金流数据提取特征"""
        features = {}
        features["capital_data_source"] = "akshare"

        latest = flow_df.iloc[-1]

        # 主力净流入（最近一日）
        if "main_inflow" in flow_df.columns:
            main_inflow = float(latest.get("main_inflow", 0))
            features["capital_main_inflow"] = main_inflow
            features["capital_main_inflow_pct"] = float(latest.get("main_inflow_pct", 0))

            # 5日主力净流入累计
            main_5d = flow_df["main_inflow"].tail(5).sum()
            features["capital_main_inflow_5d"] = float(main_5d)

            # 20日主力净流入累计
            main_20d = flow_df["main_inflow"].tail(20).sum()
            features["capital_main_inflow_20d"] = float(main_20d)

            # 资金趋势：5日均值 vs 20日均值
            main_5d_avg = flow_df["main_inflow"].tail(5).mean()
            main_20d_avg = flow_df["main_inflow"].tail(20).mean()
            if main_20d_avg != 0:
                features["capital_trend"] = float(main_5d_avg / abs(main_20d_avg))
            else:
                features["capital_trend"] = 1.0

        # 超大单占比（机构资金方向）
        if "super_large_inflow" in flow_df.columns:
            features["capital_super_large_inflow"] = float(latest.get("super_large_inflow", 0))
            features["capital_super_large_pct"] = float(latest.get("super_large_pct", 0))

        # 大单 vs 小单分歧度（越大说明散户在接盘/机构在出逃）
        if "large_inflow" in flow_df.columns and "small_inflow" in flow_df.columns:
            large_5d = flow_df["large_inflow"].tail(5).sum()
            small_5d = flow_df["small_inflow"].tail(5).sum()
            features["capital_retail_divergence"] = float(
                (small_5d - large_5d) / (abs(large_5d) + abs(small_5d) + 1)
            )

        # 资金活跃度
        if "main_inflow" in flow_df.columns:
            main_abs = flow_df["main_inflow"].tail(5).abs().mean()
            main_abs_20 = flow_df["main_inflow"].tail(20).abs().mean()
            if main_abs_20 > 0:
                features["capital_activity"] = float(main_abs / main_abs_20)
            else:
                features["capital_activity"] = 1.0

        # 连续净流入天数
        if "main_inflow" in flow_df.columns:
            consecutive = 0
            for val in flow_df["main_inflow"].iloc[::-1]:
                if val > 0:
                    consecutive += 1
                else:
                    break
            features["capital_consecutive_inflow"] = consecutive

        # 是否放量
        features["capital_is_surge"] = bool(features.get("capital_activity", 1.0) > 1.5)

        return features

    def _neutral_lhb_features(self) -> Dict:
        return {
            "capital_lhb_on_board": False,
            "capital_lhb_count_30d": 0,
            "capital_lhb_net_buy": 0.0,
            "capital_lhb_buy_amount": 0.0,
            "capital_lhb_sell_amount": 0.0,
            "capital_lhb_turnover_pct": 0.0,
            "capital_lhb_score": 0.5,
        }

    def _extract_lhb_flow(self, lhb_df: pd.DataFrame) -> Dict:
        """提取龙虎榜特征。ETF多数不会上榜，未上榜时保持中性。"""
        features = self._neutral_lhb_features()
        if lhb_df.empty:
            return features

        net_col = self._pick_column(lhb_df, ("龙虎榜净买额", "净买额", "净买入额"))
        buy_col = self._pick_column(lhb_df, ("龙虎榜买入额", "买入额"))
        sell_col = self._pick_column(lhb_df, ("龙虎榜卖出额", "卖出额"))
        turnover_col = self._pick_column(lhb_df, ("成交额占总成交比", "净买额占总成交比", "换手率"))

        features["capital_lhb_on_board"] = True
        features["capital_lhb_count_30d"] = int(len(lhb_df))

        if net_col:
            net_buy = pd.to_numeric(lhb_df[net_col], errors="coerce").fillna(0).sum()
            features["capital_lhb_net_buy"] = float(net_buy)
        if buy_col:
            features["capital_lhb_buy_amount"] = float(pd.to_numeric(lhb_df[buy_col], errors="coerce").fillna(0).sum())
        if sell_col:
            features["capital_lhb_sell_amount"] = float(pd.to_numeric(lhb_df[sell_col], errors="coerce").fillna(0).sum())
        if turnover_col:
            features["capital_lhb_turnover_pct"] = float(pd.to_numeric(lhb_df[turnover_col], errors="coerce").fillna(0).tail(5).mean())

        score = 0.5
        net_buy = features["capital_lhb_net_buy"]
        if net_buy > 0:
            score += 0.2
        elif net_buy < 0:
            score -= 0.2
        score += min(0.15, features["capital_lhb_count_30d"] * 0.03)
        features["capital_lhb_score"] = float(max(0.0, min(1.0, score)))
        return features

    def _neutral_margin_features(self) -> Dict:
        return {
            "capital_margin_source": "none",
            "capital_margin_balance": 0.0,
            "capital_margin_buy": 0.0,
            "capital_margin_repay": 0.0,
            "capital_margin_net_buy": 0.0,
            "capital_margin_change_pct": 0.0,
            "capital_short_balance": 0.0,
            "capital_margin_score": 0.5,
        }

    def _extract_margin_flow(self, margin_df: pd.DataFrame) -> Dict:
        """提取融资融券特征，兼容交易所明细和市场汇总两类字段。"""
        features = self._neutral_margin_features()
        if margin_df.empty:
            return features

        balance_col = self._pick_column(margin_df, ("融资余额", "融资余额(元)", "融资余额（元）"))
        buy_col = self._pick_column(margin_df, ("融资买入额", "融资买入额(元)", "融资买入额（元）"))
        repay_col = self._pick_column(margin_df, ("融资偿还额", "融资偿还额(元)", "融资偿还额（元）"))
        short_col = self._pick_column(margin_df, ("融券余额", "融券余量", "融券余额(元)", "融券余额（元）"))

        latest = margin_df.iloc[-1]
        features["capital_margin_source"] = "akshare_margin"

        if balance_col:
            balance_series = pd.to_numeric(margin_df[balance_col], errors="coerce").dropna()
            if not balance_series.empty:
                features["capital_margin_balance"] = float(balance_series.iloc[-1])
                if len(balance_series) >= 2 and balance_series.iloc[-2] != 0:
                    features["capital_margin_change_pct"] = float(
                        (balance_series.iloc[-1] / balance_series.iloc[-2] - 1) * 100
                    )
        if buy_col:
            features["capital_margin_buy"] = float(pd.to_numeric(pd.Series([latest[buy_col]]), errors="coerce").fillna(0).iloc[0])
        if repay_col:
            features["capital_margin_repay"] = float(pd.to_numeric(pd.Series([latest[repay_col]]), errors="coerce").fillna(0).iloc[0])
        if short_col:
            features["capital_short_balance"] = float(pd.to_numeric(pd.Series([latest[short_col]]), errors="coerce").fillna(0).iloc[0])

        features["capital_margin_net_buy"] = features["capital_margin_buy"] - features["capital_margin_repay"]

        score = 0.5
        change_pct = features["capital_margin_change_pct"]
        net_buy = features["capital_margin_net_buy"]
        if change_pct > 2 or net_buy > 0:
            score += 0.15
        elif change_pct < -2 or net_buy < 0:
            score -= 0.15
        if features["capital_short_balance"] > 0 and features["capital_margin_balance"] > 0:
            short_ratio = features["capital_short_balance"] / features["capital_margin_balance"]
            score -= min(0.15, short_ratio * 0.2)
        features["capital_margin_score"] = float(max(0.0, min(1.0, score)))
        return features

    def _extract_volume_proxy(self, df: pd.DataFrame) -> Dict:
        """降级方案：用量价数据近似资金流向"""
        features = {}
        features["capital_data_source"] = "volume_proxy"

        close = df["close"]
        volume = df["volume"]
        amount = close * volume

        avg_amount_5 = amount.rolling(5).mean().iloc[-1]
        avg_amount_20 = amount.rolling(20).mean().iloc[-1]

        if avg_amount_20 > 0:
            features["capital_activity"] = float(avg_amount_5 / avg_amount_20)
        else:
            features["capital_activity"] = 1.0

        vol_change = volume.pct_change(5).iloc[-1]
        features["volume_change"] = float(vol_change * 100)

        # 近似主力净流入：上涨日放量 = 正流入
        daily_ret = close.pct_change()
        amount_change = amount.pct_change()
        # 上涨放量计为正，下跌放量计为负
        weighted_flow = (daily_ret * amount_change).rolling(5).mean().iloc[-1]
        features["capital_main_inflow_pct"] = float(weighted_flow * 100)

        features["capital_is_surge"] = bool(features["capital_activity"] > 1.5)

        # 近似连续流入
        consecutive = 0
        for i in range(min(10, len(df) - 1)):
            idx = -(i + 1)
            if daily_ret.iloc[idx] > 0 and volume.iloc[idx] > volume.rolling(20).mean().iloc[idx]:
                consecutive += 1
            else:
                break
        features["capital_consecutive_inflow"] = consecutive
        features["capital_trend"] = features["capital_activity"]

        return features

    def _extract_north_flow(self, north_df: pd.DataFrame) -> Dict:
        """提取北向资金特征"""
        features = {}
        try:
            # 筛选北向资金（沪股通+深股通）
            north_rows = north_df[north_df.get("资金方向", north_df.iloc[:, 3]) == "北向"]
            if not north_rows.empty:
                # 成交净买额（亿元）
                net_buy = north_rows.iloc[:, 5].sum()  # 成交净买额列
                features["capital_north_net"] = float(net_buy)

                # 北向资金评分：净买入>0 为利好
                if net_buy > 5:
                    features["capital_north_score"] = 0.8
                elif net_buy > 0:
                    features["capital_north_score"] = 0.6
                elif net_buy > -5:
                    features["capital_north_score"] = 0.4
                else:
                    features["capital_north_score"] = 0.2
            else:
                features["capital_north_net"] = 0.0
                features["capital_north_score"] = 0.5
        except Exception:
            features["capital_north_net"] = 0.0
            features["capital_north_score"] = 0.5

        return features

    def _compute_capital_score(self, features: Dict) -> float:
        """计算综合资金评分 (0-1)"""
        score = 0.5  # 基准

        # 主力净流入占比（权重40%）
        main_pct = features.get("capital_main_inflow_pct", 0)
        if main_pct > 10:
            score += 0.2
        elif main_pct > 0:
            score += 0.1
        elif main_pct < -10:
            score -= 0.2
        elif main_pct < 0:
            score -= 0.1

        # 连续流入天数（权重20%）
        consecutive = features.get("capital_consecutive_inflow", 0)
        score += min(0.15, consecutive * 0.03)

        # 资金活跃度（权重20%）
        activity = features.get("capital_activity", 1.0)
        if activity > 1.5:
            score += 0.1
        elif activity < 0.5:
            score -= 0.1

        # 北向资金（权重20%）
        north_score = features.get("capital_north_score", 0.5)
        score += (north_score - 0.5) * 0.2

        # 龙虎榜资金（权重10%）
        lhb_score = features.get("capital_lhb_score", 0.5)
        score += (lhb_score - 0.5) * 0.1

        # 融资融券（权重10%）
        margin_score = features.get("capital_margin_score", 0.5)
        score += (margin_score - 0.5) * 0.1

        return float(max(0.0, min(1.0, score)))
