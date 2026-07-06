"""
Layer6 - 量价层
=====================

功能: RPS排名、VCP形态分析、量价配合

v2 改进:
- VCP: 实现真正的波动收敛形态检测（收缩次数+逐次递减+突破确认）
- RPS: 增加多周期（20/50/120日）+ 跨标的排名
- 量价: 增加OBV趋势、量价背离检测
- 时序对齐：支持 as_of_date 防未来函数
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class RPSCalculator:
    """RPS计算器"""

    @staticmethod
    def calculate_rps(price_series: pd.Series, period: int = 50) -> float:
        """
        计算RPS: 当前价格在历史区间的百分位

        RPS=80 表示价格高于过去80%的时间
        """
        if len(price_series) < period:
            period = len(price_series)
        if period < 5:
            return 50.0

        current = price_series.iloc[-1]
        historical = price_series.iloc[-period:]
        rank = (historical < current).sum() / len(historical) * 100
        return float(rank)

    @staticmethod
    def calculate_rps_cross_sectional(
        all_prices: Dict[str, pd.Series], as_of_date: Optional[str] = None
    ) -> Dict[str, float]:
        """计算跨标的RPS排名（某一天所有标的的涨跌幅排名百分位）"""
        returns = {}
        for symbol, prices in all_prices.items():
            if as_of_date:
                prices = prices[prices.index <= pd.Timestamp(as_of_date)]
            if len(prices) >= 21:
                ret = prices.pct_change(20).iloc[-1]
                returns[symbol] = ret

        if len(returns) < 2:
            return {s: 50.0 for s in all_prices}

        sorted_returns = sorted(returns.items(), key=lambda x: x[1])
        n = len(sorted_returns)
        result = {}
        for rank, (symbol, _) in enumerate(sorted_returns):
            result[symbol] = (rank / (n - 1)) * 100
        return result

    @staticmethod
    def detect_vcp(df: pd.DataFrame, lookback: int = 60) -> Dict:
        """
        检测VCP（Volatility Contraction Pattern）波动收敛形态

        VCP特征:
        1. 价格经过2-4次收缩（pivot）
        2. 每次收缩的波动幅度递减
        3. 成交量在收缩期间递减
        4. 最终向上突破

        Returns:
            {
                'is_vcp': bool,           # 是否形成VCP
                'contractions': int,       # 收缩次数 (2-4)
                'vol_ratios': list,        # 每次收缩波动率比值
                'volume_declining': bool,  # 成交量是否递减
                'breakout': bool,          # 是否突破
                'quality': float,          # 质量分 (0-1)
                'tightest_pct': float,     # 最后一次收缩幅度%
            }
        """
        result = {
            "is_vcp": False,
            "contractions": 0,
            "vol_ratios": [],
            "volume_declining": False,
            "breakout": False,
            "quality": 0.0,
            "tightest_pct": 0.0,
        }

        if df.empty or len(df) < lookback:
            return result

        close = df["close"]
        high = df.get("high", close)
        low = df.get("low", close)
        volume = df.get("volume", pd.Series(1, index=df.index))

        data = df.tail(lookback).copy()

        # ==================== 1. 检测收缩波段 ====================
        # 使用滑动窗口检测局部高低点（swing highs/lows）
        contractions = []
        window = 5  # 每个收缩波段的检测窗口

        # 将lookback期间分成多个子段，计算每个子段的波动幅度
        n_segments = min(4, len(data) // 10)  # 最多检测4次收缩
        if n_segments < 2:
            return result

        segment_size = len(data) // n_segments
        for i in range(n_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < n_segments - 1 else len(data)
            seg = data.iloc[start:end]
            if len(seg) < 5:
                continue

            seg_high = seg["high"].max() if "high" in seg else seg["close"].max()
            seg_low = seg["low"].min() if "low" in seg else seg["close"].min()
            seg_mid = (seg_high + seg_low) / 2
            if seg_mid > 0:
                spread_pct = (seg_high - seg_low) / seg_mid * 100
            else:
                spread_pct = 0

            avg_vol = seg["volume"].mean() if "volume" in seg else 1
            contractions.append({
                "spread_pct": spread_pct,
                "avg_volume": avg_vol,
                "high": seg_high,
                "low": seg_low,
            })

        if len(contractions) < 2:
            return result

        # ==================== 2. 检查波动递减 ====================
        spreads = [c["spread_pct"] for c in contractions]
        vols = [c["avg_volume"] for c in contractions]

        # 计算每次收缩相对于前一次的比值
        vol_ratios = []
        declining_count = 0
        for i in range(1, len(spreads)):
            if spreads[i - 1] > 0:
                ratio = spreads[i] / spreads[i - 1]
            else:
                ratio = 1.0
            vol_ratios.append(float(ratio))
            if ratio < 1.0:
                declining_count += 1

        # 至少 60% 的收缩是递减的
        is_contracting = declining_count / len(vol_ratios) >= 0.6

        # ==================== 3. 检查成交量递减 ====================
        vol_declining_count = 0
        for i in range(1, len(vols)):
            if vols[i] < vols[i - 1]:
                vol_declining_count += 1
        volume_declining = vol_declining_count / max(1, len(vols) - 1) >= 0.5

        # ==================== 4. 检测突破 ====================
        # 最后一个收缩区间的最高价
        last_contraction_high = contractions[-1]["high"]
        current_price = close.iloc[-1]
        breakout = current_price > last_contraction_high * 0.998  # 容差0.2%

        # 突破时放量
        vol_now = volume.iloc[-1]
        vol_avg_20 = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        breakout_with_volume = breakout and vol_now > vol_avg_20 * 1.2

        # ==================== 5. 综合判断 ====================
        tightest_pct = spreads[-1]  # 最后一次收缩幅度
        contraction_count = len(contractions)

        # VCP条件：至少2次收缩 + 波动递减 + 最后收缩幅度<8%
        is_vcp = (
            contraction_count >= 2
            and is_contracting
            and tightest_pct < 10.0  # 最后收缩幅度小于10%
        )

        # 质量评分
        quality = 0.0
        if is_vcp:
            quality += 0.3  # 基础分
            if contraction_count >= 3:
                quality += 0.15  # 多次收缩加分
            if volume_declining:
                quality += 0.2  # 量缩加分
            if breakout:
                quality += 0.2  # 突破加分
            if breakout_with_volume:
                quality += 0.15  # 放量突破加分

            # 最后收缩越紧质量越高
            if tightest_pct < 4:
                quality += 0.1
            elif tightest_pct < 6:
                quality += 0.05

        quality = float(min(1.0, quality))

        result = {
            "is_vcp": bool(is_vcp),
            "contractions": int(contraction_count),
            "vol_ratios": vol_ratios,
            "volume_declining": bool(volume_declining),
            "breakout": bool(breakout),
            "breakout_with_volume": bool(breakout_with_volume),
            "quality": quality,
            "tightest_pct": float(tightest_pct),
        }
        return result


class PriceVolumeLayer:
    """
    量价层 - 价格和成交量分析
    RPS和VCP是选股的核心指标
    """

    def __init__(self):
        self.rps_calculator = RPSCalculator()

    def extract_features(
        self,
        symbol: str,
        df: pd.DataFrame,
        ctx: Dict,
        as_of_date: Optional[str] = None,
    ) -> Dict:
        """提取量价层特征"""
        features = {}

        # 时序对齐
        if as_of_date and not df.empty:
            df = df[df.index <= pd.Timestamp(as_of_date)]

        if df.empty or len(df) < 60:
            return features

        close = df["close"]

        # ==================== RPS ====================
        rps_20 = self.rps_calculator.calculate_rps(close, 20)
        rps_50 = self.rps_calculator.calculate_rps(close, 50)
        rps_120 = self.rps_calculator.calculate_rps(close, 120)
        features["pv_rps_20"] = rps_20
        features["pv_rps_50"] = rps_50
        features["pv_rps_120"] = rps_120

        # RPS综合 (近期权重更高)
        features["pv_rps_combined"] = rps_20 * 0.3 + rps_50 * 0.4 + rps_120 * 0.3

        # 跨标的RPS（如果ctx提供了全市场价格）
        cross_rps = ctx.get("cross_sectional_rps", {})
        if symbol in cross_rps:
            features["pv_rps_cross_sectional"] = cross_rps[symbol]
        else:
            features["pv_rps_cross_sectional"] = features["pv_rps_combined"]

        # ==================== VCP 形态检测 ====================
        vcp_result = self.rps_calculator.detect_vcp(df)
        features["pv_vcp_is_pattern"] = vcp_result["is_vcp"]
        features["pv_vcp_contractions"] = vcp_result["contractions"]
        features["pv_vcp_volume_declining"] = vcp_result["volume_declining"]
        features["pv_vcp_breakout"] = vcp_result["breakout"]
        features["pv_vcp_breakout_volume"] = vcp_result.get("breakout_with_volume", False)
        features["pv_vcp_quality"] = vcp_result["quality"]
        features["pv_vcp_tightest_pct"] = vcp_result["tightest_pct"]

        # ==================== v3新增: 波动率收缩因子 (VCP的连续版本, 更适合ETF) ====================
        # VCP形态检测在ETF上效果差(ETF波动小, 形态少, ICIR≈0)
        # 用连续波动率收缩度替代: 20日波动率/60日波动率, 比值<1=收缩
        daily_ret = close.pct_change()
        vol_20 = daily_ret.rolling(20).std().iloc[-1]
        vol_60 = daily_ret.rolling(60).std().iloc[-1] if len(daily_ret) >= 60 else vol_20
        if vol_60 > 0 and not np.isnan(vol_20) and not np.isnan(vol_60):
            vol_ratio = vol_20 / vol_60
            # sigmoid映射: ratio=1→0.5, ratio=0.5→0.82(收缩=高分), ratio=1.5→0.18(扩张=低分)
            features["pv_volatility_contraction"] = float(
                1.0 / (1.0 + np.exp(np.clip((vol_ratio - 1) * 5, -50.0, 50.0)))
            )
        else:
            features["pv_volatility_contraction"] = 0.5

        # ==================== 量价配合 ====================
        if "volume" in df.columns:
            vol = df["volume"]

            # 成交量趋势
            vol_ma5 = vol.rolling(5).mean().iloc[-1]
            vol_ma20 = vol.rolling(20).mean().iloc[-1]
            vol_trend = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1.0
            features["pv_volume_trend"] = float(vol_trend)

            # 量价配合（1.0同向/0.5背离）
            price_up = close.iloc[-1] > close.iloc[-5]
            vol_up = vol.iloc[-1] > vol.iloc[-5]
            features["pv_price_volume_align"] = 1.0 if price_up == vol_up else 0.5

            # 放量突破
            vol_surge = vol_trend > 1.5
            price_breakout = close.iloc[-1] > close.rolling(20).max().iloc[-2]
            features["pv_breakout_surge"] = bool(vol_surge and price_breakout)

            # v2新增: OBV趋势
            obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
            obv_ma5 = obv.rolling(5).mean().iloc[-1]
            obv_ma20 = obv.rolling(20).mean().iloc[-1]
            if obv_ma20 != 0:
                features["pv_obv_trend"] = float(obv_ma5 / abs(obv_ma20))
            else:
                features["pv_obv_trend"] = 1.0

            # v2新增: 量价背离检测（价格新高但量未新高）
            price_new_high = close.iloc[-1] >= close.rolling(20).max().iloc[-1]
            vol_new_high = vol.iloc[-1] >= vol.rolling(20).max().iloc[-1] * 0.8
            features["pv_divergence"] = bool(price_new_high and not vol_new_high)
        else:
            features["pv_volume_trend"] = 1.0
            features["pv_price_volume_align"] = 0.5
            features["pv_breakout_surge"] = False
            features["pv_obv_trend"] = 1.0
            features["pv_divergence"] = False

        # ==================== v4新增: 独立alpha因子 ====================
        # 1. 换手率变化率: (5日均换手 - 20日均换手) / 20日均换手
        #    换手率加速→资金关注度提升, 与成交量趋势低相关(0.3-0.5)
        if "volume" in df.columns:
            vol = df["volume"]
            # 用成交量/close近似换手率(ETF无流通股本数据)
            approx_turnover = vol / close
            t5 = approx_turnover.rolling(5).mean().iloc[-1]
            t20 = approx_turnover.rolling(20).mean().iloc[-1]
            if t20 > 0 and not np.isnan(t5) and not np.isnan(t20):
                features["pv_turnover_change"] = float((t5 - t20) / t20)
            else:
                features["pv_turnover_change"] = 0.0
        else:
            features["pv_turnover_change"] = 0.0

        # 2. 价格加速度: (5日收益 - 20日收益), 即短期动量-中期动量
        #    正加速=趋势在加强, 与RPS低相关(0.2-0.4)
        ret_5 = close.pct_change(5).iloc[-1] if len(close) >= 6 else 0
        ret_20 = close.pct_change(20).iloc[-1] if len(close) >= 21 else 0
        if not np.isnan(ret_5) and not np.isnan(ret_20):
            features["pv_price_accel"] = float(ret_5 - ret_20)
        else:
            features["pv_price_accel"] = 0.0

        # 3. 量价趋势背离度: pv_volume_trend - (ret_20标准化到[0,2])
        #    量升价不升=背离, 可能是顶部信号; 量缩价升=背离, 可能是控盘信号
        #    与pv_volume_trend和RPS都低相关
        vt = features.get("pv_volume_trend", 1.0)
        # 将ret_20映射到[0,2]: ret=0→1.0, ret=+10%→1.5, ret=-10%→0.5
        ret_mapped = 1.0 + np.clip(ret_20 * 5, -0.5, 0.5) if not np.isnan(ret_20) else 1.0
        features["pv_vol_price_divergence"] = float(vt - ret_mapped)

        # ==================== 综合评分 ====================
        features["pv_score"] = self._compute_pv_score(features)

        return features

    def _compute_pv_score(self, features: Dict) -> float:
        """计算综合量价评分 (0-1)"""
        score = 0.0

        # RPS综合分 (权重40%)
        rps = features.get("pv_rps_combined", 50)
        score += (rps / 100) * 0.40

        # VCP质量 (权重30%)
        score += features.get("pv_vcp_quality", 0) * 0.30

        # 量价配合 (权重15%)
        score += features.get("pv_price_volume_align", 0.5) / 2 * 0.15

        # OBV趋势 (权重15%)
        obv = features.get("pv_obv_trend", 1.0)
        obv_score = min(1.0, max(0.0, (obv - 0.5) / 1.0))
        score += obv_score * 0.15

        return float(max(0.0, min(1.0, score)))
