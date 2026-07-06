# -*- coding: utf-8 -*-
"""
etf_screener_backtest_v3.py — ETF 智能筛选+回测系统 v3.0

v3 相对 v2.1 的关键变化:
  1. 替换 v2.1 的 5 维打分（趋势+动量+RSI+量+稳定）为 layers/ 框架的 5 层
     - L1 宏观: 大盘状态 → 整体仓位乘数 (BULL=1.0, NEUTRAL=0.5, BEAR=0.0)
     - L2 制度: rules_pass → 过滤 ST/退市/停牌
     - L3 板块: 20 日动量作为板块热点代理
     - L6 量价: pv_score（RPS 50/120 + VCP）
     - L7 技术: tech_pattern_score（MA + MACD + 布林 + RSI）
  2. 保留 v2.1 的所有止盈止损改进（移动止盈/时间止损/保本止损）
  3. 保留 v2.1 的 look-ahead bias 修复（as_of_date）
  4. 加宏观仓位乘数：BEAR 时整体减仓

作者: Matrix Agent
"""

import os
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# 把项目根加到 sys.path（这样 from layers.xxx import 才能找到）
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# 引入 layers/ 里的 8 层框架
from layers.layer1_macro import MacroLayer
from layers.layer2_rules import RulesLayer
from layers.layer3_sector import SectorLayer
from layers.layer4_capital import CapitalLayer
from layers.layer5_sentiment import SentimentLayer
from layers.layer6_price_vol import PriceVolumeLayer
from layers.layer7_technical import TechnicalLayer


# ============================================================
# v4 重做的 L1 宏观层：复合动量/斜率/波动率信号
# ============================================================

class ImprovedMacroLayer:
    """
    v4 重写的宏观层 — 比原 layer1_macro.py 更敏感

    原版只用 MA20/MA60 三均线：2020-2026 期间 97% 判 NEUTRAL
    新版用 4 个子信号合成连续得分 [0, 1]：
      1. 20 日动量  (mom_20)        权重 0.30
      2. MA20 斜率  (slope_20)      权重 0.25
      3. 20 日波动率(低波动上行加分)  权重 0.20
      4. 连续阳线  (consec_up)     权重 0.25

    分档:
      score > 0.65 → BULL   (仓位乘数 1.0)
      0.35 < score < 0.65 → NEUTRAL (0.5)
      score < 0.35 → BEAR   (0.0)
    """

    def extract_features(self, symbol: str, df: pd.DataFrame, ctx: Dict) -> Dict:
        features = {}
        if df.empty or len(df) < 60:
            features['macro_regime'] = 'NEUTRAL'
            features['macro_regime_score'] = 0.5
            return features

        close = df['close']
        cur = close.iloc[-1]

        # 1) 20 日动量
        if len(close) >= 21:
            mom_20 = (cur - close.iloc[-21]) / close.iloc[-21]
        else:
            mom_20 = 0
        # 映射 [-10%, +10%] → [0, 1]
        mom_score = max(0.0, min(1.0, (mom_20 + 0.10) / 0.20))

        # 2) MA20 斜率（最近 20 天变化率）
        ma20 = close.rolling(20).mean()
        if len(ma20) >= 21 and not np.isnan(ma20.iloc[-1]) and not np.isnan(ma20.iloc[-21]):
            slope_20 = (ma20.iloc[-1] - ma20.iloc[-21]) / ma20.iloc[-21]
        else:
            slope_20 = 0
        slope_score = max(0.0, min(1.0, (slope_20 + 0.05) / 0.10))

        # 3) 20 日波动率（低波动 = 上行稳健，加分）
        vol_20 = close.pct_change().rolling(20).std().iloc[-1]
        if np.isnan(vol_20):
            vol_20 = 0.02
        # 波动率 < 1% → 1.0; > 3% → 0.0
        vol_score = max(0.0, min(1.0, 1.0 - (vol_20 - 0.01) / 0.02))

        # 4) 连续阳线天数（最近 10 天中阳线占比）
        recent_10 = close.iloc[-10:]
        daily_ret = recent_10.pct_change().dropna()
        up_ratio = (daily_ret > 0).sum() / max(len(daily_ret), 1)
        consec_score = up_ratio  # 0-1

        # 加权汇总
        total_score = (
            mom_score * 0.30 +
            slope_score * 0.25 +
            vol_score * 0.20 +
            consec_score * 0.25
        )

        # 分档
        if total_score > 0.65:
            regime = 'BULL'
        elif total_score < 0.35:
            regime = 'BEAR'
        else:
            regime = 'NEUTRAL'

        features['macro_regime'] = regime
        features['macro_regime_score'] = float(total_score)
        features['macro_mom_20'] = float(mom_20)
        features['macro_slope_20'] = float(slope_20)
        features['macro_vol_20'] = float(vol_20)
        features['macro_up_ratio'] = float(up_ratio)
        features['macro_components'] = {
            'mom': mom_score, 'slope': slope_score,
            'vol': vol_score, 'consec': consec_score
        }
        return features

    def get_market_regime(self, market_df: pd.DataFrame) -> str:
        """便捷方法：用 extract_features 算出 regime"""
        f = self.extract_features('market', market_df, {})
        return f.get('macro_regime', 'NEUTRAL')

    def get_market_score(self, market_df: pd.DataFrame) -> float:
        f = self.extract_features('market', market_df, {})
        return f.get('macro_regime_score', 0.5)


# ============================================================
# 1. v3 评分：基于 5 层框架
# ============================================================

@dataclass
class ETFScoreV4:
    """v4 ETF 评分数据（6 层框架）"""
    code: str
    total_score: float = 0.0
    # 各层得分
    rules_pass: bool = True
    exclusion_reason: str = ""
    layer1_macro: float = 0.5        # 宏观（用 v4 重写版）
    layer3_60d_mom: float = 0.5      # 板块 60 日动量
    layer3_phase: float = 0.5        # 板块轮动阶段 (启动/高潮加分)
    layer4_capital: float = 0.5      # 资金
    layer5_sentiment: float = 0.5    # 情绪
    layer6_pv: float = 0.5           # 量价
    layer7_tech: float = 0.5         # 技术
    # 元信息
    latest_price: float = 0.0
    latest_date: str = ""
    features: Dict = field(default_factory=dict)


class LayeredETFScreener:
    """v4 评分器：6 层加权汇总（L1 改进 + L3 重做 + L4/L5 新增 + L6/L7）"""
    WEIGHTS = {
        'L1_macro': 0.15,
        'L3_momentum': 0.10,
        'L3_phase': 0.05,
        'L4_capital': 0.15,
        'L5_sentiment': 0.10,
        'L6_pv': 0.20,
        'L7_tech': 0.25,
    }

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.l1 = ImprovedMacroLayer()  # v4 重写版
        self.l2 = RulesLayer()
        self.l3 = SectorLayer()
        self.l4 = CapitalLayer()        # v4 新增
        self.l5 = SentimentLayer()      # v4 新增
        self.l6 = PriceVolumeLayer()
        self.l7 = TechnicalLayer()

    def scan_all_etfs(self, lookback_days: int = 200,
                      as_of_date: Optional[str] = None) -> Tuple[List[ETFScoreV4], Dict]:
        etf_files = [f for f in self.data_dir.glob("*.csv")
                     if f.stem.startswith("51") or f.stem.startswith("15")
                     or f.stem.startswith("16") or f.stem.startswith("50")]
        print(f"  发现 {len(etf_files)} 个 ETF 数据文件" +
              (f"（as_of_date={as_of_date}）" if as_of_date else ""))

        # 1) 大盘上下文（用 510300 当代理）
        market_df = self._load_market_df(as_of_date)
        if market_df is not None and len(market_df) >= 60:
            f1 = self.l1.extract_features('market', market_df, {})
            market_regime = f1.get('macro_regime', 'NEUTRAL')
            market_score = f1.get('macro_regime_score', 0.5)
        else:
            market_regime, market_score = 'NEUTRAL', 0.5
        ctx = {
            'market_prices_df': market_df,
            'market_regime': market_regime,
            'market_score': market_score,
        }
        print(f"  宏观状态: {market_regime} (score={market_score:.3f})  ← v4 改进版")

        # 2) 评分每个 ETF
        scores = []
        for i, file in enumerate(etf_files):
            if i % 100 == 0:
                print(f"  扫描进度: {i}/{len(etf_files)} ...")
            try:
                s = self._score_one(file.stem, lookback_days, as_of_date, ctx)
                if s is not None:
                    scores.append(s)
            except Exception:
                continue

        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores, ctx

    def _load_market_df(self, as_of_date: Optional[str]) -> Optional[pd.DataFrame]:
        p = self.data_dir / '510300.csv'
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p, parse_dates=['date']).sort_values('date')
            if as_of_date is not None:
                df = df[df['date'] <= pd.Timestamp(as_of_date)]
            if len(df) < 60:
                return None
            return df
        except Exception:
            return None

    def _score_one(self, code: str, lookback_days: int,
                   as_of_date: Optional[str], ctx: Dict) -> Optional[ETFScoreV4]:
        file_path = self.data_dir / f"{code}.csv"
        try:
            df = pd.read_csv(file_path, parse_dates=['date']).sort_values('date')
            if len(df) < 100:
                return None

            if as_of_date is not None:
                as_of = pd.Timestamp(as_of_date)
                if df['date'].min() > as_of:
                    return None
                df = df[df['date'] <= as_of]
                if len(df) < 100:
                    return None
                cutoff = as_of - pd.Timedelta(days=lookback_days)
                df = df[df['date'] >= cutoff]
            else:
                cutoff = df['date'].max() - pd.Timedelta(days=lookback_days)
                df = df[df['date'] >= cutoff]

            if len(df) < 50:
                return None

            score = ETFScoreV4(code=code)
            score.latest_price = float(df['close'].iloc[-1])
            score.latest_date = str(df['date'].iloc[-1])[:10]

            # ---- L2 制度：硬过滤 ----
            should_exclude, reason = self.l2.should_exclude(code, df)
            if should_exclude:
                return None

            # ---- L1 宏观 ----
            score.layer1_macro = ctx.get('market_score', 0.5)

            # ---- L3 板块（v4 重做）----
            close = df['close']
            # 60 日动量（更稳健）
            if len(df) >= 61:
                ret_60d = float(close.pct_change(60).iloc[-1])
                score.layer3_60d_mom = max(0.0, min(1.0, (ret_60d + 0.20) / 0.40))
            else:
                score.layer3_60d_mom = 0.5
            # 板块轮动阶段（启动/高潮加分）
            if len(df) >= 21:
                ret_5d = float(close.pct_change(5).iloc[-1])
                ret_20d = float(close.pct_change(20).iloc[-1])
                if ret_5d > 0.03 and ret_20d > 0.10:
                    score.layer3_phase = 1.0   # 高潮
                elif ret_5d > 0.01 and ret_20d > 0.05:
                    score.layer3_phase = 0.8   # 启动
                elif ret_5d < -0.03:
                    score.layer3_phase = 0.2   # 退潮
                else:
                    score.layer3_phase = 0.5   # 中性
            else:
                score.layer3_phase = 0.5

            # ---- L4 资金（v4 新增）----
            f4 = self.l4.extract_features(code, df, ctx)
            # 资金层没有 _score 字段，自己汇总
            capital_score = (
                (1.0 if f4.get('is_volume_surge', False) else 0.5) * 0.4 +
                (f4.get('volume_ratio', 1.0) - 1.0) * 0.5 + 0.5
            )
            score.layer4_capital = max(0.0, min(1.0, capital_score))
            score.features.update({f'cap_{k}': v for k, v in f4.items()})

            # ---- L5 情绪（v4 新增）----
            f5 = self.l5.extract_features(code, df, ctx)
            # ETF 涨跌幅有限制（10%/20%/30%），取近 N 日涨幅分布得分
            sentiment_raw = float(f5.get('sentiment_score', 0.5))
            score.layer5_sentiment = max(0.0, min(1.0, sentiment_raw))
            score.features.update({f'sent_{k}': v for k, v in f5.items()})

            # ---- L6 量价 ----
            f6 = self.l6.extract_features(code, df, ctx)
            score.layer6_pv = float(f6.get('pv_score', 0.5))
            score.features.update({f'pv_{k}': v for k, v in f6.items()})

            # ---- L7 技术 ----
            f7 = self.l7.extract_features(code, df, ctx)
            score.layer7_tech = float(f7.get('tech_pattern_score', 0.5))
            score.features.update({f'tech_{k}': v for k, v in f7.items()})

            # ---- 加权汇总 ----
            score.total_score = (
                self.WEIGHTS['L1_macro'] * score.layer1_macro +
                self.WEIGHTS['L3_momentum'] * score.layer3_60d_mom +
                self.WEIGHTS['L3_phase'] * score.layer3_phase +
                self.WEIGHTS['L4_capital'] * score.layer4_capital +
                self.WEIGHTS['L5_sentiment'] * score.layer5_sentiment +
                self.WEIGHTS['L6_pv'] * score.layer6_pv +
                self.WEIGHTS['L7_tech'] * score.layer7_tech
            )
            return score
        except Exception:
            return None

    def get_top_etfs(self, min_score: float = 0.3, top_n: int = 10,
                     as_of_date: Optional[str] = None) -> Tuple[List[ETFScoreV4], Dict]:
        all_scores, ctx = self.scan_all_etfs(as_of_date=as_of_date)
        filtered = [s for s in all_scores if s.total_score >= min_score]
        return filtered[:top_n], ctx


# ============================================================
# 2. v3 回测引擎：复用 v2.1 框架 + 宏观仓位乘数
# ============================================================

class ScreenerBacktestV4:
    """v4 回测引擎

    关键差异 (相对 v3):
      - 仓位乘数用 L1 连续值 [0, 1]，不是 BULL/NEUTRAL/BEAR 三态
      - 0.65+ → 1.0 倍, 0.35-0.65 → 0.5 倍, < 0.35 → 0.0 倍
    """

    def _score_to_multiplier(self, score: float) -> float:
        if score > 0.65:
            return 1.0
        elif score < 0.35:
            return 0.0
        else:
            return 0.5

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital

        # ---- 基础参数（与 v2.1 一致）----
        self.MAX_POSITION_PCT = 0.30
        self.MAX_HOLDINGS = 5
        self.REBALANCE_DAYS = 15

        # ---- 风控参数（v2.1 全部继承）----
        self.STOP_LOSS_PCT = 0.10
        self.TAKE_PROFIT_FLOOR_PCT = 0.30
        self.BREAKEVEN_TRIGGER_PCT = 0.10
        self.TRAILING_TRIGGER_PCT = 0.15
        self.TRAILING_PCT = 0.08
        self.MAX_HOLD_DAYS = 90
        self.MAX_HOLD_DAYS_HARD = 120
        self.STAGNANT_THRESHOLD = 0.05

    def run_backtest(self, selected_etfs, data_dir,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     market_regimes: Optional[Dict[str, str]] = None) -> dict:
        """
        market_regimes: {date_str: 'BULL'/'NEUTRAL'/'BEAR'} — 每个调仓日的大盘状态
                        如果没传，会用 510300 实时计算
        """
        self.cash = self.initial_capital
        self.positions: Dict[str, dict] = {}
        self.trades: List[dict] = []
        self.equity_curve: List[dict] = []
        self.partial_sells: List[dict] = []

        if start_date and end_date:
            period_start, period_end = start_date, end_date
        else:
            period_start, period_end = self._get_common_period(selected_etfs, data_dir)

        print(f"\n  回测时间: {period_start} 至 {period_end}")

        etf_data = {}
        for etf in selected_etfs:
            df = self._load_etf_data(etf.code, data_dir)
            if df is not None:
                df = df[(df['date'] >= pd.Timestamp(period_start)) &
                        (df['date'] <= pd.Timestamp(period_end))]
                if len(df) > 0:
                    etf_data[etf.code] = df

        if not etf_data:
            print("  错误: 没有可用的 ETF 数据")
            return {}

        # 如果没传 market_regimes，自己用 510300 实时算
        if market_regimes is None:
            market_regimes = self._compute_market_regimes(data_dir, period_start, period_end)

        dates = sorted(set.union(*[set(df['date']) for df in etf_data.values()]))
        last_rebalance = None
        score_sum = 0.0
        score_n = 0
        regime_distribution = {'BULL': 0, 'NEUTRAL': 0, 'BEAR': 0}

        for date in dates:
            self._update_positions(date, etf_data)

            # ---- 取当日宏观状态 ----
            date_str = str(date)[:10]
            market_info = market_regimes.get(date_str, {'regime': 'NEUTRAL', 'score': 0.5})
            if isinstance(market_info, str):
                # 兼容旧版（仅 regime）
                regime = market_info
                macro_score = 0.5
            else:
                regime = market_info.get('regime', 'NEUTRAL')
                macro_score = market_info.get('score', 0.5)
            regime_distribution[regime] += 1
            score_sum += macro_score
            score_n += 1
            multiplier = self._score_to_multiplier(macro_score)
            self.current_regime = regime
            self.current_multiplier = multiplier
            self.current_macro_score = macro_score

            total_value = self._calc_total_value(etf_data, date)
            self.equity_curve.append({
                'date': date_str, 'value': total_value,
                'pnl_pct': (total_value - self.initial_capital) / self.initial_capital * 100,
                'regime': regime,
                'macro_score': macro_score,
                'multiplier': multiplier,
            })

            days_since = 0 if last_rebalance is None else (date - last_rebalance).days
            if days_since >= self.REBALANCE_DAYS or len(self.positions) == 0:
                signals = self._collect_signals(etf_data, date, multiplier)
                if signals:
                    self._rebalance(signals, etf_data, date, multiplier)
                    last_rebalance = date

            self._check_exit_rules(date, etf_data)

        result = self._calculate_stats(period_start, period_end)
        result['regime_distribution'] = regime_distribution
        result['avg_macro_score'] = score_sum / max(score_n, 1)
        return result

    def _compute_market_regimes(self, data_dir, start, end) -> Dict[str, Dict]:
        """用 510300 实时计算每个调仓日的宏观状态（v4 改进版）"""
        p = Path(data_dir) / '510300.csv'
        if not p.exists():
            return {}
        df = pd.read_csv(p, parse_dates=['date']).sort_values('date')
        df = df[(df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))]
        if len(df) < 60:
            return {}

        l1 = ImprovedMacroLayer()
        regimes = {}
        for i, (idx, row) in enumerate(df.iterrows()):
            if i % self.REBALANCE_DAYS == 0:
                df_before = df[df['date'] <= row['date']]
                if len(df_before) < 60:
                    continue
                f = l1.extract_features('market', df_before, {})
                regimes[str(row['date'])[:10]] = {
                    'regime': f.get('macro_regime', 'NEUTRAL'),
                    'score': f.get('macro_regime_score', 0.5)
                }
        return regimes

    def _load_etf_data(self, code, data_dir):
        try:
            return pd.read_csv(Path(data_dir) / f"{code}.csv", parse_dates=['date']).sort_values('date')
        except Exception:
            return None

    def _get_common_period(self, etfs, data_dir):
        all_dates = []
        for etf in etfs:
            df = self._load_etf_data(etf.code, data_dir)
            if df is not None:
                all_dates.append((df['date'].min(), df['date'].max()))
        if not all_dates:
            return "2024-01-01", "2024-12-31"
        start = max(d[0] for d in all_dates)
        end = min(d[1] for d in all_dates)
        if start > end:
            end = max(d[1] for d in all_dates)
            start = end - pd.Timedelta(days=365)
        return str(start)[:10], str(end)[:10]

    def _update_positions(self, date, etf_data):
        for code, pos in list(self.positions.items()):
            if code not in etf_data:
                continue
            df_before = etf_data[code][etf_data[code]['date'] <= date]
            if len(df_before) == 0:
                continue
            current_price = df_before.iloc[-1]['close']
            pos['current_price'] = current_price
            pos['profit_pct'] = (current_price - pos['avg_cost']) / pos['avg_cost']
            if current_price > pos.get('highest_price', pos['avg_cost']):
                pos['highest_price'] = current_price
                pos['highest_date'] = str(date)[:10]
            buy_date = pd.Timestamp(pos['buy_date'])
            pos['days_held'] = (date - buy_date).days

    def _calc_total_value(self, etf_data, date):
        mv = 0.0
        for code, pos in self.positions.items():
            if code not in etf_data:
                continue
            df_before = etf_data[code][etf_data[code]['date'] <= date]
            if len(df_before) == 0:
                continue
            mv += pos['quantity'] * df_before.iloc[-1]['close']
        return self.cash + mv

    def _collect_signals(self, etf_data, date, multiplier):
        signals = []
        for code, df in etf_data.items():
            df_before = df[df['date'] <= date]
            if len(df_before) >= 50:
                signals.append(self._generate_signal(df_before, code))
        return signals

    def _generate_signal(self, df, code):
        if len(df) < 50:
            return {'code': code, 'strength': 0, 'action': 'HOLD'}
        close = df['close'].values
        ma20 = df['close'].rolling(20).mean().values
        ma50 = df['close'].rolling(50).mean().values
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = (100 - (100 / (1 + rs))).values
        strength = 0
        if close[-1] > ma50[-1]: strength += 30
        if ma20[-1] > ma50[-1]: strength += 20
        if not np.isnan(ma50[-1]) and not np.isnan(ma50[-20]) and ma50[-1] > ma50[-20]:
            strength += 15
        if 40 <= rsi[-1] <= 60: strength += 25
        elif rsi[-1] < 40: strength += 15
        mom = (close[-1] - close[-20]) / close[-20] * 100 if len(close) > 20 else 0
        if mom > 5: strength += min(15, mom)
        return {
            'code': code, 'strength': strength, 'rsi': rsi[-1],
            'price': close[-1], 'action': 'BUY' if strength >= 40 else 'HOLD'
        }

    def _rebalance(self, signals, etf_data, date, multiplier):
        """调仓: 仓位上限 = MAX_POSITION_PCT * multiplier"""
        if multiplier == 0:
            return  # BEAR 状态不开新仓

        trade_id = len(self.trades) + 1
        buy_signals = [s for s in signals if s['action'] == 'BUY'][:self.MAX_HOLDINGS]
        for signal in buy_signals:
            code = signal['code']
            price = signal['price']
            if code in self.positions:
                continue
            # 单只最大仓位 = 30% * 宏观乘数
            max_value = self.cash * self.MAX_POSITION_PCT * multiplier
            quantity = int(max_value / price / 100) * 100
            if quantity > 0:
                cost = quantity * price
                if cost <= self.cash:
                    self.positions[code] = {
                        'code': code, 'quantity': quantity, 'avg_cost': price,
                        'buy_date': str(date)[:10],
                        'highest_price': price, 'highest_date': str(date)[:10],
                        'days_held': 0, 'profit_pct': 0.0,
                        'current_price': price, 'time_stop_warned': False,
                    }
                    self.cash -= cost
                    self.trades.append({
                        'date': str(date)[:10], 'trade_id': trade_id,
                        'code': code, 'action': 'BUY',
                        'quantity': quantity, 'price': price, 'amount': cost
                    })
                    trade_id += 1

    def _check_exit_rules(self, date, etf_data):
        actions = {}
        for code, pos in list(self.positions.items()):
            if code not in etf_data:
                continue
            df_before = etf_data[code][etf_data[code]['date'] <= date]
            if len(df_before) == 0:
                continue
            current_price = df_before.iloc[-1]['close']
            profit = pos['profit_pct']
            drawdown_from_peak = (current_price - pos.get('highest_price', pos['avg_cost'])) / pos['highest_price'] if pos.get('highest_price', 0) > 0 else 0

            if profit <= -self.STOP_LOSS_PCT:
                actions[code] = ('FULL', 'HARD_STOP_LOSS')
                continue
            peak_profit = (pos.get('highest_price', pos['avg_cost']) - pos['avg_cost']) / pos['avg_cost']
            if peak_profit >= self.TRAILING_TRIGGER_PCT and drawdown_from_peak <= -self.TRAILING_PCT:
                actions[code] = ('FULL', f'TRAILING_STOP(peak={peak_profit*100:.1f}%,dd={drawdown_from_peak*100:.1f}%)')
                continue
            if peak_profit >= self.BREAKEVEN_TRIGGER_PCT and profit <= 0:
                actions[code] = ('FULL', f'BREAKEVEN_STOP(peak={peak_profit*100:.1f}%)')
                continue
            if profit >= self.TAKE_PROFIT_FLOOR_PCT:
                actions[code] = ('FULL', f'HARD_TAKE_PROFIT({profit*100:.1f}%)')
                continue
            if pos['days_held'] >= self.MAX_HOLD_DAYS_HARD:
                actions[code] = ('FULL', f'TIME_STOP_HARD({pos["days_held"]}d,pnl={profit*100:.1f}%)')
                continue
            elif pos['days_held'] >= self.MAX_HOLD_DAYS and profit < self.STAGNANT_THRESHOLD:
                actions[code] = ('HALF', f'TIME_STOP_HALF({pos["days_held"]}d,pnl={profit*100:.1f}%)')
                continue

        for code, (mode, reason) in actions.items():
            self._execute_exit(code, etf_data[code], date, mode, reason)

    def _execute_exit(self, code, df, date, mode, reason):
        if code not in self.positions:
            return
        pos = self.positions[code]
        df_before = df[df['date'] <= date]
        if len(df_before) == 0:
            return
        current_price = df_before.iloc[-1]['close']
        if mode == 'HALF':
            sell_qty = pos['quantity'] // 2
            if sell_qty < 100:
                return
            pos['quantity'] -= sell_qty
            revenue = sell_qty * current_price
            self.cash += revenue
            self.trades.append({
                'date': str(date)[:10], 'trade_id': len(self.trades) + 1,
                'code': code, 'action': 'SELL_HALF',
                'quantity': sell_qty, 'price': current_price,
                'amount': revenue, 'reason': reason,
            })
            self.partial_sells.append({'date': str(date)[:10], 'code': code,
                                       'reason': reason, 'remaining_qty': pos['quantity']})
        else:
            revenue = pos['quantity'] * current_price
            self.cash += revenue
            self.trades.append({
                'date': str(date)[:10], 'trade_id': len(self.trades) + 1,
                'code': code, 'action': 'SELL',
                'quantity': pos['quantity'], 'price': current_price,
                'amount': revenue, 'reason': reason,
            })
            del self.positions[code]

    def _calculate_stats(self, period_start, period_end):
        if not self.equity_curve:
            return {}
        values = [e['value'] for e in self.equity_curve]
        dates = [pd.Timestamp(e['date']) for e in self.equity_curve]
        total_return = (values[-1] - self.initial_capital) / self.initial_capital * 100
        years = (dates[-1] - dates[0]).days / 365.0
        annual_return = ((values[-1] / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        rets = pd.Series(values).pct_change().dropna()
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        peak = values[0]
        max_dd = 0
        for v in values:
            if v > peak: peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd: max_dd = dd
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] in ('SELL', 'SELL_HALF')]
        exit_reasons = {}
        for t in self.trades:
            if 'reason' in t:
                key = t['reason'].split('(')[0]
                exit_reasons[key] = exit_reasons.get(key, 0) + 1
        return {
            'period': f"{period_start} ~ {period_end}",
            'total_return': total_return, 'annual_return': annual_return,
            'sharpe_ratio': sharpe, 'max_drawdown': max_dd,
            'total_trades': len(self.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'full_sells': len([t for t in self.trades if t['action'] == 'SELL']),
            'partial_sells': len(self.partial_sells),
            'final_value': values[-1], 'exit_reasons': exit_reasons,
            'equity_curve': self.equity_curve, 'trades': self.trades,
        }


# ============================================================
# 3. 复用 v2.1 的基准对比（直接 import 即可）
# ============================================================

from backtest.etf_screener_backtest_v2 import BenchmarkComparator
# backtest/ 在 sys.path 中已经能 import（因为 ROOT 加进去了）


# ============================================================
# 4. 多周期回测（v3 专用）
# ============================================================

class MultiPeriodRunnerV4:
    """v4 多周期回测"""
    SUB_PERIODS = [
        ('2022-01-01', '2023-12-31', '2022-2023 (下行+反弹)'),
        ('2024-01-01', '2024-12-31', '2024 (急跌+反弹)'),
        ('2025-01-01', '2026-05-22', '2025-2026 (当前样本)'),
    ]

    def __init__(self, data_dir, initial_capital=10000.0,
                 min_score: float = 0.3, top_n: int = 10):
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.min_score = min_score
        self.top_n = top_n
        self.screener = LayeredETFScreener(data_dir)

    def run(self) -> List[dict]:
        out = []
        for start, end, label in self.SUB_PERIODS:
            top_etfs, ctx = self.screener.get_top_etfs(
                min_score=self.min_score, top_n=self.top_n, as_of_date=start
            )
            engine = ScreenerBacktestV4(initial_capital=self.initial_capital)
            result = engine.run_backtest(top_etfs, self.data_dir, start, end)
            if not result:
                out.append({'period': label, 'error': 'no data',
                            'selected_etfs': [e.code for e in top_etfs]})
                continue
            out.append({
                'period': label, 'range': f'{start} ~ {end}',
                'selected_etfs': [e.code for e in top_etfs[:5]],
                'market_regime': ctx.get('market_regime', 'UNKNOWN'),
                'macro_score': ctx.get('market_score', 0.5),
                'total_return': round(result['total_return'], 2),
                'sharpe': round(result['sharpe_ratio'], 3),
                'max_drawdown': round(result['max_drawdown'], 2),
                'trades': result['total_trades'],
                'final_value': round(result['final_value'], 2),
                'exit_reasons': result['exit_reasons'],
                'regime_distribution': result.get('regime_distribution', {}),
                'avg_macro_score': result.get('avg_macro_score', 0.5),
            })
        return out


# ============================================================
# 5. 主流程
# ============================================================

def main():
    print("=" * 70)
    print("ETF 智能筛选 + 回测系统 v4.0")
    print("v4 = v3 + 3 件事:")
    print("  1. L1 宏观重做: 4 维复合信号 (动量+斜率+波动率+连续阳线)")
    print("  2. L4 资金 + L5 情绪 新接入")
    print("  3. L3 板块重做: 60 日动量 + 板块轮动阶段")
    print("  仓位乘数: 连续值 [0, 1]，>0.65→1.0, <0.35→0.0")
    print("=" * 70)

    data_dir = str(Path(__file__).resolve().parent.parent / "data_cache")
    output_dir = Path(__file__).resolve().parent.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_capital = 10000.0
    top_n = 10
    min_score = 0.3
    main_start = '2020-01-01'
    main_end = '2026-05-22'

    print(f"\n[1/5] 预计算回测期: {main_start} ~ {main_end}（as_of_date={main_start}）")

    print("\n[2/5] ETF 智能筛选 (6 层框架 v4)...")
    screener = LayeredETFScreener(data_dir)
    top_etfs, ctx = screener.get_top_etfs(min_score=min_score, top_n=top_n, as_of_date=main_start)
    if not top_etfs:
        print("未找到符合条件的 ETF")
        return
    print(f"\n  筛选结果 (前 {len(top_etfs)} 只):")
    print(f"  {'代码':<8} {'L1':>5} {'L3m':>5} {'L3p':>5} {'L4':>5} {'L5':>5} {'L6':>5} {'L7':>5} {'综合':>6}")
    for etf in top_etfs:
        print(f"  {etf.code:<8} {etf.layer1_macro:>5.2f} {etf.layer3_60d_mom:>5.2f} "
              f"{etf.layer3_phase:>5.2f} {etf.layer4_capital:>5.2f} {etf.layer5_sentiment:>5.2f} "
              f"{etf.layer6_pv:>5.2f} {etf.layer7_tech:>5.2f} {etf.total_score:>6.3f}")

    print(f"\n[3/5] 主回测 (v4: 6 层框架 + 连续宏观仓位)...")
    engine = ScreenerBacktestV4(initial_capital=initial_capital)
    main_result = engine.run_backtest(top_etfs, data_dir, main_start, main_end)
    if not main_result:
        print("主回测失败")
        return
    print(f"\n  === v4 主回测结果 ===")
    print(f"  区间:           {main_result['period']}")
    print(f"  总收益:         {main_result['total_return']:+.2f}%")
    print(f"  年化收益:       {main_result['annual_return']:+.2f}%")
    print(f"  夏普:           {main_result['sharpe_ratio']:.3f}")
    print(f"  最大回撤:       {main_result['max_drawdown']:.2f}%")
    print(f"  交易次数:       {main_result['total_trades']}")
    print(f"  期末净值:       {main_result['final_value']:.2f}")
    print(f"  退出原因:       {main_result['exit_reasons']}")
    print(f"  宏观状态分布:   {main_result['regime_distribution']}")
    print(f"  平均宏观得分:   {main_result.get('avg_macro_score', 0.5):.3f}")

    print(f"\n[4/5] ETF 基准对比...")
    period_start = main_result['equity_curve'][0]['date']
    period_end = main_result['equity_curve'][-1]['date']
    comparator = BenchmarkComparator(data_dir)
    bench_results = comparator.compute_benchmarks(period_start, period_end, initial_capital)
    basket = comparator.compute_equal_weight_basket(period_start, period_end, initial_capital)
    cmp_rows = BenchmarkComparator.make_comparison(main_result, bench_results, basket)
    print(f"\n  {'标的':<28} {'收益%':>8} {'夏普':>7} {'回撤%':>7} {'超额%':>8}")
    print("  " + "-" * 60)
    for r in cmp_rows:
        print(f"  {r['name']:<28} {r['return_pct']:>+7.2f}% {r['sharpe']:>6.3f} "
              f"{r['max_drawdown']:>6.2f}% {r['excess_vs_strategy']:>+7.2f}%")

    print(f"\n[5/5] 多周期回测...")
    multi = MultiPeriodRunnerV4(data_dir, initial_capital, min_score, top_n)
    multi_results = multi.run()
    print(f"\n  {'周期':<28} {'收益%':>8} {'夏普':>7} {'回撤%':>7} {'宏观':>10} {'交易':>5}")
    print("  " + "-" * 70)
    for r in multi_results:
        if 'error' in r and 'total_return' not in r:
            print(f"  {r['period']:<28} {'无数据':>8}")
            continue
        print(f"  {r['period']:<28} {r['total_return']:>+7.2f}% {r['sharpe']:>6.3f} "
              f"{r['max_drawdown']:>6.2f}% {r.get('market_regime', '?'):>10} {r['trades']:>5d}")

    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': 'v4.0 (6层框架 L1改进+L3重做+L4新增+L5新增+L6+L7 + 连续宏观仓位)',
        'selected_etfs': [{'code': e.code, 'total_score': round(e.total_score, 3),
                            'L1': e.layer1_macro, 'L3_mom': e.layer3_60d_mom, 'L3_phase': e.layer3_phase,
                            'L4': e.layer4_capital, 'L5': e.layer5_sentiment,
                            'L6': e.layer6_pv, 'L7': e.layer7_tech} for e in top_etfs],
        'main_backtest': {
            'period': main_result['period'],
            'total_return': round(main_result['total_return'], 2),
            'annual_return': round(main_result['annual_return'], 2),
            'sharpe_ratio': round(main_result['sharpe_ratio'], 3),
            'max_drawdown': round(main_result['max_drawdown'], 2),
            'final_value': round(main_result['final_value'], 2),
            'total_trades': main_result['total_trades'],
            'exit_reasons': main_result['exit_reasons'],
            'regime_distribution': main_result['regime_distribution'],
            'avg_macro_score': main_result.get('avg_macro_score', 0.5),
        },
        'benchmark_comparison': {'benchmarks': bench_results, 'basket': basket, 'table': cmp_rows},
        'multi_period': multi_results,
    }

    out_file = output_dir / "screener_backtest_v4_results.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
