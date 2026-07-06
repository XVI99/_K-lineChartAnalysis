# -*- coding: utf-8 -*-
"""Advanced ETF factor research pipeline.

This script covers three research gates in one reproducible offline run:

1. Factor selection: IC, ICIR, quantile returns, monotonicity, turnover and
   correlation pruning, selecting 5-15 core factors.
2. Regime layering: trend bull, range, down, high volatility, low volatility,
   theme-led and broad-based regimes.
3. Transaction-cost walk-forward: rolling train/test validation with slippage,
   commission, T+1 open execution and 100-share lot sizing.

Output directory: AStockQuant/reports/advanced_factor_research/
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "advanced_factor_research"
DATA_DIR = ROOT / "data_cache"
DEFAULT_CONFIG_PATH = ROOT / "config" / "factor_research.yaml"


# ---- 默认配置（代码内 fallback；优先级: CLI > yaml > 此默认值）----
DEFAULT_CONFIG: Dict[str, Any] = {
    "date_range": {"start": "2021-01-01", "end": "2026-03-31"},
    "panel": {
        "rebalance_freq": 5,
        "lookback_min": 120,
        "fwd_days": 20,
        "n_quantiles": 5,
        "market_proxy": "510300",
        "min_lot": 100,
        "min_eligible": 20,
        "min_factor_coverage": 200,
        "min_factor_unique": 3,
        "min_total_etfs": 50,
    },
    "regime": {
        "trend": {
            "ma_window": 60, "ret_window": 60, "bull_threshold": 0.05,
            "down_threshold": -0.05, "min_history": 120,
        },
        "volatility": {
            "window": 20, "hist_window": 120, "min_hist_for_pct": 30,
            "high_vol_pct": 0.7, "low_vol_pct": 0.3,
        },
        "leadership": {
            "return_window": 20, "min_history": 21,
            "theme_dispersion": 0.035, "theme_excess": 0.03,
            "broad_breadth": 0.60, "broad_dispersion": 0.05,
        },
    },
    "factor_selection": {
        "min_n": 5, "max_n": 15, "min_ic_records": 8, "min_ls_records": 8,
        "thresholds": {
            "ic_mean": 0.01, "icir": 0.05, "long_short_ret": 0.002,
            "monotonicity": 0.5, "top_quantile_turnover": 0.90,
        },
        "relaxed_thresholds": {"ic_mean": 0.0, "icir": 0.0, "long_short_ret": 0.0},
        "correlation_cutoff": 0.75,
        "priority_factors": ["pv_rps_20", "pv_rps_combined"],
        "score_weights": {"ic": 10.0, "icir": 1.0, "monotonicity": 1.0,
                          "turnover_penalty": 0.25},
    },
    "walk_forward": {
        "train_months": 12, "test_months": 3, "min_train_dates": 10,
        "min_test_dates": 3, "top_n": 10, "initial_capital": 1_000_000.0,
        "slippage": 0.001, "commission": 0.0003, "buffer_rank": 5,
        "min_hold_periods": 3, "replace_score_margin": 0.03, "signal_stride": 1,
        "use_regime_gating": True, "rps_blend_weight": 0.65,
        "rps_source": "pv_rps_20", "min_ic_train_samples": 15,
    },
    "regime_exposure": {
        "down_high_vol": 0.0, "down": 0.3, "high_vol_mixed": 0.5,
        "theme_market": 1.0, "broad_market": 1.0, "trend_bull": 0.9, "default": 0.7,
    },
    "benchmark": {
        "buy_hold_symbols": ["510300", "510500"],
        "rps_momentum": {
            "weight_20d": 0.6, "weight_120d": 0.4, "top_n": 10,
            "min_history_20d": 21, "min_history_120d": 121,
        },
        "equal_weight_lookback": 21,
    },
    "robustness": {
        "top_n_grid": [5, 10, 15], "stride_grid": [1, 2],
        "slippage_grid": [0.0005, 0.001, 0.002],
    },
    "stats": {"annualization_factor": 252},
}

# 运行时配置（main() 中由 yaml + CLI 覆盖；模块级函数统一读 CFG）
CFG: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)


def _deep_update(base: dict, override: dict) -> dict:
    """递归合并 override 到 base（override 优先），原地修改并返回 base。"""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """加载 yaml 配置，deep_update 到 CFG 的副本。返回新的 cfg。"""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        p = Path(path)
        if p.exists() and yaml is not None:
            with open(p, "r", encoding="utf-8") as f:
                _deep_update(cfg, yaml.safe_load(f) or {})
        elif not p.exists():
            print(f"[config] 配置文件不存在，使用默认值: {p}")
    return cfg


def spearmanr(x, y) -> Tuple[float, float]:
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < 3:
        return np.nan, np.nan
    rx = s["x"].rank()
    ry = s["y"].rank()
    sx = rx - rx.mean()
    sy = ry - ry.mean()
    denom = np.sqrt((sx * sx).sum()) * np.sqrt((sy * sy).sum())
    if denom == 0:
        return np.nan, np.nan
    return float((sx * sy).sum() / denom), 0.0


def load_all_data(min_rows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    if min_rows is None:
        min_rows = CFG["panel"]["lookback_min"]
    all_data: Dict[str, pd.DataFrame] = {}
    for path in sorted(DATA_DIR.glob("*.csv")):
        symbol = path.stem
        df = pd.read_csv(path)
        if "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        cols = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in cols):
            continue
        df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) >= min_rows:
            all_data[symbol] = df
    return all_data


def get_cross_dates(all_data: Dict[str, pd.DataFrame], start: str, end: str) -> List[pd.Timestamp]:
    all_dates = sorted(set().union(*[set(df.index) for df in all_data.values()]))
    dates = pd.DatetimeIndex(all_dates)
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    return list(dates[::CFG["panel"]["rebalance_freq"]])


def build_registry():
    import AStockQuant.layers.layer4_capital as layer4_mod
    from AStockQuant.core.feature_registry import FeatureRegistry
    from AStockQuant.layers.layer1_macro import MacroLayer
    from AStockQuant.layers.layer2_rules import RulesLayer
    from AStockQuant.layers.layer3_sector import SectorLayer
    from AStockQuant.layers.layer4_capital import CapitalLayer
    from AStockQuant.layers.layer5_sentiment import SentimentLayer
    from AStockQuant.layers.layer6_price_vol import PriceVolumeLayer
    from AStockQuant.layers.layer7_technical import TechnicalLayer
    from AStockQuant.layers.layer8_micro import BeliefLayer

    # Historical research must be offline and point-in-time; live AkShare calls
    # are deliberately disabled so Layer4 uses deterministic volume proxy unless
    # historical external snapshots are added later.
    layer4_mod._AK = False

    registry = FeatureRegistry()
    registry.register("macro", MacroLayer())
    registry.register("rules", RulesLayer())
    registry.register("sector", SectorLayer())
    registry.register("capital", CapitalLayer())
    registry.register("sentiment", SentimentLayer())
    registry.register("price_vol", PriceVolumeLayer())
    registry.register("technical", TechnicalLayer())
    registry.register("belief", BeliefLayer())
    return registry


def classify_regime(as_of: pd.Timestamp, all_data: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    rt = CFG["regime"]["trend"]
    vt = CFG["regime"]["volatility"]
    ld = CFG["regime"]["leadership"]
    market = all_data.get(CFG["panel"]["market_proxy"])
    if market is None or market.empty:
        market = next(iter(all_data.values()))
    m = market[market.index <= as_of]
    if len(m) < rt["min_history"]:
        return {
            "trend_regime": "range",
            "vol_regime": "low_vol",
            "leadership_regime": "broad_market",
            "regime_label": "range_low_vol_broad_market",
        }

    close = m["close"]
    ma = close.rolling(rt["ma_window"]).mean().iloc[-1]
    retN = close.pct_change(rt["ret_window"]).iloc[-1]
    if close.iloc[-1] > ma and retN > rt["bull_threshold"]:
        trend_regime = "trend_bull"
    elif close.iloc[-1] < ma and retN < rt["down_threshold"]:
        trend_regime = "down"
    else:
        trend_regime = "range"

    ret = close.pct_change()
    vol20 = ret.rolling(vt["window"]).std().iloc[-1]
    vol_hist = ret.rolling(vt["window"]).std().tail(vt["hist_window"]).dropna()
    if len(vol_hist) >= vt["min_hist_for_pct"]:
        vol_pct = float((vol_hist <= vol20).mean())
    else:
        vol_pct = 0.5
    vol_regime = ("high_vol" if vol_pct >= vt["high_vol_pct"]
                  else "low_vol" if vol_pct <= vt["low_vol_pct"] else "mid_vol")

    returns_20 = {}
    breadth_hits = 0
    breadth_total = 0
    for sym, df in all_data.items():
        d = df[df.index <= as_of]
        if len(d) >= ld["min_history"]:
            r20 = d["close"].pct_change(ld["return_window"]).iloc[-1]
            returns_20[sym] = r20
            breadth_hits += int(r20 > 0)
            breadth_total += 1
    breadth = breadth_hits / breadth_total if breadth_total else 0.5

    from AStockQuant.layers.layer3_sector import SectorLayer

    sector_returns: Dict[str, List[float]] = {}
    for sym, r in returns_20.items():
        if np.isnan(r):
            continue
        sector = SectorLayer.get_sector(sym)[0]
        sector_returns.setdefault(sector, []).append(float(r))
    sector_mean = {k: float(np.mean(v)) for k, v in sector_returns.items() if v}
    dispersion = float(np.std(list(sector_mean.values()))) if sector_mean else 0.0
    top_sector_ret = max(sector_mean.values()) if sector_mean else 0.0
    market_ret = float(retN) if not np.isnan(retN) else 0.0
    if dispersion > ld["theme_dispersion"] and top_sector_ret - market_ret > ld["theme_excess"]:
        leadership_regime = "theme_market"
    elif breadth >= ld["broad_breadth"] and dispersion < ld["broad_dispersion"]:
        leadership_regime = "broad_market"
    else:
        leadership_regime = "mixed_market"

    return {
        "trend_regime": trend_regime,
        "vol_regime": vol_regime,
        "leadership_regime": leadership_regime,
        "regime_label": f"{trend_regime}_{vol_regime}_{leadership_regime}",
        "market_ret_60d": market_ret,
        "market_vol_pct": vol_pct,
        "market_breadth_20d": breadth,
        "sector_dispersion_20d": dispersion,
    }


def build_factor_panel(
    all_data: Dict[str, pd.DataFrame],
    cross_dates: List[pd.Timestamp],
    cache_path: Path,
    rebuild: bool = False,
) -> pd.DataFrame:
    if cache_path.exists() and not rebuild:
        panel = pd.read_csv(cache_path, dtype={"symbol": str})
        panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
        return panel

    registry = build_registry()
    market_df = all_data.get(CFG["panel"]["market_proxy"])
    records = []
    for i, as_of in enumerate(cross_dates, 1):
        eligible = {
            sym: df for sym, df in all_data.items()
            if len(df[df.index <= as_of]) >= CFG["panel"]["lookback_min"] + CFG["panel"]["fwd_days"]
        }
        if len(eligible) < CFG["panel"]["min_eligible"]:
            continue
        regime = classify_regime(as_of, all_data)
        ctx = {"market_prices_df": market_df[market_df.index <= as_of] if market_df is not None else None}
        batch = registry.extract_batch(list(eligible), eligible, context=ctx, as_of_date=as_of.strftime("%Y-%m-%d"))
        for sym, feats in batch.items():
            row = {"date": as_of.strftime("%Y-%m-%d"), "symbol": sym, **regime}
            for k, v in feats.items():
                if isinstance(v, bool):
                    row[k] = 1.0 if v else 0.0
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    row[k] = float(v)
            records.append(row)
        if i % 20 == 0 or i == len(cross_dates):
            print(f"panel progress {i}/{len(cross_dates)} records={len(records)}")

    panel = pd.DataFrame(records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return panel


def add_forward_returns(panel: pd.DataFrame, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    values = []
    for _, row in panel.iterrows():
        sym = str(row["symbol"]).zfill(6)
        as_of = pd.Timestamp(row["date"])
        df = all_data.get(sym)
        if df is None:
            values.append(np.nan)
            continue
        past = df[df.index <= as_of]
        future = df[df.index > as_of]
        fwd = CFG["panel"]["fwd_days"]
        if len(past) == 0 or len(future) < fwd:
            values.append(np.nan)
        else:
            values.append(float(future["close"].iloc[fwd - 1] / past["close"].iloc[-1] - 1))
    out = panel.copy()
    out["fwd_ret_20d"] = values
    return out


def numeric_factor_columns(panel: pd.DataFrame) -> List[str]:
    excluded = {
        "date", "symbol", "fwd_ret_20d", "market_ret_60d", "market_vol_pct",
        "market_breadth_20d", "sector_dispersion_20d",
    }
    excluded_prefixes = ("regime",)
    factors = []
    min_cov = CFG["panel"]["min_factor_coverage"]
    min_uniq = CFG["panel"]["min_factor_unique"]
    for col in panel.columns:
        if col in excluded or col.startswith(excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(panel[col]) and panel[col].notna().sum() >= min_cov:
            if panel[col].nunique(dropna=True) > min_uniq:
                factors.append(col)
    return factors


def factor_metrics(panel: pd.DataFrame, factors: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(panel["date"].unique())
    nq = CFG["panel"]["n_quantiles"]
    fs = CFG["factor_selection"]
    sw = fs["score_weights"]
    ann = CFG["stats"]["annualization_factor"]
    rfreq = CFG["panel"]["rebalance_freq"]
    min_ic = fs["min_ic_records"]
    min_ls = fs["min_ls_records"]
    rows = []
    ic_records = []
    quantile_rows = []
    for factor in factors:
        ic_list = []
        q_returns = {q: [] for q in range(nq)}
        ls_returns = []
        top_sets = []
        for dt in dates:
            sub = panel[panel["date"] == dt][["symbol", factor, "fwd_ret_20d"]].dropna()
            if len(sub) < max(20, nq * 3):
                continue
            ic, _ = spearmanr(sub[factor], sub["fwd_ret_20d"])
            if not np.isnan(ic):
                ic_list.append(ic)
                ic_records.append({"date": dt, "factor": factor, "ic": ic})
            try:
                sub["q"] = pd.qcut(sub[factor], nq, labels=False, duplicates="drop")
            except Exception:
                continue
            if sub["q"].nunique() < nq:
                continue
            for q in range(nq):
                q_returns[q].append(sub[sub["q"] == q]["fwd_ret_20d"].mean())
            q_low = sub[sub["q"] == 0]["fwd_ret_20d"].mean()
            q_high = sub[sub["q"] == nq - 1]["fwd_ret_20d"].mean()
            ls_returns.append(q_high - q_low)
            top_sets.append(set(sub[sub["q"] == nq - 1]["symbol"]))

        if len(ic_list) < min_ic or len(ls_returns) < min_ls:
            continue
        ic_arr = np.array(ic_list)
        ls_arr = np.array(ls_returns)
        q_means = [float(np.mean(q_returns[q])) if q_returns[q] else np.nan for q in range(nq)]
        monotonicity = np.mean([q_means[i] > q_means[i - 1] for i in range(1, nq)]) if not any(np.isnan(q_means)) else np.nan
        turnovers = []
        for a, b in zip(top_sets[:-1], top_sets[1:]):
            turnovers.append(1.0 - len(a & b) / max(1, len(a | b)))
        turnover = float(np.mean(turnovers)) if turnovers else np.nan
        ic_std = np.std(ic_arr, ddof=1)
        ls_std = np.std(ls_arr, ddof=1)
        rows.append({
            "factor": factor,
            "ic_mean": float(np.mean(ic_arr)),
            "icir": float(np.mean(ic_arr) / ic_std) if ic_std > 0 else 0.0,
            "ic_winrate": float(np.mean(ic_arr > 0)),
            "ic_tstat": float(np.mean(ic_arr) / (ic_std / np.sqrt(len(ic_arr)))) if ic_std > 0 else 0.0,
            "q1_ret": q_means[0], "q2_ret": q_means[1], "q3_ret": q_means[2],
            "q4_ret": q_means[3], "q5_ret": q_means[4],
            "long_short_ret": float(np.mean(ls_arr)),
            "long_short_sharpe": float(np.mean(ls_arr) / ls_std * np.sqrt(ann / rfreq)) if ls_std > 0 else 0.0,
            "monotonicity": float(monotonicity),
            "top_quantile_turnover": turnover,
            "score": float(np.mean(ic_arr) * sw["ic"] + (np.mean(ic_arr) / ic_std if ic_std > 0 else 0) * sw["icir"] + monotonicity * sw["monotonicity"] - sw["turnover_penalty"] * (turnover if not np.isnan(turnover) else 1.0)),
        })
        for q, ret in enumerate(q_means, 1):
            quantile_rows.append({"factor": factor, "quantile": q, "mean_return": ret})
    return pd.DataFrame(rows).sort_values("score", ascending=False), pd.DataFrame(ic_records), pd.DataFrame(quantile_rows)


def select_core_factors(metrics: pd.DataFrame, panel: pd.DataFrame, min_n: Optional[int] = None, max_n: Optional[int] = None) -> Tuple[List[str], pd.DataFrame]:
    fs = CFG["factor_selection"]
    if min_n is None:
        min_n = fs["min_n"]
    if max_n is None:
        max_n = fs["max_n"]
    th = fs["thresholds"]
    rth = fs["relaxed_thresholds"]
    candidates = metrics[
        (metrics["ic_mean"] >= th["ic_mean"]) &
        (metrics["icir"] >= th["icir"]) &
        (metrics["long_short_ret"] >= th["long_short_ret"]) &
        (metrics["monotonicity"] >= th["monotonicity"]) &
        (metrics["top_quantile_turnover"] <= th["top_quantile_turnover"])
    ].copy()
    if len(candidates) < min_n:
        candidates = metrics[
            (metrics["ic_mean"] > rth["ic_mean"]) &
            (metrics["icir"] > rth["icir"]) &
            (metrics["long_short_ret"] > rth["long_short_ret"])
        ].copy()
    candidates = candidates.sort_values("score", ascending=False)
    selected: List[str] = []
    numeric = candidates["factor"].tolist()
    corr = panel[numeric].corr(method="spearman").fillna(0.0) if numeric else pd.DataFrame()
    # RPS is the strongest external benchmark in ETF rotation; keep the most
    # responsive RPS factor even if it is moderately correlated with other
    # momentum features, then let the rest of the list diversify around it.
    for priority in fs["priority_factors"]:
        if priority in numeric and priority not in selected:
            selected.append(priority)
            break
    for factor in numeric:
        if len(selected) >= max_n:
            break
        if factor in selected:
            continue
        if all(abs(corr.loc[factor, prev]) < fs["correlation_cutoff"] for prev in selected):
            selected.append(factor)
    if len(selected) < min_n:
        for factor in numeric:
            if factor not in selected:
                selected.append(factor)
            if len(selected) >= min_n:
                break
    selected_df = metrics[metrics["factor"].isin(selected)].copy()
    selected_df["selected_rank"] = selected_df["factor"].map({f: i + 1 for i, f in enumerate(selected)})
    selected_df = selected_df.sort_values("selected_rank")
    return selected, selected_df


def next_trade_date(all_dates: List[pd.Timestamp], signal_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    for d in all_dates:
        if d > signal_date:
            return d
    return None


def price_at(df: pd.DataFrame, date: pd.Timestamp, field: str = "open") -> Optional[float]:
    d = df[df.index >= date]
    if d.empty:
        return None
    return float(d[field].iloc[0])


@dataclass
class WalkForwardConfig:
    train_months: int = field(default_factory=lambda: CFG["walk_forward"]["train_months"])
    test_months: int = field(default_factory=lambda: CFG["walk_forward"]["test_months"])
    top_n: int = field(default_factory=lambda: CFG["walk_forward"]["top_n"])
    initial_capital: float = field(default_factory=lambda: CFG["walk_forward"]["initial_capital"])
    slippage: float = field(default_factory=lambda: CFG["walk_forward"]["slippage"])
    commission: float = field(default_factory=lambda: CFG["walk_forward"]["commission"])
    buffer_rank: int = field(default_factory=lambda: CFG["walk_forward"]["buffer_rank"])
    min_hold_periods: int = field(default_factory=lambda: CFG["walk_forward"]["min_hold_periods"])
    replace_score_margin: float = field(default_factory=lambda: CFG["walk_forward"]["replace_score_margin"])
    signal_stride: int = field(default_factory=lambda: CFG["walk_forward"]["signal_stride"])
    use_regime_gating: bool = field(default_factory=lambda: CFG["walk_forward"]["use_regime_gating"])
    rps_blend_weight: float = field(default_factory=lambda: CFG["walk_forward"]["rps_blend_weight"])
    rps_source: str = field(default_factory=lambda: CFG["walk_forward"]["rps_source"])
    min_train_dates: int = field(default_factory=lambda: CFG["walk_forward"]["min_train_dates"])
    min_test_dates: int = field(default_factory=lambda: CFG["walk_forward"]["min_test_dates"])
    min_ic_train_samples: int = field(default_factory=lambda: CFG["walk_forward"]["min_ic_train_samples"])


def regime_exposure(row: pd.Series, enabled: bool = True) -> float:
    """Map market regime to target gross exposure."""
    if not enabled:
        return 1.0
    re = CFG["regime_exposure"]
    trend = row.get("trend_regime", "range")
    vol = row.get("vol_regime", "mid_vol")
    leadership = row.get("leadership_regime", "mixed_market")
    if trend == "down" and vol == "high_vol":
        return re["down_high_vol"]
    if trend == "down":
        return re["down"]
    if vol == "high_vol" and leadership == "mixed_market":
        return re["high_vol_mixed"]
    if leadership in ("theme_market", "broad_market"):
        return re[leadership]
    if trend == "trend_bull":
        return re["trend_bull"]
    return re["default"]


def run_walk_forward(panel: pd.DataFrame, all_data: Dict[str, pd.DataFrame], factors: List[str], cfg: WalkForwardConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(panel["date"].unique()))
    all_trade_dates = sorted(set().union(*[set(df.index) for df in all_data.values()]))
    folds = []
    equity_rows = []
    cash = cfg.initial_capital
    positions: Dict[str, int] = {}
    hold_periods: Dict[str, int] = {}
    last_equity = cfg.initial_capital
    start = dates[0]
    fold_id = 0

    while True:
        train_end = start + pd.DateOffset(months=cfg.train_months)
        test_end = train_end + pd.DateOffset(months=cfg.test_months)
        train_dates = [d for d in dates if start <= d < train_end]
        test_dates = [d for d in dates if train_end <= d < test_end]
        if len(train_dates) < cfg.min_train_dates or len(test_dates) < cfg.min_test_dates:
            break
        fold_id += 1
        train = panel[pd.to_datetime(panel["date"]).isin(train_dates)]
        weights = {}
        directions = {}
        for f in factors:
            ic_vals = []
            for dt in sorted(train["date"].unique()):
                sub = train[train["date"] == dt][[f, "fwd_ret_20d"]].dropna()
                if len(sub) >= cfg.min_ic_train_samples:
                    ic, _ = spearmanr(sub[f], sub["fwd_ret_20d"])
                    if not np.isnan(ic):
                        ic_vals.append(ic)
            mean_ic = float(np.mean(ic_vals)) if ic_vals else 0.0
            directions[f] = 1 if mean_ic >= 0 else -1
            weights[f] = abs(mean_ic)
        total_w = sum(weights.values())
        if total_w <= 0:
            weights = {f: 1 / len(factors) for f in factors}
        else:
            weights = {f: weights[f] / total_w for f in factors}

        fold_start_equity = last_equity
        fold_turnover = []
        fold_cost = 0.0
        for signal_idx, signal_dt in enumerate(test_dates):
            if signal_idx % cfg.signal_stride != 0:
                continue
            signal_str = signal_dt.strftime("%Y-%m-%d")
            trade_dt = next_trade_date(all_trade_dates, signal_dt)
            if trade_dt is None:
                continue
            sub = panel[panel["date"] == signal_str].copy()
            if sub.empty:
                continue
            factor_score = pd.Series(0.0, index=sub.index)
            for f in factors:
                ranked = sub[f].rank(pct=True)
                if directions[f] < 0:
                    ranked = 1.0 - ranked
                factor_score += ranked.fillna(0.5) * weights[f]
            rps_source = cfg.rps_source if cfg.rps_source in sub.columns else ("pv_rps_combined" if "pv_rps_combined" in sub.columns else cfg.rps_source)
            if rps_source in sub.columns:
                rps_score = sub[rps_source].rank(pct=True).fillna(0.5)
                score = cfg.rps_blend_weight * rps_score + (1.0 - cfg.rps_blend_weight) * factor_score
            else:
                score = factor_score
            sub["wf_score"] = score
            ranked_sub = sub.sort_values("wf_score", ascending=False).copy()
            ranked_sub["rank"] = np.arange(1, len(ranked_sub) + 1)
            score_map = dict(zip(ranked_sub["symbol"].astype(str).str.zfill(6), ranked_sub["wf_score"]))
            rank_map = dict(zip(ranked_sub["symbol"].astype(str).str.zfill(6), ranked_sub["rank"]))
            top_candidates = ranked_sub.head(cfg.top_n + cfg.buffer_rank)["symbol"].astype(str).str.zfill(6).tolist()
            top_core = ranked_sub.head(cfg.top_n)["symbol"].astype(str).str.zfill(6).tolist()
            exposure = regime_exposure(ranked_sub.iloc[0], cfg.use_regime_gating)

            # Mark-to-open current equity.
            equity = cash
            for sym, shares in positions.items():
                px = price_at(all_data[sym], trade_dt, "open")
                if px is not None:
                    equity += shares * px
            target_gross = equity * exposure
            target_value = target_gross / max(1, cfg.top_n)
            old_symbols = set(positions)
            retained = []
            for sym in list(positions):
                rank = rank_map.get(sym, 10_000)
                age = hold_periods.get(sym, 0)
                if exposure > 0 and (rank <= cfg.top_n + cfg.buffer_rank or age < cfg.min_hold_periods):
                    retained.append(sym)

            picks = list(retained)
            for sym in top_core:
                if sym not in picks:
                    picks.append(sym)
                if len(picks) >= cfg.top_n:
                    break

            # Replace only if the new candidate is meaningfully better than the worst retained name.
            while len(picks) < cfg.top_n and top_candidates:
                cand = top_candidates.pop(0)
                if cand in picks:
                    continue
                if not picks:
                    picks.append(cand)
                    continue
                worst = min(picks, key=lambda s: score_map.get(s, -np.inf))
                if score_map.get(cand, 0.0) >= score_map.get(worst, 0.0) + cfg.replace_score_margin:
                    if hold_periods.get(worst, 999) >= cfg.min_hold_periods:
                        picks.remove(worst)
                    picks.append(cand)
            picks = picks[:cfg.top_n] if exposure > 0 else []

            new_symbols = set(picks)
            fold_turnover.append(1.0 - len(old_symbols & new_symbols) / max(1, len(old_symbols | new_symbols)))

            # Sell names not in target; keep retained holdings to reduce turnover.
            for sym in list(positions):
                if sym in new_symbols:
                    continue
                px = price_at(all_data[sym], trade_dt, "open")
                if px is None:
                    continue
                sell_px = px * (1 - cfg.slippage)
                proceeds = positions[sym] * sell_px
                cost = proceeds * cfg.commission
                cash += proceeds - cost
                fold_cost += cost + positions[sym] * px * cfg.slippage
                del positions[sym]
                hold_periods.pop(sym, None)

            for sym in picks:
                px = price_at(all_data[sym], trade_dt, "open")
                if px is None or px <= 0:
                    continue
                buy_px = px * (1 + cfg.slippage)
                current_shares = positions.get(sym, 0)
                current_value = current_shares * px
                lot = CFG["panel"]["min_lot"]
                desired_shares = int(target_value / buy_px / lot) * lot
                delta = desired_shares - current_shares
                if abs(delta) < lot:
                    continue
                if delta > 0:
                    gross = delta * buy_px
                    cost = gross * cfg.commission
                    if gross + cost <= cash:
                        cash -= gross + cost
                        positions[sym] = current_shares + delta
                        fold_cost += cost + delta * px * cfg.slippage
                else:
                    sell_shares = abs(delta)
                    sell_px = px * (1 - cfg.slippage)
                    proceeds = sell_shares * sell_px
                    cost = proceeds * cfg.commission
                    cash += proceeds - cost
                    positions[sym] = current_shares - sell_shares
                    fold_cost += cost + sell_shares * px * cfg.slippage
                    if positions[sym] <= 0:
                        del positions[sym]
                        hold_periods.pop(sym, None)

            for sym in list(positions):
                hold_periods[sym] = hold_periods.get(sym, 0) + 1

            close_equity = cash
            for sym, shares in positions.items():
                px = price_at(all_data[sym], trade_dt, "close") or price_at(all_data[sym], trade_dt, "open")
                if px is not None:
                    close_equity += shares * px
            last_equity = close_equity
            equity_rows.append({
                "fold": fold_id,
                "signal_date": signal_str,
                "trade_date": trade_dt.strftime("%Y-%m-%d"),
                "equity": close_equity,
                "cash": cash,
                "positions": len(positions),
                "regime_label": sub["regime_label"].iloc[0],
                "trend_regime": sub["trend_regime"].iloc[0],
                "vol_regime": sub["vol_regime"].iloc[0],
                "leadership_regime": sub["leadership_regime"].iloc[0],
                "target_exposure": exposure,
            })

        fold_ret = last_equity / fold_start_equity - 1 if fold_start_equity > 0 else 0.0
        folds.append({
            "fold": fold_id,
            "train_start": train_dates[0].strftime("%Y-%m-%d"),
            "train_end": train_dates[-1].strftime("%Y-%m-%d"),
            "test_start": test_dates[0].strftime("%Y-%m-%d"),
            "test_end": test_dates[-1].strftime("%Y-%m-%d"),
            "fold_return_after_cost": fold_ret,
            "avg_turnover": float(np.mean(fold_turnover)) if fold_turnover else 0.0,
            "cost_paid": fold_cost,
            "end_equity": last_equity,
        })
        start = start + pd.DateOffset(months=cfg.test_months)
    return pd.DataFrame(folds), pd.DataFrame(equity_rows)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    cummax = equity.cummax()
    dd = equity / cummax - 1
    return float(dd.min())


def perf_stats(equity: pd.Series, initial_capital: float) -> Dict[str, float]:
    if equity.empty:
        return {"total_return": 0.0, "max_drawdown": 0.0, "ann_return": 0.0, "sharpe": 0.0}
    ret = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / initial_capital - 1)
    ann = CFG["stats"]["annualization_factor"]
    rfreq = CFG["panel"]["rebalance_freq"]
    ann_return = float((1 + total_return) ** (ann / max(1, len(equity) * rfreq)) - 1)
    sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(ann / rfreq)) if len(ret) > 2 and ret.std(ddof=1) > 0 else 0.0
    return {"total_return": total_return, "max_drawdown": max_drawdown(equity), "ann_return": ann_return, "sharpe": sharpe}


def benchmark_buy_hold(symbol: str, all_data: Dict[str, pd.DataFrame], equity_dates: List[pd.Timestamp], initial_capital: float) -> pd.Series:
    df = all_data.get(symbol)
    if df is None or df.empty or not equity_dates:
        return pd.Series(dtype=float)
    first = price_at(df, equity_dates[0], "open")
    if first is None or first <= 0:
        return pd.Series(dtype=float)
    lot = CFG["panel"]["min_lot"]
    shares = int(initial_capital / first / lot) * lot
    cash = initial_capital - shares * first
    vals = []
    for dt in equity_dates:
        px = price_at(df, dt, "close") or price_at(df, dt, "open") or first
        vals.append(cash + shares * px)
    return pd.Series(vals, index=equity_dates)


def benchmark_equal_weight(all_data: Dict[str, pd.DataFrame], equity_dates: List[pd.Timestamp], initial_capital: float) -> pd.Series:
    vals = []
    lookback = CFG["benchmark"]["equal_weight_lookback"]
    for dt in equity_dates:
        rets = []
        for df in all_data.values():
            hist = df[df.index <= dt]
            if len(hist) >= lookback:
                rets.append(hist["close"].iloc[-1] / hist["close"].iloc[-lookback] - 1)
        vals.append(vals[-1] * (1 + np.nanmean(rets)) if vals else initial_capital)
    return pd.Series(vals, index=equity_dates)


def benchmark_rps_momentum(all_data: Dict[str, pd.DataFrame], equity_dates: List[pd.Timestamp], initial_capital: float, top_n: Optional[int] = None, signal_dates: Optional[List[pd.Timestamp]] = None) -> pd.Series:
    """简单 RPS 动量基准。signal_dates 非空时用 signal_date 算 RPS、trade_date 开盘成交（无未来函数）。
    signal_dates 为空时回退到旧逻辑（用 trade_date 算 RPS，含未来函数，仅作历史对照）。"""
    rm = CFG["benchmark"]["rps_momentum"]
    if top_n is None:
        top_n = rm["top_n"]
    lot = CFG["panel"]["min_lot"]
    w20 = rm["weight_20d"]
    w120 = rm["weight_120d"]
    min_hist = rm["min_history_120d"]
    cash = initial_capital
    positions: Dict[str, int] = {}
    vals = []
    for i, dt in enumerate(equity_dates):
        rps_date = signal_dates[i] if signal_dates else dt
        scores = []
        for sym, df in all_data.items():
            hist = df[df.index <= rps_date]
            if len(hist) >= min_hist:
                rps = w20 * hist["close"].pct_change(20).iloc[-1] + w120 * hist["close"].pct_change(120).iloc[-1]
                scores.append((sym, rps))
        picks = [s for s, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]]
        equity = cash
        for sym, shares in positions.items():
            px = price_at(all_data[sym], dt, "open")
            if px is not None:
                equity += shares * px
        target = equity / max(1, len(picks))
        for sym in list(positions):
            if sym not in picks:
                px = price_at(all_data[sym], dt, "open")
                if px is not None:
                    cash += positions[sym] * px
                del positions[sym]
        for sym in picks:
            px = price_at(all_data[sym], dt, "open")
            if px is None or px <= 0:
                continue
            desired = int(target / px / lot) * lot
            delta = desired - positions.get(sym, 0)
            if delta > 0 and delta * px <= cash:
                cash -= delta * px
                positions[sym] = positions.get(sym, 0) + delta
            elif delta < 0:
                cash += abs(delta) * px
                positions[sym] = positions.get(sym, 0) + delta
                if positions[sym] <= 0:
                    del positions[sym]
        close_equity = cash
        for sym, shares in positions.items():
            px = price_at(all_data[sym], dt, "close") or price_at(all_data[sym], dt, "open")
            if px is not None:
                close_equity += shares * px
        vals.append(close_equity)
    return pd.Series(vals, index=equity_dates)


def compute_benchmarks(equity: pd.DataFrame, all_data: Dict[str, pd.DataFrame], initial_capital: float) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    dates = pd.to_datetime(equity["trade_date"]).tolist()
    bh_symbols = CFG["benchmark"]["buy_hold_symbols"]
    series = {
        "strategy": pd.Series(equity["equity"].values, index=dates),
    }
    for sym in bh_symbols:
        series[f"buy_hold_{sym}"] = benchmark_buy_hold(sym, all_data, dates, initial_capital)
    series["etf_equal_weight"] = benchmark_equal_weight(all_data, dates, initial_capital)
    signal_dates = pd.to_datetime(equity["signal_date"]).tolist() if "signal_date" in equity.columns else None
    series["simple_rps_momentum"] = benchmark_rps_momentum(all_data, dates, initial_capital, signal_dates=signal_dates)
    series["rps_momentum_lookahead"] = benchmark_rps_momentum(all_data, dates, initial_capital, signal_dates=None)
    rows = []
    strat_ret = perf_stats(series["strategy"], initial_capital)["total_return"]
    for name, s in series.items():
        stats = perf_stats(s.dropna(), initial_capital)
        stats["benchmark"] = name
        stats["excess_vs_strategy"] = strat_ret - stats["total_return"] if name != "strategy" else 0.0
        rows.append(stats)
    return pd.DataFrame(rows)[["benchmark", "total_return", "ann_return", "max_drawdown", "sharpe", "excess_vs_strategy"]]


def run_robustness_grid(panel: pd.DataFrame, all_data: Dict[str, pd.DataFrame], factors: List[str], base_cfg: WalkForwardConfig) -> pd.DataFrame:
    rows = []
    rb = CFG["robustness"]
    for top_n in rb["top_n_grid"]:
        for stride in rb["stride_grid"]:
            for slippage in rb["slippage_grid"]:
                cfg = WalkForwardConfig(
                    train_months=base_cfg.train_months,
                    test_months=base_cfg.test_months,
                    top_n=top_n,
                    initial_capital=base_cfg.initial_capital,
                    slippage=slippage,
                    commission=base_cfg.commission,
                    buffer_rank=base_cfg.buffer_rank,
                    min_hold_periods=base_cfg.min_hold_periods,
                    replace_score_margin=base_cfg.replace_score_margin,
                    signal_stride=stride,
                    use_regime_gating=base_cfg.use_regime_gating,
                    rps_blend_weight=base_cfg.rps_blend_weight,
                )
                folds, equity = run_walk_forward(panel, all_data, factors, cfg)
                stats = perf_stats(equity["equity"] if not equity.empty else pd.Series(dtype=float), cfg.initial_capital)
                rows.append({
                    "top_n": top_n,
                    "signal_stride": stride,
                    "slippage": slippage,
                    "commission": cfg.commission,
                    "folds": len(folds),
                    "avg_turnover": float(folds["avg_turnover"].mean()) if not folds.empty else np.nan,
                    **stats,
                })
    return pd.DataFrame(rows).sort_values("total_return", ascending=False)


def write_report(
    selected: pd.DataFrame,
    metrics: pd.DataFrame,
    regime_counts: pd.DataFrame,
    folds: pd.DataFrame,
    equity: pd.DataFrame,
    regime_perf: pd.DataFrame,
    initial_capital: float,
) -> None:
    lines = []
    lines.append("Advanced Factor Research Report")
    lines.append("=" * 80)
    lines.append(f"Core factors selected: {len(selected)}")
    lines.append("\nSelected factors:")
    for _, row in selected.iterrows():
        lines.append(
            f"{int(row['selected_rank']):02d}. {row['factor']} "
            f"IC={row['ic_mean']:.4f} ICIR={row['icir']:.3f} "
            f"LS={row['long_short_ret']:.4f} mono={row['monotonicity']:.2f} "
            f"turnover={row['top_quantile_turnover']:.2f}"
        )
    lines.append("\nRegime distribution:")
    lines.append("Regime schema: trend_bull/range/down + high_vol/mid_vol/low_vol + theme_market/broad_market/mixed_market")
    lines.append("Required labels covered by schema: 趋势牛=trend_bull, 震荡=range, 下跌=down, 高波动=high_vol, 低波动=low_vol, 主题行情=theme_market, 宽基行情=broad_market")
    for _, row in regime_counts.iterrows():
        lines.append(f"{row['regime_label']}: {int(row['n_dates'])} dates")
    if not equity.empty:
        total_ret = equity["equity"].iloc[-1] / initial_capital - 1
        mdd = max_drawdown(equity["equity"])
        lines.append("\nWalk-forward after cost:")
        lines.append(f"total_return={total_ret:.2%}")
        lines.append(f"max_drawdown={mdd:.2%}")
        lines.append(f"folds={len(folds)}")
        lines.append(f"final_equity={equity['equity'].iloc[-1]:,.2f}")
    if not regime_perf.empty:
        lines.append("\nWalk-forward by regime:")
        for _, row in regime_perf.iterrows():
            lines.append(f"{row['regime_label']}: n={int(row['n'])} mean_ret={row['mean_period_return']:.2%} win_rate={row['win_rate']:.2%}")
    lines.append("\nFiles: factor_metrics.csv, selected_core_factors.csv, factor_correlation.csv, regime_by_date.csv, regime_performance.csv, walk_forward_folds.csv, walk_forward_equity.csv")
    (REPORT_DIR / "advanced_factor_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    # 先从命令行提取 --config，加载 yaml 到 CFG（优先级: CLI > yaml > 默认）
    cfg_path = str(DEFAULT_CONFIG_PATH)
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            cfg_path = argv[i + 1]
        elif a.startswith("--config="):
            cfg_path = a.split("=", 1)[1]
    global CFG
    CFG = load_config(cfg_path)

    dr = CFG["date_range"]
    wf = CFG["walk_forward"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=cfg_path, help="配置文件路径 (yaml)")
    parser.add_argument("--start", default=dr["start"])
    parser.add_argument("--end", default=dr["end"])
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument("--train-months", type=int, default=wf["train_months"])
    parser.add_argument("--test-months", type=int, default=wf["test_months"])
    parser.add_argument("--top-n", type=int, default=wf["top_n"])
    parser.add_argument("--buffer-rank", type=int, default=wf["buffer_rank"])
    parser.add_argument("--min-hold-periods", type=int, default=wf["min_hold_periods"])
    parser.add_argument("--replace-score-margin", type=float, default=wf["replace_score_margin"])
    parser.add_argument("--signal-stride", type=int, default=wf["signal_stride"])
    parser.add_argument("--no-regime-gating", action="store_true")
    parser.add_argument("--rps-blend-weight", type=float, default=wf["rps_blend_weight"])
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_data = load_all_data()
    if len(all_data) < CFG["panel"]["min_total_etfs"]:
        raise RuntimeError(f"not enough ETF data: {len(all_data)}")
    cross_dates = get_cross_dates(all_data, args.start, args.end)
    cache_path = REPORT_DIR / "factor_panel.csv"
    panel = build_factor_panel(all_data, cross_dates, cache_path, rebuild=args.rebuild_panel)
    panel = add_forward_returns(panel, all_data)
    panel.to_csv(cache_path, index=False, encoding="utf-8-sig")

    factors = numeric_factor_columns(panel)
    metrics, ic_ts, qret = factor_metrics(panel, factors)
    selected, selected_df = select_core_factors(metrics, panel)
    corr = panel[selected].corr(method="spearman") if selected else pd.DataFrame()
    regime_by_date = panel[["date", "trend_regime", "vol_regime", "leadership_regime", "regime_label", "market_ret_60d", "market_vol_pct", "market_breadth_20d", "sector_dispersion_20d"]].drop_duplicates()
    regime_counts = regime_by_date.groupby("regime_label").size().reset_index(name="n_dates").sort_values("n_dates", ascending=False)

    wf_cfg = WalkForwardConfig(
        train_months=args.train_months,
        test_months=args.test_months,
        top_n=args.top_n,
        buffer_rank=args.buffer_rank,
        min_hold_periods=args.min_hold_periods,
        replace_score_margin=args.replace_score_margin,
        signal_stride=args.signal_stride,
        use_regime_gating=not args.no_regime_gating,
        rps_blend_weight=args.rps_blend_weight,
    )
    folds, equity = run_walk_forward(panel.dropna(subset=["fwd_ret_20d"]), all_data, selected, wf_cfg)
    if not equity.empty:
        equity["period_return"] = equity["equity"].pct_change().fillna(equity["equity"] / wf_cfg.initial_capital - 1)
        regime_perf = (
            equity.groupby("regime_label")["period_return"]
            .agg(n="count", mean_period_return="mean", total_period_return="sum", win_rate=lambda s: float((s > 0).mean()))
            .reset_index()
            .sort_values("mean_period_return", ascending=False)
        )
    else:
        regime_perf = pd.DataFrame(columns=["regime_label", "n", "mean_period_return", "total_period_return", "win_rate"])
    benchmarks = compute_benchmarks(equity, all_data, wf_cfg.initial_capital)
    robustness = pd.DataFrame()
    if not args.skip_robustness:
        robustness = run_robustness_grid(panel.dropna(subset=["fwd_ret_20d"]), all_data, selected, wf_cfg)

    metrics.to_csv(REPORT_DIR / "factor_metrics.csv", index=False, encoding="utf-8-sig")
    ic_ts.to_csv(REPORT_DIR / "ic_timeseries.csv", index=False, encoding="utf-8-sig")
    qret.to_csv(REPORT_DIR / "quantile_returns.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(REPORT_DIR / "selected_core_factors.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(REPORT_DIR / "factor_correlation.csv", encoding="utf-8-sig")
    regime_by_date.to_csv(REPORT_DIR / "regime_by_date.csv", index=False, encoding="utf-8-sig")
    regime_counts.to_csv(REPORT_DIR / "regime_counts.csv", index=False, encoding="utf-8-sig")
    regime_perf.to_csv(REPORT_DIR / "regime_performance.csv", index=False, encoding="utf-8-sig")
    benchmarks.to_csv(REPORT_DIR / "benchmark_comparison.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(REPORT_DIR / "robustness_grid.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(REPORT_DIR / "walk_forward_folds.csv", index=False, encoding="utf-8-sig")
    equity.to_csv(REPORT_DIR / "walk_forward_equity.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "run_metadata.json").write_text(json.dumps({
        "n_etfs": len(all_data),
        "n_cross_dates": len(cross_dates),
        "n_panel_rows": len(panel),
        "n_candidate_factors": len(factors),
        "n_selected_factors": len(selected),
        "selected_factors": selected,
        "min_lot": CFG["panel"]["min_lot"],
        "slippage": wf_cfg.slippage,
        "commission": wf_cfg.commission,
        "execution": "T+1 next open",
        "turnover_controls": {
            "buffer_rank": wf_cfg.buffer_rank,
            "min_hold_periods": wf_cfg.min_hold_periods,
            "replace_score_margin": wf_cfg.replace_score_margin,
            "signal_stride": wf_cfg.signal_stride,
            "rps_blend_weight": wf_cfg.rps_blend_weight,
        },
        "regime_gating": wf_cfg.use_regime_gating,
        "regime_schema": {
            "趋势牛": "trend_bull",
            "震荡": "range",
            "下跌": "down",
            "高波动": "high_vol",
            "低波动": "low_vol",
            "主题行情": "theme_market",
            "宽基行情": "broad_market",
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(selected_df, metrics, regime_counts, folds, equity, regime_perf, wf_cfg.initial_capital)

    try:
        from AStockQuant.llm.strategy_reporter import StrategyReporter
        reporter = StrategyReporter()
        if not equity.empty:
            strat_stats = perf_stats(equity["equity"], wf_cfg.initial_capital)
        else:
            strat_stats = {"total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "ann_return": 0.0}
        llm_report = reporter.generate_report(
            strategy_stats=strat_stats,
            benchmark_stats=benchmarks.to_dict("records") if not benchmarks.empty else [],
            regime_performance=regime_perf,
            selected_factors=selected_df,
            folds=folds,
            config={
                "rps_blend_weight": wf_cfg.rps_blend_weight,
                "top_n": wf_cfg.top_n,
                "use_regime_gating": wf_cfg.use_regime_gating,
                "train_months": wf_cfg.train_months,
                "test_months": wf_cfg.test_months,
            },
        )
        if llm_report:
            report_path = REPORT_DIR / "llm_strategy_report.md"
            reporter.save_report(llm_report, str(report_path))
    except Exception as e:
        print(f"\n[LLM] report generation skipped: {e}")

    if not benchmarks.empty:
        print("\nBenchmark comparison:")
        print(benchmarks.to_string(index=False))
    if not robustness.empty:
        print("\nRobustness grid top 10:")
        print(robustness.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
