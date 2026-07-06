# -*- coding: utf-8 -*-
"""
rps_momentum_strategy.py — RPS 主导 ETF 轮动策略

核心改造：
1. RPS 主导选股：0.6×20日收益 + 0.4×120日收益（与基准同公式）
2. 多因子仅做风控过滤（不参与打分）：排除极端负值
3. Regime gating 降仓避险（但不过度，保留趋势捕获能力）
4. 交易层降换手：持仓缓冲 + 最短持有 + 增量调仓

目标（扣成本后）：收益接近+69.48%，回撤<-18.98%，换手<0.5
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "rps_momentum_strategy"
DATA_DIR = ROOT / "data_cache"
FACTOR_PANEL_PATH = ROOT / "reports" / "advanced_factor_research" / "factor_panel.csv"


@dataclass
class StrategyConfig:
    # RPS 权重
    rps_weight_20d: float = 0.6
    rps_weight_120d: float = 0.4
    # 选股
    top_n: int = 10
    min_history: int = 121
    # 风控过滤
    use_factor_filter: bool = True
    filter_exclude_bottom_pct: float = 0.15  # 排除因子值最差的15%
    filter_factors: tuple = ("sent_combined_score", "sector_breadth", "belief_delta")
    # Regime gating
    use_regime_gating: bool = True
    regime_down_exposure: float = 0.5       # 下跌市场降到50%（不过度空仓）
    regime_down_highvol_exposure: float = 0.2  # 下跌+高波动降到20%
    regime_highvol_mixed_exposure: float = 0.7
    regime_normal_exposure: float = 1.0
    # 交易层
    rebalance_freq: int = 5
    buffer_rank: int = 10
    min_hold_periods: int = 5
    replace_score_margin: float = 0.05
    # 成本
    slippage: float = 0.001
    commission: float = 0.0003
    min_lot: int = 100
    initial_capital: float = 1_000_000.0
    # 回测范围
    start: str = "2021-01-01"
    end: str = "2026-03-31"


def load_all_data(cfg: StrategyConfig) -> Dict[str, pd.DataFrame]:
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
        if len(df) >= cfg.min_history:
            all_data[symbol] = df
    return all_data


def compute_rps_scores(all_data: Dict[str, pd.DataFrame], as_of: pd.Timestamp, cfg: StrategyConfig) -> Dict[str, float]:
    scores = {}
    for sym, df in all_data.items():
        hist = df[df.index <= as_of]
        if len(hist) < cfg.min_history:
            continue
        r20 = hist["close"].pct_change(20).iloc[-1]
        r120 = hist["close"].pct_change(120).iloc[-1]
        if np.isnan(r20) or np.isnan(r120):
            continue
        scores[sym] = cfg.rps_weight_20d * r20 + cfg.rps_weight_120d * r120
    return scores


def load_factor_panel() -> Optional[pd.DataFrame]:
    if not FACTOR_PANEL_PATH.exists():
        print(f"[WARN] factor_panel.csv not found at {FACTOR_PANEL_PATH}, factor filter disabled")
        return None
    panel = pd.read_csv(FACTOR_PANEL_PATH, dtype={"symbol": str})
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def get_factor_filter_set(panel: pd.DataFrame, as_of: pd.Timestamp, cfg: StrategyConfig) -> set:
    if not cfg.use_factor_filter or panel is None:
        return set()
    row_date = as_of.strftime("%Y-%m-%d")
    sub = panel[panel["date"] == as_of]
    if sub.empty:
        return set()
    exclude = set()
    for f in cfg.filter_factors:
        if f not in sub.columns:
            continue
        vals = sub[["symbol", f]].dropna()
        if len(vals) < 10:
            continue
        threshold = vals[f].quantile(cfg.filter_exclude_bottom_pct)
        exclude.update(vals[vals[f] <= threshold]["symbol"].astype(str).str.zfill(6).tolist())
    return exclude


def classify_regime(as_of: pd.Timestamp, all_data: Dict[str, pd.DataFrame], market_proxy: str = "510300") -> Dict[str, str]:
    market = all_data.get(market_proxy)
    if market is None or market.empty:
        market = next(iter(all_data.values()))
    m = market[market.index <= as_of]
    if len(m) < 120:
        return {"trend": "range", "vol": "mid_vol", "leadership": "mixed_market"}

    close = m["close"]
    ma60 = close.rolling(60).mean().iloc[-1]
    ret60 = close.pct_change(60).iloc[-1]
    if close.iloc[-1] > ma60 and ret60 > 0.05:
        trend = "trend_bull"
    elif close.iloc[-1] < ma60 and ret60 < -0.05:
        trend = "down"
    else:
        trend = "range"

    ret = close.pct_change()
    vol20 = ret.rolling(20).std().iloc[-1]
    vol_hist = ret.rolling(20).std().tail(120).dropna()
    vol_pct = float((vol_hist <= vol20).mean()) if len(vol_hist) >= 30 else 0.5
    vol = "high_vol" if vol_pct >= 0.7 else "low_vol" if vol_pct <= 0.3 else "mid_vol"

    returns_20 = []
    for sym, df in all_data.items():
        d = df[df.index <= as_of]
        if len(d) >= 21:
            r20 = d["close"].pct_change(20).iloc[-1]
            if not np.isnan(r20):
                returns_20.append(r20)
    breadth = sum(1 for r in returns_20 if r > 0) / max(1, len(returns_20))

    from AStockQuant.layers.layer3_sector import SectorLayer
    sector_returns: Dict[str, List[float]] = {}
    for sym, df in all_data.items():
        d = df[df.index <= as_of]
        if len(d) >= 21:
            r20 = d["close"].pct_change(20).iloc[-1]
            if not np.isnan(r20):
                sector = SectorLayer.get_sector(sym)[0]
                sector_returns.setdefault(sector, []).append(float(r20))
    sector_mean = {k: float(np.mean(v)) for k, v in sector_returns.items() if v}
    dispersion = float(np.std(list(sector_mean.values()))) if sector_mean else 0.0
    top_sector_ret = max(sector_mean.values()) if sector_mean else 0.0
    market_ret = float(ret60) if not np.isnan(ret60) else 0.0

    if dispersion > 0.035 and top_sector_ret - market_ret > 0.03:
        leadership = "theme_market"
    elif breadth >= 0.60 and dispersion < 0.05:
        leadership = "broad_market"
    else:
        leadership = "mixed_market"

    return {"trend": trend, "vol": vol, "leadership": leadership}


def get_regime_exposure(regime: Dict[str, str], cfg: StrategyConfig) -> float:
    if not cfg.use_regime_gating:
        return cfg.regime_normal_exposure
    trend = regime.get("trend", "range")
    vol = regime.get("vol", "mid_vol")
    leadership = regime.get("leadership", "mixed_market")
    if trend == "down" and vol == "high_vol":
        return cfg.regime_down_highvol_exposure
    if trend == "down":
        return cfg.regime_down_exposure
    if vol == "high_vol" and leadership == "mixed_market":
        return cfg.regime_highvol_mixed_exposure
    return cfg.regime_normal_exposure


def price_at(df: pd.DataFrame, date: pd.Timestamp, field: str = "open") -> Optional[float]:
    d = df[df.index >= date]
    if d.empty:
        return None
    return float(d[field].iloc[0])


def next_trade_date(all_dates: List[pd.Timestamp], signal_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    for d in all_dates:
        if d > signal_date:
            return d
    return None


def run_backtest(all_data: Dict[str, pd.DataFrame], panel: Optional[pd.DataFrame], cfg: StrategyConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_trade_dates = sorted(set().union(*[set(df.index) for df in all_data.values()]))
    dates = pd.DatetimeIndex(all_trade_dates)
    dates = dates[(dates >= pd.Timestamp(cfg.start)) & (dates <= pd.Timestamp(cfg.end))]
    signal_dates = list(dates[::cfg.rebalance_freq])

    cash = cfg.initial_capital
    positions: Dict[str, int] = {}
    hold_periods: Dict[str, int] = {}
    equity_rows = []
    trades_log = []

    for signal_dt in signal_dates:
        trade_dt = next_trade_date(all_trade_dates, signal_dt)
        if trade_dt is None:
            continue

        # 1. Compute RPS scores
        rps_scores = compute_rps_scores(all_data, signal_dt, cfg)
        if len(rps_scores) < cfg.top_n:
            continue

        # 2. Factor filter (exclude worst)
        exclude_set = get_factor_filter_set(panel, signal_dt, cfg) if panel is not None else set()
        filtered = {s: v for s, v in rps_scores.items() if s not in exclude_set}
        if len(filtered) < cfg.top_n:
            filtered = rps_scores  # fallback if filter too aggressive

        # 3. Rank
        ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        top_core = [s for s, _ in ranked[:cfg.top_n]]
        top_candidates = [s for s, _ in ranked[:cfg.top_n + cfg.buffer_rank]]
        score_map = {s: v for s, v in ranked}
        rank_map = {s: i + 1 for i, (s, _) in enumerate(ranked)}

        # 4. Regime gating
        regime = classify_regime(signal_dt, all_data)
        exposure = get_regime_exposure(regime, cfg)

        # 5. Mark-to-open current equity
        equity = cash
        for sym, shares in positions.items():
            px = price_at(all_data[sym], trade_dt, "open")
            if px is not None:
                equity += shares * px
        target_gross = equity * exposure
        target_value = target_gross / max(1, cfg.top_n) if exposure > 0 else 0

        # 6. Hold buffer: retain positions still in top_n+buffer or too young to sell
        retained = []
        for sym in list(positions):
            rank = rank_map.get(sym, 99999)
            age = hold_periods.get(sym, 0)
            if exposure > 0 and (rank <= cfg.top_n + cfg.buffer_rank or age < cfg.min_hold_periods):
                retained.append(sym)

        picks = list(retained)
        for sym in top_core:
            if sym not in picks:
                picks.append(sym)
            if len(picks) >= cfg.top_n:
                break

        # Fill remaining slots with score margin
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

        old_symbols = set(positions)
        new_symbols = set(picks)
        turnover = 1.0 - len(old_symbols & new_symbols) / max(1, len(old_symbols | new_symbols))

        # 7. Sell positions not in picks
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
            trades_log.append({"date": trade_dt, "action": "sell", "symbol": sym, "shares": positions[sym], "price": sell_px})
            del positions[sym]
            hold_periods.pop(sym, None)

        # 8. Buy/adjust positions (incremental)
        for sym in picks:
            px = price_at(all_data[sym], trade_dt, "open")
            if px is None or px <= 0:
                continue
            buy_px = px * (1 + cfg.slippage)
            current_shares = positions.get(sym, 0)
            desired_shares = int(target_value / buy_px / cfg.min_lot) * cfg.min_lot
            delta = desired_shares - current_shares
            if abs(delta) < cfg.min_lot:
                continue
            if delta > 0:
                gross = delta * buy_px
                cost = gross * cfg.commission
                if gross + cost <= cash:
                    cash -= gross + cost
                    positions[sym] = current_shares + delta
                    trades_log.append({"date": trade_dt, "action": "buy", "symbol": sym, "shares": delta, "price": buy_px})
            else:
                sell_shares = abs(delta)
                sell_px = px * (1 - cfg.slippage)
                proceeds = sell_shares * sell_px
                cost = proceeds * cfg.commission
                cash += proceeds - cost
                positions[sym] = current_shares - sell_shares
                trades_log.append({"date": trade_dt, "action": "sell", "symbol": sym, "shares": sell_shares, "price": sell_px})
                if positions[sym] <= 0:
                    del positions[sym]
                    hold_periods.pop(sym, None)

        # 9. Update hold periods
        for sym in list(positions):
            hold_periods[sym] = hold_periods.get(sym, 0) + 1

        # 10. Close equity
        close_equity = cash
        for sym, shares in positions.items():
            px = price_at(all_data[sym], trade_dt, "close") or price_at(all_data[sym], trade_dt, "open")
            if px is not None:
                close_equity += shares * px

        equity_rows.append({
            "signal_date": signal_dt.strftime("%Y-%m-%d"),
            "trade_date": trade_dt.strftime("%Y-%m-%d"),
            "equity": close_equity,
            "cash": cash,
            "positions": len(positions),
            "turnover": turnover,
            "regime": f"{regime['trend']}_{regime['vol']}_{regime['leadership']}",
            "exposure": exposure,
        })

    return pd.DataFrame(equity_rows), pd.DataFrame(trades_log)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    cummax = equity.cummax()
    dd = equity / cummax - 1
    return float(dd.min())


def perf_stats(equity: pd.Series, initial_capital: float, rebalance_freq: int = 5) -> Dict[str, float]:
    if equity.empty:
        return {"total_return": 0.0, "max_drawdown": 0.0, "ann_return": 0.0, "sharpe": 0.0, "avg_turnover": 0.0}
    ret = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / initial_capital - 1)
    ann_return = float((1 + total_return) ** (252 / max(1, len(equity) * rebalance_freq)) - 1)
    sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(252 / rebalance_freq)) if len(ret) > 2 and ret.std(ddof=1) > 0 else 0.0
    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown(equity),
        "ann_return": ann_return,
        "sharpe": sharpe,
    }


def benchmark_rps_momentum(all_data: Dict[str, pd.DataFrame], signal_dates: List[pd.Timestamp], trade_dates: List[pd.Timestamp], cfg: StrategyConfig) -> pd.Series:
    """简单 RPS 动量基准（无成本，用 signal_date 算 RPS，trade_date 开盘成交，无未来函数）"""
    cash = cfg.initial_capital
    positions: Dict[str, int] = {}
    vals = []
    for sig_dt, trade_dt in zip(signal_dates, trade_dates):
        scores = compute_rps_scores(all_data, sig_dt, cfg)
        picks = [s for s, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:cfg.top_n]]
        equity = cash
        for sym, shares in positions.items():
            px = price_at(all_data[sym], trade_dt, "open")
            if px is not None:
                equity += shares * px
        target = equity / max(1, len(picks))
        for sym in list(positions):
            if sym not in picks:
                px = price_at(all_data[sym], trade_dt, "open")
                if px is not None:
                    cash += positions[sym] * px
                del positions[sym]
        for sym in picks:
            px = price_at(all_data[sym], trade_dt, "open")
            if px is None or px <= 0:
                continue
            desired = int(target / px / cfg.min_lot) * cfg.min_lot
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
            px = price_at(all_data[sym], trade_dt, "close") or price_at(all_data[sym], trade_dt, "open")
            if px is not None:
                close_equity += shares * px
        vals.append(close_equity)
    return pd.Series(vals, index=trade_dates)


def main() -> int:
    parser = argparse.ArgumentParser(description="RPS 主导 ETF 轮动策略")
    parser.add_argument("--no-filter", action="store_true", help="禁用多因子风控过滤")
    parser.add_argument("--no-regime", action="store_true", help="禁用 regime gating")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--buffer-rank", type=int, default=None)
    parser.add_argument("--min-hold", type=int, default=None)
    parser.add_argument("--replace-margin", type=float, default=None)
    parser.add_argument("--down-exposure", type=float, default=None)
    parser.add_argument("--down-highvol-exposure", type=float, default=None)
    parser.add_argument("--slippage", type=float, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    cfg = StrategyConfig()
    if args.no_filter:
        cfg.use_factor_filter = False
    if args.no_regime:
        cfg.use_regime_gating = False
    if args.top_n is not None:
        cfg.top_n = args.top_n
    if args.buffer_rank is not None:
        cfg.buffer_rank = args.buffer_rank
    if args.min_hold is not None:
        cfg.min_hold_periods = args.min_hold
    if args.replace_margin is not None:
        cfg.replace_score_margin = args.replace_margin
    if args.down_exposure is not None:
        cfg.regime_down_exposure = args.down_exposure
    if args.down_highvol_exposure is not None:
        cfg.regime_down_highvol_exposure = args.down_highvol_exposure
    if args.slippage is not None:
        cfg.slippage = args.slippage
    if args.start is not None:
        cfg.start = args.start
    if args.end is not None:
        cfg.end = args.end

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== RPS Momentum Strategy ===")
    print(f"  top_n={cfg.top_n}, buffer_rank={cfg.buffer_rank}, min_hold={cfg.min_hold_periods}")
    print(f"  filter={cfg.use_factor_filter}, regime={cfg.use_regime_gating}")
    print(f"  slippage={cfg.slippage}, commission={cfg.commission}")
    print(f"  range={cfg.start} ~ {cfg.end}")

    all_data = load_all_data(cfg)
    print(f"  loaded {len(all_data)} ETFs")

    panel = load_factor_panel() if cfg.use_factor_filter else None
    if panel is not None:
        print(f"  factor panel loaded: {len(panel)} rows")

    equity_df, trades_df = run_backtest(all_data, panel, cfg)

    if equity_df.empty:
        print("ERROR: no equity data")
        return 1

    equity_series = pd.Series(equity_df["equity"].values, index=pd.to_datetime(equity_df["trade_date"]))
    stats = perf_stats(equity_series, cfg.initial_capital, cfg.rebalance_freq)
    avg_turnover = float(equity_df["turnover"].mean()) if not equity_df.empty else 0.0

    # Benchmark: simple RPS momentum (no cost, fair: signal_date RPS + trade_date open)
    signal_dates_list = pd.to_datetime(equity_df["signal_date"]).tolist()
    trade_dates_list = pd.to_datetime(equity_df["trade_date"]).tolist()
    bench_equity = benchmark_rps_momentum(all_data, signal_dates_list, trade_dates_list, cfg)
    bench_stats = perf_stats(bench_equity, cfg.initial_capital, cfg.rebalance_freq)

    print(f"\n=== Results ===")
    print(f"  Strategy (after cost):")
    print(f"    total_return = {stats['total_return']:.2%}")
    print(f"    max_drawdown = {stats['max_drawdown']:.2%}")
    print(f"    ann_return   = {stats['ann_return']:.2%}")
    print(f"    sharpe       = {stats['sharpe']:.4f}")
    print(f"    avg_turnover = {avg_turnover:.4f}")
    print(f"  Benchmark (simple RPS, no cost):")
    print(f"    total_return = {bench_stats['total_return']:.2%}")
    print(f"    max_drawdown = {bench_stats['max_drawdown']:.2%}")
    print(f"    sharpe       = {bench_stats['sharpe']:.4f}")
    print(f"  Excess vs benchmark: {stats['total_return'] - bench_stats['total_return']:.2%}")

    # Target check
    targets = {
        "return > 0": stats["total_return"] > 0,
        "return close to benchmark (>50% of bench)": stats["total_return"] > bench_stats["total_return"] * 0.5,
        "drawdown < -18.98%": stats["max_drawdown"] > -0.1898,
        "turnover < 0.5": avg_turnover < 0.5,
    }
    print(f"\n=== Target Check ===")
    for t, ok in targets.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {t}")

    # Save
    equity_df.to_csv(REPORT_DIR / "equity.csv", index=False, encoding="utf-8-sig")
    trades_df.to_csv(REPORT_DIR / "trades.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "config.json").write_text(json.dumps(
        {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
        ensure_ascii=False, indent=2, default=str
    ), encoding="utf-8")
    print(f"\n  saved to {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
