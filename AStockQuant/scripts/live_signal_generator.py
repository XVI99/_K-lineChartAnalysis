# -*- coding: utf-8 -*-
"""
实盘信号生成器
==============
每日运行: 计算最新因子 → 综合评分 → TOP N选股 → 调仓指令 → 风控预警

输入:
- data_cache/ 下所有ETF日线数据
- (可选) 当前持仓文件 current_holdings.csv

输出:
- 控制台: 当日信号摘要
- signals/YYYY-MM-DD_signals.csv: 调仓信号
- signals/YYYY-MM-DD_alerts.json: 风控预警

用法:
    python live_signal_generator.py                    # 使用最新数据
    python live_signal_generator.py --date 2026-06-20  # 指定日期
    python live_signal_generator.py --holdings my_holdings.csv  # 含当前持仓
"""

import os
import sys
import json
import warnings
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 禁用akshare网络调用
import layers.layer4_capital as _l4_mod
_l4_mod._AK = False

from layers.layer1_macro import MacroLayer
from layers.layer3_sector import SectorLayer
from layers.layer4_capital import CapitalLayer
from layers.layer5_sentiment import SentimentLayer
from layers.layer6_price_vol import PriceVolumeLayer
from layers.layer7_technical import TechnicalLayer
from layers.layer8_micro import BeliefLayer

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================== 配置 ====================
DATA_DIR = os.path.join(PROJECT_ROOT, "data_cache")
SIGNAL_DIR = os.path.join(PROJECT_ROOT, "signals")
MARKET_PROXY = "510510"

# 策略参数 (复用v4最优配置)
PORTFOLIO_FACTORS = [
    "pv_volume_trend",
    "sent_combined_score",
    "pv_turnover_change",
    "pv_price_accel",
    "pv_vol_price_divergence",
    "sector_combined_score",
]

FACTOR_DIRECTION = {
    "pv_volume_trend": 1,
    "sent_combined_score": 1,
    "pv_turnover_change": 1,
    "pv_price_accel": -1,
    "pv_vol_price_divergence": 1,
    "sector_combined_score": 1,
}

MAX_POSITIONS = 10
MAX_WEIGHT = 0.20
LOOKBACK_MIN = 120

# 风控参数 (复用回测最优)
STOP_LOSS_PCT = 0.08
TAKE_PROFIT_PCT = 0.20
TRAILING_STOP_PCT = 0.06
TRAILING_ACTIVATE = 0.10


# ==================== 1. 数据加载 ====================
def load_all_data() -> Dict[str, pd.DataFrame]:
    """加载所有ETF日线数据"""
    all_data = {}
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    for f in sorted(files):
        code = f.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f))
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            required = ["open", "high", "low", "close", "volume"]
            if all(c in df.columns for c in required):
                df = df[required].dropna()
                if len(df) >= LOOKBACK_MIN:
                    all_data[code] = df
        except Exception:
            pass
    return all_data


# ==================== 2. 因子计算 ====================
def compute_factors(
    all_data: Dict[str, pd.DataFrame],
    as_of_date: str,
) -> pd.DataFrame:
    """
    计算指定日期的所有因子

    Returns: DataFrame[symbol, factor1, factor2, ...]
    """
    as_of_ts = pd.Timestamp(as_of_date)

    # 实例化各层
    layers = {
        "macro": MacroLayer(),
        "sector": SectorLayer(),
        "capital": CapitalLayer(),
        "sentiment": SentimentLayer(),
        "price_vol": PriceVolumeLayer(),
        "technical": TechnicalLayer(),
        "belief": BeliefLayer(),
    }

    market_df = all_data.get(MARKET_PROXY)
    if market_df is None:
        market_df = all_data.get("510300")

    # 筛选有足够数据的ETF
    eligible = []
    for sym, df in all_data.items():
        df_trunc = df[df.index <= as_of_ts]
        if len(df_trunc) >= LOOKBACK_MIN:
            eligible.append(sym)

    if len(eligible) < 10:
        print(f"  警告: 仅有 {len(eligible)} 只ETF满足数据要求")
        return pd.DataFrame()

    # 预计算跨标的20日收益率
    all_returns = {}
    for sym in eligible:
        df_trunc = all_data[sym][all_data[sym].index <= as_of_ts]
        if len(df_trunc) >= 21:
            all_returns[sym] = float(df_trunc["close"].pct_change(20).iloc[-1])
        else:
            all_returns[sym] = None

    market_trunc = market_df[market_df.index <= as_of_ts] if market_df is not None else None

    ctx_base = {
        "name": "",
        "all_sector_returns": all_returns,
        "market_prices_df": market_trunc,
    }

    # 逐ETF计算因子
    records = []
    for sym in eligible:
        df_trunc = all_data[sym][all_data[sym].index <= as_of_ts]
        ctx = dict(ctx_base)

        features = {}
        layer_ctx = dict(ctx)
        for layer_name, layer in layers.items():
            try:
                feats = layer.extract_features(sym, df_trunc, layer_ctx, as_of_date=as_of_date)
                if feats:
                    features.update(feats)
                    layer_ctx.update(feats)
            except Exception:
                pass

        record = {"symbol": sym}
        for factor in PORTFOLIO_FACTORS:
            val = features.get(factor, np.nan)
            if isinstance(val, bool):
                val = 1.0 if val else 0.0
            elif isinstance(val, str):
                val = np.nan
            record[factor] = val
        records.append(record)

    return pd.DataFrame(records)


# ==================== 3. 综合评分 ====================
def compute_composite_score(factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算综合评分: 因子方向调整 → 截面排名百分位 → 等权聚合

    Returns: DataFrame with 'composite_score' column
    """
    result = factors_df.copy()

    # 因子方向调整
    for f in PORTFOLIO_FACTORS:
        if f not in result.columns:
            continue
        direction = FACTOR_DIRECTION.get(f, 1)
        if direction == -1:
            result[f] = -result[f]

    # 截面排名百分位
    for f in PORTFOLIO_FACTORS:
        if f not in result.columns:
            continue
        result[f"{f}_rank"] = result[f].rank(pct=True)

    # 等权聚合
    rank_cols = [f"{f}_rank" for f in PORTFOLIO_FACTORS if f"{f}_rank" in result.columns]
    result["composite_score"] = result[rank_cols].mean(axis=1)

    # 排序
    result = result.sort_values("composite_score", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    return result


# ==================== 4. 选股 ====================
def select_top(scored: pd.DataFrame, n: int = MAX_POSITIONS) -> pd.DataFrame:
    """选TOP N, 分配等权"""
    top = scored.head(n).copy()
    top["target_weight"] = 1.0 / n
    return top


# ==================== 5. 持仓对比 ====================
def compare_holdings(
    target: pd.DataFrame,
    current_holdings: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    对比目标持仓与当前持仓, 生成调仓指令

    Returns: DataFrame[date, symbol, action, weight, score, reason]
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    signals = []

    target_symbols = set(target["symbol"].values)

    if current_holdings is not None and len(current_holdings) > 0:
        current_symbols = set(current_holdings["symbol"].values)

        # 卖出: 当前持有但不在目标中
        for sym in current_symbols - target_symbols:
            signals.append({
                "date": today_str,
                "symbol": sym,
                "action": "SELL",
                "weight": 0.0,
                "score": 0.0,
                "reason": "调仓卖出: 不在目标TOP10",
            })

        # 持有: 当前持有且在目标中
        for sym in current_symbols & target_symbols:
            target_row = target[target["symbol"] == sym].iloc[0]
            signals.append({
                "date": today_str,
                "symbol": sym,
                "action": "HOLD",
                "weight": round(target_row["target_weight"], 4),
                "score": round(target_row["composite_score"], 4),
                "reason": "继续持有",
            })

        # 买入: 在目标中但当前未持有
        for _, row in target.iterrows():
            if row["symbol"] not in current_symbols:
                signals.append({
                    "date": today_str,
                    "symbol": row["symbol"],
                    "action": "BUY",
                    "weight": round(row["target_weight"], 4),
                    "score": round(row["composite_score"], 4),
                    "reason": "新入选TOP10",
                })
    else:
        # 无当前持仓, 全部标记为BUY
        for _, row in target.iterrows():
            signals.append({
                "date": today_str,
                "symbol": row["symbol"],
                "action": "BUY",
                "weight": round(row["target_weight"], 4),
                "score": round(row["composite_score"], 4),
                "reason": "初始建仓",
            })

    return pd.DataFrame(signals)


# ==================== 6. 风控预警 ====================
def generate_alerts(
    all_data: Dict[str, pd.DataFrame],
    current_holdings: Optional[pd.DataFrame],
    as_of_date: str,
) -> List[Dict]:
    """
    生成风控预警: 止损/止盈/追踪止损

    Args:
        current_holdings: 含 symbol, entry_price, entry_date, shares 列
    """
    alerts = []
    if current_holdings is None or len(current_holdings) == 0:
        return alerts

    as_of_ts = pd.Timestamp(as_of_date)

    for _, row in current_holdings.iterrows():
        sym = str(row["symbol"])
        entry_price = float(row["entry_price"])
        entry_date = str(row.get("entry_date", ""))

        if sym not in all_data:
            continue

        df = all_data[sym]
        df_trunc = df[df.index <= as_of_ts]
        if len(df_trunc) == 0:
            continue

        latest = df_trunc.iloc[-1]
        current_price = latest["close"]
        pnl_pct = (current_price / entry_price - 1) * 100

        # 止损预警
        stop_price = entry_price * (1 - STOP_LOSS_PCT)
        if current_price <= stop_price:
            alerts.append({
                "symbol": sym,
                "type": "STOP_LOSS",
                "severity": "HIGH",
                "entry_price": round(entry_price, 4),
                "current_price": round(current_price, 4),
                "stop_price": round(stop_price, 4),
                "pnl_pct": round(pnl_pct, 2),
                "message": f"{sym} 触发止损! 当前价{current_price:.4f} ≤ 止损价{stop_price:.4f} ({pnl_pct:+.2f}%)",
            })

        # 止盈预警
        tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
        if current_price >= tp_price:
            alerts.append({
                "symbol": sym,
                "type": "TAKE_PROFIT",
                "severity": "MEDIUM",
                "entry_price": round(entry_price, 4),
                "current_price": round(current_price, 4),
                "tp_price": round(tp_price, 4),
                "pnl_pct": round(pnl_pct, 2),
                "message": f"{sym} 触发止盈! 当前价{current_price:.4f} ≥ 止盈价{tp_price:.4f} ({pnl_pct:+.2f}%)",
            })

        # 追踪止损预警
        if pnl_pct >= TRAILING_ACTIVATE * 100:
            # 计算持仓期间最高价
            if entry_date:
                period_df = df_trunc[df_trunc.index >= pd.Timestamp(entry_date)]
            else:
                period_df = df_trunc
            if len(period_df) > 0:
                highest = period_df["high"].max()
                trail_price = highest * (1 - TRAILING_STOP_PCT)
                if current_price <= trail_price * 1.02:  # 2%接近预警
                    alerts.append({
                        "symbol": sym,
                        "type": "TRAILING_STOP",
                        "severity": "LOW",
                        "entry_price": round(entry_price, 4),
                        "current_price": round(current_price, 4),
                        "highest_price": round(highest, 4),
                        "trail_price": round(trail_price, 4),
                        "pnl_pct": round(pnl_pct, 2),
                        "message": f"{sym} 接近追踪止损! 最高价{highest:.4f} 追踪价{trail_price:.4f} 当前{current_price:.4f}",
                    })

    return alerts


# ==================== 7. 信号输出 ====================
def output_signals(
    signals: pd.DataFrame,
    scored: pd.DataFrame,
    alerts: List[Dict],
    as_of_date: str,
    market_info: Dict,
):
    """多格式输出信号"""
    os.makedirs(SIGNAL_DIR, exist_ok=True)

    today_str = as_of_date[:10] if len(as_of_date) > 10 else as_of_date

    # === 控制台输出 ===
    print("\n" + "=" * 80)
    print(f"  实盘信号 — {today_str}")
    print("=" * 80)

    # 市场状态
    print(f"\n  市场状态: {market_info.get('regime', 'N/A')}")
    print(f"  趋势强度: {market_info.get('trend', 'N/A')}")
    print(f"  数据日期: {market_info.get('data_date', 'N/A')}")

    # TOP 10
    print(f"\n  ┌─ 目标持仓 TOP {MAX_POSITIONS} ─────────────────────────────┐")
    print(f"  │ {'排名':<4} {'代码':<8} {'名称':<12} {'得分':>8} {'权重':>8} │")
    print(f"  ├{'─'*48}┤")
    for _, row in scored.head(MAX_POSITIONS).iterrows():
        sym = row["symbol"]
        name = market_info.get("names", {}).get(sym, "")
        print(f"  │ {row['rank']:<4} {sym:<8} {name:<12} {row['composite_score']:>8.4f} {row['target_weight']:>8.2%} │")
    print(f"  └{'─'*48}┘")

    # 调仓指令
    if len(signals) > 0:
        buys = signals[signals["action"] == "BUY"]
        sells = signals[signals["action"] == "SELL"]
        holds = signals[signals["action"] == "HOLD"]

        print(f"\n  [调仓指令] BUY={len(buys)} SELL={len(sells)} HOLD={len(holds)}")

        if len(sells) > 0:
            print(f"\n  [卖出]:")
            for _, row in sells.iterrows():
                print(f"     {row['symbol']:<8} {row['reason']}")

        if len(buys) > 0:
            print(f"\n  [买入]:")
            for _, row in buys.iterrows():
                print(f"     {row['symbol']:<8} 权重{row['weight']:.2%} 得分{row['score']:.4f}")

    # 风控预警
    if alerts:
        print(f"\n  [风控预警] ({len(alerts)}条):")
        for a in alerts:
            icon = {"HIGH": "[!!!]", "MEDIUM": "[!!]", "LOW": "[!]"}.get(a["severity"], "[?]")
            print(f"     {icon} [{a['type']}] {a['message']}")
    else:
        print(f"\n  [OK] 无风控预警")

    # === CSV输出 ===
    csv_path = os.path.join(SIGNAL_DIR, f"{today_str}_signals.csv")
    signals.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 完整排名
    rank_path = os.path.join(SIGNAL_DIR, f"{today_str}_full_ranking.csv")
    scored.to_csv(rank_path, index=False, encoding="utf-8-sig")

    # === JSON输出 ===
    json_path = os.path.join(SIGNAL_DIR, f"{today_str}_alerts.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": today_str,
            "market_info": market_info,
            "top_picks": scored.head(MAX_POSITIONS)[["symbol", "composite_score", "target_weight"]].to_dict("records"),
            "signals": signals.to_dict("records"),
            "alerts": alerts,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  信号已保存:")
    print(f"    调仓信号: {csv_path}")
    print(f"    完整排名: {rank_path}")
    print(f"    风控预警: {json_path}")


# ==================== 8. 市场信息 ====================
def get_market_info(all_data: Dict[str, pd.DataFrame], as_of_date: str) -> Dict:
    """获取市场状态信息"""
    market_df = all_data.get(MARKET_PROXY)
    if market_df is None:
        return {"regime": "N/A", "trend": "N/A", "data_date": as_of_date}

    df_trunc = market_df[market_df.index <= pd.Timestamp(as_of_date)]
    if len(df_trunc) < 60:
        return {"regime": "N/A", "trend": "N/A", "data_date": as_of_date}

    close = df_trunc["close"]
    data_date = df_trunc.index[-1].strftime("%Y-%m-%d")

    # 趋势强度: 20日收益率
    trend = float(close.pct_change(20).iloc[-1] * 100) if len(close) >= 21 else 0

    # 市场状态: 均线系统
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    current = close.iloc[-1]

    if current > ma20 > ma60:
        regime = "BULL(牛市)"
    elif current < ma20 < ma60:
        regime = "BEAR(熊市)"
    else:
        regime = "NEUTRAL(震荡)"

    return {
        "regime": regime,
        "trend": f"{trend:+.2f}%",
        "data_date": data_date,
        "names": {},
    }


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description="实盘信号生成器")
    parser.add_argument("--date", type=str, default=None, help="指定日期 YYYY-MM-DD, 默认最新")
    parser.add_argument("--holdings", type=str, default=None, help="当前持仓CSV路径")
    parser.add_argument("--top", type=int, default=MAX_POSITIONS, help=f"选股数量 (默认{MAX_POSITIONS})")
    parser.add_argument("--advanced", action="store_true", help="启用仓位管理系统(买多少/何时买/何时卖)")
    args = parser.parse_args()

    print("=" * 80)
    print("  实盘信号生成器 — 因子计算 → 综合评分 → TOP N选股 → 调仓指令")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    all_data = load_all_data()
    print(f"  已加载 {len(all_data)} 只ETF")

    # 2. 确定日期
    if args.date:
        as_of_date = args.date
    else:
        # 使用数据中最新的日期
        latest_dates = []
        for df in all_data.values():
            if len(df) > 0:
                latest_dates.append(df.index[-1])
        as_of_date = max(latest_dates).strftime("%Y-%m-%d") if latest_dates else datetime.now().strftime("%Y-%m-%d")

    print(f"  信号日期: {as_of_date}")

    # 3. 计算因子
    print(f"\n[2/5] 计算因子...")
    factors_df = compute_factors(all_data, as_of_date)
    if len(factors_df) == 0:
        print("  错误: 因子计算失败")
        return
    print(f"  计算完成: {len(factors_df)} 只ETF")

    # 4. 综合评分
    print(f"\n[3/5] 综合评分...")
    scored = compute_composite_score(factors_df)

    # 5. 选股
    print(f"\n[4/5] 选股 TOP {args.top}...")
    target = select_top(scored, n=args.top)

    # 6. 加载当前持仓
    current_holdings = None
    current_positions = {}
    if args.holdings and os.path.exists(args.holdings):
        current_holdings = pd.read_csv(args.holdings)
        current_holdings["symbol"] = current_holdings["symbol"].astype(str)
        print(f"  当前持仓: {len(current_holdings)} 只")

        # 转换为PositionState
        from position_manager import PositionState
        for _, row in current_holdings.iterrows():
            sym = str(row["symbol"])
            current_positions[sym] = PositionState(
                symbol=sym,
                entry_date=str(row.get("entry_date", as_of_date)),
                entry_price=float(row["entry_price"]),
                shares=int(row.get("shares", 0)),
                total_shares=int(row.get("shares", 0)),
                cost=float(row.get("cost", float(row["entry_price"]) * int(row.get("shares", 0)))),
                highest_price=float(row.get("entry_price", 0)),
                highest_date=str(row.get("entry_date", as_of_date)),
                stop_loss_price=float(row["entry_price"]) * (1 - STOP_LOSS_PCT),
                atr_trail_price=0,
                entry_score=float(row.get("score", 0)),
            )
    else:
        print(f"  无当前持仓 (全新信号)")

    # 7. 决策 (基础模式 vs 仓位管理模式)
    if args.advanced:
        print(f"\n[5/5] 仓位管理决策 (买多少/何时买/何时卖)...")
        from position_manager import PositionManager, generate_portfolio_decisions, format_decisions_output

        pm = PositionManager(all_data)
        result = generate_portfolio_decisions(scored, pm, as_of_date, current_positions, max_positions=args.top)
        print(format_decisions_output(result, as_of_date))

        # 保存决策
        os.makedirs(SIGNAL_DIR, exist_ok=True)
        decisions_df = pd.DataFrame([{
            "date": as_of_date,
            "symbol": d.symbol,
            "action": d.action,
            "target_weight": d.target_weight,
            "entry_timing": d.entry_timing,
            "exit_reason": d.exit_reason,
            "exit_ratio": d.exit_ratio,
            "stop_price": d.stop_price,
            "score": d.score,
            "message": d.message,
        } for d in result["decisions"]])
        decisions_df.to_csv(os.path.join(SIGNAL_DIR, f"{as_of_date[:10]}_decisions.csv"), index=False, encoding="utf-8-sig")
    else:
        # 基础模式: 二分类信号
        signals = compare_holdings(target, current_holdings)
        alerts = generate_alerts(all_data, current_holdings, as_of_date)
        market_info = get_market_info(all_data, as_of_date)
        print(f"\n[5/5] 输出信号...")
        output_signals(signals, target, alerts, as_of_date, market_info)

    print("\n信号生成完成!")


if __name__ == "__main__":
    main()
