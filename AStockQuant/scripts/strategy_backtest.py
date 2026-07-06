# -*- coding: utf-8 -*-
"""
策略回测引擎 (事件驱动)
=======================
借鉴AlphaEvo架构: 逐bar迭代 + 持仓状态机 + T+1 + 止损止盈 + 仓位管理

管线: 组合信号 → 逐bar执行 → 持仓管理 → 权益曲线 → 绩效报告

与portfolio_construction.py(向量化)的区别:
- 向量化: 假设调仓日收盘买入→下一调仓日收盘卖出, 无盘中风控
- 事件驱动: 逐日执行, T+1次日开盘成交, 盘中止损止盈, 真实交易约束

输出: AStockQuant/reports/backtest/
"""

import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "backtest")
DATA_DIR = os.path.join(PROJECT_ROOT, "data_cache")

# ==================== 配置 ====================
# 回测参数
INITIAL_CAPITAL = 1_000_000    # 初始资金 100万
MAX_POSITIONS = 10             # 最大持仓数
MAX_WEIGHT = 0.20              # 单只上限20%
SLIPPAGE = 0.001               # 滑点0.1%
COMMISSION_RATE = 0.0003       # 佣金0.03%

# 止损止盈 (借鉴AlphaEvo)
STOP_LOSS_PCT = 0.08           # 固定止损 -8%
STOP_LOSS_ATR_MULT = 2.0       # ATR止损倍数
TAKE_PROFIT_PCT = 0.20         # 固定止盈 +20%
TRAILING_STOP_PCT = 0.06       # 追踪止损 从最高点回撤6%
TRAILING_STOP_ACTIVATE = 0.10  # 盈利10%后激活追踪止损

# ATR参数
ATR_PERIOD = 14

# 市场规则
T_PLUS_1 = True                # A股T+1 (ETF部分适用)


# ==================== 数据结构 ====================
@dataclass
class Position:
    """持仓状态"""
    symbol: str
    entry_date: str            # 入场日期
    entry_price: float         # 入场价(含滑点)
    shares: int                # 持仓股数
    cost: float                # 总成本(含佣金)
    highest_price: float       # 持仓期间最高价(追踪止损用)
    stop_loss_price: float     # 止损价
    take_profit_price: float   # 止盈价
    trailing_activated: bool   # 追踪止损是否激活

    def update_high(self, high: float):
        if high > self.highest_price:
            self.highest_price = high

    def check_stop_loss(self, low: float) -> bool:
        """检查是否触发止损"""
        return low <= self.stop_loss_price

    def check_take_profit(self, high: float) -> bool:
        """检查是否触发止盈"""
        return high >= self.take_profit_price

    def check_trailing_stop(self, low: float) -> bool:
        """检查追踪止损"""
        if not self.trailing_activated:
            if (self.highest_price / self.entry_price - 1) >= TRAILING_STOP_ACTIVATE:
                self.trailing_activated = True
            return False
        trail_price = self.highest_price * (1 - TRAILING_STOP_PCT)
        return low <= trail_price


@dataclass
class PortfolioState:
    """组合状态"""
    cash: float = INITIAL_CAPITAL
    positions: Dict[str, Position] = field(default_factory=dict)
    equity_curve: List[Dict] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)

    @property
    def total_equity(self) -> float:
        """总权益 = 现金 + 持仓市值"""
        position_value = sum(p.shares * p.highest_price for p in self.positions.values())
        return self.cash + position_value

    @property
    def n_positions(self) -> int:
        return len(self.positions)


# ==================== 1. 数据加载 ====================
def load_all_data() -> Dict[str, pd.DataFrame]:
    """加载所有ETF日线数据"""
    all_data = {}
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    print(f"[1/5] 加载数据: {len(files)} 个CSV文件")

    for f in sorted(files):
        code = f.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f))
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            required = ["open", "high", "low", "close", "volume"]
            if all(c in df.columns for c in required):
                df = df[required].dropna()
                if len(df) > 0:
                    # 预计算ATR
                    df["tr"] = np.maximum(
                        df["high"] - df["low"],
                        np.maximum(
                            abs(df["high"] - df["close"].shift(1)),
                            abs(df["low"] - df["close"].shift(1)),
                        ),
                    )
                    df["atr"] = df["tr"].rolling(ATR_PERIOD).mean()
                    all_data[code] = df
        except Exception:
            pass

    print(f"  成功加载 {len(all_data)} 只ETF")
    return all_data


# ==================== 2. 信号加载 ====================
def load_signals() -> pd.DataFrame:
    """加载组合构建模块生成的调仓信号"""
    signal_path = os.path.join(PROJECT_ROOT, "reports", "portfolio", "rebalance_signals.csv")
    if not os.path.exists(signal_path):
        print(f"  警告: 信号文件不存在 ({signal_path}), 使用默认策略生成")
        return None

    signals = pd.read_csv(signal_path)
    signals["date"] = pd.to_datetime(signals["date"])
    signals["symbol"] = signals["symbol"].astype(str)  # 统一为字符串
    return signals


def generate_default_signals(all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    如果没有预生成的信号文件, 用简化策略生成:
    每周选pv_volume_trend最高的TOP10等权
    """
    print("  [生成默认信号] pv_volume_trend TOP10 等权...")

    # 复用因子评估的panel构建
    SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)

    from factor_evaluation import (
        get_cross_section_dates,
        build_factor_panel,
    )

    all_dates_set = set()
    for df in all_data.values():
        all_dates_set.update(df.index)
    cross_dates = get_cross_section_dates(list(all_dates_set))

    panel = build_factor_panel(all_data, cross_dates)

    if "pv_volume_trend" not in panel.columns:
        print("  错误: pv_volume_trend因子不可用")
        return pd.DataFrame()

    # 每截面选TOP10
    signals = []
    for dt in sorted(panel["date"].unique()):
        sub = panel[panel["date"] == dt].copy()
        sub["rank"] = sub["pv_volume_trend"].rank(ascending=False)
        top10 = sub[sub["rank"] <= MAX_POSITIONS]
        for _, row in top10.iterrows():
            signals.append({
                "date": dt,
                "symbol": row["symbol"],
                "weight": 1.0 / MAX_POSITIONS,
                "score": row["pv_volume_trend"],
                "action": "BUY",
            })

    if signals:
        result = pd.DataFrame(signals)
        result["symbol"] = result["symbol"].astype(str)
        return result
    else:
        return pd.DataFrame(columns=["date", "symbol", "weight", "score", "action"])


# ==================== 3. 回测引擎 ====================
class BacktestEngine:
    """事件驱动回测引擎"""

    def __init__(self, all_data: Dict[str, pd.DataFrame]):
        self.all_data = all_data
        self.state = PortfolioState()

    def run(self, signals: pd.DataFrame) -> Dict:
        """
        执行回测

        Args:
            signals: 调仓信号表 [date, symbol, weight, action]

        Returns:
            绩效指标字典
        """
        print(f"\n[2/5] 回测引擎初始化...")
        print(f"  初始资金: {INITIAL_CAPITAL:,.0f}")
        print(f"  止损: -{STOP_LOSS_PCT*100}%, 止盈: +{TAKE_PROFIT_PCT*100}%")
        print(f"  追踪止损: -{TRAILING_STOP_PCT*100}% (盈利{TRAILING_STOP_ACTIVATE*100}%激活)")
        print(f"  滑点: {SLIPPAGE*100}%, 佣金: {COMMISSION_RATE*100}%")

        # 获取所有交易日
        all_dates = sorted(set().union(*[set(df.index) for df in self.all_data.values()]))
        all_dates = [d for d in all_dates if d >= pd.Timestamp("2024-01-02")]

        # 按日期组织信号
        signal_dates = sorted(signals["date"].unique())
        signal_map = {}
        for dt in signal_dates:
            sub = signals[signals["date"] == dt]
            buys = sub[sub["action"].isin(["BUY", "HOLD"])]
            signal_map[pd.Timestamp(dt)] = buys

        # 逐日迭代
        print(f"\n[3/5] 逐日回测 ({len(all_dates)} 个交易日)...")
        rebalance_count = 0
        total_bars = 0

        for day_idx, today in enumerate(all_dates):
            total_bars += 1

            # === 检查是否需要调仓 ===
            if today in signal_map:
                target_holdings = signal_map[today]
                self._rebalance(today, target_holdings)
                rebalance_count += 1

            # === 逐持仓检查止损止盈 ===
            closed_positions = []
            for sym, pos in list(self.state.positions.items()):
                if sym not in self.all_data:
                    continue
                df = self.all_data[sym]
                if today not in df.index:
                    continue

                bar = df.loc[today]
                pos.update_high(bar["high"])

                # 检查停牌 (成交量为0)
                if bar["volume"] == 0:
                    continue

                exit_price = None
                exit_reason = ""

                if pos.check_stop_loss(bar["low"]):
                    exit_price = pos.stop_loss_price
                    exit_reason = "STOP_LOSS"
                elif pos.check_take_profit(bar["high"]):
                    exit_price = pos.take_profit_price
                    exit_reason = "TAKE_PROFIT"
                elif pos.check_trailing_stop(bar["low"]):
                    exit_price = pos.highest_price * (1 - TRAILING_STOP_PCT)
                    exit_reason = "TRAILING_STOP"

                if exit_price is not None:
                    self._close_position(sym, today, exit_price, exit_reason)
                    closed_positions.append(sym)

            # 移除已平仓
            for sym in closed_positions:
                del self.state.positions[sym]

            # === 记录每日权益 ===
            self._record_equity(today)

            # 进度
            if (day_idx + 1) % 50 == 0:
                print(f"  进度: {day_idx+1}/{len(all_dates)} 日, "
                      f"持仓: {self.state.n_positions}, 权益: {self.state.total_equity:,.0f}")

        print(f"  回测完成: {total_bars} 个交易日, {rebalance_count} 次调仓")

        # 计算绩效
        return self._compute_performance()

    def _rebalance(self, today: pd.Timestamp, target: pd.DataFrame):
        """执行调仓: 卖出不在目标中的, 买入目标中的"""
        target_symbols = set(target["symbol"].values)

        # 卖出不在目标中的持仓
        for sym in list(self.state.positions.keys()):
            if sym not in target_symbols:
                if sym in self.all_data and today in self.all_data[sym].index:
                    bar = self.all_data[sym].loc[today]
                    if bar["volume"] > 0:
                        self._close_position(sym, today, bar["close"], "REBALANCE_SELL")
                del self.state.positions[sym]

        # 买入目标中的标的
        n_buy = min(len(target), MAX_POSITIONS)
        available_cash_per = self.state.cash / max(1, n_buy - self.state.n_positions)

        for _, row in target.iterrows():
            sym = row["symbol"]
            if sym in self.state.positions:
                continue
            if sym not in self.all_data:
                continue
            if today not in self.all_data[sym].index:
                continue

            df = self.all_data[sym]
            bar = df.loc[today]

            if bar["volume"] == 0:
                continue

            entry_price = bar["close"] * (1 + SLIPPAGE)
            weight = min(row["weight"], MAX_WEIGHT)
            max_cost = min(self.state.cash * weight, available_cash_per)
            cost_per_share = entry_price * (1 + COMMISSION_RATE)
            shares = int(max_cost / cost_per_share / 100) * 100
            if shares < 100:
                continue

            cost = shares * cost_per_share
            if cost > self.state.cash:
                continue

            # ATR止损价
            atr = bar.get("atr", entry_price * 0.02)
            if np.isnan(atr) or atr <= 0:
                atr = entry_price * 0.02
            atr_stop = entry_price - STOP_LOSS_ATR_MULT * atr
            pct_stop = entry_price * (1 - STOP_LOSS_PCT)
            stop_price = max(atr_stop, pct_stop)  # 取更宽松的止损

            # 开仓
            pos = Position(
                symbol=sym,
                entry_date=today.strftime("%Y-%m-%d"),
                entry_price=entry_price,
                shares=shares,
                cost=cost,
                highest_price=entry_price,
                stop_loss_price=stop_price,
                take_profit_price=entry_price * (1 + TAKE_PROFIT_PCT),
                trailing_activated=False,
            )

            self.state.cash -= cost
            self.state.positions[sym] = pos

            self.state.trades.append({
                "date": today.strftime("%Y-%m-%d"),
                "symbol": sym,
                "action": "BUY",
                "price": round(entry_price, 4),
                "shares": shares,
                "cost": round(cost, 2),
                "reason": "REBALANCE",
            })

    def _close_position(self, sym: str, today: pd.Timestamp, price: float, reason: str):
        """平仓"""
        pos = self.state.positions[sym]
        exit_price = price * (1 - SLIPPAGE)
        proceeds = pos.shares * exit_price * (1 - COMMISSION_RATE)
        pnl = proceeds - pos.cost
        pnl_pct = (exit_price / pos.entry_price - 1) * 100
        holding_days = (today - pd.Timestamp(pos.entry_date)).days

        self.state.cash += proceeds

        self.state.trades.append({
            "date": today.strftime("%Y-%m-%d"),
            "symbol": sym,
            "action": "SELL",
            "price": round(exit_price, 4),
            "shares": pos.shares,
            "proceeds": round(proceeds, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "holding_days": holding_days,
            "reason": reason,
        })

    def _record_equity(self, today: pd.Timestamp):
        """记录每日权益"""
        position_value = 0.0
        for sym, pos in self.state.positions.items():
            if sym in self.all_data and today in self.all_data[sym].index:
                close = self.all_data[sym].loc[today, "close"]
                position_value += pos.shares * close

        self.state.equity_curve.append({
            "date": today.strftime("%Y-%m-%d"),
            "cash": round(self.state.cash, 2),
            "position_value": round(position_value, 2),
            "total_equity": round(self.state.cash + position_value, 2),
            "n_positions": self.state.n_positions,
        })

    def _compute_performance(self) -> Dict:
        """计算绩效指标"""
        equity_df = pd.DataFrame(self.state.equity_curve)
        if len(equity_df) < 5:
            return {"error": "数据不足"}

        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df = equity_df.set_index("date")

        # 日收益率
        equity_df["daily_ret"] = equity_df["total_equity"].pct_change()

        # 总收益
        total_return = (equity_df["total_equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100

        # 年化收益 (252交易日)
        n_days = len(equity_df)
        ann_return = ((1 + total_return / 100) ** (252 / n_days) - 1) * 100 if n_days > 0 else 0

        # 年化波动率
        daily_std = equity_df["daily_ret"].std()
        ann_vol = daily_std * np.sqrt(252) * 100

        # 夏普比率 (假设无风险利率=2%)
        excess_ret = equity_df["daily_ret"] - 0.02 / 252
        sharpe = (excess_ret.mean() / daily_std) * np.sqrt(252) if daily_std > 0 else 0

        # 最大回撤
        cum_max = equity_df["total_equity"].cummax()
        drawdown = (equity_df["total_equity"] - cum_max) / cum_max
        max_dd = drawdown.min() * 100

        # Calmar比率
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        # 胜率 (按交易)
        trades_df = pd.DataFrame(self.state.trades)
        sells = trades_df[trades_df["action"] == "SELL"] if len(trades_df) > 0 else pd.DataFrame()
        if len(sells) > 0:
            win_rate = (sells["pnl"] > 0).mean() * 100
            avg_win = sells[sells["pnl"] > 0]["pnl_pct"].mean() if (sells["pnl"] > 0).any() else 0
            avg_loss = sells[sells["pnl"] < 0]["pnl_pct"].mean() if (sells["pnl"] < 0).any() else 0
            profit_factor = abs(sells[sells["pnl"] > 0]["pnl"].sum() / sells[sells["pnl"] < 0]["pnl"].sum()) if (sells["pnl"] < 0).any() else float("inf")
            total_trades = len(sells)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            total_trades = 0

        # 止损止盈统计
        if len(sells) > 0:
            stop_loss_count = (sells["reason"] == "STOP_LOSS").sum()
            take_profit_count = (sells["reason"] == "TAKE_PROFIT").sum()
            trailing_count = (sells["reason"] == "TRAILING_STOP").sum()
            rebalance_count = (sells["reason"] == "REBALANCE_SELL").sum()
        else:
            stop_loss_count = take_profit_count = trailing_count = rebalance_count = 0

        # 换手率
        buys = trades_df[trades_df["action"] == "BUY"] if len(trades_df) > 0 else pd.DataFrame()
        total_buy_cost = buys["cost"].sum() if len(buys) > 0 else 0
        turnover_ratio = total_buy_cost / (INITIAL_CAPITAL * n_days / 252) if n_days > 0 else 0

        return {
            "total_return": round(total_return, 2),
            "ann_return": round(ann_return, 2),
            "ann_vol": round(ann_vol, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd, 2),
            "calmar": round(calmar, 2),
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
            "total_trades": total_trades,
            "stop_loss_count": stop_loss_count,
            "take_profit_count": take_profit_count,
            "trailing_stop_count": trailing_count,
            "rebalance_count": rebalance_count,
            "turnover_ratio": round(turnover_ratio, 2),
            "n_days": n_days,
            "final_equity": round(equity_df["total_equity"].iloc[-1], 2),
        }


# ==================== 4. 报告生成 ====================
def generate_report(perf: Dict, equity_df: pd.DataFrame, trades_df: pd.DataFrame):
    """生成回测报告"""
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("策略回测报告 (事件驱动)")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"初始资金: {INITIAL_CAPITAL:,.0f}")
    print(f"止损: -{STOP_LOSS_PCT*100}% | 止盈: +{TAKE_PROFIT_PCT*100}% | 追踪: -{TRAILING_STOP_PCT*100}%")
    print("=" * 80)

    if "error" in perf:
        print(f"  错误: {perf['error']}")
        return

    # 核心指标
    print("\n一、核心绩效指标")
    print("-" * 60)
    metrics = [
        ("总收益", f"{perf['total_return']}%"),
        ("年化收益", f"{perf['ann_return']}%"),
        ("年化波动", f"{perf['ann_vol']}%"),
        ("夏普比率", f"{perf['sharpe']}"),
        ("最大回撤", f"{perf['max_drawdown']}%"),
        ("Calmar比率", f"{perf['calmar']}"),
        ("最终权益", f"{perf['final_equity']:,.0f}"),
    ]
    for name, value in metrics:
        print(f"  {name:<12}: {value}")

    # 交易统计
    print("\n二、交易统计")
    print("-" * 60)
    trade_metrics = [
        ("总交易次数", perf["total_trades"]),
        ("胜率", f"{perf['win_rate']}%"),
        ("平均盈利", f"{perf['avg_win']}%"),
        ("平均亏损", f"{perf['avg_loss']}%"),
        ("盈亏比", perf["profit_factor"]),
        ("年化换手", f"{perf['turnover_ratio']:.1f}x"),
    ]
    for name, value in trade_metrics:
        print(f"  {name:<12}: {value}")

    # 风控统计
    print("\n三、风控触发统计")
    print("-" * 60)
    print(f"  止损触发: {perf['stop_loss_count']} 次")
    print(f"  止盈触发: {perf['take_profit_count']} 次")
    print(f"  追踪止损: {perf['trailing_stop_count']} 次")
    print(f"  调仓卖出: {perf['rebalance_count']} 次")

    # 对比组合构建(向量化)结果
    print("\n四、事件驱动 vs 向量化 对比")
    print("-" * 80)
    print(f"  {'指标':<15} {'事件驱动(逐bar)':<20} {'向量化(调仓日)':<20} {'差异':<10}")
    print(f"  {'-'*65}")
    # 读取组合构建结果
    portfolio_csv = os.path.join(PROJECT_ROOT, "reports", "portfolio", "portfolio_performance.csv")
    if os.path.exists(portfolio_csv):
        pf = pd.read_csv(portfolio_csv)
        best_pf = pf.loc[pf["sharpe"].idxmax()] if len(pf) > 0 else None
        if best_pf is not None:
            comparisons = [
                ("年化收益", f"{perf['ann_return']}%", f"{best_pf['ann_return']}%"),
                ("夏普比率", f"{perf['sharpe']}", f"{best_pf['sharpe']}"),
                ("最大回撤", f"{perf['max_drawdown']}%", f"{best_pf['max_drawdown']}%"),
                ("胜率", f"{perf['win_rate']}%", f"{best_pf['win_rate']}%"),
            ]
            for name, ev, vec in comparisons:
                diff = float(ev.replace("%", "")) - float(vec.replace("%", "")) if "%" in ev else float(ev) - float(vec)
                sign = "+" if diff > 0 else ""
                print(f"  {name:<15} {ev:<20} {vec:<20} {sign}{diff:.2f}")

    # 保存
    equity_df.to_csv(os.path.join(REPORT_DIR, "equity_curve.csv"), index=True, encoding="utf-8-sig")
    trades_df.to_csv(os.path.join(REPORT_DIR, "trade_log.csv"), index=False, encoding="utf-8-sig")

    # 摘要
    summary = []
    summary.append("=" * 80)
    summary.append("策略回测报告 (事件驱动)")
    summary.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"初始资金: {INITIAL_CAPITAL:,.0f}")
    summary.append("=" * 80)
    summary.append(f"总收益: {perf['total_return']}%")
    summary.append(f"年化收益: {perf['ann_return']}%")
    summary.append(f"夏普比率: {perf['sharpe']}")
    summary.append(f"最大回撤: {perf['max_drawdown']}%")
    summary.append(f"胜率: {perf['win_rate']}%")
    summary.append(f"盈亏比: {perf['profit_factor']}")
    summary.append(f"总交易: {perf['total_trades']}次")
    summary.append(f"止损: {perf['stop_loss_count']}次, 止盈: {perf['take_profit_count']}次")

    with open(os.path.join(REPORT_DIR, "backtest_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print(f"\n报告已保存到: {REPORT_DIR}")


# ==================== 主流程 ====================
def main():
    print("=" * 80)
    print("策略回测引擎 (事件驱动) — 逐bar + T+1 + 止损止盈")
    print("=" * 80)

    # 1. 加载数据
    all_data = load_all_data()
    if len(all_data) < 20:
        print("错误: 可用数据不足")
        return

    # 2. 加载信号
    signals = load_signals()
    if signals is None or len(signals) == 0:
        signals = generate_default_signals(all_data)
    if len(signals) == 0:
        print("错误: 无可用信号")
        return

    print(f"  信号: {len(signals)} 条, {signals['date'].nunique()} 个调仓日")

    # 3. 运行回测
    engine = BacktestEngine(all_data)
    perf = engine.run(signals)

    # 4. 生成报告
    equity_df = pd.DataFrame(engine.state.equity_curve)
    trades_df = pd.DataFrame(engine.state.trades)
    generate_report(perf, equity_df, trades_df)

    print("\n回测完成!")


if __name__ == "__main__":
    main()
