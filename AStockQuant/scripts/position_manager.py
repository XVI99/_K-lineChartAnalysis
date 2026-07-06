# -*- coding: utf-8 -*-
"""
仓位管理系统
============
从"买不买"升级为完整的仓位决策: 买多少、何时入场、卖多少、何时出场

核心决策链:
  因子评分 → 仓位大小 → 入场时机 → 持仓监控 → 出场规则

仓位分配:
  1. 得分加权: 仓位 = 基础仓位 × (得分 / 平均得分)
  2. Kelly公式: 仓位 = (胜率×平均盈利 - 败率×平均亏损) / 平均盈利
  3. 波动率目标: 仓位 = 目标波动率 / 标的波动率

入场时机:
  1. 得分阈值: 仅当综合得分 > threshold 才入场
  2. 回调入场: 得分达标后等价格回调N日再入场
  3. 量能确认: 成交量>20日均量1.2倍才入场

出场规则:
  1. 信号衰减: 得分跌破阈值 → 减仓/清仓
  2. 时间止损: 持有超过N天未盈利 → 清仓
  3. 分批止盈: +10%卖1/3, +20%卖1/3, 剩余追踪
  4. ATR追踪止损: 从最高点回撤N倍ATR → 清仓
  5. 固定止损: -8% → 清仓
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ==================== 配置 ====================
# 仓位分配
BASE_WEIGHT = 0.10          # 基础仓位 10%
MAX_WEIGHT = 0.25           # 单只上限 25%
MIN_WEIGHT = 0.03           # 单只下限 3%
POSITION_METHOD = "score_weighted"  # equal / score_weighted / kelly / vol_target
TARGET_VOL = 0.20           # 波动率目标 20% (年化)

# 入场
ENTRY_SCORE_THRESHOLD = 0.65  # 得分阈值(排名百分位), 低于此不入场
ENTRY_PULLBACK_DAYS = 3       # 回调入场: 等价格回调N日
ENTRY_VOLUME_CONFIRM = True   # 量能确认: 成交量>20日均量1.2倍
ENTRY_VOLUME_RATIO = 1.2

# 出场
EXIT_SCORE_THRESHOLD = 0.40   # 信号衰减阈值: 得分<此值→清仓
EXIT_TIME_STOP_DAYS = 20      # 时间止损: 持有N天未盈利→清仓
EXIT_PARTIAL_TP_LEVELS = [    # 分批止盈: [(盈利%, 卖出比例)]
    (0.10, 0.33),              # +10%卖1/3
    (0.20, 0.33),              # +20%再卖1/3
    (0.30, 0.34),              # +30%卖剩余
]
EXIT_ATR_TRAIL_MULT = 3.0     # ATR追踪止损倍数
EXIT_STOP_LOSS_PCT = 0.08     # 固定止损 -8%
EXIT_MAX_HOLDING_DAYS = 60    # 最大持有天数(强制清仓)

# ATR
ATR_PERIOD = 14


# ==================== 数据结构 ====================
@dataclass
class PositionState:
    """持仓状态(增强版)"""
    symbol: str
    entry_date: str
    entry_price: float
    shares: int
    total_shares: int          # 初始总股数(分批止盈用)
    cost: float
    highest_price: float
    highest_date: str
    stop_loss_price: float
    atr_trail_price: float     # ATR追踪止损价
    partial_tp_triggered: List[float] = field(default_factory=list)  # 已触发的止盈级别
    time_stop_days: int = 0    # 持有天数
    entry_score: float = 0.0   # 入场时得分

    @property
    def remaining_shares(self) -> int:
        return self.shares

    @property
    def pnl_pct(self) -> float:
        return self.highest_price / self.entry_price - 1


@dataclass
class PositionDecision:
    """单标的仓位决策"""
    symbol: str
    action: str                # BUY / SELL / HOLD / PARTIAL_SELL
    target_weight: float       # 目标仓位(0-1)
    entry_timing: str          # NOW / WAIT_PULLBACK / WAIT_VOLUME / SKIP
    exit_reason: str           # 出场原因
    exit_ratio: float          # 卖出比例(0-1)
    stop_price: float          # 止损价
    take_profit_levels: List[Tuple[float, float]]  # [(目标价, 卖出比例)]
    score: float               # 当前得分
    message: str               # 可读说明


# ==================== 1. 仓位大小计算 ====================
def calculate_position_size(
    score: float,
    avg_score: float,
    volatility: float = None,
    win_rate: float = 0.50,
    avg_win: float = 0.10,
    avg_loss: float = 0.06,
    method: str = POSITION_METHOD,
) -> float:
    """
    计算仓位大小

    Args:
        score: 当前标的综合得分(0-1)
        avg_score: 截面平均得分
        volatility: 标的年化波动率
        win_rate: 历史胜率
        avg_win: 历史平均盈利%
        avg_loss: 历史平均亏损%

    Returns:
        仓位比例 (0-1)
    """
    if method == "equal":
        return BASE_WEIGHT

    elif method == "score_weighted":
        # 得分越高仓位越大
        if avg_score > 0:
            ratio = score / avg_score
        else:
            ratio = 1.0
        weight = BASE_WEIGHT * ratio
        return np.clip(weight, MIN_WEIGHT, MAX_WEIGHT)

    elif method == "kelly":
        # Kelly公式: f = (p*b - q) / b
        # p=胜率, b=盈亏比(avg_win/avg_loss), q=1-p
        if avg_loss <= 0:
            return BASE_WEIGHT
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - p
        kelly_f = (p * b - q) / b if b > 0 else 0
        kelly_f = max(0, kelly_f)
        # 半凯利(更保守)
        weight = kelly_f * 0.5
        return np.clip(weight, MIN_WEIGHT, MAX_WEIGHT)

    elif method == "vol_target":
        # 波动率目标: weight = target_vol / stock_vol
        if volatility is None or volatility <= 0:
            return BASE_WEIGHT
        weight = TARGET_VOL / volatility
        return np.clip(weight, MIN_WEIGHT, MAX_WEIGHT)

    return BASE_WEIGHT


# ==================== 2. 入场时机判断 ====================
def evaluate_entry_timing(
    score: float,
    price_data: pd.DataFrame,
    as_of_idx: int,
) -> Tuple[str, str]:
    """
    判断入场时机

    Returns:
        (timing, message)
        timing: NOW / WAIT_PULLBACK / WAIT_VOLUME / SKIP
    """
    # 得分阈值
    if score < ENTRY_SCORE_THRESHOLD:
        return "SKIP", f"得分{score:.3f}<阈值{ENTRY_SCORE_THRESHOLD}"

    if as_of_idx < 5 or len(price_data) < 20:
        return "NOW", "数据不足, 直接入场"

    close = price_data["close"]
    volume = price_data.get("volume", pd.Series(1, index=price_data.index))

    # 回调入场检查
    if ENTRY_PULLBACK_DAYS > 0:
        recent_high = close.iloc[-ENTRY_PULLBACK_DAYS-1:-1].max()
        current = close.iloc[-1]
        if current >= recent_high * 0.98:
            return "WAIT_PULLBACK", f"价格接近{ENTRY_PULLBACK_DAYS}日高点, 等回调"

    # 量能确认
    if ENTRY_VOLUME_CONFIRM and len(volume) >= 20:
        vol_now = volume.iloc[-1]
        vol_ma20 = volume.rolling(20).mean().iloc[-1]
        if vol_now < vol_ma20 * ENTRY_VOLUME_RATIO:
            return "WAIT_VOLUME", f"量能不足(当前{vol_now:.0f}<均量{vol_ma20:.0f}×{ENTRY_VOLUME_RATIO})"

    return "NOW", "入场条件满足"


# ==================== 3. 出场规则检查 ====================
def evaluate_exit_rules(
    pos: PositionState,
    current_score: float,
    price_data: pd.DataFrame,
    as_of_idx: int,
    holding_days: int,
) -> List[Tuple[str, float, str]]:
    """
    检查所有出场规则

    Returns:
        [(exit_type, exit_ratio, reason), ...]
        exit_type: STOP_LOSS / ATR_TRAIL / TIME_STOP / SIGNAL_DECAY / PARTIAL_TP / MAX_HOLDING
        exit_ratio: 卖出比例 (1.0=全清)
    """
    exits = []

    if as_of_idx < 0 or len(price_data) == 0:
        return exits

    bar = price_data.iloc[as_of_idx] if as_of_idx < len(price_data) else price_data.iloc[-1]
    close = bar["close"]
    high = bar.get("high", close)
    low = bar.get("low", close)

    # 1. 固定止损
    stop_price = pos.entry_price * (1 - EXIT_STOP_LOSS_PCT)
    if low <= stop_price:
        exits.append(("STOP_LOSS", 1.0, f"触发固定止损: {low:.4f}≤{stop_price:.4f}"))

    # 2. ATR追踪止损
    if pos.atr_trail_price > 0 and low <= pos.atr_trail_price:
        exits.append(("ATR_TRAIL", 1.0, f"触发ATR追踪止损: {low:.4f}≤{pos.atr_trail_price:.4f}"))

    # 3. 时间止损
    if holding_days >= EXIT_TIME_STOP_DAYS and pos.pnl_pct < 0.02:
        exits.append(("TIME_STOP", 1.0, f"时间止损: 持有{holding_days}天未盈利"))

    # 4. 信号衰减
    if current_score < EXIT_SCORE_THRESHOLD:
        exits.append(("SIGNAL_DECAY", 1.0, f"信号衰减: 得分{current_score:.3f}<{EXIT_SCORE_THRESHOLD}"))

    # 5. 分批止盈
    for tp_level, sell_ratio in EXIT_PARTIAL_TP_LEVELS:
        if tp_level not in pos.partial_tp_triggered:
            tp_price = pos.entry_price * (1 + tp_level)
            if high >= tp_price:
                pos.partial_tp_triggered.append(tp_level)
                exits.append(("PARTIAL_TP", sell_ratio, f"分批止盈+{tp_level*100:.0f}%: 卖{sell_ratio*100:.0f}%"))

    # 6. 最大持有天数
    if holding_days >= EXIT_MAX_HOLDING_DAYS:
        exits.append(("MAX_HOLDING", 1.0, f"最大持有{EXIT_MAX_HOLDING_DAYS}天, 强制清仓"))

    return exits


# ==================== 4. 综合仓位决策引擎 ====================
class PositionManager:
    """仓位管理引擎"""

    def __init__(
        self,
        all_data: Dict[str, pd.DataFrame],
        win_rate: float = 0.50,
        avg_win: float = 0.10,
        avg_loss: float = 0.06,
    ):
        self.all_data = all_data
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss

    def make_decision(
        self,
        symbol: str,
        score: float,
        avg_score: float,
        as_of_date: str,
        current_position: Optional[PositionState] = None,
    ) -> PositionDecision:
        """
        对单个标的做出完整仓位决策

        Args:
            symbol: 标的代码
            score: 当前综合得分(0-1)
            avg_score: 截面平均得分
            as_of_date: 决策日期
            current_position: 当前持仓(None=未持有)

        Returns:
            PositionDecision
        """
        as_of_ts = pd.Timestamp(as_of_date)

        # 获取价格数据
        if symbol not in self.all_data:
            return PositionDecision(
                symbol=symbol, action="SKIP", target_weight=0,
                entry_timing="SKIP", exit_reason="无数据", exit_ratio=0,
                stop_price=0, take_profit_levels=[], score=score,
                message=f"{symbol}: 无价格数据"
            )

        df = self.all_data[symbol]
        df_trunc = df[df.index <= as_of_ts]
        if len(df_trunc) < 20:
            return PositionDecision(
                symbol=symbol, action="SKIP", target_weight=0,
                entry_timing="SKIP", exit_reason="数据不足", exit_ratio=0,
                stop_price=0, take_profit_levels=[], score=score,
                message=f"{symbol}: 数据不足({len(df_trunc)}行)"
            )

        as_of_idx = len(df_trunc) - 1

        # === 已有持仓: 检查出场 ===
        if current_position is not None:
            holding_days = (as_of_ts - pd.Timestamp(current_position.entry_date)).days

            # 更新持仓状态
            bar = df_trunc.iloc[-1]
            if bar["high"] > current_position.highest_price:
                current_position.highest_price = bar["high"]
                current_position.highest_date = as_of_date

            # 更新ATR追踪价
            atr = self._calc_atr(df_trunc)
            if atr > 0:
                current_position.atr_trail_price = current_position.highest_price - EXIT_ATR_TRAIL_MULT * atr

            # 检查出场规则
            exits = evaluate_exit_rules(current_position, score, df_trunc, as_of_idx, holding_days)

            if exits:
                # 取最优先的出场信号
                # 优先级: STOP_LOSS > ATR_TRAIL > SIGNAL_DECAY > TIME_STOP > PARTIAL_TP > MAX_HOLDING
                priority = {"STOP_LOSS": 0, "ATR_TRAIL": 1, "SIGNAL_DECAY": 2, "TIME_STOP": 3, "PARTIAL_TP": 4, "MAX_HOLDING": 5}
                exits.sort(key=lambda x: priority.get(x[0], 99))

                exit_type, exit_ratio, reason = exits[0]

                if exit_ratio >= 1.0:
                    # 全清
                    return PositionDecision(
                        symbol=symbol, action="SELL", target_weight=0,
                        entry_timing="NOW", exit_reason=reason, exit_ratio=1.0,
                        stop_price=current_position.stop_loss_price,
                        take_profit_levels=[], score=score,
                        message=f"[{exit_type}] {reason}"
                    )
                else:
                    # 部分卖出
                    return PositionDecision(
                        symbol=symbol, action="PARTIAL_SELL", target_weight=0,
                        entry_timing="NOW", exit_reason=reason, exit_ratio=exit_ratio,
                        stop_price=current_position.stop_loss_price,
                        take_profit_levels=[], score=score,
                        message=f"[{exit_type}] {reason} (卖{exit_ratio*100:.0f}%)"
                    )

            # 继续持有
            return PositionDecision(
                symbol=symbol, action="HOLD", target_weight=0,
                entry_timing="NOW", exit_reason="", exit_ratio=0,
                stop_price=current_position.stop_loss_price,
                take_profit_levels=[], score=score,
                message=f"持有中(第{holding_days}天, 浮盈{current_position.pnl_pct*100:+.1f}%)"
            )

        # === 无持仓: 判断是否入场 ===
        timing, timing_msg = evaluate_entry_timing(score, df_trunc, as_of_idx)

        if timing == "SKIP":
            return PositionDecision(
                symbol=symbol, action="SKIP", target_weight=0,
                entry_timing=timing, exit_reason="", exit_ratio=0,
                stop_price=0, take_profit_levels=[], score=score,
                message=timing_msg
            )

        # 计算仓位大小
        vol = self._calc_volatility(df_trunc)
        weight = calculate_position_size(
            score, avg_score, vol,
            self.win_rate, self.avg_win, self.avg_loss,
        )

        # 计算止损价
        entry_price = df_trunc["close"].iloc[-1]
        stop_price = entry_price * (1 - EXIT_STOP_LOSS_PCT)

        # 计算止盈目标价
        tp_levels = [(entry_price * (1 + level), ratio) for level, ratio in EXIT_PARTIAL_TP_LEVELS]

        if timing == "NOW":
            return PositionDecision(
                symbol=symbol, action="BUY", target_weight=weight,
                entry_timing="NOW", exit_reason="", exit_ratio=0,
                stop_price=stop_price, take_profit_levels=tp_levels,
                score=score,
                message=f"入场: 仓位{weight*100:.1f}% 止损{stop_price:.4f}"
            )
        else:
            return PositionDecision(
                symbol=symbol, action="WATCH", target_weight=weight,
                entry_timing=timing, exit_reason="", exit_ratio=0,
                stop_price=stop_price, take_profit_levels=tp_levels,
                score=score,
                message=f"观察中: {timing_msg}"
            )

    def _calc_atr(self, df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
        """计算ATR"""
        if len(df) < period:
            return 0
        high, low, close = df["high"], df["low"], df["close"]
        tr = np.maximum(
            high - low,
            np.maximum(
                abs(high - close.shift(1)),
                abs(low - close.shift(1)),
            ),
        )
        return float(tr.rolling(period).mean().iloc[-1])

    def _calc_volatility(self, df: pd.DataFrame, period: int = 20) -> float:
        """计算年化波动率"""
        if len(df) < period:
            return 0
        daily_ret = df["close"].pct_change().dropna()
        if len(daily_ret) < 5:
            return 0
        return float(daily_ret.std() * np.sqrt(252))


# ==================== 5. 组合级决策 ====================
def generate_portfolio_decisions(
    scored_df: pd.DataFrame,
    position_manager: PositionManager,
    as_of_date: str,
    current_positions: Dict[str, PositionState] = None,
    max_positions: int = 10,
) -> Dict:
    """
    对整个组合生成仓位决策

    Args:
        scored_df: 排序后的因子评分表 [symbol, composite_score, rank, ...]
        position_manager: 仓位管理器
        as_of_date: 决策日期
        current_positions: 当前持仓 {symbol: PositionState}
        max_positions: 最大持仓数

    Returns:
        {
            'decisions': [PositionDecision, ...],
            'summary': {
                'new_buys': int, 'sells': int, 'partial_sells': int,
                'holds': int, 'watches': int, 'skips': int,
                'total_target_weight': float,
                'market_regime': str,
            }
        }
    """
    if current_positions is None:
        current_positions = {}

    avg_score = scored_df["composite_score"].mean() if len(scored_df) > 0 else 0.5

    decisions = []
    buy_candidates = []

    # 1. 先处理已有持仓
    for sym, pos in current_positions.items():
        row = scored_df[scored_df["symbol"] == sym]
        if len(row) > 0:
            score = row.iloc[0]["composite_score"]
        else:
            score = 0.0  # 不在排名中=得分最低

        decision = position_manager.make_decision(sym, score, avg_score, as_of_date, pos)
        decisions.append(decision)

    # 2. 处理候选买入标的
    existing_symbols = set(current_positions.keys())
    for _, row in scored_df.iterrows():
        sym = row["symbol"]
        if sym in existing_symbols:
            continue

        score = row["composite_score"]
        decision = position_manager.make_decision(sym, score, avg_score, as_of_date)
        decisions.append(decision)

        if decision.action == "BUY":
            buy_candidates.append(decision)

    # 3. 限制买入数量
    current_hold_count = sum(1 for d in decisions if d.action in ("HOLD", "PARTIAL_SELL"))
    available_slots = max_positions - current_hold_count

    if available_slots < len(buy_candidates):
        # 按得分排序, 只保留前available_slots个BUY
        buy_candidates.sort(key=lambda x: x.score, reverse=True)
        for d in buy_candidates[available_slots:]:
            d.action = "WATCH"
            d.message = f"候选但仓位已满(前{max_positions}名之外)"

    # 4. 汇总
    summary = {
        "new_buys": sum(1 for d in decisions if d.action == "BUY"),
        "sells": sum(1 for d in decisions if d.action == "SELL"),
        "partial_sells": sum(1 for d in decisions if d.action == "PARTIAL_SELL"),
        "holds": sum(1 for d in decisions if d.action == "HOLD"),
        "watches": sum(1 for d in decisions if d.action == "WATCH"),
        "skips": sum(1 for d in decisions if d.action == "SKIP"),
        "total_target_weight": sum(d.target_weight for d in decisions if d.action == "BUY"),
    }

    return {"decisions": decisions, "summary": summary}


# ==================== 6. 格式化输出 ====================
def format_decisions_output(result: Dict, as_of_date: str) -> str:
    """格式化决策输出为可读文本"""
    decisions = result["decisions"]
    summary = result["summary"]

    lines = []
    lines.append("=" * 80)
    lines.append(f"  仓位决策报告 — {as_of_date}")
    lines.append("=" * 80)

    # 汇总
    lines.append(f"\n  持仓管理: BUY={summary['new_buys']} SELL={summary['sells']} "
                 f"HOLD={summary['holds']} WATCH={summary['watches']} "
                 f"PARTIAL={summary['partial_sells']} SKIP={summary['skips']}")

    # 卖出
    sells = [d for d in decisions if d.action == "SELL"]
    if sells:
        lines.append(f"\n  [卖出] ({len(sells)}只):")
        for d in sells:
            lines.append(f"    {d.symbol:<8} {d.message}")

    # 部分卖出
    partials = [d for d in decisions if d.action == "PARTIAL_SELL"]
    if partials:
        lines.append(f"\n  [部分止盈] ({len(partials)}只):")
        for d in partials:
            lines.append(f"    {d.symbol:<8} {d.message}")

    # 买入
    buys = [d for d in decisions if d.action == "BUY"]
    if buys:
        lines.append(f"\n  [买入] ({len(buys)}只):")
        lines.append(f"    {'代码':<8} {'仓位':>8} {'止损价':>10} {'止盈目标':>30} {'说明'}")
        lines.append(f"    {'-'*70}")
        for d in buys:
            tp_str = " → ".join(f"+{lvl*100:.0f}%卖{ratio*100:.0f}%" for lvl, ratio in EXIT_PARTIAL_TP_LEVELS)
            lines.append(f"    {d.symbol:<8} {d.target_weight:>7.1%} {d.stop_price:>10.4f} {tp_str:<30} {d.message}")

    # 持有
    holds = [d for d in decisions if d.action == "HOLD"]
    if holds:
        lines.append(f"\n  [持有] ({len(holds)}只):")
        for d in holds:
            lines.append(f"    {d.symbol:<8} {d.message}")

    # 观察
    watches = [d for d in decisions if d.action == "WATCH"]
    if watches:
        lines.append(f"\n  [观察] ({len(watches)}只, 待入场时机):")
        for d in watches[:5]:
            lines.append(f"    {d.symbol:<8} 目标仓位{d.target_weight:.1%} {d.message}")
        if len(watches) > 5:
            lines.append(f"    ... 还有{len(watches)-5}只")

    return "\n".join(lines)
