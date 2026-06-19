# -*- coding: utf-8 -*-
"""
d2_rotation.py — D2 防守轮动策略 v2.0 (优化版)

基于Serenity Bayesian Framework的防御性轮动策略

核心改进：
1. 多时间框架分析 (MA200趋势 + MA50入场)
2. RPS动量过滤 (避免逆势买入)
3. 动态仓位调整 (信念强度驱动)
4. 跟踪止损保护 (锁定利润)
5. ETF差异化配置

作者: Matrix Agent
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math

import numpy as np
import pandas as pd


class MarketPhase(Enum):
    """市场阶段枚举"""
    BULL_CONFIRM = "bull_confirm"      # 牛市确认
    BULL_RECOVERY = "bull_recovery"    # 牛市反弹
    NEUTRAL = "neutral"                # 中性
    BEAR_RALLY = "bear_rally"          # 熊市反弹
    BEAR_CONFIRM = "bear_confirm"      # 熊市确认


@dataclass
class D2Signal:
    """D2交易信号"""
    code: str
    name: str
    signal_type: str           # BUY / SELL / HOLD
    strength: float            # 信号强度 0-1
    conviction: float          # 信念度 0-1
    phase: MarketPhase
    entry_reason: str          # 入场原因描述
    stop_loss: float           # 止损价
    take_profit: float         # 止盈价
    position_size: float       # 建议仓位 (0-1)


@dataclass
class Position:
    """持仓信息"""
    code: str
    name: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    
    @property
    def pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity
    
    @property
    def pnl_pct(self) -> float:
        return (self.current_price / self.avg_cost - 1) * 100 if self.avg_cost > 0 else 0.0


class D2RotationStrategy:
    """
    D2 防守轮动策略
    
    核心逻辑：
    1. MA200判断趋势方向
    2. MA50配合RSI判断入场时机
    3. RPS过滤弱势ETF
    4. 信念强度决定仓位
    5. 跟踪止损保护利润
    """
    
    # ========== 策略参数 ==========
    MA_SHORT = 50     # 短期均线
    MA_LONG = 200     # 长期均线
    
    # 入场阈值
    RSI_OB_LEVEL = 70     # RSI超买
    RSI_OS_LEVEL = 30     # RSI超卖
    RSI_NEUTRAL_HIGH = 55
    RSI_NEUTRAL_LOW = 45
    
    RPS_MIN_BULL = 60    # 牛市RPS最低要求
    RPS_MIN_BEAR = 70    # 熊市RPS最低要求
    
    # 止损参数
    STOP_LOSS_PCT = 0.08      # 固定止损 8%
    TRAIL_STOP_PCT = 0.15     # 跟踪止损 15%
    MAX_DRAWUP = 0.20         # 最大回撤允许
    
    # 仓位参数
    MIN_CONVICTION = 0.50     # 最小信念度
    MAX_POSITION = 0.40       # 单只最大仓位
    FULL_POSITION = 0.35     # 满仓阈值
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.history: List[dict] = []
        
    # ==================== 核心分析函数 ====================
    
    def analyze_etf(self, df: pd.DataFrame, code: str, name: str) -> D2Signal:
        """
        分析单个ETF，生成交易信号
        """
        if len(df) < self.MA_LONG + 10:
            return self._create_hold_signal(code, name, "数据不足")
        
        # 计算指标
        df = self._compute_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # 判断市场阶段
        phase = self._determine_phase(df)
        
        # 计算信号强度
        signal_strength = self._calculate_signal_strength(latest, prev, phase)
        
        # 计算信念度
        conviction = self._calculate_conviction(df, latest, phase)
        
        # 生成交易信号
        return self._generate_signal(
            df, latest, code, name, phase, signal_strength, conviction
        )
    
    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        
        # 均线
        df['ma50'] = df['close'].rolling(50).mean()
        df['ma200'] = df['close'].rolling(200).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        bb_std = df['close'].rolling(20).std()
        bb_mid = df['close'].rolling(20).mean()
        df['bb_upper'] = bb_mid + 2 * bb_std
        df['bb_lower'] = bb_mid - 2 * bb_std
        
        # 动量
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_20'] = df['close'].pct_change(20)
        
        # 价格位置
        df['price_vs_ma50'] = (df['close'] - df['ma50']) / df['ma50']
        df['price_vs_ma200'] = (df['close'] - df['ma200']) / df['ma200']
        
        return df
    
    def _determine_phase(self, df: pd.DataFrame) -> MarketPhase:
        """判断市场阶段"""
        latest = df.iloc[-1]
        
        ma50 = latest['ma50']
        ma200 = latest['ma200']
        close = latest['close']
        
        # MA200趋势
        is_above_ma200 = close > ma200
        is_ma200_up = latest['ma200'] > df['ma200'].iloc[-20] if pd.notna(latest['ma200']) else False
        
        # MA50位置
        is_above_ma50 = close > ma50
        is_ma50_above_ma200 = ma50 > ma200 if pd.notna(ma50) and pd.notna(ma200) else False
        
        if is_above_ma200 and is_ma200_up:
            if is_above_ma50 and is_ma50_above_ma200:
                return MarketPhase.BULL_CONFIRM
            else:
                return MarketPhase.BULL_RECOVERY
        elif is_above_ma200 and not is_ma200_up:
            return MarketPhase.NEUTRAL
        elif not is_above_ma200:
            if is_above_ma50:
                return MarketPhase.BEAR_RALLY
            else:
                return MarketPhase.BEAR_CONFIRM
        else:
            return MarketPhase.NEUTRAL
    
    def _calculate_signal_strength(
        self, latest: pd.Series, prev: pd.Series, phase: MarketPhase
    ) -> float:
        """计算信号强度"""
        strength = 0.0
        
        # 1. RSI因素 (权重30%)
        rsi = latest.get('rsi', 50)
        if phase in [MarketPhase.BULL_CONFIRM, MarketPhase.BULL_RECOVERY]:
            # 牛市：在40-60区间偏强
            if 45 <= rsi <= 55:
                strength += 0.20
            elif rsi < 45:
                strength += 0.25  # 超卖反弹
            elif rsi > 70:
                strength -= 0.15  # 超买警告
        else:
            # 熊市：需要更强RSI才买入
            if 50 <= rsi <= 60:
                strength += 0.20
            elif rsi > 65:
                strength += 0.25  # 反弹确认
        
        # 2. 价格vs均线 (权重30%)
        price_ma50 = latest.get('price_vs_ma50', 0)
        price_ma200 = latest.get('price_vs_ma200', 0)
        
        if phase in [MarketPhase.BULL_CONFIRM, MarketPhase.BULL_RECOVERY]:
            if price_ma50 > 0 and price_ma200 > 0:
                strength += 0.30
            elif price_ma50 > 0:
                strength += 0.15
        else:
            if price_ma50 > 0 and price_ma200 < 0:
                strength += 0.25  # 反弹到MA50但未到MA200
            elif price_ma50 > 0.05:
                strength += 0.20
        
        # 3. 动量因素 (权重25%)
        mom5 = latest.get('momentum_5', 0)
        mom20 = latest.get('momentum_20', 0)
        
        if mom5 > 0 and mom20 > 0:
            strength += 0.25
        elif mom5 > 0 and mom20 < 0:
            strength += 0.10
        elif mom5 < 0 and mom20 < 0:
            strength -= 0.15
        
        # 4. 布林带位置 (权重15%)
        bb_pos = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']) \
                 if latest['bb_upper'] != latest['bb_lower'] else 0.5
        
        if 0.3 <= bb_pos <= 0.7:
            strength += 0.10
        elif bb_pos < 0.2:
            strength += 0.15  # 接近下轨，可能反弹
        elif bb_pos > 0.85:
            strength -= 0.10  # 接近上轨，注意风险
        
        return max(0.0, min(1.0, strength))
    
    def _calculate_conviction(
        self, df: pd.DataFrame, latest: pd.Series, phase: MarketPhase
    ) -> float:
        """计算信念度"""
        conviction = 0.5
        
        # 趋势一致性 (30%)
        ma_trend_count = 0
        for i in range(-5, 0):
            if len(df) > abs(i):
                row = df.iloc[i]
                if row['close'] > row['ma50'] > row['ma200']:
                    ma_trend_count += 1
        if ma_trend_count >= 4:
            conviction += 0.15
        elif ma_trend_count <= 1:
            conviction -= 0.10
        
        # RSI稳定性 (25%)
        rsi_values = df['rsi'].iloc[-10:]
        if len(rsi_values) >= 5:
            rsi_std = rsi_values.std()
            if rsi_std < 10:
                conviction += 0.15  # RSI稳定更好
            elif rsi_std > 20:
                conviction -= 0.10
        
        # 成交量确认 (20%)
        if len(df) >= 20:
            avg_vol = df['volume'].iloc[-20:].mean()
            recent_vol = df['volume'].iloc[-5:].mean()
            if recent_vol > avg_vol * 1.2:
                conviction += 0.10
            elif recent_vol < avg_vol * 0.7:
                conviction -= 0.05
        
        # 均线开口 (25%)
        ma50_slope = (latest['ma50'] - df['ma50'].iloc[-10]) / df['close'].iloc[-10] \
                     if pd.notna(latest['ma50']) else 0
        ma200_slope = (latest['ma200'] - df['ma200'].iloc[-10]) / df['close'].iloc[-10] \
                      if pd.notna(latest['ma200']) else 0
        
        if ma50_slope > 0 and ma200_slope > 0:
            conviction += 0.15
        elif ma50_slope < 0 and ma200_slope < 0:
            conviction -= 0.15
        
        return max(0.2, min(0.95, conviction))
    
    def _generate_signal(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        code: str,
        name: str,
        phase: MarketPhase,
        signal_strength: float,
        conviction: float
    ) -> D2Signal:
        """生成交易信号"""
        close = latest['close']
        ma50 = latest['ma50']
        ma200 = latest['ma200']
        rsi = latest.get('rsi', 50)
        
        # 计算止损止盈
        stop_loss = close * (1 - self.STOP_LOSS_PCT)
        take_profit = close * 1.25  # 默认25%止盈
        
        # 仓位计算
        if conviction >= 0.70:
            position_size = min(self.MAX_POSITION, conviction * 0.5)
        elif conviction >= self.MIN_CONVICTION:
            position_size = conviction * 0.35
        else:
            position_size = 0
        
        # 动态调整
        if phase in [MarketPhase.BULL_CONFIRM]:
            if rsi < 55 and signal_strength > 0.5:
                position_size = min(position_size * 1.2, self.MAX_POSITION)
        elif phase in [MarketPhase.BEAR_CONFIRM]:
            position_size *= 0.5  # 熊市降仓
        
        # ========== 信号生成 ==========
        # 买入条件
        buy_conditions = [
            phase in [MarketPhase.BULL_CONFIRM, MarketPhase.BULL_RECOVERY, MarketPhase.BEAR_RALLY],
            signal_strength >= 0.45,
            conviction >= self.MIN_CONVICTION,
            rsi < self.RSI_NEUTRAL_HIGH,
        ]
        
        if all(buy_conditions):
            return D2Signal(
                code=code,
                name=name,
                signal_type="BUY",
                strength=signal_strength,
                conviction=conviction,
                phase=phase,
                entry_reason=f"{phase.value} | RSI={rsi:.1f} | 强度={signal_strength:.2f}",
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size
            )
        
        # 卖出条件检查
        sell_reason = self._check_sell_conditions(df, latest, code)
        if sell_reason:
            return D2Signal(
                code=code,
                name=name,
                signal_type="SELL",
                strength=0.0,
                conviction=0.0,
                phase=phase,
                entry_reason=sell_reason,
                stop_loss=0.0,
                take_profit=0.0,
                position_size=0.0
            )
        
        return self._create_hold_signal(code, name, f"{phase.value} | 等待信号")
    
    def _check_sell_conditions(
        self, df: pd.DataFrame, latest: pd.Series, code: str
    ) -> Optional[str]:
        """检查是否需要卖出"""
        if code not in self.positions:
            return None
        
        pos = self.positions[code]
        close = latest['close']
        
        # 止损检查
        if close < pos.avg_cost * (1 - self.STOP_LOSS_PCT):
            return f"止损触发: {close:.3f} < {pos.avg_cost * (1 - self.STOP_LOSS_PCT):.3f}"
        
        # 跟踪止损检查
        if pos.pnl_pct > 10:
            trail_stop = pos.avg_cost * (1 + pos.pnl_pct / 100 * 0.6)
            if close < trail_stop:
                return f"跟踪止损: {close:.3f}"
        
        # 趋势破坏检查
        ma200 = latest['ma200']
        if pd.notna(ma200) and close < ma200:
            return f"跌破MA200: {close:.3f} < {ma200:.3f}"
        
        # RSI超买检查
        rsi = latest.get('rsi', 50)
        if rsi > 75:
            return f"RSI超买: {rsi:.1f}"
        
        # MA50死叉MA200检查
        ma50 = latest['ma50']
        if pd.notna(ma50) and pd.notna(ma200):
            prev_ma50 = df['ma50'].iloc[-5]
            prev_ma200 = df['ma200'].iloc[-5]
            if prev_ma50 > prev_ma200 and ma50 < ma200:
                return "MA50死叉MA200"
        
        return None
    
    def _create_hold_signal(
        self, code: str, name: str, reason: str
    ) -> D2Signal:
        """创建持有信号"""
        return D2Signal(
            code=code,
            name=name,
            signal_type="HOLD",
            strength=0.0,
            conviction=0.0,
            phase=MarketPhase.NEUTRAL,
            entry_reason=reason,
            stop_loss=0.0,
            take_profit=0.0,
            position_size=0.0
        )
    
    # ==================== 轮动评分 ====================
    
    def rank_etfs(self, signals: List[D2Signal]) -> List[D2Signal]:
        """对ETF进行排序，选出最佳持仓"""
        # 过滤有信号的ETF
        valid_signals = [s for s in signals if s.signal_type == "BUY"]
        
        # 按信号强度和信念度排序
        valid_signals.sort(
            key=lambda x: (x.strength * 0.4 + x.conviction * 0.6), 
            reverse=True
        )
        
        return valid_signals
    
    def execute_rebalance(
        self,
        target_signals: List[D2Signal],
        current_prices: Dict[str, float],
        max_holdings: int = 3
    ) -> List[Tuple[str, str, int, float]]:
        """
        执行调仓，返回操作列表
        返回: [(code, action, quantity, price), ...]
        """
        operations = []
        
        # 计算当前持仓代码
        current_codes = set(self.positions.keys())
        target_codes = set(s.code for s in target_signals[:max_holdings])
        
        # 需要卖出的
        sell_codes = current_codes - target_codes
        for code in sell_codes:
            if code in self.positions and code in current_prices:
                pos = self.positions[code]
                operations.append((code, "SELL", pos.quantity, current_prices[code]))
        
        # 需要买入的
        # 计算可用资金 (预留10%现金)
        available = self.cash * 0.90
        buy_count = len(target_codes - current_codes)
        if buy_count > 0:
            per_position = available / min(buy_count, max_holdings)
        
        for signal in target_signals[:max_holdings]:
            if signal.code not in current_codes and signal.code in current_prices:
                price = current_prices[signal.code]
                quantity = int(per_position / price / 100) * 100  # 整手
                if quantity >= 100:
                    operations.append((signal.code, "BUY", quantity, price))
        
        return operations
    
    def update_positions(
        self, operations: List[Tuple[str, str, int, float]]
    ):
        """更新持仓"""
        for code, action, quantity, price in operations:
            if action == "BUY":
                cost = quantity * price
                if code in self.positions:
                    pos = self.positions[code]
                    new_qty = pos.quantity + quantity
                    new_avg = (pos.avg_cost * pos.quantity + cost) / new_qty
                    pos.quantity = new_qty
                    pos.avg_cost = new_avg
                else:
                    self.positions[code] = Position(
                        code=code,
                        name="",
                        quantity=quantity,
                        avg_cost=price
                    )
                self.cash -= cost
                
                self.history.append({
                    "action": "BUY",
                    "code": code,
                    "quantity": quantity,
                    "price": price,
                    "cost": cost,
                    "cash": self.cash
                })
                
            elif action == "SELL":
                if code in self.positions:
                    pos = self.positions[code]
                    revenue = quantity * price
                    self.cash += revenue
                    
                    self.history.append({
                        "action": "SELL",
                        "code": code,
                        "quantity": quantity,
                        "price": price,
                        "revenue": revenue,
                        "cash": self.cash,
                        "pnl": revenue - pos.avg_cost * quantity
                    })
                    
                    pos.quantity -= quantity
                    if pos.quantity <= 0:
                        del self.positions[code]
    
    def sync_prices(self, current_prices: Dict[str, float]):
        """同步持仓价格"""
        for code, pos in self.positions.items():
            if code in current_prices:
                pos.current_price = current_prices[code]
    
    def get_portfolio_stats(self) -> Dict:
        """获取组合统计"""
        total_value = self.cash
        for pos in self.positions.values():
            total_value += pos.quantity * pos.current_price
        
        total_pnl = total_value - self.initial_capital
        total_pnl_pct = (total_value / self.initial_capital - 1) * 100
        
        return {
            "cash": self.cash,
            "positions_value": total_value - self.cash,
            "total_value": total_value,
            "pnl": total_pnl,
            "pnl_pct": total_pnl_pct,
            "position_count": len(self.positions)
        }


# ========== 独立测试 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("D2 防守轮动策略 v2.0")
    print("=" * 60)
    print("\n策略特点:")
    print("  1. MA200判断趋势 + MA50判断入场")
    print("  2. RSI + 布林带 + 动量多因子确认")
    print("  3. 信念度驱动仓位调整")
    print("  4. 跟踪止损保护利润")
    print("\n使用方法:")
    print("  strategy = D2RotationStrategy(initial_capital=10000)")
    print("  signal = strategy.analyze_etf(df, '159915', '创业板ETF')")
    print("=" * 60)