# -*- coding: utf-8 -*-
"""
etf_screener_backtest.py — ETF智能筛选+回测系统

思路：
1. 从全部ETF中扫描数据
2. 根据多维度信号（MA趋势、RSI、动量、成交量）筛选出最具有上涨潜力的ETF
3. 对筛选出的ETF进行回测验证

作者: Matrix Agent
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


@dataclass
class ETFScore:
    """ETF评分数据"""
    code: str
    name: str = ""
    
    # 趋势得分 (0-100)
    trend_score: float = 0.0
    
    # 动量得分 (0-100)
    momentum_score: float = 0.0
    
    # RSI健康度得分 (0-100)
    rsi_score: float = 0.0
    
    # 成交量得分 (0-100)
    volume_score: float = 0.0
    
    # 稳定性得分 (0-100)
    stability_score: float = 0.0
    
    # 综合得分
    total_score: float = 0.0
    
    # 附加数据
    latest_price: float = 0.0
    ma50_slope: float = 0.0      # MA50斜率
    ma200_slope: float = 0.0     # MA200斜率
    rsi_value: float = 50.0
    volume_ratio: float = 1.0
    volatility: float = 0.0
    latest_date: str = ""
    
    def __post_init__(self):
        """计算综合得分"""
        # 权重配置：趋势最重要，其次是动量和RSI
        self.total_score = (
            self.trend_score * 0.30 +
            self.momentum_score * 0.25 +
            self.rsi_score * 0.20 +
            self.volume_score * 0.15 +
            self.stability_score * 0.10
        )


class ETFScreener:
    """ETF筛选器 - 从全量ETF中筛选最佳标的"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def scan_all_etfs(self, lookback_days: int = 200) -> List[ETFScore]:
        """扫描所有ETF并评分"""
        print("=" * 60)
        print("ETF智能筛选系统 v1.0")
        print("=" * 60)
        
        # 获取所有ETF文件
        etf_files = [f for f in self.data_dir.glob("*.csv") 
                     if f.stem.startswith("51") or f.stem.startswith("15") 
                     or f.stem.startswith("16") or f.stem.startswith("50")]
        
        print(f"\n发现 {len(etf_files)} 个ETF数据文件")
        
        # 扫描评分
        scores = []
        for i, file in enumerate(etf_files):
            code = file.stem
            if i % 50 == 0:
                print(f"  进度: {i}/{len(etf_files)} ...")
            
            try:
                score = self._analyze_etf(code, lookback_days)
                if score is not None:
                    scores.append(score)
            except Exception:
                continue
        
        # 排序
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        return scores
    
    def _analyze_etf(self, code: str, lookback_days: int) -> Optional[ETFScore]:
        """分析单个ETF"""
        file_path = self.data_dir / f"{code}.csv"
        
        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            df = df.sort_values('date')
            
            # 检查数据量
            if len(df) < 100:
                return None
            
            # 只取最近lookback_days的数据
            cutoff_date = df['date'].max() - pd.Timedelta(days=lookback_days)
            df = df[df['date'] >= cutoff_date]
            
            if len(df) < 50:
                return None
            
            latest = df.iloc[-1]
            
            score = ETFScore(code=code)
            score.latest_price = latest['close']
            score.latest_date = str(latest['date'])[:10]
            
            # 计算各维度得分
            score = self._calc_trend_score(df, score)
            score = self._calc_momentum_score(df, score)
            score = self._calc_rsi_score(df, score)
            score = self._calc_volume_score(df, score)
            score = self._calc_stability_score(df, score)
            score.__post_init__()
            
            return score
            
        except Exception:
            return None
    
    def _calc_trend_score(self, df: pd.DataFrame, score: ETFScore) -> ETFScore:
        """计算趋势得分"""
        close = df['close'].values
        
        # MA计算
        ma20 = df['close'].rolling(20).mean().values
        ma50 = df['close'].rolling(50).mean().values
        ma200 = df['close'].rolling(200).mean().values
        
        latest_idx = -1
        
        # 1. 价格在均线上方
        price_above_ma50 = close[latest_idx] > ma50[latest_idx] if not np.isnan(ma50[latest_idx]) else False
        price_above_ma200 = close[latest_idx] > ma200[latest_idx] if not np.isnan(ma200[latest_idx]) else False
        
        # 2. 均线多头排列
        ma_bullish = ma50[latest_idx] > ma200[latest_idx] if not (np.isnan(ma50[latest_idx]) or np.isnan(ma200[latest_idx])) else False
        
        # 3. 均线斜率（计算最近20天的变化率）
        if not np.isnan(ma50[latest_idx]) and not np.isnan(ma50[latest_idx-20]):
            score.ma50_slope = (ma50[latest_idx] - ma50[latest_idx-20]) / ma50[latest_idx-20] * 100
        if not np.isnan(ma200[latest_idx]) and not np.isnan(ma200[latest_idx-20]):
            score.ma200_slope = (ma200[latest_idx] - ma200[latest_idx-20]) / ma200[latest_idx-20] * 100
        
        # 4. 最近N天涨幅
        gains = []
        for n in [5, 10, 20, 60]:
            if len(df) > n:
                gains.append((close[latest_idx] - close[latest_idx-n]) / close[latest_idx-n] * 100)
        
        avg_gain = np.mean(gains) if gains else 0
        
        # 计算趋势得分
        trend_points = 0
        if price_above_ma50:
            trend_points += 25
        if price_above_ma200:
            trend_points += 20
        if ma_bullish:
            trend_points += 25
        if score.ma50_slope > 0:
            trend_points += min(20, score.ma50_slope * 5)  # 最多20分
        if avg_gain > 0:
            trend_points += min(10, avg_gain)  # 最多10分
        
        score.trend_score = min(100, max(0, trend_points))
        
        return score
    
    def _calc_momentum_score(self, df: pd.DataFrame, score: ETFScore) -> ETFScore:
        """计算动量得分"""
        close = df['close'].values
        
        # 多周期动量
        mom_5 = (close[-1] - close[-6]) / close[-6] * 100 if len(close) > 5 else 0
        mom_10 = (close[-1] - close[-11]) / close[-11] * 100 if len(close) > 10 else 0
        mom_20 = (close[-1] - close[-21]) / close[-21] * 100 if len(close) > 20 else 0
        
        # 动量持续性（近期动量vs长期动量）
        momentum_accel = mom_5 > mom_10 > mom_20  # 加速还是减速
        
        # 计算动量得分
        mom_points = 0
        mom_points += min(30, max(0, mom_5 * 3))  # 短期动量
        mom_points += min(25, max(0, mom_10 * 2.5))  # 中期动量
        mom_points += min(20, max(0, mom_20 * 2))  # 长期动量
        if momentum_accel:
            mom_points += 25  # 动量加速加分
        
        score.momentum_score = min(100, max(0, mom_points))
        
        return score
    
    def _calc_rsi_score(self, df: pd.DataFrame, score: ETFScore) -> ETFScore:
        """计算RSI健康度得分"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        score.rsi_value = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
        
        # RSI得分：40-60是最佳区间
        rsi_val = score.rsi_value
        
        if 40 <= rsi_val <= 60:
            rsi_points = 100 - abs(rsi_val - 50) * 2  # 50分最高，向两边递减
        elif rsi_val < 40:
            rsi_points = max(0, 30 - (40 - rsi_val))  # 超卖但不过分
        else:  # rsi_val > 60
            rsi_points = max(0, 30 - (rsi_val - 60))  # 超买但不过分
        
        score.rsi_score = rsi_points
        
        return score
    
    def _calc_volume_score(self, df: pd.DataFrame, score: ETFScore) -> ETFScore:
        """计算成交量得分"""
        volume = df['volume'].values
        
        if len(volume) < 20:
            score.volume_score = 50
            return score
        
        # 成交量斜率（上涨趋势中放量）
        vol_ma20 = pd.Series(volume).rolling(20).mean().values
        vol_ma5 = pd.Series(volume).rolling(5).mean().values
        
        if not np.isnan(vol_ma20[-1]) and vol_ma20[-1] > 0:
            score.volume_ratio = vol_ma5[-1] / vol_ma20[-1]
        
        vol_trend = (vol_ma5[-1] - vol_ma20[-1]) / vol_ma20[-1] * 100 if vol_ma20[-1] > 0 else 0
        
        # 成交量得分
        vol_points = 50  # 基准分
        if score.volume_ratio > 1:
            vol_points += min(25, (score.volume_ratio - 1) * 25)
        if vol_trend > 0:
            vol_points += min(25, vol_trend)
        
        score.volume_score = min(100, max(0, vol_points))
        
        return score
    
    def _calc_stability_score(self, df: pd.DataFrame, score: ETFScore) -> ETFScore:
        """计算稳定性得分（低波动率+趋势一致性）"""
        close = df['close'].values
        
        if len(close) < 60:
            score.stability_score = 50
            return score
        
        # 波动率（用收益率标准差）
        returns = pd.Series(close).pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        
        score.volatility = volatility
        
        # 低波动率加分
        vol_points = 0
        if volatility < 0.2:
            vol_points += 50
        elif volatility < 0.3:
            vol_points += 35
        elif volatility < 0.4:
            vol_points += 20
        else:
            vol_points += 5
        
        # 趋势一致性（用R平方）
        x = np.arange(len(close))
        z = np.polyfit(x, close, 1)
        p = np.poly1d(z)
        y_pred = p(x)
        
        ss_res = np.sum((close - y_pred) ** 2)
        ss_tot = np.sum((close - np.mean(close)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 一致性加分
        consistency_points = r_squared * 50
        
        score.stability_score = min(100, vol_points + consistency_points)
        
        return score
    
    def get_top_etfs(self, min_score: float = 30, top_n: int = 10) -> List[ETFScore]:
        """获取评分最高的ETF"""
        all_scores = self.scan_all_etfs()
        
        # 筛选
        filtered = [s for s in all_scores if s.total_score >= min_score]
        
        return filtered[:top_n]


class ScreenerBacktest:
    """筛选后的回测引擎"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, dict] = {}
        self.trades: List[dict] = []
        self.equity_curve: List[dict] = []
        
        # 策略参数
        self.MAX_POSITION_PCT = 0.30    # 单只最大仓位30%
        self.STOP_LOSS_PCT = 0.10        # 止损10%
        self.TAKE_PROFIT_PCT = 0.25      # 止盈25%
        self.MAX_HOLDINGS = 5            # 最大持仓数
        self.REBALANCE_DAYS = 15         # 调仓周期
        
    def run_backtest(self, selected_etfs: List[ETFScore], data_dir: str) -> dict:
        """对筛选出的ETF进行回测"""
        print("\n" + "=" * 60)
        print("筛选型回测")
        print("=" * 60)
        
        print(f"\n选中的ETF ({len(selected_etfs)}只):")
        for i, etf in enumerate(selected_etfs[:10]):
            print(f"  {i+1}. {etf.code} - 综合得分: {etf.total_score:.1f}")
        
        # 获取共同的回测时间段
        start_date, end_date = self._get_common_period(selected_etfs, data_dir)
        print(f"\n回测时间: {start_date} 至 {end_date}")
        
        # 加载所有ETF数据
        etf_data = {}
        for etf in selected_etfs:
            df = self._load_etf_data(etf.code, data_dir)
            if df is not None:
                df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]
                if len(df) > 0:
                    etf_data[etf.code] = df
        
        if len(etf_data) == 0:
            print("错误: 没有可用的ETF数据")
            return {}
        
        # 生成每日的交易信号
        dates = sorted(set.union(*[set(df['date']) for df in etf_data.values()]))
        
        trade_id = 0
        last_rebalance = None
        
        for date in dates:
            # 计算当前持仓市值
            total_value = self.cash + sum(
                p['quantity'] * etf_data[p['code']].iloc[-1]['close'] 
                for p in self.positions.values() 
                if p['code'] in etf_data and len(etf_data[p['code']]) > 0
            )
            
            # 记录权益曲线
            self.equity_curve.append({
                'date': str(date)[:10],
                'value': total_value,
                'pnl_pct': (total_value - self.initial_capital) / self.initial_capital * 100
            })
            
            # 检查是否需要调仓
            days_since_rebalance = 0
            if last_rebalance is not None:
                days_since_rebalance = (date - last_rebalance).days
            
            should_rebalance = (days_since_rebalance >= self.REBALANCE_DAYS or len(self.positions) == 0)
            
            if should_rebalance:
                # 收集所有ETF的信号
                signals = []
                for code, df in etf_data.items():
                    df_before = df[df['date'] <= date]
                    if len(df_before) >= 50:
                        signal = self._generate_signal(df_before, code)
                        signals.append(signal)
                
                if signals:
                    # 按信号强度排序
                    signals.sort(key=lambda x: x['strength'], reverse=True)
                    
                    # 调仓：先清仓弱的，再买入强的
                    self._rebalance(signals, etf_data, date)
                    last_rebalance = date
                
                # 检查止损止盈
            self._check_stop_loss(date, etf_data)
            self._check_take_profit(date, etf_data)
        
        # 计算统计结果
        return self._calculate_stats()
    
    def _load_etf_data(self, code: str, data_dir: str) -> Optional[pd.DataFrame]:
        """加载ETF数据"""
        file_path = Path(data_dir) / f"{code}.csv"
        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            df = df.sort_values('date')
            return df
        except:
            return None
    
    def _get_common_period(self, etfs: List[ETFScore], data_dir: str) -> Tuple[str, str]:
        """获取所有ETF共同的回测时间段"""
        all_dates = []
        
        for etf in etfs:
            df = self._load_etf_data(etf.code, data_dir)
            if df is not None:
                all_dates.append((df['date'].min(), df['date'].max()))
        
        if not all_dates:
            return "2024-01-01", "2024-12-31"
        
        # 取最大重叠区间
        start = max(d[0] for d in all_dates)
        end = min(d[1] for d in all_dates)
        
        # 确保 start < end
        if start > end:
            # 如果所有ETF数据时长不足1年，使用最近1年的数据
            end = max(d[1] for d in all_dates)
            start = end - pd.Timedelta(days=365)
        
        return str(start)[:10], str(end)[:10]
    
    def _generate_signal(self, df: pd.DataFrame, code: str) -> dict:
        """生成交易信号"""
        if len(df) < 50:
            return {'code': code, 'strength': 0, 'action': 'HOLD'}
        
        close = df['close'].values
        latest = df.iloc[-1]
        
        # 计算指标
        ma20 = df['close'].rolling(20).mean().values
        ma50 = df['close'].rolling(50).mean().values
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = (100 - (100 / (1 + rs))).values
        
        # 信号强度
        strength = 0
        
        # 趋势信号
        if close[-1] > ma50[-1]:
            strength += 30
        if ma20[-1] > ma50[-1]:
            strength += 20
        if not np.isnan(ma50[-1]) and not np.isnan(ma50[-20]) and ma50[-1] > ma50[-20]:
            strength += 15
        
        # RSI信号
        if 40 <= rsi[-1] <= 60:
            strength += 25
        elif rsi[-1] < 40:
            strength += 15
        
        # 动量信号
        mom = (close[-1] - close[-20]) / close[-20] * 100 if len(close) > 20 else 0
        if mom > 5:
            strength += min(15, mom)
        
        return {
            'code': code,
            'strength': strength,
            'rsi': rsi[-1],
            'price': close[-1],
            'action': 'BUY' if strength >= 40 else 'HOLD'
        }
    
    def _rebalance(self, signals: List[dict], etf_data: dict, date: pd.Timestamp):
        """调仓操作"""
        trade_id = len(self.trades) + 1
        
        # 买入信号
        buy_signals = [s for s in signals if s['action'] == 'BUY'][:self.MAX_HOLDINGS]
        
        # 分配资金
        if buy_signals:
            per_position = self.cash / len(buy_signals)
        else:
            per_position = 0
        
        # 执行买入
        for signal in buy_signals:
            code = signal['code']
            price = signal['price']
            
            if code in self.positions:
                continue  # 已有持仓
            
            # 计算可买数量
            max_value = self.cash * self.MAX_POSITION_PCT
            quantity = int(max_value / price / 100) * 100  # 整手
            
            if quantity > 0:
                cost = quantity * price
                if cost <= self.cash:
                    self.positions[code] = {
                        'code': code,
                        'quantity': quantity,
                        'avg_cost': price,
                        'buy_date': str(date)[:10]
                    }
                    self.cash -= cost
                    self.trades.append({
                        'date': str(date)[:10],
                        'trade_id': trade_id,
                        'code': code,
                        'action': 'BUY',
                        'quantity': quantity,
                        'price': price,
                        'amount': cost
                    })
                    trade_id += 1
    
    def _check_stop_loss(self, date: pd.Timestamp, etf_data: dict):
        """检查止损"""
        to_sell = []
        
        for code, pos in self.positions.items():
            if code not in etf_data:
                continue
            
            df = etf_data[code]
            df_before = df[df['date'] <= date]
            
            if len(df_before) == 0:
                continue
            
            current_price = df_before.iloc[-1]['close']
            loss_pct = (current_price - pos['avg_cost']) / pos['avg_cost']
            
            if loss_pct <= -self.STOP_LOSS_PCT:
                to_sell.append(code)
        
        for code in to_sell:
            self._sell(code, etf_data[code], date)
    
    def _check_take_profit(self, date: pd.Timestamp, etf_data: dict):
        """检查止盈"""
        to_sell = []
        
        for code, pos in self.positions.items():
            if code not in etf_data:
                continue
            
            df = etf_data[code]
            df_before = df[df['date'] <= date]
            
            if len(df_before) == 0:
                continue
            
            current_price = df_before.iloc[-1]['close']
            profit_pct = (current_price - pos['avg_cost']) / pos['avg_cost']
            
            if profit_pct >= self.TAKE_PROFIT_PCT:
                to_sell.append(code)
        
        for code in to_sell:
            self._sell(code, etf_data[code], date)
    
    def _sell(self, code: str, df: pd.DataFrame, date: pd.Timestamp):
        """卖出操作"""
        if code not in self.positions:
            return
        
        pos = self.positions[code]
        df_before = df[df['date'] <= date]
        
        if len(df_before) == 0:
            return
        
        current_price = df_before.iloc[-1]['close']
        revenue = pos['quantity'] * current_price
        
        self.trades.append({
            'date': str(date)[:10],
            'trade_id': len(self.trades) + 1,
            'code': code,
            'action': 'SELL',
            'quantity': pos['quantity'],
            'price': current_price,
            'amount': revenue
        })
        
        self.cash += revenue
        del self.positions[code]
    
    def _calculate_stats(self) -> dict:
        """计算回测统计"""
        if not self.equity_curve:
            return {}
        
        values = [e['value'] for e in self.equity_curve]
        
        # 收益率
        total_return = (values[-1] - self.initial_capital) / self.initial_capital * 100
        
        # 年化收益率
        first_date = pd.Timestamp(self.equity_curve[0]['date'])
        last_date = pd.Timestamp(self.equity_curve[-1]['date'])
        years = (last_date - first_date).days / 365.0
        annual_return = ((values[-1] / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # 夏普比率
        returns = pd.Series(values).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 最大回撤
        peak = values[0]
        max_drawdown = 0
        for v in values:
            if v > peak:
                peak = v
            drawdown = (peak - v) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 交易统计
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        
        win_count = 0
        for i, sell in enumerate(sell_trades):
            if i < len(buy_trades):
                buy = buy_trades[i]
                if sell['price'] > buy['price']:
                    win_count += 1
        
        win_rate = win_count / len(sell_trades) * 100 if sell_trades else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'buy_trades': len(buy_trades),
            'final_value': values[-1],
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("ETF智能筛选 + 回测系统")
    print("=" * 70)
    
    # 配置
    data_dir = str(Path(__file__).resolve().parent.parent / "data_cache")
    initial_capital = 10000.0
    top_n = 10          # 筛选出前10只
    min_score = 25      # 最低得分阈值
    
    # 步骤1: ETF筛选
    screener = ETFScreener(data_dir)
    top_etfs = screener.get_top_etfs(min_score=min_score, top_n=top_n)
    
    if not top_etfs:
        print("未找到符合条件的ETF")
        return
    
    # 打印筛选结果
    print("\n" + "=" * 70)
    print("ETF筛选结果 (按综合得分排序)")
    print("=" * 70)
    print(f"{'排名':<4} {'代码':<8} {'趋势':<6} {'动量':<6} {'RSI':<6} {'量能':<6} {'稳定':<6} {'综合':<6}")
    print("-" * 70)
    
    for i, etf in enumerate(top_etfs):
        print(f"{i+1:<4} {etf.code:<8} {etf.trend_score:<6.1f} {etf.momentum_score:<6.1f} "
              f"{etf.rsi_value:<6.1f} {etf.volume_ratio:<6.2f} {etf.stability_score:<6.1f} {etf.total_score:<6.1f}")
    
    # 步骤2: 回测
    backtest = ScreenerBacktest(initial_capital=initial_capital)
    results = backtest.run_backtest(top_etfs, data_dir)
    
    if not results:
        print("回测失败")
        return
    
    # 打印结果
    print("\n" + "=" * 70)
    print("筛选型回测结果")
    print("=" * 70)
    print(f"  总收益率:    {results['total_return']:+.2f}%")
    print(f"  年化收益:    {results['annual_return']:+.2f}%")
    print(f"  夏普比率:    {results['sharpe_ratio']:.2f}")
    print(f"  最大回撤:    -{results['max_drawdown']:.2f}%")
    print(f"  胜率:        {results['win_rate']:.1f}%")
    print(f"  总交易次数:  {results['total_trades']}")
    print(f"  期末资金:    {results['final_value']:.2f} 元")
    
    # 保存结果
    output_dir = Path(__file__).resolve().parent.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "screener_backtest_results.json"
    
    # 准备输出数据
    output_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': 'v1.0 筛选型',
        'selected_etfs': [
            {
                'code': e.code,
                'trend_score': e.trend_score,
                'momentum_score': e.momentum_score,
                'rsi_value': e.rsi_value,
                'volume_ratio': e.volume_ratio,
                'stability_score': e.stability_score,
                'total_score': e.total_score
            } for e in top_etfs
        ],
        'backtest_result': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()