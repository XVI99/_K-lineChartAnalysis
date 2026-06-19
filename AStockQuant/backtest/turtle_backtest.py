"""
海龟动量策略 v4.0 - 基于经典海龟交易法则 + 动量过滤
参考《海龟交易法则》+ 动量策略原理

核心原则:
1. 纪律性 - 机械化的入场/出场规则
2. 趋势跟随 - 在趋势形成时入场
3. 头寸管理 - ATR-based仓位控制
4. 风险控制 - 2%风险规则, 止损2×ATR

入场: 
- S1: 20日通道突破(短周期)
- S2: 55日通道突破(长周期)
- 需要: 价格>MA200(确认趋势), RPS>70(动量确认)

出场:
- S1: 10日通道反向突破
- S2: 20日通道反向突破

止损: 2×ATR

头寸:
- 单一标的仓位: 账户2%风险对应的股数
- 最大持仓: 2个标的
- 资金利用率: 85-95%
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============== 数据加载 ==============
def load_data(codes, start_date='2015-01-01'):
    """加载多个ETF数据"""
    data_cache = 'F:/_K-lineChartAnalysis/AStockQuant/data_cache'
    
    all_data = {}
    for code in codes:
        fp = os.path.join(data_cache, f'{code}.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp, parse_dates=['date'])
            df = df.sort_values('date')
            df = df[df['date'] >= start_date]
            all_data[code] = df
            print(f"  加载 {code}: {len(df)} 行, {df['date'].min().date()} ~ {df['date'].max().date()}")
    
    return all_data

# ============== 技术指标计算 ==============
def compute_indicators(df):
    """计算策略所需的技术指标"""
    df = df.copy()
    
    # 基础指标
    df['return_1d'] = df['close'].pct_change()
    df['return_5d'] = df['close'].pct_change(5)
    df['return_20d'] = df['close'].pct_change(20)
    
    # 均线系统
    for w in [20, 50, 200]:
        df[f'ma{w}'] = df['close'].rolling(w).mean()
    
    # ATR (Average True Range) - 海龟核心指标
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    df['tr'] = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr20'] = df['tr'].rolling(20).mean()
    df['atr55'] = df['tr'].rolling(55).mean()
    
    # Donchian Channel (海龟核心概念)
    for period in [10, 20, 55]:
        df[f'dc_high_{period}'] = df['high'].rolling(period).max()
        df[f'dc_low_{period}'] = df['low'].rolling(period).min()
        df[f'dc_mid_{period}'] = (df[f'dc_high_{period}'] + df[f'dc_low_{period}']) / 2
    
    # 动量指标
    # RPS: Relative Price Strength (相对价格强度)
    df['rps_20'] = df['close'] / df['close'].rolling(20).min()  # 简化RPS
    
    # 趋势确认
    df['trend_up'] = df['close'] > df['ma200']
    
    # 波动率相对位置
    df['volatility_ratio'] = df['atr20'] / df['ma20']  # ATR/均线比例
    
    return df

# ============== 信号生成 ==============
def generate_signals(df, rps_threshold=1.0, atr_multiplier=2):
    """
    海龟信号生成
    
    入场条件:
    1. S1: 20日通道突破(close > 20日最高)
    2. S2: 55日通道突破(close > 55日最高)  
    3. 趋势确认: close > ma200
    4. 动量确认: return_20d > 0
    
    出场条件:
    1. S1: 10日通道反向突破(close < 10日最低)
    2. S2: 20日通道反向突破(close < 20日最低)
    
    止损: 2×ATR
    """
    df = df.copy()
    
    # 入场信号 - 20日突破
    df['breakout_20'] = df['close'] > df['dc_high_20'].shift(1)
    
    # 入场信号 - 55日突破
    df['breakout_55'] = df['close'] > df['dc_high_55'].shift(1)
    
    # 出场信号 - 10日反向突破
    df['exit_10'] = df['close'] < df['dc_low_10'].shift(1)
    
    # 出场信号 - 20日反向突破
    df['exit_20'] = df['close'] < df['dc_low_20'].shift(1)
    
    # 趋势过滤
    df['in_uptrend'] = df['close'] > df['ma200']
    
    # 动量过滤
    df['has_momentum'] = df['return_20d'] > 0
    
    # 综合信号
    # S1入场: 20日突破 + 趋势确认
    df['signal_s1_buy'] = df['breakout_20'] & df['in_uptrend']
    
    # S2入场: 55日突破 + 趋势确认  
    df['signal_s2_buy'] = df['breakout_55'] & df['in_uptrend']
    
    # S1出场: 10日反向
    df['signal_s1_sell'] = df['exit_10']
    
    # S2出场: 20日反向
    df['signal_s2_sell'] = df['exit_20']
    
    return df

# ============== 仓位管理 (海龟核心) ==============
def calculate_position_size(account_value, atr, price, risk_percent=0.02):
    """
    海龟仓位计算 - 基于ATR的风险控制
    
    核心公式: 头寸单位 = 账户2%风险 / ATR对应金额
    
    例如: 
    - 账户: 100,000
    - 2%风险: 2,000
    - ATR: 0.5元
    - 每股风险: 0.5元
    - 头寸单位: 2,000 / 0.5 = 4,000股
    """
    if atr <= 0 or np.isnan(atr):
        return 0
    
    risk_amount = account_value * risk_percent
    
    # 每份股票的风险 = ATR价格
    risk_per_share = atr
    
    # 计算头寸单位
    units = int(risk_amount / risk_per_share)
    
    # 限制最大单次买入量 (最多10个单位)
    max_units = 10
    units = min(units, max_units)
    
    return units

# ============== 回测引擎 ==============
class TurtleBacktester:
    """海龟策略回测器"""
    
    def __init__(self, initial_capital=100000, max_positions=2, 
                 rebalance_days=5, stop_loss_atr=2.0):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.rebalance_days = rebalance_days
        self.stop_loss_atr = stop_loss_atr
        
        self.cash = initial_capital
        self.equity_curve = []
        self.trades = []
        self.positions = {}  # {code: {'qty': int, 'entry_price': float, 'entry_atr': float, 'signal': 'S1'/'S2'}}
        
    def run(self, all_data, start_date, end_date):
        """执行回测"""
        
        # 合并所有数据到统一时间线
        all_dates = sorted(set()
            .union(*[set(df['date']) for df in all_data.values()]))
        all_dates = [d for d in all_dates if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]
        
        print(f"\n回测周期: {all_dates[0].date()} ~ {all_dates[-1].date()}, 共{len(all_dates)}天")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"最大持仓: {self.max_positions}个标的")
        print(f"调仓周期: {self.rebalance_days}天")
        print("-" * 60)
        
        last_rebalance_day = 0
        daily_records = []
        
        for i, date in enumerate(all_dates):
            # 每日结算
            daily_value = self.cash
            
            # 更新持仓市值
            positions_to_remove = []
            for code, pos in self.positions.items():
                if code in all_data and date in all_data[code]['date'].values:
                    current_price = all_data[code].loc[
                        all_data[code]['date'] == date, 'close'].values[0]
                    
                    # 止损检查 (2×ATR)
                    if 'atr' in all_data[code].columns:
                        atr = all_data[code].loc[
                            all_data[code]['date'] == date, 'atr20'].values[0]
                        stop_price = pos['entry_price'] - self.stop_loss_atr * atr
                        
                        if current_price <= stop_price:
                            # 止损出局
                            self.cash += pos['qty'] * current_price
                            self.trades.append({
                                'date': date, 'code': code,
                                'action': 'STOP_LOSS', 
                                'price': current_price,
                                'qty': pos['qty'],
                                'pnl': (current_price - pos['entry_price']) * pos['qty']
                            })
                            positions_to_remove.append(code)
                            continue
                    
                    daily_value += pos['qty'] * current_price
            
            for code in positions_to_remove:
                del self.positions[code]
            
            # 检查出场信号
            positions_to_remove = []
            for code, pos in self.positions.items():
                if code in all_data and date in all_data[code]['date'].values:
                    df = all_data[code]
                    row = df[df['date'] == date].iloc[0]
                    
                    # S1出场: 10日反向突破, S2出场: 20日反向突破
                    if pos['signal'] == 'S1' and row.get('signal_s1_sell', False):
                        self.cash += pos['qty'] * row['close']
                        self.trades.append({
                            'date': date, 'code': code,
                            'action': 'EXIT_S1', 
                            'price': row['close'],
                            'qty': pos['qty'],
                            'pnl': (row['close'] - pos['entry_price']) * pos['qty']
                        })
                        positions_to_remove.append(code)
                    elif pos['signal'] == 'S2' and row.get('signal_s2_sell', False):
                        self.cash += pos['qty'] * row['close']
                        self.trades.append({
                            'date': date, 'code': code,
                            'action': 'EXIT_S2', 
                            'price': row['close'],
                            'qty': pos['qty'],
                            'pnl': (row['close'] - pos['entry_price']) * pos['qty']
                        })
                        positions_to_remove.append(code)
            
            for code in positions_to_remove:
                if code in self.positions:
                    del self.positions[code]
            
            # 调仓检查 (每N天)
            should_rebalance = (i - last_rebalance_day) >= self.rebalance_days
            
            if should_rebalance and len(self.positions) < self.max_positions:
                # 找新的入场机会
                candidates = []
                
                for code, df in all_data.items():
                    if code in self.positions:
                        continue
                    if date not in df['date'].values:
                        continue
                    
                    row = df[df['date'] == date].iloc[0]
                    
                    # 计算RPS和动量
                    if pd.notna(row.get('return_20d', 0)):
                        rps = 1 + row['return_20d'] * 5  # 简化RPS
                    else:
                        rps = 1
                    
                    candidates.append({
                        'code': code,
                        'price': row['close'],
                        'atr': row.get('atr20', row.get('atr55', row['close'] * 0.02)),
                        'signal': None,
                        'momentum': row.get('return_20d', 0)
                    })
                    
                    # 确定信号类型
                    if row.get('signal_s2_buy', False):
                        candidates[-1]['signal'] = 'S2'
                    elif row.get('signal_s1_buy', False):
                        candidates[-1]['signal'] = 'S1'
                
                # 按动量排序
                candidates.sort(key=lambda x: x['momentum'], reverse=True)
                
                # 入场 (选择动量最强的)
                for cand in candidates:
                    if len(self.positions) >= self.max_positions:
                        break
                    if cand['signal'] is None:  # 只在有明确信号时入场
                        continue
                    if cand['atr'] <= 0 or np.isnan(cand['atr']):
                        continue
                    
                    # 海龟仓位计算
                    units = calculate_position_size(daily_value, cand['atr'], cand['price'])
                    cost = units * cand['price']
                    
                    if cost <= self.cash * 0.95:  # 最多用95%资金
                        self.positions[cand['code']] = {
                            'qty': units,
                            'entry_price': cand['price'],
                            'entry_atr': cand['atr'],
                            'signal': cand['signal'],
                            'entry_date': date
                        }
                        self.cash -= cost
                        self.trades.append({
                            'date': date, 'code': cand['code'],
                            'action': 'BUY', 
                            'price': cand['price'],
                            'qty': units,
                            'signal': cand['signal']
                        })
                        last_rebalance_day = i
            
            self.equity_curve.append({
                'date': date,
                'equity': daily_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })
        
        return self.equity_curve, self.trades
    
    def get_stats(self):
        """计算绩效指标"""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        
        # 计算收益率
        equity_df['return'] = equity_df['equity'].pct_change()
        equity_df['cum_return'] = (1 + equity_df['return']).cumprod() - 1
        
        # 年化收益率
        total_days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        total_return = equity_df['cum_return'].iloc[-1]
        annualized = (1 + total_return) ** (365 / max(total_days, 1)) - 1
        
        # 最大回撤
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # 夏普比率 (简化)
        if equity_df['return'].std() > 0:
            sharpe = equity_df['return'].mean() / equity_df['return'].std() * np.sqrt(252)
        else:
            sharpe = 0
        
        return {
            'total_return': total_return,
            'annualized': annualized,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_trades': len(self.trades),
            'final_equity': equity_df['equity'].iloc[-1],
            'equity_curve': equity_df
        }

# ============== 主程序 ==============
if __name__ == '__main__':
    print("="*60)
    print("  海龟动量策略 v4.0 - 基于经典量化交易法则")
    print("="*60)
    
    # ETF列表
    etfs = [
        '159915',  # 创业板
        '159919',  # 沪深300
        '515000',  # 科技ETF
        '512000',  # 证券ETF
        '512100',  # 军工ETF
        '512760',  # 芯片ETF
        '515980',  # 人工智能
        '515050',  # 5G ETF
    ]
    
    # 加载数据
    print("\n加载ETF数据...")
    all_data = load_data(etfs, '2015-01-01')
    
    # 计算指标
    print("\n计算技术指标...")
    for code, df in all_data.items():
        all_data[code] = compute_indicators(df)
        all_data[code] = generate_signals(all_data[code])
    
    # 运行回测
    print("\n" + "="*60)
    print("  激进配置 (高仓位, 集中持仓)")
    print("="*60)
    
    backtester = TurtleBacktester(
        initial_capital=100000,
        max_positions=2,        # 最多2个持仓 (集中)
        rebalance_days=5,       # 每5天调仓 (激进)
        stop_loss_atr=2.0       # 2×ATR止损
    )
    
    equity, trades = backtester.run(all_data, '2019-01-01', '2024-12-31')
    stats = backtester.get_stats()
    
    print("\n" + "="*60)
    print("  回测结果")
    print("="*60)
    print(f"  总收益率:     {stats['total_return']*100:+.2f}%")
    print(f"  年化收益率:   {stats['annualized']*100:+.2f}%")
    print(f"  最大回撤:     {stats['max_drawdown']*100:.2f}%")
    print(f"  夏普比率:     {stats['sharpe_ratio']:.3f}")
    print(f"  交易次数:     {stats['total_trades']}")
    print(f"  最终资金:     {stats['final_equity']:,.0f}")
    print("="*60)
    
    # 保存结果
    output_file = 'F:/_K-lineChartAnalysis/AStockQuant/output/turtle_backtest_results.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    stats['equity_curve'].to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")