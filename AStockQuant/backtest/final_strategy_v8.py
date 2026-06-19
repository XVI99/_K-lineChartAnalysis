"""
最终优化策略 v8.0 - 综合最佳配置
============================================================

基于所有回测结果的最优配置:
- 年化收益: 目标15%+
- 最大回撤: 控制在50%以内
- 夏普比率: 0.8以上

核心策略:
1. 动量选股: 20日动量排名
2. 趋势确认: MA50 > MA200
3. 仓位管理: 最多3持仓, 保留20%现金
4. 止损保护: 10%固定 + 15%跟踪
5. 调仓频率: 10天
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data(codes, start_date='2015-01-01'):
    data_cache = 'F:/_K-lineChartAnalysis/AStockQuant/data_cache'
    all_data = {}
    for code in codes:
        fp = os.path.join(data_cache, f'{code}.csv')
        if os.path.exists(fp):
            df = pd.read_csv(fp, parse_dates=['date'])
            df = df.sort_values('date')
            df = df[df['date'] >= start_date]
            all_data[code] = df
    return all_data

def compute_indicators(df):
    df = df.copy()
    
    # 收益率
    for period in [5, 10, 20, 60]:
        df[f'return_{period}d'] = df['close'].pct_change(period)
    
    # 均线
    for w in [20, 50, 120, 200]:
        df[f'ma{w}'] = df['close'].rolling(w).mean()
    
    # 趋势
    df['trend_up'] = df['close'] > df['ma50']
    df['bull_market'] = df['ma50'] > df['ma200']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 0.001)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 动量
    df['momentum'] = df['return_20d'] * 10 + df['return_60d'] * 3
    
    return df

class FinalOptimizedStrategy:
    """最终优化策略"""
    
    def __init__(self, initial_capital=100000, max_positions=3,
                 rebalance_days=10, stop_loss_pct=0.10,
                 trailing_stop_pct=0.15, cash_reserve=0.20):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.rebalance_days = rebalance_days
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.cash_reserve = cash_reserve
        
        self.cash = initial_capital
        self.equity_curve = []
        self.trades = []
        self.positions = {}
        
    def run(self, all_data, start_date, end_date):
        all_dates = sorted(set().union(*[set(df['date']) for df in all_data.values()]))
        all_dates = [d for d in all_dates if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]
        
        print(f"\n{'='*60}")
        print("  最终优化策略 v8.0 - 综合最佳配置")
        print(f"{'='*60}")
        print(f"\n回测周期: {all_dates[0].date()} ~ {all_dates[-1].date()}")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"持仓数量: {self.max_positions}个")
        print(f"调仓周期: {self.rebalance_days}天")
        print(f"止损线: {self.stop_loss_pct*100:.0f}% | 跟踪: {self.trailing_stop_pct*100:.0f}%")
        print(f"现金保留: {self.cash_reserve*100:.0f}%")
        print(f"\n策略逻辑:")
        print(f"  1. 动量选股: 20日+60日动量加权排名")
        print(f"  2. 趋势确认: MA50 > MA200 (牛市)")
        print(f"  3. 仓位管理: 单只最大60%, 均等分配")
        print(f"  4. 止损保护: 10%止损 + 15%跟踪止损")
        print("-" * 60)
        
        rebalance_counter = 0
        
        for i, date in enumerate(all_dates):
            current_equity = self.cash
            current_prices = {}
            
            # 市场趋势
            market_bull = True
            if '159915' in all_data and date in all_data['159915']['date'].values:
                mkt = all_data['159915']
                mkt_row = mkt[mkt['date'] == date].iloc[0]
                market_bull = mkt_row.get('bull_market', False)
            
            # 更新持仓
            positions_to_remove = []
            
            for code, pos in self.positions.items():
                if code in all_data and date in all_data[code]['date'].values:
                    price = all_data[code].loc[
                        all_data[code]['date'] == date, 'close'].values[0]
                    current_prices[code] = price
                    pos['peak_price'] = max(pos.get('peak_price', pos['entry_price']), price)
                    
                    # 止损检查
                    stop_triggered = (
                        price < pos['entry_price'] * (1 - self.stop_loss_pct) or
                        price < pos['peak_price'] * (1 - self.trailing_stop_pct)
                    )
                    
                    if stop_triggered:
                        self.cash += pos['qty'] * price
                        self.trades.append({
                            'date': date, 'code': code,
                            'action': 'STOP_LOSS' if price < pos['entry_price'] * (1 - self.stop_loss_pct) else 'TRAILING_STOP',
                            'price': price,
                            'qty': pos['qty'],
                            'return': price / pos['entry_price'] - 1,
                            'peak_return': price / pos['entry_price'] - 1
                        })
                        positions_to_remove.append(code)
                        continue
                    
                    current_equity += pos['qty'] * price
            
            for code in positions_to_remove:
                del self.positions[code]
            
            # 调仓 (只在牛市)
            if rebalance_counter >= self.rebalance_days:
                candidates = []
                
                for code, df in all_data.items():
                    if code in self.positions:
                        continue
                    if date not in df['date'].values:
                        continue
                    
                    try:
                        row = df[df['date'] == date].iloc[0]
                        
                        if pd.isna(row.get('momentum', np.nan)):
                            continue
                        
                        # 筛选条件
                        valid = (
                            row.get('trend_up', False) and
                            row.get('momentum', 0) > 0 and
                            row.get('rsi', 50) < 75
                        )
                        
                        if valid:
                            candidates.append({
                                'code': code,
                                'price': row['close'],
                                'momentum': row.get('momentum', 0),
                                'rsi': row.get('rsi', 50)
                            })
                    except:
                        continue
                
                if candidates and market_bull:
                    candidates.sort(key=lambda x: x['momentum'], reverse=True)
                    candidates = candidates[:self.max_positions + 1]
                    
                    # 换仓
                    current = list(self.positions.keys())
                    target = [c['code'] for c in candidates[:self.max_positions]]
                    
                    for code in current:
                        if code not in target:
                            price = current_prices.get(code,
                                all_data[code].loc[all_data[code]['date'] == date, 'close'].values[0])
                            self.cash += self.positions[code]['qty'] * price
                            self.trades.append({
                                'date': date, 'code': code,
                                'action': 'SELL',
                                'price': price,
                                'qty': self.positions[code]['qty'],
                                'return': price / self.positions[code]['entry_price'] - 1
                            })
                            del self.positions[code]
                    
                    # 买入
                    usable = self.cash * (1 - self.cash_reserve)
                    allocation = usable / max(self.max_positions - len(self.positions), 1)
                    
                    for cand in candidates:
                        if len(self.positions) >= self.max_positions:
                            break
                        if cand['code'] in self.positions:
                            continue
                        
                        qty = int(min(allocation, current_equity * 0.6) / cand['price'] / 100) * 100
                        
                        if qty > 0 and qty * cand['price'] <= self.cash * 0.9:
                            self.positions[cand['code']] = {
                                'qty': qty,
                                'entry_price': cand['price'],
                                'entry_date': date,
                                'peak_price': cand['price']
                            }
                            self.cash -= qty * cand['price']
                            self.trades.append({
                                'date': date, 'code': cand['code'],
                                'action': 'BUY',
                                'price': cand['price'],
                                'qty': qty,
                                'momentum': cand['momentum']
                            })
                
                rebalance_counter = 0
            
            rebalance_counter += 1
            
            self.equity_curve.append({
                'date': date,
                'equity': current_equity,
                'cash': self.cash,
                'positions': len(self.positions),
                'market_bull': market_bull
            })
        
        return self.equity_curve, self.trades
    
    def get_stats(self):
        if not self.equity_curve:
            return {}
        
        df = pd.DataFrame(self.equity_curve)
        df['return'] = df['equity'].pct_change()
        df['cum_return'] = (1 + df['return']).cumprod() - 1
        
        days = len(df)
        total_ret = df['cum_return'].iloc[-1]
        annualized = (1 + total_ret) ** (365 / max(days, 1)) - 1
        
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
        max_dd = df['drawdown'].min()
        
        returns = df['return'].dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 计算胜率
        if self.trades:
            buy_trades = [t for t in self.trades if t['action'] == 'BUY']
            sell_trades = [t for t in self.trades if t['action'] in ['SELL', 'STOP_LOSS', 'TRAILING_STOP']]
            wins = [t for t in sell_trades if t.get('return', 0) > 0]
            win_rate = len(wins) / len(sell_trades) if sell_trades else 0
        else:
            win_rate = 0
        
        return {
            'total_return': total_ret,
            'annualized': annualized,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'final_equity': df['equity'].iloc[-1],
            'equity_curve': df
        }

def create_final_chart(stats, equity_df, output_dir):
    """创建最终结果图表"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 权益曲线
    ax1 = axes[0]
    ax1.plot(equity_df['date'], equity_df['equity'], 'b-', linewidth=1.5, label='Equity')
    ax1.plot(equity_df['date'], equity_df['peak'], 'g--', alpha=0.5, label='Peak')
    ax1.fill_between(equity_df['date'], equity_df['peak'], equity_df['equity'],
                     alpha=0.3, color='red', label='Drawdown')
    ax1.set_title(f'Final Optimized Strategy v8.0 - Equity Curve\n'
                  f'Return: {stats["total_return"]*100:+.1f}% | '
                  f'Annual: {stats["annualized"]*100:+.1f}% | '
                  f'MaxDD: {stats["max_drawdown"]*100:.1f}%', fontsize=12)
    ax1.set_ylabel('Equity (CNY)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 回撤曲线
    ax2 = axes[1]
    ax2.fill_between(equity_df['date'], 0, equity_df['drawdown']*100, color='red', alpha=0.5)
    ax2.plot(equity_df['date'], equity_df['drawdown']*100, 'r-', linewidth=1)
    ax2.set_title('Drawdown (%)')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'final_strategy_v8.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {chart_path}")

if __name__ == '__main__':
    etfs = [
        '159915', '159919', '515000', '512000', '512100',
        '512760', '515980', '515050', '515030', '512690'
    ]
    
    print("\n加载数据...")
    all_data = load_data(etfs, '2015-01-01')
    
    print("计算指标...")
    for code in all_data:
        all_data[code] = compute_indicators(all_data[code])
    
    # 运行最终策略
    bt = FinalOptimizedStrategy(
        initial_capital=100000,
        max_positions=3,
        rebalance_days=10,
        stop_loss_pct=0.10,
        trailing_stop_pct=0.15,
        cash_reserve=0.20
    )
    
    equity, trades = bt.run(all_data, '2019-01-01', '2024-12-31')
    stats = bt.get_stats()
    
    print(f"\n{'='*60}")
    print("  最终策略 v8.0 结果")
    print(f"{'='*60}")
    print(f"  总收益率:     {stats['total_return']*100:+.2f}%")
    print(f"  年化收益率:   {stats['annualized']*100:+.2f}%")
    print(f"  最大回撤:     {stats['max_drawdown']*100:.2f}%")
    print(f"  夏普比率:     {stats['sharpe_ratio']:.3f}")
    print(f"  胜率:         {stats['win_rate']*100:.1f}%")
    print(f"  交易次数:     {stats['total_trades']}")
    print(f"  最终资金:     {stats['final_equity']:,.0f}")
    print(f"{'='*60}")
    
    # 保存结果
    output_dir = 'F:/_K-lineChartAnalysis/AStockQuant/output'
    os.makedirs(output_dir, exist_ok=True)
    
    stats['equity_curve'].to_csv(f'{output_dir}/final_strategy_v8.csv', index=False)
    create_final_chart(stats, stats['equity_curve'], output_dir)
    
    print(f"\n结果已保存到: {output_dir}/final_strategy_v8.csv")