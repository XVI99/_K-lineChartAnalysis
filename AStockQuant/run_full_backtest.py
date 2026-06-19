"""
使用新浪API完整数据进行v8.0策略回测
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 尝试导入策略
try:
    from backtest.final_strategy_v8 import FinalOptimizedStrategy
except ImportError:
    print("策略文件不可用，使用内置版本")

class FinalOptimizedStrategy:
    """优化的ETF动量策略"""
    
    def __init__(self, initial_capital=100000, max_positions=3, rebalance_days=10, 
                 stop_loss_pct=0.10, trailing_stop_pct=0.15, cash_reserve=0.20):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.rebalance_days = rebalance_days
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.cash_reserve = cash_reserve
        
    def calculate_indicators(self, df):
        """计算技术指标"""
        df = df.copy()
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA10'] = df['close'].rolling(10).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA50'] = df['close'].rolling(50).mean()
        df['MA200'] = df['close'].rolling(200).mean()
        
        # 动量指标
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_20'] = df['close'].pct_change(20)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
        
        return df
    
    def score_etf(self, row):
        """评分ETF"""
        if pd.isna(row['MA50']) or pd.isna(row['MA200']) or pd.isna(row['RSI']):
            return -100
        
        score = 0
        
        # 趋势评分
        if row['close'] > row['MA50']:
            score += 30
        if row['MA50'] > row['MA200']:
            score += 20
        
        # 动量评分
        momentum = row.get('momentum_20', 0)
        if pd.notna(momentum):
            if momentum > 0.05:
                score += 25
            elif momentum > 0.02:
                score += 15
            elif momentum > 0:
                score += 5
        
        # RSI评分
        rsi = row['RSI']
        if rsi < 30:
            score += 20
        elif rsi < 40:
            score += 10
        elif rsi > 75:
            score -= 30
        
        # 波动率评分
        vol = row.get('ATR', 0)
        price = row['close']
        if pd.notna(vol) and price > 0:
            vol_pct = vol / price
            if vol_pct < 0.02:
                score += 5
        
        return score

def load_all_etf_data():
    """加载data_cache中所有ETF数据"""
    cache_dir = 'data_cache'
    etf_data = {}
    
    if not os.path.exists(cache_dir):
        print(f"目录不存在: {cache_dir}")
        return etf_data
    
    for filename in os.listdir(cache_dir):
        if filename.endswith('.csv') and not filename.startswith('.'):
            filepath = os.path.join(cache_dir, filename)
            try:
                df = pd.read_csv(filepath)
                if 'date' in df.columns and len(df) > 100:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    etf_code = filename.replace('.csv', '')
                    etf_data[etf_code] = df
            except Exception as e:
                continue
    
    print(f"成功加载 {len(etf_data)} 个ETF数据")
    return etf_data

def run_backtest(start_date='2019-01-01', end_date='2024-12-31'):
    """运行回测"""
    print("\n" + "=" * 60)
    print("加载ETF数据...")
    print("=" * 60)
    
    etf_data = load_all_etf_data()
    
    if not etf_data:
        print("没有找到ETF数据")
        return
    
    # 获取共同交易日
    all_dates = set()
    for df in etf_data.values():
        dates = set(df['date'])
        if not all_dates:
            all_dates = dates
        else:
            all_dates &= dates
    
    # 过滤日期范围
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    all_dates = sorted([d for d in all_dates if start_dt <= d <= end_dt])
    
    print(f"共同交易日: {len(all_dates)} 天")
    print(f"日期范围: {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')}")
    
    # 计算所有ETF指标
    print("\n计算技术指标...")
    indicators_data = {}
    for code, df in etf_data.items():
        df_calc = df.copy()
        df_calc = FinalOptimizedStrategy().calculate_indicators(df_calc)
        indicators_data[code] = df_calc.set_index('date')
    
    # 创建统一的日期索引DataFrame
    strategy = FinalOptimizedStrategy()
    
    # 回测参数
    initial_capital = 100000
    cash_reserve = 0.20
    
    # 状态
    cash = initial_capital
    positions = {}  # {code: {'shares': n, 'entry_price': p, 'high_price': p}}
    equity_history = []
    trades = []
    rebalance_counter = 0
    
    print("\n" + "=" * 60)
    print("开始回测...")
    print("=" * 60)
    
    for i, date in enumerate(all_dates):
        # 每天进度
        if i % 200 == 0:
            print(f"  进度: {i}/{len(all_dates)}...")
        
        # 检查止损
        positions_to_close = []
        for code, pos in positions.items():
            if code in indicators_data and date in indicators_data[code].index:
                current_price = indicators_data[code].loc[date, 'close']
                high_price = pos['high_price']
                
                # 跟踪最高价
                if current_price > high_price:
                    pos['high_price'] = current_price
                
                # 止损检查
                entry = pos['entry_price']
                loss_pct = (current_price - entry) / entry
                trailing_trigger = (high_price - current_price) / high_price
                
                if loss_pct <= -0.10:  # 10%止损
                    positions_to_close.append((code, 'stop_loss', loss_pct))
                elif trailing_trigger >= 0.15:  # 15%移动止损
                    positions_to_close.append((code, 'trailing_stop', trailing_trigger))
        
        # 平仓
        for code, reason, loss in positions_to_close:
            pos = positions[code]
            if code in indicators_data and date in indicators_data[code].index:
                current_price = indicators_data[code].loc[date, 'close']
                shares = pos['shares']
                pnl = (current_price - pos['entry_price']) * shares
                cash += current_price * shares
                trades.append({
                    'date': date, 'code': code, 'action': 'sell',
                    'reason': reason, 'price': current_price, 'shares': shares,
                    'pnl': pnl
                })
                del positions[code]
        
        # 再平衡
        rebalance_counter += 1
        if rebalance_counter >= strategy.rebalance_days and len(positions) < strategy.max_positions:
            # 评分所有ETF
            scores = []
            for code in etf_data.keys():
                if code in indicators_data and date in indicators_data[code].index:
                    row = indicators_data[code].loc[date]
                    score = strategy.score_etf(row)
                    if score > 0:
                        scores.append((code, score, row['close']))
            
            # 按评分排序
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # 选择最佳ETF
            current_codes = set(positions.keys())
            for code, score, price in scores:
                if code not in current_codes and len(positions) < strategy.max_positions:
                    # 分配资金买入
                    available = cash * (1 - cash_reserve) / (strategy.max_positions - len(positions))
                    shares = int(available / price / 100) * 100
                    if shares > 0:
                        cost = shares * price
                        if cost <= cash * 0.95:  # 不超过95%现金
                            cash -= cost
                            positions[code] = {
                                'shares': shares, 
                                'entry_price': price, 
                                'high_price': price
                            }
                            trades.append({
                                'date': date, 'code': code, 'action': 'buy',
                                'price': price, 'shares': shares, 'pnl': 0
                            })
            
            if len(positions) < strategy.max_positions:
                rebalance_counter = 0
        
        # 计算当日市值
        total_value = cash
        for code, pos in positions.items():
            if code in indicators_data and date in indicators_data[code].index:
                current_price = indicators_data[code].loc[date, 'close']
                total_value += current_price * pos['shares']
        
        equity_history.append({
            'date': date,
            'equity': total_value,
            'cash': cash,
            'positions': len(positions)
        })
    
    # 计算统计
    equity_df = pd.DataFrame(equity_history)
    equity_df['return'] = equity_df['equity'].pct_change()
    equity_df['cum_return'] = (1 + equity_df['return']).cumprod() - 1
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
    
    # 最终统计
    final_equity = equity_df['equity'].iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    trading_days = len(equity_df)
    years = trading_days / 252
    annualized = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    max_drawdown = equity_df['drawdown'].min()
    
    # 夏普比率
    risk_free = 0.03
    excess_return = equity_df['return'].mean() * 252 - risk_free
    sharpe = excess_return / (equity_df['return'].std() * np.sqrt(252)) if equity_df['return'].std() > 0 else 0
    
    # 胜率
    sell_trades = [t for t in trades if t['action'] == 'sell' and t.get('pnl', 0) != 0]
    wins = sum(1 for t in sell_trades if t['pnl'] > 0)
    win_rate = wins / len(sell_trades) if sell_trades else 0
    
    # 输出结果
    print("\n" + "=" * 60)
    print("  FINAL STRATEGY v8.0 - 完整历史回测结果")
    print("=" * 60)
    print(f"\n回测期间: {start_date} ~ {end_date}")
    print(f"初始资金: {initial_capital:,.0f}")
    print(f"最终资金: {final_equity:,.0f}")
    print("-" * 40)
    print(f"总收益率:    {total_return*100:+.2f}%")
    print(f"年化收益率:  {annualized*100:+.2f}%")
    print(f"最大回撤:    {max_drawdown*100:+.2f}%")
    print(f"夏普比率:    {sharpe:.3f}")
    print(f"胜率:        {win_rate*100:.1f}%")
    print(f"交易次数:    {len(trades)}")
    print("-" * 40)
    
    # 保存结果
    os.makedirs('output', exist_ok=True)
    equity_df.to_csv('output/backtest_2019_2024.csv', index=False)
    
    # 绑图
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(equity_df['date'], equity_df['equity'], 'b-', label='Portfolio')
    plt.plot(equity_df['date'], equity_df['peak'], 'g--', alpha=0.5, label='Peak')
    plt.title('Final Strategy v8.0 - 2019-2024回测')
    plt.ylabel('资金')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.fill_between(equity_df['date'], equity_df['drawdown'] * 100, 0, alpha=0.3, color='red')
    plt.ylabel('回撤 (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/backtest_2019_2024.png', dpi=100)
    print(f"\n图表已保存: output/backtest_2019_2024.png")
    print(f"数据已保存: output/backtest_2019_2024.csv")
    
    return {
        'total_return': total_return,
        'annualized': annualized,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'trades': len(trades)
    }

if __name__ == '__main__':
    results = run_backtest('2019-01-01', '2024-12-31')
