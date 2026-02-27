import sys
import os
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import mplfinance as mpf
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pattern_processor import StandardizedPatternProcessor
from backtest_engine import BacktestEngine

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_data(symbol, days=500):
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    print(f"Fetching data for {symbol}...")
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                start_date=start_dt.strftime("%Y%m%d"), 
                                end_date=end_dt.strftime("%Y%m%d"), 
                                adjust="qfq")
        df = df.rename(columns={'日期': 'Date', '开盘': 'Open', '最高': 'High',
                                '最低': 'Low', '收盘': 'Close', '成交量': 'Volume'})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        return df
    except Exception as e:
        print(f"Data fetch error: {e}")
        return pd.DataFrame()

def run_strategy(symbol, days=730):
    print(f"\n[{symbol}] Processing...")
    
    # 1. Initialize
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    k_line_dir = os.path.join(base_dir, "k_line_code")
    
    processor = StandardizedPatternProcessor(k_line_dir)
    # Revert to original settings: No strict stop loss to allow breathing room
    engine = BacktestEngine(initial_capital=200000, 
                            stop_loss_pct=None, 
                            trailing_stop_pct=None)
    
    df = get_data(symbol, days=days) 
    
    if df.empty:
        return None

    # 2. Run Patterns
    # engufling
    df = processor.run_pattern("engulfing", df)
    
    # macd + signals
    df = processor.run_pattern("macd", df)
    df = processor.run_pattern("signals", df) 
    
    # belt hold
    df = processor.run_pattern("belt_hold", df)

    # 3. Construct Strategy Signals
    df["Signal"] = 0
    
    def get_col(name):
        return df[name] if name in df.columns else pd.Series(False, index=df.index)

    buy_cond = (get_col("Bull_Engulf") == 1) | \
               (get_col("BottomSignal") == 1) | \
               (get_col("Bull_BeltHold") == 1)
               
    sell_cond = (get_col("Bear_Engulf") == 1) | \
                (get_col("TopSignal") == 1) | \
                (get_col("Bear_BeltHold") == 1)
    
    df.loc[buy_cond, "Signal"] = 1
    df.loc[sell_cond, "Signal"] = -1
    
    # 4. Run Backtest
    equity_df = engine.run(df)
    metrics = engine.calculate_performance()
    
    return metrics, equity_df

def main():
    # Test multiple stocks
    targets = [
        ("600519", "贵州茅台"), # Consumption
        ("601318", "中国平安"), # Finance
        ("300750", "宁德时代"), # New Energy
        ("600030", "中信证券")  # Brokerage (High Volatility)
    ]
    
    summary = []
    
    for code, name in targets:
        try:
            metrics, equity_df = run_strategy(code)
            if metrics:
                print(f"Result for {name}: {metrics['Total Return']}")
                summary.append({
                    "Code": code,
                    "Name": name,
                    "Return": metrics['Total Return'],
                    "Drawdown": metrics['Max Drawdown'],
                    "Sharpe": metrics['Sharpe Ratio'],
                    "Trades": metrics['Trade Count']
                })
        except Exception as e:
            print(f"Error processing {code}: {e}")

    print("\n" + "="*50)
    print(f"{'Code':<10} {'Name':<10} {'Return':<10} {'Drawdown':<10} {'Sharpe':<8}")
    print("-" * 50)
    for res in summary:
        print(f"{res['Code']:<10} {res['Name']:<10} {res['Return']:<10} {res['Drawdown']:<10} {res['Sharpe']:<8}")
    print("="*50)

if __name__ == "__main__":
    main()
