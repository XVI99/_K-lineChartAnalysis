import sys
import os
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pattern_processor import StandardizedPatternProcessor
from backtest_engine import BacktestEngine

# Suppress pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_hs300_list():
    """Get Top 50 stocks from HS300 to save time for demo."""
    print("Fetching HS300 list...")
    try:
        df = ak.index_stock_cons_weight_csindex(symbol="000300")
        # Return top 30 by weight to ensure we process high-quality stocks
        # Column names might vary, usually "成分券代码"
        return df.sort_values("权重", ascending=False).head(30)["成分券代码"].tolist()
    except:
        # Fallback list if API fails
        return ["600519", "601318", "300750", "600030", "600036", "600900", "000858", "002594", "601888", "601012"]

def get_data(symbol, days=500):
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
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
    except:
        return pd.DataFrame()

def run_strategy_for_scan(processor, engine, df):
    if df.empty or len(df) < 100:
        return None

    # 1. Run Patterns
    try:
        df = processor.run_pattern("engulfing", df)
        df = processor.run_pattern("macd", df)
        df = processor.run_pattern("signals", df) 
        df = processor.run_pattern("belt_hold", df)
    except Exception:
        return None

    # 2. Construct Signals
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
    
    # 3. Backtest
    # Use simple settings (no stop loss) as per optimization result
    engine.reset()
    equity_df = engine.run(df)
    metrics = engine.calculate_performance()
    metrics["Equity_Curve"] = equity_df # Store for plotting if needed
    
    return metrics

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    k_line_dir = os.path.join(base_dir, "k_line_code")
    
    processor = StandardizedPatternProcessor(k_line_dir)
    engine = BacktestEngine(initial_capital=200000)
    
    stock_list = get_hs300_list()
    print(f"Scanning {len(stock_list)} stocks...")
    
    results = []
    
    for symbol in tqdm(stock_list):
        df = get_data(symbol, days=730)
        metrics = run_strategy_for_scan(processor, engine, df)
        
        if metrics:
            # Parse percentage strings to floats for sorting
            total_ret = float(metrics['Total Return'].strip('%'))
            drawdown = float(metrics['Max Drawdown'].strip('%'))
            
            results.append({
                "Symbol": symbol,
                "Return": total_ret,
                "Drawdown": drawdown,
                "Sharpe": float(metrics['Sharpe Ratio']),
                "Trades": metrics['Trade Count']
            })
    
    # Convert to DataFrame and Rank
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # Filter: Positive Return and Reasonable Trades (> 5)
        good_stocks = res_df[ (res_df["Return"] > 0) & (res_df["Trades"] >= 5) ]
        top_stocks = good_stocks.sort_values("Return", ascending=False).head(10)
        
        print("\n" + "="*60)
        print("TOP 10 STOCKS FOR THIS STRATEGY (Past 2 Years)")
        print("="*60)
        print(top_stocks.to_string(index=False))
        print("="*60)
        
        # Save results
        res_df.to_csv("scan_results.csv", index=False)
        print(f"Full scan results saved to 'scan_results.csv'.")
    else:
        print("No valid results found.")

if __name__ == "__main__":
    main()
