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

# Try to clear proxy settings that might be causing connection errors
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# Allowed prefixes for beginners
MAIN_BOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")

def is_beginner_friendly(symbol):
    return symbol.startswith(MAIN_BOARD_PREFIXES)

def get_scan_list():
    """Get stocks for beginner scanning."""
    scan_list = []
    
    # Main Board Stocks (Top weights from HS300)
    print("Fetching HS300 list for Main Board Stocks...")
    try:
        df = ak.index_stock_cons_weight_csindex(symbol="000300")
        df = df.sort_values("权重", ascending=False).head(50) # Scan top 50
        
        for _, row in df.iterrows():
            code = row["成分券代码"]
            name = row["成分券名称"]
            if is_beginner_friendly(code):
                scan_list.append({"code": code, "name": name, "type": "Stock"})
    except Exception as e:
        print(f"HS300 API failed ({e}), using fallback list.")
        # Fallback list of famous main board stocks
        static_stocks = [
            ("600519", "贵州茅台"), ("601318", "中国平安"), ("600036", "招商银行"), 
            ("601888", "中国中免"), ("000333", "美的集团"), ("000858", "五粮液"), 
            ("002594", "比亚迪"), ("600900", "长江电力"), ("601012", "隆基绿能"),
            ("600276", "恒瑞医药")
        ]
        for code, name in static_stocks:
             scan_list.append({"code": code, "name": name, "type": "Stock"})
             
    return scan_list

def get_data(symbol, days=500):
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    try:
        # Try stock interface for everything first (most robust)
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
        print(f"DEBUG: Stock API failed for {symbol}: {e}")
        try:
             # Fallback to ETF specific API
             df = ak.fund_etf_hist_em(symbol=symbol, period="daily", 
                                start_date=start_dt.strftime("%Y%m%d"), 
                                end_date=end_dt.strftime("%Y%m%d"), 
                                adjust="qfq")
             df = df.rename(columns={'日期': 'Date', '开盘': 'Open', '最高': 'High',
                                '最低': 'Low', '收盘': 'Close', '成交量': 'Volume'})
             df['Date'] = pd.to_datetime(df['Date'])
             df = df.set_index('Date').sort_index()
             return df
        except Exception as e2:
             print(f"DEBUG: All Data fetch failed for {symbol}: {e2}")
             return pd.DataFrame()

def run_strategy(processor, engine, df):
    if df.empty:
        print("DEBUG: DF is empty")
        return None
    if len(df) < 100:
        print(f"DEBUG: Not enough data: {len(df)}")
        return None

    # 1. Run Patterns
    try:
        df = processor.run_pattern("engulfing", df)
        df = processor.run_pattern("macd", df)
        df = processor.run_pattern("signals", df) 
        df = processor.run_pattern("belt_hold", df)
    except Exception as e:
        print(f"DEBUG: Pattern run failed: {e}")
        return None

    # 2. Construct Signals
    df["Signal"] = 0
    def get_col(name): return df[name] if name in df.columns else pd.Series(False, index=df.index)

    buy_cond = (get_col("Bull_Engulf") == 1) | (get_col("BottomSignal") == 1) | (get_col("Bull_BeltHold") == 1)
    sell_cond = (get_col("Bear_Engulf") == 1) | (get_col("TopSignal") == 1) | (get_col("Bear_BeltHold") == 1)
    
    df.loc[buy_cond, "Signal"] = 1
    df.loc[sell_cond, "Signal"] = -1
    
    # 3. Backtest
    engine.reset()
    equity_df = engine.run(df)
    metrics = engine.calculate_performance()
    return metrics

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    k_line_dir = os.path.join(base_dir, "k_line_code")
    
    processor = StandardizedPatternProcessor(k_line_dir)
    engine = BacktestEngine(initial_capital=100000) # Reset to 100k for easier comparison
    
    items = get_scan_list()
    print(f"Scanning {len(items)} items (Stocks & ETFs)...")
    
    results = []
    
    for item in tqdm(items):
        df = get_data(item["code"], days=730)
        metrics = run_strategy(processor, engine, df)
        
        if metrics:
            results.append({
                "Type": item["type"],
                "Code": item["code"],
                "Name": item["name"],
                "Return": float(metrics['Total Return'].strip('%')),
                "Drawdown": float(metrics['Max Drawdown'].strip('%')),
                "Sharpe": float(metrics['Sharpe Ratio']),
                "Trades": metrics['Trade Count']
            })
    
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # Print ETF Results
        print("\n" + "="*70)
        print("ETF PERFORMANCE (Comparison)")
        print("="*70)
        etfs = res_df[res_df["Type"]=="ETF"].sort_values("Return", ascending=False)
        print(etfs.to_string(index=False))
        
        # Print Stock Results
        print("\n" + "="*70)
        print("MAIN BOARD STOCKS (Beginner Friendly)")
        print("="*70)
        stocks = res_df[res_df["Type"]=="Stock"].sort_values("Return", ascending=False).head(10)
        print(stocks.to_string(index=False))
        
        res_df.to_csv("beginner_scan.csv", index=False)
    else:
        print("No results.")

if __name__ == "__main__":
    main()
