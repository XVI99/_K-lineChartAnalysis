import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

def test_fetch(symbol, name):
    print(f"Testing {name} ({symbol})...")
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=100)
    
    # Method 1: Stock Hist
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                start_date=start_dt.strftime("%Y%m%d"), 
                                end_date=end_dt.strftime("%Y%m%d"), 
                                adjust="qfq")
        print(f"  > stock_zh_a_hist: Success, rows={len(df)}")
    except Exception as e:
        print(f"  > stock_zh_a_hist: Failed ({e})")
        
    # Method 2: ETF Hist
    try:
        df = ak.fund_etf_hist_em(symbol=symbol, period="daily", 
                                start_date=start_dt.strftime("%Y%m%d"), 
                                end_date=end_dt.strftime("%Y%m%d"), 
                                adjust="qfq")
        print(f"  > fund_etf_hist_em: Success, rows={len(df)}")
    except Exception as e:
        print(f"  > fund_etf_hist_em: Failed ({e})")

if __name__ == "__main__":
    test_fetch("600519", "Moutai")
    test_fetch("510300", "HS300ETF")
