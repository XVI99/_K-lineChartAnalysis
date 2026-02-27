
import os
import sys

# Aggressively Clear Proxies
for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    if key in os.environ:
        del os.environ[key]

import pandas as pd

import numpy as np
import akshare as ak

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_system.backtest_engine import BacktestEngine
from quant_system.pattern_loader import PatternLoader
from quant_system.pattern_processor import StandardizedPatternProcessor
try:
    from quant_system.custom_data import fetch_daily_data
except ImportError:
    from custom_data import fetch_daily_data

def get_historical_data(symbol, start_date='20200101', end_date='20240101'):
    print(f"Fetching data for {symbol}...")
    try:
        # Use custom provider. Note: User provider works by 'count', not date range strictly for Tencent daily.
        # But our adapter 'fetch_daily_data' takes 'days'.
        # We'll request enough days to cover the period (approx 2 years = 730 days)
        # For simplicity in this script, we just ask for specific count or days.
        processed_df = fetch_daily_data(symbol, days=800)
        return processed_df
    except Exception as e:
        print(f"Data error: {e}")
        return pd.DataFrame()

def apply_strategy_logic(df, processor):
    print("Calculating Patterns & Indicators...")
    
    # 1. Run All Patterns (Same as budget_monitor.py)
    # Reversal
    df = processor.run_pattern("engulfing", df)
    df = processor.run_pattern("morning_star", df)
    df = processor.run_pattern("umbrella_lines", df)
    df = processor.run_pattern("shooting_inverted", df)
    # Trend
    df = processor.run_pattern("three_soldiers", df)
    df = processor.run_pattern("belt_hold", df)
    # Indicators
    df = processor.run_pattern("macd", df)
    df = processor.run_pattern("kdj", df)
    df = processor.run_pattern("kdj_signals", df)
    df = processor.run_pattern("rsi_patterns", df)
    df = processor.run_pattern("volume_patterns", df)
    
    # 2. Vectorized Scoring Logic
    # We need to sum up boolean columns. True=1, False=0.
    
    # Initialize Score
    df['Score'] = 0
    
    # --- Weight 3 ---
    df['Score'] += df.get('Morning_Star', 0) * 3
    df['Score'] += df.get('White_Three_Soldiers', 0) * 3
    
    # --- Weight 2 ---
    df['Score'] += df.get('Bull_Engulf', 0) * 2
    df['Score'] += df.get('BottomSignal', 0) * 2 # RSI/MACD Div
    
    # --- Weight 1 (Boosters) ---
    df['Score'] += df.get('Hammer', 0) * 1
    df['Score'] += df.get('InvertedHammer', 0) * 1
    df['Score'] += df.get('Bull_BeltHold', 0) * 1
    df['Score'] += df.get('BuySignal', 0) * 1 # KDJ Gold Cross
    df['Score'] += df.get('RSI_OverSold', 0) * 1
    
    # Volume Booster (Close > Open AND High Vol)
    vol_boost = (df.get('Is_High_Vol', 0) == 1) & (df['Close'] > df['Open'])
    df.loc[vol_boost, 'Score'] += 1
    
    # 3. Generate Signals
    df['Signal'] = 0
    
    # BUY Condition: Score >= 3 (Strict) or >= 4? Let's try 3 for backtest sensitivity.
    # The user manual says "Score >= 4 is Strong Buy", but "3" is also good.
    df.loc[df['Score'] >= 4, 'Signal'] = 1
    
    # SELL Condition: Bearish Patterns OR Stop Loss (handled by engine)
    # detecting sell signals to force exit
    sell_score = 0
    sell_score += df.get('Bear_Engulf', 0)
    sell_score += df.get('ShootingStar', 0)
    sell_score += df.get('Hanging_Man', 0)
    sell_score += df.get('SellSignal', 0) # KDJ Dead
    sell_score += df.get('TopSignal', 0)  # RSI Div Top
    
    df.loc[sell_score >= 1, 'Signal'] = -1
    
    print(f"Signals Generated: Buy={sum(df['Signal']==1)}, Sell={sum(df['Signal']==-1)}")
    return df

def main():
    # Backtest Configuration
    symbol = "000422" # 湖北宜化 (As found in scan)
    # symbol = "600519" # Moutai
    
    start_date = "20230101"
    end_date = "20240201" # 1 Year test
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    k_line_dir = os.path.join(base_dir, "k_line_code")
    
    # 1. Prepare Processor
    processor = StandardizedPatternProcessor(k_line_dir)
    
    # 2. Get Data
    df = get_historical_data(symbol, start_date, end_date)
    if df.empty:
        return
        
    # 3. Apply Strategy
    df = apply_strategy_logic(df, processor)
    
    # 4. Run Backtest Engine
    # Stop Loss: 5%, Trailing Stop: 10%
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.0003,
        slippage=0.002,
        stamp_tax=0.001,       # A-share sell-side stamp tax
        stop_loss_pct=0.05,
        trailing_stop_pct=0.10,
        time_stop_days=8,      # Exit if held >8 days with <2% profit
        execution_mode="next_open"  # Realistic: execute at next bar's open
    )
    equity_curve = engine.run(df)
    
    # 5. Report
    perf = engine.calculate_performance()
    print("\n" + "="*40)
    print(f"BACKTEST RESULTS: {symbol}")
    print("="*40)
    for k, v in perf.items():
        print(f"{k}: {v}")
    print("="*40)

if __name__ == "__main__":
    main()
