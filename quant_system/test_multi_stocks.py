"""
Multi-Stock Backtest Comparison Tool

Tests the enhanced strategy across multiple stocks to get statistically
significant results. A single stock with 6 trades is not enough to
draw conclusions.

Target: Win rate 85%+, Annualized Return 35%+
"""

import os
import sys
import warnings

# Clear proxies
for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    if key in os.environ:
        del os.environ[key]

import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_system.backtest_engine import BacktestEngine
from quant_system.pattern_processor import StandardizedPatternProcessor
from quant_system.pattern_scorer import score_pattern_quality, DEFAULT_BUY_WEIGHTS
from quant_system.signal_filter import SignalFilter, STRICT_FILTER, RELAXED_FILTER

try:
    from quant_system.custom_data import fetch_daily_data
except ImportError:
    from custom_data import fetch_daily_data

warnings.simplefilter(action='ignore', category=FutureWarning)


# Test stocks - diverse mix of large caps
TEST_STOCKS = [
    ("600519", "贵州茅台"),   # Consumption
    ("601318", "中国平安"),   # Finance
    ("600036", "招商银行"),   # Banking
    ("000858", "五粮液"),     # Consumption
    ("000333", "美的集团"),   # Home Appliances
    ("600900", "长江电力"),   # Utilities
    ("601012", "隆基绿能"),   # New Energy
    ("002594", "比亚迪"),     # EV
    ("600276", "恒瑞医药"),   # Pharma
    ("000422", "湖北宜化"),   # Chemical
]


def run_single_backtest(df, processor, signal_filter=None, engine_config=None):
    """Run backtest on a single stock DataFrame."""
    if df.empty or len(df) < 100:
        return None
    
    # Run patterns
    for pattern in ["engulfing", "morning_star", "umbrella_lines", "shooting_inverted",
                    "three_soldiers", "belt_hold", "macd", "kdj", "kdj_signals", 
                    "rsi_patterns", "volume_patterns"]:
        df = processor.run_pattern(pattern, df)
    
    # Calculate scores
    df['Score'] = 0
    df['Score'] += df.get('Morning_Star', 0) * 3
    df['Score'] += df.get('White_Three_Soldiers', 0) * 3
    df['Score'] += df.get('Bull_Engulf', 0) * 2
    df['Score'] += df.get('BottomSignal', 0) * 2
    df['Score'] += df.get('Hammer', 0) * 1
    df['Score'] += df.get('InvertedHammer', 0) * 1
    df['Score'] += df.get('Bull_BeltHold', 0) * 1
    df['Score'] += df.get('BuySignal', 0) * 1
    df['Score'] += df.get('RSI_OverSold', 0) * 1
    
    vol_boost = (df.get('Is_High_Vol', 0) == 1) & (df['Close'] > df['Open'])
    df.loc[vol_boost, 'Score'] += 1
    
    # Generate signals
    df['Signal'] = 0
    
    if signal_filter:
        # Enhanced strategy with filter
        for i in range(len(df)):
            if df['Score'].iloc[i] >= 4:
                # Calculate pattern quality
                quality = 50
                for pattern_col in DEFAULT_BUY_WEIGHTS.keys():
                    if pattern_col in df.columns and df[pattern_col].iloc[i] == 1:
                        q = score_pattern_quality(df, pattern_col, idx=i)
                        if q > quality:
                            quality = q
                
                passed, _, _ = signal_filter.filter_buy_signal(df, quality, idx=i)
                if passed:
                    df.iloc[i, df.columns.get_loc('Signal')] = 1
    else:
        # Original strategy
        df.loc[df['Score'] >= 4, 'Signal'] = 1
    
    # Sell signals
    sell_score = 0
    sell_score += df.get('Bear_Engulf', 0)
    sell_score += df.get('ShootingStar', 0)
    sell_score += df.get('Hanging_Man', 0)
    sell_score += df.get('SellSignal', 0)
    sell_score += df.get('TopSignal', 0)
    df.loc[sell_score >= 2, 'Signal'] = -1  # Require 2+ sell signals (stricter)
    
    # Run backtest
    engine = BacktestEngine(**engine_config)
    engine.run(df)
    perf = engine.calculate_performance()
    
    return perf, engine.trade_log


def main():
    print("="*70)
    print("MULTI-STOCK BACKTEST COMPARISON")
    print("Testing: Original vs Enhanced Strategy across 10 stocks")
    print("="*70)
    
    # Initialize
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    k_line_dir = os.path.join(base_dir, "k_line_code")
    processor = StandardizedPatternProcessor(k_line_dir)
    
    engine_config = {
        'initial_capital': 100000,
        'commission': 0.0003,
        'slippage': 0.002,
        'stamp_tax': 0.001,
        'stop_loss_pct': 0.05,
        'trailing_stop_pct': 0.10,
        'time_stop_days': 8,
        'execution_mode': 'next_open'
    }
    
    # Results storage
    results_original = []
    results_enhanced = []
    
    # Test each stock
    for symbol, name in TEST_STOCKS:
        print(f"\nTesting {name} ({symbol})...")
        
        try:
            df = fetch_daily_data(symbol, days=800)
            if df.empty:
                print(f"  [Skip] No data")
                continue
            
            # Original strategy
            df_orig = df.copy()
            perf_orig, trades_orig = run_single_backtest(df_orig, processor, None, engine_config)
            if perf_orig:
                results_original.append({
                    'Symbol': symbol,
                    'Name': name,
                    'Return': float(perf_orig['Total Return'].strip('%')),
                    'WinRate': float(perf_orig['Win Rate'].strip('%')),
                    'Trades': perf_orig['Trade Count'],
                    'Sharpe': float(perf_orig['Sharpe Ratio']),
                })
            
            # Enhanced strategy with STRICT filter
            df_enh = df.copy()
            perf_enh, trades_enh = run_single_backtest(df_enh, processor, STRICT_FILTER, engine_config)
            if perf_enh:
                results_enhanced.append({
                    'Symbol': symbol,
                    'Name': name,
                    'Return': float(perf_enh['Total Return'].strip('%')),
                    'WinRate': float(perf_enh['Win Rate'].strip('%')),
                    'Trades': perf_enh['Trade Count'],
                    'Sharpe': float(perf_enh['Sharpe Ratio']),
                })
            
            print(f"  Original: Return={perf_orig['Total Return']}, WinRate={perf_orig['Win Rate']}, Trades={perf_orig['Trade Count']}")
            print(f"  Enhanced: Return={perf_enh['Total Return']}, WinRate={perf_enh['Win Rate']}, Trades={perf_enh['Trade Count']}")
            
        except Exception as e:
            print(f"  [Error] {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY RESULTS")
    print("="*70)
    
    if results_original:
        df_orig = pd.DataFrame(results_original)
        print("\n[ORIGINAL STRATEGY]")
        print(f"  Avg Return: {df_orig['Return'].mean():.2f}%")
        print(f"  Avg Win Rate: {df_orig['WinRate'].mean():.2f}%")
        print(f"  Total Trades: {df_orig['Trades'].sum()}")
        print(f"  Avg Sharpe: {df_orig['Sharpe'].mean():.2f}")
        print(f"  Stocks Profitable: {sum(df_orig['Return'] > 0)}/{len(df_orig)}")
    
    if results_enhanced:
        df_enh = pd.DataFrame(results_enhanced)
        print("\n[ENHANCED STRATEGY (Strict Filter)]")
        print(f"  Avg Return: {df_enh['Return'].mean():.2f}%")
        print(f"  Avg Win Rate: {df_enh['WinRate'].mean():.2f}%")
        print(f"  Total Trades: {df_enh['Trades'].sum()}")
        print(f"  Avg Sharpe: {df_enh['Sharpe'].mean():.2f}")
        print(f"  Stocks Profitable: {sum(df_enh['Return'] > 0)}/{len(df_enh)}")
    
    # Comparison
    if results_original and results_enhanced:
        print("\n[IMPROVEMENT]")
        ret_change = df_enh['Return'].mean() - df_orig['Return'].mean()
        wr_change = df_enh['WinRate'].mean() - df_orig['WinRate'].mean()
        print(f"  Return Change: {ret_change:+.2f}%")
        print(f"  Win Rate Change: {wr_change:+.2f}%")
    
    print("="*70)
    
    # Detailed results table
    print("\n[DETAILED RESULTS]")
    print(f"{'Stock':<12} {'Orig Ret':<10} {'Enh Ret':<10} {'Orig WR':<10} {'Enh WR':<10} {'Orig Trd':<8} {'Enh Trd':<8}")
    print("-"*70)
    
    for i in range(len(results_original)):
        o = results_original[i]
        e = results_enhanced[i] if i < len(results_enhanced) else {'Return': 0, 'WinRate': 0, 'Trades': 0}
        print(f"{o['Name']:<12} {o['Return']:>8.2f}% {e['Return']:>8.2f}% {o['WinRate']:>8.2f}% {e['WinRate']:>8.2f}% {o['Trades']:>6} {e['Trades']:>6}")


if __name__ == "__main__":
    main()
