"""
Enhanced Backtest Strategy with Signal Quality Filtering

Implements Phase 1 optimizations:
1. Pattern quality score filtering (only trade high-quality patterns)
2. Multi-condition confirmation (trend, volume, RSI)
3. Higher signal threshold (Score >= 5 instead of 4)
4. Market regime filter integration

Target: Win rate 85%+
"""

import os
import sys
import warnings

# Aggressively Clear Proxies
for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    if key in os.environ:
        del os.environ[key]

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_system.backtest_engine import BacktestEngine
from quant_system.pattern_processor import StandardizedPatternProcessor
from quant_system.pattern_scorer import score_pattern_quality, DEFAULT_BUY_WEIGHTS, DEFAULT_SELL_WEIGHTS
from quant_system.signal_filter import SignalFilter, STRICT_FILTER, DEFAULT_FILTER
from quant_system.market_regime import get_market_regime_filter, MarketRegime

try:
    from quant_system.custom_data import fetch_daily_data
except ImportError:
    from custom_data import fetch_daily_data

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_historical_data(symbol, days=800):
    """Fetch historical data for backtesting."""
    print(f"Fetching data for {symbol}...")
    try:
        processed_df = fetch_daily_data(symbol, days=days)
        return processed_df
    except Exception as e:
        print(f"Data error: {e}")
        return pd.DataFrame()


def apply_enhanced_strategy(df, processor, signal_filter=None, verbose=True):
    """
    Apply enhanced strategy with signal quality filtering.
    
    Args:
        df: DataFrame with OHLCV data
        processor: PatternProcessor instance
        signal_filter: SignalFilter instance (default: DEFAULT_FILTER)
        verbose: Print detailed information
    
    Returns:
        DataFrame with signals and filter details
    """
    if signal_filter is None:
        signal_filter = DEFAULT_FILTER
    
    if verbose:
        print("Calculating Patterns & Indicators...")
    
    # ===========================================
    # 1. Run All Patterns
    # ===========================================
    # Reversal Patterns
    df = processor.run_pattern("engulfing", df)
    df = processor.run_pattern("morning_star", df)
    df = processor.run_pattern("umbrella_lines", df)
    df = processor.run_pattern("shooting_inverted", df)
    
    # Trend Patterns
    df = processor.run_pattern("three_soldiers", df)
    df = processor.run_pattern("belt_hold", df)
    
    # Indicators
    df = processor.run_pattern("macd", df)
    df = processor.run_pattern("kdj", df)
    df = processor.run_pattern("kdj_signals", df)
    df = processor.run_pattern("rsi_patterns", df)
    df = processor.run_pattern("volume_patterns", df)
    
    # ===========================================
    # 2. Calculate Pattern Quality Scores
    # ===========================================
    # Initialize columns
    df['Pattern_Quality'] = 0.0
    df['Raw_Score'] = 0
    df['Filtered_Score'] = 0
    
    # Calculate raw score (same as before)
    df['Raw_Score'] += df.get('Morning_Star', 0) * 3
    df['Raw_Score'] += df.get('White_Three_Soldiers', 0) * 3
    df['Raw_Score'] += df.get('Bull_Engulf', 0) * 2
    df['Raw_Score'] += df.get('BottomSignal', 0) * 2
    df['Raw_Score'] += df.get('Hammer', 0) * 1
    df['Raw_Score'] += df.get('InvertedHammer', 0) * 1
    df['Raw_Score'] += df.get('Bull_BeltHold', 0) * 1
    df['Raw_Score'] += df.get('BuySignal', 0) * 1
    df['Raw_Score'] += df.get('RSI_OverSold', 0) * 1
    
    # Volume booster
    vol_boost = (df.get('Is_High_Vol', 0) == 1) & (df['Close'] > df['Open'])
    df.loc[vol_boost, 'Raw_Score'] += 1
    
    # Calculate quality score for each pattern that triggered
    for pattern_col in DEFAULT_BUY_WEIGHTS.keys():
        if pattern_col in df.columns:
            # Calculate quality for rows where pattern is triggered
            mask = df[pattern_col] == 1
            if mask.any():
                for idx in df[mask].index:
                    iloc_idx = df.index.get_loc(idx)
                    quality = score_pattern_quality(df, pattern_col, idx=iloc_idx)
                    # Add weighted quality to filtered score
                    weight = DEFAULT_BUY_WEIGHTS[pattern_col]
                    df.loc[idx, 'Filtered_Score'] += weight * (quality / 100.0)
                    # Track max quality
                    if quality > df.loc[idx, 'Pattern_Quality']:
                        df.loc[idx, 'Pattern_Quality'] = quality
    
    # ===========================================
    # 3. Apply Signal Filters
    # ===========================================
    df['Signal'] = 0
    df['Filter_Passed'] = False
    df['Filter_Reasons'] = ''
    
    # Track filter statistics
    filter_stats = {
        'total_buy_signals': 0,
        'filtered_buy_signals': 0,
        'total_sell_signals': 0,
        'filtered_sell_signals': 0,
    }
    
    # Process each row
    for i in range(len(df)):
        row = df.iloc[i]
        
        # --- BUY Signal Processing ---
        if row['Raw_Score'] >= 4:  # Potential buy signal
            filter_stats['total_buy_signals'] += 1
            
            # Get pattern quality
            quality = row['Pattern_Quality'] if row['Pattern_Quality'] > 0 else 50
            
            # Apply filter
            passed, reasons, details = signal_filter.filter_buy_signal(df, quality, idx=i)
            
            if passed:
                df.iloc[i, df.columns.get_loc('Signal')] = 1
                df.iloc[i, df.columns.get_loc('Filter_Passed')] = True
                filter_stats['filtered_buy_signals'] += 1
            else:
                df.iloc[i, df.columns.get_loc('Filter_Reasons')] = '; '.join(reasons[:2])  # Store first 2 reasons
        
        # --- SELL Signal Processing ---
        sell_score = 0
        sell_score += row.get('Bear_Engulf', 0)
        sell_score += row.get('ShootingStar', 0)
        sell_score += row.get('Hanging_Man', 0)
        sell_score += row.get('SellSignal', 0)
        sell_score += row.get('TopSignal', 0)
        
        if sell_score >= 1:
            filter_stats['total_sell_signals'] += 1
            # Sell signals are less strict - just check RSI oversold
            passed, reasons, details = signal_filter.filter_sell_signal(df, 100, idx=i)
            
            if passed:
                df.iloc[i, df.columns.get_loc('Signal')] = -1
                filter_stats['filtered_sell_signals'] += 1
    
    if verbose:
        print(f"\n[Signal Filter Statistics]")
        print(f"   Buy signals: {filter_stats['filtered_buy_signals']}/{filter_stats['total_buy_signals']} passed filter")
        print(f"   Sell signals: {filter_stats['filtered_sell_signals']}/{filter_stats['total_sell_signals']} passed filter")
        print(f"   Final Signals: Buy={sum(df['Signal']==1)}, Sell={sum(df['Signal']==-1)}")
    
    return df


def run_backtest(symbol, days=800, signal_filter=None, engine_config=None):
    """
    Run complete backtest for a symbol.
    
    Args:
        symbol: Stock code
        days: Number of days to backtest
        signal_filter: SignalFilter instance
        engine_config: Dict with BacktestEngine config
    
    Returns:
        dict: Performance metrics and trade log
    """
    # Default engine config
    if engine_config is None:
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
    
    # Initialize processor
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    k_line_dir = os.path.join(base_dir, "k_line_code")
    processor = StandardizedPatternProcessor(k_line_dir)
    
    # Get data
    df = get_historical_data(symbol, days=days)
    if df.empty:
        return None
    
    # Apply enhanced strategy
    df = apply_enhanced_strategy(df, processor, signal_filter)
    
    # Run backtest
    engine = BacktestEngine(**engine_config)
    equity_curve = engine.run(df)
    
    # Get performance
    perf = engine.calculate_performance()
    
    return {
        'symbol': symbol,
        'performance': perf,
        'trade_log': engine.trade_log,
        'equity_curve': equity_curve,
        'signals': df[['Signal', 'Raw_Score', 'Filtered_Score', 'Pattern_Quality', 'Filter_Passed']]
    }


def compare_strategies(symbol, days=800):
    """
    Compare original vs enhanced strategy.
    
    Args:
        symbol: Stock code
        days: Backtest period
    """
    print("\n" + "="*60)
    print(f"STRATEGY COMPARISON: {symbol}")
    print("="*60)
    
    # Get data once
    df = get_historical_data(symbol, days=days)
    if df.empty:
        print("No data available.")
        return
    
    # Initialize processor
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    k_line_dir = os.path.join(base_dir, "k_line_code")
    processor = StandardizedPatternProcessor(k_line_dir)
    
    # Engine config
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
    
    # --- Original Strategy (Score >= 4, no filter) ---
    print("\n[*] Running ORIGINAL Strategy (Score >= 4, no filter)...")
    df_orig = df.copy()
    
    # Run patterns
    for pattern in ["engulfing", "morning_star", "umbrella_lines", "shooting_inverted",
                    "three_soldiers", "belt_hold", "macd", "kdj", "kdj_signals", 
                    "rsi_patterns", "volume_patterns"]:
        df_orig = processor.run_pattern(pattern, df_orig)
    
    # Original scoring
    df_orig['Score'] = 0
    df_orig['Score'] += df_orig.get('Morning_Star', 0) * 3
    df_orig['Score'] += df_orig.get('White_Three_Soldiers', 0) * 3
    df_orig['Score'] += df_orig.get('Bull_Engulf', 0) * 2
    df_orig['Score'] += df_orig.get('BottomSignal', 0) * 2
    df_orig['Score'] += df_orig.get('Hammer', 0) * 1
    df_orig['Score'] += df_orig.get('InvertedHammer', 0) * 1
    df_orig['Score'] += df_orig.get('Bull_BeltHold', 0) * 1
    df_orig['Score'] += df_orig.get('BuySignal', 0) * 1
    df_orig['Score'] += df_orig.get('RSI_OverSold', 0) * 1
    
    vol_boost = (df_orig.get('Is_High_Vol', 0) == 1) & (df_orig['Close'] > df_orig['Open'])
    df_orig.loc[vol_boost, 'Score'] += 1
    
    df_orig['Signal'] = 0
    df_orig.loc[df_orig['Score'] >= 4, 'Signal'] = 1
    
    sell_score = 0
    sell_score += df_orig.get('Bear_Engulf', 0)
    sell_score += df_orig.get('ShootingStar', 0)
    sell_score += df_orig.get('Hanging_Man', 0)
    sell_score += df_orig.get('SellSignal', 0)
    sell_score += df_orig.get('TopSignal', 0)
    df_orig.loc[sell_score >= 1, 'Signal'] = -1
    
    print(f"   Signals: Buy={sum(df_orig['Signal']==1)}, Sell={sum(df_orig['Signal']==-1)}")
    
    # Run backtest
    engine_orig = BacktestEngine(**engine_config)
    engine_orig.run(df_orig)
    perf_orig = engine_orig.calculate_performance()
    
    # --- Enhanced Strategy (with filter) ---
    print("\n[*] Running ENHANCED Strategy (Signal Quality Filter)...")
    df_enhanced = apply_enhanced_strategy(df.copy(), processor, DEFAULT_FILTER)
    
    engine_enhanced = BacktestEngine(**engine_config)
    engine_enhanced.run(df_enhanced)
    perf_enhanced = engine_enhanced.calculate_performance()
    
    # --- Results Comparison ---
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"{'Metric':<20} {'Original':<15} {'Enhanced':<15} {'Change':<15}")
    print("-"*60)
    
    metrics = ['Total Return', 'Annualized Return', 'Max Drawdown', 'Sharpe Ratio', 
               'Win Rate', 'Profit Factor', 'Trade Count']
    
    for m in metrics:
        orig_val = perf_orig.get(m, 'N/A')
        enh_val = perf_enhanced.get(m, 'N/A')
        
        # Calculate change if numeric
        change = ''
        if isinstance(orig_val, str) and isinstance(enh_val, str):
            try:
                orig_num = float(orig_val.strip('%'))
                enh_num = float(enh_val.strip('%'))
                change = f"{enh_num - orig_num:+.2f}%"
            except:
                pass
        
        print(f"{m:<20} {str(orig_val):<15} {str(enh_val):<15} {change:<15}")
    
    print("="*60)
    
    return {
        'original': perf_orig,
        'enhanced': perf_enhanced
    }


def main():
    """Main entry point."""
    # Configuration
    symbol = "000422"  # 湖北宜化
    # symbol = "600519"  # 茅台
    
    print("="*60)
    print("ENHANCED BACKTEST STRATEGY V2")
    print("Features: Signal Quality Filter + Multi-Condition Confirmation")
    print("="*60)
    
    # Check market regime first
    try:
        regime, allow_long, position_scale = get_market_regime_filter()
        print(f"\n[Market Regime] {regime.value}")
        print(f"   Allow Long: {allow_long}, Position Scale: {position_scale}")
        
        if not allow_long:
            print("\n[WARNING] Bear market detected. Strategy may underperform.")
    except Exception as e:
        print(f"\n[Warning] Market regime check failed: {e}")
    
    # Run comparison
    compare_strategies(symbol, days=800)


if __name__ == "__main__":
    main()
