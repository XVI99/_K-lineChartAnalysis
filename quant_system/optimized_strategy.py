"""
Optimized Trading Strategy - Three-Phase Implementation

Phase 1: Signal Quality Enhancement
- Pattern quality filter (>70)
- Multi-timeframe confirmation (weekly trend)
- Score >= 5 requirement

Phase 2: Profit/Loss Optimization
- Trailing take profit
- Partial take profit (50% at 2R)
- ATR-based stop loss (1.5*ATR)

Phase 3: Market Environment Filtering
- Market regime detection (BULL/NEUTRAL/BEAR)
- No new positions in BEAR market
- Position sizing based on market regime
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import BacktestEngine
from pattern_processor import StandardizedPatternProcessor
from pattern_scorer import (
    PatternQualityFilter,
    get_weighted_score,
    DEFAULT_BUY_WEIGHTS,
    DEFAULT_SELL_WEIGHTS,
    filter_signals_by_quality,
    score_pattern_quality
)
from signal_filter import SignalFilter, DEFAULT_FILTER
from multi_timeframe import MultiTimeframeConfirm
from market_regime import get_market_regime_filter, MarketRegime
from adaptive_signals import AdaptiveSignalGenerator, MarketStructure


# === 交易配置 ===
INITIAL_CAPITAL = 5000.0  # 初始资金5000元

# 可交易股票代码前缀 (沪A + 深A 主板)
VALID_STOCK_PREFIXES = (
    # 沪A
    'sh600', 'sh601', 'sh603', 'sh605',
    # 深A
    'sz000', 'sz001', 'sz002', 'sz003',
)


def validate_stock_code(symbol):
    """
    验证股票代码是否在可交易范围内。
    
    可交易: 沪A (600/601/603/605) + 深A (000/001/002/003)
    不可交易: 创业板(300)、科创板(688)、北交所(8/4)、ST等
    
    Args:
        symbol: 股票代码, 如 'sh600519', 'sz000001'
    
    Returns:
        tuple: (is_valid, reason)
    """
    code = symbol.lower().strip()
    
    for prefix in VALID_STOCK_PREFIXES:
        if code.startswith(prefix):
            return True, f"有效: {code} ({prefix}系列)"
    
    return False, (f"[X] 不可交易: {symbol}\n"
                   f"   仅允许: 沪A(600/601/603/605) + 深A(000/001/002/003)")


class OptimizedStrategy:
    """
    Optimized trading strategy integrating all three phases.
    """
    
    def __init__(self, config=None):
        """
        Initialize the optimized strategy.
        
        Args:
            config (dict): Strategy configuration. Uses defaults if None.
        """
        self.config = config or self.get_default_config()
        
        # Initialize components
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        k_line_dir = os.path.join(base_dir, "k_line_code")
        self.pattern_processor = StandardizedPatternProcessor(k_line_dir)
        self.quality_filter = PatternQualityFilter(
            min_quality=self.config['min_pattern_quality'],
            min_score=self.config['min_score'],
            pattern_weights=self.config.get('pattern_weights')
        )
        self.signal_filter = SignalFilter(self.config.get('signal_filter_config', DEFAULT_FILTER))
        self.mtf_confirm = MultiTimeframeConfirm(self.config.get('mtf_config'))
        
        # Phase 5: All-Weather Adaptive Engine
        self.adaptive_generator = AdaptiveSignalGenerator()
        self._regime_params = None
        
    @staticmethod
    def get_default_config():
        """Get default strategy configuration."""
        return {
            # Phase 1: Signal Quality
            'min_pattern_quality': 65,  # Minimum pattern quality score
            'min_score': 4,  # Minimum aggregate score (lowered from 5 for more trades)
            'pattern_weights': DEFAULT_BUY_WEIGHTS.copy(),
            'sell_weights': DEFAULT_SELL_WEIGHTS.copy(),
            'min_sell_score': 1.5,  # Minimum sell aggregate score to trigger exit
            'signal_filter_config': {
                'min_pattern_quality': 55,
                'require_trend_alignment': True,   # ENABLED: Close > MA20 > MA60
                'require_volume_confirm': True,     # ENABLED: Volume > avg
                'require_rsi_filter': True,          # ENABLED: RSI < 75
                'require_price_position': False,
                'require_macd_confirm': False,
                'vol_ratio_threshold': 0.8,          # Volume >= 80% of average
                'rsi_overbought': 75,
                'rsi_oversold': 25,
            },
            'mtf_config': {
                'require_weekly_trend': True,
                'require_weekly_ma_cross': False,
                'weekly_trend_ma': 20,
                'weekly_rsi_filter': False,
                'weekly_rsi_max': 70,
            },
            
            # Phase 2: Profit/Loss
            'atr_stop_loss_multiplier': 1.5,
            'atr_period': 14,
            'trailing_take_profit_pct': 0.05,
            'trailing_take_profit_trigger': 0.05,
            'partial_take_profit_pct': 0.5,
            'partial_take_profit_at': 2.0,
            'take_profit_pct': 0.15,
            
            # Phase 3: Market Environment
            'use_market_filter': True,
            'market_index': 'sh000300',
            'allow_neutral_market': True,
            
            # Phase 4: Risk Management
            'stop_loss_pct': 0.08,
            'trailing_stop_pct': 0.10,
            'time_stop_days': 20,
            'time_stop_min_profit': 0.02,
            'risk_per_trade': 0.02,
            'max_position_pct': 0.40,
            
            # Phase 5: All-Weather Adaptive Mode
            'use_adaptive_signals': True,  # Enable all-weather engine
        }
    
    def apply_strategy(self, df, verbose=True):
        """
        Apply the optimized strategy to generate signals.
        
        If use_adaptive_signals is True, combines pattern-based signals
        with adaptive RSI/BB/regime signals for all-weather profitability.
        
        Args:
            df (pd.DataFrame): OHLCV data.
            verbose (bool): Print detailed information.
            
        Returns:
            pd.DataFrame: DataFrame with Signal column added.
        """
        df = df.copy()
        
        # Step 1: Detect patterns
        if verbose:
            print("[Phase 1] Detecting patterns...")
        df = self.pattern_processor.run_all_patterns(df)
        
        # Step 2: Apply multi-timeframe confirmation
        if verbose:
            print("[Phase 1] Applying multi-timeframe confirmation...")
        df = self.mtf_confirm.add_weekly_trend_column(df)
        
        # Step 3: Generate pattern-based signals with quality filtering
        if verbose:
            print("[Phase 1] Generating signals with quality filtering...")
        df = self._generate_filtered_signals(df, verbose)
        
        # Step 4: If adaptive mode, merge with adaptive signals
        if self.config.get('use_adaptive_signals', False):
            if verbose:
                print("\n[Phase 5] 全天候自适应信号引擎启动...")
            df = self._apply_adaptive_signals(df, verbose)
        
        return df
    
    def _generate_filtered_signals(self, df, verbose=True):
        """
        Generate signals with all filters applied.
        
        Buy signals go through: PatternQualityFilter → SignalFilter → MTF Confirm
        Sell signals go through: sell pattern scoring with quality threshold
        
        Args:
            df (pd.DataFrame): DataFrame with patterns detected.
            verbose (bool): Print details.
            
        Returns:
            pd.DataFrame: DataFrame with Signal column.
        """
        df = df.copy()
        df['Signal'] = 0
        df['Signal_Reason'] = ''
        df['Pattern_Quality'] = 0.0
        df['Aggregate_Score'] = 0.0
        
        # Check market regime (Phase 3)
        market_ok = True
        self._position_scale = 1.0
        
        if self.config['use_market_filter']:
            try:
                regime, allow_long, scale = get_market_regime_filter(
                    self.config['market_index']
                )
                market_ok = allow_long or (self.config['allow_neutral_market'] and regime == MarketRegime.NEUTRAL)
                self._position_scale = scale
                
                if verbose:
                    print(f"[Phase 3] Market regime: {regime.value}, Allow: {market_ok}, Scale: {self._position_scale}")
            except Exception as e:
                if verbose:
                    print(f"[Phase 3] Market regime check failed: {e}")
                market_ok = True  # Default to allow if check fails
        
        if not market_ok:
            if verbose:
                print("[Phase 3] Market filter: No new positions allowed")
            return df
        
        # Pre-compute sell weights
        sell_weights = self.config.get('sell_weights', DEFAULT_SELL_WEIGHTS)
        min_sell_score = self.config.get('min_sell_score', 1.5)
        
        # Generate signals
        buy_count = 0
        sell_count = 0
        
        for i in range(len(df)):
            # === Check for BUY signals (full filtering pipeline) ===
            passed, score, details = self.quality_filter.check_buy_signal(df, i)
            quality = self.quality_filter.get_quality_score(df, i)
            
            df.iloc[i, df.columns.get_loc('Pattern_Quality')] = quality
            df.iloc[i, df.columns.get_loc('Aggregate_Score')] = score
            
            if passed:
                # Apply signal filter (trend, volume, RSI)
                filter_passed = self.signal_filter.filter_buy_signal(df, quality, i)
                
                # Check weekly trend (Phase 1)
                weekly_ok = True
                if 'Weekly_Trend' in df.columns:
                    weekly_trend = df['Weekly_Trend'].iloc[i]
                    weekly_ok = weekly_trend in ['BULL', 'NEUTRAL', None]
                
                if filter_passed and weekly_ok:
                    df.iloc[i, df.columns.get_loc('Signal')] = 1
                    df.iloc[i, df.columns.get_loc('Signal_Reason')] = f"Quality={quality:.0f}, Score={score:.1f}"
                    buy_count += 1
            
            # === Check for SELL signals (quality-filtered using all sell patterns) ===
            sell_score = 0.0
            sell_reasons = []
            for pattern_col, weight in sell_weights.items():
                if pattern_col not in df.columns:
                    continue
                if df[pattern_col].iloc[i] == 1:
                    p_quality = score_pattern_quality(df, pattern_col, i)
                    weighted = weight * (p_quality / 100.0)
                    sell_score += weighted
                    sell_reasons.append(f"{pattern_col}(Q={p_quality:.0f})")
            
            # Only sell if aggregate sell score exceeds threshold
            if sell_score >= min_sell_score and len(sell_reasons) > 0:
                df.iloc[i, df.columns.get_loc('Signal')] = -1
                df.iloc[i, df.columns.get_loc('Signal_Reason')] = f"Sell({sell_score:.1f}): {', '.join(sell_reasons)}"
                sell_count += 1
        
        if verbose:
            print(f"[Result] Generated {buy_count} buy signals, {sell_count} sell signals")
        
        return df
    
    def create_backtest_engine(self):
        """
        Create a BacktestEngine with Phase 2 + Phase 4 optimizations.
        Passes position_scale from market regime detection.
        
        Returns:
            BacktestEngine: Configured backtest engine.
        """
        return BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            commission=0.0003,
            slippage=0.002,
            stamp_tax=0.001,
            
            # Phase 2: Profit/Loss Optimization
            stop_loss_pct=self.config['stop_loss_pct'],
            trailing_stop_pct=self.config['trailing_stop_pct'],
            atr_stop_loss_multiplier=self.config['atr_stop_loss_multiplier'],
            atr_period=self.config['atr_period'],
            take_profit_pct=self.config['take_profit_pct'],
            trailing_take_profit_pct=self.config['trailing_take_profit_pct'],
            trailing_take_profit_trigger=self.config['trailing_take_profit_trigger'],
            partial_take_profit_pct=self.config['partial_take_profit_pct'],
            partial_take_profit_at=self.config['partial_take_profit_at'],
            
            # Time stop
            time_stop_days=self.config['time_stop_days'],
            time_stop_min_profit=self.config['time_stop_min_profit'],
            
            # Phase 4: Position Sizing (uses market regime scale)
            risk_per_trade=self.config.get('risk_per_trade', 0.02),
            position_scale=getattr(self, '_position_scale', 1.0),
            max_position_pct=self.config.get('max_position_pct', 0.40),
            
            execution_mode="next_open"
        )
    
    def _apply_adaptive_signals(self, df, verbose=True):
        """
        Apply adaptive signal generation — merges pattern signals with
        regime-based RSI/BB/Volume signals for all-weather alpha.
        
        Pattern-based signals (Signal=1/-1) are preserved.
        Adaptive signals fill in gaps where patterns are silent but 
        technical conditions indicate high-probability setups.
        """
        # Prepare pattern signal series for adaptive engine
        pattern_buys = (df['Signal'] == 1).astype(int)
        pattern_sells = (df['Signal'] == -1).astype(int)
        
        # Generate adaptive signals
        adaptive_df, regime_params = self.adaptive_generator.generate_signals(
            df, 
            pattern_buy_signals=pattern_buys,
            pattern_sell_signals=pattern_sells,
            verbose=verbose
        )
        
        # Store regime params for BacktestEngine creation
        self._regime_params = regime_params
        self._position_scale = regime_params['position_scale']
        
        # Merge signals: pattern signals take priority, adaptive fills gaps
        merged_signal = df['Signal'].copy()
        merged_reason = df['Signal_Reason'].copy()
        
        adaptive_buys = 0
        adaptive_sells = 0
        
        for i in range(len(df)):
            if merged_signal.iloc[i] == 0 and adaptive_df['Adaptive_Signal'].iloc[i] != 0:
                # No pattern signal — use adaptive signal
                merged_signal.iloc[i] = adaptive_df['Adaptive_Signal'].iloc[i]
                merged_reason.iloc[i] = f"[自适应] {adaptive_df['Adaptive_Reason'].iloc[i]}"
                if adaptive_df['Adaptive_Signal'].iloc[i] == 1:
                    adaptive_buys += 1
                else:
                    adaptive_sells += 1
        
        df['Signal'] = merged_signal
        df['Signal_Reason'] = merged_reason
        df['Regime'] = adaptive_df['Regime']
        
        if verbose:
            print(f"[Phase 5] 自适应信号补充: +{adaptive_buys}买 +{adaptive_sells}卖")
            # Count total
            total_buy = (df['Signal'] == 1).sum()
            total_sell = (df['Signal'] == -1).sum()
            print(f"[Phase 5] 合计信号: {total_buy}买 + {total_sell}卖")
        
        return df
    
    def create_backtest_engine_adaptive(self):
        """
        Create BacktestEngine with regime-adaptive parameters.
        Uses parameters from the detected market regime instead of static values.
        """
        rp = self._regime_params or {}
        
        return BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            commission=0.0003,
            slippage=0.002,
            stamp_tax=0.001,
            
            # Regime-adaptive parameters
            stop_loss_pct=rp.get('stop_loss_atr_mult', 1.5) * 0.03,  # Convert to pct approximation
            trailing_stop_pct=rp.get('trailing_stop_pct', 0.06),
            atr_stop_loss_multiplier=rp.get('stop_loss_atr_mult', 1.5),
            atr_period=self.config['atr_period'],
            take_profit_pct=rp.get('take_profit_pct', 0.10),
            trailing_take_profit_pct=rp.get('trailing_stop_pct', 0.05),
            trailing_take_profit_trigger=0.03,
            partial_take_profit_pct=0.5,
            partial_take_profit_at=2.0,
            
            time_stop_days=rp.get('time_stop_days', 15),
            time_stop_min_profit=self.config['time_stop_min_profit'],
            
            risk_per_trade=rp.get('risk_per_trade', 0.02),
            position_scale=rp.get('position_scale', 0.5),
            max_position_pct=rp.get('max_position_pct', 0.35),
            
            execution_mode="next_open"
        )
    
    def run_backtest(self, symbol, days=800, verbose=True):
        """
        Run a complete backtest with all optimizations.
        
        Args:
            symbol (str): Stock symbol.
            days (int): Number of days to backtest.
            verbose (bool): Print details.
            
        Returns:
            dict: Backtest results.
        """
        from custom_data import get_price
        
        # Validate stock code
        valid, reason = validate_stock_code(symbol)
        if not valid:
            print(reason)
            return None
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running Optimized Strategy Backtest: {symbol}")
            print(f"初始资金: {INITIAL_CAPITAL:.0f}元")
            print(f"{'='*60}")
        
        # Fetch data
        try:
            df = get_price(symbol, count=days, frequency='1d')
            if df is None or df.empty:
                print(f"Failed to fetch data for {symbol}")
                return None
            
            # Standardize columns
            df = df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume'
            })
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
        
        # Apply strategy
        df = self.apply_strategy(df, verbose)
        
        # Create backtest engine (use adaptive engine if adaptive mode)
        if self.config.get('use_adaptive_signals', False) and self._regime_params:
            engine = self.create_backtest_engine_adaptive()
        else:
            engine = self.create_backtest_engine()
        
        # Run backtest
        equity_curve = engine.run(df, signal_col="Signal", price_col="Close")
        
        if equity_curve is None:
            return None
        
        # Calculate performance
        performance = engine.calculate_performance()
        
        if verbose:
            print(f"\n{'='*60}")
            print("Backtest Results:")
            print(f"{'='*60}")
            for key, value in performance.items():
                print(f"  {key}: {value}")
            
            # Print trade log summary
            if engine.trade_log:
                print(f"\nTrade Log ({len(engine.trade_log)} trades):")
                for trade in engine.trade_log[-10:]:  # Last 10 trades
                    print(f"  {trade}")
        
        return {
            'symbol': symbol,
            'performance': performance,
            'equity_curve': equity_curve,
            'trade_log': engine.trade_log,
            'engine': engine
        }


def compare_strategies(symbol, days=800):
    """
    Compare original vs optimized strategy.
    
    Args:
        symbol (str): Stock symbol.
        days (int): Number of days.
        
    Returns:
        dict: Comparison results.
    """
    from custom_data import get_price
    from backtest_strategy import apply_strategy_logic
    
    print(f"\n{'#'*70}")
    print(f"# Strategy Comparison: {symbol}")
    print(f"{'#'*70}")
    
    # Fetch data
    try:
        df = get_price(symbol, count=days, frequency='1d')
        if df is None or df.empty:
            print(f"Failed to fetch data for {symbol}")
            return None
        
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    
    # Original strategy
    print("\n[1] Running Original Strategy...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    k_line_dir = os.path.join(base_dir, "k_line_code")
    processor = StandardizedPatternProcessor(k_line_dir)
    df_orig = apply_strategy_logic(df.copy(), processor)
    
    engine_orig = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        stop_loss_pct=0.08,
        trailing_stop_pct=0.10,
        execution_mode="next_open"
    )
    engine_orig.run(df_orig, signal_col="Signal")
    perf_orig = engine_orig.calculate_performance()
    
    # Optimized strategy
    print("\n[2] Running Optimized Strategy...")
    strategy = OptimizedStrategy()
    result_opt = strategy.run_backtest(symbol, days, verbose=False)
    
    if result_opt is None:
        return None
    
    perf_opt = result_opt['performance']
    
    # Print comparison
    print(f"\n{'='*70}")
    print("Strategy Comparison Results:")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Original':<20} {'Optimized':<20}")
    print("-" * 65)
    
    for key in perf_orig.keys():
        orig_val = perf_orig.get(key, 'N/A')
        opt_val = perf_opt.get(key, 'N/A')
        print(f"{key:<25} {str(orig_val):<20} {str(opt_val):<20}")
    
    return {
        'symbol': symbol,
        'original': perf_orig,
        'optimized': perf_opt,
        'original_trades': len(engine_orig.trade_log),
        'optimized_trades': len(result_opt['trade_log'])
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Trading Strategy')
    parser.add_argument('--symbol', type=str, default='sh600519', help='Stock symbol')
    parser.add_argument('--days', type=int, default=800, help='Number of days to backtest')
    parser.add_argument('--compare', action='store_true', help='Compare with original strategy')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_strategies(args.symbol, args.days)
    else:
        strategy = OptimizedStrategy()
        strategy.run_backtest(args.symbol, args.days)


if __name__ == "__main__":
    main()
