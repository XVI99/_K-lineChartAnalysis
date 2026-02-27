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
from custom_data import fetch_daily_data
from risk_utils import (
    calculate_atr, check_liquidity, check_trend_filter,
    check_signal_persistence, generate_trade_plan_v2
)
from market_regime import get_market_regime_filter, MarketRegime
from pattern_scorer import get_weighted_score, DEFAULT_BUY_WEIGHTS, DEFAULT_SELL_WEIGHTS

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Clear Proxies
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# Constraints
MAX_PRICE = 45.0  # Leave buffer for fees, 45 * 100 = 4500 < 5000
BUDGET = 5000.0

def get_stocks_from_csv():
    """
    Read stock list from local CSV file: ../stock_name_and_code/stock_list.csv
    Format: code,name
    """
    print("Reading stock list from local CSV...")
    try:
        # Determine strict path relative to this script
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "stock_name_and_code", "stock_list.csv")
        
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}")
            return []

        # Read CSV with str dtype for code to preserve leading zeros
        df = pd.read_csv(csv_path, dtype={"code": str, "name": str})
        
        # Filter 1: Main Board Prefixes
        main_board_prefixes = ("600", "601", "603", "605", "000", "001", "002", "003")
        df = df[df["code"].astype(str).str.startswith(main_board_prefixes)]
        
        # Exclude ST
        df = df[ ~df["name"].str.contains("ST") ]
        
        print(f"Loaded {len(df)} Main Board stocks from CSV.")
        
        # Convert to list of (symbol, name) tuples
        stock_list = list(zip(df["code"], df["name"]))
        return stock_list
        
    except Exception as e:
        print(f"Failed to read local CSV: {e}")
        return []

def get_recent_data(symbol):
    """
    Fetch recent stock data using generic custom provider (Tencent/Sina).
    """
    try:
        # Use new custom provider
        # Request slightly more data to ensure enough history for indicators (e.g. 365 days)
        df = fetch_daily_data(symbol, days=400)
        
        if df.empty:
            return pd.DataFrame()
            
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def analyze_stock(processor, symbol, name, df):
    if df.empty or len(df) < 60:
        return None

    current_price = df["Close"].iloc[-1]
    
    # 1. Price Filter (Budget Check)
    if current_price > MAX_PRICE:
        return None  # Too expensive daily check

    # === V2 ENHANCEMENTS ===
    # 2. Trend Filter (MA200) - Only buy in uptrend
    is_uptrend, ma200, _ = check_trend_filter(df, ma_period=200)
    if not is_uptrend and len(df) >= 200:
        return None  # Skip stocks in downtrend
    
    # 3. Liquidity Filter - Avoid illiquid stocks
    is_liquid, avg_turnover = check_liquidity(df, min_turnover=30_000_000, lookback=20)
    if not is_liquid:
        return None  # Skip illiquid stocks
    
    # 4. Calculate ATR for position sizing and stops
    df['ATR'] = calculate_atr(df, period=14)
    current_atr = df['ATR'].iloc[-1]
    if pd.isna(current_atr) or current_atr <= 0:
        current_atr = current_price * 0.03  # Fallback: 3% of price

    # 2. Run Patterns
    # Reversal Patterns (Strong)
    try:
        df = processor.run_pattern("engulfing", df)
        df = processor.run_pattern("morning_star", df)
        df = processor.run_pattern("umbrella_lines", df)
        df = processor.run_pattern("shooting_inverted", df)
        
        # Trend / Continuation
        df = processor.run_pattern("three_soldiers", df)
        df = processor.run_pattern("belt_hold", df)
        
        # Indicators
        df = processor.run_pattern("macd", df)
        df = processor.run_pattern("kdj", df)          # Calc K,D,J
        df = processor.run_pattern("kdj_signals", df)  # Calc Signals
        df = processor.run_pattern("rsi_patterns", df) # Calc RSI & Signals
        df = processor.run_pattern("volume_patterns", df) # Calc Vol Ratio
    except Exception as e:
        print(f"Error in patterns for {symbol}: {e}")
        return None

    # 3. Check Latest Signal (Last Row)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    buy_signal = False
    sell_signal = False
    reasons = []

    # Buy Logic — Confidence-weighted scoring (replaces hard-coded linear scores)
    score, buy_details = get_weighted_score(df, DEFAULT_BUY_WEIGHTS, idx=-1)
    for pat_name, quality, weighted in buy_details:
        buy_signal = True
        reasons.append(f"{pat_name}(Q:{quality:.0f})")

    # Volume booster (unchanged)
    if latest.get("Is_High_Vol", 0) == 1 and latest["Close"] > latest["Open"]:
        score += 0.5
        reasons.append("放量确认")

    # Sell Logic — Confidence-weighted scoring
    sell_score, sell_details = get_weighted_score(df, DEFAULT_SELL_WEIGHTS, idx=-1)
    for pat_name, quality, weighted in sell_details:
        sell_signal = True
        reasons.append(f"{pat_name}(Q:{quality:.0f})")

    # 4. Construct Result
    if buy_signal or sell_signal:
        action = "BUY" if buy_signal else "SELL"
        
        # === V2: ATR-Based Trade Plan ===
        plan = {}
        if action == "BUY":
            # Use ATR-based position sizing and stops
            plan = generate_trade_plan_v2(
                entry_price=current_price,
                atr=current_atr,
                budget=BUDGET,
                risk_pct=0.02  # Risk 2% per trade
            )
        
        return {
            "Action": action,
            "Symbol": symbol,
            "Name": name,
            "Date": latest.name.strftime("%Y-%m-%d"),
            "Price": current_price,
            "Reason": "+".join(reasons),
            "Can_Buy_Shares": plan.get("Shares", 0) if action == "BUY" else "N/A",
            "Score": score,
            "Plan": plan,
            "ATR": round(current_atr, 2),
            "Trend": "UP" if (is_uptrend if 'is_uptrend' in dir() else True) else "DOWN"
        }
    
    return None

def main():
    print(f"Starting Budget Monitor (Budget: {BUDGET} CNY, Max Price: {MAX_PRICE} CNY)...")

    # === Phase 3: Market Regime Filter ===
    try:
        regime, allow_long, position_scale = get_market_regime_filter()
        print(f"Market Regime: {regime.value} | Allow Long: {allow_long} | Scale: {position_scale}")
        if not allow_long:
            print("⛔ BEAR market detected. Skipping all buy scans.")
            return
    except Exception as e:
        print(f"[Warning] Market regime check failed ({e}), proceeding with full scan.")
        position_scale = 1.0

    effective_budget = BUDGET * position_scale
    print(f"Effective Budget (after regime scaling): {effective_budget:.0f} CNY")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    k_line_dir = os.path.join(base_dir, "k_line_code")
    processor = StandardizedPatternProcessor(k_line_dir)
    
    # Scan Stocks from Local CSV
    stocks = get_stocks_from_csv()
    alerts = []
    
    print(f"Scanning {len(stocks)} stocks from local CSV...")
    
    for symbol, name in tqdm(stocks):
        df = get_recent_data(symbol)
        result = analyze_stock(processor, symbol, name, df)
        if result:
            alerts.append(result)
            if result["Action"] == "BUY" and result["Score"] >= 2:
                # Live print only high quality signals
                print(f" >>> FOUND STRONG BUY: {name} ({symbol}) Score:{result['Score']}")
            
    print("\n" + "="*60)
    print("DAILY MONITOR REPORT")
    print("="*60)
    
    if alerts:
        # Separate BUYs and Sort by Score
        buys = [a for a in alerts if a["Action"] == "BUY"]
        buys.sort(key=lambda x: x["Score"], reverse=True)
        
        print("\n🏆 TOP 3 RECOMMENDATIONS (V2: ATR Risk Management) 🏆")
        print("-" * 50)
        for i, alert in enumerate(buys[:3]):
            print(f"#{i+1} {alert['Name']} ({alert['Symbol']}) | Score: {alert['Score']} | Trend: {alert.get('Trend', 'N/A')}")
            print(f"   Signals: {alert['Reason']}")
            p = alert['Plan']
            shares = p.get('Shares', 0)
            entry = p.get('Entry', 0)
            print(f"   [Trade Plan V2 - ATR: {p.get('ATR', 'N/A')}]")
            print(f"   ➤ Buy Zone : {entry:.2f}")
            print(f"   ➤ Stop Loss: {p.get('StopLoss', 'N/A')} (ATR-Based)")
            print(f"   ➤ Target   : {p.get('TakeProfit', 'N/A')} (R:R = 1:2)")
            print(f"   ➤ Position : {shares} shares (Cost: {int(entry * shares) if shares else 0} CNY)")
            print(f"   ➤ Risk Amt : {p.get('RiskAmount', 'N/A')} CNY (2% of Budget)")
            print("-" * 50)
            
        print("\n--- Other Signals ---")
        for alert in buys[3:]:
            print(f" {alert['Name']}({alert['Symbol']}) | Score: {alert['Score']}")
            
        sells = [a for a in alerts if a["Action"] == "SELL"]
        if sells:
            print("\n⚠️ SELL SIGNALS:")
            for s in sells:
                print(f" {s['Name']} - {s['Reason']}")
                
    else:
        print("No signals found today.")
    print("="*60)

if __name__ == "__main__":
    main()
