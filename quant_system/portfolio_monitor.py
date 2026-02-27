
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Clear Proxies
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

from quant_system.pattern_loader import PatternLoader
from quant_system.pattern_processor import StandardizedPatternProcessor

try:
    from quant_system.custom_data import fetch_daily_data
    from quant_system.risk_utils import calculate_atr, calculate_volatility_stop
except ImportError:
    from custom_data import fetch_daily_data
    from risk_utils import calculate_atr, calculate_volatility_stop

def load_holdings(csv_path="holdings.csv"):
    if not os.path.exists(csv_path):
        print("No holdings.csv found. Please create one with columns: Symbol,Name,CostPrice,Shares")
        return []
    try:
        df = pd.read_csv(csv_path, dtype={"Symbol": str})
        return df.to_dict('records')
    except Exception as e:
        print(f"Error reading holdings: {e}")
        return []

def analyze_position(processor, position, data):
    symbol = position['Symbol']
    name = position['Name']
    cost = float(position['CostPrice'])
    
    # Run Patterns
    df = data.copy()
    try:
        df = processor.run_pattern("engulfing", df)
        df = processor.run_pattern("morning_star", df) # Reversal check
        df = processor.run_pattern("kdj", df)
        df = processor.run_pattern("kdj_signals", df)
        df = processor.run_pattern("rsi_patterns", df)
        df = processor.run_pattern("macd", df)
        
        # Bearish Patterns (Sell Signals)
        df = processor.run_pattern("shooting_inverted", df)
        df = processor.run_pattern("umbrella_lines", df) # Hanging man
    except Exception as e:
        print(f"Pattern error {symbol}: {e}")
        return None

    current_price = df['Close'].iloc[-1]
    latest = df.iloc[-1]
    
    # === V2: Calculate ATR for dynamic stop ===
    df['ATR'] = calculate_atr(df, period=14)
    current_atr = df['ATR'].iloc[-1]
    if pd.isna(current_atr) or current_atr <= 0:
        current_atr = current_price * 0.03  # Fallback
    
    # Calculate dynamic stop loss (2x ATR below cost)
    dynamic_stop = calculate_volatility_stop(cost, current_atr, multiplier=2.0)
    
    # Financial Stats
    pnl = (current_price - cost) / cost
    pnl_str = f"{pnl:.2%}"
    
    action = "HOLD"
    reasons = []
    
    # === V2: ATR-Based Stop Loss Check ===
    # Trigger if price drops below cost - 2*ATR
    if current_price < dynamic_stop:
        action = "SELL (StopLoss)"
        reasons.append(f"Price < ATR Stop ({dynamic_stop:.2f})")
        
    # 2. Take Profit Check (Partial Sell?)
    # If Gain > 15%, check for reversal signs. If no reversal, HOLD trend.
    if pnl > 0.15:
        reasons.append(f"Profit > 15% ({pnl_str})")
        # Check if trend is breaking?
        if latest.get("SellSignal", 0) == 1: # KDJ Dead Cross
            action = "SELL (ProfitTake)"
            reasons.append("KDJ Dead Cross at High")
    
    # 3. Technical Sell Signals (regardless of PnL)
    sell_score = 0
    if latest.get("Bear_Engulf", 0) == 1:
        sell_score += 2
        reasons.append("Bearish Engulfing")
    if latest.get("ShootingStar", 0) == 1:
        sell_score += 1
        reasons.append("Shooting Star")
    if latest.get("Hanging_Man", 0) == 1:
        sell_score += 1
        reasons.append("Hanging Man")
    if latest.get("SellSignal", 0) == 1: # KDJ Dead
        sell_score += 1
        reasons.append("KDJ Dead Cross")
        
    if sell_score >= 2:
        action = "SELL (Technical)"
    elif sell_score == 1 and action != "SELL (StopLoss)":
        action = "WATCH (Weak Sell)"

    # 4. Buy More Signals
    buy_score = 0
    if latest.get("BuySignal", 0) == 1: # KDJ Gold Low
        buy_score += 1
        reasons.append("KDJ Gold Cross")
    if latest.get("Bull_Engulf", 0) == 1:
        buy_score += 2
        reasons.append("Bullish Engulfing")
        
    if buy_score >= 2 and "SELL" not in action:
        action = "ADD (Buy Signal)"

    return {
        "Symbol": symbol,
        "Name": name,
        "Cost": cost,
        "Current": current_price,
        "PnL": pnl_str,
        "Action": action,
        "Reasons": ", ".join(reasons)
    }

def main():
    print("="*60)
    print("PORTFOLIO MONITOR (Holdings Check)")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    k_line_dir = os.path.join(base_dir, "k_line_code")
    processor = StandardizedPatternProcessor(k_line_dir)
    
    csv_path = os.path.join(os.path.dirname(__file__), "holdings.csv")
    holdings = load_holdings(csv_path)
    
    if not holdings:
        return

    print(f"Checking {len(holdings)} positions...\n")
    
    print(f"{'Name':<10} {'Symbol':<8} {'Price':<8} {'PnL':<8} {'Action':<15} {'Reasons'}")
    print("-" * 80)
    
    for pos in holdings:
        df = fetch_daily_data(pos['Symbol'], days=300)
        if df.empty:
            print(f"Skipping {pos['Name']}: No Data")
            continue
            
        res = analyze_position(processor, pos, df)
        if res:
            # Colorize Action if possible (not in basic cmd, just symbols)
            act = res['Action']
            symbol_mark = "🔴" if "SELL" in act else "🟢" if "ADD" in act else "⚪"
            
            print(f"{res['Name']:<10} {res['Symbol']:<8} {res['Current']:<8.2f} {res['PnL']:<8} {symbol_mark} {act:<15} {res['Reasons']}")

    print("-" * 80)
    print("Legend: 🔴 SELL | 🟢 ADD | ⚪ HOLD/WATCH")
    print("="*60)

if __name__ == "__main__":
    main()
