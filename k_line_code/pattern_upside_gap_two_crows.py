# -*- coding: utf-8 -*-
"""
Upside Gap Two Crows (向上跳空两只乌鸦) Recognition Script
Pattern Characteristics (Bearish Reversal):
1. Trend: Uptrend.
2. Day 1: Long bullish candle (White).
3. Day 2: Gaps up, forms a small bearish candle (Black).
4. Day 3: Opens higher than Day 2's open, but closes lower than Day 2's close.
   However, Day 3's close is still above Day 1's close (filling the gap but not closing it).
"""

import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
from k_line_code.common.data_fetcher import fetch_stock_data

# ==========================================
# 1. Core Identification Logic
# ==========================================

def identify_upside_gap_two_crows(
        df: pd.DataFrame,
        ma_len: int = 20,
        use_trend_filter: bool = True,
        long_body_pct1: float = 0.50,    # Day 1 body > 50% of range
        small_body_pct23: float = 0.35,  # Day 2 & 3 bodies < 35% of range
        gap_tolerance_pct: float = 0.00, # Strict gap by default
        gap_vs_high_not_close: bool = False, # True: Gap calculated vs High; False: Gap vs Close
        require_o3_gt_o2_open: bool = True,   # Day 3 Open > Day 2 Open
        require_c3_lt_c2_close: bool = True,  # Day 3 Close < Day 2 Close
        require_closes_above_bar1: bool = True, # Day 2 & 3 Closes > Day 1 Close (Gap remains)
        use_day4_confirmation: bool = False,
        confirm_shift: int = 1
) -> pd.DataFrame:
    """
    Identify Upside Gap Two Crows pattern
    """
    data = df.copy()

    # --- 1. Basic Indicators ---
    data["Range"] = (data["High"] - data["Low"]).replace(0, 1e-6)
    data["Body"]  = (data["Close"] - data["Open"]).abs()
    data["MA"]    = data["Close"].rolling(ma_len).mean()

    # Helper for shifting data
    # 1=Day1 (Long White), 2=Day2 (Crow 1), 3=Day3 (Crow 2)
    # Note: Logic aligns with: Day 1 (t-2), Day 2 (t-1), Day 3 (t/current)
    def sh(col, k): return data[col].shift(k)

    # Day 1 Data (t-2)
    o1, h1, l1, c1 = sh("Open", 2), sh("High", 2), sh("Low", 2), sh("Close", 2)
    b1, r1 = sh("Body", 2), sh("Range", 2)

    # Day 2 Data (t-1)
    o2, h2, l2, c2 = sh("Open", 1), sh("High", 1), sh("Low", 1), sh("Close", 1)
    b2, r2 = sh("Body", 1), sh("Range", 1)

    # Day 3 Data (t/Current)
    o3, h3, l3, c3 = data["Open"], data["High"], data["Low"], data["Close"]
    b3, r3 = data["Body"], data["Range"]

    # --- 2. Pattern Logic ---

    # A. Trend Filter
    # Check trend before Day 1 (using Day 1's previous day, i.e., shift(3))
    if use_trend_filter:
        uptrend_before = data["Close"].shift(3) > data["MA"].shift(3)
    else:
        uptrend_before = pd.Series(True, index=data.index)

    # B. Candle Shapes
    # Day 1: Long White
    long_white1 = (c1 > o1) & (b1 >= long_body_pct1 * r1)
    # Day 2: Small Black (Bearish)
    small_black2 = (c2 < o2) & (b2 <= small_body_pct23 * r2)
    # Day 3: Small Black (Bearish)
    small_black3 = (c3 < o3) & (b3 <= small_body_pct23 * r3)

    # C. Gaps (The "Upside Gap")
    # Gap Up for Day 2 relative to Day 1
    if gap_vs_high_not_close:
        # Gap above Day 1 High
        gap_up_bar2 = l2 > h1 * (1 + gap_tolerance_pct)
        # Day 3 usually opens inside the gap or higher, let's strictly check if it's above Day 1 High
        gap_up_bar3_open = o3 > h1 * (1 + gap_tolerance_pct)
    else:
        # Gap above Day 1 Close (Standard definition often uses Close)
        gap_up_bar2 = l2 > c1 * (1 + gap_tolerance_pct)
        gap_up_bar3_open = o3 > c1 * (1 + gap_tolerance_pct)

    # D. The "Two Crows" Relationship
    # Day 3 "engulfs" Day 2's body roughly?
    # Standard Def: Day 3 opens HIGHER than Day 2 open, but closes LOWER than Day 2 close.
    cond_o3_gt_o2 = (o3 > o2 * (1 + gap_tolerance_pct)) if require_o3_gt_o2_open else pd.Series(True, index=data.index)
    cond_c3_lt_c2 = (c3 < c2 * (1 - gap_tolerance_pct)) if require_c3_lt_c2_close else pd.Series(True, index=data.index)

    # E. Gap Remains?
    # Day 3 closes lower than Day 2, but must stay ABOVE Day 1 Close (The gap is not filled)
    # If Day 3 closes below Day 1 Close, it becomes a "Two Crows" or "Dark Cloud Cover" variant, not "Upside Gap Two Crows".
    if require_closes_above_bar1:
        closes_above_bar1 = (c2 > c1) & (c3 > c1)
    else:
        closes_above_bar1 = pd.Series(True, index=data.index)

    # F. Confirmation (Optional Day 4)
    if use_day4_confirmation:
        # Day 4 Close < Day 3 Close
        day4_confirm = data["Close"].shift(-confirm_shift) < c3
    else:
        day4_confirm = pd.Series(True, index=data.index)

    # Combine All
    ugt2c = (
            uptrend_before &
            long_white1 &
            small_black2 & small_black3 &
            gap_up_bar2 & gap_up_bar3_open &
            cond_o3_gt_o2 & cond_c3_lt_c2 &
            closes_above_bar1 &
            day4_confirm
    )

    data["UpsideGapTwoCrows"] = ugt2c

    # --- 3. Resistance Calculation ---
    # Resistance is the highest high among the 3 days
    # Use max.reduce with fillna(0) to handle NaNs safely
    h1_safe, h2_safe, h3_safe = h1.fillna(0), h2.fillna(0), h3.fillna(0)
    res_level = np.maximum.reduce([h1_safe, h2_safe, h3_safe])

    data["UG2C_Res"] = np.where(data["UpsideGapTwoCrows"], res_level, np.nan)
    data["UG2C_Res_plot"] = data["UG2C_Res"].ffill()

    return data


# ==========================================
# 2. Main Execution
# ==========================================

def main():
    # --- Parameters ---
    symbol_code = '600519'          # 贵州茅台
    days_back = 365

    strategy_params = {
        'ma_len': 20,
        'use_trend_filter': True,
        'long_body_pct1': 0.50,     # Day 1 is strong
        'small_body_pct23': 0.40,   # Relaxed slightly to 0.40 to capture real-world examples
        'gap_tolerance_pct': 0.00,  # Strict gap
        'require_o3_gt_o2_open': True,
        'require_c3_lt_c2_close': True,
        'require_closes_above_bar1': True # Essential for "Upside Gap" definition
    }

    print(f"Fetching data for {symbol_code}...")

    # --- 1. Fetch Data ---
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days_back)

    try:
        df_raw = fetch_stock_data(symbol_code, days=days_back)
    except Exception as e:
        print(f"Data fetch failed: {e}")
        return

    if df_raw.empty:
        print("No data fetched.")
        return

    # Data Cleaning
    df = (
        df_raw
        .rename(columns={'日期': 'Date', '开盘': 'Open', '最高': 'High',
                         '最低': 'Low', '收盘': 'Close', '成交量': 'Volume'})
        .loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        .assign(Date=lambda x: pd.to_datetime(x['Date']))
        .set_index('Date')
        .sort_index()
    )

    print("Calculating Upside Gap Two Crows pattern...")

    # --- 2. Identify Pattern ---
    df = identify_upside_gap_two_crows(df, **strategy_params)

    # Statistics
    n_sig = df["UpsideGapTwoCrows"].sum()
    print("-" * 30)
    print(f"Pattern Count (Last {days_back} days): {n_sig}")
    print("-" * 30)

    if n_sig == 0:
        print("Tip: This pattern is rare. Try relaxing 'gap_tolerance_pct' or 'small_body_pct23'.")

    # --- 3. Visualization ---
    print("Plotting...")

    apds = [mpf.make_addplot(df["MA"], color='tab:blue', width=1.0)]

    rng = df["Range"].replace(0, 1e-6)

    # Marker: Crimson inverted triangle above High
    marks = np.where(df["UpsideGapTwoCrows"], df["High"] + rng * 0.22, np.nan)

    # Resistance Line
    res_line = df["UG2C_Res_plot"]

    if not np.all(np.isnan(marks)):
        apds.append(mpf.make_addplot(marks, type="scatter", marker="v", markersize=120, color="crimson", label="2 Crows"))

    if not np.all(np.isnan(res_line)):
        apds.append(mpf.make_addplot(res_line, color="crimson", linestyle='--', alpha=0.6))

    title_str = f'{symbol_code} Upside Gap Two Crows Pattern'

    mpf.plot(
        df,
        type="candle",
        volume=True,
        addplot=apds,
        title=title_str,
        style="yahoo",
        figsize=(14, 8),
        tight_layout=True,
        block=True
    )
    print("Done.")

if __name__ == "__main__":
    main()