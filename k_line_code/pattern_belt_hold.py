# -*- coding: utf-8 -*-
"""
捉腰带线 (Belt Hold Line) 识别与可视化脚本
形态特征：
1. 看涨：下跌趋势中，开盘价几乎就是最低价 (无下影线)，收长阳线。
2. 看跌：上升趋势中，开盘价几乎就是最高价 (无上影线)，收长阴线。
"""

import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
from k_line_code.common.data_fetcher import fetch_stock_data

# ==========================================
# 1. 核心识别逻辑
# ==========================================

def identify_belt_hold(
        df: pd.DataFrame,
        ma_len: int = 20,
        body_min_pct: float = 0.60,      # 实体至少占当日振幅的 60% (长实体)
        shadow_max_pct: float = 0.01,    # 上/下影线允许的最大比例 (通常要求几乎没有，设为 1%-3%)
        use_trend_filter: bool = True,   # 是否开启趋势过滤
        confirm_bull: bool = False,      # 是否需要次日确认 (看涨)
        confirm_bear: bool = False,      # 是否需要次日确认 (看跌)
        confirm_bars: int = 1
) -> pd.DataFrame:
    """
    识别捉腰带线形态
    """
    data = df.copy()

    # --- 1. 基础指标 ---
    data["Range"] = (data["High"] - data["Low"]).replace(0, 1e-6)
    data["Body"]  = (data["Close"] - data["Open"]).abs()
    data["MA"]    = data["Close"].rolling(ma_len).mean()

    # 计算上影线和下影线长度
    # Upper Shadow = High - Max(Open, Close)
    # Lower Shadow = Min(Open, Close) - Low
    upper_shadow = data["High"] - np.maximum(data["Open"], data["Close"])
    lower_shadow = np.minimum(data["Open"], data["Close"]) - data["Low"]

    # --- 2. 形态逻辑 ---

    # A. 看涨捉腰带 (Bullish Belt Hold)
    # 1. 收阳线
    # 2. 实体较长
    # 3. 开盘即最低 (下影线极短)
    bull_long = (data["Close"] > data["Open"]) & (data["Body"] >= body_min_pct * data["Range"])
    bull_no_lower = lower_shadow <= shadow_max_pct * data["Range"]
    bull_basic = bull_long & bull_no_lower

    # B. 看跌捉腰带 (Bearish Belt Hold)
    # 1. 收阴线
    # 2. 实体较长
    # 3. 开盘即最高 (上影线极短)
    bear_long = (data["Close"] < data["Open"]) & (data["Body"] >= body_min_pct * data["Range"])
    bear_no_upper = upper_shadow <= shadow_max_pct * data["Range"]
    bear_basic = bear_long & bear_no_upper

    # --- 3. 趋势过滤 ---
    if use_trend_filter:
        # 看涨需在下跌趋势中 (前一日收盘 < MA)
        trend_down = data["Close"].shift(1) < data["MA"].shift(1)
        # 看跌需在上升趋势中 (前一日收盘 > MA)
        trend_up = data["Close"].shift(1) > data["MA"].shift(1)
    else:
        trend_down = pd.Series(True, index=data.index)
        trend_up = pd.Series(True, index=data.index)

    # --- 4. 确认信号 (Confirmation) ---
    # 看涨确认：次日收盘价 > 当日收盘价 (继续上涨)
    if confirm_bull:
        bull_confirm = data["Close"].shift(-confirm_bars) > data["Close"]
        data["Bull_BeltHold"] = bull_basic & trend_down & bull_confirm
    else:
        data["Bull_BeltHold"] = bull_basic & trend_down

    # 看跌确认：次日收盘价 < 当日收盘价 (继续下跌)
    if confirm_bear:
        bear_confirm = data["Close"].shift(-confirm_bars) < data["Close"]
        data["Bear_BeltHold"] = bear_basic & trend_up & bear_confirm
    else:
        data["Bear_BeltHold"] = bear_basic & trend_up

    # --- 5. 支撑/阻力位 ---
    # 看涨捉腰带的开盘价(最低价)是非常强的支撑
    data["BH_Support"] = np.where(data["Bull_BeltHold"], data["Low"], np.nan)
    # 看跌捉腰带的开盘价(最高价)是非常强的阻力
    data["BH_Resistance"] = np.where(data["Bear_BeltHold"], data["High"], np.nan)

    return data


# ==========================================
# 2. 主执行程序
# ==========================================

def main():
    # --- 参数设置 ---
    symbol_code = '600519'          # 贵州茅台
    days_back = 365

    strategy_params = {
        'ma_len': 20,
        'body_min_pct': 0.60,       # 实体需占 60% 以上
        'shadow_max_pct': 0.03,     # 影线允许 3% 的误差 (稍微放宽一点适应A股)
        'use_trend_filter': True,   # 必须顺势
        'confirm_bull': False,      # 捉腰带线本身就很强，通常不需要确认
        'confirm_bear': False
    }

    print(f"正在获取 {symbol_code} 的历史数据...")

    # --- 1. 获取数据 ---
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days_back)

    try:
        df_raw = fetch_stock_data(symbol_code, days=days_back)
    except Exception as e:
        print(f"数据获取失败: {e}")
        return

    if df_raw.empty:
        print("未获取到数据。")
        return

    # 数据清洗
    # fetch_stock_data 已经返回小写列并以日期为索引，这里统一为大写列名以兼容后续逻辑
    df = df_raw.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})

    print("正在计算捉腰带线...")

    # --- 2. 计算形态 ---
    df = identify_belt_hold(df, **strategy_params)

    # 统计
    n_bull = df["Bull_BeltHold"].sum()
    n_bear = df["Bear_BeltHold"].sum()

    print("-" * 30)
    print(f"统计结果 (最近 {days_back} 天):")
    print(f"看涨捉腰带 (Bullish Belt Hold): {n_bull} 次")
    print(f"看跌捉腰带 (Bearish Belt Hold): {n_bear} 次")
    print("-" * 30)

    if n_bull + n_bear == 0:
        print("提示：未发现形态，建议放宽 body_min_pct (如 0.5) 或 shadow_max_pct (如 0.05)。")

    # --- 3. 可视化 ---
    print("正在绘图...")

    apds = [mpf.make_addplot(df["MA"], color='tab:blue', width=1.0)]

    rng = df["Range"].replace(0, 1e-6)

    # 标记位置
    bull_marks = np.where(df["Bull_BeltHold"], df["Low"] - rng * 0.18, np.nan)
    bear_marks = np.where(df["Bear_BeltHold"], df["High"] + rng * 0.18, np.nan)

    # 添加信号图层
    if not np.all(np.isnan(bull_marks)):
        apds.append(mpf.make_addplot(bull_marks, type="scatter", marker="^", markersize=100, color="seagreen", label="Bull Belt"))

    if not np.all(np.isnan(bear_marks)):
        apds.append(mpf.make_addplot(bear_marks, type="scatter", marker="v", markersize=100, color="crimson", label="Bear Belt"))

    # 添加支撑/阻力线 (阶梯线)
    if not df["BH_Resistance"].isnull().all():
        apds.append(mpf.make_addplot(df["BH_Resistance"].ffill(), color="crimson", linestyle='--', alpha=0.6))
    if not df["BH_Support"].isnull().all():
        apds.append(mpf.make_addplot(df["BH_Support"].ffill(), color="seagreen", linestyle='--', alpha=0.6))

    # 绘图
    title_str = f'{symbol_code} Belt Hold Line (Bullish/Bearish)'

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
    print("完成。")

if __name__ == "__main__":
    main()