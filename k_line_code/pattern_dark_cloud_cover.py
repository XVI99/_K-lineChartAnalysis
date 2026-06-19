# -*- coding: utf-8 -*-
"""
乌云盖顶 (Dark Cloud Cover) 形态识别与可视化脚本
逻辑说明：
1. 第一根 K 线为阳线 (Bullish)。
2. 第二根 K 线为阴线 (Bearish)。
3. 第二根 K 线高开（通常要求开盘价高于前一日最高价或收盘价）。
4. 第二根 K 线收盘价深入第一根实体内部（通常要求低于前一日实体的中点）。
"""

import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
from k_line_code.common.data_fetcher import fetch_stock_data

# ==========================================
# 1. 核心计算逻辑
# ==========================================

def calculate_dark_cloud_cover(
        df: pd.DataFrame,
        ma_len: int = 20,
        vol_ma_len: int = 20,
        gap_strict: bool = True,       # 是否严格要求开盘价 > 昨日最高价
        gap_tolerance_pct: float = 0.05, # 如果不严格，允许的偏差比例
        require_midpoint: bool = True, # 是否要求收盘价跌破昨日实体中点
        use_trend_filter: bool = True, # 是否只在上升趋势（价格 > MA）中识别
        use_vol_filter: bool = False,  # 是否要求成交量放大
        use_confirm: bool = False,     # 是否需要次日确认
        confirm_shift: int = 1
) -> pd.DataFrame:
    """
    计算乌云盖顶形态及相关阻力位
    """
    # 避免修改原始数据
    data = df.copy()

    # --- 基础均线 ---
    data["MA"] = data["Close"].rolling(ma_len).mean()
    data["VOL_MA"] = data["Volume"].rolling(vol_ma_len).mean()

    # --- 准备前后两日数据 ---
    # 1 代表前一天 (Prev), 2 代表当天 (Curr)
    o1, h1, l1, c1 = [data[c].shift(1) for c in ["Open", "High", "Low", "Close"]]
    o2, h2, l2, c2 = data["Open"], data["High"], data["Low"], data["Close"]

    # --- 形态基础条件 ---
    bull1 = c1 > o1  # 昨日收阳
    bear2 = c2 < o2  # 今日收阴

    # --- 跳空高开逻辑 ---
    if gap_strict:
        # 严格模式：今日开盘价 > 昨日最高价
        gap_ok = o2 > h1
    else:
        # 宽容模式：允许稍微低一点，或者只是高于昨日收盘价
        # 这里沿用原代码逻辑：高于 (昨日最高价 * (1 - 容差))
        gap_ok = o2 >= (h1 * (1 - gap_tolerance_pct))

    # --- 深入实体逻辑 ---
    # 是否跌破昨日实体中点
    midpoint = (o1 + c1) / 2
    if require_midpoint:
        deep_into = c2 < midpoint
    else:
        # 只要收盘价低于昨日收盘价（也就是变成阴线且重心下移）
        deep_into = c2 < c1

    # --- 过滤器 ---
    # 趋势过滤：当前必须处于均线之上（上升趋势中出现的顶部反转才有效）
    trend_ok = (data["Close"] > data["MA"]) if use_trend_filter else pd.Series(True, index=data.index)

    # 成交量过滤：今日成交量 > 均量
    vol_ok = (data["Volume"] > data["VOL_MA"]) if use_vol_filter else pd.Series(True, index=data.index)

    # --- 综合判断 ---
    dcc_basic = bull1 & bear2 & gap_ok & deep_into & trend_ok & vol_ok

    # --- 确认信号 (Future Data) ---
    if use_confirm:
        # 次日收盘价继续下跌
        confirm_ok = data["Close"].shift(-confirm_shift) < data["Close"]
        dcc = dcc_basic & confirm_ok
    else:
        dcc = dcc_basic

    data["Bear_DCC"] = dcc

    # --- 阻力位计算 ---
    # 乌云盖顶的最高点（通常是两日最高价的最大值）形成阻力
    dcc_resist = np.maximum(h1, h2)
    data["DCC_Resist"] = np.where(data["Bear_DCC"], dcc_resist, np.nan)

    # 向后填充，形成阻力线用于绘图
    data["DCC_Resist_plot"] = data["DCC_Resist"].ffill()

    return data


# ==========================================
# 2. 主程序
# ==========================================

def main():
    # --- 参数设置 ---
    symbol_code = '600519'  # 贵州茅台
    days_back = 365

    strategy_params = {
        'ma_len': 20,
        'vol_ma_len': 20,
        'gap_strict': True,         # 严格跳空
        'require_midpoint': True,   # 必须跌破中点 (经典定义)
        'use_trend_filter': True,   # 必须在 MA 之上
        'use_vol_filter': False,
        'use_confirm': False
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
    df = (df_raw
          .rename(columns={'日期': 'Date',
                           '开盘': 'Open',
                           '最高': 'High',
                           '最低': 'Low',
                           '收盘': 'Close',
                           '成交量': 'Volume'})
          .loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
          .assign(Date=lambda x: pd.to_datetime(x['Date']))
          .set_index('Date')
          .sort_index())

    print("正在计算乌云盖顶形态...")
    # --- 2. 计算 ---
    df = calculate_dark_cloud_cover(df, **strategy_params)

    # 打印发现次数
    dcc_count = df["Bear_DCC"].sum()
    print(f"在过去 {days_back} 天内发现 {dcc_count} 次乌云盖顶形态。")

    # --- 3. 可视化 ---
    print("正在绘图...")

    # 基础均线
    apds = [mpf.make_addplot(df["MA"], color='blue', width=1.0)]

    # 信号标记 (红色倒三角)
    # 计算 Range 用于确定标记位置
    rng = (df["High"] - df["Low"]).replace(0, 1e-6)
    dcc_marks = np.where(df["Bear_DCC"], df["High"] + rng * 0.15, np.nan)

    # 检查是否存在信号，防止全 NaN 报错
    if not np.all(np.isnan(dcc_marks)):
        apds.append(
            mpf.make_addplot(dcc_marks, type="scatter", marker="v", markersize=80, color='red', label='DCC')
        )

    # 阻力线
    if not np.all(np.isnan(df["DCC_Resist_plot"])):
        apds.append(
            mpf.make_addplot(df["DCC_Resist_plot"], color='red', linestyle='--', width=0.8, alpha=0.7)
        )

    # 绘图
    mpf.plot(
        df,
        type="candle",
        volume=True,
        addplot=apds,
        title=f"{symbol_code} Dark Cloud Cover (DCC) + Resistance",
        style="yahoo",
        figsize=(14, 8),
        tight_layout=True,
        block=True
    )
    print("完成。")

if __name__ == "__main__":
    main()