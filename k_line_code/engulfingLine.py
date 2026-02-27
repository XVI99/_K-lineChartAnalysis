# -*- coding: utf-8 -*-
"""
吞没形态（Engulfing Pattern）识别与可视化脚本
包含：看涨吞没 (Bullish Engulfing) & 看跌吞没 (Bearish Engulfing)
"""

import akshare as ak
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta

# ==========================================
# 1. 核心计算逻辑
# ==========================================

def calculate_engulfing(
        df: pd.DataFrame,
        ma_len: int = 20,
        vol_ma_len: int = 20,
        small_body_pct: float = 0.10,
        use_equal_ok: bool = True,
        vol_filter: bool = False,
        confirm_close: bool = False,
        confirm_shift: int = 1
) -> pd.DataFrame:
    """
    计算吞没形态及相关支撑阻力位
    """
    # 避免修改原始数据
    data = df.copy()

    # --- 基础指标计算 ---
    data["Body"] = (data["Close"] - data["Open"]).abs()
    data["Range"] = (data["High"] - data["Low"]).replace(0, np.nan)
    data["Bull"] = (data["Close"] > data["Open"]).astype(int)
    data["Bear"] = (data["Close"] < data["Open"]).astype(int)

    # 识别“小实体”（用于判断前一根K线是否为震荡/纺锤/十字星）
    data["DojiSmall"] = (data["Body"] <= small_body_pct * data["Range"]).fillna(False)

    # 均线与趋势
    data["MA"] = data["Close"].rolling(ma_len).mean()
    data["VOL_MA"] = data["Volume"].rolling(vol_ma_len).mean()
    data["DownTrend"] = data["Close"] < data["MA"]
    data["UpTrend"] = data["Close"] > data["MA"]

    # --- 准备前后两日的比较数据 ---
    # 1 代表前一天 (Previous)，2 代表当天 (Current)
    o1, h1, l1, c1 = [data[c].shift(1) for c in ["Open", "High", "Low", "Close"]]
    bull1, bear1 = data["Bull"].shift(1) == 1, data["Bear"].shift(1) == 1
    doji1 = data["DojiSmall"].shift(1)

    o2, h2, l2, c2 = data["Open"], data["High"], data["Low"], data["Close"]
    bull2, bear2 = data["Bull"] == 1, data["Bear"] == 1

    # --- 核心形态判断：包容（Engulfing） ---
    # 判断当天的实体范围是否完全包裹住了前一天的实体范围
    if use_equal_ok:
        # 允许相等（包含）
        engulf_body = (np.minimum(o2, c2) <= np.minimum(o1, c1)) & \
                      (np.maximum(o2, c2) >= np.maximum(o1, c1))
    else:
        # 严格包裹（大于/小于）
        engulf_body = (np.minimum(o2, c2) < np.minimum(o1, c1)) & \
                      (np.maximum(o2, c2) > np.maximum(o1, c1))

    # --- 颜色反转逻辑 ---
    # 通常要求一阴一阳，或者前一个是极小的十字星
    color_opposite_or_doji = ((bull2 & bear1) | (bear2 & bull1) | doji1.fillna(False))

    # --- 成交量过滤（可选） ---
    vol_ok = data["Volume"] > data["VOL_MA"] if vol_filter else pd.Series(True, index=data.index)

    # --- 基础信号 ---
    # 看涨吞没：形态满足 + 颜色反转 + 处于下降趋势 + 当天收阳 + 成交量达标
    bull_basic = engulf_body & color_opposite_or_doji & data["DownTrend"] & bull2 & vol_ok

    # 看跌吞没：形态满足 + 颜色反转 + 处于上升趋势 + 当天收阴 + 成交量达标
    bear_basic = engulf_body & color_opposite_or_doji & data["UpTrend"] & bear2 & vol_ok

    # --- 确认信号（未来数据验证） ---
    if confirm_close:
        # 看涨确认：后一天收盘价 > 当天收盘价
        bull_confirm = data["Close"].shift(-confirm_shift) > data["Close"]
        # 看跌确认：后一天收盘价 < 当天收盘价
        bear_confirm = data["Close"].shift(-confirm_shift) < data["Close"]

        data["Bull_Engulf"] = bull_basic & bull_confirm
        data["Bear_Engulf"] = bear_basic & bear_confirm
    else:
        data["Bull_Engulf"] = bull_basic
        data["Bear_Engulf"] = bear_basic

    # --- 支撑/阻力位计算 ---
    # 看涨吞没的支撑位通常取两根K线的最低点
    bull_sr = np.minimum(l1, l2)
    # 看跌吞没的阻力位通常取两根K线的最高点
    bear_sr = np.maximum(h1, h2)

    data["Bull_SR"] = np.where(data["Bull_Engulf"], bull_sr, np.nan)
    data["Bear_SR"] = np.where(data["Bear_Engulf"], bear_sr, np.nan)

    # 为了画图方便，将支撑阻力位向后填充（ffill），形成线条
    data["Bull_SR_plot"] = data["Bull_SR"].ffill()
    data["Bear_SR_plot"] = data["Bear_SR"].ffill()

    return data


# ==========================================
# 2. 主程序
# ==========================================

def main():
    # --- 参数设置 ---
    symbol_code = '601111'  # 股票代码：中国国航 (原示例)
    days_back = 365         # 回溯天数

    # 策略参数
    strategy_params = {
        'ma_len': 20,
        'vol_ma_len': 20,
        'small_body_pct': 0.10,
        'use_equal_ok': True,
        'vol_filter': False,      # 是否开启成交量过滤
        'confirm_close': False,   # 是否需要次日确认
        'confirm_shift': 1
    }

    print(f"正在获取 {symbol_code} 的历史数据...")

    # --- 1. 获取数据 ---
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days_back)

    try:
        df_raw = ak.stock_zh_a_hist(symbol=symbol_code,
                                    period='daily',
                                    start_date=start_dt.strftime('%Y%m%d'),
                                    end_date=end_dt.strftime('%Y%m%d'),
                                    adjust='qfq')
    except Exception as e:
        print(f"数据获取失败: {e}")
        return

    if df_raw.empty:
        print("未获取到数据，请检查代码或日期。")
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

    print("正在计算吞没形态...")

    # --- 2. 计算形态 ---
    df = calculate_engulfing(df, **strategy_params)

    # 统计信号数量
    bull_count = df["Bull_Engulf"].sum()
    bear_count = df["Bear_Engulf"].sum()
    print(f"识别结果: 看涨吞没 {bull_count} 次, 看跌吞没 {bear_count} 次")

    # --- 3. 可视化 (mplfinance) ---
    print("正在绘图...")

    # 构造附加图层 (Add Plots)
    apds = [
        # 主图均线
        mpf.make_addplot(df["MA"], panel=0, width=1.0, color='blue', alpha=0.5),
    ]

    # 信号标记点位置
    bull_marks = np.where(df["Bull_Engulf"], df["Low"] - df["Range"] * 0.15, np.nan)
    bear_marks = np.where(df["Bear_Engulf"], df["High"] + df["Range"] * 0.15, np.nan)

    # 添加信号标记 (如果有信号)
    if not np.all(np.isnan(bull_marks)):
        apds.append(mpf.make_addplot(bull_marks, type="scatter", marker="^", markersize=80, color='green', label='Bull Engulf'))

    if not np.all(np.isnan(bear_marks)):
        apds.append(mpf.make_addplot(bear_marks, type="scatter", marker="v", markersize=80, color='red', label='Bear Engulf'))

    # 添加支撑/阻力线 (使用 ffill 后的数据绘制连续线条)
    # 注意：为了避免线条过于杂乱，你可以选择只画最近的，或者保留全部
    apds += [
        mpf.make_addplot(df["Bull_SR_plot"], panel=0, color='green', linestyle='--', width=0.8, alpha=0.7),
        mpf.make_addplot(df["Bear_SR_plot"], panel=0, color='red', linestyle='--', width=0.8, alpha=0.7)
    ]

    # 绘图
    # 注意：style='yahoo' 比较经典，volume=True 显示成交量
    mpf.plot(
        df,
        type="candle",
        volume=True,
        addplot=apds,
        title=f"Engulfing Pattern: {symbol_code}", # 建议用英文标题防止乱码
        style="yahoo",
        figsize=(14, 8),
        tight_layout=True,
        block=True
    )
    print("完成。")

if __name__ == "__main__":
    main()