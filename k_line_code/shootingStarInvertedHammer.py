# -*- coding: utf-8 -*-
"""
流星形态 (Shooting Star) 与 倒锤子形态 (Inverted Hammer) 识别脚本
形态特征：
1. 实体很小 (Small Body)。
2. 上影线很长 (Long Upper Shadow, 通常 > 实体的2倍)。
3. 下影线很短或没有 (Tiny Lower Shadow)。
区别：
- 流星：出现在上升趋势 (看跌)。
- 倒锤子：出现在下降趋势 (看涨，通常需要次日确认)。
"""

import akshare as ak
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta

# ==========================================
# 1. 数据获取函数
# ==========================================
def get_stock_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    获取最近 N 天的 A 股复权数据并清洗格式
    """
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days)

    print(f"正在获取 {symbol} 最近 {days} 天的数据...")
    try:
        df_raw = ak.stock_zh_a_hist(symbol=symbol,
                                    period='daily',
                                    start_date=start_dt.strftime('%Y%m%d'),
                                    end_date=end_dt.strftime('%Y%m%d'),
                                    adjust='qfq')
    except Exception as e:
        print(f"数据获取失败: {e}")
        return pd.DataFrame()

    if df_raw.empty:
        print("未获取到数据，请检查代码或网络。")
        return pd.DataFrame()

    # 整理成英文列 + Date 索引
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

    return df

# ==========================================
# 2. 形态识别函数 (逻辑保持不变)
# ==========================================
def identify_shooting_inverted(df_in: pd.DataFrame,
                               ma_len: int = 20,
                               body_max_pct: float = 0.30,
                               upper_to_body_min: float = 2.0,
                               upper_min_pct: float = 0.50,
                               lower_max_pct: float = 0.10,
                               use_trend_filter: bool = True,
                               confirm_inverted: bool = True,
                               confirm_shooting: bool = False,
                               confirm_bars: int = 1) -> pd.DataFrame:
    """
    识别流星线 (Shooting Star) 和倒锤子 (Inverted Hammer)
    """
    df = df_in.copy()

    # 基础指标计算
    df["Range"] = (df["High"] - df["Low"]).replace(0, 1e-6)
    df["Body"]  = (df["Close"] - df["Open"]).abs()
    df["US"]    = df["High"] - np.maximum(df["Open"], df["Close"]) # 上影线
    df["LS"]    = np.minimum(df["Open"], df["Close"]) - df["Low"]  # 下影线
    df["MA"]    = df["Close"].rolling(ma_len).mean()

    # 形态逻辑
    small_body = df["Body"] <= body_max_pct * df["Range"]
    long_upper = (df["US"] >= upper_to_body_min * df["Body"]) & (df["US"] >= upper_min_pct * df["Range"])
    tiny_lower = df["LS"] <= lower_max_pct * df["Range"]
    shape_ok   = small_body & long_upper & tiny_lower

    # 趋势过滤
    up_trend   = df["Close"] > df["MA"]
    down_trend = df["Close"] < df["MA"]
    trend_up_ok   = up_trend if use_trend_filter else pd.Series(True, index=df.index)
    trend_down_ok = down_trend if use_trend_filter else pd.Series(True, index=df.index)

    # --- 流星 (Shooting Star) ---
    entity_low = np.minimum(df["Open"], df["Close"])
    shooting_basic = shape_ok & trend_up_ok
    # 确认信号：未来 N 日收盘价跌破实体低点
    shooting_confirm_ok = df["Close"].shift(-confirm_bars) < entity_low
    df["ShootingStar"] = shooting_basic & (shooting_confirm_ok if confirm_shooting else True)

    # --- 倒锤子 (Inverted Hammer) ---
    entity_high = np.maximum(df["Open"], df["Close"])
    inverted_basic = shape_ok & trend_down_ok
    # 确认信号：未来 N 日收盘价突破实体高点
    inverted_confirm_ok = df["Close"].shift(-confirm_bars) > entity_high
    df["InvertedHammer"] = inverted_basic & (inverted_confirm_ok if confirm_inverted else True)

    return df

# ==========================================
# 3. 可视化绘图函数
# ==========================================
def plot_patterns(df: pd.DataFrame, symbol_code: str):
    """
    绘制 K 线及流星/倒锤子标记
    """
    # 结果统计
    n_shoot = df["ShootingStar"].sum()
    n_invh = df["InvertedHammer"].sum()
    print(f'=== {symbol_code} 统计结果 ===')
    print(f'流星出现次数：{n_shoot}')
    print(f'倒锤子出现次数：{n_invh}')

    if n_shoot + n_invh == 0:
        print('当前参数下无流星/倒锤子形态，可放宽条件再试')

    # 准备绘图
    ap = [mpf.make_addplot(df["MA"], color='tab:blue', width=1.2)]

    rng = df["Range"].replace(0, 1e-6)

    # 标记坐标
    shoot_marks = np.where(df["ShootingStar"], df["High"] + rng * 0.18, np.nan)
    invh_marks = np.where(df["InvertedHammer"], df["Low"] - rng * 0.18, np.nan)

    # 只把「至少有一个有效值」的图层加进去，避免全 nan 报错
    if not np.all(np.isnan(shoot_marks)):
        ap.append(mpf.make_addplot(shoot_marks, type="scatter", marker="v", markersize=90, color="tab:red", label='Shooting Star'))
    if not np.all(np.isnan(invh_marks)):
        ap.append(mpf.make_addplot(invh_marks, type="scatter", marker="^", markersize=90, color="tab:green", label='Inverted Hammer'))

    # 绘图
    mpf.plot(df, type="candle", volume=True, addplot=ap,
             title=f'{symbol_code} Shooting Star & Inverted Hammer',
             style="yahoo", figsize=(14, 8))

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 配置
    TARGET_SYMBOL = '600519'  # 贵州茅台

    # 1. 获取数据
    df_data = get_stock_data(TARGET_SYMBOL, days=365)

    if not df_data.empty:
        # 2. 计算形态
        df_patterns = identify_shooting_inverted(df_data,
                                                 ma_len=20,
                                                 body_max_pct=0.30,
                                                 upper_to_body_min=2.0,
                                                 upper_min_pct=0.50,
                                                 lower_max_pct=0.10,
                                                 use_trend_filter=True,
                                                 confirm_inverted=True,
                                                 confirm_shooting=False)

        # 3. 绘图
        plot_patterns(df_patterns, TARGET_SYMBOL)