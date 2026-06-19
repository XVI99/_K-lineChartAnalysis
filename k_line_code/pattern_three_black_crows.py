# -*- coding: utf-8 -*-
"""
三只乌鸦 (Three Black Crows) 形态识别与可视化脚本
形态特征 (顶部强反转):
1. 趋势：处于上升趋势中。
2. 连续出现三根阴线 (Black Candles)。
3. 每日收盘价都低于前一日收盘价 (Lower Closes)。
4. 每日开盘价通常在前一日实体内部 (Opens within previous body)。
5. 实体通常较长，下影线较短。
"""

from k_line_code.common.data_fetcher import fetch_stock_data
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta

# ==========================================
# 1. 核心识别逻辑
# ==========================================

def identify_three_black_crows(
        df: pd.DataFrame,
        ma_len: int = 20,
        use_trend_filter: bool = True,
        body_min_pct: float = 0.40,     # 实体至少占当日振幅的 40% (避免选中十字星)
        shadow_max_pct: float = 0.50,   # 下影线不能太长
        open_in_prev_body: bool = True, # 严格模式：开盘价位于前一日实体内
        consecutive_lower: bool = True  # 严格模式：收盘价不断创新低
) -> pd.DataFrame:
    """
    识别三只乌鸦形态
    """
    data = df.copy()

    # --- 1. 基础指标 ---
    data["Range"] = (data["High"] - data["Low"]).replace(0, 1e-6)
    data["Body"]  = (data["Close"] - data["Open"]).abs()
    data["MA"]    = data["Close"].rolling(ma_len).mean()

    # 辅助函数：获取第前 k 天的数据
    def sh(col, k): return data[col].shift(k)

    # 获取 t(今), t-1(昨), t-2(前) 的数据
    # 0=Today, 1=Yesterday, 2=DayBefore
    c0, o0, l0, h0, b0, r0 = data["Close"], data["Open"], data["Low"], data["High"], data["Body"], data["Range"]
    c1, o1, l1, h1, b1, r1 = sh("Close", 1), sh("Open", 1), sh("Low", 1), sh("High", 1), sh("Body", 1), sh("Range", 1)
    c2, o2, l2, h2, b2, r2 = sh("Close", 2), sh("Open", 2), sh("Low", 2), sh("High", 2), sh("Body", 2), sh("Range", 2)

    # --- 2. 形态逻辑 ---

    # A. 三根都是阴线
    is_black0 = c0 < o0
    is_black1 = c1 < o1
    is_black2 = c2 < o2
    all_black = is_black0 & is_black1 & is_black2

    # B. 实体不能太小 (排除连续下跌的十字星)
    has_body0 = b0 > r0 * body_min_pct
    has_body1 = b1 > r1 * body_min_pct
    has_body2 = b2 > r2 * body_min_pct
    decent_bodies = has_body0 & has_body1 & has_body2

    # C. 收盘价不断创新低 (重心下移)
    if consecutive_lower:
        lower_closes = (c0 < c1) & (c1 < c2)
        lower_lows   = (l0 < l1) & (l1 < l2) # 可选，增加严格性
    else:
        lower_closes = pd.Series(True, index=data.index)
        lower_lows   = pd.Series(True, index=data.index)

    # D. 开盘价在前一日实体内部 (标准定义，A股常用)
    # 今天的开盘价 < 昨天的开盘价 (且最好 > 昨天的收盘价，即在实体内)
    if open_in_prev_body:
        # 只要开盘价不高于昨日开盘价即可 (允许跳空低开，但不允许大幅高开)
        opens_ok = (o0 < o1) & (o1 < o2)
    else:
        opens_ok = pd.Series(True, index=data.index)

    # E. 趋势过滤
    # 形态出现前是上升趋势
    if use_trend_filter:
        # 检查第一只乌鸦出现前的趋势 (t-3) 或者简单的均线判断
        # 这里用：第一只乌鸦(t-2)收盘价 > MA，或者 t-3 > MA
        trend_ok = sh("Close", 2) > sh("MA", 2)
    else:
        trend_ok = pd.Series(True, index=data.index)

    # 综合判断
    three_crows = all_black & decent_bodies & lower_closes & opens_ok & trend_ok

    data["ThreeBlackCrows"] = three_crows

    # --- 3. 阻力位 ---
    # 三只乌鸦的起始点 (第一根的最高价) 是极强的阻力
    data["Crow_Res"] = np.where(three_crows, h2, np.nan)
    data["Crow_Res_plot"] = data["Crow_Res"].ffill()

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
        'use_trend_filter': True,
        'body_min_pct': 0.40,       # 实体长度门槛
        'open_in_prev_body': False, # A股常有跳空低开，设为 False 可捕捉更多“跌势汹汹”的形态
        'consecutive_lower': True
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
    df = (
        df_raw
        .rename(columns={'日期': 'Date', '开盘': 'Open', '最高': 'High',
                         '最低': 'Low', '收盘': 'Close', '成交量': 'Volume'})
        .loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        .assign(Date=lambda x: pd.to_datetime(x['Date']))
        .set_index('Date')
        .sort_index()
    )

    print("正在计算三只乌鸦形态...")

    # --- 2. 计算形态 ---
    df = identify_three_black_crows(df, **strategy_params)

    # 统计
    n_sig = df["ThreeBlackCrows"].sum()
    print("-" * 30)
    print(f"统计结果 (最近 {days_back} 天):")
    print(f"三只乌鸦 (Three Black Crows): {n_sig} 次")
    print("-" * 30)

    if n_sig == 0:
        print("提示：标准三只乌鸦要求严格（3连阴且实体较大）。\n可尝试将 body_min_pct 调低 (如 0.3) 或关闭 use_trend_filter。")

    # --- 3. 可视化 ---
    print("正在绘图...")

    apds = [mpf.make_addplot(df["MA"], color='tab:blue', width=1.0)]

    rng = df["Range"].replace(0, 1e-6)

    # 标记：在第三根乌鸦的高点上方画标记
    marks = np.where(df["ThreeBlackCrows"], df["High"] + rng * 0.2, np.nan)

    # 阻力线
    res_line = df["Crow_Res_plot"]

    if not np.all(np.isnan(marks)):
        apds.append(mpf.make_addplot(marks, type="scatter", marker="v", markersize=120, color="black", label="3 Crows"))

    if not np.all(np.isnan(res_line)):
        apds.append(mpf.make_addplot(res_line, color="black", linestyle='--', alpha=0.6))

    title_str = f'{symbol_code} Three Black Crows Pattern'

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