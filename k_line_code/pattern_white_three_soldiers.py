# -*- coding: utf-8 -*-
"""
白色三兵 (Three White Soldiers) 与 前方受阻 (Advance Block) 识别脚本
形态特征：
1. 白色三兵：连续三根阳线，收盘价逐级抬高，处于下跌趋势底部或整理区间，预示反转。
2. 前方受阻：虽然也是三连阳，但实体逐级缩小或上影线变长，预示上涨动能减弱。
"""

import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
from k_line_code.common.data_fetcher import fetch_stock_data

# ==========================================
# 1. 核心识别逻辑
# ==========================================

def identify_three_soldiers(
        df: pd.DataFrame,
        ma_len: int = 20,
        body_min_pct: float = 0.50,     # 三兵要求实体至少占振幅的 50%
        use_trend_filter: bool = True,  # 是否要求出现在下跌趋势中
        shadow_tolerance: float = 0.02  # 理想的三兵下影线不宜过长
) -> pd.DataFrame:
    """
    识别白色三兵及前方受阻形态
    """
    data = df.copy()

    # --- 1. 基础指标 ---
    data["Range"] = (data["High"] - data["Low"]).replace(0, 1e-6)
    data["Body"]  = (data["Close"] - data["Open"]).abs()
    data["MA"]    = data["Close"].rolling(ma_len).mean()

    # 上影线长度
    data["US"] = data["High"] - data["Close"] # 对于阳线，上影线 = High - Close

    # 辅助函数：获取前 k 天数据
    def sh(col, k): return data[col].shift(k)

    # t(今), t-1, t-2
    c0, o0, b0, r0, us0 = data["Close"], data["Open"], data["Body"], data["Range"], data["US"]
    c1, o1, b1, r1, us1 = sh("Close", 1), sh("Open", 1), sh("Body", 1), sh("Range", 1), sh("US", 1)
    c2, o2, b2, r2, us2 = sh("Close", 2), sh("Open", 2), sh("Body", 2), sh("Range", 2), sh("US", 2)

    # --- 2. 基础形态：三连阳且重心上移 ---

    # 均为阳线
    all_white = (c0 > o0) & (c1 > o1) & (c2 > o2)

    # 收盘价逐级抬高 (Close[t] > Close[t-1])
    higher_closes = (c0 > c1) & (c1 > c2)

    # 开盘价逻辑 (标准三兵要求开盘价在前一根实体内部，A股可适当放宽为不大幅低开)
    # 这里使用宽松逻辑：开盘价 > 前一日开盘价
    higher_opens = (o0 > o1) & (o1 > o2)

    basic_shape = all_white & higher_closes & higher_opens

    # --- 3. 形态分类 ---

    # === A. 标准白色三兵 (Strong) ===
    # 1. 实体都比较长 (有力)
    strong_bodies = (b0 > body_min_pct * r0) & \
                    (b1 > body_min_pct * r1) & \
                    (b2 > body_min_pct * r2)

    # 2. 趋势过滤 (出现在下跌趋势或均线下方)
    if use_trend_filter:
        # 第一根兵出现时，处于均线下方 (反转信号)
        downtrend = c2 < sh("MA", 2)
    else:
        downtrend = pd.Series(True, index=data.index)

    data["White_Three_Soldiers"] = basic_shape & strong_bodies & downtrend

    # === B. 前方受阻 (Advance Block / Resistance) ===
    # 定义：三连阳，但后续K线实体变小，或出现长上影线

    # 条件1: 第2根或第3根实体明显变小 (例如小于第一根的 60%)
    shrinking_bodies = (b1 < 0.6 * b2) | (b2 < 0.6 * b1)

    # 条件2: 第2根或第3根出现长上影线 (上影线 > 实体)
    long_shadows = (us1 > b1) | (us2 > b0) # 注意索引: b0是今天(第3根)

    # 它是三连阳，但满足“实体缩小”或“长上影”
    # 注意：前方受阻通常发生在上升趋势中途或顶部
    data["Advance_Block"] = basic_shape & (shrinking_bodies | long_shadows)

    # 互斥处理：如果是标准三兵，就不标记为前方受阻 (优先标记三兵)
    data["Advance_Block"] = data["Advance_Block"] & (~data["White_Three_Soldiers"])

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
        'body_min_pct': 0.50,       # 实体需占振幅一半以上
        'use_trend_filter': True    # 仅识别低位反转的三兵
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

    print("正在计算形态...")

    # --- 2. 计算形态 ---
    df = identify_three_soldiers(df, **strategy_params)

    # 统计
    n_wts = df["White_Three_Soldiers"].sum()
    n_blk = df["Advance_Block"].sum()

    print("-" * 30)
    print(f"统计结果 (最近 {days_back} 天):")
    print(f"白色三兵 (White Three Soldiers): {n_wts} 次 (强势反转)")
    print(f"前方受阻 (Advance Block):        {n_blk} 次 (动能减弱)")
    print("-" * 30)

    if n_wts + n_blk == 0:
        print("提示：未发现形态，可尝试放宽 body_min_pct 或关闭 use_trend_filter。")

    # --- 3. 可视化 ---
    print("正在绘图...")

    apds = [mpf.make_addplot(df["MA"], color='tab:blue', width=1.0)]

    rng = df["Range"].replace(0, 1e-6)

    # 标记位置
    # 白色三兵：绿色向上箭头 (看涨)
    wts_marks = np.where(df["White_Three_Soldiers"], df["Low"] - rng * 0.2, np.nan)
    # 前方受阻：橙色向下箭头 (警示)
    blk_marks = np.where(df["Advance_Block"], df["High"] + rng * 0.2, np.nan)

    # 添加信号图层
    if not np.all(np.isnan(wts_marks)):
        apds.append(mpf.make_addplot(wts_marks, type="scatter", marker="^", markersize=100, color="green", label="3 Soldiers"))

    if not np.all(np.isnan(blk_marks)):
        apds.append(mpf.make_addplot(blk_marks, type="scatter", marker="v", markersize=100, color="orange", label="Blocked"))

    # 绘图
    title_str = f'{symbol_code} Three White Soldiers & Advance Block'

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