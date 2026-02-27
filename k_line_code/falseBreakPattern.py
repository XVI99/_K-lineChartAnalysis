import akshare as ak
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ==========================================
# 0. 字体配置 (解决绘图中文乱码)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# 创建支持中文的 mplfinance 样式
my_style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.family': 'SimHei'})

# ==========================================
# 1. 数据获取函数
# ==========================================
def get_stock_data(symbol: str, days: int = 365) -> pd.DataFrame:
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
        return pd.DataFrame()

    df = (df_raw
          .rename(columns={'日期': 'Date', '开盘': 'Open', '最高': 'High',
                           '最低': 'Low', '收盘': 'Close', '成交量': 'Volume'})
          .loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
          .assign(Date=lambda x: pd.to_datetime(x['Date']))
          .set_index('Date')
          .sort_index())

    return df

# ==========================================
# 2. 形态识别逻辑
# ==========================================
def identify_patterns(df_in):
    df = df_in.copy()

    # --- 基础指标 ---
    ma_len = 20
    # 防止除以零
    df["Range"] = (df["High"] - df["Low"]).replace(0, 1e-6)
    df["Body"]  = (df["Close"] - df["Open"]).abs()
    df["MA"]    = df["Close"].rolling(ma_len).mean()

    # 计算影线
    upper_shadow = df["High"] - np.maximum(df["Open"], df["Close"])
    lower_shadow = np.minimum(df["Open"], df["Close"]) - df["Low"]

    # 布林带 (用于判断假突破)
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # ================================
    # 形态 A: 捉腰带线 (Belt Hold)
    # ================================
    # 参数设置
    body_min_pct = 0.60          # 实体至少占全长的 60%
    same_side_shadow_max = 0.05  # 开盘一侧的影线极短 (光头/光脚)
    other_side_shadow_max = 0.20 # 另一侧影线也不宜过长

    # 1. 看涨捉腰带 (Bullish Belt Hold)
    # 阳线 + 实体大 + 下影线极短 + 下跌趋势
    bullish_long_body = (df["Close"] > df["Open"]) & (df["Body"] >= body_min_pct * df["Range"])
    bull_small_lower  = (lower_shadow <= same_side_shadow_max * df["Range"])
    bull_not_long_upper = (upper_shadow <= other_side_shadow_max * df["Range"])

    # 简单趋势过滤 (昨日收盘在均线下方)
    trend_down = (df["Close"].shift(1) < df["MA"].shift(1))

    df["Bull_BeltHold"] = bullish_long_body & bull_small_lower & bull_not_long_upper & trend_down
    df["BH_Support"] = np.where(df["Bull_BeltHold"], df["Low"], np.nan)

    # 2. 看跌捉腰带 (Bearish Belt Hold)
    # 阴线 + 实体大 + 上影线极短 + 上升趋势
    bearish_long_body = (df["Close"] < df["Open"]) & (df["Body"] >= body_min_pct * df["Range"])
    bear_small_upper  = (upper_shadow <= same_side_shadow_max * df["Range"])
    bear_not_long_lower = (lower_shadow <= other_side_shadow_max * df["Range"])

    # 简单趋势过滤 (昨日收盘在均线上方)
    trend_up = (df["Close"].shift(1) > df["MA"].shift(1))

    df["Bear_BeltHold"] = bearish_long_body & bear_small_upper & bear_not_long_lower & trend_up
    df["BH_Resistance"] = np.where(df["Bear_BeltHold"], df["High"], np.nan)

    # ================================
    # 形态 B: 假突破 (False Break)
    # ================================

    # 1. 破低反涨 (空头陷阱 / Bear Trap)
    # 逻辑：最低价跌破昨日布林下轨，但收盘价收回下轨上方，且为阳线
    support_breakdown = (df["Low"] < bb_lower.shift(1)) & (df["Close"] > bb_lower.shift(1))
    bullish_candle = df["Close"] > df["Open"]

    df["Breakdown_Rev"] = support_breakdown & bullish_candle

    # 目标位：近期高点
    lookback_target = 20
    recent_high = df["High"].rolling(lookback_target).max().shift(1)
    df["BD_Target"] = np.where(df["Breakdown_Rev"], recent_high, np.nan)

    # 2. 破高反跌 (多头陷阱 / Bull Trap)
    # 逻辑：最高价突破昨日布林上轨，但收盘价跌回上轨下方，且为阴线
    resistance_breakout = (df["High"] > bb_upper.shift(1)) & (df["Close"] < bb_upper.shift(1))
    bearish_candle = df["Close"] < df["Open"]

    df["Breakout_Rev"] = resistance_breakout & bearish_candle

    # 目标位：近期低点
    recent_low = df["Low"].rolling(lookback_target).min().shift(1)
    df["BO_Target"] = np.where(df["Breakout_Rev"], recent_low, np.nan)

    return df

# ==========================================
# 3. 可视化绘图函数
# ==========================================
def plot_patterns(df, symbol):
    ap = []

    # 1. 均线 MA20
    ap.append(mpf.make_addplot(df["MA"], color='tab:blue', width=1.0))

    rng = df["Range"].mean()

    # 2. 捉腰带标记
    # 看涨：绿色向上三角
    bull_marks = np.where(df["Bull_BeltHold"], df["Low"] - rng * 0.5, np.nan)
    # 看跌：红色向下三角
    bear_marks = np.where(df["Bear_BeltHold"], df["High"] + rng * 0.5, np.nan)

    if not np.all(np.isnan(bull_marks)):
        ap.append(mpf.make_addplot(bull_marks, type='scatter', marker='^',
                                   markersize=100, color='seagreen', label='看涨捉腰带'))
    if not np.all(np.isnan(bear_marks)):
        ap.append(mpf.make_addplot(bear_marks, type='scatter', marker='v',
                                   markersize=100, color='crimson', label='看跌捉腰带'))

    # 3. 假突破标记
    # 破低反涨 (空头陷阱) -> 蓝色圆点
    bd_marks = np.where(df["Breakdown_Rev"], df["Low"] - rng * 0.8, np.nan)
    # 破高反跌 (多头陷阱) -> 橙色圆点
    bo_marks = np.where(df["Breakout_Rev"], df["High"] + rng * 0.8, np.nan)

    if not np.all(np.isnan(bd_marks)):
        ap.append(mpf.make_addplot(bd_marks, type='scatter', marker='o',
                                   markersize=80, color='dodgerblue', label='破低反涨(陷阱)'))
    if not np.all(np.isnan(bo_marks)):
        ap.append(mpf.make_addplot(bo_marks, type='scatter', marker='o',
                                   markersize=80, color='orange', label='破高反跌(陷阱)'))

    # 4. 支撑/阻力线 (虚线阶梯)
    for col, color in [('BH_Support', 'seagreen'), ('BH_Resistance', 'crimson')]:
        if not df[col].isna().all():
            plot_col = pd.Series(df[col]).ffill()
            ap.append(mpf.make_addplot(plot_col, color=color, linestyle='--', width=1))

    # 5. 目标位线 (点线阶梯)
    for col, color in [('BD_Target', 'dodgerblue'), ('BO_Target', 'orange')]:
        if not df[col].isna().all():
            plot_col = pd.Series(df[col]).ffill()
            ap.append(mpf.make_addplot(plot_col, color=color, linestyle=':', width=1))

    print("正在生成图表...")
    mpf.plot(df, type="candle", volume=True, addplot=ap,
             title=f'{symbol} - 捉腰带线 与 假突破形态',
             style=my_style, figsize=(14, 8))

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519'  # 贵州茅台

    # 1. 获取数据
    df = get_stock_data(symbol_code, days=365)

    if not df.empty:
        # 2. 识别形态
        df_patterns = identify_patterns(df)

        # 3. 打印统计结果
        n_bull_bh = int(df_patterns["Bull_BeltHold"].sum())
        n_bear_bh = int(df_patterns["Bear_BeltHold"].sum())
        n_bd_rev  = int(df_patterns["Breakdown_Rev"].sum())
        n_bo_rev  = int(df_patterns["Breakout_Rev"].sum())

        print("=" * 40)
        print(f"看涨捉腰带线: {n_bull_bh} 次")
        print(f"看跌捉腰带线: {n_bear_bh} 次")
        print(f"破低反涨(空头陷阱): {n_bd_rev} 次")
        print(f"破高反跌(多头陷阱): {n_bo_rev} 次")
        print("=" * 40)

        # 4. 绘图
        if n_bull_bh + n_bear_bh + n_bd_rev + n_bo_rev > 0:
            plot_patterns(df_patterns, symbol_code)
        else:
            print("当前参数设置下未发现相关形态。")