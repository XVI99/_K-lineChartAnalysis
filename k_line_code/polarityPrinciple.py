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
# 2. 支撑/阻力水平识别 (聚类算法)
# ==========================================
def cluster_levels(values, tol):
    """把相近价位聚成一条水平线，返回中心价及被测试次数。"""
    levels = []
    counts = []
    values_sorted = sorted(values)
    if not values_sorted:
        return np.array([]), np.array([])

    current_cluster = [values_sorted[0]]
    for v in values_sorted[1:]:
        if abs(v - np.mean(current_cluster)) / np.mean(current_cluster) <= tol:
            current_cluster.append(v)
        else:
            levels.append(np.mean(current_cluster))
            counts.append(len(current_cluster))
            current_cluster = [v]

    levels.append(np.mean(current_cluster))
    counts.append(len(current_cluster))
    return np.array(levels), np.array(counts)

def identify_sr_levels(df, lookback_swing=5, tol_pct=0.003, min_touch=2):
    """
    识别关键支撑和阻力位
    """
    lows = df['Low']
    highs = df['High']

    # 局部极值判断 (Swing Lows/Highs)
    is_swing_low = (lows == lows.rolling(lookback_swing*2+1, center=True).min())
    is_swing_high = (highs == highs.rolling(lookback_swing*2+1, center=True).max())

    swing_low_points = df[is_swing_low][['Low']]
    swing_high_points = df[is_swing_high][['High']]

    # 聚类
    s_levels, s_counts = cluster_levels(swing_low_points['Low'].tolist(), tol_pct)
    r_levels, r_counts = cluster_levels(swing_high_points['High'].tolist(), tol_pct)

    # 过滤低频水平
    s_levels = s_levels[s_counts >= min_touch]
    r_levels = r_levels[r_counts >= min_touch]

    return s_levels, r_levels

# ==========================================
# 3. 极性转换检测逻辑
# ==========================================
def identify_polarity_flips(df_in, support_levels, resist_levels, retest_window=20, buffer_pct=0.0015):
    """
    检测支撑转阻力 (SupToRes) 和 阻力转支撑 (ResToSup)
    """
    df = df_in.copy()
    close = df['Close']

    # 初始化结果列
    df['SupToRes'] = np.nan
    df['ResToSup'] = np.nan

    # --- A. 支撑 -> 阻力 (跌破后反抽) ---
    for level in support_levels:
        # 1. 找到“首次有效跌破”支撑的索引
        # 条件：前一根 > level, 本根 < level
        broke_idx = df[
            (close.shift(1) > level * (1 + buffer_pct)) &
            (close < level * (1 - buffer_pct))
            ].index

        for bidx in broke_idx:
            # 2. 在跌破后的 retest_window 根K线内寻找回踩
            # 获取 bidx 之后的数据切片
            if bidx == df.index[-1]: continue

            # loc切片是包含两端的，所以我们要找 bidx 之后的
            # 获取位置索引
            loc_idx = df.index.get_loc(bidx)
            if loc_idx + 1 >= len(df): continue

            sub = df.iloc[loc_idx + 1 : loc_idx + 1 + retest_window]

            if sub.empty: continue

            # 寻找回踩：价格回到 level 附近 (反弹受阻)
            retest = sub[
                (sub['Close'] >= level * (1 - buffer_pct)) &
                (sub['Close'] <= level * (1 + buffer_pct))
                ]

            if not retest.empty:
                ridx = retest.index[0]
                # 只有当该位置还没被标记过，或者新的 level 更精准时才标记（简化处理：直接标记）
                df.at[ridx, 'SupToRes'] = level
                # 对于同一次跌破，只记录第一次回踩确认
                # 注意：这里 break 是跳出 sub 循环，继续找下一个 broke_idx
                # 但通常一个 level 在一段时间内只看一次极性转换，这里保持原逻辑
                pass

    # --- B. 阻力 -> 支撑 (突破后回踩) ---
    for level in resist_levels:
        # 1. 找到“首次有效突破”阻力的索引
        # 条件：前一根 < level, 本根 > level
        broke_idx = df[
            (close.shift(1) < level * (1 - buffer_pct)) &
            (close > level * (1 + buffer_pct))
            ].index

        for bidx in broke_idx:
            if bidx == df.index[-1]: continue

            loc_idx = df.index.get_loc(bidx)
            if loc_idx + 1 >= len(df): continue

            sub = df.iloc[loc_idx + 1 : loc_idx + 1 + retest_window]

            if sub.empty: continue

            # 寻找回踩：价格回到 level 附近 (回调获支撑)
            retest = sub[
                (sub['Close'] >= level * (1 - buffer_pct)) &
                (sub['Close'] <= level * (1 + buffer_pct))
                ]

            if not retest.empty:
                ridx = retest.index[0]
                df.at[ridx, 'ResToSup'] = level
                pass

    return df

# ==========================================
# 4. 可视化绘图函数
# ==========================================
def plot_polarity(df, symbol, s_levels, r_levels):
    ap = []

    # 1. 画出所有历史支撑水平 (绿色虚线)
    for lvl in s_levels:
        ap.append(
            mpf.make_addplot(
                pd.Series(lvl, index=df.index),
                color='seagreen',
                linestyle='--',
                width=0.8
            )
        )

    # 2. 画出所有历史阻力水平 (红色虚线)
    for lvl in r_levels:
        ap.append(
            mpf.make_addplot(
                pd.Series(lvl, index=df.index),
                color='crimson',
                linestyle='--',
                width=0.8
            )
        )

    # 3. 极性转换标记
    # 支撑 -> 阻力 (SupToRes): 红色圆点 (表示这里变成了阻力)
    sup2res_marks = np.where(~df['SupToRes'].isna(), df['SupToRes'], np.nan)
    # 阻力 -> 支撑 (ResToSup): 绿色圆点 (表示这里变成了支撑)
    res2sup_marks = np.where(~df['ResToSup'].isna(), df['ResToSup'], np.nan)

    if not np.all(np.isnan(sup2res_marks)):
        ap.append(
            mpf.make_addplot(
                sup2res_marks,
                type='scatter',
                marker='o',
                markersize=80,
                color='crimson',
                label='Sup->Res (跌破反抽)'
            )
        )

    if not np.all(np.isnan(res2sup_marks)):
        ap.append(
            mpf.make_addplot(
                res2sup_marks,
                type='scatter',
                marker='o',
                markersize=80,
                color='seagreen',
                label='Res->Sup (突破回踩)'
            )
        )

    print("正在生成图表...")
    mpf.plot(
        df,
        type='candle',
        volume=True,
        addplot=ap,
        style=my_style,
        title=f'{symbol} - 极性转换原则 (支撑↔阻力)',
        figsize=(14, 8)
    )

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519'  # 贵州茅台

    # 1. 获取数据
    df = get_stock_data(symbol_code, days=365)

    if not df.empty:
        # 2. 识别支撑/阻力位
        # 参数: lookback_swing=5 (左右5根定义高低点), tol_pct=0.3% (聚类容差), min_touch=2 (至少触碰2次)
        s_levels, r_levels = identify_sr_levels(df, lookback_swing=5, tol_pct=0.003, min_touch=2)

        print(f"识别到 {len(s_levels)} 条关键支撑带, {len(r_levels)} 条关键阻力带")

        # 3. 识别极性转换
        # 参数: retest_window=20 (突破后20天内回踩有效), buffer_pct=0.15% (回踩精度)
        df_pattern = identify_polarity_flips(df, s_levels, r_levels, retest_window=20, buffer_pct=0.0015)

        # 统计结果
        n_s2r = df_pattern['SupToRes'].count()
        n_r2s = df_pattern['ResToSup'].count()
        print("=" * 40)
        print(f"支撑转阻力 (Sup->Res): {n_s2r} 次")
        print(f"阻力转支撑 (Res->Sup): {n_r2s} 次")
        print("=" * 40)

        # 4. 绘图
        plot_polarity(df_pattern, symbol_code, s_levels, r_levels)
    else:
        print("未获取到数据。")