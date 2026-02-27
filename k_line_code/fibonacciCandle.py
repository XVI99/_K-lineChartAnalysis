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
# 2. 识别蜡烛图形态
# ==========================================
def identify_candle_patterns(df_in):
    df = df_in.copy()

    # 基础字段
    df['Range'] = (df['High'] - df['Low']).replace(0, 1e-6)
    df['Body']  = (df['Close'] - df['Open']).abs()
    upper_shadow = df["High"] - np.maximum(df["Open"], df["Close"])
    lower_shadow = np.minimum(df["Open"], df["Close"]) - df["Low"]

    prev_close = df['Close'].shift(1)
    prev_open  = df['Open'].shift(1)

    # --- 形态定义 ---
    # 1. 锤子线 (Hammer): 实体在上端，长下影
    hammer = (
            (df['Close'] > df['Open']) &  # 阳锤
            (lower_shadow >= 2 * df['Body']) &
            (upper_shadow <= df['Body'])
    )

    # 2. 流星线 (Shooting Star): 实体在下端，长上影
    shooting_star = (
            (df['Close'] < df['Open']) &  # 阴流星
            (upper_shadow >= 2 * df['Body']) &
            (lower_shadow <= df['Body'])
    )

    # 3. 刺透形态 (Piercing Line): 前阴后阳，阳线插入阴线实体一半以上
    bull_piercing = (
            (prev_close < prev_open) &
            (df['Open'] < prev_close) &
            (df['Close'] > (prev_open + prev_close) / 2) &
            (df['Close'] < prev_open)
    )

    # 4. 看涨吞没 (Bullish Engulfing): 阳包阴
    bull_engulf = (
            (prev_close < prev_open) &
            (df['Close'] > df['Open']) &
            (df['Open'] <= prev_close) &
            (df['Close'] >= prev_open)
    )

    # 5. 看跌吞没 (Bearish Engulfing): 阴包阳
    bear_engulf = (
            (prev_close > prev_open) &
            (df['Close'] < df['Open']) &
            (df['Open'] >= prev_close) &
            (df['Close'] <= prev_open)
    )

    df['Hammer']       = hammer
    df['ShootingStar'] = shooting_star
    df['BullPiercing'] = bull_piercing
    df['BullEngulf']   = bull_engulf
    df['BearEngulf']   = bear_engulf

    return df

# ==========================================
# 3. 计算斐波那契回撤与共振
# ==========================================
def identify_fib_confluence(df, window=120, tol_pct=0.005):
    """
    计算 window 窗口内的斐波那契回撤位，并寻找 K 线共振
    """
    # 1. 确定波段起点和终点
    sub = df.tail(window)
    low_idx  = sub['Low'].idxmin()
    high_idx = sub['High'].idxmax()

    # 确定趋势方向 (Up: 低点在前; Down: 高点在前)
    if low_idx < high_idx:
        trend_dir = 'up'
        start_price = df.at[low_idx, 'Low']
        end_price   = df.at[high_idx, 'High']
        print(f"检测到上升波段: {start_price} -> {end_price}")
    else:
        trend_dir = 'down'
        start_price = df.at[high_idx, 'High']
        end_price   = df.at[low_idx, 'Low']
        print(f"检测到下降波段: {start_price} -> {end_price}")

    price_move = end_price - start_price # 注意: 下降趋势时 diff 为负数

    # 2. 计算关键回撤位 (0.382, 0.5, 0.618)
    # 回撤总是相对于“终点”往回走
    # 上升趋势回撤：End - ratio * (End - Start)
    # 下降趋势回撤：End - ratio * (End - Start)  <-- 因为 diff 为负，所以 End - (-val) = End + val，逻辑通用
    # 但为了逻辑清晰，通常这样写：

    if trend_dir == 'up':
        fib_38 = end_price - 0.382 * (end_price - start_price)
        fib_50 = end_price - 0.500 * (end_price - start_price)
        fib_62 = end_price - 0.618 * (end_price - start_price)
    else:
        # 下降趋势的反弹目标位 (实际上叫回撤位也行)
        # Start(High) -> End(Low), 回撤是往上走
        fib_38 = end_price + 0.382 * (start_price - end_price)
        fib_50 = end_price + 0.500 * (start_price - end_price)
        fib_62 = end_price + 0.618 * (start_price - end_price)

    levels = {'38.2%': fib_38, '50.0%': fib_50, '61.8%': fib_62}

    for k, v in levels.items():
        print(f"{k} 回撤位: {v:.2f}")

    # 3. 寻找共振信号
    df['FibLevel'] = np.nan
    df['FibTag']   = ''

    for name, lvl in levels.items():
        # 价格在回撤位附近
        in_zone = (df['Close'] >= lvl * (1 - tol_pct)) & (df['Close'] <= lvl * (1 + tol_pct))

        if trend_dir == 'up':
            # 上升趋势的回撤是寻找“支撑” -> 看涨信号
            signal = in_zone & (df['Hammer'] | df['BullPiercing'] | df['BullEngulf'])

            # 标记
            df.loc[signal, 'FibLevel'] = lvl
            df.loc[signal, 'FibTag']   = f'{name} Bull'
        else:
            # 下降趋势的回撤是寻找“阻力” -> 看跌信号
            signal = in_zone & (df['ShootingStar'] | df['BearEngulf'])

            df.loc[signal, 'FibLevel'] = lvl
            df.loc[signal, 'FibTag']   = f'{name} Bear'

    return df, levels

# ==========================================
# 4. 可视化绘图函数
# ==========================================
def plot_fib_candle(df, symbol, fib_levels):
    ap = []

    # 1. 画斐波那契水平线
    colors = {'38.2%': 'orange', '50.0%': 'dodgerblue', '61.8%': 'purple'}

    for name, lvl in fib_levels.items():
        ap.append(
            mpf.make_addplot(
                pd.Series(lvl, index=df.index),
                color=colors.get(name, 'gray'),
                linestyle='--',
                width=1.0,
            )
        )

    # 2. 标记共振点
    # 提取有信号的点
    has_signal = df['FibTag'] != ''

    if has_signal.any():
        # 这里的颜色处理稍微复杂一点，因为 mplfinance 的 color list 需要对应每一个点
        # 我们创建两个图层：看涨和看跌

        bull_signals = df[df['FibTag'].str.contains('Bull')]
        bear_signals = df[df['FibTag'].str.contains('Bear')]

        # 看涨信号 (支撑位反转)
        if not bull_signals.empty:
            bull_series = pd.Series(np.nan, index=df.index)
            bull_series.loc[bull_signals.index] = bull_signals['Low'] * 0.99
            ap.append(
                mpf.make_addplot(
                    bull_series,
                    type='scatter',
                    marker='^',
                    markersize=100,
                    color='seagreen',
                    label='Bullish Confluence'
                )
            )

        # 看跌信号 (阻力位反转)
        if not bear_signals.empty:
            bear_series = pd.Series(np.nan, index=df.index)
            bear_series.loc[bear_signals.index] = bear_signals['High'] * 1.01
            ap.append(
                mpf.make_addplot(
                    bear_series,
                    type='scatter',
                    marker='v',
                    markersize=100,
                    color='crimson',
                    label='Bearish Confluence'
                )
            )

    print("正在生成图表...")
    mpf.plot(
        df,
        type='candle',
        volume=True,
        addplot=ap,
        style=my_style,
        title=f'{symbol} - 斐波那契回撤 + 蜡烛图共振',
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
        # 2. 识别蜡烛形态
        df = identify_candle_patterns(df)

        # 3. 计算斐波那契共振
        # window=120 表示最近约半年的大波段
        df, levels = identify_fib_confluence(df, window=120, tol_pct=0.005)

        # 4. 打印命中结果
        hits = df[df['FibTag'] != '']
        print("=" * 40)
        print(f"共发现 {len(hits)} 次共振信号:")
        if not hits.empty:
            print(hits[['Close', 'FibLevel', 'FibTag']].tail(10))
        print("=" * 40)

        # 5. 绘图
        plot_fib_candle(df, symbol_code, levels)
    else:
        print("未获取到数据。")