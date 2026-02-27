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
# 2. MACD 计算函数
# ==========================================
def calculate_macd(df_in, short_n=12, long_n=26, signal_n=9):
    df = df_in.copy()
    close = df['Close']

    # 计算 EMA
    ema_short = close.ewm(span=short_n, adjust=False).mean()
    ema_long  = close.ewm(span=long_n, adjust=False).mean()

    # 计算 DIF (快线)
    df['MACD_DIF'] = ema_short - ema_long

    # 计算 DEA (慢线/信号线)
    df['MACD_DEA'] = df['MACD_DIF'].ewm(span=signal_n, adjust=False).mean()

    # 计算 MACD 柱状图 (Histogram) -> 通常 * 2 以放大显示
    df['MACD_Hist'] = (df['MACD_DIF'] - df['MACD_DEA']) * 2

    return df

# ==========================================
# 3. 形态、交叉与背离识别
# ==========================================
def identify_signals(df_in):
    df = df_in.copy()

    # --- A. MACD 交叉 ---
    dif_prev = df['MACD_DIF'].shift(1)
    dea_prev = df['MACD_DEA'].shift(1)

    # 金叉: DIF 上穿 DEA
    df['GoldCross'] = (dif_prev < dea_prev) & (df['MACD_DIF'] > df['MACD_DEA'])
    # 死叉: DIF 下穿 DEA
    df['DeadCross'] = (dif_prev > dea_prev) & (df['MACD_DIF'] < df['MACD_DEA'])

    # --- B. 蜡烛形态 ---
    body = (df['Close'] - df['Open']).abs()
    prev_open  = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)

    # 1. 乌云盖顶 (Cloud Cover) - 看跌
    # 前阳，后阴，高开，收盘跌破前阳中点
    cloud_cover = (
            (prev_close > prev_open) &
            (df['Open'] > prev_close) &
            (df['Close'] < (prev_open + prev_close) / 2) &
            (df['Close'] > prev_open)
    )

    # 2. 看涨吞没 (Bullish Engulfing) - 看涨
    # 前阴，后阳，阳包阴
    bull_engulf = (
            (prev_close < prev_open) &
            (df['Close'] > df['Open']) &
            (df['Open'] <= prev_close) &
            (df['Close'] >= prev_open)
    )

    df['CloudCover'] = cloud_cover
    df['BullEngulf'] = bull_engulf

    # --- C. MACD 背离 (Divergence) ---
    # 简单逻辑：比较最近两个 Swing High/Low 的价格与 MACD_DIF
    lookback = 5
    # 局部高低点
    is_high = (df['High'] == df['High'].rolling(lookback*2+1, center=True).max())
    is_low  = (df['Low']  == df['Low'].rolling(lookback*2+1, center=True).min())

    swing_highs = df[is_high][['High', 'MACD_DIF']]
    swing_lows  = df[is_low][['Low', 'MACD_DIF']]

    df['TopDiv'] = False
    df['BotDiv'] = False

    # 顶背离：价格创新高，MACD DIF 未创新高
    prev_idx = None
    for idx, row in swing_highs.iterrows():
        if prev_idx is not None:
            p1, m1 = swing_highs.loc[prev_idx, 'High'], swing_highs.loc[prev_idx, 'MACD_DIF']
            p2, m2 = row['High'], row['MACD_DIF']
            if p2 > p1 and m2 < m1:
                df.at[idx, 'TopDiv'] = True
        prev_idx = idx

    # 底背离：价格创新低，MACD DIF 未创新低
    prev_idx = None
    for idx, row in swing_lows.iterrows():
        if prev_idx is not None:
            p1, m1 = swing_lows.loc[prev_idx, 'Low'], swing_lows.loc[prev_idx, 'MACD_DIF']
            p2, m2 = row['Low'], row['MACD_DIF']
            if p2 < p1 and m2 > m1:
                df.at[idx, 'BotDiv'] = True
        prev_idx = idx

    # --- D. 共振信号 ---
    # 顶部共振：死叉 + (乌云盖顶 或 顶背离)
    df['TopSignal'] = df['DeadCross'] & (df['CloudCover'] | df['TopDiv'])

    # 底部共振：金叉 + (看涨吞没 或 底背离)
    df['BottomSignal'] = df['GoldCross'] & (df['BullEngulf'] | df['BotDiv'])

    return df

# ==========================================
# 4. 可视化绘图函数
# ==========================================
def plot_macd_strategy(df, symbol):
    ap = []

    # 1. 主图信号标记
    top_marks = np.where(df['TopSignal'], df['High'] * 1.01, np.nan)
    bot_marks = np.where(df['BottomSignal'], df['Low'] * 0.99, np.nan)

    if not np.all(np.isnan(top_marks)):
        ap.append(mpf.make_addplot(top_marks, type='scatter', marker='v',
                                   color='crimson', markersize=120, label='Top Signal'))

    if not np.all(np.isnan(bot_marks)):
        ap.append(mpf.make_addplot(bot_marks, type='scatter', marker='^',
                                   color='seagreen', markersize=120, label='Bottom Signal'))

    # 2. 标记背离点 (辅助)
    div_top = np.where(df['TopDiv'], df['High'] * 1.02, np.nan)
    div_bot = np.where(df['BotDiv'], df['Low'] * 0.98, np.nan)

    if not np.all(np.isnan(div_top)):
        ap.append(mpf.make_addplot(div_top, type='scatter', marker='.', color='orange', markersize=50))
    if not np.all(np.isnan(div_bot)):
        ap.append(mpf.make_addplot(div_bot, type='scatter', marker='.', color='blue', markersize=50))

    # 3. 副图: MACD 线 (Panel 1)
    ap.append(mpf.make_addplot(df['MACD_DIF'], panel=1, color='orange', width=1.0, ylabel='MACD'))
    ap.append(mpf.make_addplot(df['MACD_DEA'], panel=1, color='deepskyblue', width=1.0))

    # 4. 副图: MACD 柱状图 (Panel 1)
    # 根据正负设置颜色
    hist_colors = ['crimson' if v < 0 else 'seagreen' for v in df['MACD_Hist']]
    ap.append(mpf.make_addplot(df['MACD_Hist'], panel=1, type='bar', color=hist_colors, alpha=0.5))

    print("正在生成图表...")
    mpf.plot(
        df,
        type='candle',
        volume=True,
        addplot=ap,
        style=my_style,
        title=f'{symbol} - MACD 背离验证 + 蜡烛共振',
        figsize=(14, 9),
        panel_ratios=(3, 1)
    )

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519'  # 贵州茅台

    # 1. 获取数据
    df = get_stock_data(symbol_code, days=500)

    if not df.empty:
        # 2. 计算 MACD
        df = calculate_macd(df)

        # 3. 识别信号
        df = identify_signals(df)

        # 4. 统计结果
        n_gold = int(df['GoldCross'].sum())
        n_dead = int(df['DeadCross'].sum())
        n_top  = int(df['TopSignal'].sum())
        n_bot  = int(df['BottomSignal'].sum())
        n_top_div = int(df['TopDiv'].sum())
        n_bot_div = int(df['BotDiv'].sum())

        print("=" * 40)
        print(f"MACD 金叉: {n_gold}, 死叉: {n_dead}")
        print(f"顶背离点: {n_top_div}, 底背离点: {n_bot_div}")
        print(f"顶部共振 (死叉+背离/乌云): {n_top}")
        print(f"底部共振 (金叉+背离/吞没): {n_bot}")
        print("=" * 40)

        if n_top > 0:
            print("\n顶部共振详情:")
            print(df[df['TopSignal']][['Close', 'MACD_DIF', 'MACD_Hist']])

        if n_bot > 0:
            print("\n底部共振详情:")
            print(df[df['BottomSignal']][['Close', 'MACD_DIF', 'MACD_Hist']])

        # 5. 绘图 (截取最近一年)
        plot_start = df.index[-1] - timedelta(days=365)
        plot_macd_strategy(df.loc[plot_start:], symbol_code)
    else:
        print("未获取到数据。")