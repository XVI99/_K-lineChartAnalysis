from k_line_code.common.data_fetcher import fetch_stock_data
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
        df_raw = fetch_stock_data(symbol, days=days)
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
# 2. RSI 计算函数 (Wilder 平滑)
# ==========================================
def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)

    gain_series = pd.Series(gain, index=series.index)
    loss_series = pd.Series(loss, index=series.index)

    # 初始平均涨跌
    avg_gain = gain_series.rolling(period).mean()
    avg_loss = loss_series.rolling(period).mean()

    # Wilder 平滑方法
    for i in range(period, len(series)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1]*(period-1) + gain_series.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1]*(period-1) + loss_series.iloc[i]) / period

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi

# ==========================================
# 3. 形态与背离识别
# ==========================================
def identify_rsi_patterns(df_in, rsi_period=14, overbought=70, oversold=30):
    df = df_in.copy()

    # 1. 计算 RSI
    df['RSI'] = calc_rsi(df['Close'], rsi_period)
    df['RSI_OverBought'] = df['RSI'] > overbought
    df['RSI_OverSold']   = df['RSI'] < oversold

    # 2. 识别蜡烛形态
    body = (df['Close'] - df['Open']).abs()
    upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])
    lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']

    prev_open  = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)

    # 乌云盖顶 (Cloud Cover) - 简化版
    cloud_cover = (
            (prev_close > prev_open) &
            (df['Open'] > prev_close) &
            (df['Close'] < (prev_open + prev_close) / 2) &
            (df['Close'] > prev_open)
    )

    # 看涨吞没 (Bullish Engulfing)
    bull_engulf = (
            (prev_close < prev_open) &
            (df['Close'] > df['Open']) &
            (df['Open'] <= prev_close) &
            (df['Close'] >= prev_open)
    )

    # 锤子线 (Hammer)
    hammer = (
            (df['Close'] >= df['Open']) &
            (lower_shadow >= 2 * body) &
            (upper_shadow <= body)
    )

    df['CloudCover'] = cloud_cover
    df['BullEngulf'] = bull_engulf
    df['Hammer']     = hammer

    # 3. 识别背离 (Divergence)
    lookback_swing = 5
    is_swing_high = (df['High'] == df['High'].rolling(lookback_swing*2+1, center=True).max())
    is_swing_low  = (df['Low']  == df['Low'].rolling(lookback_swing*2+1, center=True).min())

    swing_highs = df[is_swing_high][['High', 'RSI']]
    swing_lows  = df[is_swing_low][['Low', 'RSI']]

    df['RSI_TopDiv'] = False
    df['RSI_BotDiv'] = False

    # 顶背离: 价格新高, RSI 未新高
    prev_idx = None
    for idx, row in swing_highs.iterrows():
        if prev_idx is not None:
            prev_price, prev_rsi = swing_highs.loc[prev_idx, 'High'], swing_highs.loc[prev_idx, 'RSI']
            cur_price,  cur_rsi  = row['High'], row['RSI']
            if (cur_price > prev_price) and (cur_rsi <= prev_rsi):
                df.at[idx, 'RSI_TopDiv'] = True
        prev_idx = idx

    # 底背离: 价格新低, RSI 未新低
    prev_idx = None
    for idx, row in swing_lows.iterrows():
        if prev_idx is not None:
            prev_price, prev_rsi = swing_lows.loc[prev_idx, 'Low'], swing_lows.loc[prev_idx, 'RSI']
            cur_price,  cur_rsi  = row['Low'], row['RSI']
            if (cur_price < prev_price) and (cur_rsi >= prev_rsi):
                df.at[idx, 'RSI_BotDiv'] = True
        prev_idx = idx

    # 4. 共振信号
    # 顶部共振: 超买 + 顶背离 + 看跌形态 (这里仅用乌云盖顶示例，可自行添加更多)
    df['TopSignal'] = df['RSI_OverBought'] & df['RSI_TopDiv'] & df['CloudCover']

    # 底部共振: 超卖 + 底背离 + 看涨形态 (锤子 或 吞没)
    df['BottomSignal'] = df['RSI_OverSold'] & df['RSI_BotDiv'] & (df['Hammer'] | df['BullEngulf'])

    return df

# ==========================================
# 4. 可视化绘图函数
# ==========================================
def plot_rsi_strategy(df, symbol, overbought=70, oversold=30):
    ap = []

    # 1. 顶部信号 (红色倒三角)
    top_marks = np.where(df['TopSignal'], df['High'] * 1.02, np.nan)
    if not np.all(np.isnan(top_marks)):
        ap.append(mpf.make_addplot(top_marks, type='scatter', marker='v',
                                   color='crimson', markersize=120, label='Top Signal'))

    # 2. 底部信号 (绿色正三角)
    bottom_marks = np.where(df['BottomSignal'], df['Low'] * 0.98, np.nan)
    if not np.all(np.isnan(bottom_marks)):
        ap.append(mpf.make_addplot(bottom_marks, type='scatter', marker='^',
                                   color='seagreen', markersize=120, label='Bottom Signal'))

    # 3. RSI 指标 (在 panel 1)
    ap.append(mpf.make_addplot(df['RSI'], panel=1, color='deepskyblue', width=1.0, ylabel='RSI'))

    # 4. 超买/超卖线
    ap.append(mpf.make_addplot(pd.Series(overbought, index=df.index), panel=1,
                               color='red', linestyle='--', width=0.8))
    ap.append(mpf.make_addplot(pd.Series(oversold, index=df.index), panel=1,
                               color='green', linestyle='--', width=0.8))

    print("正在生成图表...")
    mpf.plot(
        df,
        type='candle',
        volume=True,
        addplot=ap,
        style=my_style,
        title=f'{symbol} - RSI 背离 + 蜡烛共振策略',
        figsize=(14, 9),
        panel_ratios=(3, 1) # 主图和副图高度比 3:1
    )

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519'  # 贵州茅台

    # 1. 获取数据
    df = get_stock_data(symbol_code, days=500)

    if not df.empty:
        # 2. 识别形态与背离
        df = identify_rsi_patterns(df)

        # 3. 打印统计结果
        n_top_div = int(df['RSI_TopDiv'].sum())
        n_bot_div = int(df['RSI_BotDiv'].sum())
        n_top_sig = int(df['TopSignal'].sum())
        n_bot_sig = int(df['BottomSignal'].sum())

        print("=" * 40)
        print(f"顶背离点数: {n_top_div}")
        print(f"底背离点数: {n_bot_div}")
        print(f"顶部共振信号 (超买+背离+形态): {n_top_sig}")
        print(f"底部共振信号 (超卖+背离+形态): {n_bot_sig}")
        print("=" * 40)

        if n_top_sig > 0:
            print("\n顶部共振详情:")
            print(df[df['TopSignal']][['Close', 'RSI']])

        if n_bot_sig > 0:
            print("\n底部共振详情:")
            print(df[df['BottomSignal']][['Close', 'RSI']])

        # 4. 绘图 (截取最近一年)
        plot_start = df.index[-1] - timedelta(days=365)
        plot_rsi_strategy(df.loc[plot_start:], symbol_code)
    else:
        print("未获取到数据。")