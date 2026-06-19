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
# 2. 计算 MA Oscillator 与 阈值
# ==========================================
def calculate_ma_oscillator(df_in, short_n=12, long_n=26, signal_n=9):
    df = df_in.copy()
    close = df['Close']

    def ema(series, n):
        return series.ewm(span=n, adjust=False).mean()

    # 1. 计算快慢 EMA
    df[f'EMA_{short_n}'] = ema(close, short_n)
    df[f'EMA_{long_n}']  = ema(close, long_n)

    # 2. 计算摆动指数 (Oscillator) = 快线 - 慢线
    # 这本质上就是 MACD 中的 DIFF 线（如果不算 DEA 的话）
    df['MA_Osc'] = df[f'EMA_{short_n}'] - df[f'EMA_{long_n}']

    # 3. 计算信号线 (Signal)
    df['MA_Osc_Sig'] = df['MA_Osc'].ewm(span=signal_n, adjust=False).mean()

    # 4. 计算动态超买/超卖阈值
    # 基于过去 100 天的标准差来设定阈值 (例如 1.5 倍标准差)
    osc_mean = df['MA_Osc'].rolling(100).mean()
    osc_std  = df['MA_Osc'].rolling(100).std()
    k = 1.5

    df['Osc_Upper'] = osc_mean + k * osc_std
    df['Osc_Lower'] = osc_mean - k * osc_std

    df['Osc_OverBought'] = df['MA_Osc'] > df['Osc_Upper']
    df['Osc_OverSold']   = df['MA_Osc'] < df['Osc_Lower']

    return df

# ==========================================
# 3. 形态与背离识别
# ==========================================
def identify_osc_patterns(df_in):
    df = df_in.copy()

    # --- A. 蜡烛形态 ---
    body = (df['Close'] - df['Open']).abs()
    # 防止除零
    rng  = (df['High'] - df['Low']).replace(0, 1e-6)
    upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])
    lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']

    prev_open  = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)

    # 1. 看跌吞没 (Bearish Engulfing)
    bear_engulf = (
            (prev_close > prev_open) &
            (df['Close'] < df['Open']) &
            (df['Open'] >= prev_close) &
            (df['Close'] <= prev_open)
    )

    # 2. 看涨吞没 (Bullish Engulfing)
    bull_engulf = (
            (prev_close < prev_open) &
            (df['Close'] > df['Open']) &
            (df['Open'] <= prev_close) &
            (df['Close'] >= prev_open)
    )

    # 3. 锤子线 (Hammer)
    hammer = (
            (df['Close'] >= df['Open']) &
            (lower_shadow >= 2 * body) &
            (upper_shadow <= body)
    )

    # 4. 上吊线 (Hanging Man) - 形态与锤子一样，但需出现在高位(结合背离判断)
    hanging_man = (
            (df['Close'] <= df['Open']) &
            (lower_shadow >= 2 * body) &
            (upper_shadow <= body)
    )

    df['BearEngulf'] = bear_engulf
    df['BullEngulf'] = bull_engulf
    df['Hammer']     = hammer
    df['Hanging']    = hanging_man

    # --- B. 背离识别 (Divergence) ---
    lookback_swing = 5
    is_swing_high = (df['High'] == df['High'].rolling(lookback_swing*2+1, center=True).max())
    is_swing_low  = (df['Low']  == df['Low'].rolling(lookback_swing*2+1, center=True).min())

    swing_highs = df[is_swing_high][['High', 'MA_Osc']]
    swing_lows  = df[is_swing_low][['Low', 'MA_Osc']]

    df['Osc_TopDiv'] = False
    df['Osc_BotDiv'] = False

    # 顶背离: 价格创新高，Osc 未新高
    prev_idx = None
    for idx, row in swing_highs.iterrows():
        if prev_idx is not None:
            prev_p, prev_o = swing_highs.loc[prev_idx, 'High'], swing_highs.loc[prev_idx, 'MA_Osc']
            cur_p,  cur_o  = row['High'], row['MA_Osc']
            if (cur_p > prev_p) and (cur_o <= prev_o):
                df.at[idx, 'Osc_TopDiv'] = True
        prev_idx = idx

    # 底背离: 价格创新低，Osc 未新低
    prev_idx = None
    for idx, row in swing_lows.iterrows():
        if prev_idx is not None:
            prev_p, prev_o = swing_lows.loc[prev_idx, 'Low'], swing_lows.loc[prev_idx, 'MA_Osc']
            cur_p,  cur_o  = row['Low'], row['MA_Osc']
            if (cur_p < prev_p) and (cur_o >= prev_o):
                df.at[idx, 'Osc_BotDiv'] = True
        prev_idx = idx

    # --- C. 共振信号 ---
    # 顶部: 超买 + 顶背离 + 看跌形态
    df['TopSignal'] = (
            df['Osc_OverBought'] &
            df['Osc_TopDiv'] &
            (df['BearEngulf'] | df['Hanging'])
    )

    # 底部: 超卖 + 底背离 + 看涨形态
    df['BottomSignal'] = (
            df['Osc_OverSold'] &
            df['Osc_BotDiv'] &
            (df['BullEngulf'] | df['Hammer'])
    )

    return df

# ==========================================
# 4. 可视化绘图函数
# ==========================================
def plot_osc_strategy(df, symbol, short_n=12, long_n=26):
    ap = []

    # 1. 主图 EMA
    ap.append(mpf.make_addplot(df[f'EMA_{short_n}'], color='orange', width=1.0))
    ap.append(mpf.make_addplot(df[f'EMA_{long_n}'], color='deepskyblue', width=1.0))

    # 2. 主图信号标记
    top_marks = np.where(df['TopSignal'], df['High'] * 1.02, np.nan)
    bot_marks = np.where(df['BottomSignal'], df['Low'] * 0.98, np.nan)

    if not np.all(np.isnan(top_marks)):
        ap.append(mpf.make_addplot(top_marks, type='scatter', marker='v',
                                   color='crimson', markersize=120, label='Top Signal'))

    if not np.all(np.isnan(bot_marks)):
        ap.append(mpf.make_addplot(bot_marks, type='scatter', marker='^',
                                   color='seagreen', markersize=120, label='Bottom Signal'))

    # 3. 副图 Osc 指标 (Panel 1)
    ap.append(mpf.make_addplot(df['MA_Osc'], panel=1, color='purple', width=1.0, ylabel='Osc'))
    ap.append(mpf.make_addplot(df['MA_Osc_Sig'], panel=1, color='gray', width=0.8))

    # 4. 副图 阈值通道
    ap.append(mpf.make_addplot(df['Osc_Upper'], panel=1, color='red', linestyle='--', width=0.8))
    ap.append(mpf.make_addplot(df['Osc_Lower'], panel=1, color='green', linestyle='--', width=0.8))

    print("正在生成图表...")
    mpf.plot(
        df,
        type='candle',
        volume=True,
        addplot=ap,
        style=my_style,
        title=f'{symbol} - MA Oscillator + 背离 + 蜡烛形态',
        figsize=(14, 9),
        panel_ratios=(3, 1)
    )

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519'  # 贵州茅台

    # 1. 获取数据 (取长一点以计算 rolling 统计量)
    df = get_stock_data(symbol_code, days=500)

    if not df.empty:
        # 2. 计算指标
        df = calculate_ma_oscillator(df)

        # 3. 识别形态与信号
        df = identify_osc_patterns(df)

        # 4. 统计结果
        n_top_div = int(df['Osc_TopDiv'].sum())
        n_bot_div = int(df['Osc_BotDiv'].sum())
        n_top_sig = int(df['TopSignal'].sum())
        n_bot_sig = int(df['BottomSignal'].sum())

        print("=" * 40)
        print(f"顶背离点数: {n_top_div}")
        print(f"底背离点数: {n_bot_div}")
        print(f"顶部共振信号: {n_top_sig}")
        print(f"底部共振信号: {n_bot_sig}")
        print("=" * 40)

        if n_top_sig > 0:
            print("\n顶部共振详情:")
            print(df[df['TopSignal']][['Close', 'MA_Osc']])

        if n_bot_sig > 0:
            print("\n底部共振详情:")
            print(df[df['BottomSignal']][['Close', 'MA_Osc']])

        # 5. 绘图 (截取最近一年)
        plot_start = df.index[-1] - timedelta(days=365)
        plot_osc_strategy(df.loc[plot_start:], symbol_code)
    else:
        print("未获取到数据。")