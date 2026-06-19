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
# 2. 随机指标计算函数 (%K, %D)
# ==========================================
def calculate_stochastic(df_in, n_k=14, smooth_k=3, smooth_d=3):
    df = df_in.copy()

    # 1. 计算快 %K (Fast %K)
    # Formula: (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    lowest_low  = df['Low'].rolling(n_k).min()
    highest_high = df['High'].rolling(n_k).max()

    # 防止分母为 0
    range_hl = (highest_high - lowest_low).replace(0, 1e-6)

    fast_k = (df['Close'] - lowest_low) / range_hl * 100
    df['%K_fast'] = fast_k

    # 2. 计算慢 %K (Slow %K) -> 通常就是 KDJ 中的 K 线
    df['%K'] = fast_k.rolling(smooth_k).mean()

    # 3. 计算 %D (Slow %D) -> 慢 %K 的移动平均
    df['%D'] = df['%K'].rolling(smooth_d).mean()

    # 丢弃前面的 NaN
    df = df.dropna()
    return df

# ==========================================
# 3. 识别形态与共振信号
# ==========================================
def identify_stochastic_patterns(df_in, overbought=80, oversold=20):
    df = df_in.copy()

    # --- A. 随机指标状态 ---
    df['OverBought'] = df['%D'] >= overbought
    df['OverSold']   = df['%D'] <= oversold

    # 金叉 (Bull Cross): %K 上穿 %D
    # 死叉 (Bear Cross): %K 下穿 %D
    k_prev = df['%K'].shift(1)
    d_prev = df['%D'].shift(1)

    df['BullCross'] = (k_prev < d_prev) & (df['%K'] > df['%D'])
    df['BearCross'] = (k_prev > d_prev) & (df['%K'] < df['%D'])

    # --- B. 蜡烛形态 ---
    body = (df['Close'] - df['Open']).abs()
    rng = (df['High'] - df['Low']).replace(0, 1e-6)
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

    # 2. 锤子线 (Hammer) - 简化版
    hammer = (
            (df['Close'] >= df['Open']) &
            (lower_shadow >= 2 * body) &
            (upper_shadow <= body)
    )

    df['BearEngulf'] = bear_engulf
    df['Hammer']     = hammer

    # --- C. 共振信号 ---
    # 顶部共振: 超买区域 + 死叉 + 看跌吞没
    # 注意: 有时候这三个条件不一定在同一天发生，这里为了演示简化为同一天
    df['TopSignal'] = df['OverBought'] & df['BearCross'] & df['BearEngulf']

    # 底部共振: 超卖区域 + 金叉 + 锤子线
    df['BottomSignal'] = df['OverSold'] & df['BullCross'] & df['Hammer']

    return df

# ==========================================
# 4. 可视化绘图函数
# ==========================================
def plot_stochastic_strategy(df, symbol, overbought=80, oversold=20):
    ap = []

    # 1. 主图信号标记
    top_marks = np.where(df['TopSignal'], df['High'] * 1.02, np.nan)
    bot_marks = np.where(df['BottomSignal'], df['Low'] * 0.98, np.nan)

    if not np.all(np.isnan(top_marks)):
        ap.append(mpf.make_addplot(top_marks, type='scatter', marker='v',
                                   color='crimson', markersize=120, label='Top Signal'))

    if not np.all(np.isnan(bot_marks)):
        ap.append(mpf.make_addplot(bot_marks, type='scatter', marker='^',
                                   color='seagreen', markersize=120, label='Bottom Signal'))

    # 2. 副图: %K 和 %D (Panel 1)
    ap.append(mpf.make_addplot(df['%K'], panel=1, color='orange', width=1.0, ylabel='Stoch'))
    ap.append(mpf.make_addplot(df['%D'], panel=1, color='deepskyblue', width=1.0))

    # 3. 副图: 超买超卖线
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
        title=f'{symbol} - 随机指标(Stoch) + 蜡烛共振策略',
        figsize=(14, 9),
        panel_ratios=(3, 1) # 主副图比例
    )

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519'  # 贵州茅台

    # 1. 获取数据
    df = get_stock_data(symbol_code, days=500)

    if not df.empty:
        # 2. 计算指标
        df = calculate_stochastic(df, n_k=14, smooth_k=3, smooth_d=3)

        # 3. 识别形态与信号
        df = identify_stochastic_patterns(df, overbought=80, oversold=20)

        # 4. 统计结果
        n_bull_cross = int(df['BullCross'].sum())
        n_bear_cross = int(df['BearCross'].sum())
        n_top_sig = int(df['TopSignal'].sum())
        n_bot_sig = int(df['BottomSignal'].sum())

        print("=" * 40)
        print(f"随机指标金叉次数: {n_bull_cross}")
        print(f"随机指标死叉次数: {n_bear_cross}")
        print(f"顶部共振信号: {n_top_sig}")
        print(f"底部共振信号: {n_bot_sig}")
        print("=" * 40)

        if n_top_sig > 0:
            print("\n顶部共振详情:")
            print(df[df['TopSignal']][['Close', '%K', '%D']])

        if n_bot_sig > 0:
            print("\n底部共振详情:")
            print(df[df['BottomSignal']][['Close', '%K', '%D']])

        # 5. 绘图 (截取最近一年)
        plot_start = df.index[-1] - timedelta(days=365)
        plot_stochastic_strategy(df.loc[plot_start:], symbol_code)
    else:
        print("未获取到数据。")