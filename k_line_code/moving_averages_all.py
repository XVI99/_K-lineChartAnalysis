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
# 2. 均线计算函数 (SMA / WMA / EMA)
# ==========================================
def calculate_moving_averages(df_in, periods=[5, 9, 30, 50, 200]):
    df = df_in.copy()
    close = df['Close']

    # --- A. 简单移动平均线 (SMA) ---
    def sma(series, n):
        return series.rolling(n).mean()

    # --- B. 加权移动平均线 (WMA) ---
    # 权重为 1, 2, ..., n
    def wma(series, n):
        weights = np.arange(1, n+1)
        return series.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    # --- C. 指数移动平均线 (EMA) ---
    # pandas ewm span=n 对应 alpha = 2/(n+1)
    def ema(series, n):
        return series.ewm(span=n, adjust=False).mean()

    for n in periods:
        df[f'SMA_{n}'] = sma(close, n)
        df[f'WMA_{n}'] = wma(close, n)
        df[f'EMA_{n}'] = ema(close, n)

    return df

# ==========================================
# 3. 衍生指标计算 (包络线 & 交叉)
# ==========================================
def calculate_derived_indicators(df_in, base_n=20, env_pct=0.03, short_n=5, long_n=30):
    df = df_in.copy()

    # 1. 均线包络线 (Envelopes)
    # 基于 base_n 周期 EMA 上下平移
    base_ma_col = f'EMA_{base_n}'
    if base_ma_col not in df.columns:
        # 如果前面没算，这里补算一下
        df[base_ma_col] = df['Close'].ewm(span=base_n, adjust=False).mean()

    df['Env_Upper'] = df[base_ma_col] * (1 + env_pct)
    df['Env_Lower'] = df[base_ma_col] * (1 - env_pct)

    # 2. 双均线交叉 (Golden / Death Cross)
    # 使用 EMA 作为交叉源
    short_ma = df[f'EMA_{short_n}']
    long_ma  = df[f'EMA_{long_n}']

    # 黄金交叉：短期上穿长期
    golden_cross = (short_ma.shift(1) < long_ma.shift(1)) & (short_ma > long_ma)
    # 死亡交叉：短期下穿长期
    death_cross  = (short_ma.shift(1) > long_ma.shift(1)) & (short_ma < long_ma)

    df['GoldenCross'] = golden_cross
    df['DeathCross']  = death_cross

    return df

# ==========================================
# 4. 可视化绘图函数
# ==========================================
def plot_ma_analysis(df, symbol, periods):
    ap = []

    # 1. 画 EMA 均线组
    colors_ema = {
        5: 'orange',
        9: 'deepskyblue',
        30: 'violet',
        50: 'limegreen',
        200: 'red'
    }

    for n in periods:
        if f'EMA_{n}' in df.columns and not df[f'EMA_{n}'].isna().all():
            ap.append(
                mpf.make_addplot(
                    df[f'EMA_{n}'],
                    color=colors_ema.get(n, 'blue'),
                    width=1.0
                )
            )

    # 2. 画包络线 (灰色虚线)
    if 'Env_Upper' in df.columns:
        ap.append(mpf.make_addplot(df['Env_Upper'], color='gray', linestyle='--', width=0.8))
        ap.append(mpf.make_addplot(df['Env_Lower'], color='gray', linestyle='--', width=0.8))

    # 3. 标记交叉点
    rng = (df['High'] - df['Low']).mean()

    gold_marks = np.where(df['GoldenCross'], df['Low'] - rng * 0.5, np.nan)
    dead_marks = np.where(df['DeathCross'],  df['High'] + rng * 0.5, np.nan)

    if not np.all(np.isnan(gold_marks)):
        ap.append(
            mpf.make_addplot(
                gold_marks, type='scatter', marker='^',
                color='seagreen', markersize=100, label='Golden Cross'
            )
        )

    if not np.all(np.isnan(dead_marks)):
        ap.append(
            mpf.make_addplot(
                dead_marks, type='scatter', marker='v',
                color='crimson', markersize=100, label='Death Cross'
            )
        )

    print("正在生成图表...")
    mpf.plot(
        df,
        type='candle',
        volume=True,
        addplot=ap,
        style=my_style,
        title=f'{symbol} - EMA均线组 + 包络线 + 金叉死叉',
        figsize=(14, 8)
    )

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519'  # 贵州茅台

    # 1. 获取数据 (取长一点以便计算长周期均线)
    df = get_stock_data(symbol_code, days=500)

    if not df.empty:
        # 2. 计算三种均线
        ma_periods = [5, 9, 30, 50, 200]
        df = calculate_moving_averages(df, periods=ma_periods)

        # 打印最近 5 天的 5日均线对比
        print("\n最近 5 天的 MA_5 对比 (SMA vs WMA vs EMA)：")
        print(df[['Close', 'SMA_5', 'WMA_5', 'EMA_5']].tail(5))

        # 3. 计算衍生指标
        # 基础包络线周期: 20, 宽度: 3%
        # 交叉信号: EMA_5 vs EMA_30
        df = calculate_derived_indicators(df, base_n=20, env_pct=0.03, short_n=5, long_n=30)

        n_gold = int(df['GoldenCross'].sum())
        n_dead = int(df['DeathCross'].sum())
        print("=" * 40)
        print(f"EMA(5) 上穿 EMA(30) 黄金交叉: {n_gold} 次")
        print(f"EMA(5) 下穿 EMA(30) 死亡交叉: {n_dead} 次")
        print("=" * 40)

        # 4. 绘图 (仅截取最近一年展示)
        plot_start_date = df.index[-1] - timedelta(days=365)
        df_plot = df.loc[plot_start_date:]

        plot_ma_analysis(df_plot, symbol_code, ma_periods)
    else:
        print("未获取到数据。")