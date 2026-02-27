import akshare as ak
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt  # 必须引入 matplotlib
from datetime import datetime, timedelta

# ==========================================
# 1. 数据获取函数
# ==========================================
def get_stock_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    获取最近 N 天的 A 股复权数据并清洗格式
    """
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
        print("未获取到数据，请检查代码或网络。")
        return pd.DataFrame()

    # 整理成英文列 + Date 索引
    df = (df_raw
          .rename(columns={'日期': 'Date',
                           '开盘': 'Open',
                           '最高': 'High',
                           '最低': 'Low',
                           '收盘': 'Close',
                           '成交量': 'Volume'})
          .loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
          .assign(Date=lambda x: pd.to_datetime(x['Date']))
          .set_index('Date')
          .sort_index())

    return df

# ==========================================
# 2. 反击线形态识别函数
# ==========================================
def identify_counterattack(df_in: pd.DataFrame, tol_pct: float = 0.005) -> pd.DataFrame:
    """
    识别 反击线，返回带有信号列的 DataFrame
    """
    df = df_in.copy()

    # 初始化信号列
    df['Bull_Counterattack'] = np.nan
    df['Bear_Counterattack'] = np.nan

    # 遍历数据
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]

        # 1. 看涨反击线 (Bullish)
        if (curr['Close'] > curr['Open'] and
                prev['Close'] < prev['Open'] and
                abs(curr['Close'] - prev['Close']) / prev['Close'] <= tol_pct and
                curr['Open'] < prev['Close']):

            df.iloc[i, df.columns.get_loc('Bull_Counterattack')] = curr['Close']

        # 2. 看跌反击线 (Bearish)
        if (curr['Close'] < curr['Open'] and
                prev['Close'] > prev['Open'] and
                abs(curr['Close'] - prev['Close']) / prev['Close'] <= tol_pct and
                curr['Open'] > prev['Close']):

            df.iloc[i, df.columns.get_loc('Bear_Counterattack')] = curr['Close']

    return df

# ==========================================
# 3. 第一张图：Matplotlib 折线图 (还原原代码逻辑)
# ==========================================
def plot_mpl_line_chart(df: pd.DataFrame, symbol_code: str):
    """
    绘制普通的 Matplotlib 折线图，并用箭头标注形态
    """
    # 提取有信号的点
    bull_signals = df[df['Bull_Counterattack'].notna()]
    bear_signals = df[df['Bear_Counterattack'].notna()]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, df['Close'], label='Close Price', color='grey', alpha=0.6)

    # 标注看涨信号
    for date, row in bull_signals.iterrows():
        price = row['Bull_Counterattack']
        ax.annotate('Bullish C-Attack',
                    xy=(date, price),
                    xytext=(date, price * 0.95),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                    fontsize=9, color='green', ha='center')

    # 标注看跌信号
    for date, row in bear_signals.iterrows():
        price = row['Bear_Counterattack']
        ax.annotate('Bearish C-Attack',
                    xy=(date, price),
                    xytext=(date, price * 1.05),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=9, color='red', ha='center')

    ax.set_title(f'{symbol_code} Counterattack Lines (Matplotlib Line Chart)')
    ax.set_ylabel('Price')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()  # 显示第一张图

# ==========================================
# 4. 第二张图：Mplfinance 蜡烛图
# ==========================================
def plot_mpf_candle_chart(df: pd.DataFrame, symbol_code: str):
    """
    绘制专业的 mplfinance 蜡烛图
    """
    ap = [mpf.make_addplot(df['Close'].rolling(20).mean(), color='tab:blue', width=1.2)]

    rng = (df['High'] - df['Low']).mean()

    # 看涨标记
    if not df['Bull_Counterattack'].isna().all():
        bull_signal = np.where(df['Bull_Counterattack'].notna(), df['Low'] - rng * 0.5, np.nan)
        ap.append(mpf.make_addplot(bull_signal, type='scatter', marker='^',
                                   markersize=100, color='green', label='Bullish C-Attack'))

    # 看跌标记
    if not df['Bear_Counterattack'].isna().all():
        bear_signal = np.where(df['Bear_Counterattack'].notna(), df['High'] + rng * 0.5, np.nan)
        ap.append(mpf.make_addplot(bear_signal, type='scatter', marker='v',
                                   markersize=100, color='red', label='Bearish C-Attack'))

    mpf.plot(df, type='candle', volume=True, addplot=ap,
             title=f'{symbol_code} Counterattack Lines (Candlestick)',
             style='yahoo', figsize=(14, 8)) # 显示第二张图

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    TARGET_SYMBOL = '600519'  # 贵州茅台

    # 1. 获取数据
    df_data = get_stock_data(TARGET_SYMBOL, days=365)

    if not df_data.empty:
        # 2. 计算形态
        df_patterns = identify_counterattack(df_data, tol_pct=0.005)

        # 3. 统计输出
        n_bull = df_patterns['Bull_Counterattack'].count()
        n_bear = df_patterns['Bear_Counterattack'].count()
        print(f'=== {TARGET_SYMBOL} 统计结果 ===')
        print(f'看涨反击线：{n_bull} 次')
        print(f'看跌反击线：{n_bear} 次')

        if n_bull + n_bear > 0:
            # 4. 绘图 (这里会依次调用两个绘图函数)
            print("正在生成第一张图 (Matplotlib Line Chart)...")
            plot_mpl_line_chart(df_patterns, TARGET_SYMBOL)

            print("正在生成第二张图 (Mplfinance Candlestick)...")
            plot_mpf_candle_chart(df_patterns, TARGET_SYMBOL)
        else:
            print("无形态检出，跳过绘图。")