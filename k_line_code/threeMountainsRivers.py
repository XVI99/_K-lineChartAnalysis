import akshare as ak
import pandas as pd
import numpy as np
import mplfinance as mpf
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
# 2. 形态识别函数
# ==========================================
def identify_three_peaks(df_in: pd.DataFrame, ma_len: int = 20) -> pd.DataFrame:
    """
    识别 三山顶、三尊顶、倒三尊底 (基于局部高低点逻辑)
    """
    df = df_in.copy()

    # 1. 基础指标计算 (沿用代码)
    df['Range'] = df['High'] - df['Low']
    df['Body']  = (df['Close'] - df['Open']).abs()
    df['MA']    = df['Close'].rolling(ma_len).mean()

    # 初始化结果列表
    peaks = []
    monks = []
    bottoms = []

    # 2. 循环遍历识别 (沿用原逻辑)
    # 注意：使用 iloc 遍历，range 从 1 到 len-1
    for i in range(1, len(df)-1):
        # --- 三山顶 (Three Peaks Top) ---
        # 逻辑：中间 High 高于左边 High 且 高于右边 High
        if (df['High'].iloc[i] > df['High'].iloc[i-1] and
                df['High'].iloc[i] > df['High'].iloc[i+1]):
            peaks.append((df.index[i], df['High'].iloc[i]))

        # --- 三尊顶 (Three Drunken Monks Top) ---
        # 逻辑：中间 High 高于左右，且右边明显回落 (逻辑上是三山顶的子集或严格版)
        if (df['High'].iloc[i] > df['High'].iloc[i-1] and
                df['High'].iloc[i] > df['High'].iloc[i+1] and
                df['High'].iloc[i+1] < df['High'].iloc[i]):
            monks.append((df.index[i], df['High'].iloc[i]))

        # --- 倒三尊底 (Inverted Three Peaks Bottom) ---
        # 逻辑：中间 Low 低于左边 Low 且 低于右边 Low
        if (df['Low'].iloc[i] < df['Low'].iloc[i-1] and
                df['Low'].iloc[i] < df['Low'].iloc[i+1]):
            bottoms.append((df.index[i], df['Low'].iloc[i]))

    # 3. 将结果映射回 DataFrame 用于绘图
    df['PeakMark']   = np.nan
    df['MonkMark']   = np.nan
    df['BottomMark'] = np.nan

    for date, val in peaks:
        df.loc[date, 'PeakMark'] = val

    for date, val in monks:
        df.loc[date, 'MonkMark'] = val

    for date, val in bottoms:
        df.loc[date, 'BottomMark'] = val

    return df

# ==========================================
# 3. 可视化绘图函数
# ==========================================
def plot_three_peaks(df: pd.DataFrame, symbol_code: str):
    """
    绘制 K 线及三山顶/底标记
    """
    # 统计数量
    n_peaks = df['PeakMark'].count()
    n_monks = df['MonkMark'].count()
    n_bottoms = df['BottomMark'].count()

    print(f'=== {symbol_code} 统计结果 ===')
    print(f'三山顶 (Three Peaks)：{n_peaks} 次')
    print(f'三尊顶 (Three Monks)：{n_monks} 次')
    print(f'倒三尊底 (Inverted Bottom)：{n_bottoms} 次')

    if n_peaks + n_monks + n_bottoms == 0:
        print('当前参数下无形态，可放宽条件再试')

    # 准备绘图
    ap = [mpf.make_addplot(df['MA'], color='tab:blue', width=1.2)]

    # 检查列是否全为空，避免报错
    if not df['PeakMark'].isna().all():
        ap.append(mpf.make_addplot(df['PeakMark'], type='scatter', marker='v',
                                   markersize=80, color='red', label='Three Peaks'))

    if not df['MonkMark'].isna().all():
        # 为了区分，三尊顶用更深的颜色和稍大的标记
        ap.append(mpf.make_addplot(df['MonkMark'], type='scatter', marker='v',
                                   markersize=100, color='darkred', label='Three Monks'))

    if not df['BottomMark'].isna().all():
        ap.append(mpf.make_addplot(df['BottomMark'], type='scatter', marker='^',
                                   markersize=100, color='green', label='Inverted Bottom'))

    # 绘图
    mpf.plot(df, type='candle', volume=True, addplot=ap,
             title=f'{symbol_code} Three Peaks & Inverted Bottom',
             style='yahoo', figsize=(14, 8))

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 配置
    TARGET_SYMBOL = '600519'  # 贵州茅台

    # 1. 获取数据
    df_data = get_stock_data(TARGET_SYMBOL, days=365)

    if not df_data.empty:
        # 2. 计算形态
        df_patterns = identify_three_peaks(df_data, ma_len=20)

        # 3. 绘图
        plot_three_peaks(df_patterns, TARGET_SYMBOL)