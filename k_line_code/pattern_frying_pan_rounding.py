from k_line_code.common.data_fetcher import fetch_stock_data
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
        df_raw = fetch_stock_data(symbol, days=days)
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
# 2. 形态识别函数 (平底锅 & 圆形顶)
# ==========================================
def identify_rounding_patterns(df_in: pd.DataFrame, w: int = 10) -> list:
    """
    识别 平底锅底部 (Frying Pan Bottom) 和 圆形顶部 (Rounding Top)

    逻辑 (沿用原代码):
    - 平底锅: 窗口内收盘价线性回归斜率 > 0 (圆弧上升趋势) 且 最低价逐步抬高
    - 圆形顶: 窗口内收盘价线性回归斜率 < 0 (圆弧下降趋势) 且 最高价逐步降低
    """
    df = df_in.copy()
    signals = []

    # 遍历数据 (从第 w 天开始)
    for i in range(w, len(df)):
        seg = df.iloc[i - w:i]

        # --- 1. 平底锅底部 (Frying Pan Bottom) ---
        # 逻辑：收盘近似圆弧上升（斜率 > 0） & 最低价单调递增
        slope_bottom = np.polyfit(range(w), seg['Close'], 1)[0]
        lows_inc = all(seg['Low'].iloc[j] < seg['Low'].iloc[j + 1] for j in range(w - 1))

        if slope_bottom > 0 and lows_inc:
            signals.append(('Frying Pan Bottom', df.index[i], df['Close'].iloc[i]))

        # --- 2. 圆形顶部 (Rounding Top) ---
        # 逻辑：收盘近似圆弧下降（斜率 < 0） & 最高价单调递减
        slope_top = np.polyfit(range(w), seg['Close'], 1)[0]
        highs_dec = all(seg['High'].iloc[j] > seg['High'].iloc[j + 1] for j in range(w - 1))

        if slope_top < 0 and highs_dec:
            signals.append(('Rounding Top', df.index[i], df['Close'].iloc[i]))

    return signals

# ==========================================
# 3. 可视化绘图函数
# ==========================================
def plot_rounding_patterns(df: pd.DataFrame, signals: list, symbol_code: str):
    """
    绘制 K 线及形态标记
    """
    # 统计结果
    n_sig = len(signals)
    print(f'=== {symbol_code} 统计结果 ===')
    print(f'平底锅/圆形顶共识别出 {n_sig} 次')

    if n_sig == 0:
        print('当前参数下无形态，可调宽 w 或放宽坡度阈值再试')

    # 准备绘图数据
    # 为了图例清晰，我们将 list 拆分为两个独立的 Series 图层
    df['FryingPan_Mark'] = np.nan
    df['RoundingTop_Mark'] = np.nan

    for name, date, price in signals:
        if 'Bottom' in name:
            df.loc[date, 'FryingPan_Mark'] = price
        elif 'Top' in name:
            df.loc[date, 'RoundingTop_Mark'] = price

    # 基础 MA 线
    ap = [mpf.make_addplot(df['Close'].rolling(20).mean(), color='tab:blue', width=1.2)]

    # 添加平底锅底部标记 (绿色圆点)
    if not df['FryingPan_Mark'].isna().all():
        ap.append(mpf.make_addplot(df['FryingPan_Mark'], type='scatter', marker='o',
                                   markersize=100, color='green', label='Frying Pan Bottom'))

    # 添加圆形顶部标记 (红色圆点)
    if not df['RoundingTop_Mark'].isna().all():
        ap.append(mpf.make_addplot(df['RoundingTop_Mark'], type='scatter', marker='o',
                                   markersize=100, color='red', label='Rounding Top'))

    # 绘图
    mpf.plot(df, type='candle', volume=True, addplot=ap,
             title=f'{symbol_code} Frying Pan Bottom & Rounding Top',
             style='yahoo', figsize=(14, 8))

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 配置
    TARGET_SYMBOL = '600519'  # 贵州茅台
    WINDOW_SIZE = 10       # 窗口大小 (对应原代码中的 w)

    # 1. 获取数据
    df_data = get_stock_data(TARGET_SYMBOL, days=365)

    if not df_data.empty:
        # 2. 计算形态
        # 返回的是 list: [('Name', Timestamp, Price), ...]
        signal_list = identify_rounding_patterns(df_data, w=WINDOW_SIZE)

        # 3. 绘图
        plot_rounding_patterns(df_data, signal_list, TARGET_SYMBOL)