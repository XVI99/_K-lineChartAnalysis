from k_line_code.common.data_fetcher import fetch_stock_data
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ==========================================
# 0. 字体配置 (修复中文显示乱码)
# ==========================================
# 设置 matplotlib 全局字体 (用于第二张图)
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 默认黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 设置 mplfinance 自定义样式 (用于第一张图)
# 基于 yahoo 风格，但强制使用 SimHei 字体
my_style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.family': 'SimHei'})

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
# 2. 窗口识别逻辑 (保持原逻辑)
# ==========================================
def identify_window(df):
    """
    识别跳空窗口
    返回: list of (direction, index, prev_level, curr_level, gap_size)
    """
    windows = []

    for i in range(1, len(df)):
        prev_candle = df.iloc[i - 1]
        curr_candle = df.iloc[i]

        # 向上跳空窗口 (Previous High < Current Low)
        if prev_candle['High'] < curr_candle['Low']:
            gap_size = curr_candle['Low'] - prev_candle['High']
            windows.append(('up', i, prev_candle['High'], curr_candle['Low'], gap_size))

        # 向下跳空窗口 (Previous Low > Current High)
        elif prev_candle['Low'] > curr_candle['High']:
            gap_size = prev_candle['Low'] - curr_candle['High']
            windows.append(('down', i, prev_candle['Low'], curr_candle['High'], gap_size))

    return windows

# ==========================================
# 3. 可视化 - Mplfinance 版本 (修复字体)
# ==========================================
def plot_candlestick_with_windows(df, windows, symbol_code):
    """
    使用 mplfinance 绘制蜡烛图并标记窗口
    """
    # 准备标记数据
    gap_data = pd.Series(index=df.index, dtype=float)
    gap_colors = pd.Series(index=df.index, dtype=object)

    for direction, index, prev_level, curr_level, gap_size in windows:
        date_index = df.index[index]
        gap_data[date_index] = (prev_level + curr_level) / 2
        gap_colors[date_index] = 'green' if direction == 'up' else 'red'

    ap = []
    if not gap_data.isna().all():
        ap.append(mpf.make_addplot(gap_data, type='scatter',
                                   marker='o', markersize=80,
                                   color=gap_colors.fillna('none').tolist(),
                                   alpha=0.7))

    print("正在生成 Mplfinance 图表...")
    # 注意：这里 style 使用了上面定义的 my_style
    mpf.plot(df, type='candle', volume=True, addplot=ap,
             title=f'{symbol_code} 价格图表 - 跳空窗口识别',
             style=my_style, figsize=(14, 8),
             volume_panel=1, panel_ratios=(4, 1))

# ==========================================
# 4. 可视化 - Matplotlib 版本 (修复字体)
# ==========================================
def plot_simple_candlestick(df, windows, symbol_code):
    """
    使用matplotlib直接绘制蜡烛图和窗口标记
    """
    print("正在生成 Matplotlib 图表...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # 绘制价格走势
    ax1.plot(df.index, df['Close'], label='收盘价', color='black', linewidth=1)

    # 标记窗口位置
    if windows:
        first_up = True
        first_down = True
        for direction, index, prev_level, curr_level, gap_size in windows:
            date = df.index[index]
            if direction == 'up':
                label = '向上窗口' if first_up else ""
                ax1.scatter(date, (prev_level + curr_level) / 2,
                            color='green', s=100, marker='^', alpha=0.7,
                            label=label)
                ax1.axhspan(prev_level, curr_level, alpha=0.2, color='green')
                first_up = False
            else:
                label = '向下窗口' if first_down else ""
                ax1.scatter(date, (prev_level + curr_level) / 2,
                            color='red', s=100, marker='v', alpha=0.7,
                            label=label)
                ax1.axhspan(curr_level, prev_level, alpha=0.2, color='red')
                first_down = False

    ax1.set_title(f'{symbol_code} - 价格走势与跳空窗口识别', fontsize=14)
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制成交量
    ax2.bar(df.index, df['Volume'], color='blue', alpha=0.6)
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    TARGET_SYMBOL = '600519'  # 贵州茅台

    # 1. 获取数据
    df = get_stock_data(TARGET_SYMBOL, days=365)

    if not df.empty:
        # 2. 识别窗口
        windows = identify_window(df)

        # 3. 结果统计
        n_up = len([w for w in windows if w[0] == 'up'])
        n_down = len([w for w in windows if w[0] == 'down'])
        print(f'\n向上跳空窗口：{n_up} 次')
        print(f'向下跳空窗口：{n_down} 次')

        # 4. 详细窗口信息输出
        if windows:
            print("\n详细的窗口信息：")
            print("=" * 80)
            for i, (direction, index, prev_level, curr_level, gap_size) in enumerate(windows, 1):
                date = df.index[index]
                gap_pct = (gap_size / prev_level) * 100
                print(f"窗口 {i}: {date.strftime('%Y-%m-%d')} | "
                      f"类型: {'↑向上' if direction == 'up' else '↓向下'} | "
                      f"缺口: {gap_size:.2f} ({gap_pct:.2f}%) | "
                      f"范围: {prev_level:.2f} → {curr_level:.2f}")

            # 5. 绘图
            plot_candlestick_with_windows(df, windows, TARGET_SYMBOL)
            plot_simple_candlestick(df, windows, TARGET_SYMBOL)
        else:
            print('当前数据中未识别到跳空窗口')