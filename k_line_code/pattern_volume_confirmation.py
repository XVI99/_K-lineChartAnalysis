import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from k_line_code.common.data_fetcher import fetch_stock_data

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

    # fetch_stock_data 已经返回小写列并设索引，这里统一为大写列名以兼容后续逻辑
    df = df_raw.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})

    return df

# ==========================================
# 2. 形态与成交量识别
# ==========================================
def identify_volume_patterns(df_in, vol_ma_period=20, vol_threshold=1.5):
    df = df_in.copy()

    # --- A. 基础字段 ---
    # 防止 Range 为 0
    df['Range'] = (df['High'] - df['Low']).replace(0, 1e-6)
    df['Body']  = (df['Close'] - df['Open']).abs()

    upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])
    lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']

    prev_open  = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)
    prev_high  = df['High'].shift(1)
    prev_low   = df['Low'].shift(1)

    # --- B. 相对交易量指标 ---
    # 计算过去 N 日的平均成交量
    df['Vol_MA'] = df['Volume'].rolling(vol_ma_period).mean()
    # 量比：当前量 / 均量
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA']

    # 定义“放量”：量比超过阈值 (如 1.5 倍)
    df['Is_High_Vol'] = df['Vol_Ratio'] >= vol_threshold

    # --- C. 蜡烛形态识别 ---

    # 1. 锤子线 (Hammer)
    hammer = (
            (df['Close'] >= df['Open']) &
            (lower_shadow >= 2 * df['Body']) &
            (upper_shadow <= df['Body'])
    )

    # 2. 十字线 (Doji)
    doji = (df['Body'] <= 0.1 * df['Range'])

    # 3. 看涨吞没 (Bullish Engulfing)
    bull_engulf = (
            (prev_close < prev_open) &
            (df['Close'] > df['Open']) &
            (df['Open'] <= prev_close) &
            (df['Close'] >= prev_open)
    )

    # 4. 看跌吞没 (Bearish Engulfing)
    bear_engulf = (
            (prev_close > prev_open) &
            (df['Close'] < df['Open']) &
            (df['Open'] >= prev_close) &
            (df['Close'] <= prev_open)
    )

    # 5. 缺口 (Gaps)
    gap_up   = (df['Low'] > prev_high)
    gap_down = (df['High'] < prev_low)

    # --- D. 量价共振 (形态 + 放量) ---

    # 单根K线形态：要求当日放量
    df['Hammer_Vol_OK'] = hammer & df['Is_High_Vol']
    df['Doji_Vol_OK']   = doji & df['Is_High_Vol']
    df['GapUp_Vol_OK']  = gap_up & df['Is_High_Vol']
    df['GapDown_Vol_OK']= gap_down & df['Is_High_Vol']

    # 双K线形态：吞没形态通常要求第二根K线(吞没线)成交量放大，且大于前一日
    # 逻辑：形态成立 AND 明显放量 AND 大于昨日量
    df['BullEngulf_Vol_OK'] = bull_engulf & df['Is_High_Vol'] & (df['Volume'] > df['Volume'].shift(1))
    df['BearEngulf_Vol_OK'] = bear_engulf & df['Is_High_Vol'] & (df['Volume'] > df['Volume'].shift(1))

    return df

# ==========================================
# 3. 可视化绘图函数
# ==========================================
def plot_volume_strategy(df, symbol):
    ap = []

    # 1. 均量线 (画在成交量面板，即 panel=1)
    # 注意：mplfinance 默认 volume 在 panel 1 (如果 volume=True)
    # 我们这里手动添加 MA 线到 volume panel
    # ap.append(mpf.make_addplot(df['Vol_MA'], panel=1, color='blue', width=0.8))

    # 2. 标记形态

    # 放量锤子 (绿色三角)
    hammer_marks = np.where(df['Hammer_Vol_OK'], df['Low'] * 0.99, np.nan)
    if not np.all(np.isnan(hammer_marks)):
        ap.append(mpf.make_addplot(hammer_marks, type='scatter', marker='^',
                                   color='seagreen', markersize=100, label='Vol Hammer'))

    # 放量十字 (金色菱形)
    doji_marks = np.where(df['Doji_Vol_OK'], df['High'] * 1.01, np.nan)
    if not np.all(np.isnan(doji_marks)):
        ap.append(mpf.make_addplot(doji_marks, type='scatter', marker='D',
                                   color='gold', markersize=60, label='Vol Doji'))

    # 放量看涨吞没 (蓝色圆点)
    bull_marks = np.where(df['BullEngulf_Vol_OK'], df['Low'] * 0.98, np.nan)
    if not np.all(np.isnan(bull_marks)):
        ap.append(mpf.make_addplot(bull_marks, type='scatter', marker='o',
                                   color='dodgerblue', markersize=80, label='Vol Bull Engulf'))

    # 放量看跌吞没 (红色圆点)
    bear_marks = np.where(df['BearEngulf_Vol_OK'], df['High'] * 1.02, np.nan)
    if not np.all(np.isnan(bear_marks)):
        ap.append(mpf.make_addplot(bear_marks, type='scatter', marker='o',
                                   color='crimson', markersize=80, label='Vol Bear Engulf'))

    # 放量缺口 (Gap) - 使用水平线标记
    # 为了避免图表过于混乱，我们只画水平线，不加图例
    # 向上缺口 (绿色线画在缺口下沿，即前高) -> 支撑
    # 向下缺口 (红色线画在缺口上沿，即前低) -> 阻力

    # 获取索引
    gap_up_indices = df[df['GapUp_Vol_OK']].index
    gap_down_indices = df[df['GapDown_Vol_OK']].index

    # 由于 make_addplot 不支持动态画线列表，我们构造全长的 Series
    # 这里为了演示清晰，仅用 scatter 标记缺口位置，不用线条
    gap_up_marks = np.where(df['GapUp_Vol_OK'], df['Low'], np.nan)
    gap_down_marks = np.where(df['GapDown_Vol_OK'], df['High'], np.nan)

    if not np.all(np.isnan(gap_up_marks)):
        ap.append(mpf.make_addplot(gap_up_marks, type='scatter', marker='_',
                                   color='lime', markersize=100, label='Vol Gap Up'))

    if not np.all(np.isnan(gap_down_marks)):
        ap.append(mpf.make_addplot(gap_down_marks, type='scatter', marker='_',
                                   color='magenta', markersize=100, label='Vol Gap Down'))

    print("正在生成图表...")
    mpf.plot(
        df,
        type='candle',
        volume=True,
        addplot=ap,
        style=my_style,
        title=f'{symbol} - 关键形态 + 成交量验证 (Vol Ratio > 1.5)',
        figsize=(14, 9)
    )

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519'  # 贵州茅台

    # 1. 获取数据
    df = get_stock_data(symbol_code, days=365)

    if not df.empty:
        # 2. 识别形态与成交量
        # 设定均量周期20天，放量阈值1.5倍
        df = identify_volume_patterns(df, vol_ma_period=20, vol_threshold=1.5)

        # 3. 统计结果
        print("=" * 40)
        print(f"放量锤子线: {int(df['Hammer_Vol_OK'].sum())}")
        print(f"放量十字线: {int(df['Doji_Vol_OK'].sum())}")
        print(f"放量看涨吞没: {int(df['BullEngulf_Vol_OK'].sum())}")
        print(f"放量看跌吞没: {int(df['BearEngulf_Vol_OK'].sum())}")
        print(f"放量向上缺口: {int(df['GapUp_Vol_OK'].sum())}")
        print(f"放量向下缺口: {int(df['GapDown_Vol_OK'].sum())}")
        print("=" * 40)

        # 打印最近的放量吞没形态详情
        print("\n最近的放量看涨吞没:")
        print(df[df['BullEngulf_Vol_OK']][['Close', 'Volume', 'Vol_Ratio']].tail())

        # 4. 绘图
        plot_volume_strategy(df, symbol_code)
    else:
        print("未获取到数据。")