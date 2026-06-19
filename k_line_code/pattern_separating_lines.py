from k_line_code.common.data_fetcher import fetch_stock_data
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ==========================================
# 0. 字体配置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
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
# 2. 形态识别逻辑
# ==========================================
def identify_separation_line(df, price_tolerance=0.005):
    """
    识别分手线形态
    看涨: 阴线 -> 阳线, 开盘价相同
    看跌: 阳线 -> 阴线, 开盘价相同
    """
    patterns = []
    for i in range(1, len(df)):
        first = df.iloc[i-1]  # T-1
        second = df.iloc[i]   # T

        # 计算开盘价差异百分比
        open_diff_pct = abs(first['Open'] - second['Open']) / first['Open']

        # 1. 看涨分手线 (Bullish)
        # 前阴(Close < Open) + 后阳(Close > Open) + 开盘价接近
        if (first['Close'] < first['Open']) and \
                (second['Close'] > second['Open']) and \
                (open_diff_pct <= price_tolerance):

            strength = min(
                (first['Open'] - first['Close']) / first['Open'],   # 阴线实体
                (second['Close'] - second['Open']) / second['Open'] # 阳线实体
            )
            patterns.append({
                'type': '看涨分手线',
                'index': i,
                'date': df.index[i],
                'price': second['Close'], # 记录收盘价用于绘图
                'first_open': first['Open'],
                'second_open': second['Open'],
                'strength': strength,
                'diff_pct': open_diff_pct
            })

        # 2. 看跌分手线 (Bearish)
        # 前阳(Close > Open) + 后阴(Close < Open) + 开盘价接近
        elif (first['Close'] > first['Open']) and \
                (second['Close'] < second['Open']) and \
                (open_diff_pct <= price_tolerance):

            strength = min(
                (first['Close'] - first['Open']) / first['Open'],   # 阳线实体
                (second['Open'] - second['Close']) / second['Open'] # 阴线实体
            )
            patterns.append({
                'type': '看跌分手线',
                'index': i,
                'date': df.index[i],
                'price': second['Close'],
                'first_open': first['Open'],
                'second_open': second['Open'],
                'strength': strength,
                'diff_pct': open_diff_pct
            })

    return patterns

# ==========================================
# 3. 可视化 - Mplfinance
# ==========================================
def plot_separation_mpf(df, patterns):
    if not patterns:
        return

    # 构造标记序列
    marker_series = pd.Series(np.nan, index=df.index)
    colors = []

    # 这里的逻辑稍微复杂，因为 mpf 对 list 颜色的支持需要一一对应非 NaN 值
    # 为了简单，我们只绘制位置，颜色统一处理，或者分层处理
    # 更好的方式：分两个图层
    bull_series = pd.Series(np.nan, index=df.index)
    bear_series = pd.Series(np.nan, index=df.index)

    for p in patterns:
        if p['type'] == '看涨分手线':
            bull_series[p['date']] = p['price']
        else:
            bear_series[p['date']] = p['price']

    ap = []
    if not bull_series.isna().all():
        ap.append(mpf.make_addplot(bull_series, type='scatter', marker='^',
                                   markersize=100, color='green', label='Bullish'))

    if not bear_series.isna().all():
        ap.append(mpf.make_addplot(bear_series, type='scatter', marker='v',
                                   markersize=100, color='red', label='Bearish'))

    print("正在生成 Mplfinance 图表...")
    mpf.plot(df, type='candle', volume=True, addplot=ap,
             title=f'{symbol_code} - 分手线形态识别',
             style=my_style, figsize=(14, 8))

# ==========================================
# 4. 可视化 - Matplotlib 详细分析
# ==========================================
def plot_detailed_analysis(df, patterns):
    if not patterns:
        return

    print("正在生成详细分析图表...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)

    # 绘制价格
    ax1.plot(df.index, df['Close'], label='收盘价', color='black', linewidth=1, alpha=0.6)

    for p in patterns:
        date = p['date']
        prev_date = df.index[p['index'] - 1]

        is_bullish = (p['type'] == '看涨分手线')
        color = 'green' if is_bullish else 'red'
        marker = '^' if is_bullish else 'v'

        # 1. 标记当前点
        ax1.scatter(date, p['price'], color=color, marker=marker, s=150, zorder=5)

        # 2. 连接两日开盘价 (虚线)
        ax1.plot([prev_date, date], [p['first_open'], p['second_open']],
                 color=color, linestyle='--', alpha=0.8, linewidth=2)

        # 3. 文字标注
        offset = 20 if is_bullish else -30
        ax1.annotate(p['type'], xy=(date, p['price']),
                     xytext=(0, offset), textcoords='offset points',
                     fontsize=9, color=color, ha='center',
                     arrowprops=dict(arrowstyle='->', color=color))

    ax1.set_title(f'{symbol_code} - 分手线形态详细分析', fontsize=16)
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 绘制成交量
    ax2.bar(df.index, df['Volume'], color='tab:blue', alpha=0.6)
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519'  # 贵州茅台

    # 1. 获取数据
    df = get_stock_data(symbol_code, days=365)

    if not df.empty:
        # 2. 识别形态
        # 容差设为 0.5% (0.005)，对于高价股茅台来说，开盘价差异允许在几块钱内
        patterns = identify_separation_line(df, price_tolerance=0.005)

        # 3. 打印结果
        bullish = [p for p in patterns if p['type'] == '看涨分手线']
        bearish = [p for p in patterns if p['type'] == '看跌分手线']

        print("=" * 60)
        print(f"总计识别到: 看涨分手线 {len(bullish)} 个, 看跌分手线 {len(bearish)} 个")

        if bullish:
            print("\n--- 看涨分手线列表 ---")
            for p in bullish:
                print(f"日期: {p['date'].strftime('%Y-%m-%d')} | 开盘价: {p['second_open']:.2f} | 强度: {p['strength']:.3f}")

        if bearish:
            print("\n--- 看跌分手线列表 ---")
            for p in bearish:
                print(f"日期: {p['date'].strftime('%Y-%m-%d')} | 开盘价: {p['second_open']:.2f} | 强度: {p['strength']:.3f}")

        # 4. 统计信息
        if patterns:
            avg_diff = np.mean([p['diff_pct'] for p in patterns]) * 100
            print("-" * 60)
            print(f"平均开盘价差异: {avg_diff:.3f}%")

            # 5. 绘图
            plot_separation_mpf(df, patterns)
            plot_detailed_analysis(df, patterns)
        else:
            print("\n未识别到分手线形态，请尝试调整 price_tolerance 参数。")