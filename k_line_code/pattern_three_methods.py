import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from k_line_code.common.data_fetcher import fetch_stock_data

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
# 2. 形态识别逻辑 (完全保留原逻辑)
# ==========================================
def identify_ascending_three_methods(df, min_body_ratio=0.5):
    """
    识别上升三法 (Rising Three Methods)
    逻辑：长阳 -> 三根小阴/小阳回调(在第一根范围内) -> 长阳突破
    """
    patterns = []
    # 至少需要5根K线，从第5根(索引4)开始遍历
    for i in range(4, len(df)):
        day1 = df.iloc[i-4] # 长阳
        day2 = df.iloc[i-3] # 回调
        day3 = df.iloc[i-2] # 回调
        day4 = df.iloc[i-1] # 回调
        day5 = df.iloc[i]   # 突破

        # 计算实体和波动
        day1_body = abs(day1['Close'] - day1['Open'])
        day1_range = day1['High'] - day1['Low']
        day5_body = abs(day5['Close'] - day5['Open'])

        # 1. 第一根是长阳线
        day1_body_ratio = day1_body / day1_range if day1_range > 0 else 0
        is_day1_bullish = day1['Close'] > day1['Open']

        # 2. 中间三根在第一根范围内 (High <= Day1 High, Low >= Day1 Low)
        # 注：标准定义允许影线略微刺破，这里严格按照你的代码逻辑
        day2_in = (day2['High'] <= day1['High']) and (day2['Low'] >= day1['Low'])
        day3_in = (day3['High'] <= day1['High']) and (day3['Low'] >= day1['Low'])
        day4_in = (day4['High'] <= day1['High']) and (day4['Low'] >= day1['Low'])

        # 3. 中间趋势向下 (平均收盘价 < Day1 收盘)
        middle_trend = (day2['Close'] + day3['Close'] + day4['Close']) / 3 < day1['Close']

        # 4. 第五根是长阳线且突破
        is_day5_bullish = day5['Close'] > day5['Open']
        is_breakout = day5['Close'] > day1['Close']

        if (is_day1_bullish and
                day1_body_ratio >= min_body_ratio and
                day2_in and day3_in and day4_in and
                middle_trend and
                is_day5_bullish and is_breakout):

            patterns.append({
                'type': '上升三法',
                'end_index': i,
                'end_date': df.index[i],
                'price': day5['Close']
            })
    return patterns

def identify_descending_three_methods(df, min_body_ratio=0.5):
    """
    识别下降三法 (Falling Three Methods)
    逻辑：长阴 -> 三根小阳/小阴反弹(在第一根范围内) -> 长阴跌破
    """
    patterns = []
    for i in range(4, len(df)):
        day1 = df.iloc[i-4]
        day2 = df.iloc[i-3]
        day3 = df.iloc[i-2]
        day4 = df.iloc[i-1]
        day5 = df.iloc[i]

        day1_body = abs(day1['Close'] - day1['Open'])
        day1_range = day1['High'] - day1['Low']

        # 1. 第一根长阴线
        day1_body_ratio = day1_body / day1_range if day1_range > 0 else 0
        is_day1_bearish = day1['Close'] < day1['Open']

        # 2. 中间三根在范围内
        day2_in = (day2['High'] <= day1['High']) and (day2['Low'] >= day1['Low'])
        day3_in = (day3['High'] <= day1['High']) and (day3['Low'] >= day1['Low'])
        day4_in = (day4['High'] <= day1['High']) and (day4['Low'] >= day1['Low'])

        # 3. 中间趋势向上
        middle_trend = (day2['Close'] + day3['Close'] + day4['Close']) / 3 > day1['Close']

        # 4. 第五根长阴且跌破
        is_day5_bearish = day5['Close'] < day5['Open']
        is_breakdown = day5['Close'] < day1['Close']

        if (is_day1_bearish and
                day1_body_ratio >= min_body_ratio and
                day2_in and day3_in and day4_in and
                middle_trend and
                is_day5_bearish and is_breakdown):

            patterns.append({
                'type': '下降三法',
                'end_index': i,
                'end_date': df.index[i],
                'price': day5['Close']
            })
    return patterns

# ==========================================
# 3. 可视化 - Mplfinance (修复对齐问题)
# ==========================================
def plot_patterns_mpf(df, patterns, pattern_name):
    """
    绘制蜡烛图并标记特定形态
    """
    # 构造与 df 等长的 Series，默认为 NaN
    marker_series = pd.Series(np.nan, index=df.index)

    # 填充形态出现点的价格
    for p in patterns:
        if p['type'] == pattern_name:
            marker_series[p['end_date']] = p['price']

    # 如果没有识别到形态，跳过绘图
    if marker_series.isna().all():
        print(f"未识别到 {pattern_name}，跳过绘图。")
        return

    ap = []
    color = 'green' if '上升' in pattern_name else 'red'
    marker = '^' if '上升' in pattern_name else 'v'

    ap.append(mpf.make_addplot(marker_series, type='scatter',
                               marker=marker, markersize=100,
                               color=color, label=pattern_name))

    print(f"正在生成 {pattern_name} 图表...")
    mpf.plot(df, type='candle', volume=True, addplot=ap,
             title=f'{symbol_code} - {pattern_name}',
             style=my_style, figsize=(14, 8))

# ==========================================
# 4. 可视化 - Matplotlib 详细分析
# ==========================================
def plot_detailed_analysis(df, patterns):
    """
    使用 Matplotlib 绘制详细分析图，使用色块标记 5 根 K 线范围
    """
    if not patterns:
        return

    print("正在生成详细分析图表...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)

    # 绘制价格
    ax1.plot(df.index, df['Close'], label='收盘价', color='black', linewidth=1, alpha=0.6)

    # 标记每一个形态
    for p in patterns:
        end_idx = p['end_index']
        # 获取形态的开始时间 (T-4) 和结束时间 (T)
        if end_idx >= 4:
            start_date = df.index[end_idx - 4]
            end_date = df.index[end_idx]

            # 设置颜色：上升为绿，下降为红
            is_rising = (p['type'] == '上升三法')
            color = 'green' if is_rising else 'red'
            marker = '^' if is_rising else 'v'

            # 1. 绘制背景区域 (span)
            ax1.axvspan(start_date, end_date, alpha=0.15, color=color)

            # 2. 标记结束点
            ax1.scatter(end_date, p['price'], color=color, marker=marker, s=100, zorder=5)

            # 3. 添加文字标注
            ax1.annotate(p['type'], xy=(end_date, p['price']),
                         xytext=(end_date, p['price'] * (1.02 if is_rising else 0.98)),
                         color=color, fontsize=9, ha='center',
                         arrowprops=dict(arrowstyle='-', color=color, alpha=0.5))

    ax1.set_title(f'{symbol_code} - 上升/下降三法形态详细分析', fontsize=16)
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
        # 2. 识别形态 (稍微放宽一点 min_body_ratio 以便更容易在日线中找到示例)
        # 严格的教科书定义通常要求非常大的实体，实盘中 0.5 左右比较合理
        ascending = identify_ascending_three_methods(df, min_body_ratio=0.5)
        descending = identify_descending_three_methods(df, min_body_ratio=0.5)

        all_patterns = ascending + descending

        # 3. 打印结果
        print("=" * 60)
        print(f"总计识别到: 上升三法 {len(ascending)} 个, 下降三法 {len(descending)} 个")

        if ascending:
            print("\n--- 上升三法列表 ---")
            for p in ascending:
                print(f"日期: {p['end_date'].strftime('%Y-%m-%d')} | 价格: {p['price']:.2f}")

        if descending:
            print("\n--- 下降三法列表 ---")
            for p in descending:
                print(f"日期: {p['end_date'].strftime('%Y-%m-%d')} | 价格: {p['price']:.2f}")

        # 4. 绘图
        if all_patterns:
            # 分别绘制 mplfinance 图
            plot_patterns_mpf(df, ascending, '上升三法')
            plot_patterns_mpf(df, descending, '下降三法')

            # 绘制综合分析图
            plot_detailed_analysis(df, all_patterns)
        else:
            print("\n当前参数下未识别到任何三法形态，请尝试：")
            print("1. 调整 min_body_ratio (如降低到 0.4)")
            print("2. 增加获取数据的天数 (如 days=730)")