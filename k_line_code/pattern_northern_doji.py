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
# 2. 指标计算 (RSI)
# ==========================================
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # 避免除以零
    loss = loss.replace(0, np.nan)

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 填充 NaN 以防报错
    df['RSI'] = df['RSI'].fillna(50)
    return df

# ==========================================
# 3. 北方十字线识别逻辑 (保留原逻辑)
# ==========================================
def identify_northern_doji(df,
                           min_body_ratio=0.5,   # 大阳实体比例
                           doji_threshold=0.03,  # 十字实体阈值
                           uptrend_lookback=5,   # 上升趋势期数
                           rsi_overbought=70,    # 超买阈值
                           require_gap=False,    # 是否要求跳空
                           max_doji_freq=0.1):   # 过去N天十字频率阈值
    patterns = []
    df = df.copy() # 避免修改原始数据
    n = len(df)

    # 预计算
    df['Body'] = abs(df['Close'] - df['Open'])
    df['Range'] = df['High'] - df['Low']
    # 避免 Range 为 0
    df['Body_Ratio'] = df['Body'] / df['Range'].replace(0, 1e-6)
    df['Is_Doji'] = df['Body_Ratio'] <= doji_threshold

    for i in range(uptrend_lookback, n):
        first = df.iloc[i-1]  # 前一根（大阳）
        second = df.iloc[i]   # 当前（十字）

        # 条件1: 前一根是大阳线
        is_first_long_bullish = (first['Close'] > first['Open']) and (first['Body_Ratio'] >= min_body_ratio)

        # 条件2: 当前是十字线
        is_second_doji = second['Is_Doji']

        # 条件3: 可选跳空（书籍非强制，但通常要求高位）
        is_above_first = second['Low'] >= first['Close'] if require_gap else True

        # 条件4: 上升趋势
        recent_closes = df['Close'].iloc[i-uptrend_lookback:i]
        # 简单判断：当前价格高于N天前，或者MA向上。这里沿用原代码逻辑：
        # 原代码：all(recent_closes.diff() > 0) -> 严格连续上涨
        # 为提高容错率，改为：收盘价整体呈上升趋势
        diffs = recent_closes.diff().dropna()
        is_uptrend = (diffs > 0).sum() >= (len(diffs) * 0.6) # 允许少量回调

        # 条件5: 有显著影线
        upper_shadow = second['High'] - max(second['Open'], second['Close'])
        lower_shadow = min(second['Open'], second['Close']) - second['Low']
        rng = second['Range'] if second['Range'] > 0 else 1
        shadow_ratio = max(upper_shadow, lower_shadow) / rng
        has_significant_shadows = shadow_ratio >= 0.3

        # 条件6: 超买状态
        is_overbought = second['RSI'] > rsi_overbought

        # 条件7: 稀缺性 (十字线不常出现)
        start_idx = max(0, i - uptrend_lookback*2)
        recent_dojis = df['Is_Doji'].iloc[start_idx:i].mean()
        is_rare = recent_dojis <= max_doji_freq

        if (is_first_long_bullish and is_second_doji and is_above_first and
                is_uptrend and has_significant_shadows and is_overbought and is_rare):

            # 强度计算
            strength = (min(first['Body_Ratio'], 1 - second['Body_Ratio']) +
                        (second['RSI'] - 70)/30 + (1 - recent_dojis)) / 3

            patterns.append({
                'type': '北方十字线',
                'index': i,
                'date': df.index[i],
                'first_close': first['Close'],
                'second_close': second['Close'],
                'second_high': second['High'],
                'second_low': second['Low'],
                'strength': strength,
                'rsi': second['RSI'],
                'doji_freq': recent_dojis,
                'gap_size': second['Low'] - first['Close'] if require_gap else 0
            })

    return patterns

# ==========================================
# 4. 可视化 - Mplfinance
# ==========================================
def plot_patterns_mpf(df, patterns):
    # 构造标记序列 (全长序列，非标记点为 NaN)
    marker_series = pd.Series(np.nan, index=df.index)

    for p in patterns:
        # 标记位置在最高价上方一点
        marker_series[p['date']] = p['second_high'] * 1.01

    ap = []
    if not marker_series.isna().all():
        ap.append(mpf.make_addplot(marker_series, type='scatter',
                                   marker='*', markersize=150,
                                   color='orange', label='Northern Doji'))

    print("正在生成 Mplfinance 图表...")
    mpf.plot(df, type='candle', volume=True, addplot=ap,
             title=f'{symbol_code} - 北方十字线 (Total: {len(patterns)})',
             style=my_style, figsize=(14, 8))

# ==========================================
# 5. 可视化 - Matplotlib 详细分析
# ==========================================
def plot_detailed(df, patterns):
    print("正在生成详细分析图表...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # 价格图
    ax1.plot(df.index, df['Close'], label='Close', color='black', alpha=0.7)
    ax1.plot(df.index, df['Close'].rolling(20).mean(), label='MA20', color='blue', alpha=0.5, linestyle='--')

    for p in patterns:
        # 标记星号
        ax1.scatter(p['date'], p['second_high'], color='orange', marker='*', s=200, zorder=5)

        # 添加注释框
        ax1.annotate(f"北方十字\n强度:{p['strength']:.2f}\nRSI:{p['rsi']:.0f}",
                     (p['date'], p['second_high']), xytext=(10, 20),
                     textcoords='offset points', color='black', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.3),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    ax1.set_title(f'{symbol_code} - 北方十字线详细分析')
    ax1.set_ylabel('价格')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 成交量
    ax2.bar(df.index, df['Volume'], color='gray', alpha=0.6)
    ax2.set_ylabel('成交量')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ==========================================
# 6. 主程序入口
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519'  # 贵州茅台

    # 1. 获取数据
    df = get_stock_data(symbol_code, days=730) # 建议取更长时间以捕捉更多形态

    if not df.empty:
        # 2. 计算 RSI
        df = calculate_rsi(df)

        # 3. 识别形态
        # 注意：这里稍微放宽了 is_uptrend 的逻辑以适应更多实际情况
        patterns = identify_northern_doji(df,
                                          min_body_ratio=0.5,
                                          doji_threshold=0.03,
                                          uptrend_lookback=3,
                                          rsi_overbought=70,
                                          require_gap=False,
                                          max_doji_freq=0.1)

        # 4. 打印表格结果
        if patterns:
            print("\n识别到的北方十字线形态：")
            print("=" * 60)
            data = [{
                '日期': p['date'].strftime('%Y-%m-%d'),
                '强度': f"{p['strength']:.3f}",
                'RSI': f"{p['rsi']:.1f}",
                '十字频率': f"{p['doji_freq']:.3f}",
                '缺口': f"{p['gap_size']:.2f}"
            } for p in patterns]
            print(pd.DataFrame(data).to_string(index=False))
            print(f"\n总计: {len(patterns)} 个")

            # 5. 绘图
            plot_patterns_mpf(df, patterns)
            plot_detailed(df, patterns)

            # 6. 后续表现统计
            print("\n后续表现分析 (5天/10天回报):")
            print("-" * 40)
            for p in patterns:
                idx = p['index']
                base = p['second_close']
                date_str = p['date'].strftime('%Y-%m-%d')

                res_str = f"{date_str} | "
                for days in [5, 10]:
                    if idx + days < len(df):
                        ret = (df['Close'].iloc[idx+days] - base) / base * 100
                        res_str += f"{days}日: {ret:6.2f}%  "
                    else:
                        res_str += f"{days}日:N/A     "
                print(res_str)

        else:
            print("未识别到形态。建议：")
            print("1. 降低 rsi_overbought (如 60)")
            print("2. 增加数据天数")
            print("3. 放宽 min_body_ratio")

        # 7. 参数敏感性测试
        print("\n" + "="*40)
        print("参数敏感性测试:")
        for rsi in [60, 70, 80]:
            test = identify_northern_doji(df, rsi_overbought=rsi)
            print(f"RSI阈值={rsi}: 识别到 {len(test)} 个")