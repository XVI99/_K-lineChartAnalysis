import akshare as ak
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
# 1. 数据获取与指标计算
# ==========================================
def get_stock_data(symbol: str, days: int = 730) -> pd.DataFrame:
    """获取数据，默认取2年以增加识别到稀有形态的概率"""
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

def calculate_indicators(df, rsi_period=14):
    df = df.copy()

    # 1. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)

    # 2. 实体与波动范围
    df['Body'] = abs(df['Close'] - df['Open'])
    df['Range'] = df['High'] - df['Low']
    df['Range'] = df['Range'].replace(0, 1e-6) # 避免除零
    df['Body_Ratio'] = df['Body'] / df['Range']

    return df

# ==========================================
# 2. 三星形态识别逻辑
# ==========================================
def identify_three_stars(df,
                         doji_threshold=0.03, # 实体占比阈值 (0.03 表示实体小于波动的3%视作十字)
                         lookback=10,         # 趋势判断窗口
                         rsi_overbought=70,
                         rsi_oversold=30,
                         max_doji_freq=0.2,   # 稀缺性过滤
                         require_new_extreme=True): # 中间星线是否必须创新高/新低
    patterns = []
    df = df.copy()
    n = len(df)

    # 标记十字线
    df['Is_Doji'] = df['Body_Ratio'] <= doji_threshold

    # 从第3根线开始遍历
    for i in range(lookback, n):
        # 必须连续三根都是十字线 (或极小实体)
        if not all(df['Is_Doji'].iloc[i-2 : i+1]):
            continue

        left = df.iloc[i-2]
        mid = df.iloc[i-1]
        right = df.iloc[i]

        # 为了方便绘图，记录形态结束日期
        pattern_date = right.name

        # --- 形态结构判断 ---
        # 顶部结构：中间的高点最高，且低点也相对较高 (凸起)
        is_structure_top = (mid['High'] > left['High']) and (mid['High'] > right['High'])

        # 底部结构：中间的低点最低，且高点也相对较低 (凹陷)
        is_structure_bottom = (mid['Low'] < left['Low']) and (mid['Low'] < right['Low'])

        if not (is_structure_top or is_structure_bottom):
            continue

        # --- 趋势判断 ---
        recent_window = df.iloc[i-lookback : i-2]
        recent_closes = recent_window['Close']

        # 简单趋势：当前价格区位较高/较低
        price_level = mid['Close']
        window_high = recent_window['High'].max()
        window_low = recent_window['Low'].min()

        # 是否接近新高/新低 (如果 require_new_extreme=True 则严格，否则宽松)
        if require_new_extreme:
            is_uptrend = mid['High'] >= window_high
            is_downtrend = mid['Low'] <= window_low
        else:
            # 宽松模式：只要价格在过去一段时间的上半区/下半区
            is_uptrend = price_level > recent_closes.mean()
            is_downtrend = price_level < recent_closes.mean()

        # --- 稀缺性 & RSI ---
        # 检查过去一段时间十字线的频率 (排除盘整期频繁出现的十字)
        recent_dojis = df['Is_Doji'].iloc[i-lookback*2 : i-2].mean()
        is_rare = recent_dojis <= max_doji_freq

        avg_rsi = np.mean([left['RSI'], mid['RSI'], right['RSI']])
        is_ob = avg_rsi > rsi_overbought
        is_os = avg_rsi < rsi_oversold

        # --- 最终判定 ---
        if is_structure_top and is_uptrend and is_rare:
            # 顶部三星：通常配合超买
            if is_ob or not require_new_extreme:
                strength = (1 - recent_dojis) * (avg_rsi / 100)
                patterns.append({
                    'type': '三星顶部',
                    'date': pattern_date,
                    'price': mid['High'], # 标记在中间K线的高点
                    'mid_high': mid['High'],
                    'strength': strength,
                    'rsi': avg_rsi,
                    'doji_freq': recent_dojis
                })

        elif is_structure_bottom and is_downtrend and is_rare:
            # 底部三星：通常配合超卖
            if is_os or not require_new_extreme:
                strength = (1 - recent_dojis) * (1 - avg_rsi / 100)
                patterns.append({
                    'type': '三星底部',
                    'date': pattern_date,
                    'price': mid['Low'], # 标记在中间K线的低点
                    'mid_low': mid['Low'],
                    'strength': strength,
                    'rsi': avg_rsi,
                    'doji_freq': recent_dojis
                })

    return patterns

# ==========================================
# 3. 可视化 - Mplfinance
# ==========================================
def plot_three_stars_mpf(df, patterns):
    if not patterns:
        return

    # 构造标记序列
    top_series = pd.Series(np.nan, index=df.index)
    bot_series = pd.Series(np.nan, index=df.index)

    rng = df['High'].mean() - df['Low'].mean()

    for p in patterns:
        if '顶部' in p['type']:
            top_series[p['date']] = p['price'] + rng * 0.1
        else:
            bot_series[p['date']] = p['price'] - rng * 0.1

    ap = []
    if not top_series.isna().all():
        ap.append(mpf.make_addplot(top_series, type='scatter', marker='v',
                                   markersize=200, color='red', label='Tri-Star Top'))
    if not bot_series.isna().all():
        ap.append(mpf.make_addplot(bot_series, type='scatter', marker='^',
                                   markersize=200, color='green', label='Tri-Star Bottom'))

    print("正在生成 Mplfinance 图表...")
    mpf.plot(df, type='candle', volume=True, addplot=ap,
             title=f'{symbol_code} - 三星形态 (Total: {len(patterns)})',
             style=my_style, figsize=(14, 8))

# ==========================================
# 4. 可视化 - Matplotlib 详细分析
# ==========================================
def plot_detailed(df, patterns):
    if not patterns:
        return

    print("正在生成详细分析图表...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # 价格图
    ax1.plot(df.index, df['Close'], label='Close', color='black', alpha=0.6)
    ax1.plot(df.index, df['Close'].rolling(20).mean(), label='MA20', color='blue', alpha=0.4, linestyle='--')

    for p in patterns:
        is_top = '顶部' in p['type']
        color = 'red' if is_top else 'green'
        marker = '*'

        # 标记位置
        ax1.scatter(p['date'], p['price'], color=color, marker=marker, s=300, zorder=5)

        # 文本注释
        offset = 30 if is_top else -40
        ax1.annotate(f"{p['type']}\nRSI:{p['rsi']:.0f}",
                     (p['date'], p['price']), xytext=(0, offset),
                     textcoords='offset points', ha='center',
                     fontsize=9, color='white',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                     arrowprops=dict(arrowstyle="->", color=color))

    ax1.set_title(f'{symbol_code} - 三星形态详细分析')
    ax1.set_ylabel('价格')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 成交量
    ax2.bar(df.index, df['Volume'], color='gray', alpha=0.6)
    ax2.set_ylabel('成交量')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519'  # 贵州茅台

    # 1. 获取数据 (三星极罕见，建议获取更多数据)
    df = get_stock_data(symbol_code, days=730)

    if not df.empty:
        # 2. 计算指标
        df = calculate_indicators(df)

        # 3. 识别形态
        # 默认参数较严格，如果找不到，程序会自动尝试放宽参数
        print("开始识别形态...")
        patterns = identify_three_stars(df,
                                        doji_threshold=0.03, # 实体 < 3% 波动
                                        lookback=10,
                                        rsi_overbought=70,
                                        rsi_oversold=30,
                                        max_doji_freq=0.2,
                                        require_new_extreme=True)

        # 如果没找到，尝试放宽条件 (因为A股大盘股很难走出完美三星)
        if not patterns:
            print("严格模式未找到，尝试放宽条件 (doji_threshold=0.05, require_new_extreme=False)...")
            patterns = identify_three_stars(df,
                                            doji_threshold=0.05,
                                            lookback=10,
                                            rsi_overbought=60, # 放宽 RSI
                                            rsi_oversold=40,
                                            max_doji_freq=0.3,
                                            require_new_extreme=False)

        # 4. 打印结果
        if patterns:
            print(f"\n总计识别到 {len(patterns)} 个三星形态")
            print("=" * 70)

            res_data = [{
                '日期': p['date'].strftime('%Y-%m-%d'),
                '类型': p['type'],
                '强度': f"{p['strength']:.3f}",
                'RSI': f"{p['rsi']:.1f}",
                '十字频率': f"{p['doji_freq']:.2f}"
            } for p in patterns]

            print(pd.DataFrame(res_data).to_string(index=False))

            # 5. 绘图
            plot_three_stars_mpf(df, patterns)
            plot_detailed(df, patterns)

            # 6. 后续表现
            print("\n后续表现分析 (5天/10天回报):")
            print("-" * 40)
            for p in patterns:
                idx = df.index.get_loc(p['date'])
                base = df['Close'].iloc[idx]
                date_str = p['date'].strftime('%Y-%m-%d')

                res_str = f"{date_str} ({p['type']}) | "
                for days in [5, 10]:
                    if idx + days < len(df):
                        ret = (df['Close'].iloc[idx+days] - base) / base * 100
                        res_str += f"{days}日: {ret:6.2f}%  "
                    else:
                        res_str += f"{days}日:N/A     "
                print(res_str)

        else:
            print("\n未识别到三星形态。")
            print("原因可能是形态过于罕见。建议：")
            print("1. 进一步调高 doji_threshold (如 0.06)")
            print("2. 扩大数据范围 (如 days=1000)")
            print("3. 更换波动率较大的中小盘股票代码")

        # 7. 参数敏感性测试
        print("\n" + "="*40)
        print("参数敏感性测试 (数量):")
        for th in [0.03, 0.05, 0.08]:
            p_test = identify_three_stars(df, doji_threshold=th, require_new_extreme=False)
            print(f"实体阈值={th}: {len(p_test)} 个")