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
# 1. 数据获取与预处理
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

def calculate_indicators(df, rsi_period=14):
    df = df.copy()

    # 1. 计算 RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    loss = loss.replace(0, np.nan) # 避免除零
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)

    # 2. 计算 K 线特征
    df['Body'] = abs(df['Close'] - df['Open'])
    df['Range'] = df['High'] - df['Low']

    # 防止 Range 为 0 导致除零错误
    df['Range'] = df['Range'].replace(0, 1e-6)

    df['Body_Ratio'] = df['Body'] / df['Range']

    # 计算影线
    # 上影线 = High - Max(Open, Close)
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    # 下影线 = Min(Open, Close) - Low
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    return df

# ==========================================
# 2. 十字线变体识别逻辑
# ==========================================
def identify_doji_variants(df,
                           doji_threshold=0.03,  # 实体占比阈值 (判断是否为十字线)
                           shadow_ratio=0.4,     # 影线占比阈值 (判断长腿/墓碑/蜻蜓)
                           uptrend_lookback=5,   # 趋势回顾窗口
                           rsi_overbought=70,    # 超买
                           rsi_oversold=30,      # 超卖
                           max_doji_freq=0.2,    # 稀缺性过滤
                           require_trend=True):  # 是否强制要求趋势配合

    patterns = []
    n = len(df)

    # 预计算是否为十字线
    is_doji_series = df['Body_Ratio'] <= doji_threshold

    for i in range(uptrend_lookback, n):
        curr = df.iloc[i]

        # 必须先是十字线
        if not is_doji_series.iloc[i]:
            continue

        date = df.index[i]

        # 1. 趋势判断
        recent_closes = df['Close'].iloc[i-uptrend_lookback:i]
        # 简单趋势判断：当前价格相对于过去均值的位置
        ma_short = recent_closes.mean()

        is_uptrend = (curr['Close'] > ma_short) if require_trend else True
        is_downtrend = (curr['Close'] < ma_short) if require_trend else True

        # 2. 稀缺性检查
        start_idx = max(0, i - uptrend_lookback*2)
        recent_doji_freq = is_doji_series.iloc[start_idx:i].mean()
        is_rare = recent_doji_freq <= max_doji_freq

        # 3. 超买超卖
        is_ob = curr['RSI'] > rsi_overbought
        is_os = curr['RSI'] < rsi_oversold

        # 基础强度 (实体越小、越稀缺，强度越高)
        strength_base = (1 - curr['Body_Ratio']) * (1 - recent_doji_freq)

        # 归一化影线比例
        u_ratio = curr['Upper_Shadow'] / curr['Range']
        l_ratio = curr['Lower_Shadow'] / curr['Range']

        # --- 形态 A: 长腿十字线 (Long-Legged Doji / Rickshaw Man) ---
        # 特征：上下影线都很长，且长度相近
        if (u_ratio >= shadow_ratio and l_ratio >= shadow_ratio):
            p_type = '长腿十字线'
            # 如果在高位超买，偏空；在低位超卖，偏多
            direction = '看跌(高位)' if (is_uptrend and is_ob) else ('看涨(低位)' if (is_downtrend and is_os) else '犹豫')

            patterns.append({
                'type': p_type,
                'sub_type': direction,
                'date': date,
                'price': curr['Close'],
                'high': curr['High'],
                'low': curr['Low'],
                'strength': strength_base,
                'rsi': curr['RSI']
            })

        # --- 形态 B: 墓碑十字线 (Gravestone Doji) ---
        # 特征：上影线极长，下影线极短/无，收盘在低位
        # 宽松判定：上影线占比大，下影线占比极小
        elif (u_ratio >= shadow_ratio * 1.2 and l_ratio <= 0.1):
            # 墓碑通常在高位看跌最有效
            if (is_uptrend and is_ob) or not require_trend:
                patterns.append({
                    'type': '墓碑十字线',
                    'sub_type': '看跌',
                    'date': date,
                    'price': curr['High'], # 标记在顶部
                    'high': curr['High'],
                    'low': curr['Low'],
                    'strength': strength_base * 1.2,
                    'rsi': curr['RSI']
                })

        # --- 形态 C: 蜻蜓十字线 (Dragonfly Doji) ---
        # 特征：下影线极长，上影线极短/无，收盘在高位
        elif (l_ratio >= shadow_ratio * 1.2 and u_ratio <= 0.1):
            # 蜻蜓通常在低位看涨最有效
            if (is_downtrend and is_os) or not require_trend:
                patterns.append({
                    'type': '蜻蜓十字线',
                    'sub_type': '看涨',
                    'date': date,
                    'price': curr['Low'], # 标记在底部
                    'high': curr['High'],
                    'low': curr['Low'],
                    'strength': strength_base * 1.2,
                    'rsi': curr['RSI']
                })

    return patterns

# ==========================================
# 3. 可视化 - Mplfinance
# ==========================================
def plot_variants_mpf(df, patterns):
    if not patterns:
        return

    # 为不同形态创建不同的 Series 图层
    # 墓碑(红) - 标记在 High 上方
    gravestone_s = pd.Series(np.nan, index=df.index)
    # 蜻蜓(绿) - 标记在 Low 下方
    dragonfly_s = pd.Series(np.nan, index=df.index)
    # 长腿(橙) - 标记在 High/Low 中间或上方
    longleg_s = pd.Series(np.nan, index=df.index)

    rng = df['High'].mean() - df['Low'].mean()

    for p in patterns:
        if '墓碑' in p['type']:
            gravestone_s[p['date']] = p['high'] + rng * 0.2
        elif '蜻蜓' in p['type']:
            dragonfly_s[p['date']] = p['low'] - rng * 0.2
        elif '长腿' in p['type']:
            longleg_s[p['date']] = p['high'] + rng * 0.2

    ap = []
    if not gravestone_s.isna().all():
        ap.append(mpf.make_addplot(gravestone_s, type='scatter', marker='v',
                                   markersize=100, color='red', label='Gravestone'))
    if not dragonfly_s.isna().all():
        ap.append(mpf.make_addplot(dragonfly_s, type='scatter', marker='^',
                                   markersize=100, color='green', label='Dragonfly'))
    if not longleg_s.isna().all():
        ap.append(mpf.make_addplot(longleg_s, type='scatter', marker='*',
                                   markersize=120, color='orange', label='Long-Legged'))

    print("正在生成 Mplfinance 图表...")
    mpf.plot(df, type='candle', volume=True, addplot=ap,
             title=f'{symbol_code} - 十字线变体 (Total: {len(patterns)})',
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
    ax1.plot(df.index, df['Close'].rolling(20).mean(), label='MA20', color='blue', linestyle='--', alpha=0.4)

    colors = {'长腿': 'orange', '墓碑': 'red', '蜻蜓': 'green'}
    markers = {'长腿': '*', '墓碑': 'v', '蜻蜓': '^'}

    for p in patterns:
        base_type = p['type'].split('十字')[0] # 提取"长腿"/"墓碑"/"蜻蜓"
        c = colors.get(base_type, 'gray')
        m = markers.get(base_type, 'o')

        # 决定标注的 Y 坐标
        y_pos = p['high'] if '墓碑' in p['type'] or '长腿' in p['type'] else p['low']

        ax1.scatter(p['date'], y_pos, color=c, marker=m, s=150, zorder=5)

        ax1.annotate(f"{p['type']}\n{p['sub_type']}\nRSI:{p['rsi']:.0f}",
                     (p['date'], y_pos), xytext=(0, 20 if '蜻蜓' not in p['type'] else -30),
                     textcoords='offset points', ha='center',
                     fontsize=8, color='black',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=c, alpha=0.3),
                     arrowprops=dict(arrowstyle="->", color=c))

    ax1.set_title(f'{symbol_code} - 十字线变体详细分析')
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
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519'  # 贵州茅台

    # 1. 获取数据
    df = get_stock_data(symbol_code, days=365)

    if not df.empty:
        # 2. 计算指标
        df = calculate_indicators(df)

        # 3. 识别形态
        # 参数说明：
        # doji_threshold: 实体占比小于 3% 视为十字
        # shadow_ratio: 影线占比大于 40% (对长腿) 或 48% (对墓碑/蜻蜓 1.2倍)
        # require_trend: 是否必须配合趋势和超买超卖才算有效信号
        patterns = identify_doji_variants(df,
                                          doji_threshold=0.03,
                                          shadow_ratio=0.4,
                                          uptrend_lookback=5,
                                          rsi_overbought=70,
                                          rsi_oversold=30,
                                          max_doji_freq=0.2,
                                          require_trend=False) # 设为False以展示更多形态，实战建议True

        # 4. 打印结果
        if patterns:
            print(f"\n总计识别到 {len(patterns)} 个形态")
            print("=" * 70)

            # 转换为 DataFrame 展示
            res_data = [{
                '日期': p['date'].strftime('%Y-%m-%d'),
                '类型': p['type'],
                '子类型': p['sub_type'],
                '强度': f"{p['strength']:.2f}",
                'RSI': f"{p['rsi']:.1f}"
            } for p in patterns]

            print(pd.DataFrame(res_data).to_string(index=False))

            # 5. 绘图
            plot_variants_mpf(df, patterns)
            plot_detailed(df, patterns)

            # 6. 统计参数敏感性
            print("\n" + "="*40)
            print("参数敏感性测试 (数量):")
            for th in [0.02, 0.03, 0.05]:
                p_test = identify_doji_variants(df, doji_threshold=th, require_trend=False)
                print(f"实体阈值={th}: {len(p_test)} 个")

        else:
            print("未识别到十字线变体。建议：")
            print("1. 调高 doji_threshold (放宽十字线定义)")
            print("2. 调低 shadow_ratio (放宽影线长度要求)")
            print("3. 设置 require_trend=False")