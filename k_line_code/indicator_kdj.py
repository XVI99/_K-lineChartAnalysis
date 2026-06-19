from k_line_code.common.data_fetcher import fetch_stock_data
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ==========================================
# 0. 字体配置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
my_style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.family': 'SimHei'})

# ==========================================
# 1. 数据获取
# ==========================================
def get_stock_data(symbol: str, days: int = 365) -> pd.DataFrame:
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days)
    print(f"正在获取 {symbol} 数据...")
    try:
        df_raw = fetch_stock_data(symbol, days=days)
        df = (df_raw.rename(columns={'日期': 'Date', '开盘': 'Open', '最高': 'High',
                                     '最低': 'Low', '收盘': 'Close', '成交量': 'Volume'})
              .loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
              .assign(Date=lambda x: pd.to_datetime(x['Date']))
              .set_index('Date').sort_index())
        return df
    except Exception as e:
        print(f"数据获取失败: {e}")
        return pd.DataFrame()

# ==========================================
# 2. KDJ 计算函数 (核心)
# ==========================================
def calculate_kdj(df_in, N=9, M1=3, M2=3):
    """
    计算 KDJ 指标
    N: 计算 RSV 的周期 (通常为 9)
    M1: K 值的平滑周期 (通常为 3)
    M2: D 值的平滑周期 (通常为 3)
    """
    df = df_in.copy()

    # 1. 计算 RSV (Raw Stochastic Value)
    low_min = df['Low'].rolling(window=N).min()
    high_max = df['High'].rolling(window=N).max()

    # 防止分母为0
    range_hl = (high_max - low_min).replace(0, 1e-6)

    rsv = (df['Close'] - low_min) / range_hl * 100

    # 2. 计算 K, D, J
    # 在国内软件(如同花顺)中，K值和D值的计算通常使用类似 EMA 的平滑算法
    # 公式: K = 2/3 * 前一日K + 1/3 * 今日RSV
    # 这等同于 pandas 的 ewm(com=2) 或 alpha=1/3

    # 初始化 K 和 D，通常初始值为 50
    # 但直接用 pandas 的 ewm 可以自动处理
    # adjust=False 对应递归公式

    df['K'] = rsv.ewm(com=M1-1, adjust=False).mean()
    df['D'] = df['K'].ewm(com=M2-1, adjust=False).mean()

    # J = 3K - 2D
    df['J'] = 3 * df['K'] - 2 * df['D']

    return df

# ==========================================
# 3. 信号识别
# ==========================================
def identify_kdj_signals(df_in):
    df = df_in.copy()

    k_prev = df['K'].shift(1)
    d_prev = df['D'].shift(1)
    j_prev = df['J'].shift(1)

    # --- 1. 金叉与死叉 ---
    # 金叉：K, J 同时上穿 D
    df['GoldCross'] = (k_prev < d_prev) & (df['K'] > df['D'])
    # 死叉：K, J 同时下穿 D
    df['DeadCross'] = (k_prev > d_prev) & (df['K'] < df['D'])

    # --- 2. 超买与超卖 (J线敏感度最高) ---
    # J > 100 通常视为严重超买 (钝化区)
    # J < 0   通常视为严重超卖 (钝化区)
    df['OverBought_J'] = df['J'] > 100
    df['OverSold_J']   = df['J'] < 0

    # --- 3. 底部共振信号 (J线超卖 + 金叉) ---
    df['BuySignal'] = df['GoldCross'] & (df['D'] < 30) # 低位金叉

    # --- 4. 顶部共振信号 (J线超买 + 死叉) ---
    df['SellSignal'] = df['DeadCross'] & (df['D'] > 70) # 高位死叉

    return df

# ==========================================
# 4. 可视化
# ==========================================
def plot_kdj(df, symbol):
    ap = []

    # 1. 绘制 K, D, J 线 (在 Panel 1)
    ap.append(mpf.make_addplot(df['K'], panel=1, color='orange', width=1.0, label='K')) # K: 黄色
    ap.append(mpf.make_addplot(df['D'], panel=1, color='deepskyblue', width=1.0, label='D')) # D: 蓝色
    ap.append(mpf.make_addplot(df['J'], panel=1, color='purple', width=1.2, label='J')) # J: 紫色 (最活跃)

    # 2. 绘制超买超卖参考线
    ap.append(mpf.make_addplot(pd.Series(100, index=df.index), panel=1, color='red', linestyle=':', width=0.8))
    ap.append(mpf.make_addplot(pd.Series(0, index=df.index), panel=1, color='green', linestyle=':', width=0.8))
    # 20/50/80 也是常用参考
    ap.append(mpf.make_addplot(pd.Series(50, index=df.index), panel=1, color='gray', linestyle='--', width=0.5))

    # 3. 标记买卖信号
    buy_marks = np.where(df['BuySignal'], df['Low'] * 0.98, np.nan)
    sell_marks = np.where(df['SellSignal'], df['High'] * 1.02, np.nan)

    if not np.all(np.isnan(buy_marks)):
        ap.append(mpf.make_addplot(buy_marks, type='scatter', marker='^', color='red', markersize=80, label='Low Gold Cross'))

    if not np.all(np.isnan(sell_marks)):
        ap.append(mpf.make_addplot(sell_marks, type='scatter', marker='v', color='green', markersize=80, label='High Dead Cross'))

    mpf.plot(df, type='candle', volume=True, addplot=ap,
             style=my_style, figsize=(14, 9),
             title=f'{symbol} KDJ指标分析 (9,3,3)',
             panel_ratios=(3, 1),
             ylabel='Price', ylabel_lower='KDJ')

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519' # 贵州茅台

    # 1. 获取数据
    df = get_stock_data(symbol_code, days=300)

    if not df.empty:
        # 2. 计算 KDJ
        df = calculate_kdj(df, N=9, M1=3, M2=3)

        # 3. 识别信号
        df = identify_kdj_signals(df)

        # 4. 打印最近信号
        last_rows = df.tail(5)
        print("\n最近 5 天 KDJ 数据:")
        print(last_rows[['Close', 'K', 'D', 'J', 'GoldCross', 'DeadCross']])

        buy_count = int(df['BuySignal'].sum())
        sell_count = int(df['SellSignal'].sum())
        print(f"\n统计: 低位金叉(买点) {buy_count} 次, 高位死叉(卖点) {sell_count} 次")

        # 5. 绘图
        plot_kdj(df, symbol_code)
    else:
        print("无数据")