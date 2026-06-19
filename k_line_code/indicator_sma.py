from k_line_code.common.data_fetcher import fetch_stock_data
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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

    df = (df_raw
          .rename(columns={'日期': 'Date', '开盘': 'Open', '最高': 'High',
                           '最低': 'Low', '收盘': 'Close', '成交量': 'Volume'})
          .loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
          .assign(Date=lambda x: pd.to_datetime(x['Date']))
          .set_index('Date')
          .sort_index())

    return df

# ==========================================
# 2. 计算简单移动平均线 (SMA)
# ==========================================
def calculate_sma(df_in, periods=None):
    """
    计算指定周期的 SMA
    """
    if periods is None:
        periods = [5, 9, 30, 50, 200]

    df = df_in.copy()

    for n in periods:
        # SMA_n = 最近 n 个收盘价的算术平均
        df[f'SMA_{n}'] = df['Close'].rolling(n).mean()

    return df

# ==========================================
# 3. 可视化绘图函数
# ==========================================
def plot_sma(df, symbol, periods):
    ap = []

    # 定义不同周期的颜色
    colors = {
        5:   'orange',
        9:   'deepskyblue',
        30:  'violet',
        50:  'limegreen',
        200: 'red'
    }

    # 添加均线图层
    for n in periods:
        # 确保数据不全为空，否则 mplfinance 会报错
        if not df[f'SMA_{n}'].isna().all():
            ap.append(
                mpf.make_addplot(
                    df[f'SMA_{n}'],
                    color=colors.get(n, 'blue'), # 默认蓝色
                    width=1.0,
                    label=f'SMA {n}' # 注意：mplfinance 图例显示需要特定设置，这里仅作标识
                )
            )

    print("正在生成图表...")

    # 标题
    title_str = f'{symbol} 简单移动平均线 ({ "/".join(map(str, periods)) })'

    mpf.plot(
        df,
        type='candle',
        volume=True,
        addplot=ap,
        style=my_style,
        title=title_str,
        figsize=(14, 8)
    )

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519'  # 贵州茅台

    # 1. 获取数据 (获取稍长一点的时间，保证 200日均线 能计算出来)
    # 因为 SMA_200 需要前200天的数据，如果只取365天，前200天 SMA_200 都是 NaN
    df = get_stock_data(symbol_code, days=500)

    if not df.empty:
        # 2. 计算 SMA
        ma_periods = [5, 9, 30, 50, 200]
        df = calculate_sma(df, periods=ma_periods)

        # 3. 打印最近 5 天的数据检查
        cols = ['Close'] + [f'SMA_{n}' for n in ma_periods]
        print("\n最近 5 天的均线数值：")
        print(df[cols].tail(5))

        # 4. 绘图
        # 为了图表好看，截取最近一年的数据进行展示 (均线已经计算好了)
        plot_start_date = df.index[-1] - timedelta(days=365)
        df_plot = df.loc[plot_start_date:]

        plot_sma(df_plot, symbol_code, ma_periods)
    else:
        print("未获取到数据。")