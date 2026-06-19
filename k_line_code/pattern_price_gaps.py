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
# 2. 缺口识别函数
# ==========================================
def identify_gaps(df_in: pd.DataFrame, min_gap_pct: float = 0.002) -> pd.DataFrame:
    """
    识别 向上缺口 (Up Gap) 和 向下缺口 (Down Gap)

    逻辑 (沿用原代码):
    - UpGap: 当前 Low > 前一日 High
    - DownGap: 当前 High < 前一日 Low
    - 阈值: 缺口幅度 / 前一日参考价 >= min_gap_pct
    """
    df_gap = df_in.copy()

    # 初始化列
    df_gap['UpGap']    = np.nan
    df_gap['DownGap']  = np.nan
    df_gap['GapColor'] = 'none' # 默认为透明
    df_gap['GapY']     = np.nan

    # 遍历数据 (从第 1 天开始)
    for i in range(1, len(df_gap)):
        prev = df_gap.iloc[i-1]
        curr = df_gap.iloc[i]

        # 计算缺口绝对值
        up_gap_val   = curr['Low'] - prev['High']
        down_gap_val = prev['Low'] - curr['High']

        # 1. 向上跳空
        if up_gap_val > 0 and (up_gap_val / prev['High']) >= min_gap_pct:
            df_gap.loc[df_gap.index[i], 'UpGap']    = up_gap_val
            df_gap.loc[df_gap.index[i], 'GapColor'] = 'green'
            # 标记坐标设为缺口的中间位置
            df_gap.loc[df_gap.index[i], 'GapY']     = (prev['High'] + curr['Low']) / 2

        # 2. 向下跳空
        if down_gap_val > 0 and (down_gap_val / prev['High']) >= min_gap_pct:
            df_gap.loc[df_gap.index[i], 'DownGap']  = down_gap_val
            df_gap.loc[df_gap.index[i], 'GapColor'] = 'red'
            # 标记坐标设为缺口的中间位置
            df_gap.loc[df_gap.index[i], 'GapY']     = (prev['Low'] + curr['High']) / 2

    return df_gap

# ==========================================
# 3. 可视化绘图函数
# ==========================================
def plot_gaps(df: pd.DataFrame, symbol_code: str):
    """
    绘制 K 线及缺口标记
    """
    # 统计结果
    n_up  = df['UpGap'].notna().sum()
    n_down = df['DownGap'].notna().sum()

    print(f'=== {symbol_code} 统计结果 ===')
    print(f'向上跳空 (Up Gap)：{n_up} 次')
    print(f'向下跳空 (Down Gap)：{n_down} 次')

    if n_up + n_down == 0:
        print('当前参数下无缺口，可调小 min_gap_pct 再试')

    # 准备绘图
    ap = []

    # 检查是否有缺口数据
    if not df['GapY'].isna().all():
        # 使用 scatter 画缺口位置
        # 注意：这里直接传入颜色列表，mplfinance 会根据列表为每个点上色
        # 'none' 的颜色值会让非缺口点不可见，从而实现只显示红/绿点的效果
        ap.append(mpf.make_addplot(df['GapY'], type='scatter',
                                   marker='o', markersize=80,
                                   color=df['GapColor'].tolist()))

    # 绘图
    mpf.plot(df, type='candle', volume=True, addplot=ap,
             title=f'{symbol_code} Price Gaps (Up=Green, Down=Red)',
             style='yahoo', figsize=(14, 8))

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 配置
    TARGET_SYMBOL = '600519'  # 贵州茅台
    GAP_THRESHOLD = 0.002     # 缺口阈值 (0.2%)

    # 1. 获取数据
    df_data = get_stock_data(TARGET_SYMBOL, days=365)

    if not df_data.empty:
        # 2. 计算形态
        df_patterns = identify_gaps(df_data, min_gap_pct=GAP_THRESHOLD)

        # 3. 绘图
        plot_gaps(df_patterns, TARGET_SYMBOL)