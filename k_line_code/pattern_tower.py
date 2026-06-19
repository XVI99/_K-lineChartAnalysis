import pandas as pd
from k_line_code.common.data_fetcher import fetch_stock_data
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
# 2. 塔形形态识别函数
# ==========================================
def identify_tower_patterns(df_in: pd.DataFrame, w: int = 5, body_pct: float = 0.60) -> list:
    """
    识别 塔形顶 (Tower Top) 和 塔形底 (Tower Bottom)

    逻辑 (完全保留原代码):
    - 塔形顶：前段(w)为大阳线，后段(w)为小实体
    - 塔形底：前段(w)为小实体，后段(w)为大阴线
    """
    df = df_in.copy()

    # 补算辅助列
    df['Range'] = df['High'] - df['Low']
    df['Body']  = (df['Close'] - df['Open']).abs()

    signals = []

    # 循环遍历 (从 2*w 开始，因为需要回溯两段 w)
    for i in range(2 * w, len(df)):
        up_seg   = df.iloc[i - 2 * w : i - w]   # 前段 (First half)
        down_seg = df.iloc[i - w : i]           # 后段 (Second half)

        # --- 逻辑定义 ---
        # 前段是大阳线 (全红 & 实体大)
        up_big = (up_seg['Close'] > up_seg['Open']).all() and \
                 (up_seg['Body'] >= body_pct * up_seg['Range']).all()

        # 前段是小实体
        up_small = (up_seg['Body'] <= 0.35 * up_seg['Range']).all()

        # 后段是大阴线 (全绿 & 实体大)
        down_big = (down_seg['Close'] < down_seg['Open']).all() and \
                   (down_seg['Body'] >= body_pct * down_seg['Range']).all()

        # 后段是小实体
        down_small = (down_seg['Body'] <= 0.35 * down_seg['Range']).all()

        # --- 形态判断 ---

        # 1. 塔形顶 (Tower Top)
        # 逻辑：前段大阳线 + 后段小实体
        if up_big and down_small:
            signals.append(('Tower Top', df.index[i], df['Close'].iloc[i]))

        # 2. 塔形底 (Tower Bottom)
        # 逻辑：前段小实体 + 后段大阴线 (注：这里保留了你代码中 if down_big and up_small 的组合)
        if down_big and up_small:
            signals.append(('Tower Bottom', df.index[i], df['Close'].iloc[i]))

    return signals

# ==========================================
# 3. 可视化绘图函数
# ==========================================
def plot_tower_patterns(df: pd.DataFrame, signals: list, symbol_code: str):
    """
    绘制 K 线及塔形形态标记
    """
    # 统计结果
    n_sig = len(signals)
    print(f'=== {symbol_code} 统计结果 ===')
    print(f'塔形顶/底共识别出 {n_sig} 次')

    if n_sig == 0:
        print('当前参数下无塔形形态，可调宽 w 或放宽 body_pct 再试')

    # 准备绘图数据 (拆分图层以便显示图例)
    df['TowerTop_Mark'] = np.nan
    df['TowerBottom_Mark'] = np.nan

    for name, date, price in signals:
        if 'Top' in name:
            df.loc[date, 'TowerTop_Mark'] = price
        elif 'Bottom' in name:
            df.loc[date, 'TowerBottom_Mark'] = price

    # 基础线 (如 MA)
    ap = [mpf.make_addplot(df['Close'].rolling(20).mean(), color='tab:blue', width=1.2)]

    # 添加塔形顶标记 (红色圆点)
    if not df['TowerTop_Mark'].isna().all():
        ap.append(mpf.make_addplot(df['TowerTop_Mark'], type='scatter', marker='o',
                                   markersize=100, color='red', label='Tower Top'))

    # 添加塔形底标记 (绿色圆点)
    if not df['TowerBottom_Mark'].isna().all():
        ap.append(mpf.make_addplot(df['TowerBottom_Mark'], type='scatter', marker='o',
                                   markersize=100, color='green', label='Tower Bottom'))

    # 绘图
    mpf.plot(df, type='candle', volume=True, addplot=ap,
             title=f'{symbol_code} Tower Top & Tower Bottom',
             style='yahoo', figsize=(14, 8))

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 配置
    TARGET_SYMBOL = '600519'  # 贵州茅台
    WINDOW_SIZE = 3           # 窗口 w (原代码示例为 5，建议可由 3-5 微调)
    BODY_PCT = 0.60           # 实体占比阈值

    # 1. 获取数据
    df_data = get_stock_data(TARGET_SYMBOL, days=365)

    if not df_data.empty:
        # 2. 计算形态
        # 注意：w=5 对于日线来说条件非常苛刻（连续5天大阳线/大阴线），
        # 如果跑不出结果，可以尝试把 w 改小，比如 w=3
        signal_list = identify_tower_patterns(df_data, w=WINDOW_SIZE, body_pct=BODY_PCT)

        # 3. 绘图
        plot_tower_patterns(df_data, signal_list, TARGET_SYMBOL)