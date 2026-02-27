import akshare as ak
import pandas as pd
import numpy as np
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

# ==========================================
# 2. 箱体识别与突破检测
# ==========================================
def identify_box_breakout(df_in, window=30, max_span_pct=0.06, buf_pct=0.002):
    df = df_in.copy()

    # 初始化列
    df['BoxLow']    = np.nan
    df['BoxHigh']   = np.nan
    df['Target_Up']   = np.nan
    df['Target_Down'] = np.nan
    df['BreakUp']   = False
    df['BreakDown'] = False
    df['Up_Invalid'] = False
    df['Dn_Invalid'] = False

    high = df['High']
    low  = df['Low']

    # 1. 识别潜在箱体 (Rolling 窗口内的高低差)
    rolling_max = high.rolling(window).max()
    rolling_min = low.rolling(window).min()
    # 波动幅度百分比
    span_pct = (rolling_max - rolling_min) / rolling_min

    # 标记符合条件的横盘区域
    is_box = span_pct <= max_span_pct

    # 2. 提取最近的一个有效箱体
    # 倒序查找连续的 True 区段
    box_end = None
    box_start = None
    in_box = False

    for i in range(len(is_box)-1, -1, -1):
        if is_box.iloc[i] and not in_box:
            in_box = True
            box_end = is_box.index[i]
        if in_box and not is_box.iloc[i]:
            box_start = is_box.index[i+1] # 这里的+1是为了取回变成True的那个点
            break

    # 如果没找到完整的结束点（比如一直横盘到最早的数据），就取最早
    if in_box and box_start is None:
        box_start = df.index[0]

    if box_start is None or box_end is None:
        print("未识别到明显的箱体区间。")
        return df

    # 3. 确定箱体参数
    box_df = df.loc[box_start:box_end]
    box_low  = box_df['Low'].min()
    box_high = box_df['High'].max()
    box_height = box_high - box_low

    print(f"识别到箱体: {box_start.strftime('%Y-%m-%d')} ~ {box_end.strftime('%Y-%m-%d')}")
    print(f"箱体范围: {box_low:.2f} - {box_high:.2f}, 高度: {box_height:.2f}")

    # 标记箱体范围 (从开始一直延伸到最后，方便画图)
    df.loc[box_start:, 'BoxLow']  = box_low
    df.loc[box_start:, 'BoxHigh'] = box_high

    # 4. 检测突破 (在箱体结束时间之后)
    after_box = df.loc[box_end:]

    if after_box.empty:
        return df

    # 向上突破: 收盘价 > 箱体顶 * (1 + 缓冲)
    up_break_idx = after_box[after_box['Close'] > box_high * (1 + buf_pct)].index
    # 向下突破: 收盘价 < 箱体底 * (1 - 缓冲)
    down_break_idx = after_box[after_box['Close'] < box_low * (1 - buf_pct)].index

    # 处理向上突破
    if len(up_break_idx) > 0:
        bidx_up = up_break_idx[0] # 首次突破点
        df.at[bidx_up, 'BreakUp'] = True

        # 目标价 = 箱体顶 + 箱体高度
        target = box_high + box_height
        df['Target_Up'] = np.where(df.index >= bidx_up, target, np.nan)
        print(f"向上突破日期: {bidx_up.strftime('%Y-%m-%d')}, 目标价: {target:.2f}")

        # 检查失效 (突破后收盘跌回箱体顶下方)
        # 只检查突破之后的日子
        check_invalid = df.loc[bidx_up:].iloc[1:]
        inv_idx = check_invalid[check_invalid['Close'] < box_high].index
        if len(inv_idx) > 0:
            first_inv = inv_idx[0]
            df.at[first_inv, 'Up_Invalid'] = True
            print(f"注意: 向上突破在 {first_inv.strftime('%Y-%m-%d')} 失效 (跌回箱体)")

    # 处理向下突破
    if len(down_break_idx) > 0:
        bidx_dn = down_break_idx[0] # 首次突破点
        df.at[bidx_dn, 'BreakDown'] = True

        # 目标价 = 箱体底 - 箱体高度
        target = box_low - box_height
        df['Target_Down'] = np.where(df.index >= bidx_dn, target, np.nan)
        print(f"向下突破日期: {bidx_dn.strftime('%Y-%m-%d')}, 目标价: {target:.2f}")

        # 检查失效 (突破后收盘涨回箱体底上方)
        check_invalid = df.loc[bidx_dn:].iloc[1:]
        inv_idx = check_invalid[check_invalid['Close'] > box_low].index
        if len(inv_idx) > 0:
            first_inv = inv_idx[0]
            df.at[first_inv, 'Dn_Invalid'] = True
            print(f"注意: 向下突破在 {first_inv.strftime('%Y-%m-%d')} 失效 (涨回箱体)")

    return df

# ==========================================
# 3. 可视化绘图函数
# ==========================================
def plot_box_breakout(df, symbol):
    ap = []

    # 1. 画箱体边界
    if not df['BoxLow'].isna().all():
        ap.append(mpf.make_addplot(df['BoxLow'], color='seagreen', linestyle='--', width=1.0))
        ap.append(mpf.make_addplot(df['BoxHigh'], color='crimson', linestyle='--', width=1.0))

    # 2. 画目标价
    if not df['Target_Up'].isna().all():
        ap.append(mpf.make_addplot(df['Target_Up'], color='orange', linestyle=':', width=1.0))
    if not df['Target_Down'].isna().all():
        ap.append(mpf.make_addplot(df['Target_Down'], color='dodgerblue', linestyle=':', width=1.0))

    # 3. 标记突破点
    up_marks = np.where(df['BreakUp'], df['High'] * 1.01, np.nan)
    dn_marks = np.where(df['BreakDown'], df['Low'] * 0.99, np.nan)

    if not np.all(np.isnan(up_marks)):
        ap.append(mpf.make_addplot(up_marks, type='scatter', marker='^',
                                   color='seagreen', markersize=120, label='Breakout Up'))
    if not np.all(np.isnan(dn_marks)):
        ap.append(mpf.make_addplot(dn_marks, type='scatter', marker='v',
                                   color='crimson', markersize=120, label='Breakout Down'))

    # 4. 标记失效点
    up_inv = np.where(df['Up_Invalid'], df['Low'] * 0.99, np.nan)
    dn_inv = np.where(df['Dn_Invalid'], df['High'] * 1.01, np.nan)

    if not np.all(np.isnan(up_inv)):
        ap.append(mpf.make_addplot(up_inv, type='scatter', marker='x',
                                   color='red', markersize=100, label='Invalid Up'))
    if not np.all(np.isnan(dn_inv)):
        ap.append(mpf.make_addplot(dn_inv, type='scatter', marker='x',
                                   color='blue', markersize=100, label='Invalid Down'))

    print("正在生成图表...")
    mpf.plot(
        df,
        type='candle',
        volume=True,
        addplot=ap,
        style=my_style,
        title=f'{symbol} - 箱体突破与目标测算',
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
        # 2. 识别箱体与突破
        # window=30: 30天内波动小
        # max_span_pct=0.06: 波动幅度小于 6% 视为横盘
        df = identify_box_breakout(df, window=30, max_span_pct=0.06)

        # 3. 统计信息
        if df['BreakUp'].any() or df['BreakDown'].any():
            # 4. 绘图
            plot_box_breakout(df, symbol_code)
        else:
            print("未检测到有效突破信号。建议调整 max_span_pct (放宽箱体定义) 或 window (缩短周期)。")
    else:
        print("未获取到数据。")