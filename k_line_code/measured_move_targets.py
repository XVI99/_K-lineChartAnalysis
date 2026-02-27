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
# 2. 目标价测算函数
# ==========================================

def calculate_equal_move(df, date_a, date_b, date_c, up=True):
    """
    计算对等运动目标 (Measured Move)
    A->B 为第一波趋势，B->C 为回调，C->D 为第二波趋势
    目标 D = C + (B - A)
    """
    try:
        # 获取价格
        if up:
            # 上涨趋势：A(低) -> B(高) -> C(低) -> D(高)
            price_a = df.loc[date_a]['Low']
            price_b = df.loc[date_b]['High']
            price_c = df.loc[date_c]['Low']

            height = price_b - price_a
            target = price_c + height
        else:
            # 下跌趋势：A(高) -> B(低) -> C(高) -> D(低)
            price_a = df.loc[date_a]['High']
            price_b = df.loc[date_b]['Low']
            price_c = df.loc[date_c]['High']

            height = price_a - price_b
            target = price_c - height

        return height, target

    except KeyError:
        print(f"日期错误: 请确保日期 {date_a}, {date_b}, {date_c} 在数据中存在。")
        return 0, 0

def calculate_flag_target(df, date_pole_start, date_pole_end, date_breakout, bullish=True):
    """
    计算旗形/尖旗形目标 (Flag / Pennant)
    目标 = 突破点 + 旗杆高度
    """
    try:
        if bullish:
            # 看涨旗形：旗杆从低到高
            price_start = df.loc[date_pole_start]['Low']
            price_end   = df.loc[date_pole_end]['High']
            # 突破点通常取旗形底边或突破时的低点（保守算法）
            price_break = df.loc[date_breakout]['Low']

            pole_height = price_end - price_start
            target = price_break + pole_height
        else:
            # 看跌旗形：旗杆从高到低
            price_start = df.loc[date_pole_start]['High']
            price_end   = df.loc[date_pole_end]['Low']
            price_break = df.loc[date_breakout]['High']

            pole_height = price_start - price_end
            target = price_break - pole_height

        return pole_height, target

    except KeyError:
        print("日期错误，请检查输入日期。")
        return 0, 0

def calculate_triangle_target(base_price, peak_price, up_triangle=True):
    """
    计算三角形突破目标
    上升三角形：目标 = 水平阻力 + (水平阻力 - 最低点)
    下降三角形：目标 = 水平支撑 - (最高点 - 水平支撑)
    """
    height = abs(base_price - peak_price)

    if up_triangle:
        target = base_price + height
    else:
        target = base_price - height

    return height, target

# ==========================================
# 3. 示例应用与绘图准备
# ==========================================
def apply_patterns(df):
    """
    这里模拟手动识别出的形态日期。
    在实际应用中，你需要根据肉眼观察或算法识别出的日期来填入。
    """
    df = df.copy()

    # --- 示例 1: 上涨对等运动 (假设数据) ---
    # 为了演示，我们取最近一段时间的高低点
    # 假设: A=100天前低点, B=60天前高点, C=30天前低点
    try:
        date_a = df.index[-60].strftime('%Y-%m-%d')
        date_b = df.index[-40].strftime('%Y-%m-%d')
        date_c = df.index[-20].strftime('%Y-%m-%d')

        h_eq, t_eq = calculate_equal_move(df, date_a, date_b, date_c, up=True)

        # 在图表上画出目标线 (从 C 点开始画)
        df['EqTarget'] = np.nan
        df.loc[date_c:, 'EqTarget'] = t_eq

        print(f"对等运动 (AB=CD): A({date_a}) -> B({date_b}) -> C({date_c})")
        print(f"波段高度: {h_eq:.2f}, 目标位: {t_eq:.2f}")

    except Exception as e:
        print(f"对等运动计算跳过: {e}")

    # --- 示例 2: 看跌旗形 (假设数据) ---
    # 假设: 旗杆起点=120天前, 终点=100天前, 突破点=90天前
    try:
        d_p_s = df.index[-120].strftime('%Y-%m-%d')
        d_p_e = df.index[-100].strftime('%Y-%m-%d')
        d_brk = df.index[-90].strftime('%Y-%m-%d')

        # 假设这是一段下跌
        if df.loc[d_p_s]['High'] > df.loc[d_p_e]['Low']:
            h_flag, t_flag = calculate_flag_target(df, d_p_s, d_p_e, d_brk, bullish=False)

            df['FlagTarget'] = np.nan
            df.loc[d_brk:, 'FlagTarget'] = t_flag

            print(f"看跌旗形: 杆始({d_p_s}) -> 杆终({d_p_e}) -> 突破({d_brk})")
            print(f"旗杆高度: {h_flag:.2f}, 目标位: {t_flag:.2f}")

    except Exception as e:
        print(f"旗形计算跳过: {e}")

    # --- 示例 3: 三角形 (假设数值) ---
    # 假设当前价格 1500，形成上升三角形，阻力 1550，低点 1450
    # h_tri, t_tri = calculate_triangle_target(1550, 1450, up_triangle=True)
    # df['TriTarget'] = t_tri  # 画一条水平线

    return df

# ==========================================
# 4. 可视化绘图函数
# ==========================================
def plot_targets(df, symbol):
    ap = []

    # 1. 对等运动目标 (橙色虚线)
    if 'EqTarget' in df.columns and not df['EqTarget'].isna().all():
        ap.append(mpf.make_addplot(df['EqTarget'], color='orange', linestyle='--', width=1.5))

    # 2. 旗形目标 (蓝色虚线)
    if 'FlagTarget' in df.columns and not df['FlagTarget'].isna().all():
        ap.append(mpf.make_addplot(df['FlagTarget'], color='dodgerblue', linestyle='--', width=1.5))

    # 3. 三角形目标 (紫色虚线) - 如果有的话
    if 'TriTarget' in df.columns and not df['TriTarget'].isna().all():
        ap.append(mpf.make_addplot(df['TriTarget'], color='purple', linestyle='--', width=1.5))

    print("正在生成图表...")
    mpf.plot(
        df,
        type='candle',
        volume=True,
        addplot=ap,
        style=my_style,
        title=f'{symbol} - 形态目标测算 (对等运动/旗形)',
        figsize=(14, 9)
    )

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    symbol_code = '600519'  # 贵州茅台

    # 1. 获取数据
    df = get_stock_data(symbol_code, days=365)

    if not df.empty:
        # 2. 应用测算逻辑 (注意：这里是演示用的自动选点，实战需手动指定日期)
        df_plot = apply_patterns(df)

        # 3. 绘图
        plot_targets(df_plot, symbol_code)
    else:
        print("未获取到数据。")