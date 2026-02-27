# -*- coding: utf-8 -*-
"""
启明星 (Morning Star) 形态识别与可视化脚本
形态特征 (三根 K 线)：
1. 趋势：处于下降趋势中。
2. K1：长阴线。
3. K2：小实体（星线），向下跳空（或位置较低）。
4. K3：长阳线，收盘价深入 K1 实体内部。
"""

import akshare as ak
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta

# ==========================================
# 1. 核心识别逻辑
# ==========================================

def identify_morning_star(
        df: pd.DataFrame,
        ma_len: int = 20,
        body_small_ratio: float = 0.3,
        pierce_ratio: float = 0.5
) -> pd.DataFrame:
    """
    识别启明星形态

    参数:
    - ma_len: 均线周期，用于判断下跌趋势
    - body_small_ratio: 中间星线实体的最大比例（相对于全长）
    - pierce_ratio: 第三根阳线刺入第一根阴线的深度比例
    """
    # 避免修改原始 DataFrame
    data = df.copy()

    # 定义三根 K 线：Prev2(K1), Prev1(K2), Curr(K3)
    prev2 = data.shift(2)   # K1: 前前日
    prev1 = data.shift(1)   # K2: 前日
    curr  = data            # K3: 今日

    # --- 1. 趋势判断 ---
    # 使用 K1 时刻的收盘价与当时的 MA 比较
    # 注意：这里需要计算完整的 rolling 序列，再进行 shift 比较
    ma = data['Close'].rolling(ma_len).mean()
    # 逻辑：K1 发生时，收盘价 < 均线
    downtrend = prev2['Close'] < ma.shift(2)

    # --- 预计算实体与振幅 ---
    # 防止除以 0，将 0 替换为 nan
    def get_body_range(d):
        body = (d['Close'] - d['Open']).abs()
        rng = (d['High'] - d['Low']).replace(0, np.nan)
        return body, rng

    body1, range1 = get_body_range(prev2)
    body2, range2 = get_body_range(prev1)
    body3, range3 = get_body_range(curr)

    # --- 2. K1: 长阴线 ---
    # 收阴 & 实体占整根 K 线的一半以上
    cond1 = (prev2['Close'] < prev2['Open']) & ((body1 / range1) > 0.5)

    # --- 3. K2: 小实体星线 ---
    # 实体较小 (body <= range * 0.3)
    cond2_small = (body2 / range2) <= body_small_ratio
    # 位置较低 (这里使用宽松条件：K2 最高价 < K1 最高价，或者 K2 处于 K1 下半部分)
    # A股很难完全跳空，这里沿用你的逻辑：High2 < High1
    cond2_pos = prev1['High'] < prev2['High']
    cond2 = cond2_small & cond2_pos

    # --- 4. K3: 长阳线 & 深入实体 ---
    # 收阳
    cond3_bull = curr['Close'] > curr['Open']
    # 实体较长
    cond3_long = (body3 / range3) > 0.5
    # 深入 K1 实体的程度
    # K1 是阴线，Top=Open, Bottom=Close。深入计算：Close3 > Open1 + (Close1 - Open1) * ratio
    # 注意：(Close1 - Open1) 是负数，所以是 Open 向下减
    mid1 = prev2['Open'] + (prev2['Close'] - prev2['Open']) * pierce_ratio
    cond3_pierce = curr['Close'] > mid1

    cond3 = cond3_bull & cond3_long & cond3_pierce

    # --- 综合结果 ---
    data['Morning_Star'] = downtrend & cond1 & cond2 & cond3

    # 将 MA 存入 data 以便画图
    data['MA'] = ma

    return data


# ==========================================
# 2. 主执行程序
# ==========================================

def main():
    # --- 参数配置 ---
    symbol_code = '600519'          # 贵州茅台
    days_back = 365

    # 策略灵敏度参数
    strategy_params = {
        'ma_len': 20,
        'body_small_ratio': 0.3,    # 星线实体不能超过 K 线全长的 30%
        'pierce_ratio': 0.5         # 必须反弹回第一根阴线的 50% 以上
    }

    print(f"正在获取 {symbol_code} 的历史数据...")

    # --- 1. 获取数据 ---
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days_back)

    try:
        df_raw = ak.stock_zh_a_hist(
            symbol=symbol_code,
            period='daily',
            start_date=start_dt.strftime('%Y%m%d'),
            end_date=end_dt.strftime('%Y%m%d'),
            adjust='qfq'
        )
    except Exception as e:
        print(f"数据获取失败: {e}")
        return

    if df_raw.empty:
        print("未获取到数据，请检查代码或日期。")
        return

    # 数据清洗
    df = (
        df_raw
        .rename(columns={
            '日期': 'Date', '开盘': 'Open', '最高': 'High',
            '最低': 'Low', '收盘': 'Close', '成交量': 'Volume'
        })
        .loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        .assign(Date=lambda x: pd.to_datetime(x['Date']))
        .set_index('Date')
        .sort_index()
    )

    print("正在识别启明星形态...")

    # --- 2. 识别形态 ---
    df = identify_morning_star(df, **strategy_params)

    # 统计结果
    n_sig = df['Morning_Star'].sum()
    print(f'启明星形态出现次数：{n_sig}')

    if n_sig > 0:
        print("最近 3 次信号详情：")
        print(df[df['Morning_Star']][['Open', 'High', 'Low', 'Close']].tail(3))
    else:
        print('当前参数下未发现启明星形态，建议放宽 body_small_ratio 或 pierce_ratio。')

    # --- 3. 可视化 ---
    print("正在绘图...")

    apds = []

    # 1. 绘制均线 (辅助查看趋势)
    apds.append(mpf.make_addplot(df['MA'], color='blue', width=1.0))

    # 2. 绘制信号箭头
    if n_sig > 0:
        rng = (df['High'] - df['Low']).replace(0, 1e-6)
        # 标记位置：在最低价下方 15% 处
        mark = np.where(df['Morning_Star'],
                        df['Low'] - rng * 0.15,
                        np.nan)

        apds.append(
            mpf.make_addplot(
                mark,
                type='scatter',
                marker='^',
                markersize=100,
                color='red',        # 启明星通常看涨，用红色突出
                label='Morning Star'
            )
        )

    mpf.plot(
        df,
        type='candle',
        volume=True,
        addplot=apds,
        title=f'{symbol_code} Morning Star Pattern', # 建议用英文标题
        style='yahoo',
        figsize=(14, 8),
        tight_layout=True,
        block=True
    )
    print("完成。")

if __name__ == "__main__":
    main()