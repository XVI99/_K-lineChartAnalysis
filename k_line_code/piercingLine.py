# -*- coding: utf-8 -*-
"""
刺透形态 (Piercing Line) 识别与可视化脚本
形态特征：
1. 出现在下跌趋势中。
2. 第一根为大阴线。
3. 第二根为阳线，低开（低于前一日最低价或收盘价），收盘价刺入前一日阴线实体的50%以上。
"""

import akshare as ak
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta

# ==========================================
# 1. 形态识别逻辑
# ==========================================

def piercing_line_strict(data: pd.DataFrame) -> pd.Series:
    """
    【严格版】刺透形态：
    1. 前阴（Close1 < Open1）
    2. 今阳（Close2 > Open2）
    3. 今开 < 前低（严格跳空低开）
    4. 今收 > 前实体 50%
    """
    df = data.copy()
    prev = df.shift(1)

    cond1 = prev['Close'] < prev['Open']           # 前阴
    cond2 = df['Close'] > df['Open']               # 今阳
    cond3 = df['Open'] < prev['Low']               # 今开 < 前低

    # 前实体中点
    mid_prev = (prev['Open'] + prev['Close']) / 2
    cond4 = df['Close'] > mid_prev                 # 今收 > 前实体 50%

    return cond1 & cond2 & cond3 & cond4


def piercing_line_enhanced(
        data: pd.DataFrame,
        ma_len: int = 20,
        use_trend_filter: bool = True,
        allow_soft_gap: bool = True,
        gap_tolerance_pct: float = 0.01,
        pierce_ratio: float = 0.5,
) -> pd.Series:
    """
    【实战增强版】刺透形态：
    - 在严格定义基础上加入：
      1) 趋势过滤：要求处于下跌趋势（Close < MA）
      2) 宽松缺口：允许今开略高于前低（可调 gap_tolerance_pct）
      3) 可调穿透比例 pierce_ratio（>0.5 更保守）
    """
    df = data.copy()
    prev = df.shift(1)

    # 1. 基础条件：前阴 / 今阳
    cond1 = prev['Close'] < prev['Open']
    cond2 = df['Close'] > df['Open']

    # 2. 缺口条件
    if allow_soft_gap:
        # 允许今开略高于前低（例如：虽然没创新低，但大幅低开在昨日收盘价之下也算，这里用 Low 做基准）
        # 逻辑：Open <= Prev_Low * (1 + 0.01)
        cond3 = df['Open'] <= prev['Low'] * (1 + gap_tolerance_pct)
    else:
        # 严格小于前低
        cond3 = df['Open'] < prev['Low']

    # 3. 穿透比例（默认 50%）
    # 注意：前一根是阴线，Top是Open，Bottom是Close。
    # 我们希望 Current Close > Open - (Entity * ratio)
    # 或者直接用数学公式：Open + (Close - Open) * ratio
    mid_prev = prev['Open'] + (prev['Close'] - prev['Open']) * pierce_ratio
    cond4 = df['Close'] > mid_prev

    # 4. 趋势过滤：下跌趋势中才认定为底部反转
    if use_trend_filter:
        ma = df['Close'].rolling(ma_len).mean()
        # 这里简单定义为收盘价在均线下方
        downtrend = df['Close'] < ma
    else:
        downtrend = pd.Series(True, index=df.index)

    return cond1 & cond2 & cond3 & cond4 & downtrend


# ==========================================
# 2. 主执行程序
# ==========================================

def main():
    # --- 参数配置 ---
    symbol_code = '600519'          # 贵州茅台
    days_back = 365

    # 策略参数
    strategy_params = {
        'ma_len': 20,
        'use_trend_filter': True,
        'allow_soft_gap': True,     # 开启宽松缺口模式，增加识别率
        'gap_tolerance_pct': 0.01,  # 容忍 1% 的偏差
        'pierce_ratio': 0.5         # 穿透 50%
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
            '日期': 'Date',
            '开盘': 'Open',
            '最高': 'High',
            '最低': 'Low',
            '收盘': 'Close',
            '成交量': 'Volume'
        })
        .loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        .assign(Date=lambda x: pd.to_datetime(x['Date']))
        .set_index('Date')
        .sort_index()
    )

    print("正在计算刺透形态...")

    # --- 2. 计算信号 ---
    # 这里使用增强版函数，如果想用严格版，可调用 piercing_line_strict(df)
    signal = piercing_line_enhanced(df, **strategy_params)
    df['Piercing'] = signal

    # 统计结果
    n_sig = signal.sum()
    print(f'刺透形态出现次数：{n_sig}')

    if n_sig > 0:
        print("最近 5 次信号详情：")
        print(df[signal][['Open', 'High', 'Low', 'Close']].tail())
    else:
        print('当前参数下无刺透形态，建议放宽 allow_soft_gap 或 gap_tolerance_pct 再试。')

    # --- 3. 可视化 ---
    print("正在绘图...")

    apds = []

    # 如果开启了趋势过滤，把均线也画上去辅助观察
    if strategy_params['use_trend_filter']:
        ma_col = df['Close'].rolling(strategy_params['ma_len']).mean()
        apds.append(mpf.make_addplot(ma_col, color='blue', width=1.0))

    # 绿色箭头标注信号
    if n_sig > 0:
        rng = (df['High'] - df['Low']).replace(0, 1e-6)
        # 标记位置在最低价下方一点点
        mark = np.where(signal, df['Low'] - rng * 0.15, np.nan)

        apds.append(
            mpf.make_addplot(
                mark,
                type='scatter',
                marker='^',
                markersize=80,
                color='green',
                label='Piercing'
            )
        )

    mpf.plot(
        df,
        type='candle',
        volume=True,
        addplot=apds,
        title=f'{symbol_code} Piercing Line Pattern', # 建议使用英文标题
        style='yahoo',
        figsize=(14, 8),
        tight_layout=True,
        block=True
    )
    print("完成。")

if __name__ == "__main__":
    main()