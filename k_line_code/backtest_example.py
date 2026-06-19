# backtest_example.py
# 示例回测脚本，演示如何使用 Backtrader 对已实现的形态识别进行回测

import backtrader as bt
from k_line_code.common.data_fetcher import fetch_stock_data
from k_line_code.ma_oscillator_pattern import identify_osc_patterns

class OscillatorStrategy(bt.Strategy):
    params = (('sma_period', 20),)

    def __init__(self):
        # 使用收盘价的移动平均线作为基准
        self.sma = bt.ind.SMA(self.datas[0].close, period=self.p.sma_period)
        # 记录信号
        self.signal = None

    def next(self):
        # 将 DataFrame 中的形态信号映射到回测框架
        # 这里假设已经在 DataFrame 中加入了 "Osc_Signal" 列，1 表示买入，-1 表示卖出
        # 为演示简化，仅使用收盘价与 SMA 的交叉
        if not self.position:
            if self.datas[0].close[0] > self.sma[0]:
                self.buy()
        else:
            if self.datas[0].close[0] < self.sma[0]:
                self.close()

if __name__ == "__main__":
    # 读取历史数据（示例使用贵州茅台）
    symbol = "600519"
    df = fetch_stock_data(symbol, days=365)
    # 计算形态信号（这里使用已有的指标函数）
    df = identify_osc_patterns(df)
    # 将 DataFrame 转换为 Backtrader 数据源
    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(OscillatorStrategy)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    print("起始资金: %.2f" % cerebro.broker.getvalue())
    cerebro.run()
    print("结束资金: %.2f" % cerebro.broker.getvalue())
    cerebro.plot(style='candlestick')
