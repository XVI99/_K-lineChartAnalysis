import akshare as ak
import pandas as pd
import time
import logging
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ========== 日志 ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ========== 主板前缀判断 ==========
SH_MAIN = ("600", "601", "603", "605")
SZ_MAIN = ("000", "001", "002", "003")

def is_mainboard(symbol: str) -> bool:
    """新人默认可交易的主板"""
    return symbol.startswith(SH_MAIN) or symbol.startswith(SZ_MAIN)

# ========== 策略函数（假设已写好） ==========
try:
    from k_line_code.kdj_analysis import get_stock_data, calculate_kdj, identify_kdj_signals
    from k_line_code.stochastic_pattern import calculate_stochastic, identify_stochastic_patterns
    from new_strategy import build_combined_signals
except ImportError as e:
    logging.error(f"策略模块导入失败: {e}")
    exit(1)

# ========== 1. 取主板列表 ==========
def get_mainboard_list():
    """返回 [(代码, 名称), ...] 只含主板"""
    logging.info("正在获取全市场实时行情...")
    df = ak.stock_zh_a_spot_em()
    df = df[df["最新价"] > 0]          # 去掉停牌或退市
    df = df[df["代码"].apply(is_mainboard)]
    return list(zip(df["代码"], df["名称"]))

# ========== 2. 单票扫描 ==========
def analyze_single_stock(code: str, name: str):
    try:
        df = get_stock_data(code, days=300)
        if df.empty or len(df) < 50:
            return None

        df = calculate_kdj(df)
        df = identify_kdj_signals(df)

        df_stoch = calculate_stochastic(df)
        df_stoch = identify_stochastic_patterns(df_stoch)

        df_all = df.join(df_stoch[["%K", "%D", "TopSignal", "BottomSignal"]]
                         .rename(columns={"%K": "Stoch_K", "%D": "Stoch_D"}))
        df_all = build_combined_signals(df_all)

        last = df_all.iloc[-1]
        if last.get("StrongLong"):
            sig = "强做多"
        elif last.get("StrongShort"):
            sig = "强做空"
        else:
            return None

        return dict(代码=code, 名称=name,
                    日期=last.name.strftime("%Y-%m-%d"),
                    信号类型=sig, 收盘价=round(last["Close"], 2),
                    KDJ_J=round(last["J"], 2),
                    Stoch_K=round(last["Stoch_K"], 2))

    except Exception as e:
        logging.debug(f"{code} 分析失败: {e}")
        return None

# ========== 3. 批量扫描 ==========
if __name__ == "__main__":
    stocks = get_mainboard_list()
    logging.info(f"主板共 {len(stocks)} 只")

    results = []
    for code, name in tqdm(stocks, desc="扫描中"):
        res = analyze_single_stock(code, name)
        if res:
            results.append(res)
            logging.info(f"发现信号: {code} {name}  {res['信号类型']}")
        # time.sleep(0.)          # 控制频率，可按需调整或改用线程池
    # 保存
    if results:
        fname = f"mainboard_signals_{datetime.now():%Y%m%d_%H%M}.csv"
        pd.DataFrame(results).to_csv(fname, index=False, encoding="utf-8-sig")
        logging.info(f"结果已保存至 {fname}，共 {len(results)} 只股票")
    else:
        logging.info("今日主板无共振信号")