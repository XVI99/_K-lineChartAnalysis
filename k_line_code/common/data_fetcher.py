"""公共数据获取模块

提供统一的股票历史数据获取函数，使用 akshare 如可用，若不可用返回空 DataFrame。
"""
import pandas as pd
import datetime

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


def fetch_stock_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """获取指定股票的历史行情数据

    参数:
        symbol: 股票代码，例如 '600519'
        days:   向前获取的天数
    返回:
        包含列 ['open', 'high', 'low', 'close', 'volume']，索引为 datetime 的 DataFrame。
    """
    if not AKSHARE_AVAILABLE:
        # akshare 未安装，返回空 DataFrame，调用方可自行处理 fallback
        return pd.DataFrame()
    try:
        end_dt = datetime.datetime.now()
        start_dt = end_dt - datetime.timedelta(days=days)
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_dt.strftime("%Y%m%d"),
            adjust="qfq",
        )
        if df.empty:
            return pd.DataFrame()
        df = df.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"])  # type: ignore[arg-type]
        df.set_index("date", inplace=True)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()
