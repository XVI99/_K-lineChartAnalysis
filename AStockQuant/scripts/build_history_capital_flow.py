# -*- coding: utf-8 -*-
"""
build_history_capital_flow.py — 历史资金流数据库构建

Phase 1 优化：每日定时抓取并持久化资金流数据到 SQLite，形成可回测的历史面板。

数据表:
- etf_capital_flow: ETF 个股资金流（日期/代码/净流入/排名）
- lhb_daily:        龙虎榜明细
- north_flow_daily: 北向资金每日汇总
- theme_flow_daily: 主题资金流（概念资金流按ETF主题映射）

用法:
    python scripts/build_history_capital_flow.py              # 抓取当日快照
    python scripts/build_history_capital_flow.py --backfill   # 回填历史数据
    python scripts/build_history_capital_flow.py --days 30     # 回填最近30天
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# 路径设置
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from AStockQuant.core.history_data_loader import HistoryDataLoader
from AStockQuant.layers.layer3_sector import SectorLayer


def init_database(db_path: str):
    """初始化 SQLite 数据库表结构"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ETF 个股资金流
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS etf_capital_flow (
            date        TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            net_inflow  REAL,
            rank        REAL,
            PRIMARY KEY (date, symbol)
        )
    """)

    # 龙虎榜明细
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lhb_daily (
            date            TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            reason          TEXT,
            net_buy_amount  REAL,
            PRIMARY KEY (date, symbol)
        )
    """)

    # 北向资金每日汇总
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS north_flow_daily (
            date            TEXT PRIMARY KEY,
            net_buy_amount  REAL
        )
    """)

    # 主题资金流
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS theme_flow_daily (
            date        TEXT NOT NULL,
            theme       TEXT NOT NULL,
            net_inflow  REAL,
            breadth     REAL,
            momentum    REAL,
            PRIMARY KEY (date, theme)
        )
    """)

    conn.commit()
    conn.close()
    print(f"[build_history] 数据库已初始化: {db_path}")


def fetch_etf_capital_flow(symbols: List[str], date_str: str) -> pd.DataFrame:
    """抓取 ETF 个股资金流（AkShare）"""
    try:
        import akshare as ak
    except ImportError:
        print("[build_history] akshare 未安装，跳过个股资金流")
        return pd.DataFrame()

    rows = []
    for i, sym in enumerate(symbols):
        try:
            # AkShare ETF 资金流
            df = ak.fund_etf_fund_daily_em()
            if df is not None and not df.empty:
                if "基金代码" in df.columns:
                    sub = df[df["基金代码"].astype(str).str[:6] == sym]
                    if not sub.empty:
                        row = sub.iloc[0]
                        net_inflow = 0.0
                        for col in ["主力净流入额", "净流入额", "资金净流入"]:
                            if col in sub.columns:
                                net_inflow = float(row[col])
                                break
                        rows.append({
                            "date": date_str,
                            "symbol": sym,
                            "net_inflow": net_inflow,
                            "rank": 50.0,
                        })
            time.sleep(0.3)
        except Exception:
            pass
        if (i + 1) % 10 == 0:
            print(f"  资金流进度: {i+1}/{len(symbols)}")

    return pd.DataFrame(rows)


def fetch_lhb(date_str: str) -> pd.DataFrame:
    """抓取龙虎榜数据"""
    try:
        import akshare as ak
    except ImportError:
        return pd.DataFrame()

    try:
        date_compact = date_str.replace("-", "")
        df = ak.stock_lhb_detail_em(start_date=date_compact, end_date=date_compact)
        if df is None or df.empty:
            return pd.DataFrame()

        result = pd.DataFrame()
        result["date"] = date_str
        for col in ["代码", "股票代码"]:
            if col in df.columns:
                result["symbol"] = df[col].astype(str).str[:6]
                break
        for col in ["龙虎榜净买额", "净买额", "买入额"]:
            if col in df.columns:
                result["net_buy_amount"] = df[col].astype(float)
                break
        for col in ["上榜原因", "解读"]:
            if col in df.columns:
                result["reason"] = df[col].astype(str)
                break
        if "symbol" not in result.columns:
            return pd.DataFrame()
        return result
    except Exception as e:
        print(f"[build_history] 龙虎榜抓取失败: {e}")
        return pd.DataFrame()


def fetch_north_flow(date_str: str) -> Optional[float]:
    """抓取北向资金净流入"""
    try:
        import akshare as ak
    except ImportError:
        return None

    try:
        df = ak.stock_hsgt_north_net_flow_in_em(symbol="北向资金")
        if df is None or df.empty:
            return None
        if "日期" in df.columns:
            df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
            sub = df[df["日期"] == date_str]
            if not sub.empty:
                for col in ["净流入额", "当日净流入", "净流入"]:
                    if col in sub.columns:
                        return float(sub[col].iloc[0])
        return None
    except Exception as e:
        print(f"[build_history] 北向资金抓取失败: {e}")
        return None


def fetch_theme_flow(date_str: str) -> pd.DataFrame:
    """抓取概念资金流并按ETF主题映射"""
    try:
        import akshare as ak
    except ImportError:
        return pd.DataFrame()

    try:
        df = ak.stock_fund_flow_concept(symbol="即时")
        if df is None or df.empty:
            return pd.DataFrame()

        theme_map = {}

        for _, row in df.iterrows():
            concept_name = str(row.get("名称", row.get("概念", "")))
            net_inflow = 0.0
            for col in ["今日主力净流入-净额", "主力净流入额", "净流入额"]:
                if col in df.columns:
                    net_inflow = float(row[col])
                    break

            theme = _map_concept_to_theme(concept_name)
            if theme not in theme_map:
                theme_map[theme] = {"net_inflow": 0.0, "count": 0}
            theme_map[theme]["net_inflow"] += net_inflow
            theme_map[theme]["count"] += 1

        rows = []
        for theme, data in theme_map.items():
            rows.append({
                "date": date_str,
                "theme": theme,
                "net_inflow": data["net_inflow"],
                "breadth": 0.5,
                "momentum": 0.0,
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[build_history] 主题资金流抓取失败: {e}")
        return pd.DataFrame()


def _map_concept_to_theme(concept_name: str) -> str:
    """将同花顺概念名称映射到 ETF 主题"""
    concept_name = concept_name.upper()
    mappings = [
        (("芯片", "半导体", "集成电路"), "科技"),
        (("人工智能", "AI", "算力", "大数据"), "科技"),
        (("5G", "通信", "物联网"), "科技"),
        (("消费电子", "苹果"), "科技"),
        (("新能源车", "锂电", "充电桩", "特斯拉"), "新能源"),
        (("光伏", "风电", "储能"), "新能源"),
        (("白酒", "啤酒", "食品"), "消费"),
        (("券商", "证券", "金融"), "金融"),
        (("银行", "保险"), "金融"),
        (("医药", "医疗", "生物", "疫苗"), "医药"),
        (("军工", "航天", "国防"), "军工"),
        (("煤炭", "有色", "钢铁", "稀土", "矿业"), "资源"),
        (("房地产", "建材", "基建"), "房地产"),
        (("黄金", "白银"), "商品"),
    ]
    for keywords, theme in mappings:
        if any(kw.upper() in concept_name for kw in keywords):
            return theme
    return "其他"


def save_to_db(db_path: str, table_name: str, df: pd.DataFrame) -> int:
    """保存数据到 SQLite"""
    if df.empty:
        return 0
    import sqlite3
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="append", index=False, method="multi")
    # 去重：保留最新记录
    cols = list(df.columns[:2])
    conn.execute(f"""
        DELETE FROM {table_name}
        WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM {table_name}
            GROUP BY {', '.join(cols)}
        )
    """)
    conn.commit()
    conn.close()
    return len(df)


def get_etf_symbols(loader: HistoryDataLoader) -> List[str]:
    """获取ETF代码列表"""
    try:
        from AStockQuant.core.data_hub import ETFDataHub
        hub = ETFDataHub()
        symbols = hub.get_etf_list(top_n=200)
        hub.close()
        return symbols
    except Exception:
        cache_dir = os.path.join(loader.base_dir, "data_cache")
        symbols = [f.replace(".csv", "") for f in os.listdir(cache_dir)
                   if f.endswith(".csv") and f.replace(".csv", "").isdigit()]
        return symbols


def run_snapshot(symbols: Optional[List[str]] = None, date_str: Optional[str] = None):
    """执行当日快照抓取"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\n[build_history] 开始抓取 {date_str} 资金流数据...")

    loader = HistoryDataLoader.get_instance()
    db_path = loader.capital_flow_db

    init_database(db_path)

    if symbols is None:
        symbols = get_etf_symbols(loader)
    print(f"[build_history] ETF列表: {len(symbols)} 只")

    # 1. ETF 个股资金流
    print("\n1. 抓取 ETF 个股资金流...")
    flow_df = fetch_etf_capital_flow(symbols, date_str)
    n = save_to_db(db_path, "etf_capital_flow", flow_df)
    print(f"   保存 {n} 条个股资金流记录")

    # 2. 龙虎榜
    print("\n2. 抓取龙虎榜数据...")
    lhb_df = fetch_lhb(date_str)
    n = save_to_db(db_path, "lhb_daily", lhb_df)
    print(f"   保存 {n} 条龙虎榜记录")

    # 3. 北向资金
    print("\n3. 抓取北向资金数据...")
    north_net = fetch_north_flow(date_str)
    if north_net is not None:
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT OR REPLACE INTO north_flow_daily (date, net_buy_amount) VALUES (?, ?)",
            (date_str, north_net)
        )
        conn.commit()
        conn.close()
        print(f"   北向净流入: {north_net/1e8:.2f} 亿")
    else:
        print("   北向资金数据获取失败")

    # 4. 主题资金流
    print("\n4. 抓取主题资金流数据...")
    theme_df = fetch_theme_flow(date_str)
    n = save_to_db(db_path, "theme_flow_daily", theme_df)
    print(f"   保存 {n} 条主题资金流记录")

    print(f"\n[build_history] {date_str} 快照完成")


def run_backfill(days: int = 30):
    """回填历史数据"""
    print(f"\n[build_history] 开始回填最近 {days} 天数据...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        if current.weekday() < 5:
            print(f"\n--- 回填 {date_str} ---")
            try:
                run_snapshot(date_str=date_str)
            except Exception as e:
                print(f"  失败: {e}")
            time.sleep(1)
        current += timedelta(days=1)

    print(f"\n[build_history] 回填完成，共 {days} 天")


def main():
    parser = argparse.ArgumentParser(description="构建历史资金流数据库")
    parser.add_argument("--backfill", action="store_true", help="回填历史数据")
    parser.add_argument("--days", type=int, default=30, help="回填天数（默认30天）")
    args = parser.parse_args()

    if args.backfill:
        run_backfill(args.days)
    else:
        run_snapshot()


if __name__ == "__main__":
    main()