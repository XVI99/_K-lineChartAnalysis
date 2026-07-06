# -*- coding: utf-8 -*-
"""
build_sector_history.py — 板块轮动历史数据构建

Phase 1 优化：每日缓存行业板块涨跌幅排名，形成可回测的板块轮动历史。

数据存储: data_cache/history/sector_history.parquet
字段: date, sector, pct_change, volume, turnover_rank, phase

用法:
    python scripts/build_sector_history.py              # 抓取当日快照
    python scripts/build_sector_history.py --backfill   # 回填历史数据
    python scripts/build_sector_history.py --days 30     # 回填最近30天
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

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from AStockQuant.core.history_data_loader import HistoryDataLoader


def fetch_sector_daily(date_str: str) -> pd.DataFrame:
    """抓取当日行业板块行情（AkShare）"""
    try:
        import akshare as ak
    except ImportError:
        print("[build_sector] akshare 未安装")
        return pd.DataFrame()

    try:
        # 行业板块行情
        df = ak.stock_board_industry_name_em()
        if df is None or df.empty:
            return pd.DataFrame()

        result = pd.DataFrame()
        result["date"] = date_str

        # 板块名称
        for col in ["板块名称", "名称"]:
            if col in df.columns:
                result["sector"] = df[col].astype(str)
                break

        # 涨跌幅
        for col in ["涨跌幅", "涨幅"]:
            if col in df.columns:
                result["pct_change"] = df[col].astype(float)
                break

        # 成交量
        for col in ["成交量", "总成交量"]:
            if col in df.columns:
                result["volume"] = df[col].astype(float)
                break

        # 成交额排名
        for col in ["成交额", "总成交额"]:
            if col in df.columns:
                amounts = df[col].astype(float)
                result["turnover_rank"] = amounts.rank(ascending=False, method="min").astype(float)
                break

        if "sector" not in result.columns:
            return pd.DataFrame()

        return result
    except Exception as e:
        print(f"[build_sector] 板块行情抓取失败: {e}")
        return pd.DataFrame()


def save_to_parquet(parquet_path: str, new_df: pd.DataFrame):
    """追加数据到 Parquet 文件（去重）"""
    if new_df.empty:
        return 0

    # 读取已有数据
    if os.path.exists(parquet_path):
        try:
            existing = pd.read_parquet(parquet_path)
            # 合并并去重
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date", "sector"], keep="last")
            combined = combined.sort_values(["date", "sector"]).reset_index(drop=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df

    combined.to_parquet(parquet_path, index=False)
    return len(new_df)


def run_snapshot(date_str: Optional[str] = None):
    """执行当日快照抓取"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\n[build_sector] 开始抓取 {date_str} 板块行情...")

    loader = HistoryDataLoader.get_instance()
    parquet_path = loader.sector_parquet

    df = fetch_sector_daily(date_str)
    n = save_to_parquet(parquet_path, df)
    print(f"[build_sector] 保存 {n} 条板块记录到 {parquet_path}")


def run_backfill(days: int = 30):
    """回填历史数据"""
    print(f"\n[build_sector] 开始回填最近 {days} 天板块数据...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        if current.weekday() < 5:
            print(f"\n--- 回填 {date_str} ---")
            try:
                run_snapshot(date_str)
            except Exception as e:
                print(f"  失败: {e}")
            time.sleep(1)
        current += timedelta(days=1)

    print(f"\n[build_sector] 回填完成")


def main():
    parser = argparse.ArgumentParser(description="构建板块轮动历史数据")
    parser.add_argument("--backfill", action="store_true", help="回填历史数据")
    parser.add_argument("--days", type=int, default=30, help="回填天数")
    args = parser.parse_args()

    if args.backfill:
        run_backfill(args.days)
    else:
        run_snapshot()


if __name__ == "__main__":
    main()