# -*- coding: utf-8 -*-
"""
build_sentiment_history.py — 新闻情绪历史数据构建

Phase 1 优化：每日调用 LLM 生成全市场情绪快照并存储，形成可回测的情绪历史。

数据存储: data_cache/history/sentiment_history.parquet
字段: date, symbol, sentiment_score, confidence, summary

用法:
    python scripts/build_sentiment_history.py              # 抓取当日快照
    python scripts/build_sentiment_history.py --symbols 510300,159915  # 指定标的
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from AStockQuant.core.history_data_loader import HistoryDataLoader


def fetch_sentiment_for_symbol(symbol: str, date_str: str) -> Dict:
    """获取单只标的的情绪分析

    优先使用 LLM NewsSentimentAnalyzer，失败时回退到基于价量的简单情绪。
    """
    # 尝试 LLM 情绪分析
    try:
        from AStockQuant.llm.news_sentiment import NewsSentimentAnalyzer
        analyzer = NewsSentimentAnalyzer()

        # 获取相关新闻
        news_list = analyzer.fetch_news_for_symbol(symbol)
        if news_list:
            result = analyzer.analyze_sentiment(symbol, news_list)
            return {
                "date": date_str,
                "symbol": symbol,
                "sentiment_score": float(result.get("score", 0.5)),
                "confidence": float(result.get("confidence", 0.0)),
                "summary": str(result.get("summary", ""))[:500],
            }
    except Exception as e:
        print(f"  [sentiment] LLM 分析失败 {symbol}: {e}")

    # 回退：基于近期价量的简单情绪
    try:
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data_cache"
        )
        csv_path = os.path.join(cache_dir, f"{symbol}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if len(df) >= 5:
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                ret_5d = df["close"].pct_change(5).iloc[-1]
                score = 1.0 / (1.0 + np.exp(-np.clip(ret_5d * 20, -50, 50)))
                return {
                    "date": date_str,
                    "symbol": symbol,
                    "sentiment_score": float(score),
                    "confidence": 0.3,
                    "summary": f"基于5日收益率 {ret_5d:.2%} 的简单情绪",
                }
    except Exception:
        pass

    return {
        "date": date_str,
        "symbol": symbol,
        "sentiment_score": 0.5,
        "confidence": 0.0,
        "summary": "",
    }


def save_to_parquet(parquet_path: str, new_df: pd.DataFrame):
    """追加数据到 Parquet 文件（去重）"""
    if new_df.empty:
        return 0

    if os.path.exists(parquet_path):
        try:
            existing = pd.read_parquet(parquet_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date", "symbol"], keep="last")
            combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df

    combined.to_parquet(parquet_path, index=False)
    return len(new_df)


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
    """执行当日情绪快照抓取"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\n[build_sentiment] 开始抓取 {date_str} 情绪数据...")

    loader = HistoryDataLoader.get_instance()
    parquet_path = loader.sentiment_parquet

    if symbols is None:
        symbols = get_etf_symbols(loader)
    print(f"[build_sentiment] ETF列表: {len(symbols)} 只")

    rows = []
    for i, sym in enumerate(symbols):
        result = fetch_sentiment_for_symbol(sym, date_str)
        rows.append(result)
        if (i + 1) % 10 == 0:
            print(f"  情绪进度: {i+1}/{len(symbols)}")
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    n = save_to_parquet(parquet_path, df)
    print(f"[build_sentiment] 保存 {n} 条情绪记录到 {parquet_path}")


def main():
    parser = argparse.ArgumentParser(description="构建新闻情绪历史数据")
    parser.add_argument("--symbols", type=str, help="指定标的（逗号分隔）")
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else None
    run_snapshot(symbols=symbols)


if __name__ == "__main__":
    main()