# -*- coding: utf-8 -*-
"""
cache_etf_pcf.py — 从交易所官方获取 ETF 申购赎回清单（PCF）

数据源：
  - 上交所 (SSE): http://query.sse.com.cn/infodisplay/queryLatestETFListNew.do
  - 深交所 (SZSE): http://www.szse.cn/api/disc/announcement/ann
  - 中证指数公司: 通过 akshare 间接获取指数成分股权重

PCF (Portfolio Composition File) 是 ETF 申购赎回的核心数据文件：
  - 每日开盘前由交易所发布
  - 包含成分股代码/数量/现金替代标志/溢价比例
  - 可用于计算 IOPV（盘中估算净值）和折溢价套利边界

免费，无需 API Token。

用法:
  python scripts/cache_etf_pcf.py                    # 获取当日全部ETF PCF
  python scripts/cache_etf_pcf.py --codes 510300,159915  # 指定ETF
  python scripts/cache_etf_pcf.py --market sse       # 只获取上交所
  python scripts/cache_etf_pcf.py --market szse      # 只获取深交所
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "external_cache" / "pcf"


def _get_no_proxy_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": None, "https": None}
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    })
    return session


def fetch_sse_etf_list(session: requests.Session) -> pd.DataFrame:
    """获取上交所 ETF 列表及基本信息"""
    url = "http://query.sse.com.cn/infodisplay/queryLatestETFListNew.do"
    params = {
        "isPagination": "true",
        "pageNo": "1",
        "pageSize": "2000",
        "BANCODE": "",
        "stdisplay": "ETF",
    }
    headers = {"Referer": "http://www.sse.com.cn/disclosure/fund/etflist/"}
    try:
        resp = session.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("result", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        print(f"[PCF] SSE ETF list failed: {e}")
        return pd.DataFrame()


def fetch_sse_pcf(session: requests.Session, fund_code: str) -> Optional[dict]:
    """获取上交所单只 ETF 的 PCF 申购赎回清单"""
    url = "http://query.sse.com.cn/infodisplay/queryETFPCFInfo.do"
    params = {
        "isPagination": "false",
        "BANCODE": fund_code,
        "date": datetime.now().strftime("%Y-%m-%d"),
    }
    headers = {"Referer": "http://www.sse.com.cn/disclosure/fund/etflist/"}
    try:
        resp = session.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        print(f"[PCF] SSE PCF {fund_code} failed: {e}")
        return None


def fetch_szse_etf_list(session: requests.Session) -> pd.DataFrame:
    """获取深交所 ETF 列表（通过 akshare 间接获取）"""
    try:
        import akshare as ak
        df = ak.fund_etf_category_sina()
        if df is None or df.empty:
            return pd.DataFrame()
        sz_codes = df[df["代码"].astype(str).str.startswith("1")]
        return sz_codes
    except Exception as e:
        print(f"[PCF] SZSE ETF list failed: {e}")
        return pd.DataFrame()


def fetch_szse_pcf(session: requests.Session, fund_code: str) -> Optional[dict]:
    """获取深交所单只 ETF 的 PCF（通过 akshare fund_etf_fund_info_em 获取净值信息作为替代）"""
    try:
        import akshare as ak
        today = datetime.now().strftime("%Y%m%d")
        df = ak.fund_etf_fund_info_em(fund=fund_code, start_date=today, end_date=today)
        if df is None or df.empty:
            return None
        row = df.iloc[0].to_dict()
        return {
            "code": fund_code,
            "date": row.get("净值日期", ""),
            "unit_nav": float(row.get("单位净值", 0)),
            "cum_nav": float(row.get("累计净值", 0)),
            "purchase_status": str(row.get("申购状态", "")),
            "redeem_status": str(row.get("赎回状态", "")),
        }
    except Exception as e:
        print(f"[PCF] SZSE PCF {fund_code} failed: {e}")
        return None


def fetch_index_constituents(session: requests.Session, index_code: str) -> pd.DataFrame:
    """获取中证指数公司指数成分股权重（通过 akshare）"""
    try:
        import akshare as ak
        df = ak.index_stock_cons(symbol=index_code)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"[PCF] Index constituents {index_code} failed: {e}")
        return pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="获取交易所 ETF PCF 申购赎回清单")
    parser.add_argument("--codes", default="", help="逗号分隔的ETF代码，留空获取全部")
    parser.add_argument("--market", default="all", choices=["all", "sse", "szse"], help="交易所范围")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = _get_no_proxy_session()

    today_str = datetime.now().strftime("%Y%m%d")
    print("=" * 60)
    print("  ETF PCF 申购赎回清单获取器")
    print(f"  日期: {today_str}  交易所: {args.market}")
    print("=" * 60)

    all_results: List[dict] = []
    codes_filter = set()
    if args.codes.strip():
        codes_filter = {c.strip() for c in args.codes.split(",") if c.strip()}

    # ---- 上交所 ----
    if args.market in ("all", "sse"):
        print("\n[SSE] 获取上交所 ETF 列表...")
        sse_list = fetch_sse_etf_list(session)
        if not sse_list.empty:
            print(f"  上交所 ETF 数量: {len(sse_list)}")
            code_col = "BANCODE" if "BANCODE" in sse_list.columns else sse_list.columns[0]
            sse_codes = sse_list[code_col].astype(str).tolist()
            if codes_filter:
                sse_codes = [c for c in sse_codes if c in codes_filter]
            for i, code in enumerate(sse_codes[:50]):
                print(f"  [{i+1}/{len(sse_codes)}] SSE PCF {code}...", end=" ")
                pcf = fetch_sse_pcf(session, code)
                if pcf:
                    all_results.append({"market": "SSE", "code": code, "pcf": pcf})
                    print("OK")
                else:
                    print("FAIL")
                time.sleep(0.1)
        else:
            print("  上交所 ETF 列表为空")

    # ---- 深交所 ----
    if args.market in ("all", "szse"):
        print("\n[SZSE] 获取深交所 ETF 列表...")
        szse_list = fetch_szse_etf_list(session)
        if not szse_list.empty:
            print(f"  深交所 ETF 数量: {len(szse_list)}")
            szse_codes = szse_list["代码"].astype(str).tolist()
            if codes_filter:
                szse_codes = [c for c in szse_codes if c in codes_filter]
            for i, code in enumerate(szse_codes[:50]):
                print(f"  [{i+1}/{len(szse_codes)}] SZSE PCF {code}...", end=" ")
                pcf = fetch_szse_pcf(session, code)
                if pcf:
                    all_results.append({"market": "SZSE", "code": code, "pcf": pcf})
                    print("OK")
                else:
                    print("FAIL")
                time.sleep(0.15)
        else:
            print("  深交所 ETF 列表为空")

    # ---- 保存结果 ----
    print("\n" + "=" * 60)
    print(f"  获取完成: {len(all_results)} 只 ETF")
    print("=" * 60)

    if all_results:
        output_file = output_dir / f"etf_pcf_{today_str}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "date": today_str,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "count": len(all_results),
                    "data": all_results,
                    "warning": "PCF data from exchange official sources. Use for live review only.",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  结果已保存: {output_file}")
    else:
        print("  未获取到任何 PCF 数据")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
