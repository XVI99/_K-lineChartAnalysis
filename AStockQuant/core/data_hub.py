# -*- coding: utf-8 -*-
"""
data_hub.py — 数据中枢（股票 + ETF）

整合 akshare / 腾讯API / 新浪API / 东方财富API 四数据源，
支持在线实时拉取 + 离线 CSV 回测双模式。
自动识别股票/ETF/指数类型，调用对应 API。

新增功能（v2）：
- get_stock_list(): 获取 A 股主板股票列表（沪A + 深A）
- get_broad_etf_universe(): 宽基 ETF 标的池构建（基于 universe_builder.py）
- classify_etf_type(): ETF 类型分类（宽基/行业/跨境/债券）
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from AStockQuant.core.cache_manager import CacheManager

try:
    import akshare as ak
    _AKSHARE = True
except ImportError:
    _AKSHARE = False

# ==================== Disable System Proxy Session ====================

def _get_no_proxy_session():
    session = __import__("requests").Session()
    session.trust_env = False
    session.proxies = {"http": None, "https": None}
    return session

_no_proxy_session = None

def _request_get(url, timeout=15, **kwargs):
    global _no_proxy_session
    if _no_proxy_session is None:
        _no_proxy_session = _get_no_proxy_session()
    return _no_proxy_session.get(url, timeout=timeout, **kwargs)




ETF_SH_PREFIXES = (
    "510", "511", "512", "513", "515", "516", "517", "518",
    "560", "561", "562", "563", "588",
)
ETF_SZ_PREFIXES = ("159",)
DEFAULT_ETF_PREFIXES = ETF_SH_PREFIXES + ETF_SZ_PREFIXES


def _normalize_symbol(symbol: str) -> str:
    code = str(symbol or "").strip()
    code = code.replace(".XSHG", "").replace(".XSHE", "")
    code = code.replace(".SH", "").replace(".SZ", "")
    return re.sub(r"^(sh|sz)", "", code, flags=re.IGNORECASE)


def _is_etf_symbol(symbol: str) -> bool:
    code = _normalize_symbol(symbol)
    return code.startswith(DEFAULT_ETF_PREFIXES)


def _get_market_id(symbol: str) -> int:
    code = _normalize_symbol(symbol)
    if code.startswith(("5", "6", "9", "11")):
        return 1
    return 0


def _to_eastmoney_secid(symbol: str) -> str:
    code = _normalize_symbol(symbol)
    return f"{_get_market_id(code)}.{code}"


def _pick_existing_column(df: pd.DataFrame, candidates) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    column_aliases = {
        "date": ("date", "\u65e5\u671f"),
        "open": ("open", "\u5f00\u76d8"),
        "high": ("high", "\u6700\u9ad8"),
        "low": ("low", "\u6700\u4f4e"),
        "close": ("close", "\u6536\u76d8"),
        "volume": ("volume", "\u6210\u4ea4\u91cf"),
    }

    rename_map = {}
    for target, aliases in column_aliases.items():
        source = _pick_existing_column(df, aliases)
        if source and source != target:
            rename_map[source] = target
    if rename_map:
        df = df.rename(columns=rename_map)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date")
    else:
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df.index.name = "date"

    keep_cols = [col for col in ("open", "high", "low", "close", "volume") if col in df.columns]
    if len(keep_cols) < 5:
        return pd.DataFrame()

    for col in keep_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[keep_cols].dropna().sort_index()


def _normalize_quote_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    canonical_columns = {
        "\u4ee3\u7801": ("code", "\u4ee3\u7801", "\u8bc1\u5238\u4ee3\u7801", "\u57fa\u91d1\u4ee3\u7801"),
        "\u540d\u79f0": ("name", "\u540d\u79f0", "\u8bc1\u5238\u540d\u79f0", "\u57fa\u91d1\u540d\u79f0"),
        "\u6700\u65b0\u4ef7": ("latest_price", "\u6700\u65b0\u4ef7", "\u6700\u65b0"),
        "\u6da8\u8dcc\u5e45": ("change_pct", "\u6da8\u8dcc\u5e45"),
        "\u6da8\u8dcc\u989d": ("change_amount", "\u6da8\u8dcc\u989d"),
        "\u6210\u4ea4\u91cf": ("volume", "\u6210\u4ea4\u91cf"),
        "\u6210\u4ea4\u989d": ("amount", "\u6210\u4ea4\u989d"),
        "\u4eca\u5f00": ("open", "\u4eca\u5f00"),
        "\u6700\u9ad8": ("high", "\u6700\u9ad8"),
        "\u6700\u4f4e": ("low", "\u6700\u4f4e"),
        "\u6628\u6536": ("pre_close", "\u6628\u6536"),
    }

    rename_map = {}
    for target, aliases in canonical_columns.items():
        source = _pick_existing_column(df, aliases)
        if source and source != target:
            rename_map[source] = target
    if rename_map:
        df = df.rename(columns=rename_map)

    code_col = "\u4ee3\u7801"
    if code_col not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df[code_col] = df[code_col].astype(str).map(_normalize_symbol)

    numeric_cols = [
        "\u6700\u65b0\u4ef7", "\u6da8\u8dcc\u5e45", "\u6da8\u8dcc\u989d",
        "\u6210\u4ea4\u91cf", "\u6210\u4ea4\u989d",
        "\u4eca\u5f00", "\u6700\u9ad8", "\u6700\u4f4e", "\u6628\u6536",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ordered_cols = [col for col in canonical_columns if col in df.columns]
    return df[ordered_cols].drop_duplicates(subset=[code_col]).reset_index(drop=True)


def _rate_limit(interval: float = 1.5, batch_limit: int = 10, batch_sleep: float = 5):
    pass


class DataHub:
    """数据中枢 — 股票/ETF/指数四源降级 + 在线/离线双模式"""

    def __init__(
        self,
        use_cache: bool = True,
        cache_expire: int = 3600,
        current_date: Optional[str] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        mode: str = "online",
        local_dir: str = None,
    ):
        self._cache = CacheManager(
            host=redis_host, port=redis_port, db=redis_db,
            expire_seconds=cache_expire,
        ) if use_cache else None
        self.current_date = current_date
        self.mode = mode
        self.local_dir = local_dir

    def set_date(self, date_str: str):
        self.current_date = date_str
        if self._cache:
            self._cache.clear()

    def set_offline_mode(self, local_dir: str):
        self.mode = "offline"
        self.local_dir = local_dir

    def load_local_data(self, symbol: str, date: str = None) -> pd.DataFrame:
        if not self.local_dir:
            return pd.DataFrame()
        csv_path = os.path.join(self.local_dir, f"{symbol}.csv")
        if not os.path.exists(csv_path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(csv_path)
            if date and "date" in df.columns:
                df = df[df["date"] == date]
            return df
        except Exception:
            return pd.DataFrame()

    def load_local_data_full(self, symbol: str) -> pd.DataFrame:
        if not self.local_dir:
            return pd.DataFrame()
        csv_path = os.path.join(self.local_dir, f"{symbol}.csv")
        if not os.path.exists(csv_path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(csv_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            return df.sort_index()
        except Exception:
            return pd.DataFrame()

    def load_local_data_for_date(
        self, symbols: List[str], date: str
    ) -> Dict[str, dict]:
        result = {}
        for symbol in symbols:
            df = self.load_local_data(symbol)
            if df.empty:
                continue
            if date and "date" in df.columns:
                row = df[df["date"] == date]
                if row.empty:
                    continue
                row = row.iloc[0]
            else:
                row = df.iloc[-1]
            result[symbol] = {
                "close": float(row.get("close", 0)),
                "volume": int(row.get("volume", 0)),
                "pct_change": float(row.get("pct_change", 0)),
                "open": float(row.get("open", row.get("close", 0))),
                "high": float(row.get("high", row.get("close", 0))),
                "low": float(row.get("low", row.get("close", 0))),
            }
        return result

    def get_etf_list(
        self, top_n: int = 0, prefixes: Optional[tuple] = None
    ) -> List[str]:
        prefixes = prefixes or DEFAULT_ETF_PREFIXES

        if _AKSHARE:
            for attempt in range(3):
                try:
                    df = ak.fund_etf_spot_em()
                    code_col = _pick_existing_column(df, ("\u4ee3\u7801", "code"))
                    if not code_col:
                        raise KeyError("ETF quote dataframe missing code column")
                    all_codes = (
                        df[code_col].astype(str).map(_normalize_symbol).tolist()
                    )
                    all_codes = [c for c in all_codes if c.startswith(prefixes)]
                    all_codes = list(dict.fromkeys(all_codes))
                    return all_codes[:top_n] if top_n > 0 else all_codes
                except Exception as e:
                    print(f"[DataHub] akshare ETF list failed ({attempt + 1}/3): {e}")
                    time.sleep(1)

        try:
            all_codes = self._fetch_eastmoney_etf_list()
            all_codes = [c for c in all_codes if c.startswith(prefixes)]
            all_codes = list(dict.fromkeys(all_codes))
            if all_codes:
                return all_codes[:top_n] if top_n > 0 else all_codes
        except Exception as e:
            print(f"[DataHub] Eastmoney ETF list failed: {e}")

        raise RuntimeError("All ETF universe providers are unavailable.")

    def get_stock_data_ex(
        self, symbol: str, days: int = 500, adjust: str = "qfq"
    ) -> pd.DataFrame:
        cache_key = f"etf_ex_{symbol}_{days}_{adjust}_{self.current_date or 'latest'}"
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        df = pd.DataFrame()
        if _AKSHARE:
            try:
                end_dt = datetime.now()
                start_dt = end_dt - timedelta(days=days)
                code = _normalize_symbol(symbol)
                df = ak.fund_etf_hist_em(
                    symbol=code,
                    period="daily",
                    start_date=start_dt.strftime("%Y%m%d"),
                    end_date=end_dt.strftime("%Y%m%d"),
                    adjust=adjust,
                )
                df = _normalize_ohlcv_frame(df)
            except Exception:
                df = pd.DataFrame()

        if df.empty:
            df = self._fetch_eastmoney_hist(symbol, days)
        if df.empty:
            df = self._fetch_tencent(symbol, days)
        if df.empty:
            df = self._fetch_sina(symbol, days)

        if self.current_date and not df.empty:
            dt = pd.to_datetime(self.current_date)
            df = df[df.index <= dt]
            df = df.tail(days)

        if not df.empty and self._cache:
            self._cache.put(cache_key, df)
        return df

    def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        symbols = [_normalize_symbol(s) for s in symbols if _normalize_symbol(s)]
        if not symbols:
            return pd.DataFrame()

        cache_key = (
            f"realtime_{'_'.join(sorted(symbols))}_{self.current_date or 'latest'}"
        )
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        df = pd.DataFrame()
        if _AKSHARE:
            try:
                raw = ak.fund_etf_spot_em()
                raw = _normalize_quote_frame(raw)
                if not raw.empty:
                    df = raw[raw["\u4ee3\u7801"].isin(symbols)].reset_index(drop=True)
            except Exception as e:
                print(f"[DataHub] akshare realtime quote failed: {e}")

        if df.empty:
            df = self._fetch_eastmoney_spot(symbols)
        if df.empty:
            df = self._fetch_sina_spot(symbols)

        if not df.empty and self._cache:
            self._cache.put(cache_key, df)
        return df

    @staticmethod
    def _to_tx_code(symbol: str) -> str:
        s = _normalize_symbol(symbol)
        return f"sh{s}" if _get_market_id(s) == 1 else f"sz{s}"

    def _fetch_tencent(self, symbol: str, days: int) -> pd.DataFrame:
        code = self._to_tx_code(symbol)
        try:
            url = (
                f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                f"?param={code},day,,,{days},qfq"
            )
            _rate_limit()
            resp = _request_get(url, timeout=15)
            resp.raise_for_status()
            data = json.loads(resp.content)
            if "data" not in data or code not in data["data"]:
                return pd.DataFrame()
            stk = data["data"][code]
            buf = stk.get("qfqday") or stk.get("day", [])
            if not buf:
                return pd.DataFrame()

            safe_buf = [row[:6] for row in buf if len(row) >= 6]
            if not safe_buf:
                return pd.DataFrame()

            cols = ["time", "open", "close", "high", "low", "volume"]
            df = pd.DataFrame(safe_buf, columns=cols)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df.index.name = "date"
            for c in ["open", "close", "high", "low", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[["open", "high", "low", "close", "volume"]]
            df["volume"] = df["volume"] * 100
            return df.dropna()
        except Exception as e:
            print(f"[DataHub] Tencent API failed: {e}")
            return pd.DataFrame()

    def _fetch_sina(self, symbol: str, days: int) -> pd.DataFrame:
        code = self._to_tx_code(symbol)
        try:
            url = (
                f"http://money.finance.sina.com.cn/quotes_service/api/"
                f"json_v2.php/CN_MarketData.getKLineData?"
                f"symbol={code}&scale=240&ma=5&datalen={days}"
            )
            _rate_limit()
            resp = _request_get(url, timeout=15)
            resp.raise_for_status()
            data = json.loads(resp.content)
            df = pd.DataFrame(
                data, columns=["day", "open", "high", "low", "close", "volume"]
            )
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = df[c].astype(float)
            df["day"] = pd.to_datetime(df["day"])
            df.set_index("day", inplace=True)
            df.index.name = "date"
            return df
        except Exception as e:
            print(f"[DataHub] Sina K-line failed: {e}")
            return pd.DataFrame()

    def _fetch_eastmoney_hist(self, symbol: str, days: int) -> pd.DataFrame:
        code = _normalize_symbol(symbol)
        try:
            url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            params = {
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": "101",
                "fqt": "1",
                "beg": "0",
                "end": "20500101",
                "lmt": str(days),
            }
            params["secid"] = _to_eastmoney_secid(code)
            _rate_limit()
            resp = _request_get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if data.get("data", {}).get("klines"):
                klines = data["data"]["klines"]
                records = [k.split(",") for k in klines]
                df = pd.DataFrame(
                    records,
                    columns=[
                        "date", "open", "close", "high", "low", "volume",
                        "amount", "amplitude", "turnover", "_", "_",
                    ],
                )
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df[["open", "high", "low", "close", "volume"]]
                return df.dropna()
            return pd.DataFrame()
        except Exception as e:
            print(f"[DataHub] Eastmoney K-line failed: {e}")
            return pd.DataFrame()

    def _fetch_eastmoney_etf_list(self) -> List[str]:
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        page = 1
        etf_codes: List[str] = []
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://quote.eastmoney.com/center/gridlist.html#fund_etf",
        }

        while True:
            params = {
                "pn": page, "pz": 1000, "po": 1, "np": 1,
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": 2, "invt": 2, "fid": "f12",
                "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024,b:MK0827",
                "fields": "f12",
            }
            resp = _request_get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            diff = data.get("data", {}).get("diff", [])
            if not diff:
                break
            for item in diff:
                code = _normalize_symbol(item.get("f12", ""))
                if code:
                    etf_codes.append(code)
            page += 1

        return etf_codes

    def _fetch_eastmoney_spot(self, symbols: List[str]) -> pd.DataFrame:
        records = []
        url = "https://push2.eastmoney.com/api/qt/stock/get"

        try:
            for symbol in symbols[:20]:
                params = {
                    "secid": _to_eastmoney_secid(symbol),
                    "fields": "f43,f44,f45,f46,f47,f48,f57,f58,f60,f169,f170",
                }
                _rate_limit()
                resp = _request_get(url, params=params, timeout=10)
                resp.raise_for_status()
                item = (resp.json() or {}).get("data") or {}
                if not item:
                    continue
                records.append({
                    "\u4ee3\u7801": _normalize_symbol(item.get("f57", symbol)),
                    "\u540d\u79f0": item.get("f58", ""),
                    "\u6700\u65b0\u4ef7": float(item.get("f43", 0) or 0) / 100,
                    "\u6da8\u8dcc\u5e45": float(item.get("f170", 0) or 0) / 100,
                    "\u6da8\u8dcc\u989d": float(item.get("f169", 0) or 0) / 100,
                    "\u6210\u4ea4\u91cf": float(item.get("f47", 0) or 0),
                    "\u6210\u4ea4\u989d": float(item.get("f48", 0) or 0),
                    "\u4eca\u5f00": float(item.get("f46", 0) or 0) / 100,
                    "\u6700\u9ad8": float(item.get("f44", 0) or 0) / 100,
                    "\u6700\u4f4e": float(item.get("f45", 0) or 0) / 100,
                    "\u6628\u6536": float(item.get("f60", 0) or 0) / 100,
                })
            return pd.DataFrame(records)
        except Exception as e:
            print(f"[DataHub] Eastmoney realtime quote failed: {e}")
            return pd.DataFrame()

    def _fetch_sina_spot(self, symbols: List[str]) -> pd.DataFrame:
        try:
            url = "https://hq.sinajs.cn/list="
            codes = [self._to_tx_code(symbol) for symbol in symbols[:100]]
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Referer": "https://finance.sina.com.cn/",
                "Accept": "*/*",
            }
            _rate_limit()
            resp = _request_get(url + ",".join(codes), headers=headers, timeout=10)
            resp.raise_for_status()
            resp.encoding = "gbk"
            lines = resp.text.strip().split("\n")

            records = []
            for line in lines:
                match = re.search(r'"([^"]*)"', line)
                if not match:
                    continue
                parts = match.group(1).split(",")
                if len(parts) <= 10:
                    continue
                sym_match = re.search(r'hq_s[zy]_(sh\d{6}|sz\d{6})', line)
                code = sym_match.group(1)[2:] if sym_match else ""
                latest = float(parts[3]) if parts[3] else 0.0
                prev_close = float(parts[2]) if parts[2] else 0.0
                change_amount = latest - prev_close if latest and prev_close else 0.0
                change_pct = (
                    (change_amount / prev_close * 100) if prev_close else 0.0
                )
                records.append({
                    "\u4ee3\u7801": code,
                    "\u540d\u79f0": parts[0],
                    "\u6700\u65b0\u4ef7": latest,
                    "\u6da8\u8dcc\u5e45": change_pct,
                    "\u6da8\u8dcc\u989d": change_amount,
                    "\u6210\u4ea4\u91cf": float(parts[8]) if parts[8] else 0.0,
                    "\u6210\u4ea4\u989d": float(parts[9]) if parts[9] else 0.0,
                    "\u4eca\u5f00": float(parts[1]) if parts[1] else 0.0,
                    "\u6700\u9ad8": float(parts[4]) if parts[4] else 0.0,
                    "\u6700\u4f4e": float(parts[5]) if parts[5] else 0.0,
                    "\u6628\u6536": prev_close,
                })
            return pd.DataFrame(records)
        except Exception as e:
            print(f"[DataHub] Sina realtime quote failed: {e}")
            return pd.DataFrame()

    def get_stock_list(self) -> List[str]:
        """获取 A 股主板股票列表（沪A + 深A）"""
        if not _AKSHARE:
            return []

        for attempt in range(3):
            try:
                df = ak.stock_info_a_code_name()
                if df.empty:
                    continue

                code_col = _pick_existing_column(df, ("代码", "code"))
                if not code_col:
                    continue

                prefixes = ("600", "601", "603", "605", "000", "001", "002", "003")
                codes = df[code_col].astype(str).tolist()
                codes = [c for c in codes if c.startswith(prefixes)]
                return list(dict.fromkeys(codes))
            except Exception as e:
                print(f"[DataHub] get_stock_list failed ({attempt + 1}/3): {e}")
                time.sleep(1)
        return []

    # ==================== 宽基 ETF 标的池构建 ====================

    # 宽基指数定义（BROAD_BASED_INDICES）
    BROAD_BASED_INDICES: Dict[str, Dict] = {
        "510050": {"name": "上证50ETF", "index": "000016", "type": "SSE50"},
        "510300": {"name": "沪深300ETF", "index": "000300", "type": "CSI300"},
        "510500": {"name": "中证500ETF", "index": "000905", "type": "CSI500"},
        "512100": {"name": "中证1000ETF", "index": "000852", "type": "CSI1000"},
        "159915": {"name": "创业板ETF", "index": "399006", "type": "GEM"},
        "159628": {"name": "深证100ETF", "index": "399330", "type": "SZSE100"},
        "588000": {"name": "科创50ETF", "index": "000688", "type": "STAR50"},
        "159788": {"name": "中证A500ETF", "name2": "中证A500ETF(中欧)", "index": "000510", "type": "CSI_A500"},
        "159692": {"name": "中证A500ETF", "name2": "中证A500ETF(嘉实)", "index": "000510", "type": "CSI_A500"},
    }

    # 需要排除的行业/主题 ETF 前缀模式
    INDUSTRY_ETFS: Tuple[str, ...] = (
        "512660",  # 军工
        "512800",  # 银行
        "512690",  # 消费
        "512760",  # 芯片
        "512980",  # 医疗
        "515050",  # 5G
        "512220",  # 通信
        "512200",  # 房地产
        "512580",  # 煤炭
        "512680",  # 军工
        "512170",  # 医疗
        "512000",  # 券商
    )

    def classify_etf_type(self, symbol: str, name: str = "") -> Dict[str, any]:
        """
        分类 ETF 类型。

        返回：
            {
                "is_broad_based": bool,
                "is_industry": bool,
                "is_cross_border": bool,
                "is_bond": bool,
                "is_money": bool,
                "index_type": str,
                "index_name": str,
            }
        """
        name_upper = (name or "").upper()

        # 宽基 ETF
        if symbol in self.BROAD_BASED_INDICES:
            info = self.BROAD_BASED_INDICES[symbol]
            return {
                "is_broad_based": True,
                "is_industry": False,
                "is_cross_border": False,
                "is_bond": False,
                "is_money": False,
                "index_type": info.get("type", "OTHER"),
                "index_name": info.get("name", name or "Unknown"),
            }

        # 排除类型检测
        is_bond = any(kw in name_upper for kw in ["债券", "国债", "信用债", "政金债"])
        is_money = any(kw in name_upper for kw in ["货币", "现金", "Money"])
        is_gold = any(kw in name_upper for kw in ["黄金", "Gold"])
        is_qdii = any(kw in name_upper for kw in ["QDII", "纳斯达克", "标普", "恒生", "日经", "印度"])

        # 行业/主题检测
        is_industry = (
            symbol in self.INDUSTRY_ETFS
            or any(
                kw in name_upper
                for kw in [
                    "银行", "证券", "保险", "地产", "煤炭", "钢铁", "有色", "化工",
                    "军工", "芯片", "半导体", "光伏", "新能源", "汽车", "医药",
                    "医疗", "消费", "食品", "白酒", "家电", "纺织", "服装",
                    "基建", "工程", "机械", "5G", "通信", "人工智能", "AI",
                    "机器人", "数字经济", "云计算", "大数据", "网络安全",
                ]
            )
        )

        return {
            "is_broad_based": False,
            "is_industry": is_industry,
            "is_cross_border": is_qdii,
            "is_bond": is_bond,
            "is_money": is_money or is_gold,
            "index_type": "OTHER",
            "index_name": name or "Unknown",
        }

    def get_broad_etf_universe(self) -> pd.DataFrame:
        """
        构建宽基 ETF 标的池（从 data_cache 扫描已下载的文件）。

        规则：
        1. 只选宽基 ETF，不选行业主题 ETF
        2. 标的必须在本地有数据文件
        3. 不根据未来收益选择成分

        返回：
            pd.DataFrame: 宽基 ETF 标的池
        """
        if not self.local_dir or not os.path.exists(self.local_dir):
            return pd.DataFrame()

        # 扫描 data_cache 中的 CSV 文件
        csv_files = [
            f.replace(".csv", "")
            for f in os.listdir(self.local_dir)
            if f.endswith(".csv")
            and "_backup" not in f
            and "_test" not in f
        ]

        results = []
        for symbol in csv_files:
            classification = self.classify_etf_type(symbol)
            if classification["is_broad_based"]:
                info = self.BROAD_BASED_INDICES.get(symbol, {})
                results.append({
                    "symbol": symbol,
                    "name": info.get("name", classification["index_name"]),
                    "index_type": info.get("type", "OTHER"),
                    "index_code": info.get("index", ""),
                })

        df = pd.DataFrame(results)
        return df.sort_values("index_type").reset_index(drop=True)

    def save_etf_universe(self, filepath: str = None) -> str:
        """
        保存 ETF 标的池到 CSV + JSON 文件。

        参数：
            filepath: CSV 文件路径，默认保存在 local_dir/eligible_etf_universe.csv

        返回：
            str: 保存的文件路径
        """
        df = self.get_broad_etf_universe()
        if df.empty:
            return ""

        if filepath is None:
            filepath = os.path.join(self.local_dir or ".", "eligible_etf_universe.csv")

        df.to_csv(filepath, index=False, encoding="utf-8-sig")

        json_path = filepath.replace(".csv", ".json")
        universe_data = {
            "build_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_count": len(df),
            "symbols": df["symbol"].tolist(),
            "details": df.to_dict("records"),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(universe_data, f, ensure_ascii=False, indent=2)

        return filepath


ETFDataHub = DataHub
