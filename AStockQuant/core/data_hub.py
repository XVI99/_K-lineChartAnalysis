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
import socket
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

try:
    import baostock as bs
    _BAOSTOCK = True
except ImportError:
    _BAOSTOCK = False

try:
    import tushare as ts
    _TUSHARE = True
except ImportError:
    _TUSHARE = False

try:
    import efinance as ef
    _EFINANCE = True
except ImportError:
    _EFINANCE = False

try:
    from pytdx.hq import TdxHq_API as _TdxHq_API
    _PYTDX = True
except ImportError:
    _PYTDX = False

try:
    import yfinance as yf
    _YFINANCE = True
except ImportError:
    _YFINANCE = False

try:
    import tickflow as _tickflow
    _TICKFLOW = True
except ImportError:
    _TICKFLOW = False

try:
    import socks as _socks
    _SOCKS = True
except ImportError:
    _SOCKS = False

# ==================== SOCKS5 Tunnel for Raw-Socket Libraries ====================

def _detect_socks5_proxy() -> Optional[Tuple[str, int]]:
    """自动探测本地 SOCKS5 代理（Clash/Mihomo mixed-port）"""
    import socket as _socket
    for port in (7897, 7890, 1080, 10808):
        try:
            s = _socket.create_connection(("127.0.0.1", port), timeout=0.5)
            s.close()
            return ("127.0.0.1", port)
        except OSError:
            continue
    return None

_DEFAULT_SOCKS5 = _detect_socks5_proxy() if _SOCKS else None

import contextlib

@contextlib.contextmanager
def _socks_tunnel(proxy: Optional[Tuple[str, int]] = None, timeout: int = 15):
    """临时将全局 socket 替换为 SOCKS5 代理，用于 baostock/pytdx 等原生 TCP 库。

    背景：当系统有 VMware 虚拟网卡或 VPN 劫持默认路由时，baostock/pytdx 的
    原生 TCP 连接会超时。通过 Clash 的 mixed-port (SOCKS5) 隧道转发可绕过此问题。

    参数：
        proxy:  (host, port) 元组，None 则自动探测
        timeout: socket 超时秒数，防止协议通信 hang 住
    """
    proxy = proxy or _DEFAULT_SOCKS5
    if not proxy or not _SOCKS:
        yield
        return

    original_socket = socket.socket
    original_create = socket.create_connection
    original_timeout = socket.getdefaulttimeout()
    _socks.set_default_proxy(_socks.SOCKS5, proxy[0], proxy[1])

    # 自定义 socksocket 子类，强制在创建时设置 timeout
    # 否则 baostock 内部 socket.recv() 会永久阻塞
    class _TimedSockSocket(_socks.socksocket):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.settimeout(timeout)

    socket.socket = _TimedSockSocket
    socket.setdefaulttimeout(timeout)

    def _socks_create_connection(address, timeout=None, source_address=None):
        sock = _TimedSockSocket()
        sock.settimeout(timeout or 15)
        if source_address:
            sock.bind(source_address)
        sock.connect(address)
        return sock

    socket.create_connection = _socks_create_connection
    try:
        yield
    finally:
        socket.socket = original_socket
        socket.create_connection = original_create
        socket.setdefaulttimeout(original_timeout)
        _socks.socksocket.default_proxy = None

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
    """简单限流：sleep interval 秒，避免高频请求被数据源 ban IP。

    batch_limit / batch_sleep 保留以兼容签名，当前未按批次聚合
    （调用方均为单次 fetch，简单 sleep 即可）。如需更精细的批次限流，
    可在此处维护全局计数器按 batch_sleep 休眠。
    """
    time.sleep(interval)


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
        data_source_config: Optional[Dict] = None,
    ):
        self._cache = CacheManager(
            host=redis_host, port=redis_port, db=redis_db,
            expire_seconds=cache_expire,
        ) if use_cache else None
        self.current_date = current_date
        self.mode = mode
        # 默认落盘目录：AStockQuant/data_cache/
        if local_dir is None:
            local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")
        self.local_dir = local_dir
        os.makedirs(self.local_dir, exist_ok=True)

        # 数据源配置
        dsc = data_source_config or {}
        self._baostock_enabled = dsc.get("baostock", {}).get("enabled", True) and _BAOSTOCK
        self._tushare_enabled = dsc.get("tushare", {}).get("enabled", False) and _TUSHARE
        # token 优先读环境变量 TUSHARE_TOKEN，其次 config.yaml（已建议留空）
        self._tushare_token = os.environ.get("TUSHARE_TOKEN") or dsc.get("tushare", {}).get("token", "")
        self._efinance_enabled = dsc.get("efinance", {}).get("enabled", True) and _EFINANCE
        self._tsanghi_enabled = dsc.get("tsanghi", {}).get("enabled", False)
        self._tsanghi_token = dsc.get("tsanghi", {}).get("token", "")
        self._pytdx_enabled = dsc.get("pytdx", {}).get("enabled", True) and _PYTDX
        self._yfinance_enabled = dsc.get("yfinance", {}).get("enabled", True) and _YFINANCE
        self._tickflow_enabled = dsc.get("tickflow", {}).get("enabled", True) and _TICKFLOW
        self._bs_logged_in = False
        self._pytdx_api: Optional[object] = None
        self._tickflow_client = None

        # SOCKS5 代理配置（用于 baostock/pytdx 等原生 TCP 库）
        proxy_cfg = dsc.get("proxy", {}).get("socks5", {})
        if proxy_cfg.get("enabled", True) and proxy_cfg.get("host"):
            self._socks5_proxy = (proxy_cfg["host"], proxy_cfg.get("port", 7897))
        else:
            self._socks5_proxy = _DEFAULT_SOCKS5
        if self._socks5_proxy:
            print(f"[DataHub] SOCKS5 代理已启用: {self._socks5_proxy[0]}:{self._socks5_proxy[1]} (baostock/pytdx)")

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

    def _load_local_csv(self, symbol: str) -> pd.DataFrame:
        """从本地 CSV 加载数据（用于离线优先模式）"""
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

    def _save_local_csv(self, symbol: str, df: pd.DataFrame) -> None:
        """保存数据到本地 CSV（自动落盘）"""
        if df.empty or not self.local_dir:
            return
        try:
            csv_path = os.path.join(self.local_dir, f"{symbol}.csv")
            df.to_csv(csv_path, index=True, index_label="date", encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _is_local_fresh(df: pd.DataFrame) -> bool:
        """检查本地数据是否足够新（最后一天距今 <= 2 个日历日）"""
        if df.empty:
            return False
        try:
            last_date = df.index[-1]
            if hasattr(last_date, "to_pydatetime"):
                last_date = last_date.to_pydatetime()
            else:
                last_date = pd.Timestamp(last_date).to_pydatetime()
            age = (datetime.now() - last_date).days
            return age <= 2  # 周末/节假日容忍 2 天
        except Exception:
            return False

    def get_stock_data_ex(
        self, symbol: str, days: int = 500, adjust: str = "qfq"
    ) -> pd.DataFrame:
        cache_key = f"etf_ex_{symbol}_{days}_{adjust}_{self.current_date or 'latest'}"
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # 1. 离线优先：检查本地 CSV 是否足够新
        local_df = self._load_local_csv(symbol)
        if self._is_local_fresh(local_df):
            # 本地数据足够新，直接用（截取所需天数）
            result = local_df.tail(days)
            if self.current_date:
                dt = pd.to_datetime(self.current_date)
                result = result[result.index <= dt]
                result = result.tail(days)
            if not result.empty and self._cache:
                self._cache.put(cache_key, result)
            return result

        # 2. 本地数据不存在或过期 → 在线获取
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
        if df.empty:
            df = self._fetch_baostock(symbol, days)
        if df.empty:
            df = self._fetch_tushare(symbol, days)
        if df.empty:
            df = self._fetch_efinance(symbol, days)
        if df.empty:
            df = self._fetch_tickflow(symbol, days)
        if df.empty:
            df = self._fetch_yfinance(symbol, days)
        if df.empty:
            df = self._fetch_pytdx(symbol, days)

        # 3. 在线获取失败 → 回退到本地 CSV（即使过期）
        if df.empty and not local_df.empty:
            df = local_df.tail(days)

        if self.current_date and not df.empty:
            dt = pd.to_datetime(self.current_date)
            df = df[df.index <= dt]
            df = df.tail(days)

        # 4. 自动落盘 + Redis 缓存
        if not df.empty:
            # 与本地数据合并后保存（增量更新，保留历史）
            if not local_df.empty:
                combined = pd.concat([local_df, df])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                self._save_local_csv(symbol, combined)
            else:
                self._save_local_csv(symbol, df)
            if self._cache:
                self._cache.put(cache_key, df)
        return df

    def get_index_data(self, code: str = "000300", days: int = 500) -> pd.DataFrame:
        """获取指数日K数据（如沪深300=000300，用于 market regime 判断）。

        优先 akshare，失败返回空 DataFrame（调用方需做空值兜底，如
        MarketScanner._detect_regime 对空 df 返回 NEUTRAL）。
        """
        cache_key = f"idx_{code}_{days}_{self.current_date or 'latest'}"
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        raw = _normalize_symbol(code)
        # akshare 指数代码：沪 sh + 6位（000300/000001），深 sz + 6位（399001/399006）
        ak_code = f"sz{raw}" if raw.startswith("399") else f"sh{raw}"

        if _AKSHARE:
            try:
                df = ak.stock_zh_index_daily(symbol=ak_code)
                if df is not None and not df.empty:
                    df = df.rename(columns=str.lower)
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date")
                    for c in ["open", "high", "low", "close", "volume"]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                    df = df[keep].dropna().sort_index().tail(days)
                    if self.current_date:
                        df = df[df.index <= pd.to_datetime(self.current_date)]
                    if not df.empty:
                        if self._cache:
                            self._cache.put(cache_key, df)
                        return df
            except Exception as e:
                print(f"[DataHub] akshare index {code} failed: {e}")

        return pd.DataFrame()

    def batch_stock_data(
        self, stock_list: List[str], days: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """批量获取多只标的的日K数据，返回 {symbol: DataFrame}。

        失败的标的会被跳过（不包含在返回 dict 中）。内部复用
        get_stock_data_ex，享有完整的离线优先 + 10源降级链 + 增量落盘。
        """
        result: Dict[str, pd.DataFrame] = {}
        for sym in stock_list:
            try:
                df = self.get_stock_data_ex(sym, days=days)
                if df is not None and not df.empty:
                    result[sym] = df
            except Exception as e:
                print(f"[DataHub] batch fetch {sym} failed: {e}")
        return result

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

    # ==================== 新增降级源：Baostock / Tushare / efinance ====================

    def _fetch_baostock(self, symbol: str, days: int) -> pd.DataFrame:
        """从 Baostock 获取 ETF/股票历史日线（免费，无需 token）"""
        if not self._baostock_enabled:
            return pd.DataFrame()
        try:
            import baostock as bs
            from baostock.util import socketutil as _bs_sock
            from baostock.common import context as _bs_ctx
            import baostock.common.contants as _bs_cons
            import baostock.data.messageheader as _bs_msgheader
            import zlib as _zlib
            code = _normalize_symbol(symbol)
            bs_code = f"sh.{code}" if _get_market_id(code) == 1 else f"sz.{code}"
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=days)

            # 直接 monkey-patch baostock 的 SocketUtil.connect 和 send_msg，
            # 使其通过 SOCKS5 隧道连接并设置 timeout，且修复 recv 空循环 bug
            _original_connect = _bs_sock.SocketUtil.connect
            _original_send_msg = _bs_sock.send_msg

            if self._socks5_proxy and _SOCKS:
                proxy_host, proxy_port = self._socks5_proxy

                def _socks_connect(self):
                    sock = _socks.socksocket()
                    sock.set_proxy(_socks.SOCKS5, proxy_host, proxy_port)
                    sock.settimeout(15)
                    sock.connect((_bs_cons.BAOSTOCK_SERVER_IP, _bs_cons.BAOSTOCK_SERVER_PORT))
                    setattr(_bs_ctx, "default_socket", sock)

                def _socks_send_msg(msg):
                    """修复版 send_msg：处理 recv 超时和连接关闭"""
                    try:
                        if hasattr(_bs_ctx, "default_socket"):
                            default_socket = getattr(_bs_ctx, "default_socket")
                            if default_socket is not None:
                                msg = msg + "\n"
                                default_socket.send(bytes(msg, encoding='utf-8'))
                                receive = b""
                                while True:
                                    recv = default_socket.recv(8192)
                                    if not recv:
                                        break
                                    receive += recv
                                    if receive[-13:] == b"<![CDATA[]]>\n":
                                        break
                                if not receive:
                                    return None
                                head_bytes = receive[0:_bs_cons.MESSAGE_HEADER_LENGTH]
                                head_str = bytes.decode(head_bytes)
                                head_arr = head_str.split(_bs_cons.MESSAGE_SPLIT)
                                if head_arr[1] in _bs_cons.COMPRESSED_MESSAGE_TYPE_TUPLE:
                                    head_inner_length = int(head_arr[2])
                                    body_str = bytes.decode(_zlib.decompress(
                                        receive[_bs_cons.MESSAGE_HEADER_LENGTH:
                                                _bs_cons.MESSAGE_HEADER_LENGTH + head_inner_length]))
                                    return head_str + body_str
                                else:
                                    return bytes.decode(receive)
                            else:
                                return None
                        else:
                            return None
                    except Exception:
                        return None

                _bs_sock.SocketUtil.connect = _socks_connect
                _bs_sock.send_msg = _socks_send_msg

            try:
                if not self._bs_logged_in:
                    lg = bs.login()
                    if lg.error_code != "0":
                        print(f"[DataHub] Baostock login failed: {lg.error_msg}，该源将在本次运行中禁用")
                        self._baostock_enabled = False
                        return pd.DataFrame()
                    self._bs_logged_in = True

                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,open,high,low,close,volume",
                    start_date=start_dt.strftime("%Y-%m-%d"),
                    end_date=end_dt.strftime("%Y-%m-%d"),
                    frequency="d",
                )
                rows = []
                while rs.error_code == "0" and rs.next():
                    rows.append(rs.get_row_data())
            finally:
                _bs_sock.SocketUtil.connect = _original_connect
                _bs_sock.send_msg = _original_send_msg

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["volume"] = df["volume"] * 100
            return df[["open", "high", "low", "close", "volume"]].dropna().sort_index()
        except (socket.timeout, TimeoutError, OSError) as e:
            print(f"[DataHub] Baostock 超时/网络错误，该源将禁用: {e}")
            self._baostock_enabled = False
            return pd.DataFrame()
        except Exception as e:
            print(f"[DataHub] Baostock failed: {e}")
            return pd.DataFrame()

    def _fetch_tushare(self, symbol: str, days: int) -> pd.DataFrame:
        """从 Tushare Pro 获取 ETF 日线（需 token）"""
        if not self._tushare_enabled or not self._tushare_token:
            return pd.DataFrame()
        try:
            import tushare as ts
            ts.set_token(self._tushare_token)
            pro = ts.pro_api()

            code = _normalize_symbol(symbol)
            ts_code = f"{code}.SH" if _get_market_id(code) == 1 else f"{code}.SZ"
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=days)

            df = pro.fund_daily(
                ts_code=ts_code,
                start_date=start_dt.strftime("%Y%m%d"),
                end_date=end_dt.strftime("%Y%m%d"),
            )
            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(columns={"trade_date": "date", "vol": "volume"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["volume"] = df["volume"] * 100
            return df[["open", "high", "low", "close", "volume"]].dropna().sort_index()
        except Exception as e:
            print(f"[DataHub] Tushare failed: {e}")
            return pd.DataFrame()

    def _fetch_efinance(self, symbol: str, days: int) -> pd.DataFrame:
        """从 efinance 获取 ETF/股票历史行情（免费，无需 token）"""
        if not self._efinance_enabled:
            return pd.DataFrame()
        try:
            import efinance as ef
            code = _normalize_symbol(symbol)
            df = ef.stock.get_quote_history(code, kctypes="1")
            if df is None or df.empty:
                return pd.DataFrame()

            rename_map = {
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
            }
            df = df.rename(columns=rename_map)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            for c in ["open", "high", "low", "close", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            if len(keep) < 5:
                return pd.DataFrame()
            return df[keep].tail(days).dropna().sort_index()
        except Exception as e:
            print(f"[DataHub] efinance failed: {e}")
            return pd.DataFrame()

    def _fetch_tsanghi(self, symbol: str, days: int) -> pd.DataFrame:
        """从 Tsanghi 沧海数据获取 ETF 日线（需 token）。

        注意: 本源目前未接入 get_stock_data_ex 的降级链（tsanghi 默认
        enabled: false）。如需启用，在 config.yaml 设 tsanghi.enabled=true
        并填 token，然后在 get_stock_data_ex 的源列表中追加 self._fetch_tsanghi。
        """
        if not self._tsanghi_enabled or not self._tsanghi_token:
            return pd.DataFrame()
        try:
            code = _normalize_symbol(symbol)
            market = "XSHG" if _get_market_id(code) == 1 else "XSHE"
            url = (
                f"https://tsanghi.com/api/fin/etf/{market}/daily"
                f"?token={self._tsanghi_token}&ticker={code}"
                f"&limit={days}"
            )
            resp = _request_get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("data", [])
            if not records:
                return pd.DataFrame()
            df = pd.DataFrame(records)
            if "date" not in df.columns:
                return pd.DataFrame()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            for c in ["open", "high", "low", "close", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            if len(keep) < 5:
                return pd.DataFrame()
            return df[keep].dropna().sort_index()
        except Exception as e:
            print(f"[DataHub] Tsanghi failed: {e}")
            return pd.DataFrame()

    def _fetch_pytdx(self, symbol: str, days: int) -> pd.DataFrame:
        """从通达信行情接口获取 ETF/股票日线（免费，无需 token，通过 SOCKS5 隧道）"""
        if not self._pytdx_enabled:
            return pd.DataFrame()
        try:
            import pytdx.hq as _pytdx_hq
            import pytdx.base_socket_client as _pytdx_bsc
            code = _normalize_symbol(symbol)
            market = _get_market_id(code)  # 1=SH, 0=SZ
            count = min(days, 800)  # pytdx 单次最多 ~800 根

            # pytdx 的 TrafficStatSocket 继承自 socket.socket，
            # 全局 monkey-patch 不影响它（类继承在定义时已绑定）。
            # 需要直接替换为 SOCKS5 兼容版本。
            _original_ts = _pytdx_bsc.TrafficStatSocket
            if self._socks5_proxy and _SOCKS:
                proxy_host, proxy_port = self._socks5_proxy

                class _SocksTrafficStatSocket(_socks.socksocket):
                    def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None):
                        super().__init__(family, type, proto, fileno)
                        self.settimeout(15)
                        self.send_pkg_num = 0
                        self.recv_pkg_num = 0
                        self.send_pkg_bytes = 0
                        self.recv_pkg_bytes = 0
                        self.first_pkg_send_time = None
                        self.last_api_send_bytes = 0
                        self.last_api_recv_bytes = 0

                _pytdx_bsc.TrafficStatSocket = _SocksTrafficStatSocket
                _socks.set_default_proxy(_socks.SOCKS5, proxy_host, proxy_port)

            try:
                if self._pytdx_api is None:
                    api = _TdxHq_API()
                    api.need_setup = False
                    connected = False
                    for ip, port in [
                        ("119.147.212.81", 7709), ("14.215.128.18", 7709),
                        ("59.173.18.77", 7709), ("180.153.39.51", 7709),
                    ]:
                        try:
                            if api.connect(ip, port, time_out=8):
                                connected = True
                                break
                        except Exception:
                            continue
                    if not connected:
                        print("[DataHub] Pytdx 所有服务器连接失败，该源将禁用")
                        self._pytdx_enabled = False
                        return pd.DataFrame()
                    self._pytdx_api = api

                bars = self._pytdx_api.get_security_bars(4, market, code, 0, count)
            finally:
                _pytdx_bsc.TrafficStatSocket = _original_ts
                if _SOCKS:
                    _socks.socksocket.default_proxy = None

            if not bars:
                return pd.DataFrame()
            df = self._pytdx_api.to_df(bars)
            df["date"] = pd.to_datetime(df["datetime"].str[:10])
            df = df.set_index("date")
            df = df.rename(columns={"vol": "volume"})
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[["open", "high", "low", "close", "volume"]].dropna().sort_index().tail(days)
        except (socket.timeout, TimeoutError, OSError) as e:
            print(f"[DataHub] Pytdx 超时/网络错误，该源将禁用: {e}")
            self._pytdx_enabled = False
            return pd.DataFrame()
        except Exception as e:
            print(f"[DataHub] Pytdx failed: {e}")
            return pd.DataFrame()

    def _fetch_tickflow(self, symbol: str, days: int) -> pd.DataFrame:
        """从 TickFlow 免费版获取 ETF/股票日K线（免费，无需 token，HTTP API）"""
        if not self._tickflow_enabled:
            return pd.DataFrame()
        try:
            if self._tickflow_client is None:
                self._tickflow_client = _tickflow.TickFlow.free()

            code = _normalize_symbol(symbol)
            suffix = ".SH" if _get_market_id(code) == 1 else ".SZ"
            tf_symbol = f"{code}{suffix}"

            df = self._tickflow_client.klines.get(
                symbol=tf_symbol, period="1d", count=min(days, 500), as_dataframe=True
            )
            if df is None or df.empty:
                return pd.DataFrame()

            df["date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("date")
            for c in ["open", "high", "low", "close", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            if len(keep) < 5:
                return pd.DataFrame()
            return df[keep].dropna().sort_index().tail(days)
        except Exception as e:
            print(f"[DataHub] TickFlow failed: {e}")
            return pd.DataFrame()

    def _fetch_yfinance(self, symbol: str, days: int) -> pd.DataFrame:
        """从 Yahoo Finance 获取 A 股 ETF/股票日线（免费，HTTP API，可能被限流）"""
        if not self._yfinance_enabled:
            return pd.DataFrame()
        try:
            code = _normalize_symbol(symbol)
            suffix = ".SS" if _get_market_id(code) == 1 else ".SZ"
            yf_symbol = f"{code}{suffix}"

            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=f"{min(days, 500)}d")
            if df is None or df.empty:
                return pd.DataFrame()

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            for c in ["open", "high", "low", "close", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            if len(keep) < 5:
                return pd.DataFrame()
            return df[keep].dropna().sort_index().tail(days)
        except Exception as e:
            print(f"[DataHub] YFinance failed: {e}")
            return pd.DataFrame()

    def close(self):
        """释放资源（Baostock 登出 / Pytdx 断开 / TickFlow 关闭）"""
        if self._bs_logged_in and _BAOSTOCK:
            try:
                import baostock as bs
                from baostock.util import socketutil as _bs_sock
                from baostock.common import context as _bs_ctx
                import baostock.common.contants as _bs_cons
                if self._socks5_proxy and _SOCKS:
                    proxy_host, proxy_port = self._socks5_proxy
                    def _socks_connect(self):
                        sock = _socks.socksocket()
                        sock.set_proxy(_socks.SOCKS5, proxy_host, proxy_port)
                        sock.settimeout(10)
                        sock.connect((_bs_cons.BAOSTOCK_SERVER_IP, _bs_cons.BAOSTOCK_SERVER_PORT))
                        setattr(_bs_ctx, "default_socket", sock)
                    _bs_sock.SocketUtil.connect = _socks_connect
                bs.logout()
                self._bs_logged_in = False
            except Exception:
                pass
        if self._pytdx_api is not None:
            try:
                self._pytdx_api.disconnect()
            except Exception:
                pass
            self._pytdx_api = None
        if self._tickflow_client is not None:
            try:
                self._tickflow_client.close()
            except Exception:
                pass
            self._tickflow_client = None

    def __del__(self):
        """析构时确保 Baostock 登出，防止会话泄露"""
        try:
            self.close()
        except Exception:
            pass

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
