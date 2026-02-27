"""
Data Cache Module — SQLite-based local caching for stock data.

Eliminates redundant API calls during scanning by storing daily OHLCV data
in a local SQLite database. Only fetches new data when the cache is stale.
"""
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_cache.db")


class DataCache:
    """SQLite-based cache for daily stock data."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the cache table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_cache (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_meta (
                symbol TEXT PRIMARY KEY,
                last_updated TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def get_last_updated(self, symbol):
        """Get the last update timestamp for a symbol."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT last_updated FROM cache_meta WHERE symbol = ?", (symbol,)
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def is_stale(self, symbol, max_age_hours=18):
        """Check if cached data is stale (older than max_age_hours)."""
        last = self.get_last_updated(symbol)
        if last is None:
            return True
        last_dt = datetime.fromisoformat(last)
        return (datetime.now() - last_dt).total_seconds() > max_age_hours * 3600

    def save_data(self, symbol, df):
        """Save DataFrame to cache. Upserts rows."""
        if df.empty:
            return

        conn = self._get_conn()
        try:
            for idx, row in df.iterrows():
                date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)
                conn.execute("""
                    INSERT OR REPLACE INTO daily_cache (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (symbol, date_str, row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))

            conn.execute("""
                INSERT OR REPLACE INTO cache_meta (symbol, last_updated)
                VALUES (?, ?)
            """, (symbol, datetime.now().isoformat()))

            conn.commit()
        finally:
            conn.close()

    def load_data(self, symbol, days=400):
        """Load cached data for a symbol."""
        conn = self._get_conn()
        try:
            cutoff = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y-%m-%d")
            df = pd.read_sql_query(
                "SELECT date, open as 'Open', high as 'High', low as 'Low', "
                "close as 'Close', volume as 'Volume' "
                "FROM daily_cache WHERE symbol = ? AND date >= ? ORDER BY date",
                conn, params=(symbol, cutoff)
            )
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df.index.name = 'Date'
            return df
        finally:
            conn.close()

    def get_or_fetch(self, symbol, days=400, fetch_func=None):
        """
        Get data from cache or fetch from API if stale.

        Args:
            symbol (str): Stock code.
            days (int): Number of days of data needed.
            fetch_func (callable): Function that fetches data from API.
                Signature: fetch_func(symbol, days) -> pd.DataFrame

        Returns:
            pd.DataFrame: OHLCV data.
        """
        if not self.is_stale(symbol):
            cached = self.load_data(symbol, days)
            if not cached.empty and len(cached) >= days * 0.5:
                return cached

        # Fetch fresh data
        if fetch_func is None:
            from custom_data import fetch_daily_data
            fetch_func = fetch_daily_data

        df = fetch_func(symbol, days)
        if df is not None and not df.empty:
            self.save_data(symbol, df)
        return df if df is not None else pd.DataFrame()

    def batch_prefetch(self, symbols, days=400, fetch_func=None, max_workers=4):
        """
        Prefetch data for multiple symbols in parallel.

        Args:
            symbols (list[str]): List of stock codes.
            days (int): Number of days.
            fetch_func (callable): API fetch function.
            max_workers (int): Thread pool size.

        Returns:
            dict: {symbol: pd.DataFrame}
        """
        results = {}

        def _fetch_one(sym):
            return sym, self.get_or_fetch(sym, days, fetch_func)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, s): s for s in symbols}
            for future in as_completed(futures):
                try:
                    sym, df = future.result()
                    results[sym] = df
                except Exception as e:
                    sym = futures[future]
                    results[sym] = pd.DataFrame()

        return results

    def clear_cache(self):
        """Clear all cached data."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM daily_cache")
            conn.execute("DELETE FROM cache_meta")
            conn.commit()
        finally:
            conn.close()

    def get_cache_stats(self):
        """Get cache statistics."""
        conn = self._get_conn()
        try:
            n_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM daily_cache").fetchone()[0]
            n_rows = conn.execute("SELECT COUNT(*) FROM daily_cache").fetchone()[0]
            return {"symbols": n_symbols, "rows": n_rows}
        finally:
            conn.close()
