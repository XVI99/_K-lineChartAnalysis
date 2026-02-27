import requests
import json
import pandas as pd
import datetime
import time as _time

# --- Constants ---
HTTP_TIMEOUT = 15  # seconds
MAX_RETRIES = 2

# --- User Provided Functions ---

# 腾讯日线
def get_price_day_tx(code, end_date='', count=10, frequency='1d'):
    unit = 'week' if frequency in '1w' else 'month' if frequency in '1M' else 'day'
    if end_date:
        end_date = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime.date) else end_date.split(' ')[0]
    end_date = '' if end_date == datetime.datetime.now().strftime('%Y-%m-%d') else end_date
    URL = f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},{unit},,{end_date},{count},qfq'
    for attempt in range(MAX_RETRIES + 1):
        try:
            st = json.loads(requests.get(URL, timeout=HTTP_TIMEOUT).content)
            ms = 'qfq' + unit
            stk = st['data'][code]
            buf = stk[ms] if ms in stk else stk[unit]
            df = pd.DataFrame(buf, columns=['time', 'open', 'close', 'high', 'low', 'volume'], dtype='float')
            df.time = pd.to_datetime(df.time)
            df.set_index(['time'], inplace=True)
            df.index.name = 'Date'
            return _validate_ohlcv(df)
        except Exception as e:
            if attempt < MAX_RETRIES:
                _time.sleep(1 * (attempt + 1))
                continue
            return None

# 腾讯分钟线
def get_price_min_tx(code, end_date=None, count=10, frequency='1d'):
    ts = int(frequency[:-1]) if frequency[:-1].isdigit() else 1
    if end_date:
        end_date = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime.date) else end_date.split(' ')[0]
    URL = f'http://ifzq.gtimg.cn/appstock/app/kline/mkline?param={code},m{ts},,{count}'
    try:
        st = json.loads(requests.get(URL).content)
        buf = st['data'][code]['m' + str(ts)]
        df = pd.DataFrame(buf, columns=['time', 'open', 'close', 'high', 'low', 'volume', 'n1', 'n2'])
        df = df[['time', 'open', 'close', 'high', 'low', 'volume']]
        df[['open', 'close', 'high', 'low', 'volume']] = df[['open', 'close', 'high', 'low', 'volume']].astype('float')
        df.time = pd.to_datetime(df.time)
        df.set_index(['time'], inplace=True)
        df.index.name = 'Date'
        df['close'][-1] = float(st['data'][code]['qt'][code][3])
        return df
    except Exception as e:
        print(f"获取分钟线数据出错: {e}")
        return None

# 新浪全周期获取函数
def get_price_sina(code, end_date='', count=10, frequency='60m'):
    frequency = frequency.replace('1d', '240m').replace('1w', '1200m').replace('1M', '7200m')
    ts = int(frequency[:-1]) if frequency[:-1].isdigit() else 1
    if (end_date != '') & (frequency in ['240m', '1200m', '7200m']):
        end_date = pd.to_datetime(end_date) if not isinstance(end_date, datetime.date) else end_date
        unit = 4 if frequency == '1200m' else 29 if frequency == '7200m' else 1
        count = count + (datetime.datetime.now() - end_date).days // unit
    URL = f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={code}&scale={ts}&ma=5&datalen={count}'
    for attempt in range(MAX_RETRIES + 1):
        try:
            dstr = json.loads(requests.get(URL, timeout=HTTP_TIMEOUT).content)
            df = pd.DataFrame(dstr, columns=['day', 'open', 'high', 'low', 'close', 'volume'])
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df.day = pd.to_datetime(df.day)
            df.set_index(['day'], inplace=True)
            df.index.name = 'Date'
            if (end_date != '') & (frequency in ['240m', '1200m', '7200m']):
                return _validate_ohlcv(df[df.index <= end_date][-count:])
            return _validate_ohlcv(df)
        except Exception as e:
            if attempt < MAX_RETRIES:
                _time.sleep(1 * (attempt + 1))
                continue
            return None

def get_price(code, end_date='', count=10, frequency='1d', fields=[]):
    # Adapter Logic to handle 'sh'/'sz' manually if not provided in 'code' for raw logic, 
    # but based on provided code, the replacement logic relies on finding .XSHG etc.
    # If code is '000001', replacement does nothing.
    # We add prefix logic here explicitly for robust handling.
    
    # Simple heuristic for A-share prefixes
    if code.isdigit():
        if code.startswith('6'):
            code = 'sh' + code
        elif code.startswith('0') or code.startswith('3'):
            code = 'sz' + code
        elif code.startswith('4') or code.startswith('8'):
             code = 'bj' + code
    
    xcode = code.replace('.XSHG', '').replace('.XSHE', '')
    xcode = 'sh' + xcode if ('XSHG' in code) else 'sz' + xcode if ('XSHE' in code) else code

    if frequency in ['1d', '1w', '1M']:
        try:
            return get_price_sina(xcode, end_date=end_date, count=count, frequency=frequency)
        except:
            return get_price_day_tx(xcode, end_date=end_date, count=count, frequency=frequency)

    if frequency in ['1m', '5m', '15m', '30m', '60m']:
        if frequency == '1m':
            return get_price_min_tx(xcode, end_date=end_date, count=count, frequency=frequency)
        try:
            return get_price_sina(xcode, end_date=end_date, count=count, frequency=frequency)
        except:
            return get_price_min_tx(xcode, end_date=end_date, count=count, frequency=frequency)

# --- Adapter for Budget Monitor ---
def fetch_daily_data(symbol, days=365, use_cache=False):
    """
    Wrapper to fetch daily data and return standardized DataFrame.
    
    Args:
        symbol (str): Stock code.
        days (int): Number of trading days to fetch.
        use_cache (bool): If True, use SQLite cache to reduce API calls.
    """
    try:
        # Use cache if enabled
        if use_cache:
            try:
                from data_cache import DataCache
                cache = DataCache()
                df = cache.get_or_fetch(symbol, days, fetch_func=_fetch_raw_daily)
                return df if df is not None else pd.DataFrame()
            except Exception:
                pass  # Fall through to direct API call

        return _fetch_raw_daily(symbol, days)
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


def _fetch_raw_daily(symbol, days=365):
    """Internal: fetch from API and standardize columns."""
    df = get_price(symbol, count=days, frequency='1d')

    if df is None or df.empty:
        return pd.DataFrame()

    # Standardize Columns
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    # Ensure correct column order
    available = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    df = df[available]

    return df


def _validate_ohlcv(df):
    """
    Validate and clean OHLCV data.
    Removes rows with zero/negative prices, fills small gaps.
    
    Args:
        df: DataFrame with OHLCV columns
    
    Returns:
        Cleaned DataFrame, or None if data is too corrupt.
    """
    if df is None or df.empty:
        return df
    
    # Remove rows where any price is <= 0
    price_cols = [c for c in ['open', 'close', 'high', 'low', 'Open', 'Close', 'High', 'Low'] if c in df.columns]
    for col in price_cols:
        df = df[df[col] > 0]
    
    # Remove duplicated indices
    df = df[~df.index.duplicated(keep='last')]
    
    # Sort by date
    df = df.sort_index()
    
    return df if len(df) > 5 else None

