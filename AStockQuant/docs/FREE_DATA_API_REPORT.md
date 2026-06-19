# A股/ETF免费数据API调研报告

## 📊 一、Python开源库

### 1. AkShare (推荐⭐⭐⭐⭐⭐)

**定位**: 国产最强金融数据Python库，完全免费开源

**特点**:
- 无需注册，即装即用
- 数据源覆盖新浪财经、东方财富、同花顺等
- 支持A股、ETF、期货、期权、基金、债券、宏观经济等
- 支持实时行情和历史K线数据

**安装**:
```bash
pip install akshare
```

**ETF数据接口示例**:
```python
import akshare as ak

# 方法1: 东方财富 ETF历史数据 (推荐)
df = ak.fund_etf_hist_em(symbol="510300", adjust="qfq")
# 返回: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 涨跌幅等

# 方法2: 新浪ETF列表
etf_list = ak.fund_etf_category_sina()

# 方法3: 东方财富ETF实时数据
etf_spot = ak.fund_etf_fund_spot_em()

# 方法4: 股票日线数据 (包含ETF)
stock_df = ak.stock_zh_a_hist(symbol="510300", period="daily", 
                               start_date="20200101", end_date="20250531", 
                               adjust="qfq")
```

**数据源对比**:
| 接口函数 | 数据源 | 特点 |
|---------|--------|------|
| `fund_etf_hist_em` | 东方财富 | 历史K线，稳定 |
| `fund_etf_fund_daily` | 东方财富 | 日线数据 |
| `fund_etf_category_sina` | 新浪财经 | ETF列表 |
| `stock_zh_a_hist` | 网易财经 | 股票/ETF历史 |

---

### 2. Tushare (推荐⭐⭐⭐⭐)

**定位**: 专业量化数据平台，数据最全

**特点**:
- 需要注册获取API Token
- 免费版有调用限制（日2000次/分钟50次）
- 数据质量高，覆盖全面

**安装**:
```bash
pip install tushare
```

**使用示例**:
```python
import tushare as ts

# 初始化 (需要注册获取token)
pro = ts.pro_api('your_token_here')

# 获取ETF日线数据
df = pro.fund_daily(ts_code='510300.SH', start_date='20200101', end_date='20250531')
```

**注册地址**: https://tushare.pro/register

---

### 3. Baostock (推荐⭐⭐⭐)

**定位**: 免费A股数据，稳定可靠

**特点**:
- 无需Token，但需显式登录/登出
- 数据涵盖沪深A股
- 适合日线级别数据需求

**安装**:
```bash
pip install baostock
```

**使用示例**:
```python
import baostock as bs

# 登录
lg = bs.login()
print(lg.error_code, lg.error_msg)

# 获取日线数据
rs = bs.query_history_k_data_plus("sh.510300",
    "date,code,open,high,low,close,volume",
    start_date='2020-01-01', end_date='2025-05-31',
    frequency="d")

# 登出
bs.logout()
```

---

### 4. yfinance (国际ETF专用)

**定位**: Yahoo Finance API，适合美股/港股ETF

**安装**:
```bash
pip install yfinance
```

**使用示例**:
```python
import yfinance as yf

# 获取国际ETF
spy = yf.Ticker("SPY")
hist = spy.history(start="2020-01-01", end="2025-05-31")
```

---

## 📡 二、直接API接口

### 1. 新浪财经API

**接口地址**: `http://hq.sinajs.cn/list=sh600519`

**数据内容**: 实时价格、成交量、涨跌幅等

**使用示例**:
```python
import requests

# 实时行情
codes = ['sh510300', 'sz159915']  # sh=上海, sz=深圳
url = f'http://hq.sinajs.cn/list={",".join(codes)}'
headers = {'Referer': 'http://finance.sina.com.cn'}
response = requests.get(url, headers=headers)
```

**注意**: 频繁请求可能被封IP，建议加入延时

---

### 2. 腾讯证券API

**接口地址**: `https://qt.gtimg.cn/q=sz000001`

**数据内容**: A股/港股实时行情

**使用示例**:
```python
import requests

url = 'https://qt.gtimg.cn/q=sz510300,sh510500'
response = requests.get(url)
```

---

### 3. 东方财富直连API

**基础URL**: `https://api.doanything.com/`

**特点**: 需要开发者自行申请

---

### 4. Tsanghi API (全球ETF)

**接口地址**: `https://tsanghi.com/api/fin/etf/XSHG/daily`

**特点**:
- 覆盖全球15000+ETF
- 支持JSON/CSV格式
- 需要注册获取Token

**使用示例**:
```python
import requests
import os

token = os.getenv('ETF_API_TOKEN')  # 注册获取
url = f'https://tsanghi.com/api/fin/etf/XSHG/daily?token={token}&ticker=512480'
response = requests.get(url)
```

---

## 🔧 三、量化终端内置API

### 1. QMT (迅投QMT)

**特点**:
- 专业量化交易终端
- 内置Python环境
- 支持ETF分钟级/Tick级数据

**数据接口**:
```python
from xtquant import xtdata

# 下载历史数据
xtdata.download_history_data('510300.SH')

# 获取分钟数据
data = xtdata.get_market_data(['close'], ['510300.SH'], 
                               start_date='20200101', end_date='20250531')
```

---

### 2. MiniQMT

**特点**:
- 国金证券量化终端
- 支持增量数据下载
- 高精度历史行情

---

## 📋 四、数据接口汇总表

| 数据源 | 是否免费 | 是否需要注册 | 数据类型 | 稳定性 | 推荐度 |
|--------|---------|-------------|----------|--------|--------|
| **AkShare** | ✅ 免费 | ❌ 不需要 | 全市场 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Tushare** | ✅ 免费(有限制) | ✅ 需要Token | 全市场 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Baostock** | ✅ 免费 | ❌ 不需要 | A股日线 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **新浪API** | ✅ 免费 | ❌ 不需要 | 实时行情 | ⭐⭐⭐ | ⭐⭐⭐ |
| **腾讯API** | ✅ 免费 | ❌ 不需要 | 实时行情 | ⭐⭐⭐ | ⭐⭐⭐ |
| **Tsanghi** | ✅ 免费(有限制) | ✅ 需要Token | 全球ETF | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **yfinance** | ✅ 免费 | ❌ 不需要 | 美股/港股 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 🎯 五、推荐使用方案

### 方案1: AkShare单一方案 (推荐个人用户)
```python
import akshare as ak

# ETF历史数据
df = ak.fund_etf_hist_em(symbol="510300", adjust="qfq")
print(df.head())
```

### 方案2: AkShare + Baostock组合 (专业用户)
```python
import akshare as ak
import baostock as bs

# AkShare获取ETF
etf_df = ak.fund_etf_hist_em(symbol="510300")

# Baostock获取股票
bs.login()
stock_rs = bs.query_history_k_data_plus("sh.600519", ...)
```

### 方案3: Tushare Pro (企业级应用)
```python
import tushare as ts

pro = ts.pro_api('your_pro_token')
df = pro.fund_daily(ts_code='510300.SH')
```

---

## ⚠️ 六、常见问题处理

### 1. 网络连接问题
```python
# 设置代理或重试机制
import requests

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

# 重试装饰器
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def fetch_data():
    return ak.fund_etf_hist_em(symbol="510300")
```

### 2. 访问频率限制
```python
import time

# 添加延时避免被封
for code in codes:
    df = ak.fund_etf_hist_em(symbol=code)
    time.sleep(1)  # 每秒1次请求
```

### 3. 数据格式处理
```python
import pandas as pd

# 转换日期格式
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').sort_index()
```

---

## 📁 七、项目集成建议

针对AStockQuant项目，推荐以下数据获取策略：

```python
class DataSource:
    """多数据源ETF数据获取"""
    
    def __init__(self):
        self.sources = [
            ('akshare', self._from_akshare),
            ('baostock', self._from_baostock),
            ('tushare', self._from_tushare),
        ]
    
    def _from_akshare(self, symbol: str) -> pd.DataFrame:
        """东方财富数据源"""
        return ak.fund_etf_hist_em(symbol=symbol, adjust="qfq")
    
    def _from_baostock(self, symbol: str) -> pd.DataFrame:
        """Baostock数据源"""
        # 需要格式转换 sh.510300 -> 510300
        bs.login()
        code = f"sh.{symbol}" if symbol.startswith('5') else f"sz.{symbol}"
        rs = bs.query_history_k_data_plus(code, ...)
        bs.logout()
        return pd.DataFrame(rs.data)
    
    def get_etf_data(self, symbol: str, fallback: bool = True) -> pd.DataFrame:
        """优先主数据源，失败时自动切换"""
        for name, func in self.sources:
            try:
                df = func(symbol)
                if df is not None and len(df) > 0:
                    print(f"成功从 {name} 获取数据")
                    return df
            except Exception as e:
                print(f"{name} 获取失败: {e}")
                continue
        
        raise ConnectionError("所有数据源均无法获取数据")
```

---

*报告生成时间: 2026-05-31*
*数据来源: 公开技术文档和网络搜索结果*