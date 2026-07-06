# 数据源配置与 API 密钥指南

本指南说明 AStockQuant 项目中所有数据源的配置方式，以及如何获取和填写需要 API Token 的数据源。

---

## 一、数据源降级链总览

`DataHub` 按以下顺序依次尝试获取数据，前一源失败自动降级到下一个：

```
akshare → 东方财富 → 腾讯 → 新浪 → baostock → tushare → efinance
→ tickflow → yfinance → pytdx
```

所有配置在 `config.yaml` 的 `data_sources` 节：

```yaml
data_sources:
  proxy:
    socks5:
      enabled: true              # 自动探测 Clash/Mihomo mixed-port
      host: "127.0.0.1"
      port: 7897
  baostock:
    enabled: true              # 免费，无需注册（通过 SOCKS5 隧道）
  tushare:
    enabled: false             # ← 改为 true 启用
    token: ""                  # ← 填入你的 Token
  efinance:
    enabled: true              # 免费，无需注册
  tsanghi:
    enabled: false             # ← 改为 true 启用
    token: ""                  # ← 填入你的 Token
  tickflow:
    enabled: true              # 免费版提供历史日K线，无需 token
  yfinance:
    enabled: true              # Yahoo Finance，免费（可能被限流）
  pytdx:
    enabled: true              # 通达信行情接口，免费（通过 SOCKS5 隧道）
  exchange_pcf:
    enabled: true              # 交易所PCF，免费
```

---

## 二、无需 Token 的数据源（开箱即用）

以下源**不需要任何注册或密钥**，`enabled: true` 即可直接使用：

| 数据源 | 安装状态 | 说明 |
|--------|---------|------|
| **akshare** | 已安装 | 主数据源，聚合东财/新浪/同花顺 |
| **东方财富直连** | 无需安装 | HTTP API，`data_hub.py` 内置 |
| **腾讯 API** | 无需安装 | HTTP API，`data_hub.py` 内置 |
| **新浪 API** | 无需安装 | HTTP API，`data_hub.py` 内置 |
| **baostock** | 已安装 | 免费A股日线，需 login/logout（自动处理，SOCKS5 隧道） |
| **efinance** | 已安装 | 东方财富封装，接口简洁 |
| **tickflow** | 已安装 | 免费版历史日K线，HTTP API，支持 ETF/股票 |
| **yfinance** | 已安装 | Yahoo Finance，HTTP API，A股用 `.SS`/`.SZ` 后缀 |
| **pytdx** | 已安装 | 通达信行情接口，SOCKS5 隧道，原生 TCP 协议 |
| **交易所 PCF** | 无需安装 | 上交所/深交所官方清单（`scripts/cache_etf_pcf.py`） |

### baostock 使用说明
- 自动在首次调用时 `login()`，在 `DataHub.close()` 时 `logout()`
- 支持 ETF（`sh.510300`）和股票（`sh.600519`）
- 数据为不复权日线，如需前复权请用 akshare 或腾讯源

### efinance 使用说明
- 通过 `ef.stock.get_quote_history()` 获取
- 支持股票和 ETF，返回中文列名（已自动归一化）

### tickflow 使用说明
- 免费版提供历史日K线数据（非实时），无需注册
- 符号格式：`510300.SH`（沪市）、`159915.SZ`（深市）
- HTTP API，不受代理环境影响

### yfinance 使用说明
- Yahoo Finance 国际版，A股用 `.SS`（沪）/`.SZ`（深）后缀
- HTTP API，可能被 Yahoo 限流（返回空数据时自动降级）

### pytdx 使用说明
- 通达信行情接口，原生 TCP 协议（端口 7709）
- 通过 SOCKS5 隧道连接（绕过 VMware/VPN 路由劫持）
- 部分服务器可能不返回 ETF 数据，失败时自动降级

### SOCKS5 代理隧道（baostock / pytdx 专用）

**背景**：当系统有 VMware 虚拟网卡或 VPN 劫持默认路由时，baostock/pytdx 的原生 TCP 连接会超时。DataHub 通过 Clash 的 mixed-port (SOCKS5) 隧道转发 TCP 连接绕过此问题。

**自动探测**：DataHub 启动时自动探测本地 SOCKS5 代理端口（7897/7890/1080/10808），无需手动配置。

**手动配置**（如自动探测失败）：
```yaml
data_sources:
  proxy:
    socks5:
      enabled: true
      host: "127.0.0.1"
      port: 7897                 # Clash Verge 默认 mixed-port
```

**依赖**：`pip install PySocks`（已包含在项目依赖中）

**工作原理**：
- baostock: monkey-patch `SocketUtil.connect` + `send_msg`，使用 SOCKS5 socket + 15s 超时
- pytdx: 替换 `TrafficStatSocket` 为 SOCKS5 兼容版本（继承 `socks.socksocket`）
- 两者连接失败/超时后自动禁用该源，降级链继续

### 交易所 PCF 获取
```bash
# 获取当日全部 ETF PCF 清单
python scripts/cache_etf_pcf.py

# 指定 ETF 代码
python scripts/cache_etf_pcf.py --codes 510300,159915

# 只获取上交所
python scripts/cache_etf_pcf.py --market sse
```
产出保存在 `external_cache/pcf/etf_pcf_YYYYMMDD.json`。

---

## 三、需要 Token 的数据源

### 1. Tushare Pro（推荐）

**简介**：专业量化数据平台，数据质量高，覆盖全面。

**免费额度**：
- 日调用 2000 次
- 每分钟 50 次
- 支持 ETF 日线、股票日线、财务数据等

**获取 Token 步骤**：

1. **注册账号**
   - 访问 https://tushare.pro/register
   - 填写手机号/邮箱注册

2. **获取 Token**
   - 登录后访问 https://tushare.pro/user/token
   - 复制你的 API Token（一串字符，如 `a1b2c3d4e5...`）

3. **填写到 config.yaml**
   ```yaml
   data_sources:
     tushare:
       enabled: true                    # ← 改为 true
       token: "你的Token粘贴在这里"      # ← 粘贴 Token
   ```

4. **验证**
   ```bash
   cd AStockQuant
   python -c "
   from core.config_loader import load_config
   cfg = load_config()
   dsc = cfg.get_data_source_config()
   print('Tushare enabled:', dsc['tushare']['enabled'])
   print('Token length:', len(dsc['tushare']['token']))
   "
   ```

**注意事项**：
- 免费版有调用频率限制（每分钟50次），DataHub 已内置降级机制
- 部分高级接口需要积分（如分钟数据需 2000 积分），日线数据免费可用
- Token 是个人凭证，切勿提交到 git（`config.yaml` 已在 .gitignore 之外，建议将 token 部分改为环境变量读取）

---

### 2. Tsanghi 沧海数据（可选）

**简介**：全球 15000+ ETF 数据，支持 JSON/CSV 格式。

**获取 Token 步骤**：

1. **注册账号**
   - 访问 https://tsanghi.com
   - 注册账号

2. **获取 Token**
   - 登录后在个人中心找到 API Token
   - 复制 Token

3. **填写到 config.yaml**
   ```yaml
   data_sources:
     tsanghi:
       enabled: true                    # ← 改为 true
       token: "你的TsanghiToken"        # ← 粘贴 Token
   ```

---

## 四、使用环境变量（推荐的安全做法）

如果你不想把 Token 明文写在 `config.yaml` 中（避免误提交到 git），可以使用环境变量：

### 方法：在 data_hub.py 初始化时传入

```python
import os
from AStockQuant.core.config_loader import load_config
from AStockQuant.core.data_hub import DataHub

cfg = load_config()
dsc = cfg.get_data_source_config()

# 用环境变量覆盖 token
dsc["tushare"]["token"] = os.getenv("TUSHARE_TOKEN", dsc["tushare"]["token"])
dsc["tsanghi"]["token"] = os.getenv("TSANGHI_TOKEN", dsc["tsanghi"]["token"])

hub = DataHub(data_source_config=dsc)
```

### 设置环境变量

**Windows PowerShell（临时）**：
```powershell
$env:TUSHARE_TOKEN = "你的Token"
python your_script.py
```

**Windows PowerShell（永久）**：
```powershell
[Environment]::SetEnvironmentVariable("TUSHARE_TOKEN", "你的Token", "User")
```

**Linux / macOS**：
```bash
export TUSHARE_TOKEN="你的Token"
```

---

## 五、各数据源能力对比

| 数据源 | 需Token | ETF日线 | 股票日线 | 实时行情 | 历史长度 | 稳定性 | 代理需求 |
|--------|---------|---------|---------|---------|---------|--------|---------|
| akshare | 否 | ✅ | ✅ | ✅ | 5-10年 | ★★★★ | HTTP（系统代理） |
| 东方财富 | 否 | ✅ | ✅ | ✅ | 5-10年 | ★★★★ | HTTP（系统代理） |
| 腾讯 | 否 | ✅ | ✅ | ✅ | 5-8年 | ★★★ | HTTP（系统代理） |
| 新浪 | 否 | ✅ | ✅ | ✅ | 3-5年 | ★★★ | HTTP（系统代理） |
| **baostock** | 否 | ✅ | ✅ | ❌ | 8-10年 | ★★★★ | SOCKS5 隧道 |
| **tushare** | 是 | ✅ | ✅ | ✅ | 10年+ | ★★★★★ | HTTP（系统代理） |
| **efinance** | 否 | ✅ | ✅ | ✅ | 5-10年 | ★★★ | HTTP（系统代理） |
| **tickflow** | 否 | ✅ | ✅ | ❌ | 30天 | ★★★★ | HTTP（系统代理） |
| **yfinance** | 否 | ✅ | ✅ | ✅ | 5-10年 | ★★ | HTTP（系统代理） |
| **pytdx** | 否 | ✅ | ✅ | ✅ | 800根 | ★★ | SOCKS5 隧道 |
| **tsanghi** | 是 | ✅ | ❌ | ❌ | 10年+ | ★★★ | HTTP（系统代理） |

---

## 六、如何验证数据源是否正常工作

```bash
cd AStockQuant

# 测试所有已启用的数据源
python -c "
from core.config_loader import load_config
from core.data_hub import DataHub

cfg = load_config()
dsc = cfg.get_data_source_config()
print('=== 数据源配置 ===')
for name, conf in dsc.items():
    print(f'  {name}: enabled={conf.get(\"enabled\", \"N/A\")}')

hub = DataHub(data_source_config=dsc)
df = hub.get_stock_data_ex('510300', days=10)
print(f'\n=== 510300 最近10天数据 ===')
print(df)
print(f'\n数据源: {len(df)} 行')
hub.close()
"
```

---

## 七、常见问题

### Q: baostock 报 "login failed"？
A: 在 VMware/VPN 环境下，baostock 的原生 TCP 连接可能被路由劫持。DataHub 通过 SOCKS5 隧道（Clash mixed-port 7897）转发连接。如果仍失败（协议层不兼容），DataHub 会自动禁用 baostock 并降级到下一个源。检查 Clash 是否运行在 `127.0.0.1:7897`。

### Q: pytdx 返回 0 条数据？
A: pytdx 通过 SOCKS5 隧道连接通达信服务器成功，但部分服务器可能不返回 ETF 历史数据。DataHub 会自动降级到下一个源。如需 pytdx 数据，尝试更换服务器 IP 或关闭 VPN 后重试。

### Q: yfinance 报 "Too Many Requests"？
A: Yahoo Finance 有严格的请求频率限制。DataHub 会捕获限流错误并自动降级。如需大量获取数据，建议优先使用 akshare 或 tickflow。

### Q: 如何确认 SOCKS5 代理是否工作？
A: 启动 DataHub 时会打印 `[DataHub] SOCKS5 代理已启用: 127.0.0.1:7897`。如未打印，检查 Clash 是否运行、端口是否正确。也可手动测试：
```bash
python -c "import socks; s=socks.socksocket(); s.set_proxy(socks.SOCKS5,'127.0.0.1',7897); s.settimeout(5); s.connect(('baostock.com',9001)); print('OK', s.getpeername()); s.close()"
```

### Q: tushare 报 "权限不足"？
A: 免费版有接口权限限制。ETF 日线 (`fund_daily`) 免费可用，分钟数据需要积分。检查 https://tushare.pro/document/1?doc_id=290 的权限说明。

### Q: 如何添加更多数据源？
A: 在 `data_hub.py` 中新增 `_fetch_xxx()` 方法，并在 `get_stock_data_ex()` 的降级链中添加调用。参考 `_fetch_baostock()` 的实现模式。

---

*文档更新时间: 2026-06-21*
