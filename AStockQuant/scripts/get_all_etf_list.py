# 获取所有ETF列表
import requests
import json

def _get_no_proxy_session():
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": None, "https": None}
    return session

print("获取所有ETF列表...")

# 东方财富ETF列表API (HTTP)
url = "http://push2.eastmoney.com/api/qt/clist/get"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/center/gridlist.html#fund_etf"
}

params = {
    "pn": 1,
    "pz": 5000,  # 获取更多
    "po": 1,
    "np": 1,
    "ut": "bd1d9ddb04089700cf9c27f6f7426281",
    "fltt": 2,
    "invt": 2,
    "fid": "f12",
    "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024,b:MK0827",
    "fields": "f12,f14"
}

try:
    session = _get_no_proxy_session()
    resp = session.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    etf_list = []
    diff = data.get("data", {}).get("diff", [])
    
    for item in diff:
        code = item.get("f12", "")
        name = item.get("f14", "")
        if code:
            etf_list.append({"code": code, "name": name})
    
    print(f"获取到 {len(etf_list)} 只ETF")
    
    # 保存ETF列表
    with open(r"F:\_K-lineChartAnalysis\AStockQuant\all_etf_list.json", "w", encoding="utf-8") as f:
        json.dump(etf_list, f, ensure_ascii=False, indent=2)
    
    print("ETF列表已保存")
    
    # 打印前20个ETF
    print("\n前20个ETF:")
    for i, etf in enumerate(etf_list[:20]):
        print(f"  {i+1}. {etf['code']} - {etf['name']}")
    
except Exception as e:
    print(f"Failed: {e}")
    
    # 备用：使用腾讯API获取ETF列表
    print("\n尝试使用腾讯API...")
    try:
        # 从新浪获取ETF列表
        url2 = "https://hq.sinajs.cn/list=fund_sh,fund_sz"
        headers2 = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn"
        }
        session2 = _get_no_proxy_session()
        resp2 = session2.get(url2, headers=headers2, timeout=15)
        print(f"Sina response: {resp2.status_code}")
    except Exception as e2:
        print(f"Sina also failed: {e2}")