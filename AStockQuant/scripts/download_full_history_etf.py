"""
下载具有完整历史数据的ETF (2019-2024)
"""
import os
import requests
import pandas as pd
import json
import time
from datetime import datetime

# 禁用代理
session = requests.Session()
session.trust_env = False
session.proxies = {"http": None, "https": None}

data_cache = 'F:/_K-lineChartAnalysis/AStockQuant/data_cache'
os.makedirs(data_cache, exist_ok=True)

# 主要宽基ETF列表 (具有完整历史数据)
broad_etfs = [
    # 沪深300系列
    '510300', '510310', '159919',
    # 中证500系列
    '510500', '159922',
    # 创业板系列
    '159915', '159949',
    # 科创50
    '588000', '588050',
    # 上证50
    '510050',
    # 纳指
    '513100', '159941',
    # 恒生
    '159920', '510900',
    # 中概
    '513050', '159941',
    # 行业ETF
    '512760',  # 芯片
    '515980',  # 人工智能
    '515050',  # 5G
    '515030',  # 新能源车
    '512690',  # 酒
    '512000',  # 证券
    '512100',  # 军工
    '515220',  # 煤炭
    '512880',  # 银行
    '512980',  # 消费
    '159869',  # 游戏
    '159805',  # 芯片
    # 债券
    '511010', '511260',
    # 黄金
    '518880', '159934',
    # 货币
    '511990',
]

def get_etf_kline_full(code, days=2000):
    """获取ETF完整K线数据"""
    try:
        # 腾讯API - 获取更长时间
        url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_dayqfq&param={code},day,,,{days},qfq"
        resp = session.get(url, timeout=15)
        text = resp.text
        
        # 解析数据
        if 'hq_str_' + code not in text:
            return None
            
        data_str = text.split('hq_str_' + code + '="')[1].split('"')[0]
        data_str = data_str.replace('day_qfq', '"day_qfq"').replace('"', '"')
        
        # 解析K线
        import re
        items = re.findall(r'\[(\d+,\d+,\d+,\d+,\d+,\d+)\]', data_str)
        if not items:
            return None
            
        records = []
        for item in items:
            parts = item.split(',')
            if len(parts) >= 6:
                timestamp = int(parts[0])
                date = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d')
                open_ = float(parts[1])
                close = float(parts[2])
                high = float(parts[3])
                low = float(parts[4])
                volume = float(parts[5])
                records.append([date, open_, close, high, low, volume])
        
        return records
    except Exception as e:
        return None

def get_etf_name(code):
    """获取ETF名称"""
    try:
        url = f"http://qt.gtimg.cn/q={code}"
        resp = session.get(url, timeout=10)
        text = resp.text
        match = re.search(f'hq_str_{code}="([^"]+)"', text)
        if match:
            parts = match.group(1).split('~')
            if len(parts) > 1:
                return parts[1]
        return code
    except:
        return code

import re

print("="*60)
print("  下载完整历史ETF数据 (2019-2024)")
print("="*60)
print("")
print(f"目标: {len(broad_etfs)} 个宽基ETF")
print(f"每个ETF: 2000天K线数据 (约8年)")
print("-"*60)

success_count = 0
results = []

for i, code in enumerate(broad_etfs):
    print(f"[{i+1}/{len(broad_etfs)}] 下载 {code}...", end=" ")
    
    data = get_etf_kline_full(code, 2000)
    
    if data and len(data) > 100:
        df = pd.DataFrame(data, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # 保存
        filepath = os.path.join(data_cache, f'{code}.csv')
        df.to_csv(filepath, index=False)
        
        # 检查数据范围
        start_year = df['date'].min().year
        end_year = df['date'].max().year
        days_count = len(df)
        
        print(f"OK ({start_year}-{end_year}, {days_count}天)")
        success_count += 1
        results.append({'code': code, 'start': start_year, 'end': end_year, 'days': days_count})
    else:
        print(f"失败")
    
    time.sleep(0.2)

print("")
print("="*60)
print(f"  下载完成: {success_count}/{len(broad_etfs)} 个ETF")
print("="*60)

# 保存结果
if results:
    print("\n成功下载的ETF:")
    print("-"*60)
    for r in sorted(results, key=lambda x: x['start']):
        print(f"  {r['code']}: {r['start']}-{r['end']} ({r['days']}天)")