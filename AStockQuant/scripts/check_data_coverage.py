"""
检查ETF数据覆盖情况
"""
import os
import pandas as pd
from datetime import datetime

cache_dir = 'data_cache'
etf_files = [f for f in os.listdir(cache_dir) if f.endswith('.csv')]

print(f"data_cache中有 {len(etf_files)} 个文件")

# 检查每个ETF的日期范围
date_ranges = []
for f in etf_files[:50]:  # 只检查前50个
    try:
        df = pd.read_csv(os.path.join(cache_dir, f))
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            min_date = df['date'].min()
            max_date = df['date'].max()
            count = len(df)
            date_ranges.append({
                'file': f,
                'min': min_date,
                'max': max_date,
                'count': count
            })
    except:
        pass

date_ranges.sort(key=lambda x: x['min'])

print("\n前20个ETF的日期范围:")
for d in date_ranges[:20]:
    print(f"  {d['file']}: {d['min'].strftime('%Y-%m-%d')} ~ {d['max'].strftime('%Y-%m-%d')} ({d['count']}条)")

print("\n后20个ETF的日期范围:")
for d in date_ranges[-20:]:
    print(f"  {d['file']}: {d['min'].strftime('%Y-%m-%d')} ~ {d['max'].strftime('%Y-%m-%d')} ({d['count']}条)")

# 检查2019-2024都有哪些ETF
etfs_2019 = [d['file'] for d in date_ranges if d['min'].year <= 2019]
etfs_2024 = [d['file'] for d in date_ranges if d['max'].year >= 2024]

print(f"\n2019年之前开始的ETF: {len(etfs_2019)}")
print(f"2024年仍然存在的ETF: {len(etfs_2024)}")

# 找出覆盖2019-2024完整区间的ETF
etfs_complete = [d['file'] for d in date_ranges if d['min'].year <= 2019 and d['max'].year >= 2024]
print(f"覆盖2019-2024完整区间的ETF: {len(etfs_complete)}")
print(f"这些ETF: {etfs_complete}")
