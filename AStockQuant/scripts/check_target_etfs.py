"""
检查特定ETF的数据覆盖
"""
import os
import pandas as pd

# 我们用新浪API下载的关键ETF
target_etfs = ['510300', '510500', '159915', '512000', '512010', '159941', 
               '513500', '518880', '510050', '512200', '159928']

cache_dir = 'data_cache'

print("检查目标ETF的日期范围:\n")
for etf in target_etfs:
    filepath = os.path.join(cache_dir, f'{etf}.csv')
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            min_date = df['date'].min()
            max_date = df['date'].max()
            count = len(df)
            print(f"  {etf}: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')} ({count}条)")
    else:
        print(f"  {etf}: 文件不存在")

# 检查其他可能有历史数据的文件
print("\n\n查找有2019年之前数据的CSV文件:")
count = 0
for f in os.listdir(cache_dir):
    if f.endswith('.csv'):
        try:
            filepath = os.path.join(cache_dir, f)
            df = pd.read_csv(filepath)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                if df['date'].min().year < 2019:
                    min_d = df['date'].min()
                    max_d = df['date'].max()
                    print(f"  {f}: {min_d.strftime('%Y-%m-%d')} ~ {max_d.strftime('%Y-%m-%d')}")
                    count += 1
                    if count > 30:
                        print("  ... (显示前30个)")
                        break
        except:
            pass

if count == 0:
    print("  没有找到2019年之前的数据")
