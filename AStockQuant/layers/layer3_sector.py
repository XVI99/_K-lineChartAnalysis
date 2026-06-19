"""
Layer3 - 板块层
=====================

功能: 行业板块轮动分析，热点判断

分析:
- 标的所属板块
- 板块动量
- 板块热度
- 是否为龙头股
- 板块轮动阶段
"""

import pandas as pd
import numpy as np
from typing import Dict, List

class SectorLayer:
    """
    板块层 - 行业板块轮动分析
    热点板块中的标的更值得关注
    """
    
    # 板块映射 (简化的ETF分类)
    SECTOR_ETFS = {
        '159915': '创业板',    # 创业板
        '159919': '沪深300',   # 沪深300
        '515000': '科技',      # 科技
        '512000': '证券',      # 证券
        '512100': '军工',      # 军工
        '512760': '芯片',      # 芯片
        '515980': '人工智能',  # AI
        '515050': '5G',        # 5G
        '515030': '新能源',     # 新能源
        '512690': '消费',      # 消费/酒
    }
    
    def extract_features(self, symbol: str, df: pd.DataFrame, ctx: Dict) -> Dict:
        """提取板块层特征"""
        features = {}
        
        # 确定板块
        sector = self.SECTOR_ETFS.get(symbol, '其他')
        features['sector'] = sector
        
        if df.empty or len(df) < 20:
            return features
        
        # 计算板块动量
        close = df['close']
        ret_5d = close.pct_change(5).iloc[-1]
        ret_20d = close.pct_change(20).iloc[-1]
        ret_60d = close.pct_change(60).iloc[-1] if len(df) >= 60 else ret_20d
        
        features['sector_momentum'] = ret_20d * 100
        features['sector_momentum_short'] = ret_5d * 100
        features['sector_momentum_long'] = ret_60d * 100
        
        # 板块是否热门 (短期动量 > 5%)
        features['sector_is_hot'] = ret_5d > 0.05
        
        # 计算板块综合得分 (0-1)
        momentum_score = min(1.0, max(0.0, (ret_20d * 10 + 0.5)))
        features['sector_combined_score'] = momentum_score
        
        # 判断板块轮动阶段
        if ret_5d > 0.03 and ret_20d > 0.10:
            features['sector_phase'] = 'hot'  # 高潮期
        elif ret_5d > 0.01 and ret_20d > 0.05:
            features['sector_phase'] = 'warming'  # 启动期
        elif ret_5d < -0.03:
            features['sector_phase'] = 'cooling'  # 退潮期
        else:
            features['sector_phase'] = 'neutral'  # 中性
        
        # 是否为龙头 (动量最强的前20%)
        features['sector_is_leader'] = ret_20d > 0.15
        
        return features
    
    def get_hot_sectors(self, all_data: Dict[str, pd.DataFrame]) -> List[str]:
        """获取当前热门板块"""
        hot_sectors = []
        
        for symbol, df in all_data.items():
            if len(df) < 20:
                continue
            ret_5d = df['close'].pct_change(5).iloc[-1]
            if ret_5d > 0.05:
                sector = self.SECTOR_ETFS.get(symbol, '其他')
                if sector not in hot_sectors:
                    hot_sectors.append(sector)
        
        return hot_sectors