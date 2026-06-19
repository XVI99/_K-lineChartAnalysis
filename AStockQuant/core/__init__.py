# -*- coding: utf-8 -*-
"""core 子包 — 数据流、缓存、特征注册、配置加载"""

try:
    from AStockQuant.core.config_loader import ConfigLoader, load_config
    from AStockQuant.core.data_hub import ETFDataHub as DataHub
except ImportError:
    # 本地运行时使用相对导入
    from .config_loader import ConfigLoader, load_config
    from .data_hub import ETFDataHub as DataHub
from AStockQuant.core.feature_registry import FeatureRegistry
from AStockQuant.core.cache_manager import CacheManager

try:
    import akshare as ak
    _AKSHARE_AVAILABLE = True
except ImportError:
    _AKSHARE_AVAILABLE = False
    ak = None

__all__ = [
    "ConfigLoader",
    "load_config",
    "DataHub",
    "FeatureRegistry",
    "CacheManager",
    "ak",
    "_AKSHARE_AVAILABLE",
]
