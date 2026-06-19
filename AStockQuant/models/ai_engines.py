# -*- coding: utf-8 -*-
"""
ai_engines.py — AI引擎统一入口

本文件提供与原 quant_system/advanced_ai.py 的兼容性导入，
实际功能已拆分到 deep_learning.py 和 reinforcement.py。

保留此文件是为了向后兼容旧的导入路径（如 from quant_system.advanced_ai import ...）。
新代码请直接导入：
    from AStockQuant.models.deep_learning import DeepLearningSignalEngine
    from AStockQuant.models.reinforcement import PPOAllocationEngine
"""

from AStockQuant.models.deep_learning import (
    DeepLearningSignalEngine,
    TemporalEnsembleSignalEngine,
    build_feature_label_dataset,
    build_sequence_dataset,
    extract_latest_features,
    extract_latest_sequence,
    DLTrainReport,
)

from AStockQuant.models.reinforcement import (
    ReinforcementAllocator,
    RiskAwareReinforcementAllocator,
    PPOAllocationEngine,
    build_ppo_inputs,
)

# 为了向后兼容，也导出以下函数（如果存在的话）
try:
    from AStockQuant.models.deep_learning import _sigmoid, _compute_rsi
except ImportError:
    pass

__all__ = [
    # 深度学习
    "DeepLearningSignalEngine",
    "TemporalEnsembleSignalEngine",
    "build_feature_label_dataset",
    "build_sequence_dataset",
    "extract_latest_features",
    "extract_latest_sequence",
    "DLTrainReport",
    # 强化学习
    "ReinforcementAllocator",
    "RiskAwareReinforcementAllocator",
    "PPOAllocationEngine",
    "build_ppo_inputs",
]