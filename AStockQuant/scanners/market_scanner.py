# -*- coding: utf-8 -*-
"""
market_scanner.py — 全市场扫描器 (主入口)

组装 DataHub + 九层因子 + AI引擎 + 融合器, 执行全市场扫描。
相当于原 single_scanner.py 的 SingleScanner 类, 但解耦后更干净。
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

# 将项目根目录添加到系统路径
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if proj_root not in sys.path:
    sys.path.append(proj_root)

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.data_hub import ETFDataHub as DataHub
from core.feature_registry import FeatureRegistry
from core.config_loader import ConfigLoader
from layers.layer1_macro import MacroLayer
from layers.layer2_rules import RulesLayer
from layers.layer3_sector import SectorLayer
from layers.layer4_capital import CapitalLayer
from layers.layer5_sentiment import SentimentLayer
from layers.layer6_price_vol import PriceVolumeLayer, RPSCalculator
from layers.layer7_technical import TechnicalLayer
from layers.layer8_micro import BeliefLayer
from scanners.fusion_engine import FusionEngine, MarketRegime, SignalType
from models.deep_learning import (
    DeepLearningSignalEngine,
    TemporalEnsembleSignalEngine,
    build_feature_label_dataset,
    build_sequence_dataset,
    extract_latest_features,
    extract_latest_sequence,
)
from models.reinforcement import (
    RiskAwareReinforcementAllocator,
    PPOAllocationEngine,
    build_ppo_inputs,
)


@dataclass
class ScanResult:
    """扫描结果数据类"""
    symbol: str
    signal: str
    confidence: float
    composite_score: float
    ml_prob: float
    dl_prob: float
    seq_prob: float
    model_uncertainty: float
    rps_value: float
    vcp_quality: float
    pattern_score: float
    latest_price: float
    change_pct: float
    max_buyable_shares: int
    # 新增: 各层核心信号
    macro_regime_score: float = 0.0      # Layer1: 宏观环境分
    sector_is_hot: bool = False          # Layer3: 所属板块是否热
    sector_momentum: float = 0.5         # Layer3: 板块动量得分
    sector_is_leader: bool = False       # Layer3: 是否为龙头股
    sector_phase: str = "neutral"        # Layer3: 板块轮动阶段
    capital_score: float = 0.5           # Layer4: 资金综合得分
    capital_lhb: bool = False            # Layer4: 是否上龙虎榜
    sent_market_score: float = 0.5       # Layer5: 市场情绪分
    sent_is_limit_up: bool = False       # Layer5: 今日涨停
    sent_consecutive_days: float = 1.0   # Layer5: 连板天数
    rl_weight: float = 0.0
    ppo_weight: float = 0.0
    # === 贝叶斯信念因子 ===
    belief_posterior: float = 0.5          # P(H|E) 后验概率
    belief_level: str = "neutral"         # 信念水平标签
    belief_kl: float = 0.0               # KL 散度（漂移度量）
    belief_evidence_count: float = 0.0     # 累计证据数
    belief_signal: str = "NEUTRAL"         # 信念信号（贝叶斯直接分类）
    bayesian_confidence: float = 0.5       # 贝叶斯融合置信度
    bayesian_score: float = 0.5           # 贝叶斯综合得分


class MarketScanner:
    """全市场扫描器 — 系统主入口"""

    def __init__(self, budget: float = None, max_price: float = None,
                 enable_macro: bool = None, enable_sector: bool = None,
                 enable_capital: bool = None, enable_sentiment: bool = None,
                 config_path: str = None):
        # 加载配置
        self.config = ConfigLoader.get_instance(config_path)
        
        # 从配置文件读取交易参数（如果未显式传入）
        trading_cfg = self.config.get_trading_config()
        if budget is None:
            budget = trading_cfg["budget"]
        if max_price is None:
            max_price = trading_cfg["max_price"]
        
        # 从配置文件读取层级开关（如果未显式传入）
        layer_cfg = self.config.get_layer_config()
        if enable_macro is None:
            enable_macro = layer_cfg["macro"]
        if enable_sector is None:
            enable_sector = layer_cfg["sector"]
        if enable_capital is None:
            enable_capital = layer_cfg["capital"]
        if enable_sentiment is None:
            enable_sentiment = layer_cfg["sentiment"]
        
        # 核心组件
        self.data_hub = DataHub()
        self.registry = FeatureRegistry()
        
        # 从配置文件读取融合权重
        fusion_weights = self.config.get_fusion_weights()
        self.fusion = FusionEngine(
            ml_weight=fusion_weights["ml_weight"],
            rps_weight=fusion_weights["rps_weight"],
            vcp_weight=fusion_weights["vcp_weight"],
            pattern_weight=fusion_weights["pattern_weight"],
        )

        # 从配置文件读取制度层配置
        rules_config = self.config.get("rules_config", {})
        allowed_boards = tuple(rules_config.get("allowed_boards", ["etf_sh", "etf_sz"]))
        allowed_boards = tuple(board for board in allowed_boards if board in ("etf_sh", "etf_sz")) or ("etf_sh", "etf_sz")
        exclude_risks = tuple(rules_config.get("exclude_risk_levels", ["st", "delisting", "suspended"]))
        
        # 从配置文件读取板块层配置
        sector_config = self.config.get("sector_config", {})
        # （预留：未来可在此处使用配置参数）

        # 层级注册 (制度层 / 量价层 / 技术层 / 贝叶斯信念层 — 始终启用)
        self.rules_layer = RulesLayer(
            allowed_boards=allowed_boards,
            exclude_risk_levels=exclude_risks
        )
        self.pv_layer = PriceVolumeLayer()
        self.tech_layer = TechnicalLayer()
        self.belief_layer = BeliefLayer()  # 贝叶斯信念引擎（L8）
        self.registry.register("rules", self.rules_layer)
        self.registry.register("price_vol", self.pv_layer)
        self.registry.register("technical", self.tech_layer)
        self.registry.register("belief", self.belief_layer)

        # 可选层 (网络数据依赖，可独立开关)
        self.sector_layer = SectorLayer() if enable_sector else None
        self.capital_layer = CapitalLayer() if enable_capital else None
        self.sentiment_layer = SentimentLayer() if enable_sentiment else None
        # 注意：macro_layer 始终不自动启用（执行较慢，由用户显式控制）
        self.macro_layer = MacroLayer() if enable_macro else None

        if self.macro_layer:
            self.registry.register("macro", self.macro_layer)
        if self.sector_layer:
            self.registry.register("sector", self.sector_layer)
        if self.capital_layer:
            self.registry.register("capital", self.capital_layer)
        if self.sentiment_layer:
            self.registry.register("sentiment", self.sentiment_layer)

        # AI 引擎 - 从配置文件读取参数
        model_cfg = self.config.get_model_config()
        dl_cfg = model_cfg["deep_learning"]
        temporal_cfg = model_cfg["temporal"]
        rl_cfg = model_cfg["rl_allocator"]
        ppo_cfg = model_cfg["ppo"]
        
        self.dl_engine = DeepLearningSignalEngine(epochs=dl_cfg["epochs"], lr=dl_cfg["lr"])
        self.seq_engine = TemporalEnsembleSignalEngine(
            lookback=temporal_cfg["lookback"], 
            epochs=temporal_cfg["epochs"], 
            lr=temporal_cfg["lr"], 
            mode=temporal_cfg["mode"]
        )
        self.rl_allocator = RiskAwareReinforcementAllocator(
            alpha=rl_cfg["alpha"],
            beta=rl_cfg["beta"],
            gamma_risk=rl_cfg["gamma_risk"],
            max_weight=rl_cfg["max_weight"],
            turnover_penalty=rl_cfg["turnover_penalty"],
        )
        self.ppo_allocator = PPOAllocationEngine(
            max_weight=ppo_cfg["max_weight"],
            commission=ppo_cfg["commission"],
            risk_aversion=ppo_cfg["risk_aversion"],
        )

        self.budget = budget
        self.max_price = max_price

    def _detect_regime(self, index_df: pd.DataFrame) -> MarketRegime:
        try:
            if index_df.empty or len(index_df) < 60:
                return MarketRegime.NEUTRAL
            close = index_df["close"]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1]
            cur = close.iloc[-1]
            if cur > ma20 > ma60:
                return MarketRegime.BULL
            elif cur < ma20 < ma60:
                return MarketRegime.BEAR
            return MarketRegime.NEUTRAL
        except Exception:
            return MarketRegime.NEUTRAL

    def _analyze_stock(
        self,
        symbol: str,
        df: pd.DataFrame,
        market_prices_df: pd.DataFrame,
        regime: MarketRegime,
        dl_prob: Optional[float] = None,
        seq_prob: Optional[float] = None,
    ) -> Optional[ScanResult]:
        try:
            if df.empty or len(df) < 100:
                return None

            # 通过 FeatureRegistry 提取所有层的特征
            ctx = {"market_prices_df": market_prices_df}
            features = self.registry.extract_features(symbol, df, ctx)

            # 如果制度层拦截, 直接跳过
            if features.get("rules_pass") is False:
                return None

            # ---- 第 6 / 7 层核心因子 ----
            rps_value = float(features.get("pv_rps_50", 50.0))
            vcp_quality = float(features.get("pv_vcp_quality", 0.0))
            pattern_score = float(features.get("tech_pattern_score", 0.0))

            # ---- 第 8 层: 贝叶斯信念因子 ----
            belief_posterior = float(features.get("belief_posterior", 0.5))
            belief_level = str(features.get("belief_level", "neutral"))
            belief_kl = float(features.get("belief_kl", 0.0))
            belief_evidence_count = float(features.get("belief_evidence_count", 0.0))
            belief_confidence = float(features.get("belief_confidence", 0.5))

            # ---- 贝叶斯融合 ----
            bay_sig, bay_conf, bay_score = self.fusion.fuse_bayesian(
                belief_posterior=belief_posterior,
                rps_value=rps_value,
                vcp_quality=vcp_quality,
                pattern_score=pattern_score,
                regime=regime,
            )

            # ---- 第 1 层: 宏观环境调节 ----
            macro_score = float(features.get("macro_regime_score", 0.0))

            # ---- 第 3 层: 板块热度调节 ----
            sector_is_hot = bool(features.get("sector_is_hot", False))
            sector_momentum = float(features.get("sector_combined_score", 0.5))
            sector_is_leader = bool(features.get("sector_is_leader", False))
            sector_phase = str(features.get("sector_phase", "neutral"))

            # ---- 第 4 层: 资金层调节 ----
            capital_score = float(features.get("capital_score", 0.5))
            capital_lhb = bool(features.get("capital_lhb_on_board", False))

            # ---- 第 5 层: 情绪层调节 ----
            sent_market_score = float(features.get("sentiment_score", 0.5))
            sent_is_limit_up = bool(features.get("sent_is_limit_up", False))
            sent_consecutive = int(features.get("sent_consecutive_days", 0))

            # ML 基础概率
            base_ml = 0.5 + (rps_value - 50) / 200 + pattern_score / 40
            # 叠加板块、资金、情绪、宏观的调节量
            base_ml += macro_score * 0.05
            base_ml += (sector_momentum - 0.5) * 0.12  # 提高板块权重
            base_ml += (capital_score - 0.5) * 0.08
            base_ml += (sent_market_score - 0.5) * 0.06
            base_ml = max(0.1, min(0.9, base_ml))

            dl_p = base_ml if dl_prob is None else max(0.01, min(0.99, float(dl_prob)))
            seq_p = dl_p if seq_prob is None else max(0.01, min(0.99, float(seq_prob)))

            if regime == MarketRegime.BULL:
                wb, wd, ws = 0.20, 0.30, 0.50
            elif regime == MarketRegime.BEAR:
                wb, wd, ws = 0.45, 0.35, 0.20
            else:
                wb, wd, ws = 0.30, 0.35, 0.35

            ml_prob = max(0.01, min(0.99, wb * base_ml + wd * dl_p + ws * seq_p))
            uncertainty = abs(dl_p - seq_p)

            signal, conf, comp = self.fusion.fuse(ml_prob, rps_value, vcp_quality, pattern_score, regime)

            # === 贝叶斯信号融合（优先级更高）===
            # 如果贝叶斯信念和传统信号不一致，以贝叶斯为准
            # 贝叶斯信号通常更具概率优势（基于序贯更新）
            use_bayesian = bay_conf > conf * 0.9 and belief_evidence_count >= 3
            if use_bayesian:
                signal = bay_sig
                conf = bay_conf
                comp = bay_score

            # 热板块 + 龙头股 + 机构龙虎榜买入 → 置信度提升
            if sector_is_hot:
                conf = min(1.0, conf + 0.06)  # 提高热门板块加成
            if sector_is_leader:
                conf = min(1.0, conf + 0.08)  # 龙头股额外加成
            if capital_lhb:
                conf = min(1.0, conf + 0.08)
            # 市场情绪极差时降低置信度
            if sent_market_score < 0.3:
                conf = max(0.0, conf - 0.05)

            # KL 漂移告警：信念与先验严重偏离，触发降权
            if belief_kl > 0.15:
                conf = max(0.0, conf - 0.10)
            elif belief_kl > 0.05:
                conf = max(0.0, conf - 0.03)

            # 板块轮动阶段调整
            if sector_phase == "warming":  # 启动期，增加信心
                conf = min(1.0, conf + 0.03)
            elif sector_phase == "hot":    # 高潮期，谨慎
                conf = max(0.0, conf - 0.02)
            elif sector_phase == "cooling": # 退潮期，大幅降低
                conf = max(0.0, conf - 0.08)

            close = df["close"]
            price = float(close.iloc[-1])
            chg = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
            shares = int(self.budget / price / 100) * 100 if price <= self.max_price else 0

            return ScanResult(
                symbol=symbol, signal=signal.name, confidence=conf,
                composite_score=comp, ml_prob=round(ml_prob, 3),
                dl_prob=round(dl_p, 3), seq_prob=round(seq_p, 3),
                model_uncertainty=round(uncertainty, 3),
                rps_value=rps_value, vcp_quality=round(vcp_quality, 2),
                pattern_score=round(pattern_score, 2),
                latest_price=round(price, 2), change_pct=round(chg, 2),
                max_buyable_shares=shares,
                macro_regime_score=round(macro_score, 3),
                sector_is_hot=sector_is_hot,
                sector_momentum=round(sector_momentum, 3),
                sector_is_leader=sector_is_leader,
                sector_phase=sector_phase,
                capital_score=round(capital_score, 3),
                capital_lhb=capital_lhb,
                sent_market_score=round(sent_market_score, 3),
                sent_is_limit_up=sent_is_limit_up,
                sent_consecutive_days=sent_consecutive,
                # 贝叶斯因子
                belief_posterior=round(belief_posterior, 4),
                belief_level=belief_level,
                belief_kl=round(belief_kl, 4),
                belief_evidence_count=belief_evidence_count,
                belief_signal=bay_sig.name,
                bayesian_confidence=round(bay_conf, 3),
                bayesian_score=round(bay_score, 3),
            )
        except Exception as e:
            print(f"  分析 {symbol} 失败: {e}")
            return None

    def scan(self, top_n: int = 30, min_confidence: float = 0.55) -> pd.DataFrame:
        """执行全市场扫描"""
        print("=" * 60)
        print("AStockQuant 全市场扫描")
        print("=" * 60)
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"预算: {self.budget}元 | 最高价: {self.max_price}元")
        print(f"已注册层: {self.registry.active_layers}")
        print()

        # 1. 股票列表
        stock_list = self.data_hub.get_etf_list(top_n)
        print(f"扫描股票数: {len(stock_list)}")

        # 2. 指数 & 市场环境
        index_df = self.data_hub.get_index_data("000300")
        regime = self._detect_regime(index_df)
        print(f"市场环境: {regime.value}")

        # 3. 批量数据
        print("获取市场数据...")
        stock_data_map = self.data_hub.batch_stock_data(stock_list)
        market_prices = {s: d["close"] for s, d in stock_data_map.items()}
        market_prices_df = pd.DataFrame(market_prices) if market_prices else pd.DataFrame()

        # 4. 训练 DL
        print("训练深度学习模型...")
        X_tr, y_tr = build_feature_label_dataset(stock_data_map)
        r1 = self.dl_engine.fit(X_tr, y_tr)
        print(f"  DL: trained={r1.trained} backend={r1.backend} samples={r1.samples}")

        # 5. 训练序列模型
        print("训练时序模型...")
        X_seq, y_seq = build_sequence_dataset(stock_data_map)
        r2 = self.seq_engine.fit(X_seq, y_seq)
        print(f"  SEQ: trained={r2.trained} backend={r2.backend} samples={r2.samples}")

        # 6. 预测
        dl_probs, seq_probs = {}, {}
        for sym, df in stock_data_map.items():
            dl_probs[sym] = self.dl_engine.predict_proba(extract_latest_features(df))
            seq_probs[sym] = self.seq_engine.predict_proba(extract_latest_sequence(df))

        # 7. 分析
        print("\n分析股票...")
        results = []
        for sym in tqdm(stock_data_map.keys(), desc="分析中"):
            r = self._analyze_stock(
                sym, stock_data_map[sym], market_prices_df, regime,
                dl_probs.get(sym), seq_probs.get(sym),
            )
            if r:
                results.append(r)

        if not results:
            print("无有效结果")
            return pd.DataFrame()

        rdf = pd.DataFrame([vars(r) for r in results])

        # 8. RL 分配
        for _, row in rdf.iterrows():
            self.rl_allocator.warmup_with_daily_change(row["symbol"], float(row["change_pct"]))

        score_map = {
            row["symbol"]: float(0.6 * row["confidence"] + 0.4 * row["ml_prob"])
            for _, row in rdf.iterrows()
            if row["signal"] in ["STRONG_BUY", "BUY"]
        }
        risk_map = {
            row["symbol"]: float(row["model_uncertainty"] + max(0, 0.3 - row["vcp_quality"] / 100))
            for _, row in rdf.iterrows() if row["symbol"] in score_map
        }
        rl_w = self.rl_allocator.allocate_with_risk(score_map, risk_map)
        rdf["rl_weight"] = rdf["symbol"].map(rl_w).fillna(0.0)

        # 9. PPO 分配
        ppo_data = build_ppo_inputs(rdf, stock_data_map)
        ppo_syms = ppo_data["symbols"]
        if len(ppo_syms) >= 2 and len(ppo_data["state_matrix"]) > 30:
            self.ppo_allocator.train(ppo_data["state_matrix"], ppo_data["returns_matrix"])
            ppo_w = self.ppo_allocator.infer_weights(
                ppo_syms, ppo_data["latest_state"],
                {s: score_map.get(s, 0.5) for s in ppo_syms},
            )
        else:
            ppo_w = {}
        rdf["ppo_weight"] = rdf["symbol"].map(ppo_w).fillna(0.0)

        # 10. 最终排名
        rdf["final_weight"] = 0.7 * rdf["ppo_weight"] + 0.3 * rdf["rl_weight"]
        rdf["ai_rank_score"] = (
            0.45 * rdf["confidence"] + 0.30 * rdf["ml_prob"]
            + 0.25 * rdf["final_weight"] - 0.10 * rdf["model_uncertainty"]
        )
        rdf = rdf.sort_values(["signal", "ai_rank_score", "composite_score"], ascending=[False, False, False])

        # 11. 输出
        buys = rdf[
            rdf["signal"].isin(["STRONG_BUY", "BUY"])
            & (rdf["confidence"] >= min_confidence)
            & (rdf["max_buyable_shares"] > 0)
        ]
        print(f"\n买入信号: {len(buys)} 只")
        if len(buys) > 0:
            print("\n【推荐买入股票】")
            print("-" * 60)
            for _, row in buys.head(10).iterrows():
                print(f"  {row['symbol']}: {row['signal']}  置信度={row['confidence']:.1%}  "
                      f"AI={row['ai_rank_score']:.3f}  价格={row['latest_price']:.2f}元")

        # 12. 贝叶斯漂移告警
        drift_alerts = self.belief_layer.get_drift_alerts(min_kl=0.05)
        if drift_alerts:
            print(f"\n【信念漂移告警】{len(drift_alerts)} 只标的")
            for alert in drift_alerts[:5]:
                print(f"  {alert['symbol']}: KL={alert['kl']:.3f}  "
                      f"后验={alert['posterior']:.3f} 先验={alert['base_prior']:.3f}  "
                      f"漂移={alert['drift_pct']:+.1f}%  级别={alert['level']}")

        # 13. 全市场贝叶斯汇总
        belief_summary = self.belief_layer.get_market_summary()
        print(f"\n【贝叶斯市场汇总】")
        print(f"  标的数: {belief_summary['count']}  "
              f"均值后验: {belief_summary['mean_posterior']:.3f}  "
              f"均值KL: {belief_summary['mean_kl']:.4f}")
        print(f"  看涨: {belief_summary['bullish_count']}  "
              f"看跌: {belief_summary['bearish_count']}  "
              f"强烈看涨: {belief_summary['strongly_bullish_count']}  "
              f"强烈看跌: {belief_summary['strongly_bearish_count']}")

        out_dir = os.path.dirname(os.path.dirname(__file__))
        out_file = os.path.join(out_dir, f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        rdf.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"\n结果已保存: {out_file}")
        return rdf

    # ==================== 诊断方法 ====================

    def diagnose(self) -> dict:
        """
        诊断扫描器状态（整合自 test_layer_activation.py）。

        返回：
            dict: {
                "budget": float,
                "max_price": float,
                "active_layers": List[str],
                "layer_objects": Dict[str, str],
                "ai_engines": Dict[str, str],
                "fusion_weights": Dict[str, float],
            }
        """
        layer_objs = {
            "macro": self.macro_layer,
            "sector": self.sector_layer,
            "capital": self.capital_layer,
            "sentiment": self.sentiment_layer,
            "rules": self.rules_layer,
            "price_vol": self.pv_layer,
            "technical": self.tech_layer,
            "belief": self.belief_layer,  # 贝叶斯信念层（L8）
        }
        return {
            "budget": self.budget,
            "max_price": self.max_price,
            "active_layers": self.registry.active_layers,
            "layer_objects": {
                name: "[OK]  已加载" if obj is not None else "[!!] 未加载"
                for name, obj in layer_objs.items()
            },
            "ai_engines": {
                "dl_engine_backend": getattr(self.dl_engine, "backend", "unknown"),
                "seq_engine_backend": getattr(self.seq_engine, "backend", "unknown"),
                "rl_allocator_type": type(self.rl_allocator).__name__,
                "ppo_backend": getattr(self.ppo_allocator, "backend", "unknown"),
            },
            "fusion_weights": {
                "ml_weight": self.fusion.ml_w,
                "rps_weight": self.fusion.rps_w,
                "vcp_weight": self.fusion.vcp_w,
                "pattern_weight": self.fusion.pat_w,
            },
        }

    def print_diagnosis(self):
        """打印诊断报告"""
        d = self.diagnose()
        print("\n" + "=" * 60)
        print("MarketScanner 诊断报告")
        print("=" * 60)
        print(f"预算: {d['budget']} 元 | 最高价: {d['max_price']} 元")
        print("\n【已注册的层级】")
        for layer in d["active_layers"]:
            print(f"  [OK] {layer}")
        print("\n【层级对象状态】")
        for name, status in d["layer_objects"].items():
            print(f"  {name:12s}: {status}")
        print("\n【AI 引擎配置】")
        for key, val in d["ai_engines"].items():
            print(f"  {key:20s}: {val}")
        print("\n【信号融合权重】")
        for key, val in d["fusion_weights"].items():
            print(f"  {key:20s}: {val:.2f}")
        print("=" * 60)


# ==================== 主函数 ====================
def main():
    try:
        scanner = MarketScanner(budget=5000, max_price=45.0)
        return scanner.scan(top_n=30, min_confidence=0.55)
    except Exception as e:
        print(f"扫描错误: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


if __name__ == "__main__":
    main()
