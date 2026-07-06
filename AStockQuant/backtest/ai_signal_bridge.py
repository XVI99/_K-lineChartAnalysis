# -*- coding: utf-8 -*-
"""
ai_signal_bridge.py — 连接 DL/SEQ 引擎与 walkforward 回测的桥接层

把 MarketScanner 里的 DeepLearningSignalEngine / TemporalEnsembleSignalEngine
接入 walk-forward 回测, 让 AI 选股模型在历史回测中被验证.

时序对齐原则 (防 look-ahead):
- fit(stock_data_map, as_of_date): 只看 <= as_of_date 的数据训练引擎
- predict_proba(stock_data_map, dates): 对每个 date, 取 df[df['date'] <= date]
  (含当日 close, 符合 "T 日 close 出信号, T+1 open 执行") 预测
- 每 fold 训练一次引擎, fold 内所有 config 共享同一份 ai_prob

详见 docs/AI_BACKTEST_INTEGRATION.md
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.deep_learning import (
    DeepLearningSignalEngine,
    TemporalEnsembleSignalEngine,
    build_feature_label_dataset,
    build_sequence_dataset,
    extract_latest_features,
    extract_latest_sequence,
)


class AISignalBridge:
    """连接 DL/SEQ 引擎与 walkforward 回测的桥接层.

    每个实例对应一个 walk-forward fold:
      bridge.fit(map, train_end)        -> 训练引擎 (只见 <= train_end 的数据)
      bridge.predict_proba(map, dates)  -> 产出 (date, code, ai_prob) 表
    """

    def __init__(
        self,
        dl_epochs: int = 20,
        seq_epochs: int = 12,
        seq_mode: str = "ensemble",
        min_samples_dl: int = 80,
        min_samples_seq: int = 140,
        horizon: int = 5,
        lookback: int = 30,
        min_window: int = 30,
        verbose: bool = True,
    ) -> None:
        self.dl_epochs = dl_epochs
        self.seq_epochs = seq_epochs
        self.seq_mode = seq_mode
        self.min_samples_dl = min_samples_dl
        self.min_samples_seq = min_samples_seq
        self.horizon = horizon
        self.lookback = lookback
        self.min_window = min_window
        self.verbose = verbose

        self.dl_engine = DeepLearningSignalEngine(epochs=dl_epochs, lr=1e-3)
        self.seq_engine = TemporalEnsembleSignalEngine(
            lookback=lookback, epochs=seq_epochs, lr=8e-4, mode=seq_mode
        )
        self._dl_trained = False
        self._seq_trained = False
        # 特征/序列缓存: fold 内同一 (code, date) 不重复 extract
        self._feat_cache: Dict[Tuple[str, pd.Timestamp], Optional[np.ndarray]] = {}
        self._seq_cache: Dict[Tuple[str, pd.Timestamp], Optional[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def fit(
        self, stock_data_map: Dict[str, pd.DataFrame], as_of_date: pd.Timestamp
    ) -> Dict[str, object]:
        """按 as_of_date 截断数据, 训练 DL + SEQ 引擎.

        严格只看 <= as_of_date 的数据, 防止 look-ahead.
        """
        t0 = time.time()
        truncated: Dict[str, pd.DataFrame] = {}
        for code, df in stock_data_map.items():
            if df is None or df.empty:
                continue
            sub = df[df["date"] <= as_of_date]
            if len(sub) < self.min_window + self.horizon + 5:
                continue
            truncated[code] = sub

        if not truncated:
            self._dl_trained = False
            self._seq_trained = False
            report = {
                "dl": {"trained": False, "samples": 0, "note": "无可用数据"},
                "seq": {"trained": False, "samples": 0, "note": "无可用数据"},
            }
            self._log(f"fit(as_of={as_of_date.date()}): 无可用数据, 跳过训练")
            return report

        # --- 训练 DL (MLP) ---
        X, y = build_feature_label_dataset(
            truncated, horizon=self.horizon, min_window=self.min_window
        )
        if len(X) >= self.min_samples_dl:
            r_dl = self.dl_engine.fit(X, y)
            self._dl_trained = bool(r_dl.trained)
            dl_report = {
                "trained": r_dl.trained, "backend": r_dl.backend,
                "samples": r_dl.samples, "note": r_dl.note,
            }
        else:
            self._dl_trained = False
            dl_report = {
                "trained": False, "samples": int(len(X)),
                "note": f"DL 样本不足 {len(X)} < {self.min_samples_dl}",
            }

        # --- 训练 SEQ (LSTM + Transformer) ---
        X_seq, y_seq = build_sequence_dataset(
            truncated, lookback=self.lookback, horizon=self.horizon
        )
        if len(X_seq) >= self.min_samples_seq:
            r_seq = self.seq_engine.fit(X_seq, y_seq)
            self._seq_trained = bool(r_seq.trained)
            seq_report = {
                "trained": r_seq.trained, "backend": r_seq.backend,
                "samples": r_seq.samples, "note": r_seq.note,
            }
        else:
            self._seq_trained = False
            seq_report = {
                "trained": False, "samples": int(len(X_seq)),
                "note": f"SEQ 样本不足 {len(X_seq)} < {self.min_samples_seq}",
            }

        # 清空预测缓存 (新引擎, 旧特征向量失效)
        self._feat_cache.clear()
        self._seq_cache.clear()

        self._log(
            f"fit(as_of={as_of_date.date()}): DL={dl_report['trained']}({dl_report['samples']}s) "
            f"SEQ={seq_report['trained']}({seq_report['samples']}s) "
            f"elapsed={time.time()-t0:.1f}s"
        )
        return {"dl": dl_report, "seq": seq_report}

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    def predict_proba(
        self, stock_data_map: Dict[str, pd.DataFrame], dates: List[pd.Timestamp]
    ) -> pd.DataFrame:
        """对指定日期列表, 产出 (date, code, dl_prob, seq_prob, ai_prob) DataFrame.

        对每个 (date, code): 取 df[df['date'] <= date] (含当日 close),
        extract_latest_features + dl_engine.predict_proba -> dl_prob,
        extract_latest_sequence + seq_engine.predict_proba -> seq_prob,
        ai_prob = 0.5 * dl_prob + 0.5 * seq_prob.
        未训练或样本不足的引擎贡献 0.5 (中性).
        """
        if not dates:
            return pd.DataFrame(columns=["date", "code", "dl_prob", "seq_prob", "ai_prob"])

        rows: List[Dict[str, object]] = []
        for date in dates:
            for code, df in stock_data_map.items():
                if df is None or df.empty:
                    continue
                sub = df[df["date"] <= date]
                if len(sub) < self.lookback + 5:
                    continue

                # DL 概率
                dl_prob = 0.5
                if self._dl_trained:
                    feat = self._feat_cache.get((code, date), "missing")
                    if feat == "missing":
                        feat = extract_latest_features(sub, min_window=self.min_window)
                        self._feat_cache[(code, date)] = feat
                    if feat is not None:
                        dl_prob = float(self.dl_engine.predict_proba(feat))

                # SEQ 概率
                seq_prob = 0.5
                if self._seq_trained:
                    seq = self._seq_cache.get((code, date), "missing")
                    if seq == "missing":
                        seq = extract_latest_sequence(sub, lookback=self.lookback)
                        self._seq_cache[(code, date)] = seq
                    if seq is not None:
                        seq_prob = float(self.seq_engine.predict_proba(seq))

                ai_prob = 0.5 * dl_prob + 0.5 * seq_prob
                rows.append({
                    "date": date, "code": code,
                    "dl_prob": dl_prob, "seq_prob": seq_prob, "ai_prob": ai_prob,
                })

        if not rows:
            return pd.DataFrame(columns=["date", "code", "dl_prob", "seq_prob", "ai_prob"])
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [AISignalBridge] {msg}")


def inject_ai_prob(panel: pd.DataFrame, ai_prob_df: pd.DataFrame) -> pd.DataFrame:
    """把 ai_prob 注入 panel 的副本, 返回带 ai_prob 列的 panel.

    panel: MultiIndex (date, code) 的 DataFrame.
    ai_prob_df: 含 date/code/ai_prob 列.
    未覆盖的行 ai_prob = NaN (score_candidates 里 fillna(0.5) 兜底).
    """
    out = panel.copy()
    if ai_prob_df.empty:
        out["ai_prob"] = np.nan
        return out
    ai_lookup = ai_prob_df.set_index(["date", "code"])["ai_prob"]
    # reindex 按 MultiIndex 对齐, 缺失返回 NaN
    out["ai_prob"] = ai_lookup.reindex(out.index).values
    return out
