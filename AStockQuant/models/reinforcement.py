# -*- coding: utf-8 -*-
"""
reinforcement.py — 强化学习仓位分配器

从原 advanced_ai.py 迁移, 包含:
- ReinforcementAllocator     (UCB 多臂老虎机)
- RiskAwareReinforcementAllocator (风险约束 + 换手惩罚)
- PPOAllocationEngine        (Stable Baselines3 PPO)
- build_ppo_inputs           (辅助函数)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class ReinforcementAllocator:
    """UCB 多臂老虎机仓位分配"""

    def __init__(self, alpha: float = 0.25, beta: float = 0.15):
        self.alpha = alpha
        self.beta = beta
        self.q_values: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.total_steps = 0

    def update(self, symbol: str, reward: float):
        self.total_steps += 1
        n = self.counts.get(symbol, 0) + 1
        q = self.q_values.get(symbol, 0.0)
        self.q_values[symbol] = q + (reward - q) / n
        self.counts[symbol] = n

    def warmup_with_daily_change(self, symbol: str, change_pct: float):
        self.update(symbol, float(np.tanh(change_pct / 5.0)))

    def allocate(self, symbol_scores: Dict[str, float]) -> Dict[str, float]:
        if not symbol_scores:
            return {}
        prefs = {}
        total = max(1, self.total_steps)
        for s, score in symbol_scores.items():
            q = self.q_values.get(s, 0.0)
            n = self.counts.get(s, 0)
            ucb = np.sqrt(2.0 * np.log(total + 1) / (n + 1))
            prefs[s] = float(score + self.alpha * q + self.beta * ucb)
        vals = np.array(list(prefs.values()), dtype=np.float64)
        vals = vals - np.max(vals)
        exps = np.exp(vals)
        w = exps / exps.sum() if exps.sum() > 0 else np.ones_like(exps) / len(exps)
        return {s: float(w[i]) for i, s in enumerate(prefs)}


class RiskAwareReinforcementAllocator(ReinforcementAllocator):
    """风险约束 + 换手惩罚 强化分配器"""

    def __init__(self, alpha=0.35, beta=0.20, gamma_risk=0.35, max_weight=0.25, turnover_penalty=0.30):
        super().__init__(alpha, beta)
        self.gamma_risk = gamma_risk
        self.max_weight = max_weight
        self.turnover_penalty = turnover_penalty
        self._prev: Dict[str, float] = {}

    def allocate_with_risk(self, symbol_scores, risk_map=None):
        if not symbol_scores:
            return {}
        risk_map = risk_map or {}
        prefs = {}
        total = max(1, self.total_steps)
        for s, score in symbol_scores.items():
            q = self.q_values.get(s, 0.0)
            n = self.counts.get(s, 0)
            ucb = np.sqrt(2.0 * np.log(total + 1) / (n + 1))
            rp = self.gamma_risk * float(risk_map.get(s, 0.0))
            tc = self.turnover_penalty * abs(self._prev.get(s, 0.0))
            prefs[s] = float(score + self.alpha * q + self.beta * ucb - rp - tc)
        vals = np.array(list(prefs.values()), dtype=np.float64)
        vals = vals - np.max(vals)
        exps = np.exp(vals)
        raw = exps / exps.sum() if exps.sum() > 0 else np.ones_like(exps) / len(exps)
        clipped = np.minimum(raw, self.max_weight)
        clipped = clipped / clipped.sum() if clipped.sum() > 0 else np.ones_like(clipped) / len(clipped)
        out = {}
        for i, s in enumerate(prefs):
            old = self._prev.get(s, 0.0)
            out[s] = float(max(0.0, (1 - self.turnover_penalty) * clipped[i] + self.turnover_penalty * old))
        tw = sum(out.values())
        if tw > 0:
            out = {k: v / tw for k, v in out.items()}
        self._prev = out.copy()
        return out


class PPOAllocationEngine:
    """PPO 策略级仓位分配器"""

    def __init__(self, max_weight=0.25, commission=0.001, risk_aversion=0.25):
        self.max_weight = max_weight
        self.commission = commission
        self.risk_aversion = risk_aversion
        self._model = None
        self._trained = False
        self.backend = "fallback"
        self._fallback = RiskAwareReinforcementAllocator(max_weight=max_weight)
        self._gym = None
        self._sb3 = None
        self._init_deps()

    def _init_deps(self):
        try:
            import importlib
            gym = importlib.import_module("gymnasium")
            spaces = importlib.import_module("gymnasium.spaces")
            sb3 = importlib.import_module("stable_baselines3")
            self._gym = (gym, spaces)
            self._sb3 = getattr(sb3, "PPO")
            self.backend = "ppo"
        except Exception:
            pass

    def train(self, state_matrix, returns_matrix, timesteps=3000) -> bool:
        if state_matrix is None or returns_matrix is None:
            return False
        if len(state_matrix) < 40 or returns_matrix.shape[1] < 2 or self.backend != "ppo":
            return False
        gym, spaces = self._gym
        PPO = self._sb3
        mw = self.max_weight

        class PortfolioEnv(gym.Env):
            metadata = {"render_modes": []}
            def __init__(self, sm, rm, comm, ra):
                super().__init__()
                self.sm, self.rm = sm, rm
                self.comm, self.ra = comm, ra
                self.t = self.nav = 0
                self.max_nav = 1.0
                self.pw = np.ones(rm.shape[1], dtype=np.float32) / rm.shape[1]
                self.observation_space = spaces.Box(-10, 10, (sm.shape[1],), np.float32)
                self.action_space = spaces.Box(-1.0, 1.0, (rm.shape[1],), np.float32)
            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.t, self.nav, self.max_nav = 0, 1.0, 1.0
                self.pw = np.ones(self.rm.shape[1], dtype=np.float32) / self.rm.shape[1]
                return self.sm[0].astype(np.float32), {}
            def step(self, action):
                x = np.clip(np.array(action, dtype=np.float64), -5, 5)
                e = np.exp(x - x.max())
                w = np.minimum(e / (e.sum() + 1e-12), mw)
                w = w / (w.sum() + 1e-12)
                r = float(w @ self.rm[self.t])
                tc = self.comm * float(np.abs(w - self.pw).sum())
                self.nav *= (1 + r - tc)
                self.max_nav = max(self.max_nav, self.nav)
                dd = 1 - self.nav / (self.max_nav + 1e-12)
                self.pw = w.astype(np.float32)
                self.t += 1
                done = self.t >= len(self.sm) - 1
                obs = self.sm[min(self.t, len(self.sm)-1)].astype(np.float32)
                return obs, float((r - tc) - self.ra * dd), done, False, {}

        env = PortfolioEnv(state_matrix, returns_matrix, self.commission, self.risk_aversion)
        self._model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4,
                          n_steps=min(256, max(64, len(state_matrix) // 2)),
                          batch_size=64, gamma=0.995, gae_lambda=0.95, ent_coef=0.005)
        self._model.learn(total_timesteps=max(500, timesteps))
        self._trained = True
        return True

    def infer_weights(self, symbols, latest_state, score_map=None):
        if not symbols:
            return {}
        if self.backend == "ppo" and self._trained and self._model and latest_state is not None and len(latest_state) > 0:
            action, _ = self._model.predict(latest_state.astype(np.float32), deterministic=True)
            x = np.clip(np.array(action, dtype=np.float64), -5, 5)
            e = np.exp(x - x.max())
            w = np.minimum(e / (e.sum() + 1e-12), self.max_weight)
            w = w / (w.sum() + 1e-12)
            return {s: float(w[i]) for i, s in enumerate(symbols)}
        sm = score_map or {s: 1.0 for s in symbols}
        return self._fallback.allocate_with_risk(sm, {s: 0.1 for s in symbols})


def build_ppo_inputs(results_df, stock_data_map, top_k=10, lookback=90):
    """从扫描结果构建 PPO 输入"""
    empty = {"symbols": [], "state_matrix": np.empty((0, 1), np.float32),
             "returns_matrix": np.empty((0, 1), np.float32), "latest_state": np.empty((1,), np.float32)}
    if results_df is None or results_df.empty:
        return empty
    sc = "ai_rank_score" if "ai_rank_score" in results_df.columns else "confidence"
    syms = [s for s in results_df.sort_values(sc, ascending=False)["symbol"] if s in stock_data_map][:top_k]
    if not syms:
        return empty
    cm = {s: stock_data_map[s]["close"].astype(float).tail(lookback + 30) for s in syms}
    cdf = pd.DataFrame(cm).dropna(how="all").ffill().dropna()
    if cdf.empty or len(cdf) < 40:
        return {**empty, "symbols": syms}
    rets = cdf.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    mom5 = cdf.pct_change(5).replace([np.inf, -np.inf], 0).fillna(0)
    vol10 = rets.rolling(10).std().replace([np.inf, -np.inf], 0).fillna(0)
    mg = (cdf / cdf.rolling(20).mean() - 1).replace([np.inf, -np.inf], 0).fillna(0)
    af = np.concatenate([np.stack([rets[s].values, mom5[s].values, vol10[s].values, mg[s].values], 1) for s in syms], 1)
    mm = rets.mean(1).values
    mv = pd.Series(mm).rolling(10).std().fillna(0).values
    br = (rets > 0).mean(1).values
    mmom = cdf.mean(1).pct_change(10).replace([np.inf, -np.inf], 0).fillna(0).values
    mf = np.stack([mm, mv, br, mmom], 1)
    sm = np.concatenate([af, mf], 1).astype(np.float32)[20:]
    rm = rets[syms].values.astype(np.float32)[20:]
    if len(sm) == 0:
        return {**empty, "symbols": syms}
    return {"symbols": syms, "state_matrix": sm, "returns_matrix": rm, "latest_state": sm[-1]}
