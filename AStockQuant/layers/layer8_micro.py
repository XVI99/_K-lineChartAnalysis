"""
Layer8 - 贝叶斯信念层
=====================

功能: 序贯贝叶斯更新，信念动态调整
"""

import pandas as pd
import numpy as np
from typing import Dict
from collections import deque


class BeliefLayer:
    """贝叶斯信念层 - 序贯更新概率信念"""
    
    BASE_PRIOR = 0.50
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.belief_history: Dict[str, deque] = {}
    
    def _bayes_update(self, prior: float, likelihood: float) -> float:
        if prior <= 0 or prior >= 1:
            return 0.5
        if likelihood <= 0 or likelihood >= 1:
            return prior
        numerator = likelihood * prior
        denominator = likelihood * prior + (1 - likelihood) * (1 - prior)
        if denominator > 0:
            posterior = numerator / denominator
        else:
            posterior = prior
        return max(0.01, min(0.99, posterior))
    
    def extract_features(self, symbol: str, df: pd.DataFrame, ctx: Dict) -> Dict:
        features = {}
        
        if symbol not in self.belief_history:
            self.belief_history[symbol] = deque(maxlen=self.max_history)
        
        if df.empty or len(df) < 20:
            return features
        
        close = df['close']
        recent_return = close.pct_change(20).iloc[-1]
        volume_surge = False
        if 'volume' in df.columns:
            vol = df['volume']
            vol_now = vol.iloc[-1]
            vol_ma = vol.rolling(20).mean().iloc[-1]
            volume_surge = vol_now > vol_ma * 1.5
        
        if recent_return > 0.05:
            likelihood = 0.65
        elif recent_return > 0.02:
            likelihood = 0.55
        elif recent_return < -0.05:
            likelihood = 0.35
        elif recent_return < -0.02:
            likelihood = 0.45
        else:
            likelihood = 0.50
        
        if volume_surge:
            if recent_return > 0:
                likelihood = min(0.85, likelihood + 0.10)
            else:
                likelihood = max(0.15, likelihood - 0.10)
        
        history = self.belief_history[symbol]
        if len(history) > 0:
            current_prior = history[-1][1]
        else:
            current_prior = self.BASE_PRIOR
        
        posterior = self._bayes_update(current_prior, likelihood)
        
        if len(df) >= 2:
            current_date = df['date'].iloc[-1] if 'date' in df.columns else None
            if current_date is not None:
                history.append((current_date, posterior))
        
        features['belief_posterior'] = posterior
        features['belief_base_prior'] = self.BASE_PRIOR
        features['belief_evidence_count'] = len(history)
        
        if posterior >= 0.75:
            level = 'strongly_bullish'
        elif posterior >= 0.60:
            level = 'bullish'
        elif posterior <= 0.25:
            level = 'strongly_bearish'
        elif posterior <= 0.40:
            level = 'bearish'
        else:
            level = 'neutral'
        
        features['belief_level'] = level
        
        if posterior > 0 and posterior < 1:
            kl = posterior * np.log(posterior / self.BASE_PRIOR) + \
                 (1 - posterior) * np.log((1 - posterior) / (1 - self.BASE_PRIOR))
            features['belief_kl'] = max(0, kl)
        else:
            features['belief_kl'] = 0.0
        
        evidence_weight = min(1.0, len(history) / 20)
        features['belief_confidence'] = evidence_weight
        features['belief_signal'] = level.upper()
        
        return features
    
    def get_market_summary(self) -> Dict:
        if not self.belief_history:
            return {}
        
        all_posteriors = []
        for symbol, history in self.belief_history.items():
            if len(history) > 0:
                all_posteriors.append(history[-1][1])
        
        if not all_posteriors:
            return {}
        
        return {
            'count': len(all_posteriors),
            'mean_posterior': np.mean(all_posteriors),
            'median_posterior': np.median(all_posteriors),
            'std_posterior': np.std(all_posteriors),
            'bullish_count': sum(1 for p in all_posteriors if p > 0.6),
            'bearish_count': sum(1 for p in all_posteriors if p < 0.4),
        }