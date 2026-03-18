"""
Composite reward aggregator.

R(x, y) = λ1 * R_style(y) + λ2 * R_content(x, y) + λ3 * R_fluency(y) - λ4 * R_degeneration(y)
"""

from typing import Dict, List, Union

from .config import RewardConfig
from .sarcasm_reward import SarcasmReward
from .content_reward import ContentReward
from .fluency_reward import FluencyReward
from .degeneration_reward import DegenerationReward


class CompositeReward:
    def __init__(self, config: RewardConfig | None = None):
        if config is None:
            config = RewardConfig()
        self.config = config

        self.lambda_style = config.lambda_style
        self.lambda_content = config.lambda_content
        self.lambda_fluency = config.lambda_fluency
        self.lambda_degen = config.lambda_degeneration

        self.sarcasm_reward = SarcasmReward(config)
        self.content_reward = ContentReward(config)
        self.fluency_reward = FluencyReward(config)
        self.degen_reward = DegenerationReward(config)

    def compute(self, original: str, generated: str) -> Dict[str, float]:
        """Compute composite reward for a single (original, generated) pair."""
        s = self.sarcasm_reward.score(generated)
        c = self.content_reward.score(original, generated)
        f = self.fluency_reward.score(original, generated)
        d = self.degen_reward.score(generated)

        total = (
            self.lambda_style * s
            + self.lambda_content * c
            + self.lambda_fluency * f
            - self.lambda_degen * d
        )

        return {
            "total": total,
            "sarcasm": s,
            "content": c,
            "fluency": f,
            "degeneration": d,
        }

    def compute_batch(
        self, originals: List[str], generateds: List[str]
    ) -> List[Dict[str, float]]:
        """Compute composite reward for a batch of pairs."""
        assert len(originals) == len(generateds), "Batch sizes must match"

        s_scores = self.sarcasm_reward.score(generateds)
        c_scores = self.content_reward.score(originals, generateds)
        f_scores = self.fluency_reward.score(originals, generateds)
        d_scores = self.degen_reward.score(generateds)

        results = []
        for s, c, f, d in zip(s_scores, c_scores, f_scores, d_scores):
            total = (
                self.lambda_style * s
                + self.lambda_content * c
                + self.lambda_fluency * f
                - self.lambda_degen * d
            )
            results.append({
                "total": total,
                "sarcasm": s,
                "content": c,
                "fluency": f,
                "degeneration": d,
            })
        return results
