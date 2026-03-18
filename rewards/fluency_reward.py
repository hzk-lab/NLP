"""
R_fluency(y): Fluency reward.

Combines two sub-scores:
  1. Perplexity  – GPT-2 perplexity mapped to a 0-1 score (lower PPL = higher reward)
  2. Length control – Gaussian penalty on output/input token-length ratio
"""

import math
from typing import List, Union

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from .config import RewardConfig


class FluencyReward:
    def __init__(self, config: RewardConfig):
        self.device = torch.device(config.device)
        self.w_ppl = config.fluency_ppl_weight
        self.w_len = config.fluency_length_weight
        self.ppl_clip = config.ppl_clip
        self.max_length = config.ppl_max_length
        self.target_ratio = config.length_target_ratio
        self.sigma = config.length_sigma

        self.tokenizer = GPT2TokenizerFast.from_pretrained(config.ppl_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(config.ppl_model_name).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Sub-scores
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _perplexity_scores(self, texts: List[str]) -> List[float]:
        """Compute per-sample perplexity and map to 0-1 reward."""
        scores = []
        for text in texts:
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            input_ids = enc["input_ids"]
            if input_ids.size(1) < 2:
                scores.append(0.0)
                continue

            outputs = self.model(input_ids=input_ids, labels=input_ids)
            nll = outputs.loss.item()  # mean token-level NLL
            ppl = min(math.exp(nll), self.ppl_clip)

            # Map PPL → reward:  1 / (1 + log(ppl))
            reward = 1.0 / (1.0 + math.log(max(ppl, 1.0)))
            scores.append(reward)
        return scores

    def _length_scores(
        self, originals: List[str], generateds: List[str]
    ) -> List[float]:
        """Gaussian penalty on output/input token-length ratio."""
        scores = []
        for orig, gen in zip(originals, generateds):
            orig_len = max(len(orig.split()), 1)
            gen_len = max(len(gen.split()), 1)
            ratio = gen_len / orig_len
            score = math.exp(
                -((ratio - self.target_ratio) ** 2) / (2 * self.sigma ** 2)
            )
            scores.append(score)
        return scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        originals: Union[str, List[str]],
        generateds: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """Compute weighted fluency score (0-1)."""
        single = isinstance(originals, str)
        if single:
            originals, generateds = [originals], [generateds]

        ppl_scores = self._perplexity_scores(generateds)
        len_scores = self._length_scores(originals, generateds)

        combined = [
            self.w_ppl * p + self.w_len * l
            for p, l in zip(ppl_scores, len_scores)
        ]

        return combined[0] if single else combined
