"""
R_degeneration(y): Degeneration penalty.

Higher score = more degenerate = larger penalty in the composite formula.

Combines two sub-scores:
  1. Repetition rate – fraction of repeated n-grams (n=2,3)
  2. Nonsense score – special-char ratio, uppercase ratio, consecutive duplicate tokens
"""

import re
from collections import Counter
from typing import List, Union

from .config import RewardConfig


class DegenerationReward:
    def __init__(self, config: RewardConfig):
        self.w_rep = config.degen_repetition_weight
        self.w_nonsense = config.degen_nonsense_weight
        self.ngram_sizes = config.ngram_sizes
        self.special_char_threshold = config.special_char_threshold
        self.uppercase_threshold = config.uppercase_threshold

    # ------------------------------------------------------------------
    # Sub-scores
    # ------------------------------------------------------------------

    @staticmethod
    def _ngram_repetition(tokens: List[str], n: int) -> float:
        """Fraction of n-grams that appear more than once."""
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        return repeated / len(ngrams)

    def _repetition_score(self, text: str) -> float:
        """Average repetition rate across configured n-gram sizes."""
        tokens = text.lower().split()
        if len(tokens) == 0:
            return 1.0  # empty output is maximally degenerate
        rates = [self._ngram_repetition(tokens, n) for n in self.ngram_sizes]
        return sum(rates) / len(rates)

    def _nonsense_score(self, text: str) -> float:
        """Heuristic nonsense detection based on surface patterns."""
        if len(text) == 0:
            return 1.0

        chars = list(text)
        total_chars = len(chars)

        # Special character ratio (non-alphanumeric, non-space)
        special = sum(1 for c in chars if not c.isalnum() and not c.isspace())
        special_ratio = special / total_chars
        special_flag = min(special_ratio / self.special_char_threshold, 1.0)

        # Uppercase ratio (among alpha chars)
        alpha_chars = [c for c in chars if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            upper_flag = min(upper_ratio / self.uppercase_threshold, 1.0)
        else:
            upper_flag = 1.0  # no alpha chars → suspicious

        # Consecutive duplicate tokens (e.g. "the the the")
        tokens = text.lower().split()
        if len(tokens) >= 2:
            consec_dups = sum(
                1 for a, b in zip(tokens, tokens[1:]) if a == b
            )
            consec_ratio = consec_dups / (len(tokens) - 1)
        else:
            consec_ratio = 0.0

        return (special_flag + upper_flag + consec_ratio) / 3.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, generated: Union[str, List[str]]) -> Union[float, List[float]]:
        """
        Compute degeneration score (0-1).
        Higher = more degenerate → will be *subtracted* in the composite reward.
        """
        single = isinstance(generated, str)
        if single:
            generated = [generated]

        results = []
        for text in generated:
            rep = self._repetition_score(text)
            nonsense = self._nonsense_score(text)
            results.append(self.w_rep * rep + self.w_nonsense * nonsense)

        return results[0] if single else results
