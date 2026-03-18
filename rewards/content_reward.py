"""
R_content(x, y): Content preservation reward.

Combines three complementary methods to assess whether the generated
sarcastic headline preserves the factual content of the original:
  1. BERTScore  – token-level soft matching (robust to style shifts)
  2. NLI Entailment – bidirectional entailment probability
  3. Entity Overlap – named-entity F1 between original and generated
"""

from typing import List, Tuple, Union

import torch
from bert_score import BERTScorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy

from .config import RewardConfig


class ContentReward:
    def __init__(self, config: RewardConfig):
        self.device = torch.device(config.device)
        self.w_bert = config.content_bertscore_weight
        self.w_nli = config.content_nli_weight
        self.w_entity = config.content_entity_weight

        # BERTScore
        self.bert_scorer = BERTScorer(
            model_type=config.bertscore_model_type,
            device=config.device,
            rescale_with_baseline=False,
        )

        # NLI cross-encoder
        self.nli_tokenizer = AutoTokenizer.from_pretrained(config.nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            config.nli_model_name
        ).to(self.device)
        self.nli_model.eval()
        # Label mapping for cross-encoder/nli-distilroberta-base:
        # 0 = contradiction, 1 = entailment, 2 = neutral
        self.entailment_idx = 1

        # spaCy NER
        self.nlp = spacy.load(config.spacy_model)

    # ------------------------------------------------------------------
    # Sub-scores
    # ------------------------------------------------------------------

    def _bertscore(
        self, originals: List[str], generateds: List[str]
    ) -> List[float]:
        """Token-level BERTScore F1."""
        _, _, f1 = self.bert_scorer.score(generateds, originals)
        return f1.tolist()

    @torch.no_grad()
    def _nli_entailment(
        self, originals: List[str], generateds: List[str]
    ) -> List[float]:
        """Bidirectional entailment probability (averaged)."""
        scores = []
        for orig, gen in zip(originals, generateds):
            # direction 1: premise=original, hypothesis=generated
            enc_fwd = self.nli_tokenizer(
                orig, gen, truncation=True, max_length=256, return_tensors="pt"
            ).to(self.device)
            logits_fwd = self.nli_model(**enc_fwd).logits
            p_fwd = torch.softmax(logits_fwd, dim=-1)[0, self.entailment_idx].item()

            # direction 2: premise=generated, hypothesis=original
            enc_bwd = self.nli_tokenizer(
                gen, orig, truncation=True, max_length=256, return_tensors="pt"
            ).to(self.device)
            logits_bwd = self.nli_model(**enc_bwd).logits
            p_bwd = torch.softmax(logits_bwd, dim=-1)[0, self.entailment_idx].item()

            scores.append((p_fwd + p_bwd) / 2.0)
        return scores

    def _entity_overlap(
        self, originals: List[str], generateds: List[str]
    ) -> List[float]:
        """Named-entity F1 between original and generated headlines."""
        scores = []
        for orig, gen in zip(originals, generateds):
            ents_orig = {ent.text.lower() for ent in self.nlp(orig).ents}
            ents_gen = {ent.text.lower() for ent in self.nlp(gen).ents}

            if len(ents_orig) == 0 and len(ents_gen) == 0:
                scores.append(1.0)  # neither has entities → content trivially preserved
                continue
            if len(ents_orig) == 0 or len(ents_gen) == 0:
                scores.append(0.0)
                continue

            overlap = len(ents_orig & ents_gen)
            f1 = 2 * overlap / (len(ents_orig) + len(ents_gen))
            scores.append(f1)
        return scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        originals: Union[str, List[str]],
        generateds: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """Compute weighted content-preservation score (0-1)."""
        single = isinstance(originals, str)
        if single:
            originals, generateds = [originals], [generateds]

        bert_scores = self._bertscore(originals, generateds)
        nli_scores = self._nli_entailment(originals, generateds)
        entity_scores = self._entity_overlap(originals, generateds)

        combined = [
            self.w_bert * b + self.w_nli * n + self.w_entity * e
            for b, n, e in zip(bert_scores, nli_scores, entity_scores)
        ]

        return combined[0] if single else combined
