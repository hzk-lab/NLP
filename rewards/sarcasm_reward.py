"""
R_style(y): Sarcasm / style reward.

Loads a fine-tuned DistilBERT sarcasm classifier and returns
P(sarcastic | headline) as the reward signal.
"""

from typing import List, Union

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from .config import RewardConfig


class SarcasmReward:
    def __init__(self, config: RewardConfig):
        self.device = torch.device(config.device)
        self.max_length = config.classifier_max_length
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(config.classifier_checkpoint)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            config.classifier_checkpoint
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, generated: Union[str, List[str]]) -> Union[float, List[float]]:
        """Return P(sarcastic) for one or more generated headlines."""
        single = isinstance(generated, str)
        if single:
            generated = [generated]

        encodings = self.tokenizer(
            generated,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**encodings).logits
        probs = F.softmax(logits, dim=-1)
        # label 1 = sarcastic
        sarcasm_probs = probs[:, 1].cpu().tolist()

        return sarcasm_probs[0] if single else sarcasm_probs
