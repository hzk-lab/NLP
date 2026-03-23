"""
Dataset for Disentangled Sarcasm Generator.

Loads headlines + is_sarcastic labels. No pairing needed — disentanglement
is achieved purely through L_recon, L_style, and L_adv losses.
"""

import json
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BartTokenizerFast


class SarcasmHeadlineDataset(Dataset):
    def __init__(
        self,
        headlines: List[str],
        labels: List[int],
        tokenizer: BartTokenizerFast,
        max_length: int = 64,
    ):
        self.headlines = headlines
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        headline = self.headlines[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            headline,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # For reconstruction, labels = input_ids with padding tokens set to -100
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "is_sarcastic": torch.tensor(label, dtype=torch.long),
        }


def load_data(path: str) -> Tuple[List[str], List[int]]:
    """Load headlines and labels from the JSON-lines dataset."""
    headlines, labels = [], []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line.strip())
            headlines.append(obj["headline"])
            labels.append(obj["is_sarcastic"])
    return headlines, labels


def create_splits(
    data_path: str,
    tokenizer: BartTokenizerFast,
    max_length: int = 64,
    seed: int = 42,
) -> Tuple[SarcasmHeadlineDataset, SarcasmHeadlineDataset, SarcasmHeadlineDataset]:
    """Load data and return train/val/test datasets (80/10/10 stratified split)."""
    headlines, labels = load_data(data_path)

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        headlines, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=seed, stratify=temp_labels
    )

    train_ds = SarcasmHeadlineDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds = SarcasmHeadlineDataset(val_texts, val_labels, tokenizer, max_length)
    test_ds = SarcasmHeadlineDataset(test_texts, test_labels, tokenizer, max_length)

    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds
