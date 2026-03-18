"""
Train a DistilBERT sarcasm classifier on the Sarcasm Headlines Dataset.

Usage:
    python train_classifier.py \
        --data_path Sarcasm_Headlines_Dataset.json \
        --output_dir checkpoints/sarcasm_classifier \
        --epochs 3 \
        --batch_size 32 \
        --lr 2e-5
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)


class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_data(path):
    headlines, labels = [], []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line.strip())
            headlines.append(obj["headline"])
            labels.append(obj["is_sarcastic"])
    return headlines, labels


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, f1, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Sarcasm_Headlines_Dataset.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sarcasm_classifier")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load & split ---
    headlines, labels = load_data(args.data_path)
    print(f"Loaded {len(headlines)} samples  |  sarcastic: {sum(labels)}  non-sarcastic: {len(labels) - sum(labels)}")

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        headlines, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=args.seed, stratify=temp_labels
    )
    print(f"Train: {len(train_texts)}  |  Val: {len(val_texts)}  |  Test: {len(test_texts)}")

    # --- Tokenizer & datasets ---
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = SarcasmDataset(val_texts, val_labels, tokenizer, args.max_length)
    test_dataset = SarcasmDataset(test_texts, test_labels, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # --- Model ---
    model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Train ---
    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} Step {step+1}/{len(train_loader)}  loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  -> Saved best model (val_f1={val_f1:.4f})")

    # --- Test ---
    print("\n--- Test Set Evaluation ---")
    model = DistilBertForSequenceClassification.from_pretrained(args.output_dir).to(device)
    test_acc, test_f1, test_labels_true, test_preds = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}  |  Test F1: {test_f1:.4f}")
    print(classification_report(test_labels_true, test_preds, target_names=["not_sarcastic", "sarcastic"]))


if __name__ == "__main__":
    main()
