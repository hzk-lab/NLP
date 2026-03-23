"""
Two-phase training script for DisentangledBART.

Phase 1 (warm-up):  L_recon only — learn to reconstruct headlines through the bottleneck.
Phase 2 (disentangle): L_recon + α * L_style + β * L_adv — separate content from style.

Usage:
    python -m generator.train \
        --data_path Sarcasm_Headlines_Dataset.json \
        --output_dir checkpoints/disentangled_bart \
        --phase1_epochs 3 \
        --phase2_epochs 6 \
        --batch_size 32 \
        --lr 3e-5
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizerFast, get_linear_schedule_with_warmup

from .dataset import create_splits
from .model import DisentangledBART, DisentangledBARTConfig


def run_epoch(model, dataloader, optimizer, scheduler, device, phase, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_style = 0.0
    total_adv = 0.0
    steps = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        is_sarcastic = batch["is_sarcastic"].to(device)

        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            is_sarcastic=is_sarcastic,
            phase=phase,
        )

        loss = result["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_recon += result["loss_recon"].item()
        if "loss_style" in result:
            total_style += result["loss_style"].item()
        if "loss_adv" in result:
            total_adv += result["loss_adv"].item()
        steps += 1

    n = max(steps, 1)
    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "style": total_style / n,
        "adv": total_adv / n,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, phase):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    steps = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        is_sarcastic = batch["is_sarcastic"].to(device)

        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            is_sarcastic=is_sarcastic,
            phase=phase,
        )
        total_loss += result["loss"].item()
        total_recon += result["loss_recon"].item()
        steps += 1

    n = max(steps, 1)
    return {"loss": total_loss / n, "recon": total_recon / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Sarcasm_Headlines_Dataset.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/disentangled_bart")
    parser.add_argument("--pretrained_model", type=str, default="facebook/bart-base")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--phase1_epochs", type=int, default=3)
    parser.add_argument("--phase2_epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--alpha_style", type=float, default=1.0)
    parser.add_argument("--alpha_adv", type=float, default=0.5)
    parser.add_argument("--grl_alpha", type=float, default=1.0)
    parser.add_argument("--content_dim", type=int, default=256)
    parser.add_argument("--style_dim", type=int, default=64)
    parser.add_argument("--num_prefix", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    tokenizer = BartTokenizerFast.from_pretrained(args.pretrained_model)
    train_ds, val_ds, _ = create_splits(args.data_path, tokenizer, args.max_length, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    # --- Model ---
    model_config = DisentangledBARTConfig(
        pretrained_model=args.pretrained_model,
        content_dim=args.content_dim,
        style_dim=args.style_dim,
        num_prefix_tokens=args.num_prefix,
        grl_alpha=args.grl_alpha,
        alpha_style=args.alpha_style,
        alpha_adv=args.alpha_adv,
    )
    model = DisentangledBART(model_config).to(device)

    # ======================================================================
    # Phase 1: Auto-encoder warm-up
    # ======================================================================
    print("\n" + "=" * 60)
    print("Phase 1: Auto-encoder warm-up")
    print("=" * 60)

    total_steps_p1 = len(train_loader) * args.phase1_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps_p1 * args.warmup_ratio), total_steps_p1
    )

    best_val_loss = float("inf")
    for epoch in range(args.phase1_epochs):
        metrics = run_epoch(model, train_loader, optimizer, scheduler, device, phase=1)
        val_metrics = evaluate(model, val_loader, device, phase=1)
        print(
            f"[P1] Epoch {epoch+1}/{args.phase1_epochs}  "
            f"train_loss={metrics['loss']:.4f}  train_recon={metrics['recon']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_phase1.pt"))
            print(f"  -> Saved best phase-1 model (val_loss={best_val_loss:.4f})")

    # ======================================================================
    # Phase 2: Disentanglement training
    # ======================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Disentanglement training")
    print("=" * 60)

    total_steps_p2 = len(train_loader) * args.phase2_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps_p2 * args.warmup_ratio), total_steps_p2
    )

    best_val_loss = float("inf")
    for epoch in range(args.phase2_epochs):
        metrics = run_epoch(model, train_loader, optimizer, scheduler, device, phase=2)
        val_metrics = evaluate(model, val_loader, device, phase=2)
        print(
            f"[P2] Epoch {epoch+1}/{args.phase2_epochs}  "
            f"train_loss={metrics['loss']:.4f}  recon={metrics['recon']:.4f}  "
            f"style={metrics['style']:.4f}  adv={metrics['adv']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_phase2.pt"))
            print(f"  -> Saved best phase-2 model (val_loss={best_val_loss:.4f})")

    # Save final model + config
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final.pt"))
    torch.save(vars(model_config), os.path.join(args.output_dir, "config.pt"))
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nTraining complete. Models saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
