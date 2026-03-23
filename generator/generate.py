"""
Inference script for DisentangledBART sarcasm generator.

1. Loads trained model
2. Computes the mean sarcastic style vector from all sarcastic headlines in the dataset
3. For a given non-sarcastic headline: extracts z_c, replaces z_s with the sarcastic
   style vector, and decodes to produce a sarcastic version.

Usage:
    python -m generator.generate \
        --checkpoint_dir checkpoints/disentangled_bart \
        --data_path Sarcasm_Headlines_Dataset.json \
        --input "obama visits arlington national cemetery to honor veterans"

    # Or batch mode from file (one headline per line):
    python -m generator.generate \
        --checkpoint_dir checkpoints/disentangled_bart \
        --data_path Sarcasm_Headlines_Dataset.json \
        --input_file headlines.txt
"""

import argparse
import json
import sys

import torch
from transformers import BartTokenizerFast

from .model import DisentangledBART, DisentangledBARTConfig


@torch.no_grad()
def compute_sarcastic_style_vector(model, tokenizer, data_path, device, max_length=64):
    """Compute the mean z_s across all sarcastic headlines in the dataset."""
    style_vectors = []
    with open(data_path, "r") as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj["is_sarcastic"] != 1:
                continue
            enc = tokenizer(
                obj["headline"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            _, z_s, _, _ = model.encode(input_ids, attention_mask)
            style_vectors.append(z_s.cpu())

    all_z_s = torch.cat(style_vectors, dim=0)  # (N_sarc, style_dim)
    mean_z_s = all_z_s.mean(dim=0, keepdim=True)  # (1, style_dim)
    print(f"Computed sarcastic style vector from {all_z_s.size(0)} headlines")
    return mean_z_s


def generate_sarcastic(model, tokenizer, headline, sarc_style, device, max_length=64, num_beams=4):
    """Generate a sarcastic version of a single headline."""
    enc = tokenizer(
        headline,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    z_c, _, enc_hidden, _ = model.encode(input_ids, attention_mask)

    target_z_s = sarc_style.to(device)
    if target_z_s.size(0) != z_c.size(0):
        target_z_s = target_z_s.expand(z_c.size(0), -1)

    output_ids = model.generate_from_embeddings(
        z_c, target_z_s, enc_hidden, attention_mask,
        max_length=max_length, num_beams=num_beams,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/disentangled_bart")
    parser.add_argument("--data_path", type=str, default="Sarcasm_Headlines_Dataset.json")
    parser.add_argument("--input", type=str, default=None, help="Single headline to transform")
    parser.add_argument("--input_file", type=str, default=None, help="File with one headline per line")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--weights", type=str, default="best_phase2.pt", help="Which weights to load")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    config_dict = torch.load(
        f"{args.checkpoint_dir}/config.pt", map_location="cpu", weights_only=False
    )
    model_config = DisentangledBARTConfig(**config_dict)
    model = DisentangledBART(model_config).to(device)
    state_dict = torch.load(
        f"{args.checkpoint_dir}/{args.weights}", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = BartTokenizerFast.from_pretrained(args.checkpoint_dir)
    print("Model loaded.\n")

    # --- Compute sarcastic style vector ---
    sarc_style = compute_sarcastic_style_vector(
        model, tokenizer, args.data_path, device, args.max_length
    )

    # --- Generate ---
    headlines = []
    if args.input:
        headlines = [args.input]
    elif args.input_file:
        with open(args.input_file, "r") as f:
            headlines = [line.strip() for line in f if line.strip()]
    else:
        print("Enter headlines (one per line, Ctrl-D to finish):")
        headlines = [line.strip() for line in sys.stdin if line.strip()]

    for headline in headlines:
        sarcastic = generate_sarcastic(
            model, tokenizer, headline, sarc_style, device,
            args.max_length, args.num_beams,
        )
        print(f"Original:  {headline}")
        print(f"Sarcastic: {sarcastic}")
        print()


if __name__ == "__main__":
    main()
