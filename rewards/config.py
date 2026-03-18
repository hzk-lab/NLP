from dataclasses import dataclass, field
import torch


@dataclass
class RewardConfig:
    # --- Composite reward weights (λ1..λ4) ---
    lambda_style: float = 0.35
    lambda_content: float = 0.30
    lambda_fluency: float = 0.20
    lambda_degeneration: float = 0.15

    # --- Content reward internal sub-weights ---
    content_bertscore_weight: float = 0.4
    content_nli_weight: float = 0.4
    content_entity_weight: float = 0.2

    # --- Fluency reward internal sub-weights ---
    fluency_ppl_weight: float = 0.6
    fluency_length_weight: float = 0.4

    # --- Degeneration reward internal sub-weights ---
    degen_repetition_weight: float = 0.6
    degen_nonsense_weight: float = 0.4

    # --- Sarcasm classifier ---
    classifier_checkpoint: str = "checkpoints/sarcasm_classifier"
    classifier_model_name: str = "distilbert-base-uncased"
    classifier_max_length: int = 128

    # --- Content preservation models ---
    nli_model_name: str = "cross-encoder/nli-distilroberta-base"
    bertscore_model_type: str = "distilbert-base-uncased"
    spacy_model: str = "en_core_web_sm"

    # --- Fluency / PPL ---
    ppl_model_name: str = "gpt2"
    ppl_max_length: int = 128
    ppl_clip: float = 1000.0

    # --- Length control ---
    length_target_ratio: float = 1.0
    length_sigma: float = 0.3

    # --- Degeneration ---
    ngram_sizes: list = field(default_factory=lambda: [2, 3])
    special_char_threshold: float = 0.3
    uppercase_threshold: float = 0.5

    # --- Device ---
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
