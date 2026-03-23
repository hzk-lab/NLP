"""
DisentangledBART — BART with content/style bottleneck for sarcasm style transfer.

Architecture:
  BART Encoder → mean-pool → content_proj → z_c  (dim 256)
                            → style_proj  → z_s  (dim 64)
  Fusion(z_c, z_s) → prefix tokens → prepend to encoder hidden states → BART Decoder

Auxiliary heads:
  - Style classifier on z_s  (predicts is_sarcastic)
  - Adversarial classifier on z_c via GRL (forces z_c to be style-invariant)
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartConfig


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    """Forward: identity. Backward: negate gradients scaled by alpha."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

@dataclass
class DisentangledBARTConfig:
    pretrained_model: str = "facebook/bart-base"
    content_dim: int = 256
    style_dim: int = 64
    num_prefix_tokens: int = 4
    grl_alpha: float = 1.0           # gradient reversal strength
    alpha_style: float = 1.0         # weight for L_style
    alpha_adv: float = 0.5           # weight for L_adv


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DisentangledBART(nn.Module):
    def __init__(self, config: DisentangledBARTConfig | None = None):
        super().__init__()
        if config is None:
            config = DisentangledBARTConfig()
        self.config = config

        self.bart = BartForConditionalGeneration.from_pretrained(config.pretrained_model)
        hidden_size = self.bart.config.d_model  # 768 for bart-base

        # Bottleneck projections
        self.content_proj = nn.Sequential(
            nn.Linear(hidden_size, config.content_dim),
            nn.ReLU(),
        )
        self.style_proj = nn.Sequential(
            nn.Linear(hidden_size, config.style_dim),
            nn.ReLU(),
        )

        # Fusion: map (z_c || z_s) back to hidden_size, then expand to prefix sequence
        self.num_prefix = config.num_prefix_tokens
        self.fusion = nn.Sequential(
            nn.Linear(config.content_dim + config.style_dim, hidden_size * self.num_prefix),
            nn.ReLU(),
        )

        # Auxiliary classifiers
        self.style_classifier = nn.Linear(config.style_dim, 2)
        self.adv_classifier = nn.Linear(config.content_dim, 2)

    # ------------------------------------------------------------------

    def _mean_pool(self, hidden_states, attention_mask):
        """Mean pool encoder hidden states, respecting the attention mask."""
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (hidden_states * mask).sum(dim=1)   # (B, H)
        lengths = mask.sum(dim=1).clamp(min=1)       # (B, 1)
        return summed / lengths

    def encode(self, input_ids, attention_mask):
        """Run BART encoder and return z_c, z_s, and full encoder hidden states."""
        encoder_outputs = self.bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = encoder_outputs.last_hidden_state  # (B, L, H)
        pooled = self._mean_pool(hidden_states, attention_mask)  # (B, H)

        z_c = self.content_proj(pooled)   # (B, content_dim)
        z_s = self.style_proj(pooled)     # (B, style_dim)

        return z_c, z_s, hidden_states, encoder_outputs

    def fuse_and_prefix(self, z_c, z_s):
        """Fuse z_c and z_s into prefix token embeddings."""
        fused = self.fusion(torch.cat([z_c, z_s], dim=-1))  # (B, H * num_prefix)
        B = z_c.size(0)
        H = self.bart.config.d_model
        prefix = fused.view(B, self.num_prefix, H)  # (B, num_prefix, H)
        return prefix

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        is_sarcastic: Optional[torch.Tensor] = None,
        phase: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        phase=1: only reconstruction loss (autoencoder warm-up)
        phase=2: reconstruction + style classification + adversarial
        """
        z_c, z_s, enc_hidden, encoder_outputs = self.encode(input_ids, attention_mask)

        # Build prefix and prepend to encoder hidden states
        prefix = self.fuse_and_prefix(z_c, z_s)  # (B, num_prefix, H)
        extended_hidden = torch.cat([prefix, enc_hidden], dim=1)  # (B, num_prefix + L, H)

        # Extend attention mask to cover prefix tokens
        B = input_ids.size(0)
        prefix_mask = torch.ones(B, self.num_prefix, device=input_ids.device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Decode
        from transformers.modeling_outputs import BaseModelOutput
        decoder_input_ids = self.bart.prepare_decoder_input_ids_from_labels(labels) if labels is not None else None

        outputs = self.bart(
            encoder_outputs=BaseModelOutput(last_hidden_state=extended_hidden),
            attention_mask=extended_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        result = {"logits": outputs.logits}
        total_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=input_ids.device)
        result["loss_recon"] = total_loss.clone()

        if phase == 2 and is_sarcastic is not None:
            # L_style: style classifier on z_s
            style_logits = self.style_classifier(z_s)
            loss_style = F.cross_entropy(style_logits, is_sarcastic)
            result["loss_style"] = loss_style

            # L_adv: adversarial classifier on z_c via gradient reversal
            z_c_rev = grad_reverse(z_c, self.config.grl_alpha)
            adv_logits = self.adv_classifier(z_c_rev)
            loss_adv = F.cross_entropy(adv_logits, is_sarcastic)
            result["loss_adv"] = loss_adv

            total_loss = total_loss + self.config.alpha_style * loss_style + self.config.alpha_adv * loss_adv

        result["loss"] = total_loss
        return result

    @torch.no_grad()
    def generate_from_embeddings(
        self,
        z_c: torch.Tensor,
        z_s: torch.Tensor,
        encoder_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 64,
        num_beams: int = 4,
    ) -> torch.Tensor:
        """Generate token IDs given pre-computed z_c, z_s, and encoder hidden states."""
        prefix = self.fuse_and_prefix(z_c, z_s)
        extended_hidden = torch.cat([prefix, encoder_hidden], dim=1)

        B = z_c.size(0)
        prefix_mask = torch.ones(B, self.num_prefix, device=z_c.device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        from transformers.modeling_outputs import BaseModelOutput
        generated = self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=extended_hidden),
            attention_mask=extended_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
        return generated
