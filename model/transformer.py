"""MedBharat-1: Full model assembly.

Architecture:
  Input → DualScript Embedding → Shared Layers (8) → MoE Router → Expert Layers → Output Head

Improvements over LLaMA 3.1:
  1. Medical Sparse Attention (4 heads with learned medical entity bias)
  2. Mixture of Medical Experts (6 domain experts, top-2 routing)
  3. Language embedding (English/Hindi/Mixed)
  4. Designed for curriculum training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.config import MedBharatConfig
from model.experts import TransformerBlock, RMSNorm, MixtureOfMedicalExperts


class MedBharatModel(nn.Module):
    """MedBharat-1: Curriculum-trained Mixture of Medical Experts."""

    def __init__(self, config: MedBharatConfig):
        super().__init__()
        self.config = config

        # --- Embedding Layer ---
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.language_embedding = nn.Embedding(config.n_languages, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        # --- Shared Transformer Layers (all tokens pass through) ---
        self.shared_layers = nn.ModuleList([
            TransformerBlock(config, intermediate_size=config.intermediate_size)
            for _ in range(config.n_shared_layers)
        ])
        self.shared_norm = RMSNorm(config.d_model, config.rms_norm_eps)

        # --- Mixture of Medical Experts ---
        self.moe = MixtureOfMedicalExperts(config)
        self.moe_norm = RMSNorm(config.d_model, config.rms_norm_eps)

        # --- Output Head ---
        self.output_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding weights with output head
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Gradient checkpointing flag
        self._gradient_checkpointing = config.gradient_checkpointing

    def _init_weights(self, module):
        """Initialize weights — standard normal with small std."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self._gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        language_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """Forward pass.

        Args:
            input_ids: [batch, seq] token indices
            attention_mask: [batch, seq] 1 for real tokens, 0 for padding
            language_ids: [batch, seq] language indicator (0=en, 1=hi, 2=mix)
            labels: [batch, seq] target token ids for loss computation

        Returns:
            dict with 'logits', 'loss', 'aux_loss', 'total_loss'
        """
        batch_size, seq_len = input_ids.shape

        # --- Embedding ---
        hidden = self.token_embedding(input_ids)

        if language_ids is not None:
            hidden = hidden + self.language_embedding(language_ids)

        hidden = self.embed_dropout(hidden)

        # --- Shared Layers ---
        for layer in self.shared_layers:
            if self._gradient_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, attention_mask, use_reentrant=False)
            else:
                hidden = layer(hidden, attention_mask)

        hidden = self.shared_norm(hidden)

        # --- Mixture of Experts ---
        # Note: checkpoint works with tuple returns, but we use a wrapper
        # to ensure clean gradient flow through the MoE routing
        if self._gradient_checkpointing and self.training:
            def _moe_forward(h, mask):
                return self.moe(h, mask)
            moe_output, aux_loss = checkpoint(
                _moe_forward, hidden, attention_mask, use_reentrant=False
            )
        else:
            moe_output, aux_loss = self.moe(hidden, attention_mask)

        # Residual connection from shared layers
        hidden = hidden + self.moe_norm(moe_output)

        # --- Output ---
        hidden = self.output_norm(hidden)
        logits = self.lm_head(hidden)

        # --- Loss ---
        loss = None
        total_loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            total_loss = loss + aux_loss

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss,
            "total_loss": total_loss,
        }

    def count_parameters(self) -> dict:
        """Count total and per-component parameters."""
        counts = {}
        counts["token_embedding"] = sum(p.numel() for p in self.token_embedding.parameters())
        counts["language_embedding"] = sum(p.numel() for p in self.language_embedding.parameters())
        counts["shared_layers"] = sum(p.numel() for p in self.shared_layers.parameters())
        counts["moe_router"] = sum(p.numel() for p in self.moe.router.parameters())
        for i, expert in enumerate(self.moe.experts):
            counts[f"expert_{self.config.expert_names[i]}"] = sum(
                p.numel() for p in expert.parameters()
            )
        counts["output_head"] = sum(p.numel() for p in self.output_norm.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())

        # Active params = everything except non-selected experts
        single_expert = counts[f"expert_{self.config.expert_names[0]}"]
        counts["active_per_forward"] = (
            counts["total"]
            - self.config.n_experts * single_expert
            + self.config.top_k_experts * single_expert
        )
        return counts

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Simple autoregressive generation."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids[:, -self.config.max_seq_len:]

            # Forward
            outputs = self(idx_cond)
            logits = outputs["logits"][:, -1, :]  # last token logits

            # Temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -1e4

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_mask] = -1e4
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == self.config.eos_token_id:
                break

        return input_ids
