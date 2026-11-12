"""Mixture of Medical Experts (MoME) for MedBharat-1.

6 expert modules, each a small transformer stack.
A learned router selects top-2 experts per token.
Load-balancing auxiliary loss ensures all experts get used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import MedBharatConfig
from model.attention import MedicalSparseAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (like LLaMA)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (like LLaMA)."""

    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer block: RMSNorm -> Attention -> RMSNorm -> FFN."""

    def __init__(self, config: MedBharatConfig, intermediate_size: int = None):
        super().__init__()
        inter = intermediate_size or config.intermediate_size
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = MedicalSparseAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config.d_model, inter)

    def forward(self, x, attention_mask=None):
        # Pre-norm architecture
        x = x + self.attention(self.attention_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ExpertModule(nn.Module):
    """A single expert: a stack of transformer layers specialized for one domain."""

    def __init__(self, config: MedBharatConfig, expert_name: str = ""):
        super().__init__()
        self.expert_name = expert_name
        self._gradient_checkpointing = config.gradient_checkpointing
        self.layers = nn.ModuleList([
            TransformerBlock(config, intermediate_size=config.expert_intermediate_size)
            for _ in range(config.n_expert_layers)
        ])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            if torch.is_grad_enabled() and self._gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)
        return x

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self._gradient_checkpointing = False


class ExpertRouter(nn.Module):
    """Learned router that selects top-k experts per token.

    Uses a simple linear projection + softmax.
    Includes load-balancing auxiliary loss to prevent expert collapse.
    """

    def __init__(self, config: MedBharatConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.top_k = config.top_k_experts
        self.n_experts = config.n_experts
        self.aux_loss_weight = config.router_aux_loss_weight

    def forward(self, hidden_states: torch.Tensor):
        """Route tokens to experts.

        Args:
            hidden_states: [batch, seq, d_model]

        Returns:
            router_weights: [batch, seq, top_k] — weights for selected experts
            expert_indices: [batch, seq, top_k] — which experts selected
            aux_loss: scalar — load balancing loss
        """
        # Compute routing logits
        router_logits = self.gate(hidden_states)  # [batch, seq, n_experts]

        # Top-k selection
        router_weights, expert_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        router_weights = F.softmax(router_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

        # Load-balancing auxiliary loss
        # Encourages uniform expert utilization
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        # fraction of tokens routed to each expert
        tokens_per_expert = routing_probs.mean(dim=[0, 1])  # [n_experts]
        # ideal: 1/n_experts for each
        aux_loss = self.n_experts * (tokens_per_expert * tokens_per_expert).sum()
        aux_loss = self.aux_loss_weight * aux_loss

        return router_weights, expert_indices, aux_loss


class MixtureOfMedicalExperts(nn.Module):
    """The MoME layer: routes tokens through top-k expert modules."""

    def __init__(self, config: MedBharatConfig):
        super().__init__()
        self.config = config
        self.router = ExpertRouter(config)
        self.experts = nn.ModuleList([
            ExpertModule(config, expert_name=name)
            for name in config.expert_names
        ])

    def forward(self, hidden_states: torch.Tensor, attention_mask=None):
        """Forward pass through MoE layer.

        Each token gets processed by top-2 experts, results weighted and summed.
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Route
        router_weights, expert_indices, aux_loss = self.router(hidden_states)
        # router_weights: [batch, seq, top_k]
        # expert_indices: [batch, seq, top_k]

        # Initialize output
        final_output = torch.zeros_like(hidden_states)

        # Process each expert
        for expert_idx in range(self.config.n_experts):
            expert = self.experts[expert_idx]

            # Find which tokens go to this expert (across all top-k slots)
            # mask: True where this expert was selected
            expert_mask = (expert_indices == expert_idx)  # [batch, seq, top_k]

            if not expert_mask.any():
                continue

            # Get weights for tokens routed to this expert
            # We process ALL tokens through the expert for simplicity
            # (batched is more efficient than sparse indexing for GPU)
            expert_output = expert(hidden_states, attention_mask)

            # Weight by router probability and mask
            for k in range(self.config.top_k_experts):
                slot_mask = expert_mask[:, :, k].unsqueeze(-1)  # [batch, seq, 1]
                slot_weight = router_weights[:, :, k].unsqueeze(-1)  # [batch, seq, 1]
                final_output = final_output + expert_output * slot_mask * slot_weight

        return final_output, aux_loss
