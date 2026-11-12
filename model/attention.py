"""Medical Sparse Attention for MedBharat-1.

Standard multi-head attention with an addition:
- 12 heads: standard causal attention (like LLaMA)
- 4 heads:  medical-focus attention with learnable bias
  These heads learn to attend more strongly to medical entity tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import MedBharatConfig


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) — same as LLaMA."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MedicalSparseAttention(nn.Module):
    """Multi-head attention with medical-focus heads.

    - Standard heads (12): normal causal self-attention
    - Medical heads (4): causal attention + learned position-independent bias
      that helps the model attend to medically important tokens
    """

    def __init__(self, config: MedBharatConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_medical_heads = config.n_medical_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

        # Medical attention bias: learnable per-head bias that modulates attention
        # These 4 heads learn a query-dependent bias for medical entity focus
        self.medical_gate = nn.Linear(config.d_model, config.n_medical_heads, bias=True)

        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_seq_len, config.rope_theta
        )
        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask (use -1e4 instead of -inf for MPS compatibility)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), -1e4, device=hidden_states.device),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        # Padding mask
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0, -1e4
            )

        # Medical-focus bias for the last n_medical_heads
        # Compute a per-token "medical importance" score
        med_bias = self.medical_gate(hidden_states)  # [batch, seq, n_medical_heads]
        med_bias = torch.sigmoid(med_bias)  # 0-1 importance
        # Shape: [batch, n_medical_heads, 1, seq] — bias on key positions
        med_bias = med_bias.permute(0, 2, 1).unsqueeze(2)

        # Apply medical bias only to the medical heads (last n_medical_heads)
        n_std = self.n_heads - self.n_medical_heads
        std_attn = attn_weights[:, :n_std, :, :]
        med_attn = attn_weights[:, n_std:, :, :] + med_bias
        attn_weights = torch.cat([std_attn, med_attn], dim=1)

        # Softmax + dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_proj(attn_output)
