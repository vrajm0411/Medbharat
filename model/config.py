from dataclasses import dataclass


@dataclass
class MedBharatConfig:
    """MedBharat-1 model configuration.

    Architecture: Curriculum-trained Mixture of Medical Experts
    with Dual-Script support and Medical Sparse Attention.

    Total params:  ~500M
    Active params: ~260M per forward pass (top-2 experts)
    """

    # Vocabulary & Embedding
    vocab_size: int = 48_000          # 32K English + 12K Hindi + 4K medical symbols
    max_seq_len: int = 256            # reduced for 8GB RAM + MPS
    d_model: int = 512                # reduced from 1024
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Shared Transformer Layers (bottom stack, all inputs go through these)
    n_shared_layers: int = 6          # reduced from 8
    n_heads: int = 8                  # reduced from 16
    n_medical_heads: int = 2          # reduced from 4
    head_dim: int = 64                # d_model / n_heads
    intermediate_size: int = 2048     # reduced from 4096

    # Mixture of Medical Experts (top stack, routed)
    n_experts: int = 6                # DIAG, PHARMA, TREAT, AYUSH, ANATOMY, GENERAL
    n_expert_layers: int = 4          # reduced from 8
    top_k_experts: int = 2            # activate 2 experts per token
    expert_intermediate_size: int = 1024  # reduced from 2048
    router_aux_loss_weight: float = 0.01  # load balancing loss weight

    # Language embedding (for dual-script)
    n_languages: int = 3              # 0=English, 1=Hindi, 2=Mixed/Hinglish

    # Normalization & Activation
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0       # RoPE base frequency

    # Regularization
    dropout: float = 0.0              # no dropout during pre-training (standard)
    attention_dropout: float = 0.0

    # Training
    gradient_checkpointing: bool = False  # disabled for MPS compatibility
    use_cache: bool = False           # disable KV cache during training

    # Expert names (for logging / interpretability)
    expert_names: tuple = (
        "diagnosis",
        "pharmacology",
        "treatment",
        "ayush",
        "anatomy",
        "general",
    )

    @property
    def total_params_estimate(self) -> str:
        emb = self.vocab_size * self.d_model
        shared = self.n_shared_layers * (
            4 * self.d_model * self.d_model  # attention
            + 3 * self.d_model * self.intermediate_size  # SwiGLU
        )
        expert_single = self.n_expert_layers * (
            4 * self.d_model * self.d_model
            + 3 * self.d_model * self.expert_intermediate_size
        )
        experts_total = self.n_experts * expert_single
        router = self.d_model * self.n_experts
        total = emb + shared + experts_total + router
        active = emb + shared + self.top_k_experts * expert_single + router
        return (
            f"Total: {total / 1e6:.0f}M params | "
            f"Active: {active / 1e6:.0f}M params per forward"
        )
