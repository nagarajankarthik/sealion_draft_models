import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3Model, DFlash2Qwen3ForCausalLM

class DFlash2Qwen3AdaFlashModel(DFlash2Qwen3Model):
    def __init__(self, *, vllm_config: VllmConfig, start_layer_id: int = 0,
                 prefix: str = "", **kwargs) -> None:
        super().__init__(vllm_config=vllm_config, 
                         start_layer_id=start_layer_id, 
                         prefix=prefix,kwargs=**kwargs)
        self.thresh_head = None
        draft_config = self.config.dflash_config
        vocab_size = vllm_config.model_config.get_vocab_size()
        spec_config = vllm_config.speculative_config
        hf_config = spec_config.draft_model_config.hf_config
        hidden_size = (
            getattr(hf_config, "target_hidden_size", None) or hf_config.hidden_size
        )
        if draft_config.use_thresh_head_two_model:
            self.thresh_head = ThreshHeadTwoModel(
                hidden_size=hidden_size,
                bottleneck_dim=int(draft_config.get("thresh_head_bottleneck_dim", 256)),
            )
        elif draft_config.use_thresh_head_subsequent:
            self.thresh_head = ThreshHeadSubsequent(
                vocab_size=vocab_size,
                bottleneck_dim=int(draft_config.get("thresh_head_bottleneck_dim", 256)),
                top_k=int(draft_config.get("thresh_head_topk", 5)),
            )
        else:
            raise ValueError(f"Unknown threshold head type: {draft_config.thresh_head}")


class DFlash2Qwen3AdaFlashForCausalLM(DFlash2Qwen3ForCausalLM):

    model_cls = DFlash2Qwen3AdaFlashModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

# ---------------------------------------------------------------------------
# Threshold Head modules (adaptive threshold prediction)
# ---------------------------------------------------------------------------

class ThreshHeadSubsequent(nn.Module):
    """Predicts per-block threshold from draft logit statistics.
    Extracts softmax features (max_prob, entropy, top-k probs) per position,
    then aggregates over block and projects to a scalar threshold."""
    def __init__(self, vocab_size: int, bottleneck_dim: int = 256, top_k: int = 5):
        super().__init__()
        self.top_k = top_k
        # Per-position features: max_prob + entropy + top_k probs = 2 + top_k
        feat_dim = (2 + top_k)
        # Block-level: concat per-position features across block_size, then project
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: [bs, block_size, vocab_size] -> [bs, 1]."""
        probs = torch.softmax(logits.float(), dim=-1)
        max_prob = probs.max(dim=-1).values  # [bs, block_size]
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [bs, block_size]
        topk_probs = probs.topk(self.top_k, dim=-1).values  # [bs, block_size, top_k]
        # Concat features: [bs, block_size, 2 + top_k]
        features = torch.cat([max_prob.unsqueeze(-1), entropy.unsqueeze(-1), topk_probs], dim=-1)
        # Mean-pool across block positions
        pooled = features.mean(dim=-2)  # [bs, 2 + top_k]
        return torch.sigmoid(self.proj(pooled.to(self.proj[0].weight.dtype)))  # [bs, 1]


class ThreshHeadTwoModel(nn.Module):
    """Predicts per-block adaptive threshold from draft hidden states.
    Concatenates mean-pool and last-position features, projects through
    two-layer bottleneck with residual, outputs sigmoid."""
    def __init__(self, hidden_size: int, bottleneck_dim: int = 256):
        super().__init__()
        self.down = nn.Linear(hidden_size * 2, bottleneck_dim)
        self.res = nn.Sequential(nn.Linear(bottleneck_dim, bottleneck_dim), nn.SiLU())
        self.out = nn.Linear(bottleneck_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [bs, block_size, hidden_size] -> [bs, 1]."""
        feat = torch.cat([x.mean(dim=-2), x[:, -1]], dim=-1)  # [bs, 2*hidden_size]
        h = nn.functional.silu(self.down(feat))  # [bs, bottleneck_dim]
        h = h + self.res(h)  # residual
        return torch.sigmoid(self.out(h))  # [bs, 1]


