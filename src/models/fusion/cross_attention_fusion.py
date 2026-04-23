"""
Cross-attention fusion layer for combining two-branch embeddings.

Replaces concat+linear fusion with pairwise mutual cross-attention.
Each branch attends to the other branch (2 ops total),
then enriched branches are combined via per-cell-type learned weighted sum.

Cross-attention is scale-invariant (softmax normalizes attention scores).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseCrossAttention(nn.Module):
    """Single pairwise cross-attention: query branch attends to key-value branch.

    Standard multi-head attention with pre-LayerNorm on Q and K.
    Supports three attention modes:
    - "standard": normal softmax(Q·K^T/√d) — emphasizes correlated features
    - "reverse": softmax(-Q·K^T/√d) — CrossFuse (Li et al., 2024), emphasizes
      complementary/uncorrelated features
    - "blend": α·softmax(scores) + (1-α)·softmax(-scores) — learnable per-head
      blend of standard and reverse attention

    Args:
        d_model: Embedding dimension (shared by Q and KV branches).
        n_heads: Number of attention heads.
        dropout: Dropout on attention weights.
        mode: Attention mode ("standard", "reverse", "blend").
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        mode: str = "standard",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        if mode not in ("standard", "reverse", "blend"):
            raise ValueError(
                f"mode must be 'standard', 'reverse', or 'blend', got '{mode}'"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.mode = mode

        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

        # Blend mode: learnable α per head, initialized at 0.5 (equal blend)
        if mode == "blend":
            self.blend_logits = nn.Parameter(torch.zeros(n_heads))

    def forward(
        self,
        query: torch.Tensor,   # [B, N_q, d_model]
        kv: torch.Tensor,      # [B, N_kv, d_model]
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query branch embeddings [B, N_q, d_model]
            kv: Key-value branch embeddings [B, N_kv, d_model]
            return_attention: If True, return attention weights alongside output.

        Returns:
            attended: [B, N_q, d_model]
            attention_weights: [B, n_heads, N_q, N_kv] (only if return_attention)
        """
        B, N_q, _ = query.shape
        N_kv = kv.shape[1]

        # Pre-LayerNorm
        q = self.q_norm(query)
        k = self.k_norm(kv)
        v = kv  # Values are not normalized (standard practice)

        # Project to multi-head format
        q = self.q_proj(q).view(B, N_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(k).view(B, N_kv, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(v).view(B, N_kv, self.n_heads, self.d_head).transpose(1, 2)
        # q: [B, H, N_q, d_head], k/v: [B, H, N_kv, d_head]

        drop_p = self.dropout if self.training else 0.0

        if self.mode == "standard":
            # Standard softmax attention via SDPA
            attended = F.scaled_dot_product_attention(
                q, k, v, dropout_p=drop_p,
            )
        elif self.mode == "reverse":
            # CrossFuse: softmax(-scores) — attend to LEAST similar (complementary)
            # SDPA doesn't support negated scores directly, so compute manually
            scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / (self.d_head ** 0.5)
            attn_weights_r = F.softmax(-scores, dim=-1)
            if drop_p > 0 and self.training:
                attn_weights_r = F.dropout(attn_weights_r, p=drop_p)
            attended = torch.einsum("bhqk,bhkd->bhqd", attn_weights_r, v)
        elif self.mode == "blend":
            # Learnable blend: α·softmax(scores) + (1-α)·softmax(-scores)
            scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / (self.d_head ** 0.5)
            attn_standard = F.softmax(scores, dim=-1)
            attn_reverse = F.softmax(-scores, dim=-1)
            # α per head: sigmoid(blend_logits) ∈ (0, 1)
            alpha = torch.sigmoid(self.blend_logits).view(1, self.n_heads, 1, 1)
            attn_blended = alpha * attn_standard + (1 - alpha) * attn_reverse
            if drop_p > 0 and self.training:
                attn_blended = F.dropout(attn_blended, p=drop_p)
            attended = torch.einsum("bhqk,bhkd->bhqd", attn_blended, v)

        # Reshape and project output
        attended = attended.transpose(1, 2).reshape(B, N_q, self.d_model)
        attended = self.out_proj(attended)

        if return_attention:
            with torch.no_grad():
                scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / (self.d_head ** 0.5)
                if self.mode == "standard":
                    attn_weights = F.softmax(scores.float(), dim=-1)
                elif self.mode == "reverse":
                    attn_weights = F.softmax(-scores.float(), dim=-1)
                else:  # blend
                    alpha = torch.sigmoid(self.blend_logits).view(1, self.n_heads, 1, 1)
                    attn_weights = (
                        alpha * F.softmax(scores.float(), dim=-1)
                        + (1 - alpha) * F.softmax(-scores.float(), dim=-1)
                    )
            return attended, attn_weights

        return attended


class CrossAttentionFusionLayer(nn.Module):
    """
    Pairwise mutual cross-attention fusion for two branches (HGT + CT).

    Each branch attends to the other branch (2 operations),
    producing enriched representations. Enriched branches are combined
    via per-cell-type learned weighted sum (B2).

    Scale-invariant: softmax normalizes attention scores regardless of
    input activation magnitudes.

    Args:
        d_embed: Branch embedding dimension.
        d_fused: Output fused dimension.
        n_cell_types: Number of cell types (default: 31).
        n_heads: Number of attention heads (default: 4).
        dropout: Dropout probability (default: 0.1).
    """

    # Branch names for indexing B2 weights and attention maps
    BRANCH_NAMES = ("hgt", "cell_transformer")

    def __init__(
        self,
        d_embed: int,
        d_fused: int,
        n_cell_types: int = 31,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_pma_seeds: int = 1,
        attention_mode: str = "standard",
    ):
        super().__init__()

        if d_embed <= 0:
            raise ValueError(f"d_embed must be positive, got {d_embed}")
        if d_fused <= 0:
            raise ValueError(f"d_fused must be positive, got {d_fused}")
        if n_cell_types <= 0:
            raise ValueError(f"n_cell_types must be positive, got {n_cell_types}")

        self.d_embed = d_embed
        self.d_fused = d_fused
        self.n_cell_types = n_cell_types
        self.n_heads = n_heads
        self.n_pma_seeds = n_pma_seeds
        self.attention_mode = attention_mode

        # Project cell_emb from n_pma_seeds * d_embed → d_embed if n_pma_seeds > 1
        d_cell_emb = n_pma_seeds * d_embed
        if d_cell_emb != d_embed:
            self.cell_input_proj = nn.Linear(d_cell_emb, d_embed)
        else:
            self.cell_input_proj = None

        # 2 pairwise cross-attention operations
        # Named: {query_branch}_from_{kv_branch}
        self.hgt_from_ct = PairwiseCrossAttention(d_embed, n_heads, dropout, mode=attention_mode)
        self.ct_from_hgt = PairwiseCrossAttention(d_embed, n_heads, dropout, mode=attention_mode)

        # Per-branch enrichment LayerNorms (applied after residual sum)
        self.hgt_enrich_norm = nn.LayerNorm(d_embed)
        self.ct_enrich_norm = nn.LayerNorm(d_embed)

        # B2: per-cell-type learned branch weights [2, n_cell_types]
        # softmax(dim=0) → each cell type's weights sum to 1 across branches
        self.branch_weight_logits = nn.Parameter(torch.zeros(2, n_cell_types))

        # Output projection if d_embed != d_fused
        if d_embed != d_fused:
            self.output_proj = nn.Linear(d_embed, d_fused)
        else:
            self.output_proj = None

        # Post-fusion normalization
        self.layer_norm = nn.LayerNorm(d_fused)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        hgt_emb: torch.Tensor,          # [B, 31, d_embed]
        cell_emb: torch.Tensor,         # [B, 31, d_embed]
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Fuse two branches via pairwise mutual cross-attention.

        Args:
            hgt_emb: [B, n_cell_types, d_embed]
            cell_emb: [B, n_cell_types, d_embed]
            return_attention: If True, return dict of attention maps and branch weights.

        Returns:
            fused: [B, n_cell_types, d_fused]
            attention_info: dict (only if return_attention=True) with keys:
                'branch_weights': [2, n_cell_types] softmax-normalized
                'hgt_from_ct', 'ct_from_hgt': [B, n_heads, 31, 31] attention maps
        """
        # Input validation
        if hgt_emb.dim() != 3:
            raise ValueError(f"Expected 3D hgt_emb, got shape {hgt_emb.shape}")
        if cell_emb.dim() != 3:
            raise ValueError(f"Expected 3D cell_emb, got shape {cell_emb.shape}")

        B, C, D = hgt_emb.shape
        if C != self.n_cell_types:
            raise ValueError(f"Expected {self.n_cell_types} cell types, got {C}")
        if D != self.d_embed:
            raise ValueError(
                f"Expected d_embed={self.d_embed} for hgt_emb, got {D}"
            )

        # Project cell_emb to d_embed if n_pma_seeds > 1
        if self.cell_input_proj is not None:
            cell_emb = self.cell_input_proj(cell_emb)

        attention_maps = {} if return_attention else None

        # ── 2 pairwise cross-attention operations ────────────────────────
        def _cross_attn(module, query, kv, name):
            if return_attention:
                out, attn = module(query, kv, return_attention=True)
                attention_maps[name] = attn
                return out
            return module(query, kv, return_attention=False)

        hgt_from_ct = _cross_attn(self.hgt_from_ct, hgt_emb, cell_emb, "hgt_from_ct")
        ct_from_hgt = _cross_attn(self.ct_from_hgt, cell_emb, hgt_emb, "ct_from_hgt")

        # ── Per-branch enrichment (residual + LayerNorm) ─────────────────
        hgt_enriched = self.hgt_enrich_norm(hgt_emb + hgt_from_ct)
        ct_enriched = self.ct_enrich_norm(cell_emb + ct_from_hgt)

        # ── B2: per-cell-type learned weighted sum ───────────────────────
        # branch_weights: [2, n_cell_types], each column sums to 1
        branch_weights = F.softmax(self.branch_weight_logits, dim=0)
        # Reshape for broadcasting: [2, 1, n_cell_types, 1]
        w = branch_weights.unsqueeze(1).unsqueeze(-1)  # [2, 1, C, 1]

        # Stack enriched branches: [2, B, C, d_embed]
        stacked = torch.stack([hgt_enriched, ct_enriched], dim=0)

        # Weighted sum: [B, C, d_embed]
        fused = (w * stacked).sum(dim=0)

        # ── Output projection if needed ──────────────────────────────────
        if self.output_proj is not None:
            fused = self.output_proj(fused)

        fused = self.dropout_layer(self.layer_norm(fused))

        if return_attention:
            attention_maps["branch_weights"] = branch_weights.detach()
            return fused, attention_maps

        return fused

    def get_branch_weights(self) -> torch.Tensor:
        """Get per-cell-type branch importance weights.

        Returns:
            [2, n_cell_types] tensor, softmax-normalized along dim=0.
            Row order: hgt, cell_transformer.
        """
        return F.softmax(self.branch_weight_logits, dim=0).detach()

    def get_branch_weight_dict(self) -> dict[str, torch.Tensor]:
        """Get branch weights as named dict for interpretability.

        Returns:
            Dict mapping branch name to [n_cell_types] weight tensor.
        """
        weights = self.get_branch_weights()
        return {name: weights[i] for i, name in enumerate(self.BRANCH_NAMES)}

    def extra_repr(self) -> str:
        return (
            f"d_embed={self.d_embed}, d_fused={self.d_fused}, "
            f"n_cell_types={self.n_cell_types}, n_heads={self.n_heads}"
        )
