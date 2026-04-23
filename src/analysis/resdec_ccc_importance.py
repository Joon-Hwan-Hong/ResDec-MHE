"""CCC (cell-cell communication) interpretability for ResDec-H3.

Three deterministic pieces (plus one ablation driver that hits a real checkpoint):

1. ``extract_hgt_edge_attention(model, batch, device, n_edge_types)`` — runs the
   encoder with ``return_hgt_attention=True`` (already supported by
   :class:`CognitiveResilienceModel` and :class:`HGTEncoderTensor`) and
   aggregates the list of per-layer ``[E_total, H]`` attention tensors into
   per-edge-type summaries: a ``[n_edge_types]`` head-averaged mean and a
   ``[n_layers, n_edge_types]`` per-layer variant, plus edge-type counts.

2. ``drop_edges_of_type(batch, edge_type_idx)`` — returns a shallow copy of
   ``batch`` with rows of ``ccc_edge_index``, ``ccc_edge_type`` and
   ``ccc_edge_attr`` belonging to ``edge_type_idx`` removed. Because HGT uses
   flat concatenated edges + scatter-softmax, physically deleting rows *is*
   the ablation — no mask-fiddling needed. All non-edge keys are passed through.

3. ``per_edge_type_ablation(lit_module, val_dataloader, n_edge_types, device,
   tabpfn_val_map)`` — for each edge type ``k``, walks the val dataloader,
   drops type-``k`` edges from every batch, runs the ResDec composite forward
   (head residual + TabPFN outer), accumulates predictions + targets, and
   returns the per-edge-type R² delta (baseline R² − ablated R²). Baseline is
   computed in the same pass (first loop iteration, ``k == None``).

4. ``liana_correlation(our_ranking, liana_df, score_col)`` — pure Pandas join
   on ``(source_ct, target_ct)`` with ``(source, target)`` + Pearson / Spearman
   on the aligned vectors. Reports ``n_pairs`` and ``n_missing`` so downstream
   readers can see the intersection size.

The orchestration in ``scripts/redesign/interpretability/ccc_composite_attribution.py``
sweeps all 5 folds: per-fold ckpt load → extract edge attention → run ablation
sweep → aggregate → correlate against subject-aggregated LIANA scores.

Unit tests (``tests/unit/analysis/test_resdec_ccc_importance.py``) exercise the
deterministic pieces with a synthetic encoder + toy edge batch. Full ablation
is validated end-to-end by running the orchestration script on the canonical
checkpoints.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. HGT edge-attention extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_hgt_edge_attention(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
    n_edge_types: int,
) -> dict[str, np.ndarray]:
    """Run one forward pass with ``return_hgt_attention=True`` and aggregate.

    Relies on :class:`CognitiveResilienceModel` (and :class:`HGTEncoderTensor`)
    propagating ``return_hgt_attention`` through to each HGT layer, which emits
    a ``[E_total, H]`` attention tensor per layer. We aggregate by edge type
    (head-averaged mean) per layer, then across layers.

    Args:
        model: Module exposing ``forward(..., return_hgt_attention=True)``.
            Typically the full :class:`CognitiveResilienceModel` (not the
            Lightning wrapper).
        batch: A collate output with ``ccc_edge_index``, ``ccc_edge_type``,
            ``ccc_edge_attr`` and any other inputs the encoder needs (already
            on ``device``).
        device: Compute device.
        n_edge_types: Total number of edge types (index range: ``[0, n_edge_types)``).

    Returns:
        Dict with keys:

        - ``per_edge_type_attention``: ``[n_edge_types]`` head-averaged mean attention
          across all HGT layers. NaN for edge types absent in the batch.
        - ``per_layer_attention``: ``[n_layers, n_edge_types]`` per-layer mean
          attention (head-averaged). NaN where the type has zero edges.
        - ``per_edge_type_counts``: ``[n_edge_types]`` integer counts of edges
          per type in the batch (unchanged across layers).
    """
    edge_type = batch["ccc_edge_type"]
    E = edge_type.shape[0]
    if E == 0:
        return {
            "per_edge_type_attention": np.full(n_edge_types, np.nan, dtype=np.float64),
            "per_layer_attention": np.full((0, n_edge_types), np.nan, dtype=np.float64),
            "per_edge_type_counts": np.zeros(n_edge_types, dtype=np.int64),
        }

    # ``model`` expected to be in eval mode; caller's responsibility. We still
    # wrap in no_grad for memory + safety.
    with torch.no_grad():
        out = model(**_forward_kwargs(batch, return_hgt_attention=True))

    attn_list = out.get("hgt_attention")
    if attn_list is None or len(attn_list) == 0:
        raise RuntimeError(
            "Model forward did not return 'hgt_attention'. Ensure the encoder "
            "propagates return_hgt_attention=True (CognitiveResilienceModel does)."
        )

    # Stack over layers: [n_layers, E_total, H]
    attn_stack = torch.stack([a.detach().to(device=device, dtype=torch.float32) for a in attn_list], dim=0)
    n_layers = attn_stack.shape[0]

    # Head-average: [n_layers, E_total]
    per_edge = attn_stack.mean(dim=-1).cpu().numpy()
    et_np = edge_type.cpu().numpy()

    # Aggregate per (layer, edge_type)
    per_layer_attention = np.full((n_layers, n_edge_types), np.nan, dtype=np.float64)
    counts = np.zeros(n_edge_types, dtype=np.int64)
    for k in range(n_edge_types):
        mask = et_np == k
        if not mask.any():
            continue
        counts[k] = int(mask.sum())
        for ell in range(n_layers):
            per_layer_attention[ell, k] = float(per_edge[ell, mask].mean())

    # Across-layer mean per type (nan-mean ignores missing layers; since each
    # type is either present in every layer or absent in every layer, nan-mean
    # is equivalent to plain mean when the type is present). For absent types
    # all layers are NaN → nanmean emits "Mean of empty slice" warning; silence
    # it since NaN is the intended output for missing types.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        with np.errstate(invalid="ignore"):
            per_edge_type_attention = np.nanmean(per_layer_attention, axis=0)

    return {
        "per_edge_type_attention": per_edge_type_attention,
        "per_layer_attention": per_layer_attention,
        "per_edge_type_counts": counts,
    }


def _forward_kwargs(batch: dict, *, return_hgt_attention: bool = False) -> dict:
    """Select keys that :class:`CognitiveResilienceModel.forward` accepts.

    Matches :data:`src.training.resdec_lightning_module._ENCODER_KWARG_KEYS`
    plus the interpretability flag(s).
    """
    keys = (
        "region_pseudobulk",
        "region_mask",
        "pseudobulk",
        "ccc_edge_index",
        "ccc_edge_type",
        "ccc_edge_attr",
        "cell_type_mask",
        "pathology",
        "cognition",
        "cell_data",
        "cell_offsets",
    )
    kwargs = {k: batch.get(k) for k in keys if k in batch}
    kwargs["return_hgt_attention"] = return_hgt_attention
    return kwargs


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cell-type-pair aggregation (for LIANA correlation)
# ─────────────────────────────────────────────────────────────────────────────


def aggregate_attention_by_celltype_pair(
    attention: torch.Tensor,  # [E, H] head-level attention
    edge_index: torch.Tensor,  # [2, E]
    edge_type: torch.Tensor,  # [E]
    n_nodes_per_graph: int,
) -> pd.DataFrame:
    """Collapse a flat HGT attention tensor to (source_ct, target_ct, edge_type) means.

    The collate function offsets node indices per sample
    (``local_idx + sample_idx * n_nodes_per_graph``) so we modulo back to get
    per-graph cell-type indices.

    Args:
        attention: ``[E, H]`` attention values (will be head-averaged).
        edge_index: ``[2, E]`` batch-offset source/target node indices.
        edge_type: ``[E]`` edge-type indices.
        n_nodes_per_graph: Number of cell types (31) used to de-offset node indices.

    Returns:
        DataFrame with columns
        ``["source_ct_idx", "target_ct_idx", "edge_type", "mean_attention", "n_edges"]``.
    """
    if attention.ndim != 2:
        raise ValueError(f"attention must be 2D [E, H], got shape {tuple(attention.shape)}")

    attn_avg = attention.detach().cpu().float().mean(dim=-1).numpy()  # [E]
    ei = edge_index.detach().cpu().numpy()  # [2, E]
    et = edge_type.detach().cpu().numpy()  # [E]

    src_ct = ei[0] % n_nodes_per_graph
    tgt_ct = ei[1] % n_nodes_per_graph

    df = pd.DataFrame(
        {
            "source_ct_idx": src_ct,
            "target_ct_idx": tgt_ct,
            "edge_type": et,
            "attention": attn_avg,
        }
    )
    grouped = df.groupby(["source_ct_idx", "target_ct_idx", "edge_type"], sort=True)
    return grouped.agg(
        mean_attention=("attention", "mean"),
        n_edges=("attention", "size"),
    ).reset_index()


def aggregate_attention_by_edge_type(
    attention: torch.Tensor,  # [E, H]
    edge_type: torch.Tensor,  # [E]
    n_edge_types: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Head-averaged mean attention per edge type.

    Args:
        attention: ``[E, H]``.
        edge_type: ``[E]``.
        n_edge_types: Total number of types.

    Returns:
        Tuple of ``(mean_per_type [n_edge_types], counts_per_type [n_edge_types])``.
        Types with zero edges get NaN + 0.
    """
    if attention.shape[0] == 0:
        return (
            np.full(n_edge_types, np.nan, dtype=np.float64),
            np.zeros(n_edge_types, dtype=np.int64),
        )
    attn_avg = attention.detach().cpu().float().mean(dim=-1).numpy()
    et_np = edge_type.detach().cpu().numpy()
    mean_per_type = np.full(n_edge_types, np.nan, dtype=np.float64)
    counts = np.zeros(n_edge_types, dtype=np.int64)
    for k in range(n_edge_types):
        mask = et_np == k
        counts[k] = int(mask.sum())
        if counts[k] > 0:
            mean_per_type[k] = float(attn_avg[mask].mean())
    return mean_per_type, counts


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-edge-type ablation
# ─────────────────────────────────────────────────────────────────────────────


def drop_edges_of_type(batch: dict, edge_type_idx: int) -> dict:
    """Return a shallow copy of ``batch`` with edges of ``edge_type_idx`` removed.

    All non-edge keys pass through unchanged (including device and dtype). If no
    ``ccc_edge_*`` keys are present, ``batch`` is returned as-is.
    """
    if "ccc_edge_type" not in batch or batch["ccc_edge_type"] is None:
        return batch

    et = batch["ccc_edge_type"]
    keep = et != edge_type_idx
    new_batch = dict(batch)
    new_batch["ccc_edge_type"] = et[keep]
    if "ccc_edge_index" in batch and batch["ccc_edge_index"] is not None:
        new_batch["ccc_edge_index"] = batch["ccc_edge_index"][:, keep]
    if "ccc_edge_attr" in batch and batch["ccc_edge_attr"] is not None:
        new_batch["ccc_edge_attr"] = batch["ccc_edge_attr"][keep]
    return new_batch


def _run_composite_inference(
    lit_module: Any,
    batch: dict,
    device: torch.device,
    tabpfn_val_map: Optional[dict[str, tuple[float, float]]] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the ResDec composite forward (head residual + optional TabPFN outer).

    Mirrors :meth:`ResDecLightningModule.validation_step` but without logging.

    Args:
        lit_module: The ResDec Lightning module (already moved to ``device`` + ``eval``).
        batch: Collate output already on ``device``.
        device: Compute device.
        tabpfn_val_map: Optional ``{subject_id: (y_tabpfn, sigma_tabpfn)}``; when
            provided the residual head's output is added to TabPFN. When ``None``,
            the module's own ``tabpfn_val_map`` is used if TabPFN was enabled.

    Returns:
        Tuple ``(y_pred [B], y_true [B], subject_ids list-of-str)``.
    """
    with torch.no_grad():
        out = lit_module.forward(batch)
    pred = out["prediction"].detach().cpu().float()
    target = batch["cognition"].detach().cpu().float()
    if target.dim() == 2 and target.shape[-1] == 1:
        target = target.squeeze(-1)

    # TabPFN compose: prefer explicit map arg, else fall back to the module's.
    effective_map = tabpfn_val_map
    if effective_map is None and getattr(lit_module, "_tabpfn_enabled", False):
        effective_map = lit_module.tabpfn_val_map

    subject_ids = list(batch["subject_ids"])
    if effective_map:
        y_tabpfn = np.array(
            [effective_map[sid][0] for sid in subject_ids], dtype=np.float32,
        )
        pred_np = pred.numpy().astype(np.float32) + y_tabpfn
    else:
        pred_np = pred.numpy().astype(np.float32)
    return pred_np, target.numpy().astype(np.float32), subject_ids


def per_edge_type_ablation(
    lit_module: Any,
    val_dataloader: Any,
    n_edge_types: int,
    device: torch.device,
    edge_type_names: Optional[list[str]] = None,
) -> dict:
    """For each edge type ``k``, run inference with type-``k`` edges dropped.

    One baseline pass + ``n_edge_types`` ablated passes. R² is computed over the
    concatenated val set per pass. Returns both absolute R² and the delta
    (baseline − ablated) per type — positive delta means removing the type
    hurts the model (= edge type was useful).

    Args:
        lit_module: Lightning module with ``forward(batch) -> dict`` (must be
            on ``device`` and in ``eval`` mode).
        val_dataloader: Iterable yielding collate-batched dicts.
        n_edge_types: Number of edge types to sweep.
        device: Compute device.
        edge_type_names: Optional human-readable names for the reported table.

    Returns:
        Dict with:

        - ``baseline_r2``: float — R² on the full (unablated) val set.
        - ``per_edge_type``: list of dicts
          ``{edge_type_idx, edge_type_name, ablated_r2, r2_delta, n_edges_ablated}``.
    """
    from sklearn.metrics import r2_score

    lit_module = lit_module.to(device).eval()

    def _move_batch(b: dict) -> dict:
        """Move tensors in ``b`` to ``device`` (floats untouched — Lightning
        handles AMP separately; we run in eval/fp32)."""
        out: dict = {}
        for k, v in b.items():
            if torch.is_tensor(v):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out

    # Cache the batches once so we don't iterate the dataloader twice (slow on
    # big datasets + sampler state matters). Edges are small — the memory cost
    # is in the pseudobulk + cell_data tensors which share the loader's buffer.
    cached_batches: list[dict] = []
    for b in val_dataloader:
        cached_batches.append(_move_batch(b))

    # Baseline pass
    preds_base: list[np.ndarray] = []
    targets_base: list[np.ndarray] = []
    for b in cached_batches:
        p, t, _sids = _run_composite_inference(lit_module, b, device)
        preds_base.append(p)
        targets_base.append(t)
    y_pred_base = np.concatenate(preds_base, axis=0)
    y_true_base = np.concatenate(targets_base, axis=0)
    baseline_r2 = float(r2_score(y_true_base, y_pred_base))
    logger.info("Ablation baseline R²=%.4f  (n=%d)", baseline_r2, len(y_true_base))

    # Count total edges per type across the val set (diagnostic).
    total_edges_per_type = np.zeros(n_edge_types, dtype=np.int64)
    for b in cached_batches:
        et = b.get("ccc_edge_type")
        if et is None or et.numel() == 0:
            continue
        et_np = et.detach().cpu().numpy()
        for k in range(n_edge_types):
            total_edges_per_type[k] += int((et_np == k).sum())

    # Ablated passes
    per_edge_results: list[dict] = []
    for k in range(n_edge_types):
        preds_abl: list[np.ndarray] = []
        targets_abl: list[np.ndarray] = []
        for b in cached_batches:
            b_abl = drop_edges_of_type(b, edge_type_idx=k)
            p, t, _sids = _run_composite_inference(lit_module, b_abl, device)
            preds_abl.append(p)
            targets_abl.append(t)
        y_pred_abl = np.concatenate(preds_abl, axis=0)
        y_true_abl = np.concatenate(targets_abl, axis=0)
        ablated_r2 = float(r2_score(y_true_abl, y_pred_abl))
        et_name = (
            edge_type_names[k]
            if edge_type_names is not None and k < len(edge_type_names)
            else f"edge_type_{k}"
        )
        per_edge_results.append(
            {
                "edge_type_idx": int(k),
                "edge_type_name": et_name,
                "ablated_r2": ablated_r2,
                "r2_delta": float(baseline_r2 - ablated_r2),
                "n_edges_ablated": int(total_edges_per_type[k]),
            }
        )
        logger.info(
            "Ablation type %d (%s): ablated R²=%.4f, Δ=%.4f (dropped %d edges)",
            k,
            et_name,
            ablated_r2,
            baseline_r2 - ablated_r2,
            total_edges_per_type[k],
        )

    return {
        "baseline_r2": baseline_r2,
        "baseline_n": int(len(y_true_base)),
        "per_edge_type": per_edge_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. LIANA correlation
# ─────────────────────────────────────────────────────────────────────────────


def liana_correlation(
    our_ranking: pd.DataFrame,
    liana_df: pd.DataFrame,
    score_col: str = "magnitude_rank",
    higher_is_better: bool = False,
) -> dict:
    """Correlate our (source_ct, target_ct)→importance against LIANA scores.

    Args:
        our_ranking: DataFrame with columns ``["source_ct", "target_ct", "importance"]``
            (higher = more important in our model).
        liana_df: DataFrame with at least ``["source", "target", score_col]``.
            May have multiple rows per (source, target) — aggregated by mean first.
        score_col: Score column to use. Default ``"magnitude_rank"`` (LIANA's
            rank percentile; lower = more important).
        higher_is_better: Whether a higher ``score_col`` means more important.
            For ``magnitude_rank`` / ``specificity_rank`` this is ``False`` (LIANA
            ranks are percentiles where lower = more important); the function
            inverts the sign so a positive correlation means our importance
            agrees with LIANA's most-important calls.

    Returns:
        Dict with ``pearson_r``, ``spearman_rho``, ``n_pairs``, ``n_missing``,
        ``score_col``.
    """
    if not {"source_ct", "target_ct", "importance"}.issubset(our_ranking.columns):
        raise ValueError(
            "our_ranking must have columns ['source_ct', 'target_ct', 'importance']"
        )
    if not {"source", "target", score_col}.issubset(liana_df.columns):
        raise ValueError(
            f"liana_df must have columns ['source', 'target', '{score_col}']"
        )

    # Aggregate LIANA by (source, target) first — LIANA has one row per LR pair,
    # but we're comparing cell-type-pair-level importance. Use mean of the score.
    liana_agg = (
        liana_df.groupby(["source", "target"], as_index=False)[score_col].mean()
        .rename(columns={"source": "source_ct", "target": "target_ct"})
    )

    # Optionally flip sign so "higher importance = bigger number" for correlation.
    if not higher_is_better:
        liana_agg[score_col] = -liana_agg[score_col]

    # Left-join on cell-type pairs; count unmatched.
    joined = our_ranking.merge(liana_agg, on=["source_ct", "target_ct"], how="inner")
    n_pairs = len(joined)
    n_missing = len(our_ranking) - n_pairs
    if n_missing > 0:
        logger.warning(
            "liana_correlation: dropped %d/%d pairs absent from LIANA reference "
            "(n_pairs kept = %d)",
            n_missing,
            len(our_ranking),
            n_pairs,
        )

    if n_pairs == 0:
        return {
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "n_pairs": 0,
            "n_missing": int(n_missing),
            "score_col": score_col,
        }

    from scipy.stats import pearsonr, spearmanr

    our_vec = joined["importance"].to_numpy(dtype=np.float64)
    liana_vec = joined[score_col].to_numpy(dtype=np.float64)

    # Degenerate input → NaN (matches scipy's contract).
    if np.std(our_vec) == 0 or np.std(liana_vec) == 0:
        pearson_r = float("nan")
        spearman_rho = float("nan")
    else:
        pearson_r = float(pearsonr(our_vec, liana_vec).statistic)
        spearman_rho = float(spearmanr(our_vec, liana_vec).correlation)

    return {
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "n_pairs": int(n_pairs),
        "n_missing": int(n_missing),
        "score_col": score_col,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LIANA reference data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_liana_reference(
    liana_dir: Path,
    subject_ids: Optional[list[str]] = None,
    score_col: str = "magnitude_rank",
) -> pd.DataFrame:
    """Load LIANA per-subject parquets and aggregate by (source, target).

    Reads all ``liana_<subject>.parquet`` files in ``liana_dir`` (or just the
    provided ``subject_ids``), keeps ``[source, target, score_col]`` columns,
    and returns the concatenated frame without aggregation. Caller can group
    however they like (mean across subjects, per-subject, etc.).

    Args:
        liana_dir: Directory containing ``liana_<subject>.parquet`` files.
        subject_ids: Optional whitelist. When ``None``, loads all parquets
            in the directory (excluding ``liana_NA.parquet`` which has no
            subject_id).
        score_col: Which score column to retain (default ``magnitude_rank``).

    Returns:
        Concatenated DataFrame with columns ``[source, target, score_col, subject_id]``.
    """
    liana_dir = Path(liana_dir)
    if not liana_dir.is_dir():
        raise FileNotFoundError(f"LIANA directory not found: {liana_dir}")

    if subject_ids is None:
        paths = sorted(
            p for p in liana_dir.glob("liana_*.parquet")
            if not p.name.endswith("_NA.parquet")
        )
    else:
        paths = [liana_dir / f"liana_{sid}.parquet" for sid in subject_ids]
        missing = [p for p in paths if not p.exists()]
        if missing:
            logger.warning(
                "LIANA: %d/%d subject parquets missing (e.g. %s)",
                len(missing), len(paths), missing[0].name,
            )
            paths = [p for p in paths if p.exists()]

    if not paths:
        raise FileNotFoundError(f"No LIANA parquets matched in {liana_dir}")

    dfs = []
    for p in paths:
        df = pd.read_parquet(p, columns=["source", "target", score_col, "subject_id"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


__all__ = [
    "aggregate_attention_by_celltype_pair",
    "aggregate_attention_by_edge_type",
    "drop_edges_of_type",
    "extract_hgt_edge_attention",
    "liana_correlation",
    "load_liana_reference",
    "per_edge_type_ablation",
]
