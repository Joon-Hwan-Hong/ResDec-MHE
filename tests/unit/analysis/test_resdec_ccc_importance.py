"""Unit tests for :mod:`src.analysis.resdec_ccc_importance`.

Contract (from docs/plans/2026-04-22-resdec-h3-phase5-finish.md Task C.4):

1. ``extract_hgt_edge_attention`` — drives the encoder with ``return_hgt_attention=True``
   and aggregates per-edge-type attention across HGT layers, returning per-layer and
   per-type tensors indexed by ``(source_ct, target_ct, edge_type)``.
2. ``per_edge_type_ablation`` — for each edge type ``k``, runs inference with type-``k``
   edges dropped from every batch, computes val-R² delta vs. baseline composite R².
   Deterministic tests only cover the edge-masking primitive + R² bookkeeping;
   full model inference is exercised by the orchestration script.
3. ``liana_correlation`` — pure Pandas join + Pearson/Spearman on ranked importance;
   must report ``n_pairs`` (intersection size).

Full ablation involves a ResDec checkpoint + dataloader round-trip, which is too
slow and memory-hungry for unit tests. Those paths are exercised manually via the
orchestration script and recorded in the run log for reproducibility.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import torch

from src.analysis.resdec_ccc_importance import (
    aggregate_attention_by_celltype_pair,
    aggregate_attention_by_edge_type,
    drop_edges_of_type,
    extract_hgt_edge_attention,
    liana_correlation,
    load_liana_reference,
)
from src.data.constants import ALL_EDGE_TYPES, CELL_TYPE_ORDER, N_EDGE_TYPES
from src.training.resdec_lightning_module import ENCODER_KWARG_KEYS


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: synthetic encoder model + batch
# ─────────────────────────────────────────────────────────────────────────────


class _FakeHGTModel(torch.nn.Module):
    """Minimal stand-in for CognitiveResilienceModel exposing the attention API.

    Emits deterministic attention values that depend only on ``edge_type`` so
    the aggregation assertions have closed-form expected values.
    """

    def __init__(self, n_heads: int = 2, n_layers: int = 3, n_edge_types: int = 5):
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_edge_types = n_edge_types

    def forward(
        self,
        *,
        ccc_edge_index: torch.Tensor,
        ccc_edge_type: torch.Tensor,
        ccc_edge_attr: torch.Tensor,
        return_hgt_attention: bool = False,
        **kwargs,
    ) -> dict:
        # Sanity: any extra kwargs must be in the public encoder contract —
        # catches regressions where the extractor starts passing private keys.
        unexpected = set(kwargs) - set(ENCODER_KWARG_KEYS)
        assert not unexpected, (
            f"_FakeHGTModel received non-encoder kwargs: {sorted(unexpected)}; "
            f"allowed = {sorted(ENCODER_KWARG_KEYS)}"
        )
        E = ccc_edge_type.shape[0]
        # Deterministic: layer ℓ attention = (edge_type + 1) * (ℓ + 1)
        # Use dtype float to simulate real scatter-softmax output.
        attn_list: list[torch.Tensor] = []
        for layer in range(self.n_layers):
            vals = (ccc_edge_type.float() + 1.0) * (layer + 1)
            attn = vals.unsqueeze(-1).expand(E, self.n_heads).contiguous()
            attn_list.append(attn)
        out: dict = {
            "mean": torch.zeros(1, 1),
            "attended": torch.zeros(1, 8),
        }
        if return_hgt_attention:
            out["hgt_attention"] = attn_list
        return out


def _make_toy_batch(device: torch.device) -> dict:
    """Construct a small toy batch — 2 subjects, 4 edges total, 3 edge types."""
    # ccc_edge_index: [2, E_total], node indices already offset per-subject
    # Subject 0 has cell_types 0-1 → node indices [0, 1]
    # Subject 1 has cell_types 0-1 → node indices [n_cell_types, n_cell_types+1]
    ccc_edge_index = torch.tensor(
        [[0, 1, 31, 32], [1, 0, 32, 31]], dtype=torch.long, device=device
    )
    ccc_edge_type = torch.tensor([0, 1, 2, 0], dtype=torch.long, device=device)
    ccc_edge_attr = torch.ones(4, 1, device=device)
    return {
        "ccc_edge_index": ccc_edge_index,
        "ccc_edge_type": ccc_edge_type,
        "ccc_edge_attr": ccc_edge_attr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# extract_hgt_edge_attention
# ─────────────────────────────────────────────────────────────────────────────


def test_extract_hgt_edge_attention_shape():
    """Extractor returns dict with expected keys / shapes."""
    device = torch.device("cpu")
    model = _FakeHGTModel(n_heads=2, n_layers=3, n_edge_types=5).to(device).eval()
    batch = _make_toy_batch(device)

    result = extract_hgt_edge_attention(model, batch, device=device, n_edge_types=5)

    assert "per_edge_type_attention" in result
    assert "per_layer_attention" in result
    assert "per_edge_type_counts" in result

    # per_edge_type_attention: [n_edge_types]
    per_et = result["per_edge_type_attention"]
    assert per_et.shape == (5,)
    # Only types 0, 1, 2 are present in the batch (4 edges total)
    assert np.isfinite(per_et[[0, 1, 2]]).all()

    # per_layer_attention: [n_layers, n_edge_types]
    per_layer = result["per_layer_attention"]
    assert per_layer.shape == (3, 5)

    # per_edge_type_counts: [n_edge_types] — integer edge counts
    counts = result["per_edge_type_counts"]
    assert counts.shape == (5,)
    assert counts[0] == 2 and counts[1] == 1 and counts[2] == 1
    assert counts[3] == 0 and counts[4] == 0


def test_extract_hgt_edge_attention_values():
    """Fake model returns attention = (edge_type + 1) * (layer + 1); verify aggregation."""
    device = torch.device("cpu")
    model = _FakeHGTModel(n_heads=2, n_layers=3, n_edge_types=5).to(device).eval()
    batch = _make_toy_batch(device)

    result = extract_hgt_edge_attention(model, batch, device=device, n_edge_types=5)

    # Attention per (layer, edge_type): (edge_type + 1) * (layer + 1), averaged over heads (→ identical).
    # Expected per_layer_attention[layer, et] = (et + 1) * (layer + 1)
    expected = np.array(
        [
            [1.0, 2.0, 3.0, np.nan, np.nan],  # layer 0
            [2.0, 4.0, 6.0, np.nan, np.nan],  # layer 1
            [3.0, 6.0, 9.0, np.nan, np.nan],  # layer 2
        ]
    )
    per_layer = result["per_layer_attention"]
    # Layers × edge types 0..2 are present; 3,4 should be NaN (no edges).
    np.testing.assert_allclose(per_layer[:, :3], expected[:, :3])
    assert np.isnan(per_layer[:, 3:]).all()

    # per_edge_type_attention is mean over layers per type (nan-mean over present only)
    per_et = result["per_edge_type_attention"]
    np.testing.assert_allclose(per_et[:3], np.array([2.0, 4.0, 6.0]))  # mean of [1,2,3], [2,4,6], [3,6,9]
    assert np.isnan(per_et[3:]).all()


def test_extract_hgt_edge_attention_pair_breakdown():
    """With return_pair_breakdown=True the extractor also returns a (source_ct,
    target_ct, edge_type) DataFrame whose row count == number of unique triples."""
    device = torch.device("cpu")
    model = _FakeHGTModel(n_heads=2, n_layers=3, n_edge_types=5).to(device).eval()
    batch = _make_toy_batch(device)

    result = extract_hgt_edge_attention(
        model,
        batch,
        device=device,
        n_edge_types=5,
        n_nodes_per_graph=31,
        return_pair_breakdown=True,
    )

    pair_df = result["per_pair_attention"]
    # 4 toy edges, all distinct (src_ct, tgt_ct, edge_type) triples.
    assert len(pair_df) == 4
    assert set(pair_df.columns) >= {
        "source_ct_idx", "target_ct_idx", "edge_type", "mean_attention", "n_edges",
    }
    # n_edges per row sums to 4 (total edges in batch).
    assert int(pair_df["n_edges"].sum()) == 4


def test_extract_hgt_edge_attention_pair_breakdown_requires_n_nodes():
    """return_pair_breakdown=True without n_nodes_per_graph raises ValueError."""
    device = torch.device("cpu")
    model = _FakeHGTModel(n_heads=2, n_layers=3, n_edge_types=5).to(device).eval()
    batch = _make_toy_batch(device)
    with pytest.raises(ValueError, match="n_nodes_per_graph"):
        extract_hgt_edge_attention(
            model, batch, device=device, n_edge_types=5,
            return_pair_breakdown=True,
        )


def test_extract_hgt_edge_attention_zero_edges():
    """Zero-edge batch returns a dict with all NaN attention and zero counts."""
    device = torch.device("cpu")
    model = _FakeHGTModel(n_heads=2, n_layers=3, n_edge_types=5).to(device).eval()
    batch = {
        "ccc_edge_index": torch.zeros(2, 0, dtype=torch.long, device=device),
        "ccc_edge_type": torch.zeros(0, dtype=torch.long, device=device),
        "ccc_edge_attr": torch.zeros(0, 1, device=device),
    }

    result = extract_hgt_edge_attention(model, batch, device=device, n_edge_types=5)
    assert np.isnan(result["per_edge_type_attention"]).all()
    assert (result["per_edge_type_counts"] == 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# drop_edges_of_type (ablation primitive)
# ─────────────────────────────────────────────────────────────────────────────


def test_drop_edges_of_type_removes_only_target_type():
    """Dropping type 1 keeps edges of type 0, 2."""
    device = torch.device("cpu")
    batch = _make_toy_batch(device)
    new_batch = drop_edges_of_type(batch, edge_type_idx=1)

    assert new_batch["ccc_edge_type"].shape[0] == 3  # 4 original − 1 of type 1
    assert 1 not in new_batch["ccc_edge_type"].tolist()
    # Edge attributes and indices should match the filtered rows
    assert new_batch["ccc_edge_attr"].shape[0] == 3
    assert new_batch["ccc_edge_index"].shape == (2, 3)


def test_drop_edges_of_type_all_removed():
    """Dropping the only type produces an empty edge batch."""
    device = torch.device("cpu")
    batch = {
        "ccc_edge_index": torch.tensor([[0], [1]], dtype=torch.long, device=device),
        "ccc_edge_type": torch.tensor([3], dtype=torch.long, device=device),
        "ccc_edge_attr": torch.ones(1, 1, device=device),
    }
    new_batch = drop_edges_of_type(batch, edge_type_idx=3)
    assert new_batch["ccc_edge_type"].shape[0] == 0
    assert new_batch["ccc_edge_index"].shape[1] == 0
    assert new_batch["ccc_edge_attr"].shape[0] == 0


def test_drop_edges_of_type_preserves_other_keys():
    """Non-edge keys (pathology, pseudobulk, etc.) pass through unchanged."""
    device = torch.device("cpu")
    batch = _make_toy_batch(device)
    batch["pathology"] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    batch["subject_ids"] = ["S0", "S1"]
    new_batch = drop_edges_of_type(batch, edge_type_idx=0)
    assert torch.equal(new_batch["pathology"], batch["pathology"])
    assert new_batch["subject_ids"] == ["S0", "S1"]


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_attention_by_celltype_pair
# ─────────────────────────────────────────────────────────────────────────────


def test_aggregate_attention_by_celltype_pair_shape_and_values():
    """Given a flat [E, H] attention and edge index/type, aggregate to
    (source_ct, target_ct, edge_type) → mean."""
    device = torch.device("cpu")
    # 4 edges, all within 1 subject graph (no batch offset).
    ccc_edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)
    ccc_edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    attention = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # [E, 1 head]

    df = aggregate_attention_by_celltype_pair(
        attention=attention,
        edge_index=ccc_edge_index,
        edge_type=ccc_edge_type,
        n_nodes_per_graph=31,
    )

    # 4 rows, one per (source, target, edge_type). (Or fewer if pairs repeat;
    # here all four are distinct (src, tgt, et) triples.)
    assert len(df) == 4
    for col in ("source_ct_idx", "target_ct_idx", "edge_type", "mean_attention", "n_edges"):
        assert col in df.columns


def test_aggregate_attention_by_celltype_pair_batch_offset():
    """Node indices that exceed n_nodes_per_graph get modulo'd back to per-graph indices."""
    ccc_edge_index = torch.tensor([[0, 31, 62], [1, 32, 63]], dtype=torch.long)
    ccc_edge_type = torch.tensor([0, 0, 0], dtype=torch.long)
    attention = torch.ones(3, 1)

    df = aggregate_attention_by_celltype_pair(
        attention=attention,
        edge_index=ccc_edge_index,
        edge_type=ccc_edge_type,
        n_nodes_per_graph=31,
    )
    # All three edges are the same local (0 → 1, type 0) pair, so aggregated output has 1 row
    assert len(df) == 1
    assert df["source_ct_idx"].iloc[0] == 0
    assert df["target_ct_idx"].iloc[0] == 1
    assert df["n_edges"].iloc[0] == 3


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_attention_by_edge_type
# ─────────────────────────────────────────────────────────────────────────────


def test_aggregate_attention_by_edge_type_basic():
    """Mean attention per edge type index, plus counts."""
    attention = torch.tensor([[1.0], [3.0], [5.0], [7.0]])  # [E, 1 head]
    edge_type = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    mean_per_type, counts = aggregate_attention_by_edge_type(
        attention=attention, edge_type=edge_type, n_edge_types=5,
    )
    # Type 0: mean(1, 3) = 2, count=2. Type 1: 5, count=1. Type 2: 7, count=1. 3,4: NaN, 0.
    np.testing.assert_allclose(mean_per_type[:3], np.array([2.0, 5.0, 7.0]))
    assert np.isnan(mean_per_type[3:]).all()
    np.testing.assert_array_equal(counts, np.array([2, 1, 1, 0, 0]))


# ─────────────────────────────────────────────────────────────────────────────
# liana_correlation
# ─────────────────────────────────────────────────────────────────────────────


def _ranking_df(src: list[str], tgt: list[str], imp: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"source_ct": src, "target_ct": tgt, "importance": imp})


def _liana_df(src: list[str], tgt: list[str], score: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"source": src, "target": tgt, "score": score})


def test_liana_correlation_perfect_rank_match():
    """Identical rank orders → Spearman ρ = 1.0, Pearson in [−1, 1].

    Uses ``higher_is_better=True`` so the (synthetic) score column is treated
    as "higher = more important" (no sign flip).
    """
    our = _ranking_df(
        src=["A", "A", "B", "C"], tgt=["B", "C", "A", "B"], imp=[4.0, 3.0, 2.0, 1.0]
    )
    liana = _liana_df(
        src=["A", "A", "B", "C"], tgt=["B", "C", "A", "B"], score=[40.0, 30.0, 20.0, 10.0]
    )
    result = liana_correlation(our, liana, score_col="score", higher_is_better=True)
    assert result["n_pairs"] == 4
    assert result["spearman_rho"] == pytest.approx(1.0, abs=1e-10)
    assert result["pearson_r"] == pytest.approx(1.0, abs=1e-10)


def test_liana_correlation_sign_flip_on_rank_score():
    """LIANA's magnitude_rank is "lower = more important"; with default
    ``higher_is_better=False`` the sign is inverted so that perfect agreement
    (our imp ↑, liana rank ↓) yields positive correlation."""
    our = _ranking_df(
        src=["A", "A", "B", "C"], tgt=["B", "C", "A", "B"], imp=[4.0, 3.0, 2.0, 1.0]
    )
    # LIANA-style: lower magnitude_rank = more important.
    liana = _liana_df(
        src=["A", "A", "B", "C"], tgt=["B", "C", "A", "B"],
        score=[0.001, 0.01, 0.1, 0.5],
    )
    result = liana_correlation(our, liana, score_col="score", higher_is_better=False)
    assert result["n_pairs"] == 4
    # Sign-flipped LIANA ranks are monotonic-increasing with our importance.
    assert result["spearman_rho"] == pytest.approx(1.0, abs=1e-10)


def test_liana_correlation_anti_correlated():
    """Reversed rank orders → Spearman ρ = -1.0."""
    our = _ranking_df(
        src=["A", "A", "B", "C"], tgt=["B", "C", "A", "B"], imp=[4.0, 3.0, 2.0, 1.0]
    )
    liana = _liana_df(
        src=["A", "A", "B", "C"], tgt=["B", "C", "A", "B"], score=[10.0, 20.0, 30.0, 40.0]
    )
    result = liana_correlation(our, liana, score_col="score", higher_is_better=True)
    assert result["n_pairs"] == 4
    assert result["spearman_rho"] == pytest.approx(-1.0, abs=1e-10)


def test_liana_correlation_handles_missing_pairs(caplog):
    """Pairs in ``our_ranking`` absent from ``liana_df`` are dropped; a WARNING is emitted."""
    our = _ranking_df(
        src=["A", "A", "Z"], tgt=["B", "C", "Y"], imp=[3.0, 2.0, 1.0]
    )
    liana = _liana_df(src=["A", "A"], tgt=["B", "C"], score=[30.0, 20.0])
    with caplog.at_level(logging.WARNING):
        result = liana_correlation(our, liana, score_col="score", higher_is_better=True)
    assert result["n_pairs"] == 2
    assert result["n_missing"] == 1
    # WARNING emitted about dropped pairs
    assert any("missing" in r.getMessage().lower() or "drop" in r.getMessage().lower()
               for r in caplog.records)


def test_liana_correlation_empty_intersection_returns_nan():
    """Zero overlap returns NaN statistics + n_pairs=0."""
    our = _ranking_df(src=["A"], tgt=["B"], imp=[1.0])
    liana = _liana_df(src=["C"], tgt=["D"], score=[1.0])
    result = liana_correlation(our, liana, score_col="score", higher_is_better=True)
    assert result["n_pairs"] == 0
    assert np.isnan(result["spearman_rho"])
    assert np.isnan(result["pearson_r"])


def test_liana_correlation_constant_ranking_returns_nan():
    """Degenerate ranking (all identical values) → NaN correlation (scipy contract)."""
    our = _ranking_df(src=["A", "B", "C"], tgt=["X", "Y", "Z"], imp=[1.0, 1.0, 1.0])
    liana = _liana_df(src=["A", "B", "C"], tgt=["X", "Y", "Z"], score=[10.0, 20.0, 30.0])
    result = liana_correlation(our, liana, score_col="score", higher_is_better=True)
    assert result["n_pairs"] == 3
    # Spearman of constant vector is undefined → NaN.
    assert np.isnan(result["spearman_rho"])


def test_liana_correlation_result_has_aggregation_level_field():
    """M5: downstream consumers rely on ``aggregation_level`` for figure captions."""
    our = _ranking_df(src=["A", "B"], tgt=["C", "D"], imp=[1.0, 2.0])
    liana = _liana_df(src=["A", "B"], tgt=["C", "D"], score=[10.0, 20.0])
    result = liana_correlation(our, liana, score_col="score", higher_is_better=True)
    assert "aggregation_level" in result
    assert result["aggregation_level"] == "population_mean_source_target"


# ─────────────────────────────────────────────────────────────────────────────
# load_liana_reference (M2)
# ─────────────────────────────────────────────────────────────────────────────


def test_load_liana_reference_roundtrip(tmp_path):
    """Happy path: write a LIANA parquet, round-trip back with expected columns."""
    df = pd.DataFrame({
        "source": ["A", "B"],
        "target": ["C", "D"],
        "magnitude_rank": [0.1, 0.5],
        "subject_id": ["s1", "s1"],
    })
    df.to_parquet(tmp_path / "liana_s1.parquet")
    loaded = load_liana_reference(tmp_path, subject_ids=["s1"])
    assert len(loaded) == 2
    assert set(loaded.columns) >= {"source", "target", "magnitude_rank", "subject_id"}


def test_load_liana_reference_missing_subject_raises(tmp_path):
    """Requesting a nonexistent subject leaves zero parquets → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_liana_reference(tmp_path, subject_ids=["nonexistent"])


def test_load_liana_reference_missing_score_col_raises(tmp_path):
    """Schema check raises ValueError with the available columns listed."""
    df = pd.DataFrame({
        "source": ["A"], "target": ["B"],
        "magnitude_rank": [0.1], "subject_id": ["s1"],
    })
    df.to_parquet(tmp_path / "liana_s1.parquet")
    with pytest.raises(ValueError, match="missing required columns"):
        load_liana_reference(tmp_path, subject_ids=["s1"], score_col="lrscore")


# ─────────────────────────────────────────────────────────────────────────────
# drop_edges_of_type edge cases (M3)
# ─────────────────────────────────────────────────────────────────────────────


def test_drop_edges_of_type_no_edges_key_returns_as_is():
    """If the batch has no ccc_edge_type key, drop_edges_of_type is a no-op."""
    batch = {"pathology": torch.ones(2, 3)}
    out = drop_edges_of_type(batch, edge_type_idx=0)
    assert out is batch  # shallow no-op: same object, no copy
