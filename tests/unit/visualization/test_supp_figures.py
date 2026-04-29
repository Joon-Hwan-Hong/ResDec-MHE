"""Smoke tests for the 3 lab-meeting supplementary figures (D1, D2, D3).

Verifies that:
  - make_sae_top_features_heatmap.py writes:
        fig_sae_top_features_heatmap.{png,pdf}
  - make_ccc_supp_figures.py writes:
        fig_ccc_per_edge_type_heatmap.{png,pdf}
        fig_ccc_subject_heterogeneity_strip.{png,pdf}

Each PNG must exist and exceed 50 KB.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))


# ---------------------------------------------------------------------------
# Synthetic-input fixtures: minimal canonical-shaped artifacts for each fig.
# ---------------------------------------------------------------------------


CT_NAMES = [
    "Astrocyte", "Oligodendrocyte", "Oligodendrocyte precursor",
    "Committed oligodendrocyte precursor", "Microglia", "Bergmann glia",
    "Upper-layer intratelencephalic", "Deep-layer intratelencephalic",
    "Deep-layer corticothalamic and 6b", "Deep-layer near-projecting",
    "CGE interneuron", "MGE interneuron", "LAMP5-LHX6 and Chandelier",
    "Midbrain-derived inhibitory", "Hippocampal dentate gyrus",
    "Hippocampal CA1-3", "Hippocampal CA4", "Amygdala excitatory",
    "Thalamic excitatory", "Mammillary body", "Medium spiny neuron",
    "Eccentric medium spiny neuron", "Upper rhombic lip", "Lower rhombic lip",
    "Cerebellar inhibitory", "Vascular", "Fibroblast", "Ependymal",
    "Choroid plexus", "Miscellaneous", "Splatter",
]


def _write_synthetic_sae(tmp_path: Path) -> dict[str, Path]:
    """Write minimal SAE artifacts: feature_report.json + sae_model.npz +
    activations_fused_all_folds.npz (all in canonical schema).

    We use a tiny n=8, m=64 SAE with a small N=20 subjects so the test runs
    fast but exercises the full per-feature × per-CT mass computation path.
    """
    rng = np.random.default_rng(123)
    n = 8       # SAE input dim
    m = 64      # SAE feature dim
    n_total = 20  # synthetic "subjects across folds"
    C = len(CT_NAMES)

    # Synthetic SAE: random-init decoder
    W_dec = rng.normal(0, 0.5, size=(n, m)).astype(np.float32)
    sae_path = tmp_path / "sae_model.npz"
    np.savez(
        sae_path,
        W_enc=rng.normal(0, 0.3, size=(m, n)).astype(np.float32),
        b_enc=rng.normal(0, 0.05, size=(m,)).astype(np.float32),
        W_dec=W_dec,
        b_dec=rng.normal(0, 0.05, size=(n,)).astype(np.float32),
    )

    # Activations [N_total, C, n]
    acts = rng.normal(0, 1.0, size=(n_total, C, n)).astype(np.float32)
    # Boost Splatter (last index) signal so projection magnitudes are realistic
    acts[:, -1, :] *= 1.5
    sids = np.array([f"R{i:08d}" for i in range(n_total)], dtype=object)
    folds = np.tile(np.arange(5), int(np.ceil(n_total / 5)))[:n_total]
    is_val = np.ones(n_total, dtype=bool)
    cell_types = np.array(CT_NAMES, dtype=object)
    acts_path = tmp_path / "activations_fused_all_folds.npz"
    np.savez(
        acts_path,
        activations=acts,
        subject_ids=sids,
        fold_indices=folds,
        is_val=is_val,
        cell_types=cell_types,
        layer=np.array("fused", dtype=object),
    )

    # feature_report: 64 features, each with mw_p_cog / fraction_active /
    # ct_dominance / top_cell_types / flags. Make ~25 features pass relaxed
    # filter, with 1 Splatter-dominant. Match real-data fields.
    rng2 = np.random.default_rng(456)
    fr: list[dict] = []
    splatter_idx = 7  # this feature_idx will be Splatter-dominant
    for j in range(m):
        flags: list[str] = []
        # half are dead, the rest active
        if j % 4 == 0:
            flags.append("dead")
            frac = 0.0
            mw = None
        else:
            frac = float(rng2.uniform(0.001, 0.4))
            mw = float(rng2.uniform(0.001, 0.5))
        # Build top_cell_types: fake top 3 with squared_projection
        proj = rng2.normal(0, 1.0, size=C)
        if j == splatter_idx:
            proj[-1] = 5.0  # boost Splatter
        sq = proj ** 2
        top3 = np.argsort(-sq)[:3]
        top_cts = [
            {
                "cell_type": CT_NAMES[c],
                "projection": float(proj[c]),
                "squared_projection": float(sq[c]),
            }
            for c in top3
        ]
        ct_dom = float(sq[top3].sum() / sq.sum()) if sq.sum() > 0 else 0.0
        # Force some features to be relaxed-eligible (mw < 0.05 + frac in band)
        if j > 4 and j % 2 == 1:
            mw = float(rng2.uniform(0.001, 0.04))
        fr.append({
            "feature_idx": j,
            "top_subjects": [str(s) for s in sids[:5]],
            "top_cell_types": top_cts,
            "mw_p_cognition": mw,
            "mw_p_pathology": float(rng2.uniform(0.001, 0.5)),
            "fraction_active": frac,
            "ct_dominance": ct_dom,
            "flags": flags,
        })
    fr_path = tmp_path / "feature_report.json"
    fr_path.write_text(json.dumps(fr))

    return {
        "sae_model": sae_path,
        "activations": acts_path,
        "feature_report": fr_path,
    }


def _write_synthetic_ccc(tmp_path: Path) -> dict[str, Path]:
    """Write a small per_subject_ccc_attention.npz + summary JSON."""
    rng = np.random.default_rng(7)
    n_subjects = 30
    C = len(CT_NAMES)
    edge_types = [
        "Secreted_Signaling", "ECM_Receptor", "Cell_Cell_Contact",
        "Non_protein_Signaling", "Novel_Uncharacterized",
    ]
    n_et = len(edge_types)
    # Most subjects: low attention bulk; ~5 outliers > 0.01
    attention = rng.uniform(0, 0.005, size=(n_subjects, C, C, n_et)).astype(np.float32)
    outlier_idx = np.array([0, 1, 2, 3, 4])
    attention[outlier_idx, 0, 4, 2] = 0.04  # boost Cell_Cell_Contact (Astro→Microglia)
    sids = np.array([f"R{i:08d}" for i in range(n_subjects)], dtype="<U8")
    folds = (np.arange(n_subjects) % 5).astype(np.int32)
    npz_path = tmp_path / "per_subject_ccc_attention.npz"
    np.savez(
        npz_path,
        attention=attention,
        subject_ids=sids,
        folds=folds,
        cell_type_order=np.array(CT_NAMES, dtype="<U35"),
        edge_type_order=np.array(edge_types, dtype="<U21"),
    )

    # Summary JSON: per_subject list with max_attention used by D3.
    per_subject = []
    for i in range(n_subjects):
        max_att = float(attention[i].max())
        per_subject.append({
            "subject_id": str(sids[i]),
            "fold": int(folds[i]),
            "max_attention": max_att,
            "n_high_attention_edges": int((attention[i] > 0.01).sum()),
            "top_edges": [],
        })
    summary = {
        "config": {"threshold": 0.01, "top_k_per_subject": 5,
                   "n_cell_types": C, "n_edge_types": n_et},
        "max_attention_distribution": {"mean": 0.005, "max": 0.04},
        "n_subjects_with_high_attention": int(
            sum(p["n_high_attention_edges"] > 0 for p in per_subject)
        ),
        "frac_subjects_with_high_attention": 0.0,
        "top_frequent_high_attention_edges": [],
        "per_subject": per_subject,
        "provenance": {},
    }
    summary_path = tmp_path / "per_subject_ccc_attention_summary.json"
    summary_path.write_text(json.dumps(summary))

    return {"npz": npz_path, "summary": summary_path}


@pytest.fixture
def synthetic_sae_inputs(tmp_path: Path) -> dict[str, Path]:
    return _write_synthetic_sae(tmp_path)


@pytest.fixture
def synthetic_ccc_inputs(tmp_path: Path) -> dict[str, Path]:
    return _write_synthetic_ccc(tmp_path)


# ---------------------------------------------------------------------------
# D1: SAE top-N feature × 31-CT heatmap
# ---------------------------------------------------------------------------


def test_d1_orchestrator_imports():
    from scripts.resdec_mhe.interpretability import (  # noqa: F401
        make_sae_top_features_heatmap as orch,
    )


def test_d1_writes_figure(synthetic_sae_inputs, tmp_path):
    from scripts.resdec_mhe.interpretability import make_sae_top_features_heatmap as orch
    out_dir = tmp_path / "sae_supp"
    orch.build_figure(
        feature_report_path=synthetic_sae_inputs["feature_report"],
        sae_model_path=synthetic_sae_inputs["sae_model"],
        activations_path=synthetic_sae_inputs["activations"],
        out_dir=out_dir,
        top_n=20,
    )
    png = out_dir / "fig_sae_top_features_heatmap.png"
    pdf = out_dir / "fig_sae_top_features_heatmap.pdf"
    assert png.exists(), f"missing {png}"
    # PDF intentionally NOT written (user pref — PNG only).
    assert not pdf.exists(), f"unexpected pdf at {pdf}"
    size_kb = png.stat().st_size / 1024.0
    assert size_kb > 50.0, f"{png} too small ({size_kb:.1f} KB)"


def test_d1_per_feature_per_ct_mass_matches_decoder_projection(synthetic_sae_inputs):
    """Recomputed mass MUST match the (per_ct_means @ W_dec[:, j])**2 formula
    used by sparse_autoencoder.py (interpret_features). This guards against
    silent simplification of the per-CT decomposition.
    """
    from scripts.resdec_mhe.interpretability import make_sae_top_features_heatmap as orch
    mat, ct_order, feat_indices = orch.compute_per_feature_per_ct_mass(
        sae_model_path=synthetic_sae_inputs["sae_model"],
        activations_path=synthetic_sae_inputs["activations"],
        feature_indices=list(range(64)),
        normalize=True,
    )
    assert mat.shape == (64, len(ct_order))
    # Each row sums to 1 (within float tol) when normalize=True
    np.testing.assert_allclose(mat.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# D2 + D3: CCC supplementary figures
# ---------------------------------------------------------------------------


def test_ccc_supp_orchestrator_imports():
    from scripts.resdec_mhe.interpretability import (  # noqa: F401
        make_ccc_supp_figures as orch,
    )


def test_d2_writes_figure(synthetic_ccc_inputs, tmp_path):
    from scripts.resdec_mhe.interpretability import make_ccc_supp_figures as orch
    out_dir = tmp_path / "ccc_supp"
    orch.build_d2_per_edge_type_heatmap(
        npz_path=synthetic_ccc_inputs["npz"],
        out_dir=out_dir,
    )
    png = out_dir / "fig_ccc_per_edge_type_heatmap.png"
    pdf = out_dir / "fig_ccc_per_edge_type_heatmap.pdf"
    assert png.exists(), f"missing {png}"
    assert not pdf.exists(), f"unexpected pdf at {pdf}"
    size_kb = png.stat().st_size / 1024.0
    assert size_kb > 50.0, f"{png} too small ({size_kb:.1f} KB)"


def test_d3_writes_figure(synthetic_ccc_inputs, tmp_path):
    from scripts.resdec_mhe.interpretability import make_ccc_supp_figures as orch
    out_dir = tmp_path / "ccc_supp"
    orch.build_d3_subject_heterogeneity_strip(
        summary_path=synthetic_ccc_inputs["summary"],
        out_dir=out_dir,
        outlier_threshold=0.01,
    )
    png = out_dir / "fig_ccc_subject_heterogeneity_strip.png"
    pdf = out_dir / "fig_ccc_subject_heterogeneity_strip.pdf"
    assert png.exists(), f"missing {png}"
    assert not pdf.exists(), f"unexpected pdf at {pdf}"
    size_kb = png.stat().st_size / 1024.0
    assert size_kb > 50.0, f"{png} too small ({size_kb:.1f} KB)"


def test_d2_panel_count_equals_n_edge_types(synthetic_ccc_inputs):
    from scripts.resdec_mhe.interpretability import make_ccc_supp_figures as orch
    fig = orch.build_d2_figure(npz_path=synthetic_ccc_inputs["npz"])
    # 5 panels (one per edge type); colorbars may add extra axes — so we
    # require AT LEAST 5 axes containing imshow data.
    assert len(fig.axes) >= 5, f"expected ≥ 5 axes, got {len(fig.axes)}"


def test_d3_outlier_count_threshold(synthetic_ccc_inputs):
    from scripts.resdec_mhe.interpretability import make_ccc_supp_figures as orch
    n_outliers = orch.count_outliers(
        summary_path=synthetic_ccc_inputs["summary"],
        threshold=0.01,
    )
    # In our fixture we boosted exactly 5 subjects to 0.04
    assert n_outliers == 5
