"""Smoke test for the interpretability capstone 4-panel composite figure.

The orchestrator script reads four canonical JSON inputs and writes a single
4-panel PNG + PDF.  This test verifies that:
  1. The orchestrator module imports.
  2. Building the figure with synthetic in-memory data writes a non-empty PNG.
  3. The resulting figure has 4 axes (one per panel).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

# Add worktree root to sys.path so `import scripts...` works from tests.
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

# Import the orchestrator module under test.
from scripts.resdec_mhe.interpretability import (  # noqa: E402
    make_interpretability_capstone_composite as orch,
)


def _write_synthetic_inputs(tmp_path: Path) -> dict[str, Path]:
    """Write minimal-but-valid JSON inputs for all four panels."""
    consensus = {
        "row_cts": ["Splatter", "Fibroblast", "Vascular", "Microglia"],
        "methods": ["IG", "GradientSHAP", "Wasserstein", "CMI"],
        "ranks": {
            "Splatter": {"IG": 1, "GradientSHAP": 1, "Wasserstein": 1, "CMI": 5},
            "Fibroblast": {"IG": 2, "GradientSHAP": 2, "Wasserstein": 3, "CMI": 2},
            "Vascular": {"IG": 7, "GradientSHAP": 6, "Wasserstein": 8, "CMI": 4},
            "Microglia": {"IG": 12, "GradientSHAP": 13, "Wasserstein": 25, "CMI": 27},
        },
        "top5_counts": {
            "Splatter": 4,
            "Fibroblast": 4,
            "Vascular": 1,
            "Microglia": 0,
        },
    }
    wasserstein = {
        "n_resilient": 129,
        "n_vulnerable": 129,
        "per_cell_type": [
            {
                "cell_type": "Splatter",
                "wasserstein_per_gene_mean": 0.0436,
                "wasserstein_per_gene_top10": [
                    ["CTNNA2", 0.310],
                    ["PTPRN", 0.260],
                    ["SHANK2", 0.256],
                    ["TLE5", 0.255],
                    ["ROR1", 0.251],
                    ["ZNF804B", 0.250],
                    ["KIRREL3", 0.241],
                    ["TMSB10", 0.239],
                    ["PDE5A", 0.235],
                    ["KCND2", 0.234],
                ],
            },
            {
                "cell_type": "Astrocyte",
                "wasserstein_per_gene_mean": 0.0166,
                "wasserstein_per_gene_top10": [
                    ["DPP10", 0.177],
                ],
            },
        ],
    }
    xref = {
        "trained": {
            "label": "synthetic",
            "n_total": 2048,
            "relaxed": {
                "n_features": 323,
                "per_ct_counts": {
                    "Microglia": 28,
                    "Oligodendrocyte precursor": 23,
                    "Deep-layer intratelencephalic": 22,
                    "Splatter": 1,
                    "Fibroblast": 9,
                    "Vascular": 9,
                },
            },
        },
    }
    perm = {
        "canonical_mean_r2": 0.4436,
        "n_permutations": 10,
        "null_mean_r2_per_perm": [
            -0.336, -0.241, -0.144, -0.365, -0.321,
            -0.453, -0.297, -0.249, -0.229, -0.304,
        ],
        "null_mean": -0.2944,
        "null_std": 0.0845,
        "z_under_null": 8.73,
        "p_value_one_sided": 0.0909,
    }

    paths = {
        "consensus": tmp_path / "consensus_heatmap_data.json",
        "wasserstein": tmp_path / "wasserstein_per_celltype_pseudobulk.json",
        "xref": tmp_path / "feature_xref_consensus.json",
        "permutation": tmp_path / "permutation_summary.json",
    }
    paths["consensus"].write_text(json.dumps(consensus))
    paths["wasserstein"].write_text(json.dumps(wasserstein))
    paths["xref"].write_text(json.dumps(xref))
    paths["permutation"].write_text(json.dumps(perm))
    return paths


def test_orchestrator_module_imports():
    """The orchestrator module must be importable and expose `build_figure`."""
    assert hasattr(orch, "build_figure")
    assert hasattr(orch, "main")


def test_build_figure_returns_4_axes(tmp_path):
    """`build_figure` must return a matplotlib Figure with exactly 4 axes."""
    paths = _write_synthetic_inputs(tmp_path)
    fig = orch.build_figure(
        consensus_path=paths["consensus"],
        wasserstein_path=paths["wasserstein"],
        xref_path=paths["xref"],
        permutation_path=paths["permutation"],
    )
    # 4 panels.
    assert len(fig.axes) == 4
    plt.close(fig)


def test_main_writes_png_only(tmp_path):
    """End-to-end: orchestrator writes >50KB PNG + PDF in out_dir."""
    paths = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "composite"
    rc = orch.main(
        argv=[
            f"--consensus-data={paths['consensus']}",
            f"--wasserstein-json={paths['wasserstein']}",
            f"--xref-json={paths['xref']}",
            f"--permutation-summary={paths['permutation']}",
            f"--out-dir={out_dir}",
        ]
    )
    assert rc == 0
    png = out_dir / "fig_interpretability_capstone.png"
    pdf = out_dir / "fig_interpretability_capstone.pdf"
    assert png.exists(), f"missing PNG at {png}"
    # PDF intentionally NOT written (user pref — PNG only).
    assert not pdf.exists(), f"unexpected PDF at {pdf}"
    assert png.stat().st_size > 50_000, (
        f"PNG too small ({png.stat().st_size} bytes) — figure likely empty"
    )


def test_panel_letters_present(tmp_path):
    """All four letters A/B/C/D must appear in the figure."""
    paths = _write_synthetic_inputs(tmp_path)
    fig = orch.build_figure(
        consensus_path=paths["consensus"],
        wasserstein_path=paths["wasserstein"],
        xref_path=paths["xref"],
        permutation_path=paths["permutation"],
    )
    letters = {t.get_text() for ax in fig.axes for t in ax.texts}
    for ltr in ("A", "B", "C", "D"):
        assert ltr in letters, f"panel letter {ltr!r} missing; got {letters!r}"
    plt.close(fig)
