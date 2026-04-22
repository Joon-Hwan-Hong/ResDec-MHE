"""Build enriched flat-feature vectors for TabPFN / XGBoost.

Used by the scoping experiment that tests whether TabPFN-2.6 benefits from
CCC + composition + pathology + region_mask on top of the gene-expression-only
pseudobulk baseline.

Feature sets:
    A             -> pseudobulk.flatten()                         [148_335]
    A+C           -> + cell-type proportions                      [148_366]
    A+C+E         -> + CCC dense (31*31*5) + CCC aggregate (18)   [153_189]
    A+C+E+P+R     -> + pathology (3) + region_mask (6)            [153_198]

Pathology columns are ``["gpath", "amylsqrt", "tangsqrt"]`` (same as in
``src.data.datasets.CognitiveResilienceDataset``). Region mask is the 6-dim
boolean region indicator baked into each precomputed .pt file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.data.tabpfn_input import flatten_pseudobulk

logger = logging.getLogger(__name__)

# ── Constants shared with scripts/analysis/run_baselines.py ─────────────────
N_CCC_TYPES = 5
N_CELL_TYPES = 31

# Pathology columns used by CognitiveResilienceDataset (src/data/datasets.py).
PATHOLOGY_COLUMNS = ("gpath", "amylsqrt", "tangsqrt")

# Region mask length baked into precomputed .pt files.
N_REGIONS = 6

FEATURE_SETS = ("A", "A+C", "A+C+E", "A+C+E+P+R")


# ── Per-component extractors ────────────────────────────────────────────────

def extract_composition(pt_data: dict) -> np.ndarray:
    """Cell-type proportions [31], cell_counts normalised to sum to 1."""
    cell_counts = pt_data["cell_counts"].float()
    total = cell_counts.sum()
    if total > 0:
        proportions = cell_counts / total
    else:
        proportions = torch.zeros_like(cell_counts)
    return proportions.numpy().astype(np.float32)


def extract_ccc_dense(pt_data: dict) -> np.ndarray:
    """Dense CCC tensor [31 * 31 * 5] = 4805, flattened in (sender, receiver,
    type) order.

    Each cell ``[s, r, t]`` is the SUM of ``edge_attr[:, 0]`` over edges with
    source=s, target=r, edge_type=t. Most cells will be 0 because CCC edges
    are sparse.
    """
    edge_index = pt_data["ccc_edge_index"]  # [2, n_edges]
    edge_type = pt_data["ccc_edge_type"]    # [n_edges]
    edge_attr = pt_data["ccc_edge_attr"]    # [n_edges, edge_dim]

    dense = torch.zeros(
        (N_CELL_TYPES, N_CELL_TYPES, N_CCC_TYPES), dtype=torch.float32
    )
    n_edges = edge_index.shape[1]
    if n_edges == 0:
        return dense.flatten().numpy()

    src = edge_index[0].long()
    dst = edge_index[1].long()
    t = edge_type.long()
    # edge_attr is [n_edges, 1]; take the scalar per edge.
    attr = edge_attr[:, 0].float()

    # Vectorised scatter-add: flatten (src, dst, t) to a 1-D index and
    # accumulate attr values into a flat tensor, then reshape.
    flat_idx = (
        src * (N_CELL_TYPES * N_CCC_TYPES)
        + dst * N_CCC_TYPES
        + t
    )
    flat = dense.flatten()
    flat.scatter_add_(0, flat_idx, attr)
    return flat.numpy()


def extract_ccc_aggregate(pt_data: dict) -> np.ndarray:
    """CCC graph summary features [18].

    Matches ``scripts/analysis/run_baselines.extract_features_e``:
    Per-type (5 types): edge count, mean edge attribute, std edge attribute
    Global node-degree: mean, std, max (based on source nodes).
    Total: 5 + 5 + 5 + 3 = 18.
    """
    edge_index = pt_data["ccc_edge_index"]
    edge_type = pt_data["ccc_edge_type"]
    edge_attr = pt_data["ccc_edge_attr"]

    n_edges = edge_index.shape[1]

    counts = np.zeros(N_CCC_TYPES, dtype=np.float32)
    mean_attrs = np.zeros(N_CCC_TYPES, dtype=np.float32)
    std_attrs = np.zeros(N_CCC_TYPES, dtype=np.float32)

    for t in range(N_CCC_TYPES):
        mask = edge_type == t
        c = int(mask.sum().item())
        counts[t] = c
        if c > 0:
            attrs_t = edge_attr[mask]
            mean_attrs[t] = attrs_t.mean().item()
            std_attrs[t] = attrs_t.std().item() if c > 1 else 0.0

    # Node degree statistics (source-node out-degree).
    if n_edges > 0:
        src_nodes = edge_index[0].long()
        degrees = torch.bincount(src_nodes, minlength=N_CELL_TYPES).float()
    else:
        degrees = torch.zeros(N_CELL_TYPES, dtype=torch.float32)

    degree_mean = float(degrees.mean().item())
    degree_std = float(degrees.std().item()) if N_CELL_TYPES > 1 else 0.0
    degree_max = float(degrees.max().item())

    return np.concatenate([
        counts, mean_attrs, std_attrs,
        np.array([degree_mean, degree_std, degree_max], dtype=np.float32),
    ]).astype(np.float32)


def extract_region_mask(pt_data: dict) -> np.ndarray:
    """Region-mask boolean indicator [6], cast to float32 (0.0 / 1.0)."""
    mask = pt_data["region_mask"]
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    if mask.shape[0] != N_REGIONS:
        raise ValueError(
            f"Expected region_mask of length {N_REGIONS}, got {mask.shape}"
        )
    return mask.float().numpy().astype(np.float32)


# ── Assembly ────────────────────────────────────────────────────────────────

FEATURE_SET_SIZES = {
    "A": 31 * 4785,                                 # 148_335
    "A+C": 31 * 4785 + 31,                          # 148_366
    "A+C+E": 31 * 4785 + 31 + 31 * 31 * 5 + 18,     # 153_189
    "A+C+E+P+R": (
        31 * 4785 + 31 + 31 * 31 * 5 + 18
        + len(PATHOLOGY_COLUMNS) + N_REGIONS         # 153_198
    ),
}


def build_features(
    pt_data: dict,
    feature_set: str,
    pathology_vec: np.ndarray | None = None,
) -> np.ndarray:
    """Return the concatenated feature vector for ``feature_set``.

    Args:
        pt_data: Loaded ``.pt`` dict (see ``data/precomputed/R*.pt``).
        feature_set: One of :data:`FEATURE_SETS`.
        pathology_vec: Required ``float32`` array of shape ``(3,)`` when the
            feature set requests "+P" (i.e. ``"A+C+E+P+R"``). Order must
            match :data:`PATHOLOGY_COLUMNS`. Pass ``None`` otherwise.

    Returns:
        ``float32`` 1-D array of length :data:`FEATURE_SET_SIZES`
        ``[feature_set]``.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set={feature_set!r}. "
            f"Supported: {list(FEATURE_SETS)}"
        )

    parts: list[np.ndarray] = [flatten_pseudobulk(pt_data).numpy()]

    if feature_set in {"A+C", "A+C+E", "A+C+E+P+R"}:
        parts.append(extract_composition(pt_data))

    if feature_set in {"A+C+E", "A+C+E+P+R"}:
        parts.append(extract_ccc_dense(pt_data))
        parts.append(extract_ccc_aggregate(pt_data))

    if feature_set == "A+C+E+P+R":
        if pathology_vec is None:
            raise ValueError(
                "pathology_vec is required for feature_set='A+C+E+P+R'."
            )
        pathology_vec = np.asarray(pathology_vec, dtype=np.float32)
        if pathology_vec.shape != (len(PATHOLOGY_COLUMNS),):
            raise ValueError(
                f"pathology_vec must be shape {(len(PATHOLOGY_COLUMNS),)}, "
                f"got {pathology_vec.shape}"
            )
        parts.append(pathology_vec)
        parts.append(extract_region_mask(pt_data))

    out = np.concatenate(parts).astype(np.float32)
    expected = FEATURE_SET_SIZES[feature_set]
    if out.shape[0] != expected:
        raise RuntimeError(
            f"Assembled feature vector has length {out.shape[0]} but "
            f"expected {expected} for feature_set={feature_set!r}"
        )
    return out


# ── Loaders that mirror src.data.feature_loaders ────────────────────────────

def load_pathology(
    meta_csv: Path,
    subject_ids: Iterable[str],
    columns: tuple[str, ...] = PATHOLOGY_COLUMNS,
    id_col: str = "ROSMAP_IndividualID",
) -> dict[str, np.ndarray]:
    """Load pathology vectors per subject.

    Drops subjects with NaN in ANY pathology column (to keep the enriched
    feature matrix dense).

    Returns: ``{subject_id -> np.ndarray[len(columns)] float32}``.
    """
    subject_ids = list(subject_ids)
    df = pd.read_csv(meta_csv)
    wanted = set(subject_ids)
    df = df[df[id_col].isin(wanted)]

    out: dict[str, np.ndarray] = {}
    n_null = 0
    for _, r in df.iterrows():
        vals = [r.get(c) for c in columns]
        if any(pd.isna(v) for v in vals):
            n_null += 1
            continue
        out[r[id_col]] = np.asarray(vals, dtype=np.float32)

    logger.info(
        "load_pathology (%s): %d/%d subjects with all-non-null pathology "
        "(null_any=%d, not_in_meta=%d)",
        list(columns), len(out), len(subject_ids),
        n_null, len(wanted) - len(df),
    )
    return out


def load_enriched_features(
    precomputed_dir: Path,
    subject_ids: Iterable[str],
    feature_set: str,
    pathology: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Per-subject enriched feature vectors for the given feature set.

    Mirrors :func:`src.data.feature_loaders.load_flat_features` but emits the
    concatenated feature vector defined by ``feature_set``.

    Subjects without a precomputed ``.pt`` file are skipped. When
    ``feature_set="A+C+E+P+R"``, subjects not present in ``pathology`` are
    also skipped (so the caller must load pathology first).
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set={feature_set!r}. Supported: {list(FEATURE_SETS)}"
        )

    subject_ids = list(subject_ids)
    need_pathology = feature_set == "A+C+E+P+R"
    out: dict[str, np.ndarray] = {}
    missing_pt: list[str] = []
    missing_pathology: list[str] = []

    for sid in subject_ids:
        pt_path = precomputed_dir / f"{sid}.pt"
        if not pt_path.exists():
            missing_pt.append(sid)
            continue

        pathology_vec: np.ndarray | None = None
        if need_pathology:
            if pathology is None or sid not in pathology:
                missing_pathology.append(sid)
                continue
            pathology_vec = pathology[sid]

        pt = torch.load(pt_path, weights_only=False)
        out[sid] = build_features(pt, feature_set, pathology_vec)

    logger.info(
        "load_enriched_features(%s): %d/%d subjects loaded "
        "(missing_pt=%d, missing_pathology=%d, dim=%d)",
        feature_set, len(out), len(subject_ids),
        len(missing_pt), len(missing_pathology),
        FEATURE_SET_SIZES[feature_set],
    )
    if missing_pt:
        head = missing_pt[:10]
        logger.warning(
            "Missing .pt files (%d%s): %s",
            len(missing_pt),
            "" if len(missing_pt) <= 10 else f", first 10 of {len(missing_pt)}",
            head,
        )
    if missing_pathology:
        head = missing_pathology[:10]
        logger.warning(
            "Missing pathology values (%d%s): %s",
            len(missing_pathology),
            "" if len(missing_pathology) <= 10 else f", first 10 of {len(missing_pathology)}",
            head,
        )
    return out
