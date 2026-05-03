"""Gene-name list loading utilities.

Single source of truth for the "load gene names from precomputed feature
metadata, fall back to ``gene_<i>`` placeholders" pattern shared by
``captum_composite_attribution.py`` and ``gradient_shap_smoothgrad_attribution.py``
(File 5 deferred fix in
``docs/code_reviews/2026-05-02_full_review_C_FIXES.md``).

The historical signature returned only the names list, so a downstream
consumer could not tell whether the names were placeholders or the real
gene symbols. The new signature returns a 2-tuple
``(names, used_real_names)`` so the consumer can stamp a
``_no_gene_names`` flag in the summary JSON when placeholders were used.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_gene_names(
    precomputed_dir: Path,
    n_genes: int,
    fallback_paths: tuple[Path, ...] = (Path("data/canonical/gene_names.json"),),
) -> tuple[list[str], bool]:
    """Try to load ``n_genes`` real gene symbols; fall back to placeholders.

    Parameters
    ----------
    precomputed_dir
        Directory containing the canonical ``gene_names.{npy,json}`` or
        ``feature_names.json`` sidecar (written by ``precompute_features``
        in ``src/data/datasets.py``).
    n_genes
        The model's actual gene-axis length. Names are truncated to this.
    fallback_paths
        Additional absolute / project-relative paths to probe before
        giving up. Defaults to the canonical project gene-names file.

    Returns
    -------
    (names, used_real_names)
        ``names``: length-``n_genes`` list of strings.
        ``used_real_names``: ``True`` if a sidecar was found and the
        first ``n_genes`` real symbols are returned. ``False`` if the
        list contains ``gene_<i>`` placeholders.
    """
    candidates = [
        precomputed_dir / "gene_names.npy",
        precomputed_dir / "gene_names.json",
        precomputed_dir / "feature_names.json",
        *fallback_paths,
    ]
    for p in candidates:
        if not p.exists():
            continue
        if p.suffix == ".npy":
            names = np.load(p, allow_pickle=True).tolist()
        else:
            names = json.loads(p.read_text())
        if isinstance(names, list) and len(names) >= n_genes:
            logger.info("Loaded %d gene names from %s", n_genes, p)
            return [str(n) for n in names[:n_genes]], True
    logger.warning(
        "No gene-name file found in expected locations (%s). "
        "Using gene_<i> placeholders.",
        ", ".join(str(p) for p in candidates),
    )
    return [f"gene_{i}" for i in range(n_genes)], False
