"""Captum attribution for TabPFN-2.6 predictions.

Given the cached TabPFN setup (top-2K features per outer fold), fits TabPFN on
outer-train, computes per-subject feature attributions on val via Captum's
FeatureAblation, and re-hydrates the 2000 flat feature indices back to
`(cell_type, gene)` pairs for biological interpretation.

Method choice
-------------
We use **Captum's FeatureAblation** rather than Integrated Gradients.

Rationale (verified by inspecting ``tabpfn/regressor.py`` and
``tabpfn/inference.py`` at v7.1.1):

- ``TabPFNRegressor.predict()`` uses ``InferenceEngineCachePreprocessing``
  which runs the forward pass under ``torch.inference_mode(True)`` (see
  ``inference.py::_call_model`` line ~774), disabling autograd on outputs.
  Output is coerced to ``np.ndarray`` in ``regressor.py::_logits_to_output``.
  This means the default ``reg.predict(X)`` path is non-differentiable.
- The only gradient-capable path is ``forward(use_inference_mode=False)``
  with ``InferenceEngineBatchedNoPreprocessing``, which is the fine-tuning
  path. Driving Captum IG through that path would require constructing
  preprocessed torch-tensor batches via the fine-tuning dataset APIs and
  bypassing the sklearn preprocessing (ordinal encoding, z-scoring, NaN
  imputation), which risks diverging from the actual predict() output
  distribution we care about attributing.
- FeatureAblation treats ``reg.predict`` as a black box: for each feature,
  it replaces the feature's value with a baseline (cohort mean) and
  measures the change in prediction. This matches exactly what TabPFN
  "sees" at predict time — no divergence from the actual baseline model.

FeatureAblation cost scales as O(n_features + 1) predict calls per attribution
call. For 2000 features and ~10 val subjects batched together, that is 2001
forward passes over the whole val batch.

Used for complementarity analysis against head-residual attributions in the
interpretability pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import FeatureAblation
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from src.data.feature_loaders import load_flat_features, load_targets
from src.data.splits import load_splits

logger = logging.getLogger(__name__)

# Pseudobulk shape is [N_CT, N_GENES] = [31, 4785] flattened.
# Flat feature index i = ct * N_GENES + gene.
N_GENES: int = 4785
N_CT: int = 31


def hydrate_feature_indices(top_k_indices: list[int] | np.ndarray) -> pd.DataFrame:
    """Map flat pseudobulk feature indices back to ``(cell_type_id, gene_id)``.

    The pseudobulk tensor has shape ``[N_CT, N_GENES]`` (row-major). When
    flattened, index ``i`` corresponds to ``ct = i // N_GENES`` and
    ``gene = i % N_GENES``.

    Args:
        top_k_indices: 1-D iterable of flat feature indices in ``[0, N_CT * N_GENES)``.

    Returns:
        DataFrame with columns ``feature_idx``, ``ct_id``, ``gene_id`` (one
        row per input index, same order as input).
    """
    idx_arr = np.asarray(list(top_k_indices), dtype=np.int64)
    return pd.DataFrame(
        {
            "feature_idx": idx_arr,
            "ct_id": idx_arr // N_GENES,
            "gene_id": idx_arr % N_GENES,
        }
    )


def _fit_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: str = "cuda",
    seed: int = 42,
) -> TabPFNRegressor:
    """Fit TabPFN-2.6 on outer-train set for a given fold.

    Uses the model-version-specific factory so defaults (model_path,
    n_estimators, softmax_temperature) match the V2.6 preset — matches
    ``scripts/resdec_mhe/tabpfn/compute_outer.py`` which intentionally
    targets V2.6 for the paper's standalone baseline.
    """
    reg = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2_6,
        device=device,
        random_state=seed,
    )
    reg.fit(X_train, y_train)
    return reg


def _make_predict_fn(reg: TabPFNRegressor):
    """Return a torch-tensor-in / torch-tensor-out wrapper around ``reg.predict``.

    Captum's FeatureAblation accepts black-box forward functions that map
    ``(B, F) -> (B,)``. TabPFN's ``predict()`` takes numpy and returns numpy,
    so we convert at the boundary. Because FeatureAblation never needs
    gradients through this forward (it perturbs inputs directly), the
    ``torch.inference_mode`` inside ``predict()`` is not a problem.
    """

    def predict_fn(X: torch.Tensor) -> torch.Tensor:
        X_np = X.detach().cpu().numpy().astype(np.float32)
        # "median" is the calibrated point prediction used in the baseline
        # (see scripts/resdec_mhe/tabpfn/compute_outer.py::_predict_with_sigma).
        y_np = np.asarray(reg.predict(X_np, output_type="median"))
        return torch.as_tensor(y_np, dtype=torch.float32, device=X.device)

    return predict_fn


def _compute_attributions(
    reg: TabPFNRegressor,
    X_val: np.ndarray,
    *,
    baselines: np.ndarray,
    device: str,
) -> np.ndarray:
    """Compute per-subject per-feature attributions via FeatureAblation.

    Args:
        reg: Fitted TabPFN regressor.
        X_val: ``[n_val, n_features]`` validation features (top-2K).
        baselines: ``[n_features]`` baseline vector (cohort mean of train).
        device: torch device string (e.g. ``"cuda:1"``).

    Returns:
        ``np.ndarray`` shape ``[n_val, n_features]``; ``attr[i, j]`` is the
        change in predicted median when feature j is replaced by
        ``baselines[j]`` (positive = feature pushes prediction up).
    """
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    baselines_t = torch.as_tensor(baselines, dtype=torch.float32, device=device)
    # FeatureAblation expects baselines broadcastable to the input shape.
    baselines_bcast = baselines_t.unsqueeze(0).expand_as(X_val_t).contiguous()

    predict_fn = _make_predict_fn(reg)
    ablator = FeatureAblation(predict_fn)
    attributions = ablator.attribute(
        inputs=X_val_t,
        baselines=baselines_bcast,
        perturbations_per_eval=1,  # one feature at a time; batch across subjects
        show_progress=False,
    )
    return attributions.detach().cpu().numpy().astype(np.float32)


def attribute_tabpfn_fold(
    fold_idx: int,
    precomputed_dir: Path,
    meta_csv: Path,
    splits_path: Path,
    top_k_dir: Path,
    n_val_subjects: int | None = None,
    device: str = "cuda",
    method: str = "feature_ablation",
    seed: int = 42,
    tabpfn_model_cache_dir: Path | str | None = None,
) -> dict:
    """Run TabPFN attribution for a single fold.

    The fold's TabPFN is fit on all outer-train subjects (top-2K features),
    then per-subject FeatureAblation attributions are computed on the first
    ``n_val_subjects`` val subjects (or all val subjects when ``None``).

    Args:
        fold_idx: Outer CV fold index (0..4).
        precomputed_dir: Dir containing ``<subject>.pt`` pseudobulk files.
        meta_csv: Path to ROSMAP metadata CSV (cognition target).
        splits_path: Path to splits JSON.
        top_k_dir: Dir containing ``top_2000_features_fold{k}.json`` files.
        n_val_subjects: If set, attribute only the first N val subjects
            (smoke-run). If ``None``, attribute all val subjects.
        device: Torch device string (TabPFN requires a cuda device for
            reasonable throughput).
        method: Attribution method. Only ``"feature_ablation"`` is currently
            implemented — see module docstring for rationale.
        seed: TabPFN ``random_state``.
        tabpfn_model_cache_dir: Override for the TabPFN model cache. If
            ``None``, the existing ``TABPFN_MODEL_CACHE_DIR`` env var is
            used; if neither is set, an explicit error is raised. No
            host-specific default is hardcoded — set the env var or pass
            this kwarg explicitly.

    Returns:
        Dict with keys:
          - ``attributions``: ``np.ndarray`` ``[n_val, 2000]``.
          - ``val_subject_ids``: list of val subject IDs attributed.
          - ``feature_schema``: DataFrame mapping top-2K indices to (ct, gene).
          - ``mean_abs_attrib``: ``np.ndarray`` ``[2000]`` cohort mean |attr|.
          - ``top_attrib_per_subject``: dict ``{subject_id -> list of top-20}``
            where each entry is ``{feature_idx, ct_id, gene_id, score}``.
          - ``method``: the attribution method tag.
          - ``fold_idx``: fold index.
    """
    if method != "feature_ablation":
        raise NotImplementedError(
            f"Only 'feature_ablation' is implemented (got {method!r}). "
            "See module docstring for why IG is not used."
        )

    if tabpfn_model_cache_dir is not None:
        os.environ["TABPFN_MODEL_CACHE_DIR"] = str(tabpfn_model_cache_dir)
    elif "TABPFN_MODEL_CACHE_DIR" not in os.environ:
        raise RuntimeError(
            "TABPFN_MODEL_CACHE_DIR is not set. Either export it in the "
            "environment or pass tabpfn_model_cache_dir=... to "
            "attribute_tabpfn_fold(). No host-specific default is hardcoded."
        )

    splits = load_splits(splits_path)
    fold_split = splits["folds"][fold_idx]
    all_ids = fold_split["train"] + fold_split["val"]
    features = load_flat_features(Path(precomputed_dir), all_ids)
    targets = load_targets(Path(meta_csv), all_ids)

    train_ids = [s for s in fold_split["train"] if s in features and s in targets]
    val_ids_all = [s for s in fold_split["val"] if s in features and s in targets]
    val_ids = val_ids_all if n_val_subjects is None else val_ids_all[:n_val_subjects]

    top_k = json.loads(
        (Path(top_k_dir) / f"top_2000_features_fold{fold_idx}.json").read_text()
    )["indices"]

    X_train = np.stack([features[s] for s in train_ids])[:, top_k].astype(np.float32)
    y_train = np.array([targets[s] for s in train_ids], dtype=np.float32)
    X_val = np.stack([features[s] for s in val_ids])[:, top_k].astype(np.float32)

    logger.info(
        "fold %d: fitting TabPFN on %d train, attributing %d val (top-2K features)",
        fold_idx, X_train.shape[0], X_val.shape[0],
    )
    reg = _fit_tabpfn(X_train, y_train, device=device, seed=seed)

    # Baseline: cohort mean of training features in the top-2K subspace.
    # This corresponds to "replace with cohort average" — the standard
    # FeatureAblation baseline for continuous features.
    baselines = X_train.mean(axis=0)

    logger.info(
        "fold %d: running FeatureAblation (%d subjects x %d features = %d predict calls)",
        fold_idx, X_val.shape[0], X_val.shape[1], X_val.shape[1] + 1,
    )
    attributions = _compute_attributions(
        reg, X_val, baselines=baselines, device=device,
    )

    feature_schema = hydrate_feature_indices(top_k)
    mean_abs = np.abs(attributions).mean(axis=0)  # [n_features]

    top_attrib_per_subject: dict[str, list[dict]] = {}
    for i, sid in enumerate(val_ids):
        abs_scores = np.abs(attributions[i])
        sorted_top = np.argsort(-abs_scores)[:20]
        top_attrib_per_subject[sid] = [
            {
                "feature_idx": int(top_k[j]),
                "ct_id": int(top_k[j] // N_GENES),
                "gene_id": int(top_k[j] % N_GENES),
                "score": float(attributions[i, j]),
            }
            for j in sorted_top
        ]

    return {
        "attributions": attributions,
        "val_subject_ids": list(val_ids),
        "feature_schema": feature_schema,
        "mean_abs_attrib": mean_abs,
        "top_attrib_per_subject": top_attrib_per_subject,
        "method": method,
        "fold_idx": fold_idx,
    }
