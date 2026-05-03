"""GradientSHAP + SmoothGrad attribution-method robustness companion to
``captum_composite_attribution.py``.

For each fold, loads the same max-R² ``best-*.ckpt`` and reuses the
``_ResDecCompositeWrapper`` (encoder + ResDec-MHE head as pseudobulk → scalar
``f̂_1``). Runs **two** Captum methods on the val pseudobulk:

  * **GradientSHAP** (``captum.attr.GradientShap``) — n_samples gaussian-noise
    samples, baselines = stack of [zeros, 5 random gaussian baselines drawn from
    a stdev-matched per-fold distribution]. SmoothGrad-flavored sampling around
    the input combined with a baseline-distribution expectation.
  * **SmoothGrad** (``captum.attr.NoiseTunnel(IntegratedGradients)`` with
    ``nt_type="smoothgrad"``, ``nt_samples=10``, ``stdevs=0.1``) — averages IG
    over noisy copies of the input. Tests whether the IG ranking is stable to
    input-space noise.

Goal: confirm that the Splatter-dominant + LAMP5-LHX6 / Chandelier secondary
findings from IG are **robust to attribution-method choice**.

Output (default ``outputs/canonical/interpretability/captum_robustness/``):
  - gradientshap_attributions.npz       — keys: subject_ids [N], attributions
                                          [N, C, G] (float32),
                                          predictions_residual [N], fold [N]
  - smoothgrad_attributions.npz         — same schema
  - attribution_methods_comparison.json — per-method top-K lists +
                                          Spearman ρ matrix between IG /
                                          GradientSHAP / SmoothGrad rankings
                                          (over the C × G mean-|attribution|
                                          vector). Includes a provenance block
                                          (git_sha, args, captum version).

Usage
-----
    PYTHONPATH=<worktree-root> \\
    CUDA_VISIBLE_DEVICES=0 \\
    uv run python scripts/resdec_mhe/interpretability/gradient_shap_smoothgrad_attribution.py \\
        --pred-root outputs/canonical/p5_canonical_seed42 \\
        --out-dir outputs/canonical/interpretability/captum_robustness \\
        --n-steps 50 --internal-batch-size 4

Smoke-test mode (single fold, tiny budget)::

    uv run python scripts/resdec_mhe/interpretability/gradient_shap_smoothgrad_attribution.py \\
        --smoke --gs-n-samples 2 --sg-n-samples 2 --n-steps 8

The smoke run aborts (sys.exit 2) if wall time exceeds ``--smoke-max-min``
(default 10).

Arguments mirror ``captum_composite_attribution.py`` plus a few method-specific
knobs (``--gs-n-samples``, ``--gs-n-baselines``, ``--gs-stdevs``,
``--sg-n-samples``, ``--sg-stdevs``, ``--ig-vs-path``).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import GradientShap, IntegratedGradients, NoiseTunnel
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from tqdm import tqdm

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))

import captum  # noqa: E402  - for provenance version stamp
from src.data.constants import CELL_TYPE_ORDER  # noqa: E402
from src.data.datamodule import CognitiveResilienceDataModule  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.training.resdec_lightning_module import ResDecLightningModule  # noqa: E402
from src.utils.cell_types import pad_cell_type_names  # noqa: E402
from src.utils.provenance import git_sha, pick_max_r2_ckpt  # noqa: E402

# Reuse the wrapper + summary helpers from the canonical IG orchestrator —
# same input contract (pseudobulk [B, C, G] → residual scalar) so the only
# axis that varies is the attribution method itself.
from scripts.resdec_mhe.interpretability.captum_composite_attribution import (  # noqa: E402
    _ResDecCompositeWrapper,
    _load_gene_names,
    summarize,
)

logger = logging.getLogger(__name__)


class _BatchTilingWrapper(torch.nn.Module):
    """Wraps ``_ResDecCompositeWrapper`` and auto-tiles fixed inputs.

    Captum's GradientSHAP / NoiseTunnel internally replicate the input along
    the batch dimension (n_samples copies). The encoder strictly validates
    that ``cell_type_mask``, ``cell_offsets``, ``pathology`` etc. all share
    the **same** batch size as ``pseudobulk``. This wrapper detects the size
    mismatch in ``forward()`` and tiles the cached fixed inputs along dim 0
    to match.

    For the ragged ``cell_data`` / ``cell_offsets`` pair: ``cell_offsets``
    is replicated K times (each copy still points into the same flat
    ``cell_data``), giving K identical cell-context replicas. This is the
    correct semantics for IG/GS/SG since only ``pseudobulk`` is being noised.
    """

    def __init__(self, base: "_ResDecCompositeWrapper"):
        super().__init__()
        self.base = base
        self._original_fixed: dict = {}

    def set_fixed_inputs(self, batch: dict) -> None:
        # Capture both sides: forward to the inner wrapper, and keep our own
        # copy for tiling logic.
        self.base.set_fixed_inputs(batch)
        self._original_fixed = dict(self.base._fixed_kwargs)
        self._B0 = int(batch["pseudobulk"].shape[0])

    def _tile_fixed(self, tile_factor: int) -> dict:
        """Tile each cached fixed kwarg along dim 0 by ``tile_factor``.

        Strategy
        --------
        Goal: lay out tiled ``cell_data`` and ``cell_offsets`` so that the
        cell_transformer's linear traversal ``arange(cell_data.shape[0])``
        consistently lands in the right (sample, replica, type) group when
        the batch dim is expanded by Captum to ``B*K``.

        The cell_transformer expects:
          * ``cell_data``: flat ``[total_cells_internal, n_genes]``.
          * ``cell_offsets``: ``[B*K, n_types+1]`` with absolute pointers
            into ``cell_data``.
          * ``counts_flat.sum() == total_cells_internal``.

        Our layout (per-sample K-tile so that ``arange`` traversal is
        consistent):
          * ``cell_data_tiled``: for each sample i, repeat its slice
            ``cell_data[orig_offsets[i, 0] : orig_offsets[i, n_types]]`` K
            times consecutively. Sample 0's K replicas occupy
            ``[0, K * count_0)``; sample 1's K replicas occupy
            ``[K * count_0, K * count_0 + K * count_1)``; etc.
          * ``cell_offsets_tiled``: for row ``i*K + k``, type t,
            ``offset = sample_starts_tiled[i] + k * count_i + within_sample[i, t]``,
            where ``within_sample[i, t] = orig_offsets[i, t] - orig_offsets[i, 0]``
            and ``sample_starts_tiled[i] = K * sum_{j<i} count_j``.

        Other fixed kwargs (``cell_type_mask``, ``pathology``) tile via
        standard ``repeat_interleave`` along dim 0. ``ccc_*`` graph edges
        are batch-invariant and pass through unchanged.
        """
        if tile_factor <= 1:
            return dict(self._original_fixed)

        tiled: dict = {}
        cell_offsets = self._original_fixed.get("cell_offsets")
        cell_data = self._original_fixed.get("cell_data")

        if cell_offsets is None or cell_data is None:
            cd_tiled = None
            co_tiled = None
        else:
            B0 = int(cell_offsets.shape[0])
            K = tile_factor
            device = cell_offsets.device

            # Per-sample counts: [B0]. orig_offsets[i, n_types] - orig_offsets[i, 0]
            sample_counts = (cell_offsets[:, -1] - cell_offsets[:, 0]).to(
                device=device, dtype=cell_offsets.dtype,
            )
            # Cumulative starts after tiling: [B0]. K * cumsum_excl(sample_counts).
            cum = torch.cumsum(sample_counts, dim=0)  # [B0]
            sample_starts_tiled = torch.cat(
                [torch.zeros(1, dtype=cum.dtype, device=device), K * cum[:-1]],
            )  # [B0]

            # Per-sample slice + K-repeat to build tiled flat cell_data.
            # Done sample-by-sample to keep memory tight and the layout
            # exactly matching cell_offsets_tiled.
            sample_blocks = []
            for i in range(B0):
                s = int(cell_offsets[i, 0].item())
                e = int(cell_offsets[i, -1].item())
                if e > s:
                    block = cell_data[s:e]  # [count_i, F]
                    sample_blocks.append(block.repeat(K, *([1] * (block.dim() - 1))))
                # else: empty sample — skipped (count = 0 contributes nothing).
            cd_tiled = torch.cat(sample_blocks, dim=0) if sample_blocks else cell_data

            # Tiled cell_offsets:
            i_idx = torch.arange(B0, device=device).repeat_interleave(K)  # [B0*K]
            k_idx = torch.arange(K, device=device).repeat(B0)  # [B0*K]
            # within_sample[i, t] = orig_offsets[i, t] - orig_offsets[i, 0]
            within_sample = cell_offsets - cell_offsets[:, :1]  # [B0, n_types+1]
            within_sample_tiled = within_sample[i_idx]  # [B0*K, n_types+1]
            shift = (
                sample_starts_tiled[i_idx]
                + k_idx.to(sample_counts.dtype) * sample_counts[i_idx]
            )  # [B0*K]
            co_tiled = within_sample_tiled + shift.unsqueeze(1)

        for k, v in self._original_fixed.items():
            if v is None or not torch.is_tensor(v):
                tiled[k] = v
                continue
            if k.startswith("ccc_"):
                # Graph-level edge tensors — not batch-indexed.
                tiled[k] = v
                continue
            if k == "cell_data":
                tiled[k] = cd_tiled if cd_tiled is not None else v
                continue
            if k == "cell_offsets":
                tiled[k] = co_tiled if co_tiled is not None else v
                continue
            # Default: tile dim-0 by tile_factor.
            tiled[k] = v.repeat_interleave(tile_factor, dim=0)
        return tiled

    def forward(self, pseudobulk: torch.Tensor) -> torch.Tensor:
        B_in = int(pseudobulk.shape[0])
        if B_in == self._B0:
            return self.base(pseudobulk)
        if B_in % self._B0 != 0:
            raise ValueError(
                f"BatchTilingWrapper: pseudobulk batch {B_in} is not a "
                f"multiple of cached fixed batch {self._B0}; cannot tile."
            )
        tile_factor = B_in // self._B0
        # Temporarily swap in tiled fixed kwargs.
        saved = self.base._fixed_kwargs
        self.base._fixed_kwargs = self._tile_fixed(tile_factor)
        try:
            out = self.base(pseudobulk)
        finally:
            self.base._fixed_kwargs = saved
        return out


def _make_gradient_shap_baselines(
    pseudobulk: torch.Tensor, n_baselines: int, stdev_scale: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Stack of [zeros] + ``n_baselines`` gaussian baselines.

    GradientSHAP samples baselines from this stack. We anchor with one true
    zero baseline (matching the IG reference point) and add ``n_baselines``
    random gaussians scaled to the input's stdev.
    """
    zeros = torch.zeros_like(pseudobulk[:1])  # [1, C, G]
    if n_baselines <= 0:
        return zeros
    base_std = pseudobulk.std().item() * stdev_scale
    if base_std == 0.0:
        base_std = 1e-3
    rand = torch.randn(
        (n_baselines,) + tuple(pseudobulk.shape[1:]),
        generator=rng, device=pseudobulk.device, dtype=pseudobulk.dtype,
    ) * base_std
    return torch.cat([zeros, rand], dim=0)  # [1 + n_baselines, C, G]


def attribute_one_fold(
    args: argparse.Namespace, fold: int, device: torch.device,
) -> dict:
    """Run GradientSHAP + SmoothGrad on the val loader of ``fold``.

    Returns
    -------
    dict with keys ``subject_ids``, ``gradientshap``, ``smoothgrad``,
    ``predictions_residual``, ``fold``.
    """
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(args.config),
    )
    OmegaConf.set_struct(cfg, False)
    cfg.model.head.type = "deterministic"
    cfg.data.fold = int(fold)

    fold_dir = Path(args.pred_root) / f"fold{fold}"
    ckpt_path = pick_max_r2_ckpt(fold_dir / "checkpoints")
    logger.info("fold %d: loading %s", fold, ckpt_path.name)

    splits = load_splits(str(args.splits_path))
    metadata_csv = pd.read_csv(Path(cfg.data.metadata_path) / "metadata.csv")
    dm = CognitiveResilienceDataModule(
        config=cfg, metadata=metadata_csv, splits=splits,
        fold_idx=fold,
        precomputed_dir=cfg.data.precomputed_dir,
        adata=None,
    )
    dm.setup(stage="fit")

    model = ResDecLightningModule.load_from_checkpoint(
        str(ckpt_path), config=cfg, map_location="cpu",
    ).to(device).eval()
    # Captum requires fp32 for stable gradients (bf16 mantissa too short).
    model = model.float()

    base_wrapper = _ResDecCompositeWrapper(model).to(device)
    wrapper = _BatchTilingWrapper(base_wrapper).to(device)
    gshap = GradientShap(wrapper)
    smoothgrad = NoiseTunnel(IntegratedGradients(wrapper))

    # Per-fold deterministic generator for the gradient-SHAP baseline draws.
    rng = torch.Generator(device=device).manual_seed(int(args.baseline_seed) + fold)

    sids_all: list[str] = []
    gs_all: list[np.ndarray] = []
    sg_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []

    val_loader = dm.val_dataloader()
    for batch in tqdm(val_loader, desc=f"fold {fold} GS+SG", unit="batch"):
        sids = list(batch["subject_ids"])
        batch_d = {
            k: (v.to(device).float() if (torch.is_tensor(v) and v.is_floating_point())
                else v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        wrapper.set_fixed_inputs(batch_d)
        pseudobulk = batch_d["pseudobulk"]  # [B, C, G]

        baselines = _make_gradient_shap_baselines(
            pseudobulk,
            n_baselines=int(args.gs_n_baselines),
            stdev_scale=float(args.gs_stdevs),
            rng=rng,
        )
        gs_attr = gshap.attribute(
            pseudobulk,
            baselines=baselines,
            n_samples=int(args.gs_n_samples),
            stdevs=float(args.gs_stdevs),
        )

        sg_attr = smoothgrad.attribute(
            pseudobulk,
            nt_type="smoothgrad",
            nt_samples=int(args.sg_n_samples),
            nt_samples_batch_size=int(args.sg_nt_batch_size) or None,
            stdevs=float(args.sg_stdevs),
            # IG kwargs forwarded via **kwargs:
            baselines=torch.zeros_like(pseudobulk),
            n_steps=int(args.n_steps),
            internal_batch_size=int(args.internal_batch_size),
        )

        with torch.no_grad():
            pred = wrapper(pseudobulk)

        sids_all.extend(sids)
        gs_all.append(gs_attr.detach().cpu().numpy())
        sg_all.append(sg_attr.detach().cpu().numpy())
        preds_all.append(pred.detach().cpu().numpy())

    return {
        "subject_ids": np.array(sids_all, dtype=object),
        "gradientshap": np.concatenate(gs_all, axis=0).astype(np.float32),
        "smoothgrad": np.concatenate(sg_all, axis=0).astype(np.float32),
        "predictions_residual": np.concatenate(preds_all, axis=0).astype(np.float32),
        "fold": np.full(len(sids_all), fold, dtype=np.int32),
    }


def _spearman_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ between two flattened mean-|attribution| vectors."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    rho, _ = spearmanr(a.flatten(), b.flatten())
    return float(rho)


def _build_comparison(
    method_arrays: dict[str, np.ndarray],
    ct_names: list[str],
    gene_names: list[str],
    top_k: int,
) -> dict:
    """Per-method top-K + cross-method Spearman ρ matrix.

    ``method_arrays``: {method_name: [N, C, G] attribution array}.
    Comparison metric: mean |attribution| over the N axis → [C, G] flattened
    to a length-CG ranking vector. Spearman over those.
    """
    summaries: dict[str, dict] = {}
    flat_means: dict[str, np.ndarray] = {}
    for name, arr in method_arrays.items():
        per_ct = np.abs(arr).mean(axis=0)  # [C, G]
        flat_means[name] = per_ct.flatten()
        # Top-K (ct, gene) pairs. kind="stable" so ties break by original
        # index (matches captum_composite_attribution.summarize convention).
        flat_idx = np.argsort(-per_ct.flatten(), kind="stable")[:top_k]
        top_pairs = []
        for k in flat_idx:
            c, g = divmod(int(k), per_ct.shape[1])
            top_pairs.append({
                "cell_type": ct_names[c] if c < len(ct_names) else f"ct_{c}",
                "gene": gene_names[g] if g < len(gene_names) else f"gene_{g}",
                "mean_abs_attribution": float(per_ct[c, g]),
            })
        summaries[name] = {
            "n_subjects": int(arr.shape[0]),
            "n_cell_types": int(arr.shape[1]),
            "n_genes": int(arr.shape[2]),
            "top_pairs": top_pairs,
        }

    # Pairwise Spearman over flattened C × G means.
    # Upper-triangular only (and diagonal) — full matrix is redundant since
    # ρ(a, b) == ρ(b, a). Consumer can mirror if symmetric access is needed.
    method_names = list(method_arrays.keys())
    rho_matrix: dict[str, dict[str, float]] = {m: {} for m in method_names}
    for i, mi in enumerate(method_names):
        rho_matrix[mi][mi] = 1.0
        for j in range(i + 1, len(method_names)):
            mj = method_names[j]
            rho_matrix[mi][mj] = _spearman_rank_corr(
                flat_means[mi], flat_means[mj],
            )
    return {
        "per_method_top_pairs": summaries,
        # Upper triangle (i<=j) only — mirror at the consumer if needed.
        "spearman_rho_over_celltype_x_gene_means": rho_matrix,
    }


def _load_ig_attribution(ig_npz_path: Path) -> np.ndarray | None:
    """Load IG attributions for cross-method Spearman comparison.

    Returns ``None`` if the file is missing — comparison just skips IG.
    """
    if not ig_npz_path.exists():
        logger.warning("IG comparison file not found: %s — skipping IG axis.",
                       ig_npz_path)
        return None
    d = np.load(ig_npz_path, allow_pickle=True)
    return d["attributions"]


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Smoke mode: one fold, abort if wall too long.
    folds = [0] if args.smoke else list(range(5))
    logger.info("Running folds: %s (smoke=%s)", folds, args.smoke)

    t0 = time.time()
    sids_per: list[np.ndarray] = []
    gs_per: list[np.ndarray] = []
    sg_per: list[np.ndarray] = []
    preds_per: list[np.ndarray] = []
    folds_per: list[np.ndarray] = []
    for f in folds:
        d = attribute_one_fold(args, f, device)
        sids_per.append(d["subject_ids"])
        gs_per.append(d["gradientshap"])
        sg_per.append(d["smoothgrad"])
        preds_per.append(d["predictions_residual"])
        folds_per.append(d["fold"])
        elapsed_min = (time.time() - t0) / 60.0
        logger.info("fold %d done — %s GS / %s SG / elapsed %.2f min",
                    f, d["gradientshap"].shape, d["smoothgrad"].shape, elapsed_min)
        if args.smoke and elapsed_min > float(args.smoke_max_min):
            logger.error(
                "SMOKE ABORT: elapsed %.1f min > --smoke-max-min %.1f. "
                "Likely cause: too-large n_samples or n_steps for a smoke run; "
                "try --gs-n-samples 2 --sg-n-samples 2 --n-steps 8.",
                elapsed_min, float(args.smoke_max_min),
            )
            return 2

    sids = np.concatenate(sids_per)
    gs = np.concatenate(gs_per, axis=0)
    sg = np.concatenate(sg_per, axis=0)
    preds = np.concatenate(preds_per)
    fold_arr = np.concatenate(folds_per)

    n_subj, n_ct, n_genes = gs.shape
    logger.info(
        "Total: %d subjects × %d cell types × %d genes  "
        "(%.1f MB GS, %.1f MB SG)",
        n_subj, n_ct, n_genes, gs.nbytes / 1e6, sg.nbytes / 1e6,
    )

    gs_npz = out_dir / "gradientshap_attributions.npz"
    sg_npz = out_dir / "smoothgrad_attributions.npz"
    np.savez(gs_npz, subject_ids=sids, attributions=gs,
             predictions_residual=preds, fold=fold_arr)
    np.savez(sg_npz, subject_ids=sids, attributions=sg,
             predictions_residual=preds, fold=fold_arr)
    logger.info("Wrote %s", gs_npz)
    logger.info("Wrote %s", sg_npz)

    # Cell-type + gene names for summaries.
    ct_names = pad_cell_type_names(CELL_TYPE_ORDER, n_ct)

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(args.config),
    )
    gene_names, used_real_gene_names = _load_gene_names(
        Path(cfg.data.precomputed_dir), n_genes
    )

    # Per-method summaries reuse the same logic as captum_composite_attribution.
    gs_summary = summarize(gs_npz, ct_names, gene_names, top_k=int(args.top_k))
    sg_summary = summarize(sg_npz, ct_names, gene_names, top_k=int(args.top_k))
    # Stamp gene-name provenance so consumers can detect placeholder use.
    gs_summary["_no_gene_names"] = not used_real_gene_names
    sg_summary["_no_gene_names"] = not used_real_gene_names
    (out_dir / "gradientshap_attribution_summary.json").write_text(
        json.dumps(gs_summary, indent=2),
    )
    (out_dir / "smoothgrad_attribution_summary.json").write_text(
        json.dumps(sg_summary, indent=2),
    )

    # Cross-method comparison — load IG if available so we get a 3-method
    # rho matrix instead of just GradientSHAP × SmoothGrad.
    method_arrays: dict[str, np.ndarray] = {
        "gradientshap": gs,
        "smoothgrad": sg,
    }
    ig_loaded = False
    ig_skip_reason: str | None = None
    if args.ig_npz_path:
        ig_arr = _load_ig_attribution(Path(args.ig_npz_path))
        if ig_arr is None:
            ig_skip_reason = f"file_missing: {args.ig_npz_path}"
        else:
            # Spearman is computed on the [C, G] mean-|attr| vector — IG can
            # have a different N as long as it covers the same (C, G).
            if ig_arr.shape[1:] == gs.shape[1:]:
                method_arrays["integrated_gradients"] = ig_arr
                ig_loaded = True
            else:
                ig_skip_reason = (
                    f"shape_mismatch: IG (C,G)={ig_arr.shape[1:]} vs "
                    f"GS (C,G)={gs.shape[1:]}"
                )
                logger.warning(
                    "IG (C, G)=%s != GS (C, G)=%s — skipping IG axis.",
                    ig_arr.shape[1:], gs.shape[1:],
                )
    else:
        ig_skip_reason = "no_ig_npz_path_passed"

    comparison = _build_comparison(method_arrays, ct_names, gene_names,
                                   top_k=int(args.top_k))
    # Stamp ig_loaded explicitly so a consumer notices when the comparison
    # is 2-way (GS × SG only) rather than the canonical 3-way (with IG).
    comparison["ig_loaded"] = bool(ig_loaded)
    comparison["ig_skip_reason"] = ig_skip_reason

    def _provenance_arg_serializer(v: object) -> object:
        """Best-effort JSON-safe serializer for argparse Namespace values.

        Path → str; primitives pass through; other types fall back to
        ``str(v)`` so the JSON write always succeeds.
        """
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (str, int, float, bool, type(None))):
            return v
        return str(v)

    comparison["provenance"] = {
        "git_sha": git_sha(_WORKTREE_ROOT),
        "captum_version": captum.__version__,
        "args": {k: _provenance_arg_serializer(v) for k, v in vars(args).items()},
        "folds_run": folds,
        "wall_time_minutes": round((time.time() - t0) / 60.0, 3),
    }
    cmp_path = out_dir / "attribution_methods_comparison.json"
    cmp_path.write_text(json.dumps(comparison, indent=2))
    logger.info("Wrote %s", cmp_path)

    print()
    print("=== GradientSHAP top-5 (cell_type, gene) ===")
    for i, p in enumerate(gs_summary["top_cell_type_gene_pairs"][:5], 1):
        print(f"  {i}. {p['cell_type']:<40s}  {p['gene']:<20s}  "
              f"{p['mean_abs_attribution']:.6f}")
    print()
    print("=== SmoothGrad top-5 (cell_type, gene) ===")
    for i, p in enumerate(sg_summary["top_cell_type_gene_pairs"][:5], 1):
        print(f"  {i}. {p['cell_type']:<40s}  {p['gene']:<20s}  "
              f"{p['mean_abs_attribution']:.6f}")
    print()
    print("=== Spearman ρ (over [C, G] mean-|attr| vector) ===")
    # rho_matrix is now upper-triangular (i<=j); skip mi == mj (always 1.0).
    for mi, row in comparison["spearman_rho_over_celltype_x_gene_means"].items():
        for mj, rho in row.items():
            if mi != mj:
                print(f"  {mi:<22s} vs {mj:<22s}  ρ={rho:.4f}")

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Captum GradientSHAP + SmoothGrad robustness companion.",
    )
    # Same canonical flags as captum_composite_attribution.py:
    p.add_argument("--config", default="configs/resdec_mhe/canonical.yaml")
    p.add_argument("--pred-root", default="outputs/canonical/p5_canonical_seed42")
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--out-dir",
                   default="outputs/canonical/interpretability/captum_robustness")
    p.add_argument("--n-steps", type=int, default=50,
                   help="IG interpolation steps (used by SmoothGrad's wrapped IG).")
    p.add_argument("--internal-batch-size", type=int, default=4)
    p.add_argument("--top-k", type=int, default=50)

    # GradientSHAP knobs:
    p.add_argument("--gs-n-samples", type=int, default=20,
                   help="GradientSHAP n_samples (gaussian draws around input).")
    p.add_argument("--gs-n-baselines", type=int, default=5,
                   help="Number of random gaussian baselines (in addition to zero).")
    p.add_argument("--gs-stdevs", type=float, default=0.1,
                   help="GradientSHAP stdev scale (relative to input stdev).")

    # SmoothGrad (NoiseTunnel + IG) knobs:
    p.add_argument("--sg-n-samples", type=int, default=10,
                   help="SmoothGrad nt_samples (noisy input copies).")
    p.add_argument("--sg-stdevs", type=float, default=0.1,
                   help="SmoothGrad input-noise stdev.")
    p.add_argument("--sg-nt-batch-size", type=int, default=0,
                   help="NoiseTunnel batch size for noisy samples (0 = None).")

    # Cross-method comparison:
    p.add_argument(
        "--ig-npz-path",
        default="outputs/canonical/interpretability/captum_ig/composite_attributions.npz",
        help=("Path to existing IG composite_attributions.npz for 3-way "
              "Spearman comparison. Pass empty string to skip."),
    )
    p.add_argument("--baseline-seed", type=int, default=42,
                   help="Seed for GradientSHAP baseline gaussian draws.")

    # Smoke-test mode:
    p.add_argument("--smoke", action="store_true",
                   help="Run only fold 0; abort if wall > --smoke-max-min.")
    p.add_argument("--smoke-max-min", type=float, default=10.0,
                   help="Smoke-test wall-time cap (minutes).")
    sys.exit(main(p.parse_args()))
