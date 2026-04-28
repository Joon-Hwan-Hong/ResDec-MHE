"""Captum Integrated Gradients composite attribution for ResDec-MHE canonical model.

For each fold, loads the max-R² ``best-*.ckpt``, wraps the encoder + ResDec-MHE
head as a single-input module that takes pseudobulk ``[B, n_cell_types, n_genes]``
as the differentiable input and returns the head's residual scalar ``f̂_1`` (the
neural correction on top of TabPFN). Runs Integrated Gradients with a zero
baseline, then aggregates across all 5 folds → per-subject ``[N, n_cell_types,
n_genes]`` attribution matrix.

Why we attribute ``f̂_1`` rather than the composite ``ŷ = ŷ_tabpfn + f̂_1``:
TabPFN-2.6 predictions are pre-cached (in-context learning, not differentiable
in our PyTorch graph). IG on f̂_1 isolates **what the encoder + head learned to
contribute on top of TabPFN** — exactly the biologically interpretable signal
unique to our architecture.

Output (default ``outputs/canonical/interpretability/``):
  - composite_attributions.npz       — keys: subject_ids [N], attributions [N, C, G],
                                        predictions_residual [N], folds [N]
  - composite_attribution_summary.json
                                      — global top-K genes, per-cell-type top-K,
                                        top-K (cell_type, gene) pairs
  - top_pairs_table.csv               — top-100 (cell_type, gene) pairs by mean |attribution|

Usage
-----
    PYTHONPATH=<worktree-root> \\
    CUDA_VISIBLE_DEVICES=0 \\
    uv run python scripts/resdec_mhe/interpretability/captum_composite_attribution.py \\
        --pred-root outputs/canonical/p5_canonical_seed42 \\
        --out-dir outputs/canonical/interpretability \\
        --n-steps 50 --internal-batch-size 4

Arguments
---------
    --config <path>            Phase YAML merged on top of configs/default.yaml
                               (default: canonical configs/resdec_mhe/canonical.yaml).
    --pred-root <path>         Per-fold output dir with fold{0..4}/checkpoints/best-*.ckpt.
    --splits-path <path>       Splits JSON (default: outputs/splits.json).
    --out-dir <path>           Output directory (created if missing).
    --n-steps <int>            IG interpolation steps (default 50; lower = faster but noisier).
    --internal-batch-size <int> IG's micro-batch size (default 4; raise if VRAM permits).
    --top-k <int>              Top-K rows to surface in summary tables (default 50).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from omegaconf import OmegaConf
from tqdm import tqdm

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))

from src.data.constants import CELL_TYPE_ORDER, N_REGIONS, PFC_REGION_IDX
from src.data.datamodule import CognitiveResilienceDataModule
from src.data.splits import load_splits
from src.training.resdec_lightning_module import ResDecLightningModule

logger = logging.getLogger(__name__)
_BEST_CKPT_RE = re.compile(r"^best-(\d+)-(\d+\.\d+)\.ckpt$")


def _pick_max_r2_ckpt(ckpt_dir: Path) -> Path:
    best: tuple[Path, float] | None = None
    for p in ckpt_dir.glob("best-*.ckpt"):
        m = _BEST_CKPT_RE.match(p.name)
        if not m:
            continue
        r2 = float(m.group(2))
        if best is None or r2 > best[1]:
            best = (p, r2)
    if best is None:
        raise FileNotFoundError(f"No best-*.ckpt files in {ckpt_dir}")
    return best[0]


class _ResDecCompositeWrapper(torch.nn.Module):
    """Wraps ResDecLightningModule's encoder + head as pseudobulk → scalar.

    Builds region_pseudobulk on the fly from the pseudobulk input (placing it
    in PFC region only, matching the cohort's mostly-PFC sampling). All other
    encoder inputs (CCC edges, cell_data, pathology, cell_type_mask) are held
    fixed via ``set_fixed_inputs()`` so they're treated as constants by autograd.

    Returns f̂_1 (the head's residual scalar) — TabPFN base is added at val
    time outside the gradient graph.
    """

    def __init__(self, lit_module: ResDecLightningModule):
        super().__init__()
        self.lit_module = lit_module
        self._fixed_kwargs: dict = {}
        self._d_metadata = lit_module._d_metadata

    def set_fixed_inputs(self, batch: dict) -> None:
        """Cache non-pseudobulk batch keys for the encoder forward."""
        self._fixed_kwargs = {
            "ccc_edge_index": batch.get("ccc_edge_index"),
            "ccc_edge_type": batch.get("ccc_edge_type"),
            "ccc_edge_attr": batch.get("ccc_edge_attr"),
            "cell_type_mask": batch.get("cell_type_mask"),
            "pathology": batch.get("pathology"),
            "cell_data": batch.get("cell_data"),
            "cell_offsets": batch.get("cell_offsets"),
        }

    def forward(self, pseudobulk: torch.Tensor) -> torch.Tensor:
        """pseudobulk: [B, n_cell_types, n_genes] → scalar [B] head output."""
        B, n_ct, n_genes = pseudobulk.shape
        device, dtype = pseudobulk.device, pseudobulk.dtype

        # Place pseudobulk in PFC region; all other regions zero. region_mask
        # marks only PFC as present. This matches the cohort's mostly-PFC
        # sampling (RegionHandler's masked softmax handles the zero regions).
        region_pseudobulk = torch.zeros(B, N_REGIONS, n_ct, n_genes,
                                        device=device, dtype=dtype)
        region_pseudobulk[:, PFC_REGION_IDX, :, :] = pseudobulk
        region_mask = torch.zeros(B, N_REGIONS, dtype=torch.bool, device=device)
        region_mask[:, PFC_REGION_IDX] = True

        kwargs = dict(self._fixed_kwargs)
        kwargs["region_pseudobulk"] = region_pseudobulk
        kwargs["region_mask"] = region_mask
        # cognition not needed for forward-only; pop if present.
        kwargs.pop("cognition", None)

        enc_out = self.lit_module.encoder.forward_encoder_only(**kwargs)
        z = enc_out["attended"]  # [B, d_subject]

        # Metadata wiring is not needed for attribution — fall back to zeros
        # (FiLM is near-identity init at zero metadata, so prediction is
        # well-defined).
        metadata = torch.zeros(B, self._d_metadata, device=device, dtype=dtype)

        head_out = self.lit_module.head(z, metadata)
        return head_out["prediction"]  # [B]


def attribute_one_fold(args: argparse.Namespace, fold: int,
                       device: torch.device) -> dict:
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(args.config),
    )
    OmegaConf.set_struct(cfg, False)
    cfg.model.head.type = "deterministic"
    cfg.data.fold = int(fold)

    fold_dir = Path(args.pred_root) / f"fold{fold}"
    ckpt_path = _pick_max_r2_ckpt(fold_dir / "checkpoints")
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

    # IG requires fp32 for stable gradients (bf16 mantissa is too short).
    model = model.float()

    wrapper = _ResDecCompositeWrapper(model).to(device)
    ig = IntegratedGradients(wrapper)

    sids_all: list[str] = []
    attrs_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []

    val_loader = dm.val_dataloader()
    for batch in tqdm(val_loader, desc=f"fold {fold} IG", unit="batch"):
        sids = list(batch["subject_ids"])
        batch_d = {
            k: (v.to(device).float() if (torch.is_tensor(v) and v.is_floating_point())
                else v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        wrapper.set_fixed_inputs(batch_d)
        pseudobulk = batch_d["pseudobulk"]  # [B, n_ct, n_genes]

        baseline = torch.zeros_like(pseudobulk)
        attr = ig.attribute(
            pseudobulk,
            baselines=baseline,
            n_steps=int(args.n_steps),
            internal_batch_size=int(args.internal_batch_size),
        )
        with torch.no_grad():
            pred = wrapper(pseudobulk)

        sids_all.extend(sids)
        attrs_all.append(attr.detach().cpu().numpy())
        preds_all.append(pred.detach().cpu().numpy())

    return {
        "subject_ids": np.array(sids_all, dtype=object),
        "attributions": np.concatenate(attrs_all, axis=0).astype(np.float32),
        "predictions_residual": np.concatenate(preds_all, axis=0).astype(np.float32),
        "fold": np.full(len(sids_all), fold, dtype=np.int32),
    }


def summarize(attr_npz_path: Path, ct_names: list[str], gene_names: list[str],
              top_k: int = 50) -> dict:
    """Build top-K summaries from the saved per-subject attribution array."""
    d = np.load(attr_npz_path, allow_pickle=True)
    attr = d["attributions"]  # [N, C, G]
    abs_attr = np.abs(attr)

    global_importance = abs_attr.mean(axis=(0, 1))  # [G]
    per_ct_importance = abs_attr.mean(axis=0)       # [C, G]

    # Global top-K genes (averaged over subjects + cell types).
    top_global_idx = np.argsort(-global_importance)[:top_k]
    top_global = [
        {"gene": gene_names[i] if i < len(gene_names) else f"gene_{i}",
         "mean_abs_attribution": float(global_importance[i])}
        for i in top_global_idx
    ]

    # Per-cell-type top-K genes.
    top_per_ct: dict[str, list[dict]] = {}
    for c, ct in enumerate(ct_names[:per_ct_importance.shape[0]]):
        idx = np.argsort(-per_ct_importance[c])[:top_k]
        top_per_ct[ct] = [
            {"gene": gene_names[i] if i < len(gene_names) else f"gene_{i}",
             "mean_abs_attribution": float(per_ct_importance[c, i])}
            for i in idx
        ]

    # Top-K (cell_type, gene) pairs across the entire C × G heatmap.
    flat = per_ct_importance.flatten()
    top_pair_idx = np.argsort(-flat)[:top_k]
    top_pairs = []
    for k in top_pair_idx:
        c, g = divmod(int(k), per_ct_importance.shape[1])
        top_pairs.append({
            "cell_type": ct_names[c] if c < len(ct_names) else f"ct_{c}",
            "gene": gene_names[g] if g < len(gene_names) else f"gene_{g}",
            "mean_abs_attribution": float(per_ct_importance[c, g]),
        })

    # Per-cell-type total importance (summed over genes).
    per_ct_total = per_ct_importance.sum(axis=1)
    ct_rank = [
        {"cell_type": ct_names[c] if c < len(ct_names) else f"ct_{c}",
         "total_abs_attribution": float(per_ct_total[c])}
        for c in np.argsort(-per_ct_total)
    ]

    return {
        "n_subjects": int(attr.shape[0]),
        "n_cell_types": int(attr.shape[1]),
        "n_genes": int(attr.shape[2]),
        "top_global_genes": top_global,
        "top_genes_per_cell_type": top_per_ct,
        "top_cell_type_gene_pairs": top_pairs,
        "cell_types_ranked_by_total_attribution": ct_rank,
    }


def _load_gene_names(precomputed_dir: Path, n_genes: int) -> list[str]:
    """Try to load the gene-name list from precomputed feature metadata.

    Supports both .npy (written by ``precompute_features`` in datasets.py) and
    .json sidecars. Falls back to ``gene_<i>`` placeholders if none are found —
    downstream interpretability output will be unreadable until a real sidecar
    is added.
    """
    candidates = [precomputed_dir / "gene_names.npy",
                  precomputed_dir / "gene_names.json",
                  precomputed_dir / "feature_names.json",
                  Path("data/canonical/gene_names.json")]
    for p in candidates:
        if not p.exists():
            continue
        if p.suffix == ".npy":
            names = np.load(p, allow_pickle=True).tolist()
        else:
            names = json.loads(p.read_text())
        if isinstance(names, list) and len(names) >= n_genes:
            logger.info("Loaded %d gene names from %s", n_genes, p)
            return [str(n) for n in names[:n_genes]]
    logger.warning(
        "No gene-name file found in expected locations (%s). Using gene_<i> placeholders.",
        ", ".join(str(p) for p in candidates),
    )
    return [f"gene_{i}" for i in range(n_genes)]


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    sids_per: list[np.ndarray] = []
    attrs_per: list[np.ndarray] = []
    preds_per: list[np.ndarray] = []
    folds_per: list[np.ndarray] = []
    for f in range(5):
        d = attribute_one_fold(args, f, device)
        sids_per.append(d["subject_ids"])
        attrs_per.append(d["attributions"])
        preds_per.append(d["predictions_residual"])
        folds_per.append(d["fold"])
        logger.info("fold %d: %s attributions extracted", f, d["attributions"].shape)

    sids = np.concatenate(sids_per)
    attr = np.concatenate(attrs_per, axis=0)
    preds = np.concatenate(preds_per)
    folds = np.concatenate(folds_per)

    n_subj, n_ct, n_genes = attr.shape
    logger.info("Total: %d subjects × %d cell types × %d genes (%.1f MB)",
                n_subj, n_ct, n_genes, attr.nbytes / 1e6)

    out_npz = out_dir / "composite_attributions.npz"
    np.savez(out_npz, subject_ids=sids, attributions=attr,
             predictions_residual=preds, fold=folds)
    logger.info("Wrote %s", out_npz)

    # Cell-type names from constants (truncate to actual n_ct).
    ct_names = list(CELL_TYPE_ORDER)[:n_ct]
    if len(ct_names) < n_ct:
        ct_names = ct_names + [f"ct_{c}" for c in range(len(ct_names), n_ct)]

    # Gene names — best-effort from precomputed_dir; fall back to placeholders.
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(args.config),
    )
    gene_names = _load_gene_names(Path(cfg.data.precomputed_dir), n_genes)

    summary = summarize(out_npz, ct_names, gene_names, top_k=int(args.top_k))
    summary_path = out_dir / "composite_attribution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", summary_path)

    # Top-100 (cell_type, gene) pairs CSV — paper-table-ready.
    top_pairs_top100 = summary["top_cell_type_gene_pairs"][:min(100, len(summary["top_cell_type_gene_pairs"]))]
    pd.DataFrame(top_pairs_top100).to_csv(out_dir / "top_pairs_table.csv", index=False)
    logger.info("Wrote %s", out_dir / "top_pairs_table.csv")

    print()
    print("=== Top-10 (cell_type, gene) pairs by mean |attribution| ===")
    for i, p in enumerate(summary["top_cell_type_gene_pairs"][:10], 1):
        print(f"  {i:>2}. {p['cell_type']:<40s}  {p['gene']:<20s}  {p['mean_abs_attribution']:.6f}")

    print()
    print("=== Cell types ranked by total attribution mass ===")
    for i, c in enumerate(summary["cell_types_ranked_by_total_attribution"][:10], 1):
        print(f"  {i:>2}. {c['cell_type']:<40s}  {c['total_abs_attribution']:.4f}")

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Captum IG composite attribution.")
    p.add_argument("--config", default="configs/resdec_mhe/canonical.yaml")
    p.add_argument("--pred-root", default="outputs/canonical/p5_canonical_seed42")
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--out-dir", default="outputs/canonical/interpretability")
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--internal-batch-size", type=int, default=4)
    p.add_argument("--top-k", type=int, default=50)
    sys.exit(main(p.parse_args()))
