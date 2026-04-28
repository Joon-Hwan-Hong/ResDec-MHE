"""Deterministic A/B confirmation of §31.7 SDPA backward non-determinism mechanism.

Per MASTER-INFO §31.7: the diff-test 5-fold ~0.07 R² gap is hypothesised to be
caused by SDPA's non-deterministic backward atomic-add reduction order, perturbed
by the no_grad einsum+softmax block changing cudaMallocAsync pool layout. To
raise confidence to 100% via a single definitive experiment, this script re-runs
the diff-test fold-0 vs canonical fold-0 A/B with::

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)

If the gap **DISAPPEARS** under deterministic mode → mechanism CONFIRMED.
If it **persists** → other mechanism still active.

Notes on code-path tested
-------------------------
The d4 refactor of `src/models/full_model.py:549` (drops the
``or self.return_attention_in_training`` clause from the
``return_attention_weights=`` argument) eliminates the no_grad einsum+softmax
re-compute during training irrespective of deterministic mode. If the refactor
is already applied when this script runs, the diff_test config will produce
canonical R² regardless of deterministic-mode setting — confirming the OPPOSITE
mechanism (SDPA backward non-determinism is irrelevant; the no_grad block was
the perturbing event).

The script logs the current git SHA AND the literal text of `full_model.py:549`
so the caller can attribute the result to the correct code state.

Usage
-----
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \\
        uv run python \\
            scripts/resdec_mhe/interpretability/diff_test_deterministic_confirmation.py \\
            --output-root outputs/canonical/p5_diff_test_mechanism/deterministic_confirmation
"""
# CRITICAL: deterministic env vars must be set BEFORE any torch import.
# pyflakes:ignore  (intentional pre-torch import side-effect)
from __future__ import annotations

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import lightning.pytorch as pl  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.data.datamodule import CognitiveResilienceDataModule  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.training.callbacks import MinEpochEarlyStopping  # noqa: E402
from src.training.resdec_lightning_module import ResDecLightningModule  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _git_sha() -> str:
    """Return short git SHA of the worktree HEAD ('unknown' on failure)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover  - non-fatal probe
        return "unknown"


def _full_model_signature() -> dict[str, str | bool]:
    """Probe ``src/models/full_model.py`` to determine which code path is active.

    Returns a dict with:
      - ``line549_text``: the literal text of the
        ``return_attention_weights=...`` line at ~L549 (post-d4 refactor)
        or ~L546 (pre-d4 refactor).
      - ``has_or_clause``: True iff the legacy
        ``or self.return_attention_in_training`` clause is present (i.e., the
        d4 refactor has NOT been applied → diff_test config still triggers the
        no_grad block during training).
    """
    fm_path = PROJECT_ROOT / "src" / "models" / "full_model.py"
    text = fm_path.read_text()
    has_or_clause = "or self.return_attention_in_training" in text

    # Find the canonical CognitiveResilienceModel.forward call site for the
    # pathology_attention invocation (there are two — line ~549 and ~676 — both
    # gated by the same flag in the legacy code).
    line_text = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("return_attention_weights="):
            line_text = stripped
            break
    return {
        "path": str(fm_path),
        "line_text": line_text,
        "has_or_clause": bool(has_or_clause),
    }


def _setup_determinism(seed: int) -> None:
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def _load_cfg(config_path: Path, fold: int, seed: int) -> dict:
    default_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "default.yaml")
    override_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(default_cfg, override_cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.model.head.type = "deterministic"
    cfg.experiment.seed = int(seed)
    cfg.data.fold = int(fold)
    return cfg


def _train_one(
    label: str,
    config_path: Path,
    splits_path: Path,
    output_dir: Path,
    fold: int,
    seed: int,
) -> dict:
    """Train a single fold to convergence with deterministic mode and return
    a summary dict (val/r2 + wall time + config + metadata).

    Mirrors the canonical training entry in
    ``scripts/resdec_mhe/training/train.py`` but skips CSV logging,
    permutation overrides, frozen-encoder branch, and per-subject prediction
    npz writing — we only need val/r2 at end of training for the A/B.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log"

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    try:
        logger.info("[%s] starting training for fold %d, seed %d", label, fold, seed)
        logger.info("[%s] config = %s", label, config_path)
        _setup_determinism(seed)

        cfg = _load_cfg(config_path, fold, seed)

        splits = load_splits(str(splits_path))
        metadata_csv = Path(cfg.data.metadata_path) / "metadata.csv"
        metadata = pd.read_csv(metadata_csv)

        precomputed_dir = cfg.data.get("precomputed_dir", None)
        if precomputed_dir is None:
            raise ValueError(
                "data.precomputed_dir is None in merged config — required for "
                "live-encoder path."
            )

        dm = CognitiveResilienceDataModule(
            config=cfg,
            metadata=metadata,
            splits=splits,
            fold_idx=int(fold),
            precomputed_dir=precomputed_dir,
            adata=None,
        )
        model = ResDecLightningModule(cfg)

        # ModelCheckpoint with best-by-val/r2 (mirrors canonical training).
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            save_last=True,
            save_top_k=1,
            monitor="val/r2",
            mode="max",
            filename="best-{epoch}-{val/r2:.4f}",
            auto_insert_metric_name=False,
        )
        callbacks: list = [checkpoint_cb]
        es_cfg = cfg.training.get("early_stopping", None)
        if es_cfg is not None:
            es_cb = MinEpochEarlyStopping(
                min_epochs=int(es_cfg.get("min_epochs", 3)),
                monitor=str(es_cfg.get("monitor", "val/r2")),
                mode=str(es_cfg.get("mode", "max")),
                patience=int(es_cfg.get("patience", 5)),
                min_delta=float(es_cfg.get("min_delta", 0.0)),
                verbose=True,
            )
            callbacks.append(es_cb)

        trainer = pl.Trainer(
            max_epochs=int(cfg.training.max_epochs),
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=True,
            callbacks=callbacks,
            enable_progress_bar=False,
            precision=str(cfg.training.get("precision", "bf16-mixed")),
            enable_model_summary=False,
            # Lightning's deterministic flag: error rather than warn so we
            # surface any non-deterministic op we're missing. warn_only=False
            # is set on torch.use_deterministic_algorithms(...) above; here we
            # use Lightning's "True" not "warn" to match.
            deterministic=True,
            default_root_dir=str(output_dir),
        )
        t0 = time.time()
        trainer.fit(model, datamodule=dm)
        t_fit = time.time() - t0

        val_results = trainer.validate(model, datamodule=dm, verbose=False)
        wall_seconds = time.time() - t0

        summary = {
            "label": label,
            "config": str(config_path),
            "fold": int(fold),
            "seed": int(seed),
            "max_epochs": int(cfg.training.max_epochs),
            "val_results": val_results,
            "fit_seconds": float(t_fit),
            "wall_seconds": float(wall_seconds),
            "deterministic_mode": True,
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG", ""
            ),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.info("[%s] DONE — wall=%.1fs val/r2=%s",
                    label, wall_seconds,
                    val_results[0].get("val/r2") if val_results else "?")
        return summary
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


def _val_r2_from_summary(summary: dict) -> float | None:
    val = summary.get("val_results")
    if not val:
        return None
    return float(val[0].get("val/r2", float("nan")))


def main(args: argparse.Namespace) -> None:
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Code-state probe: did the d4 refactor of full_model.py:549 happen?
    sha = _git_sha()
    fm_sig = _full_model_signature()
    logger.info("git SHA = %s", sha)
    logger.info("full_model.py signature = %s", fm_sig)

    canonical_dir = out_root / "canonical"
    diff_test_dir = out_root / "diff_test"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    diff_test_dir.mkdir(parents=True, exist_ok=True)

    splits_path = Path(args.splits_path)
    if not splits_path.exists():
        raise FileNotFoundError(f"splits not found: {splits_path}")

    canonical_cfg = Path(args.canonical_config)
    diff_test_cfg = Path(args.diff_test_config)
    if not canonical_cfg.exists():
        raise FileNotFoundError(f"canonical config not found: {canonical_cfg}")
    if not diff_test_cfg.exists():
        raise FileNotFoundError(f"diff_test config not found: {diff_test_cfg}")

    # Sequential A then B on GPU 0.
    canonical_summary = _train_one(
        label="canonical",
        config_path=canonical_cfg,
        splits_path=splits_path,
        output_dir=canonical_dir,
        fold=int(args.fold),
        seed=int(args.seed),
    )
    diff_test_summary = _train_one(
        label="diff_test",
        config_path=diff_test_cfg,
        splits_path=splits_path,
        output_dir=diff_test_dir,
        fold=int(args.fold),
        seed=int(args.seed),
    )

    canonical_r2 = _val_r2_from_summary(canonical_summary)
    diff_test_r2 = _val_r2_from_summary(diff_test_summary)
    delta = (
        canonical_r2 - diff_test_r2
        if canonical_r2 is not None and diff_test_r2 is not None
        else None
    )

    # Verdict: if |Δ R²| < 0.005 with the legacy or-clause STILL active, the
    # SDPA-backward-non-determinism mechanism is CONFIRMED. If the d4 refactor
    # is already applied, the gap was eliminated by removing the no_grad block,
    # not by deterministic mode — we report this distinctly.
    threshold = float(args.gap_threshold)
    if delta is None:
        verdict = "INCONCLUSIVE"
        rationale = "missing val/r2 in one or both runs"
    elif fm_sig["has_or_clause"]:
        if abs(delta) < threshold:
            verdict = "MECHANISM CONFIRMED"
            rationale = (
                f"|Δ R²| = {abs(delta):.4f} < {threshold:.3f} on legacy code "
                "(or-clause active). SDPA-backward non-determinism + "
                "cudaMallocAsync pool perturbation by the no_grad einsum+softmax "
                "block fully accounts for the original ~0.07 R² gap; "
                "deterministic mode eliminates it."
            )
        else:
            verdict = "MECHANISM NOT CONFIRMED"
            rationale = (
                f"|Δ R²| = {abs(delta):.4f} >= {threshold:.3f} on legacy code. "
                "Deterministic SDPA backward did NOT close the gap; another "
                "mechanism (kernel selection, allocator-pool latency, etc.) "
                "is still active. Re-investigate."
            )
    else:
        if abs(delta) < threshold:
            verdict = "INCONCLUSIVE (refactor active)"
            rationale = (
                f"|Δ R²| = {abs(delta):.4f} < {threshold:.3f}, BUT the d4 "
                "refactor has already eliminated the no_grad block during "
                "training. The gap was closed by removing the perturbing block, "
                "not by deterministic mode. To literally confirm the SDPA "
                "backward mechanism, re-run on the legacy code (revert the "
                "two-line change at full_model.py:549 + 676)."
            )
        else:
            verdict = "ANOMALY"
            rationale = (
                f"|Δ R²| = {abs(delta):.4f} >= {threshold:.3f} despite the "
                "d4 refactor being applied. Diff-test config should now match "
                "canonical regardless of determinism. Investigate why the gap "
                "persists."
            )

    report = {
        "git_sha": sha,
        "full_model_signature": fm_sig,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "deterministic_mode": True,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", ""
        ),
        "canonical": {
            "config": str(canonical_cfg),
            "val_r2": canonical_r2,
            "wall_seconds": canonical_summary.get("wall_seconds"),
        },
        "diff_test": {
            "config": str(diff_test_cfg),
            "val_r2": diff_test_r2,
            "wall_seconds": diff_test_summary.get("wall_seconds"),
        },
        "delta_r2_canonical_minus_diff_test": delta,
        "abs_delta_r2": abs(delta) if delta is not None else None,
        "gap_threshold_for_confirmation": threshold,
        "verdict": verdict,
        "rationale": rationale,
        "comparison_to_non_determ_baseline": {
            "non_deterministic_canonical_r2_5fold_mean": 0.4436,
            "non_deterministic_diff_test_r2_5fold_mean": 0.3760,
            "non_deterministic_gap_5fold_mean": 0.0676,
            "note": (
                "Per §31.7 the original 5-fold mean gap was ~0.07 R². The "
                "fold-0 single-fold deterministic delta should be ≪ that if "
                "the mechanism is confirmed (legacy code) or zero if the "
                "refactor is applied."
            ),
        },
    }

    out_path = out_root / "comparison_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote comparison report to %s", out_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--output-root",
        default="outputs/canonical/p5_diff_test_mechanism/deterministic_confirmation",
        help="Root output directory; canonical/ and diff_test/ subdirs are "
             "created with summary.json + log + checkpoints. Comparison "
             "report is written to comparison_report.json at the root.",
    )
    p.add_argument(
        "--canonical-config",
        default="configs/resdec_mhe/canonical.yaml",
        help="Path to the canonical config YAML.",
    )
    p.add_argument(
        "--diff-test-config",
        default="configs/resdec_mhe/diff_test_no_reg_with_flag.yaml",
        help="Path to the diff-test config YAML.",
    )
    p.add_argument(
        "--splits-path",
        default="outputs/splits.json",
        help="Path to the 5-fold splits JSON.",
    )
    p.add_argument(
        "--fold", type=int, default=0,
        help="Fold index (0-4); defaults to 0 per the §31.7 mechanism probe.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed for both A and B runs (must match for the comparison to be "
             "valid).",
    )
    p.add_argument(
        "--gap-threshold", type=float, default=0.005,
        help="If |canonical - diff_test| val/r2 falls below this and the "
             "legacy or-clause is active, the mechanism is CONFIRMED.",
    )
    main(p.parse_args())
