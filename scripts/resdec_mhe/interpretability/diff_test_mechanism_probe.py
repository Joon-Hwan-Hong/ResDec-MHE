"""Diff-test mechanism probe: per-step trajectory comparator.

Investigates the deeper mechanistic reason why
`model.return_attention_in_training=True` triggers a ~0.07 R² regression even
though the SDPA fast path forward is mathematically unchanged. The probe trains
fold 0 of either canonical or diff-test config, with a deterministic seeded
state, and dumps per-step diagnostics to JSONL.

The five hypotheses tested are:
- H1: CUDA kernel selection drift (extra einsum changes downstream cuBLAS heuristics)
- H2: RNG state advance from no_grad einsum + softmax (subsequent dropout differs)
- H3: bf16 numerical drift (.float() softmax cast not bit-identical to SDPA's
       internal fp32 softmax)
- H4: Memory allocator pool fragmentation (intermediate tensors → different
       allocs in backward → reproducibility loss)
- H5: Backward-pass interaction with autograd bookkeeping

The probe captures, per training step:
    - step index, epoch, loss
    - sum of grad-norm per parameter group
    - attention_weights mean / std for first batch
    - prediction mean / std
    - global CUDA RNG state hash (sha256 of the full state)
    - global CPU RNG state hash
    - bit-exact tensor diff at step 1 (saved separately)

Usage
-----
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \\
        uv run python scripts/resdec_mhe/interpretability/diff_test_mechanism_probe.py \\
            --config configs/resdec_mhe/canonical.yaml \\
            --output-dir outputs/canonical/p5_diff_test_mechanism/canonical \\
            --max-epochs 5

The actual A/B run is two such invocations with different configs (canonical
vs diff_test_no_reg_with_flag).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.datamodule import CognitiveResilienceDataModule
from src.data.splits import load_splits
from src.training.resdec_lightning_module import ResDecLightningModule

logger = logging.getLogger(__name__)


def _cuda_rng_state_hash() -> str:
    """sha256 hash of the full CUDA RNG state."""
    state = torch.cuda.get_rng_state()
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()[:16]


def _cpu_rng_state_hash() -> str:
    """sha256 hash of the full CPU RNG state."""
    state = torch.get_rng_state()
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()[:16]


def _np_rng_state_hash() -> str:
    """sha256 hash of the NumPy RNG state."""
    state = np.random.get_state(legacy=True)
    # state is a tuple ('MT19937', uint32-array, pos, has_gauss, cached_gaussian)
    payload = state[1].tobytes() + bytes([state[2] % 256, state[3], 0])
    return hashlib.sha256(payload).hexdigest()[:16]


def _grad_norm_by_group(model: torch.nn.Module) -> dict[str, float]:
    """Sum of |grad|² by parameter group prefix (encoder.* / head.* / etc.)."""
    groups: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        prefix = name.split(".")[0]  # 'encoder' or 'head' etc.
        groups[prefix] = groups.get(prefix, 0.0) + float(p.grad.norm().item() ** 2)
    return {k: v ** 0.5 for k, v in groups.items()}


class TrajectoryProbeCallback(pl.Callback):
    """Lightning callback that dumps per-step JSONL records.

    Records are appended to ``out_dir / 'trajectory.jsonl'``. A separate
    ``step1_tensors.pt`` file dumps the bit-exact tensors from step 1 (first
    batch's attention_weights, predictions, parameter checksums).
    """

    def __init__(self, out_dir: Path):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "trajectory.jsonl"
        # Truncate prior runs.
        self.jsonl_path.write_text("")
        self.step1_path = self.out_dir / "step1_tensors.pt"
        self.step_idx: int = 0
        self._latest_loss: float | None = None
        self._latest_attn_stats: dict[str, float] | None = None
        self._latest_pred_stats: dict[str, float] | None = None

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: dict,
        batch_idx: int,
    ) -> None:
        # Pre-step record (RNG state + param checksum BEFORE forward).
        self._pre_record = {
            "step": self.step_idx,
            "epoch": trainer.current_epoch,
            "phase": "pre_step",
            "cuda_rng": _cuda_rng_state_hash(),
            "cpu_rng": _cpu_rng_state_hash(),
            "np_rng": _np_rng_state_hash(),
            "param_checksum": self._param_checksum(pl_module),
        }

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: dict,
        batch_idx: int,
    ) -> None:
        rec = dict(self._pre_record)
        rec["phase"] = "post_step"

        # Loss extraction. PL gives us {"loss": tensor} or just tensor.
        if isinstance(outputs, dict) and "loss" in outputs:
            loss_val = float(outputs["loss"].item())
        elif isinstance(outputs, torch.Tensor):
            loss_val = float(outputs.item())
        else:
            loss_val = float("nan")
        rec["loss"] = loss_val
        rec["grad_norms"] = _grad_norm_by_group(pl_module)
        rec["post_cuda_rng"] = _cuda_rng_state_hash()
        rec["post_cpu_rng"] = _cpu_rng_state_hash()
        rec["post_param_checksum"] = self._param_checksum(pl_module)
        if self._latest_attn_stats is not None:
            rec["attn"] = self._latest_attn_stats
        if self._latest_pred_stats is not None:
            rec["pred"] = self._latest_pred_stats

        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        # On step 1 only: dump tensors for bit-exact comparison.
        if self.step_idx == 0 and not self.step1_path.exists():
            self._dump_step1(pl_module, batch)

        self.step_idx += 1

    @staticmethod
    def _param_checksum(model: torch.nn.Module) -> str:
        """sha256 hash over all trainable parameters' contents (concatenated)."""
        h = hashlib.sha256()
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Cast to fp32 for stable hashing across precision modes.
            arr = p.detach().to(torch.float32).cpu().numpy().tobytes()
            h.update(name.encode())
            h.update(arr)
        return h.hexdigest()[:16]

    def _dump_step1(self, model: ResDecLightningModule, batch: dict) -> None:
        """Replay one forward in eval-mode to grab attention_weights without
        perturbing training trajectory; ALSO save current parameters.

        Saves and restores CUDA + CPU RNG state so the eval-replay forward does
        not advance the training-loop RNG state.
        """
        # Save RNG state so the eval replay does not perturb subsequent steps.
        cuda_state = (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        )
        cpu_state = torch.get_rng_state()
        np_state = np.random.get_state()

        # Collect the batch's first-row attention_weights from a no-grad replay.
        was_training = model.training
        model.eval()
        with torch.no_grad():
            # Forward through encoder to get attention_weights tensor.
            # Move batch to the module's device (model.device on LightningModule).
            _device = model.device
            batch_dev = {
                k: (v.to(_device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            enc_kwargs = model._batch_to_encoder_kwargs(batch_dev)
            enc_out = model.encoder(**enc_kwargs)
            attn = enc_out.get("attention_weights")
            attended = enc_out.get("attended")
        if was_training:
            model.train()

        # Restore RNG state.
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        torch.set_rng_state(cpu_state)
        np.random.set_state(np_state)

        param_state = {
            name: p.detach().to(torch.float32).cpu().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        torch.save(
            {
                "attention_weights": (
                    attn.detach().to(torch.float32).cpu() if attn is not None else None
                ),
                "attended": (
                    attended.detach().to(torch.float32).cpu()
                    if attended is not None
                    else None
                ),
                "params": param_state,
            },
            self.step1_path,
        )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Per-epoch flush marker.
        rec = {
            "step": self.step_idx,
            "epoch": trainer.current_epoch,
            "phase": "epoch_end",
            "cuda_rng": _cuda_rng_state_hash(),
            "param_checksum": self._param_checksum(pl_module),
        }
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


class _AttentionStatsHook:
    """Lightweight forward hook on encoder.pathology_attention to grab
    attention_weights mean / std on every step. Wired to the callback's
    cache so the callback emits these in JSONL.

    Hook-firing-order assumption (read carefully before reusing this probe):

    * ``register_forward_hook`` runs hooks in the order they were registered,
      AFTER the wrapped module's forward returns and BEFORE control returns
      to the caller. We rely on this to ensure ``_latest_attn_stats`` and
      ``_latest_pred_stats`` are populated before the Lightning callback's
      ``on_train_batch_end`` fires (which is invoked AFTER the full step,
      including the head forward pass).
    * Both hooks read tensors from a single forward; PL Lightning runs the
      whole forward / loss / backward / step sequence inside one
      ``training_step``, so by the time ``on_train_batch_end`` is called,
      both hooks have run exactly once for this step.
    * If a future refactor of ResDec-MHE moves the attention call to a
      different module or runs the head before pathology_attention, this
      ordering invariant breaks and the probe will record stale stats from
      the *previous* batch. Add a step-id check or rebind hooks to the
      correct module if that happens.
    """

    def __init__(self, callback: TrajectoryProbeCallback):
        self.callback = callback

    def __call__(self, module: torch.nn.Module, inputs, outputs) -> None:
        # outputs = (attended, attention_weights | None)
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            return
        attn = outputs[1]
        if attn is None:
            self.callback._latest_attn_stats = {"present": False}
            return
        attn_f = attn.detach().to(torch.float32)
        self.callback._latest_attn_stats = {
            "present": True,
            "mean": float(attn_f.mean().item()),
            "std": float(attn_f.std().item()),
            "max": float(attn_f.max().item()),
            "min": float(attn_f.min().item()),
            "shape": list(attn.shape),
        }


class _PredictionStatsHook:
    """Forward hook on the head's last layer to grab prediction stats.

    Same hook-firing-order assumption as :class:`_AttentionStatsHook` —
    ResDec-MHE's training_step calls encoder → head exactly once per batch,
    so this hook fires exactly once per training step and the cached stats
    are fresh by the time ``on_train_batch_end`` is invoked.
    """

    def __init__(self, callback: TrajectoryProbeCallback):
        self.callback = callback

    def __call__(self, module: torch.nn.Module, inputs, outputs) -> None:
        if isinstance(outputs, dict) and "prediction" in outputs:
            pred = outputs["prediction"].detach().to(torch.float32)
            self.callback._latest_pred_stats = {
                "mean": float(pred.mean().item()),
                "std": float(pred.std().item()),
            }


def main(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------ #
    # Determinism setup                                                  #
    # ------------------------------------------------------------------ #
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    seed = int(args.seed)
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # warn_only=True so any algorithm without a deterministic impl falls back
    # rather than aborting the run (we want the trajectory regardless).
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------ #
    # Config loading                                                     #
    # ------------------------------------------------------------------ #
    default_cfg = OmegaConf.load("configs/default.yaml")
    override_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(default_cfg, override_cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.model.head.type = "deterministic"
    cfg.training.max_epochs = int(args.max_epochs)
    cfg.experiment.seed = seed
    cfg.data.fold = int(args.fold)

    torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------ #
    # Splits + metadata                                                  #
    # ------------------------------------------------------------------ #
    splits_path = Path(args.splits_path)
    metadata_csv = Path(cfg.data.metadata_path) / "metadata.csv"
    splits = load_splits(str(splits_path))
    metadata = pd.read_csv(metadata_csv)

    precomputed_dir = args.precomputed_dir or cfg.data.get("precomputed_dir", None)
    dm = CognitiveResilienceDataModule(
        config=cfg,
        metadata=metadata,
        splits=splits,
        fold_idx=int(args.fold),
        precomputed_dir=precomputed_dir,
        adata=None,
    )
    model = ResDecLightningModule(cfg)

    # ------------------------------------------------------------------ #
    # Callback + hooks                                                   #
    # ------------------------------------------------------------------ #
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cb = TrajectoryProbeCallback(out_dir=out_dir)
    attn_hook = _AttentionStatsHook(cb)
    pred_hook = _PredictionStatsHook(cb)
    if hasattr(model.encoder, "pathology_attention"):
        model.encoder.pathology_attention.register_forward_hook(attn_hook)
    if hasattr(model, "head"):
        model.head.register_forward_hook(pred_hook)

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[cb],
        enable_progress_bar=False,
        precision=str(cfg.training.get("precision", "bf16-mixed")),
        enable_model_summary=False,
        deterministic="warn",
        default_root_dir=str(out_dir),
    )
    trainer.fit(model, datamodule=dm)

    # Final summary record.
    summary = {
        "config": args.config,
        "fold": int(args.fold),
        "seed": seed,
        "max_epochs": int(cfg.training.max_epochs),
        "n_steps_recorded": cb.step_idx,
        "compute_attention_with_grad": bool(
            getattr(
                getattr(model.encoder, "pathology_attention", None),
                "compute_attention_with_grad",
                False,
            )
        ),
        # NOTE: ``return_attention_in_training`` is a CONSTRUCTION-TIME flag
        # on the encoder. ResDec-MHE sets it once when the encoder module is
        # built from config and never mutates it during training (no hot-swap
        # path exists). The summary records its value at write time, but the
        # whole probe assumes the attribute is the same as at module init.
        "return_attention_in_training": bool(
            getattr(model.encoder, "return_attention_in_training", False)
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default=None)
    main(p.parse_args())
