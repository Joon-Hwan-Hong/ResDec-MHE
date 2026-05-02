"""Compare two trajectory.jsonl files (canonical vs diff_test) and identify
the first step of divergence and the signal that diverged first.

Reads:
    outputs/canonical/p5_diff_test_mechanism/canonical/trajectory.jsonl
    outputs/canonical/p5_diff_test_mechanism/canonical/step1_tensors.pt
    outputs/canonical/p5_diff_test_mechanism/diff_test/trajectory.jsonl
    outputs/canonical/p5_diff_test_mechanism/diff_test/step1_tensors.pt

Writes:
    outputs/canonical/p5_diff_test_mechanism/comparison_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    # Resolve to absolute so a relative path is taken from cwd loudly rather
    # than silently against the worktree root.
    path = Path(path).resolve()
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _post_step_records(records: list[dict]) -> list[dict]:
    """Filter for ``phase == "post_step"`` records.

    Asserts the filter result is non-empty so a future schema rename
    (``post_step`` → ``post_train_batch_end`` etc.) fails loudly instead
    of silently returning [], which would emit "no divergence detected"
    instead of failing.
    """
    post = [r for r in records if r.get("phase") == "post_step"]
    assert post, (
        f"No post_step records found in trajectory ({len(records)} total "
        "records walked). Check that diff_test_mechanism_probe still emits "
        "phase='post_step' — schema may have drifted."
    )
    return post


# Per-signal divergence epsilons. Each is set just above the expected
# floor for that signal so we surface meaningful divergences while
# ignoring fp32-precision-floor noise:
#   _EPS_LOSS    = 1e-3  — bf16 forward loss precision floor (~1e-4 in
#                          practice; 10x headroom).
#   _EPS_ATTN    = 1e-6  — fp32 attention softmax output precision floor.
#   _EPS_PRED    = 1e-5  — head's residual prediction (fp32) precision.
#   _EPS_GRADNRM = 1e-4  — bf16 backward grad-norm noise floor (10x looser
#                          than forward because bf16 backward kernels are
#                          intentionally non-deterministic at this scale).
_EPS_LOSS: float = 1e-3
_EPS_ATTN: float = 1e-6
_EPS_PRED: float = 1e-5
_EPS_GRADNRM: float = 1e-4


def _epsilon_diff(a: float, b: float, eps: float = _EPS_LOSS) -> bool:
    return abs(a - b) > eps


def _find_first_divergence(
    canon: list[dict], diff: list[dict], eps_loss: float = _EPS_LOSS,
) -> dict[str, Any]:
    """Walk both trajectories step-by-step, find first divergence by signal."""
    n = min(len(canon), len(diff))
    out: dict[str, Any] = {
        "first_step_pre_cuda_rng_diff": None,
        "first_step_post_cuda_rng_diff": None,
        "first_step_post_param_diff": None,
        "first_step_loss_diff": None,
        "first_step_attn_mean_diff": None,
        "first_step_pred_mean_diff": None,
        "first_step_grad_norm_encoder_diff": None,
    }
    for i in range(n):
        a, b = canon[i], diff[i]
        if (
            out["first_step_pre_cuda_rng_diff"] is None
            and a.get("cuda_rng") != b.get("cuda_rng")
        ):
            out["first_step_pre_cuda_rng_diff"] = i
        if (
            out["first_step_post_cuda_rng_diff"] is None
            and a.get("post_cuda_rng") != b.get("post_cuda_rng")
        ):
            out["first_step_post_cuda_rng_diff"] = i
        if (
            out["first_step_post_param_diff"] is None
            and a.get("post_param_checksum") != b.get("post_param_checksum")
        ):
            out["first_step_post_param_diff"] = i
        if (
            out["first_step_loss_diff"] is None
            and a.get("loss") is not None
            and b.get("loss") is not None
            and _epsilon_diff(float(a["loss"]), float(b["loss"]), eps_loss)
        ):
            out["first_step_loss_diff"] = {
                "step": i,
                "canonical": float(a["loss"]),
                "diff_test": float(b["loss"]),
                "delta": float(a["loss"]) - float(b["loss"]),
            }
        ax = a.get("attn") or {}
        bx = b.get("attn") or {}
        if (
            out["first_step_attn_mean_diff"] is None
            and ax.get("present") and bx.get("present")
            and "mean" in ax and "mean" in bx
            and _epsilon_diff(float(ax["mean"]), float(bx["mean"]), _EPS_ATTN)
        ):
            out["first_step_attn_mean_diff"] = {
                "step": i,
                "canonical_mean": float(ax["mean"]),
                "diff_test_mean": float(bx["mean"]),
            }
        ap = a.get("pred") or {}
        bp = b.get("pred") or {}
        if (
            out["first_step_pred_mean_diff"] is None
            and "mean" in ap and "mean" in bp
            and _epsilon_diff(float(ap["mean"]), float(bp["mean"]), _EPS_PRED)
        ):
            out["first_step_pred_mean_diff"] = {
                "step": i,
                "canonical_mean": float(ap["mean"]),
                "diff_test_mean": float(bp["mean"]),
            }
        a_gn = (a.get("grad_norms") or {}).get("encoder")
        b_gn = (b.get("grad_norms") or {}).get("encoder")
        if (
            out["first_step_grad_norm_encoder_diff"] is None
            and a_gn is not None and b_gn is not None
            and _epsilon_diff(float(a_gn), float(b_gn), _EPS_GRADNRM)
        ):
            out["first_step_grad_norm_encoder_diff"] = {
                "step": i,
                "canonical": float(a_gn),
                "diff_test": float(b_gn),
            }
    return out


def _step_snapshot(records: list[dict], step: int) -> dict[str, Any]:
    if step >= len(records):
        return {"missing": True, "step": step}
    r = records[step]
    return {
        "step": step,
        "loss": r.get("loss"),
        "cuda_rng": r.get("cuda_rng"),
        "post_cuda_rng": r.get("post_cuda_rng"),
        "param_checksum": r.get("param_checksum"),
        "post_param_checksum": r.get("post_param_checksum"),
        "grad_norms": r.get("grad_norms"),
        "attn": r.get("attn"),
        "pred": r.get("pred"),
    }


def _compare_step1_tensors(
    canon_path: Path, diff_path: Path
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not canon_path.exists() or not diff_path.exists():
        out["missing"] = True
        out["canon_exists"] = canon_path.exists()
        out["diff_exists"] = diff_path.exists()
        return out
    canon = torch.load(canon_path, map_location="cpu", weights_only=False)
    diff = torch.load(diff_path, map_location="cpu", weights_only=False)

    # Compare attention_weights
    a_attn = canon.get("attention_weights")
    b_attn = diff.get("attention_weights")
    if a_attn is not None and b_attn is not None:
        delta = (a_attn - b_attn).abs()
        out["attn_diff_max"] = float(delta.max().item())
        out["attn_diff_mean"] = float(delta.mean().item())
        out["attn_canon_mean"] = float(a_attn.mean().item())
        out["attn_diff_test_mean"] = float(b_attn.mean().item())
        out["attn_bit_exact"] = bool(out["attn_diff_max"] == 0.0)

    # Compare attended
    a_at = canon.get("attended")
    b_at = diff.get("attended")
    if a_at is not None and b_at is not None:
        delta = (a_at - b_at).abs()
        out["attended_diff_max"] = float(delta.max().item())
        out["attended_diff_mean"] = float(delta.mean().item())
        out["attended_bit_exact"] = bool(out["attended_diff_max"] == 0.0)

    # Compare params (count of params where checksum differs).
    # F9: stack per-name max-abs-deltas into one tensor and pull a single
    # ``.item()`` rather than one sync per name. The "first diff" name still
    # honours ``sorted(common)`` order (we walk the names in the same order
    # we built the deltas tensor, so ``argmax(deltas > 0)`` over a contiguous
    # sorted list is identical to the prior loop's first-hit semantics).
    a_params = canon.get("params") or {}
    b_params = diff.get("params") or {}
    common = sorted(set(a_params.keys()) & set(b_params.keys()))
    if common:
        deltas = torch.stack([
            (a_params[name] - b_params[name]).abs().max() for name in common
        ])
        deltas_np = deltas.detach().cpu().numpy()
        n_diff = int((deltas_np > 0.0).sum())
        max_param_delta = float(deltas_np.max())
        first_idx = int(np.argmax(deltas_np > 0.0)) if n_diff > 0 else -1
        first_diff_param = common[first_idx] if first_idx >= 0 else None
    else:
        n_diff = 0
        max_param_delta = 0.0
        first_diff_param = None
    out["param_n_diff"] = n_diff
    out["param_n_total"] = len(common)
    out["param_max_delta"] = float(max_param_delta)
    out["param_first_diff_name"] = first_diff_param
    return out


def _compare_at_steps(
    canon: list[dict], diff: list[dict], steps: list[int]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in steps:
        out[f"step_{s}"] = {
            "canonical": _step_snapshot(canon, s),
            "diff_test": _step_snapshot(diff, s),
        }
    return out


def _interpret_mechanism(
    div: dict[str, Any], step1: dict[str, Any]
) -> dict[str, str]:
    """Map evidence pattern to most likely hypothesis (H1-H5)."""
    notes: list[str] = []

    # Step-1 attn_diff: if non-zero, the no-grad block has produced
    # different attention_weights even at parameter-equal init. That points
    # to numerical drift (H3).
    s1_attn_max = step1.get("attn_diff_max", None)
    s1_attended_max = step1.get("attended_diff_max", None)
    if s1_attended_max is not None and s1_attended_max == 0.0:
        notes.append(
            "Step-1 ATTENDED (the SDPA fast-path output that feeds the head) "
            "is BIT-EXACT identical between canonical and diff-test. The "
            "extra no-grad einsum block does NOT change forward semantics."
        )
    if s1_attn_max is not None and s1_attn_max > 0.0:
        notes.append(
            f"Step-1 attention_weights tensor differs by max={s1_attn_max:.3e}. "
            "The diff-test config returns attention_weights via the explicit "
            "no_grad einsum+softmax block while canonical returns None. This "
            "tensor is detached and not used for loss, so it cannot change "
            "the gradient directly."
        )

    pre_rng = div.get("first_step_pre_cuda_rng_diff")
    post_rng = div.get("first_step_post_cuda_rng_diff")
    loss_div = div.get("first_step_loss_diff")
    grad_div = div.get("first_step_grad_norm_encoder_diff")
    pred_div = div.get("first_step_pred_mean_diff")
    param_div = div.get("first_step_post_param_diff")

    # H2: RNG state advance.
    if pre_rng == 1 and post_rng == 0:
        notes.append(
            "PRE-step-1 CUDA RNG hashes are IDENTICAL (so post-step-0 didn't "
            "advance global RNG yet differently). "
            f"POST-step-0 CUDA RNG diverges first at step {post_rng}. "
            "This is H2 evidence: the no-grad einsum/softmax block DOES "
            "advance the global CUDA generator (e.g. via temporary alloc on "
            "the default stream), making downstream dropout draws differ."
        )
    elif post_rng is not None and post_rng <= 1:
        notes.append(
            f"POST-step-{post_rng} CUDA RNG diverges. "
            "Consistent with H2 (RNG advance from no_grad block) — "
            "downstream dropout uses different draws."
        )

    # H3: bf16 numerical drift (would manifest as IMMEDIATE step-1 loss / grad
    # divergence even WITHOUT RNG divergence).
    if (
        post_rng is not None and post_rng > 0
        and grad_div is not None and grad_div.get("step", 999) == 0
    ):
        notes.append(
            "Step-0 encoder grad-norm diverges before any RNG divergence. "
            "Consistent with H3 (numerical drift from .float() softmax cast)."
        )

    # H1: kernel selection drift would show as cuBLAS heuristic-driven
    # numerical drift in matmul outputs even without RNG advance.
    if (
        s1_attended_max is not None and s1_attended_max > 0.0 and s1_attended_max < 1e-3
        and pre_rng is None
    ):
        notes.append(
            "Step-1 attended differs by tiny non-zero amount with no RNG "
            "divergence. Could be H1 (kernel heuristic drift) or H3 "
            "(intra-step numerical noise in fused attention)."
        )

    # H4: allocator pool effects would manifest as slow drift, not
    # immediate divergence, and would track param checksum diverging.
    if param_div is not None and param_div > 1 and post_rng is None:
        notes.append(
            f"Param checksum diverges at step {param_div} but RNG hashes "
            "match throughout. Suggests H4 (allocator-pool / kernel-heuristic "
            "drift accumulating)."
        )

    # H5: backward-pass interaction would NOT change forward but would change
    # gradient values directly (so loss step-0 same, grad step-0 different).
    if (
        loss_div is not None and loss_div.get("step", 0) > 0
        and grad_div is not None and grad_div.get("step", 999) == 0
    ):
        notes.append(
            "Step-0 loss matches but step-0 encoder grad diverges. "
            "Consistent with H5 (autograd bookkeeping interaction)."
        )

    if not notes:
        notes.append(
            "No clear divergence pattern detected — trajectories may be "
            "identical (or run too short to diverge)."
        )

    # Pick a most-likely hypothesis label by simple heuristics.
    # Step-0 grad norm noise + identical post-step RNG + identical attended
    # is the signature of "non-determinic backward kernel + GPU state side-channel"
    # (H1 or H4) — the diff_test extra no_grad block changes allocator/launch
    # context such that SDPA backward (which is explicitly non-deterministic per
    # the FlashAttention warning) takes a different reduction order.
    if pre_rng == 1 or (post_rng is not None and post_rng <= 1):
        hypothesis_id = "H2"
        hypothesis_label = "H2 (RNG state advance from no_grad block) — most likely"
    elif s1_attended_max is not None and s1_attended_max > 0.0:
        hypothesis_id = "H3"
        hypothesis_label = "H3 (bf16 numerical drift) — most likely"
    elif (
        s1_attended_max is not None and s1_attended_max == 0.0
        and (post_rng is None)
        and param_div == 0
    ):
        # Attended bit-exact identical, RNG hashes match throughout, BUT params
        # diverge after step 0 → encoder grad differs at machine epsilon → the
        # non-deterministic SDPA backward kernel produced different output. The
        # diff_test config's only ADDITIONAL gpu work is the no_grad einsum +
        # softmax block, which alters allocator pool / kernel-launch context
        # in a way that changes which non-deterministic backward path is taken.
        hypothesis_id = "H1_H4"
        hypothesis_label = (
            "H1/H4 (non-deterministic SDPA backward + GPU state side-channel "
            "from no_grad block) — most likely"
        )
    elif param_div is not None and param_div > 1:
        hypothesis_id = "H1_H4_possible"
        hypothesis_label = "H1 / H4 (kernel heuristic / allocator drift) — possible"
    else:
        hypothesis_id = "unclear"
        hypothesis_label = "Unclear from this trajectory; need longer run or finer probe"

    return {
        "likely_hypothesis_id": hypothesis_id,
        "likely_hypothesis": hypothesis_label,
        "evidence_notes": notes,
    }


def main(args: argparse.Namespace) -> None:
    canon_dir = Path(args.canonical_dir)
    diff_dir = Path(args.diff_test_dir)

    canon_records = _load_jsonl(canon_dir / "trajectory.jsonl")
    diff_records = _load_jsonl(diff_dir / "trajectory.jsonl")
    canon_post = _post_step_records(canon_records)
    diff_post = _post_step_records(diff_records)

    div = _find_first_divergence(canon_post, diff_post)
    snap = _compare_at_steps(
        canon_post, diff_post, steps=[0, 1, 9, 99, 999]
    )
    step1 = _compare_step1_tensors(
        canon_dir / "step1_tensors.pt", diff_dir / "step1_tensors.pt"
    )

    interp = _interpret_mechanism(div, step1)

    report = {
        "n_steps_canonical": len(canon_post),
        "n_steps_diff_test": len(diff_post),
        "first_divergence": div,
        "step_snapshots": snap,
        "step1_bit_exact_diff": step1,
        "interpretation": interp,
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Wrote %s", out_path)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--canonical-dir", required=True,
        help="Directory containing canonical trajectory.jsonl + step1_tensors.pt",
    )
    p.add_argument(
        "--diff-test-dir", required=True,
        help="Directory containing diff_test trajectory.jsonl + step1_tensors.pt",
    )
    p.add_argument(
        "--output-path",
        default="outputs/canonical/p5_diff_test_mechanism/comparison_report.json",
    )
    main(p.parse_args())
