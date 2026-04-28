"""Compute KSG-MI bootstrap-derived SE + flag the upward bias for the paper.

Background
----------
The KSG conditional-MI estimator under bootstrap-with-replacement has a
well-known UPWARD bias: duplicate points in the resampled set have
distance 0, which inflates the k-NN counts that the KSG digamma sum is
computed from. Empirical evidence in our run: bootstrap median is
~0.28-0.30 nats above observed CMI for every CT (consistent across CTs).

Naive percentile CIs are therefore biased upward and should NOT be
reported. The Efron 1987 basic-bootstrap reflection (``2·obs - q``)
OVER-corrects: applied to our data it yields negative-CI bounds for
multiple CTs (impossible since CMI ≥ 0). The reflection assumes the
bootstrap distribution captures variability symmetrically around the
true value, which fails when bias dominates variance.

What this script does instead
-----------------------------
1. Treats observed CMI as the point estimate (KSG on the original sample
   IS unbiased for the population MI — only the resampled bootstrap is
   biased).
2. Derives a bootstrap-based SE from the percentile width via the normal
   approximation: SE ≈ (q_97.5 − q_2.5) / (2 · z_{0.975}) ≈ width / 3.92.
   This is valid for variance estimation even when the bootstrap location
   is biased (variance corrections survive shifts).
3. Reports the bias estimate (``boot_median − obs``) as a diagnostic.
4. Does NOT report a bias-corrected CI. The paper text should cite
   observed CMI + SE and flag the upward-biased CI as a known limitation
   of bootstrap-with-replacement for non-parametric MI estimators.

Reads
-----
``outputs/canonical/interpretability/ct_ranking_nulls/cmi_bootstrap_ci.json``

Writes
------
``outputs/canonical/interpretability/ct_ranking_nulls/cmi_bootstrap_ci_se.json``
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


_Z_975 = 1.959963984540054  # qnorm(0.975); precise for the SE conversion


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--in-path",
        default="outputs/canonical/interpretability/ct_ranking_nulls/cmi_bootstrap_ci.json",
    )
    p.add_argument(
        "--out-path",
        default="outputs/canonical/interpretability/ct_ranking_nulls/cmi_bootstrap_ci_se.json",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    if not in_path.exists():
        logger.error("input not found: %s", in_path)
        return 1
    d = json.loads(in_path.read_text())
    out: dict = {
        "n_boot": d.get("n_boot"),
        "n_subjects": d.get("n_subjects"),
        "method": "se_from_percentile_width_normal_approximation",
        "note": (
            "KSG-MI bootstrap is upward-biased due to duplicate points under "
            "resample-with-replacement. We DO NOT report a bias-corrected CI "
            "(basic-bootstrap reflection over-corrects, producing impossible "
            "negative-CMI bounds for several CTs). Instead we report observed "
            "CMI + SE derived from bootstrap percentile width "
            "(SE ≈ (q_97.5 − q_2.5) / 3.92). Variance estimation survives "
            "additive location bias. Cite bias_estimate to flag the limitation."
        ),
        "per_ct": {},
    }
    for ct, v in d.get("per_ct", {}).items():
        obs = float(v["observed_cmi"])
        q_lo = float(v["ci_2_5"])
        q_hi = float(v["ci_97_5"])
        boot_median = float(v["ci_50"])
        bias = boot_median - obs
        # SE from the percentile width assuming approximately Gaussian
        # bootstrap distribution shape. Robust to additive bias.
        se = (q_hi - q_lo) / (2.0 * _Z_975)
        out["per_ct"][ct] = {
            "observed_cmi": obs,
            "se_from_bootstrap_width": se,
            "ci_95_normal_approx_lo": obs - _Z_975 * se,
            "ci_95_normal_approx_hi": obs + _Z_975 * se,
            "naive_percentile_ci_lo": q_lo,
            "naive_percentile_ci_hi": q_hi,
            "naive_percentile_median": boot_median,
            "bias_estimate_nats": bias,
            "n_valid_boots": v.get("n_valid_boots"),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s (%d CTs)", out_path, len(out["per_ct"]))

    print("\nTop 10 CTs by observed CMI (point estimate ± SE from bootstrap width):")
    print("(Naive percentile median shown to flag upward bias.)")
    items = sorted(
        out["per_ct"].items(), key=lambda kv: -kv[1]["observed_cmi"],
    )
    for ct, v in items[:10]:
        se = v["se_from_bootstrap_width"]
        bias = v["bias_estimate_nats"]
        print(
            f"  {ct:42s}  obs={v['observed_cmi']:.4f} ± {se:.4f}  "
            f"naive_boot_median={v['naive_percentile_median']:.4f} "
            f"(bias={bias:+.4f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
