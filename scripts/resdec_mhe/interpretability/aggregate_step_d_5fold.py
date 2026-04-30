#!/usr/bin/env python
"""Aggregate Step D 5-fold F1 counterfactual replication.

Loads:
    fold 0 from outputs/canonical/interpretability/counterfactuals_optimized_{rel,abs}_delta{0p5,0p3}/counterfactuals_fold0.json
    folds 1-4 from outputs/canonical/interpretability/counterfactuals_optimized_{rel,abs}_delta{0p5,0p3}_fold{N}/counterfactuals_fold{N}.json

Writes:
    outputs/canonical/interpretability/f1_5fold_summary.json
    outputs/canonical/interpretability/f1_5fold_summary.md

Computes per-fold × per-mode × per-delta × per-regime:
    - success rate (fraction of subjects that hit target within tol)
    - mean ± std n_steps_used
    - mean ± std l2_distance
    - mean ± std lambda_used
"""
from __future__ import annotations
import json
from pathlib import Path
import statistics

WT = Path(__file__).resolve().parents[3]
INTERP = WT / "outputs/canonical/interpretability"


def _path_for(fold: int, mode: str, delta_str: str) -> Path:
    if fold == 0:
        return INTERP / f"counterfactuals_optimized_{mode}_delta{delta_str}/counterfactuals_fold0.json"
    return INTERP / f"counterfactuals_optimized_{mode}_delta{delta_str}_fold{fold}/counterfactuals_fold{fold}.json"


def _summarize(rows: list[dict]) -> dict:
    """Per-regime aggregate."""
    out = {}
    for regime in ("resilient", "vulnerable"):
        sub = [r for r in rows if r.get("regime") == regime]
        if not sub:
            out[regime] = {"n": 0}
            continue
        succ = [r for r in sub if r.get("success")]
        steps = [r["n_steps_used"] for r in sub if r.get("n_steps_used") is not None]
        l2 = [r["l2_distance"] for r in sub if r.get("l2_distance") is not None]
        lam = [r["lambda_used"] for r in sub if r.get("lambda_used") is not None]

        def _ms(vs):
            if not vs:
                return None, None
            sd = float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0
            return float(statistics.mean(vs)), sd

        m_s, sd_s = _ms(steps)
        m_l, sd_l = _ms(l2)
        m_lam, sd_lam = _ms(lam)

        out[regime] = {
            "n": len(sub),
            "n_success": len(succ),
            "success_rate": round(len(succ) / len(sub), 3),
            "mean_steps": round(m_s, 1) if m_s is not None else None,
            "std_steps": round(sd_s, 1) if sd_s is not None else None,
            "mean_l2": round(m_l, 4) if m_l is not None else None,
            "std_l2": round(sd_l, 4) if sd_l is not None else None,
            "mean_lambda": round(m_lam, 1) if m_lam is not None else None,
            "std_lambda": round(sd_lam, 1) if sd_lam is not None else None,
        }
    return out


def main():
    out: dict = {"per_fold": {}, "across_folds": {}}

    folds = [0, 1, 2, 3, 4]
    modes = ("relative", "absolute")
    deltas = (("0.5", "0p5"), ("0.3", "0p3"))

    # Per-fold aggregation
    for fold in folds:
        out["per_fold"][f"fold_{fold}"] = {}
        for mode in modes:
            for delta, delta_str in deltas:
                p = _path_for(fold, mode, delta_str)
                if not p.exists():
                    out["per_fold"][f"fold_{fold}"][f"{mode}_delta{delta}"] = {"missing": True, "path": str(p)}
                    continue
                d = json.loads(p.read_text())
                rows = d.get("results", [])
                summary = _summarize(rows)
                summary["n_total"] = len(rows)
                summary["path"] = str(p.relative_to(WT))
                out["per_fold"][f"fold_{fold}"][f"{mode}_delta{delta}"] = summary

    # Across-folds aggregation: stack all folds for each (mode, delta)
    for mode in modes:
        for delta, delta_str in deltas:
            stacked = []
            for fold in folds:
                p = _path_for(fold, mode, delta_str)
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                stacked.extend(d.get("results", []))
            if not stacked:
                continue
            agg = _summarize(stacked)
            agg["n_total"] = len(stacked)
            agg["folds_included"] = [f for f in folds if _path_for(f, mode, delta_str).exists()]
            out["across_folds"][f"{mode}_delta{delta}"] = agg

    # Write JSON
    out_json = INTERP / "f1_5fold_summary.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_json}")

    # Build markdown table
    md = ["# F1 5-fold counterfactual replication summary\n"]
    md.append("Source: outputs/canonical/interpretability/counterfactuals_optimized_*_fold{0..4}/")
    md.append("")
    md.append("## Per-regime aggregate across all 5 folds")
    md.append("")
    md.append("| Mode | δ | Regime | n_total | Success rate | Mean ± std steps | Mean ± std L2 | Mean ± std λ |")
    md.append("|---|---|---|---|---|---|---|---|")
    for mode in modes:
        for delta, _ in deltas:
            key = f"{mode}_delta{delta}"
            cell = out["across_folds"].get(key, {})
            for regime in ("resilient", "vulnerable"):
                r = cell.get(regime, {})
                if not r or r.get("n") == 0:
                    continue
                md.append(
                    f"| {mode} | {delta} | {regime} | {r['n']} | "
                    f"{r['success_rate']:.0%} ({r['n_success']}/{r['n']}) | "
                    f"{r['mean_steps']:.1f} ± {r['std_steps']:.1f} | "
                    f"{r['mean_l2']:.3f} ± {r['std_l2']:.3f} | "
                    f"{r['mean_lambda']:.0f} ± {r['std_lambda']:.0f} |"
                )

    md.append("")
    md.append("## Per-fold × per-mode × per-δ × per-regime")
    md.append("")
    md.append("| Fold | Mode | δ | Regime | n | Success | Mean steps | Mean L2 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for fold in folds:
        fkey = f"fold_{fold}"
        if fkey not in out["per_fold"]:
            continue
        for mode in modes:
            for delta, _ in deltas:
                key = f"{mode}_delta{delta}"
                cell = out["per_fold"][fkey].get(key, {})
                if cell.get("missing"):
                    continue
                for regime in ("resilient", "vulnerable"):
                    r = cell.get(regime, {})
                    if not r or r.get("n") == 0 or r.get("mean_steps") is None or r.get("mean_l2") is None:
                        continue
                    md.append(
                        f"| {fold} | {mode} | {delta} | {regime} | {r['n']} | "
                        f"{r['n_success']}/{r['n']} | "
                        f"{r['mean_steps']:.1f} | "
                        f"{r['mean_l2']:.3f} |"
                    )

    out_md = INTERP / "f1_5fold_summary.md"
    out_md.write_text("\n".join(md))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
