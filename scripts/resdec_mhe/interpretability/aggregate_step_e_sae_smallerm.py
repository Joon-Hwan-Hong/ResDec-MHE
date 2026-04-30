#!/usr/bin/env python
"""Aggregate Step E SAE smaller-m sweep (180 configs).

For each config (arch, layer, expansion m, K, seed): compute
    - n_features = m * K
    - n_interpretable (features passing canonical mw_p < 1e-4 AND 0.05 ≤ fraction_active ≤ 0.5 filter)
    - n_splatter_dominant (ct_dominance ≥ 0.3 with top_cell_type == "Splatter")
    - max ct_dominance overall
    - mean fve (full)
    - mean l0 (full)
    - dead_fraction (full)

Output:
    outputs/canonical/sae/stability_smaller_m/aggregate_summary.json
    outputs/canonical/sae/stability_smaller_m/aggregate_summary.md
"""
from __future__ import annotations
import json
from pathlib import Path
import statistics

WT = Path(__file__).resolve().parents[3]
ROOT = WT / "outputs/canonical/sae/stability_smaller_m"
MW_P_THRESHOLD = 1e-4
FA_LOW, FA_HIGH = 0.05, 0.5
CT_DOMINANCE_THRESHOLD = 0.3


def _is_interpretable(feat: dict) -> bool:
    """Canonical filter from the 1/323 work: MW p < 1e-4 (cognition or pathology)
    AND fraction_active in [0.05, 0.5]."""
    mw_c = feat.get("mw_p_cognition")
    mw_p = feat.get("mw_p_pathology")
    if (mw_c is None or mw_c >= MW_P_THRESHOLD) and (mw_p is None or mw_p >= MW_P_THRESHOLD):
        return False
    fa = feat.get("fraction_active")
    if fa is None or not (FA_LOW <= fa <= FA_HIGH):
        return False
    return True


def _aggregate_config_dir(d: Path) -> dict | None:
    """Returns aggregate stats for one (arch, layer, exp, k, seed) directory."""
    fr_path = d / "feature_report.json"
    rm_path = d / "reconstruction_metrics.json"
    if not fr_path.exists() or not rm_path.exists():
        return None
    features = json.loads(fr_path.read_text())
    rm = json.loads(rm_path.read_text())

    n_total = len(features)
    interpretable = [f for f in features if _is_interpretable(f)]
    n_interpretable = len(interpretable)

    # max_ct_dominance over ALL features (not interpretable-filtered) — the
    # "overall" qualifier in the field name is literal. Useful to diagnose
    # whether the interpretability filter strips CT-dominant features.
    max_ct_dominance_overall = 0.0
    for f in features:
        cd = f.get("ct_dominance", 0.0) or 0.0
        if cd > max_ct_dominance_overall:
            max_ct_dominance_overall = cd

    # Count Splatter-dominant features (top CT == Splatter, ct_dominance >= threshold);
    # interpretable-only because Splatter-dominance is a property of the polysemantic
    # claim about meaningful features.
    n_splatter_dominant = 0
    splatter_max_dominance = 0.0
    for f in interpretable:
        cd = f.get("ct_dominance", 0.0) or 0.0
        top_cts = f.get("top_cell_types") or []
        if top_cts:
            top_ct = top_cts[0] if isinstance(top_cts[0], str) else (top_cts[0].get("cell_type") if isinstance(top_cts[0], dict) else None)
            if top_ct == "Splatter":
                if cd >= CT_DOMINANCE_THRESHOLD:
                    n_splatter_dominant += 1
                if cd > splatter_max_dominance:
                    splatter_max_dominance = cd

    cfg = rm.get("config", {})
    full = rm.get("full", {})
    return {
        "config": {
            "architecture": cfg.get("architecture"),
            "layer": rm.get("layer"),
            "expansion": cfg.get("expansion"),
            "k": cfg.get("k"),
            "seed": cfg.get("seed"),
        },
        "n_features_total": n_total,
        "n_interpretable": n_interpretable,
        "frac_interpretable": round(n_interpretable / max(n_total, 1), 4),
        "n_splatter_dominant": n_splatter_dominant,
        "splatter_dominance_rate": round(n_splatter_dominant / max(n_interpretable, 1), 4),
        "max_ct_dominance_overall": round(max_ct_dominance_overall, 4),
        "splatter_max_dominance": round(splatter_max_dominance, 4),
        "fve": round(full.get("fve", 0.0), 4),
        "l0_mean": round(full.get("l0_mean", 0.0), 1),
        "dead_fraction": round(full.get("dead_fraction", 0.0), 4),
    }


def main():
    rows = []
    archs = sorted([d.name for d in ROOT.iterdir() if d.is_dir()])
    for arch in archs:
        arch_dir = ROOT / arch
        for layer_dir in sorted([d for d in arch_dir.iterdir() if d.is_dir()]):
            for cfg_dir in sorted([d for d in layer_dir.iterdir() if d.is_dir()]):
                agg = _aggregate_config_dir(cfg_dir)
                if agg:
                    rows.append(agg)

    print(f"aggregated {len(rows)} configs")

    # Group across seeds: (arch, layer, expansion, k) → mean ± std of metrics
    grouped: dict = {}
    for r in rows:
        c = r["config"]
        key = (c["architecture"], c["layer"], c["expansion"], c["k"])
        grouped.setdefault(key, []).append(r)

    grouped_summary = []
    # Sort tuples (arch:str, layer:str, expansion:int, k:int) lexicographically:
    # alphabetical by arch/layer, numeric within each str-prefix group.
    for key, seed_rows in sorted(grouped.items()):
        n_seeds = len(seed_rows)

        def _ms(field):
            vs = [r[field] for r in seed_rows]
            return float(statistics.mean(vs)), float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0

        n_int_m, n_int_s = _ms("n_interpretable")
        spl_dom_m, spl_dom_s = _ms("n_splatter_dominant")
        spl_dom_max_m, spl_dom_max_s = _ms("splatter_max_dominance")
        fve_m, fve_s = _ms("fve")
        l0_m, l0_s = _ms("l0_mean")
        dead_m, _ = _ms("dead_fraction")
        max_cd_m, _ = _ms("max_ct_dominance_overall")

        grouped_summary.append({
            "architecture": key[0],
            "layer": key[1],
            "expansion": key[2],
            "k": key[3],
            "n_features": key[2] * key[3],
            "n_seeds": n_seeds,
            "n_interpretable_mean": round(n_int_m, 1),
            "n_interpretable_std": round(n_int_s, 1),
            "n_splatter_dominant_mean": round(spl_dom_m, 2),
            "n_splatter_dominant_std": round(spl_dom_s, 2),
            "splatter_max_dominance_mean": round(spl_dom_max_m, 4),
            "max_ct_dominance_overall_mean": round(max_cd_m, 4),
            "fve_mean": round(fve_m, 4),
            "l0_mean_mean": round(l0_m, 1),
            "dead_fraction_mean": round(dead_m, 4),
        })

    out_json = ROOT / "aggregate_summary.json"
    out_json.write_text(json.dumps({
        "n_configs": len(rows),
        "n_grouped": len(grouped_summary),
        "filter": {
            "mw_p_threshold": MW_P_THRESHOLD,
            "fraction_active_range": [FA_LOW, FA_HIGH],
            "ct_dominance_threshold_for_dominant": CT_DOMINANCE_THRESHOLD,
        },
        "per_config": rows,
        "grouped_across_seeds": grouped_summary,
    }, indent=2))
    print(f"Wrote {out_json}")

    # Build markdown summary
    md = ["# SAE smaller-m sweep — Step E aggregate (180 configs)\n"]
    md.append(f"Filter: MW p < {MW_P_THRESHOLD:.0e} (cognition or pathology) AND fraction_active in [{FA_LOW}, {FA_HIGH}].")
    md.append(f"Splatter-dominant = top_cell_type == Splatter AND ct_dominance >= {CT_DOMINANCE_THRESHOLD}.\n")
    md.append("## Grouped across seeds (3 seeds per config)\n")
    md.append("| Arch | Layer | exp m | K | n_feat | n_seeds | n_interp ± std | n_splatter_dom ± std | spl_max_dom | max_cd | fve | l0_mean | dead |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in grouped_summary:
        md.append(
            f"| {r['architecture']} | {r['layer']} | {r['expansion']} | {r['k']} | {r['n_features']} | "
            f"{r['n_seeds']} | {r['n_interpretable_mean']:.1f} ± {r['n_interpretable_std']:.1f} | "
            f"{r['n_splatter_dominant_mean']:.2f} ± {r['n_splatter_dominant_std']:.2f} | "
            f"{r['splatter_max_dominance_mean']:.3f} | "
            f"{r['max_ct_dominance_overall_mean']:.3f} | "
            f"{r['fve_mean']:.3f} | {r['l0_mean_mean']:.1f} | {r['dead_fraction_mean']:.3f} |"
        )

    out_md = ROOT / "aggregate_summary.md"
    out_md.write_text("\n".join(md))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
