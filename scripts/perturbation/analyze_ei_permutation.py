#!/usr/bin/env python3
"""
Permutation test sobre los resultados de E-I (perturbación direccional,
along vs orthogonal a v_identidad) — cierra la lectura preliminar de
EI_RUNBOOK.md con significancia formal, mismo método (mean_difference,
n_permutations=1000) que TIER0_REPORT.md / EE_EH_WINDOW_REPORT.md.

Uso:
  python analyze_ei_permutation.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from statistical_tests import GeometricStatisticalTests  # noqa: E402

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results" / "perturbation_ei_L30_medium"
DETAILS_PATH = RESULTS_DIR / "details.json"

T_INJ_VALUES = [50, 128, 200]
GROUPS = ["axis", "axis_pec_only"]
METRICS = ["tau_tokens", "displacement_l2", "sampen_delta", "w1_norms_post"]


def extract(details, group, direction, t_inj, metric):
    lst = details[group][direction][str(t_inj)]
    return [x[metric] for x in lst if x.get(metric) is not None]


def main():
    with open(DETAILS_PATH) as f:
        data = json.load(f)
    details = data["details"]

    stat = GeometricStatisticalTests(n_permutations=1000)
    out = {}

    for metric in METRICS:
        print("\n" + "=" * 78)
        print(f"E-I — permutation test sobre '{metric}' (along vs orthogonal)")
        print("=" * 78)
        print(f"{'grupo':16s} {'t_inj':>6s} {'mean_along':>11s} {'mean_orthog':>12s} "
              f"{'Δ':>8s} {'p':>8s} {'d':>7s} {'interp':>12s}")
        print("-" * 78)
        out[metric] = {}
        for group in GROUPS:
            out[metric][group] = {}
            for t_inj in T_INJ_VALUES:
                a = extract(details, group, "along", t_inj, metric)
                o = extract(details, group, "orthogonal", t_inj, metric)
                if len(a) < 2 or len(o) < 2:
                    print(f"{group:16s} {t_inj:6d}  [datos insuficientes]")
                    continue
                res = stat.full_comparison(a, o, "along", "orthogonal")
                perm = res["permutation_test"]
                out[metric][group][t_inj] = {
                    "n_along": len(a), "n_orthogonal": len(o),
                    "mean_along": res["mean_along"], "mean_orthogonal": res["mean_orthogonal"],
                    "delta": perm["observed_statistic"], "p_value": perm["p_value"],
                    "cohens_d": res["cohens_d"], "interpretation": res["effect_size_interpretation"],
                }
                print(f"{group:16s} {t_inj:6d} {res['mean_along']:11.2f} {res['mean_orthogonal']:12.2f} "
                      f"{perm['observed_statistic']:+8.2f} {perm['p_value']:8.4f} "
                      f"{res['cohens_d']:+7.3f} {res['effect_size_interpretation']:>12s}")

    out_path = RESULTS_DIR / "EI_PERMUTATION_REPORT.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
