#!/usr/bin/env python3
"""
T2 — tercera señal de identidad (E-H2: dinámica temporal de v̂) para
witness_soul_md, axis_pec_only_v2 y automata_neutro_v2, mismo método que
scripts/fase0/analyze_EH2_serie_temporal.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "fase0"))
from tier0_metrics import project_trajectory  # noqa: E402
from analyze_EH2_serie_temporal import series_metrics, agg  # noqa: E402
from statistical_tests import GeometricStatisticalTests  # noqa: E402

T2_DIR = Path(
    "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
    "gemma4_31b_combined/perturbation/results_t2/trajectories"
)
V_HAT_PATH = Path(__file__).parent.parent / "perturbation" / "v_identidad.npy"
GROUPS = ["axis_pec_only_v2", "automata_neutro_v2", "witness_soul_md",
          "soul_jarvis", "soul_elena_financial", "soul_solidity_auditor"]

PAIRS = [
    ("witness_soul_md", "axis_pec_only_v2"),
    ("witness_soul_md", "automata_neutro_v2"),
    ("soul_jarvis", "axis_pec_only_v2"),
    ("soul_jarvis", "automata_neutro_v2"),
    ("soul_elena_financial", "axis_pec_only_v2"),
    ("soul_elena_financial", "automata_neutro_v2"),
    ("soul_solidity_auditor", "axis_pec_only_v2"),
    ("soul_solidity_auditor", "automata_neutro_v2"),
]


def load(name):
    d = np.load(T2_DIR / f"{name}_baseline_embeddings.npz", allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int)


def main():
    v_hat = np.load(V_HAT_PATH)
    tester = GeometricStatisticalTests(n_permutations=1000, random_seed=42)
    print(f"{'grupo':22s} {'autocorr':>9s} {'frac(p>0)':>10s} {'ráfaga_media':>13s} {'ráfaga_máx':>11s} {'mean_p':>8s}")
    per_group_rows = {}
    for g in GROUPS:
        emb, lengths = load(g)
        rows = []
        for i, L in enumerate(lengths):
            if L < 8:
                continue
            p = project_trajectory(emb[i, :L], v_hat)
            m = series_metrics(p)
            if m is not None:
                rows.append(m)
        per_group_rows[g] = rows
        autocorr = agg(rows, "autocorr_lag1")[0]
        frac = agg(rows, "frac_positive")[0]
        mburst = agg(rows, "mean_burst_len")[0]
        xburst = agg(rows, "max_burst_len")[0]
        meanp = agg(rows, "mean_p")[0]
        print(f"{g:22s} {autocorr['mean']:9.3f} {frac['mean']:10.3f} "
              f"{mburst['mean']:13.2f} {xburst['mean']:11.1f} {meanp['mean']:+8.3f}")

    print("\n=== Comparaciones (permutación, ráfaga media + frac_positive) ===")
    pairs_out = {}
    summary = {}
    for g in GROUPS:
        summary[g] = {
            "autocorr_lag1": agg(per_group_rows[g], "autocorr_lag1")[0],
            "frac_positive": agg(per_group_rows[g], "frac_positive")[0],
            "mean_burst_len": agg(per_group_rows[g], "mean_burst_len")[0],
            "max_burst_len": agg(per_group_rows[g], "max_burst_len")[0],
            "mean_p": agg(per_group_rows[g], "mean_p")[0],
        }
    for a, b in PAIRS:
        burst_a = agg(per_group_rows[a], "mean_burst_len")[1]
        burst_b = agg(per_group_rows[b], "mean_burst_len")[1]
        frac_a = agg(per_group_rows[a], "frac_positive")[1]
        frac_b = agg(per_group_rows[b], "frac_positive")[1]
        res_burst = tester.full_comparison(burst_a, burst_b, a, b)
        res_frac = tester.full_comparison(frac_a, frac_b, a, b)
        print(f"  {a:22s} vs {b:20s}  "
              f"ráfaga d={res_burst['cohens_d']:+6.2f}(p={res_burst['permutation_test']['p_value']:.4f})  "
              f"frac d={res_frac['cohens_d']:+6.2f}(p={res_frac['permutation_test']['p_value']:.4f})")
        pairs_out[f"{a}_vs_{b}"] = {
            "mean_burst_len": {"d": res_burst["cohens_d"], "p": res_burst["permutation_test"]["p_value"]},
            "frac_positive": {"d": res_frac["cohens_d"], "p": res_frac["permutation_test"]["p_value"]},
        }

    out = {"summary": summary, "pairs": pairs_out}
    out_path = Path(__file__).parent / "t2_eh2_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
