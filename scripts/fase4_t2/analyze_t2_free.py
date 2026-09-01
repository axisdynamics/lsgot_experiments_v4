#!/usr/bin/env python3
"""
T2 — análisis de trayectoria libre (v̂, PR, RQA, t=0) para las réplicas
axis_pec_only_v2, automata_neutro_v2, + witness_soul_md (exploratorio),
comparado contra las condiciones originales del panel limpio.

Datos: scripts/perturbation/results_t2/trajectories/*_baseline_embeddings.npz
(las mismas trayectorias que alimentan H4_rev — capa final, N<=256).
"""
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from tier0_metrics import compute_trajectory_tier0, persona_vector, project_trajectory, _agg  # noqa: E402
from statistical_tests import GeometricStatisticalTests  # noqa: E402

T2_DIR = Path(
    "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
    "gemma4_31b_combined/perturbation/results_t2/trajectories"
)
ORIG_DIR = Path(
    "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
    "gemma4_31b_combined/results_local/sia_extended_v5"
)
V_HAT_PATH = Path(__file__).parent.parent / "perturbation" / "v_identidad.npy"

T2_GROUPS = ["axis_pec_only_v2", "automata_neutro_v2", "witness_soul_md"]
ORIG_GROUPS = ["axis_pec_only", "automata_neutro", "vanilla", "axis"]

# Pares T2 (réplica vs su original) + comparaciones cruzadas de interés
PAIRS = [
    ("axis_pec_only_v2", "automata_neutro_v2"),       # la disociación central, en la réplica
    ("axis_pec_only_v2", "axis_pec_only"),             # v2 vs v1 — ¿mismo perfil?
    ("automata_neutro_v2", "automata_neutro"),         # v2 vs v1 — ¿mismo perfil?
    ("axis_pec_only_v2", "vanilla"),
    ("automata_neutro_v2", "vanilla"),
    ("witness_soul_md", "axis_pec_only"),
    ("witness_soul_md", "automata_neutro"),
    ("witness_soul_md", "vanilla"),
]


def load_t2(name):
    d = np.load(T2_DIR / f"{name}_baseline_embeddings.npz", allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int)


def load_orig(name):
    d = np.load(ORIG_DIR / f"{name}_embeddings.npz", allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int)


def main():
    v_hat = np.load(V_HAT_PATH)
    tester = GeometricStatisticalTests(n_permutations=1000, random_seed=42)

    emb, lengths = {}, {}
    for g in T2_GROUPS:
        emb[g], lengths[g] = load_t2(g)
    for g in ORIG_GROUPS:
        emb[g], lengths[g] = load_orig(g)

    print(f"{'grupo':22s} {'n':>3s} {'hurst':>8s} {'determ':>8s} {'PR':>8s} {'proj_v̂':>10s} {'proj_t0':>10s}")
    per_group_metrics = {}
    per_group_t0 = {}
    for g in T2_GROUPS + ORIG_GROUPS:
        rows = []
        t0_vals = []
        for i, L in enumerate(lengths[g]):
            if L < 3:
                continue
            traj = emb[g][i, :L]
            m = compute_trajectory_tier0(traj, v_identidad=v_hat)
            rows.append(m)
            proj_series = project_trajectory(traj, v_hat)
            t0_vals.append(float(proj_series[0]))
        per_group_metrics[g] = rows
        per_group_t0[g] = t0_vals
        hurst_s = _agg([r.hurst for r in rows])
        det_s = _agg([r.determinism for r in rows])
        pr_s = _agg([r.participation_ratio for r in rows])
        proj_s = _agg([r.identity_projection for r in rows])
        t0_mean = float(np.mean(t0_vals)) if t0_vals else float("nan")
        t0_sd = float(np.std(t0_vals)) if t0_vals else float("nan")
        print(f"{g:22s} {len(rows):3d} {hurst_s['mean']:+8.4f} {det_s['mean']:8.4f} "
              f"{pr_s['mean']:8.3f} {proj_s['mean']:+10.4f} {t0_mean:+10.4f}")

    print("\n=== Comparaciones clave (permutación, PR + proj media + proj t=0) ===")
    results = {}
    for a, b in PAIRS:
        pr_a = [r.participation_ratio for r in per_group_metrics[a]]
        pr_b = [r.participation_ratio for r in per_group_metrics[b]]
        proj_a = [r.identity_projection for r in per_group_metrics[a]]
        proj_b = [r.identity_projection for r in per_group_metrics[b]]
        det_a = [r.determinism for r in per_group_metrics[a]]
        det_b = [r.determinism for r in per_group_metrics[b]]
        t0_a = per_group_t0[a]
        t0_b = per_group_t0[b]

        res_pr = tester.full_comparison(pr_a, pr_b, a, b)
        res_proj = tester.full_comparison(proj_a, proj_b, a, b)
        res_det = tester.full_comparison(det_a, det_b, a, b)
        res_t0 = tester.full_comparison(t0_a, t0_b, a, b)

        key = f"{a}_vs_{b}"
        results[key] = {
            "PR": {"d": res_pr["cohens_d"], "p": res_pr["permutation_test"]["p_value"]},
            "proj_mean": {"d": res_proj["cohens_d"], "p": res_proj["permutation_test"]["p_value"]},
            "determinism": {"d": res_det["cohens_d"], "p": res_det["permutation_test"]["p_value"]},
            "proj_t0": {"d": res_t0["cohens_d"], "p": res_t0["permutation_test"]["p_value"]},
        }
        print(f"  {a:20s} vs {b:20s}  "
              f"PR d={res_pr['cohens_d']:+6.2f}(p={res_pr['permutation_test']['p_value']:.4f})  "
              f"proj d={res_proj['cohens_d']:+6.2f}(p={res_proj['permutation_test']['p_value']:.4f})  "
              f"det d={res_det['cohens_d']:+6.2f}(p={res_det['permutation_test']['p_value']:.4f})  "
              f"t0 d={res_t0['cohens_d']:+6.2f}(p={res_t0['permutation_test']['p_value']:.4f})")

    out = {"pairs": results,
           "summary": {g: {"PR_mean": float(np.mean([r.participation_ratio for r in per_group_metrics[g]])),
                            "proj_mean": float(np.mean([r.identity_projection for r in per_group_metrics[g]])),
                            "determinism_mean": float(np.mean([r.determinism for r in per_group_metrics[g]])),
                            "t0_mean": float(np.mean(per_group_t0[g])),
                            "n": len(per_group_metrics[g])}
                       for g in T2_GROUPS + ORIG_GROUPS}}
    out_path = Path(__file__).parent / "t2_free_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
