#!/usr/bin/env python3
"""
T2 — ¿witness_soul_md recupera por la MISMA ruta que su propia línea base,
o solo converge a un punto parecido por otro camino? Fréchet normalizado
con test de permutación (igual método que §3.6 del paper), comparando
witness_soul_md contra axis_pec_only_v2 (identidad) y automata_neutro_v2
(restricción) en vez de solo reportar la media cruda.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from statistical_tests import GeometricStatisticalTests  # noqa: E402

T2_DIR = Path(
    "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
    "gemma4_31b_combined/perturbation/results_t2/trajectories"
)
T_INJ_VALUES = [50, 128, 200]
GROUPS = ["axis_pec_only_v2", "automata_neutro_v2", "witness_soul_md"]


def load_traj_npz(name):
    d = np.load(T2_DIR / f"{name}.npz", allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int), d["prompt_ids"]


def paired_indices(base_ids, pert_ids):
    base_pos = {pid: i for i, pid in enumerate(base_ids)}
    return [(base_pos[pid], j) for j, pid in enumerate(pert_ids) if pid in base_pos]


def discrete_frechet(P, Q):
    n, m = len(P), len(Q)
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)
    ca = np.empty((n, m), dtype=np.float64)
    ca[0, 0] = D[0, 0]
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], D[0, j])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], D[i, 0])
        row_prev, row, d_row = ca[i - 1], ca[i], D[i]
        prev_val = row[0]
        for j in range(1, m):
            v = max(min(row_prev[j], row_prev[j - 1], prev_val), d_row[j])
            row[j] = v
            prev_val = v
    return float(ca[n - 1, m - 1])


def mean_step_velocity(traj):
    return float(np.mean(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


def frechet_norm_per_prompt(group, t_inj):
    base_emb, base_len, base_ids = load_traj_npz(f"{group}_baseline_embeddings")
    pert_emb, pert_len, pert_ids = load_traj_npz(f"{group}_perturbed_t{t_inj}_embeddings")
    pairs = paired_indices(base_ids, pert_ids)
    vals = []
    for ib, ip in pairs:
        Lb, Lp = base_len[ib], pert_len[ip]
        if t_inj >= Lb or t_inj >= Lp:
            continue
        P = base_emb[ib, t_inj:Lb]
        Q = pert_emb[ip, t_inj:Lp]
        if len(P) < 2 or len(Q) < 2:
            continue
        d = discrete_frechet(P, Q)
        v = mean_step_velocity(P)
        if v and v > 1e-8:
            vals.append(d / v)
    return vals


def main():
    tester = GeometricStatisticalTests(n_permutations=1000, random_seed=42)
    per_group = {g: {} for g in GROUPS}
    for g in GROUPS:
        for t in T_INJ_VALUES:
            per_group[g][t] = frechet_norm_per_prompt(g, t)

    print(f"{'grupo':22s}" + "".join(f"  t={t:3d} (media±sd, n)" for t in T_INJ_VALUES))
    for g in GROUPS:
        line = f"{g:22s}"
        for t in T_INJ_VALUES:
            v = per_group[g][t]
            line += f"  {np.mean(v):.3f}±{np.std(v):.3f}(n={len(v):2d})    "
        print(line)

    print("\n=== witness_soul_md vs identidad y vs restricción (permutación, Fréchet normalizado) ===")
    for a, b in [("witness_soul_md", "axis_pec_only_v2"), ("witness_soul_md", "automata_neutro_v2")]:
        for t in T_INJ_VALUES:
            res = tester.full_comparison(per_group[a][t], per_group[b][t], a, b)
            d = res["cohens_d"]
            p = res["permutation_test"]["p_value"]
            print(f"  {a} vs {b}  t_inj={t:3d}  d={d:+.2f} ({res['effect_size_interpretation']})  p={p:.4f}")


if __name__ == "__main__":
    main()
