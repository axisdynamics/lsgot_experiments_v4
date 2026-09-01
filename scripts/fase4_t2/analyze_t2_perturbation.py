#!/usr/bin/env python3
"""
T2 — análisis de perturbación (H4_rev) para axis_pec_only_v2,
automata_neutro_v2 y witness_soul_md (exploratorio): τ geométrico,
recovery_rate, τ_identidad/recovery_id, y Fréchet normalizado — mismos
parámetros exactos que la corrida original (capa final, L30 inyección,
sigma=medium=5.979170192466191, t_inj=[50,128,200]).

Compara contra los valores ya publicados de axis_pec_only/automata_neutro
(paper/lsgot_4.md §3.3, §3.5, §3.6; AUTOMATA_NEUTRO_REPORT.md).
"""
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "perturbation"))

from tier0_metrics import persona_vector, project_trajectory  # noqa: E402
from recovery_analyzer import compute_recovery  # noqa: E402

T2_DIR = Path(
    "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
    "gemma4_31b_combined/perturbation/results_t2/trajectories"
)
V_HAT_PATH = Path(__file__).parent.parent / "perturbation" / "v_identidad.npy"

T_INJ_VALUES = [50, 128, 200]
GROUPS = ["axis_pec_only_v2", "automata_neutro_v2", "witness_soul_md"]
TAU_MIN_WINDOW = 5
TAU_THRESHOLD = 0.95


def load_traj_npz(name):
    path = T2_DIR / f"{name}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int), d["prompt_ids"]


def paired_indices(base_ids, pert_ids):
    base_pos = {pid: i for i, pid in enumerate(base_ids)}
    return [(base_pos[pid], j) for j, pid in enumerate(pert_ids) if pid in base_pos]


def discrete_frechet(P, Q):
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return float("nan")
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)
    ca = np.empty((n, m), dtype=np.float64)
    ca[0, 0] = D[0, 0]
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], D[0, j])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], D[i, 0])
        row_prev = ca[i - 1]
        row = ca[i]
        d_row = D[i]
        prev_val = row[0]
        for j in range(1, m):
            v = max(min(row_prev[j], row_prev[j - 1], prev_val), d_row[j])
            row[j] = v
            prev_val = v
    return float(ca[n - 1, m - 1])


def mean_step_velocity(traj):
    if len(traj) < 2:
        return float("nan")
    return float(np.mean(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


def find_tau_identity(proj_post, proj_reference, threshold=TAU_THRESHOLD):
    if len(proj_post) < TAU_MIN_WINDOW:
        return None
    target = threshold * proj_reference
    cum = 0.0
    for w in range(1, len(proj_post) + 1):
        cum += proj_post[w - 1]
        mean_w = cum / w
        ok = (mean_w >= target) if proj_reference >= 0 else (mean_w <= target)
        if ok and w >= TAU_MIN_WINDOW:
            return w
    return None


def main():
    v_hat = np.load(V_HAT_PATH)
    out = {}

    for group in GROUPS:
        base = load_traj_npz(f"{group}_baseline_embeddings")
        if base is None:
            print(f"[skip] {group}: sin baseline")
            continue
        base_emb, base_len, base_ids = base
        out[group] = {}

        for t_inj in T_INJ_VALUES:
            pert = load_traj_npz(f"{group}_perturbed_t{t_inj}_embeddings")
            if pert is None:
                continue
            pert_emb, pert_len, pert_ids = pert
            pairs = paired_indices(base_ids, pert_ids)

            taus_geom, n_recover_geom = [], 0
            taus_id, n_recover_id = [], 0
            frechet_norm = []

            for ib, ip in pairs:
                Lb, Lp = base_len[ib], pert_len[ip]
                if t_inj >= Lb or t_inj >= Lp:
                    continue
                base_traj = base_emb[ib, :Lb]
                pert_traj = pert_emb[ip, :Lp]

                # tau geometrico + recovery_rate
                r = compute_recovery(base_traj, pert_traj, t_inj)
                if r.get("status") == "ok":
                    if r["tau_tokens"] is not None:
                        taus_geom.append(r["tau_tokens"])
                        n_recover_geom += 1

                # tau identidad
                proj_base_post = project_trajectory(base_traj[t_inj:], v_hat)
                proj_pert_post = project_trajectory(pert_traj[t_inj:], v_hat)
                if len(proj_base_post) >= TAU_MIN_WINDOW:
                    ref = float(np.mean(proj_base_post))
                    tau_id = find_tau_identity(proj_pert_post, ref)
                    if tau_id is not None:
                        taus_id.append(tau_id)
                        n_recover_id += 1

                # Frechet normalizado (post t_inj)
                P = base_traj[t_inj:Lb]
                Q = pert_traj[t_inj:Lp]
                if len(P) >= 2 and len(Q) >= 2:
                    d = discrete_frechet(P, Q)
                    v = mean_step_velocity(P)
                    if v and v > 1e-8:
                        frechet_norm.append(d / v)

            n_total = len(pairs)
            row = {
                "n_total": n_total,
                "tau_geom_mean": float(np.mean(taus_geom)) if taus_geom else None,
                "recovery_rate_geom": n_recover_geom / n_total if n_total else None,
                "tau_identity_mean": float(np.mean(taus_id)) if taus_id else None,
                "recovery_rate_id": n_recover_id / n_total if n_total else None,
                "frechet_norm_mean": float(np.mean(frechet_norm)) if frechet_norm else None,
                "frechet_norm_n": len(frechet_norm),
            }
            out[group][t_inj] = row
            print(f"{group:20s} t_inj={t_inj:3d}  "
                  f"tau_geom={row['tau_geom_mean']!s:>8s} rate_geom={row['recovery_rate_geom']:.2f}  "
                  f"tau_id={row['tau_identity_mean']!s:>8s} rate_id={row['recovery_rate_id']:.2f}  "
                  f"frechet_norm={row['frechet_norm_mean']:.3f} (n={row['frechet_norm_n']})"
                  if row['frechet_norm_mean'] is not None else
                  f"{group:20s} t_inj={t_inj:3d}  incomplete")

    out_path = Path(__file__).parent / "t2_perturbation_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
