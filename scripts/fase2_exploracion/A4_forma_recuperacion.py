#!/usr/bin/env python3
"""
A4 — Forma completa de la recuperación tras perturbación (no solo tau/
recovery_rate escalares). Para cada prompt, coseno entre la trayectoria
PERTURBADA y su propia trayectoria BASELINE (mismo prompt, mismo step
absoluto) en función de k = pasos desde la inyección. Busca: decaimiento
monótono, overshoot (se aleja antes de acercarse), oscilación.
"""
import json
from pathlib import Path

import numpy as np

D = Path("/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/perturbation/results/perturbation_sia_L30_medium/trajectories")
GROUPS = ["axis", "vanilla", "automata_neutro"]
T_INJS = [50, 128, 200]
WINDOW = 80  # pasos post-inyección a inspeccionar


def load(name):
    d = np.load(D / f"{name}_embeddings.npz", allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int), d["prompt_ids"]


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


results = {}
for g in GROUPS:
    base_emb, base_len, base_pids = load(f"{g}_baseline")
    base_by_pid = {int(pid): i for i, pid in enumerate(base_pids)}
    results[g] = {}
    for t_inj in T_INJS:
        pert_emb, pert_len, pert_pids = load(f"{g}_perturbed_t{t_inj}")
        curves = []
        for j, pid in enumerate(pert_pids):
            pid = int(pid)
            if pid not in base_by_pid:
                continue
            bi = base_by_pid[pid]
            Lb, Lp = base_len[bi], pert_len[j]
            end = min(Lb, Lp, t_inj + WINDOW)
            if end <= t_inj:
                continue
            curve = []
            for t in range(t_inj, end):
                c = cos(base_emb[bi, t], pert_emb[j, t])
                curve.append(c)
            if len(curve) >= WINDOW // 2:
                curves.append(curve)
        # promediar curvas (recortar a la más corta común)
        min_len = min(len(c) for c in curves) if curves else 0
        if min_len == 0:
            continue
        arr = np.array([c[:min_len] for c in curves])
        mean_curve = arr.mean(axis=0)
        results[g][t_inj] = {
            "n_prompts": len(curves),
            "mean_curve": mean_curve.tolist(),
            "min_cos": float(mean_curve.min()),
            "argmin_k": int(np.argmin(mean_curve)),
            "final_cos": float(mean_curve[-1]),
            "cos_at_k5": float(mean_curve[min(5, len(mean_curve)-1)]),
            "cos_at_k20": float(mean_curve[min(20, len(mean_curve)-1)]) if len(mean_curve) > 20 else None,
        }
        print(f"{g:18s} t_inj={t_inj:3d}  n={len(curves):2d}  "
              f"min_cos={results[g][t_inj]['min_cos']:.4f} @k={results[g][t_inj]['argmin_k']:3d}  "
              f"cos@k5={results[g][t_inj]['cos_at_k5']:.4f}  "
              f"cos@k20={results[g][t_inj]['cos_at_k20']:.4f}  "
              f"final={results[g][t_inj]['final_cos']:.4f}")

Path("A4_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
print("\nGuardado: A4_results.json")
