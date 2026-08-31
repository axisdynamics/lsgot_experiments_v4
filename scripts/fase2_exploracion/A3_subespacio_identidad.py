#!/usr/bin/env python3
"""
A3 — Subespacio de identidad multidimensional, no solo v_hat.

v_hat es UNA dirección (diferencia de medias axis-generic_long). Si
identidad organiza un subespacio real (E-J: 28 grados de ángulo
principal vs generic), un PCA sobre las 20 diferencias PAREADAS
(axis[i] - generic_long[i], mismo prompt) debería mostrar cuánta
varianza explica PC1 (~v_hat) vs PC2, PC3... Si PC1 domina, v_hat es un
buen resumen; si no, hay estructura adicional que v_hat no captura.
"""
import sys, json
from pathlib import Path

import numpy as np

FREE_DIR = Path("/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local/sia_extended_v5")
V_HAT_PATH = Path(__file__).parent.parent / "perturbation" / "v_identidad.npy"
PRIORITY_SUBSET = [1, 3, 6, 10, 14, 21, 23, 27, 31, 39, 41, 45, 51, 59, 61, 65, 71, 79, 91, 98]


def load(g):
    d = np.load(FREE_DIR / f"{g}_embeddings.npz")
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int)


def per_prompt_mean(emb, lengths, exclude_t0=True):
    start = 1 if exclude_t0 else 0
    rows = []
    for i, L in enumerate(lengths):
        rows.append(emb[i, start:L].mean(axis=0) if L > start else emb[i, :L].mean(axis=0))
    return np.array(rows)


v_hat = np.load(V_HAT_PATH)

emb_axis, len_axis = load("axis")
emb_gl, len_gl = load("generic_long")
mean_axis = per_prompt_mean(emb_axis, len_axis)
mean_gl = per_prompt_mean(emb_gl, len_gl)

diffs = mean_axis - mean_gl  # (20, 5376), pareado por prompt (mismo orden PRIORITY_SUBSET)
print("diffs shape:", diffs.shape)

# PCA sobre las 20 diferencias
diffs_centered = diffs - diffs.mean(axis=0)
U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
var_explained = (S ** 2) / (S ** 2).sum()
cum_var = np.cumsum(var_explained)

print("\n=== Varianza explicada por componente (PCA de las 20 diferencias axis-generic_long) ===")
for i in range(min(10, len(S))):
    print(f"  PC{i+1}: {var_explained[i]*100:5.1f}%  (acumulado: {cum_var[i]*100:5.1f}%)")

# cos(PC1, v_hat) -- v_hat fue calculado como la MEDIA de las diferencias, no PC1;
# deberían ser parecidos si PC1 domina
pc1 = Vt[0]
cos_pc1_vhat = float(np.dot(pc1 / np.linalg.norm(pc1), v_hat / np.linalg.norm(v_hat)))
print(f"\ncos(PC1, v_hat) = {cos_pc1_vhat:.4f}  (v_hat = media de las diferencias; PC1 = primer eje de varianza)")

# ¿el subespacio multidim separa MEJOR que v_hat solo?
# proyectar automata_neutro y vanilla sobre v_hat (1D) vs sobre PC1-3 (3D, norma de la proyección)
def load_mean(g):
    emb, lengths = load(g)
    return per_prompt_mean(emb, lengths)

conditions = {"axis": mean_axis, "generic_long": mean_gl,
              "automata_neutro": load_mean("automata_neutro"),
              "vanilla": load_mean("vanilla"),
              "axis_pec_only": load_mean("axis_pec_only")}

k = 3
subspace = Vt[:k]  # (k, 5376), ya ortonormal (de SVD)

print(f"\n=== Separación: proyección 1D (v_hat) vs norma en subespacio {k}D (PC1-{k}) ===")
results = {}
for name, means in conditions.items():
    proj_1d = means @ v_hat  # (20,)
    proj_kd = means @ subspace.T  # (20, k)
    norm_kd = np.linalg.norm(proj_kd, axis=1)  # (20,)
    results[name] = {
        "proj_1d_mean": float(np.mean(proj_1d)), "proj_1d_std": float(np.std(proj_1d)),
        "norm_kd_mean": float(np.mean(norm_kd)), "norm_kd_std": float(np.std(norm_kd)),
    }
    print(f"  {name:18s} v_hat(1D)={np.mean(proj_1d):+8.3f}±{np.std(proj_1d):.3f}   "
          f"||proj PC1-{k}||={np.mean(norm_kd):7.3f}±{np.std(norm_kd):.3f}")

# separación en unidades de d de Cohen, 1D vs subespacio (usando norma kd como "score")
from scipy import stats as sp_stats
def cohend(a, b):
    a, b = np.array(a), np.array(b)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else 0.0

axis_1d = conditions["axis"] @ v_hat
an_1d = conditions["automata_neutro"] @ v_hat
axis_kd = np.linalg.norm(conditions["axis"] @ subspace.T, axis=1)
an_kd = np.linalg.norm(conditions["automata_neutro"] @ subspace.T, axis=1)
d_1d = cohend(axis_1d, an_1d)
d_kd = cohend(axis_kd, an_kd)
print(f"\naxis vs automata_neutro: d (v_hat 1D) = {d_1d:+.3f}   d (norma subespacio {k}D) = {d_kd:+.3f}")

out = {
    "variance_explained": var_explained.tolist(),
    "cum_variance": cum_var.tolist(),
    "cos_pc1_vhat": cos_pc1_vhat,
    "projections": results,
    "d_axis_vs_automata_1d": d_1d,
    "d_axis_vs_automata_kd": d_kd,
}
Path("A3_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("\nGuardado: A3_results.json")
