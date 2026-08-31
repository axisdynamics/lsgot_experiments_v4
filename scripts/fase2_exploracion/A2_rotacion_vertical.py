#!/usr/bin/env python3
"""
A2 — Rotación vertical: cuánto gira la representación del MISMO token
al pasar de L5 a L55 (trayectoria "vertical" a través de capas, no
"horizontal" a través del tiempo de generación).

Usa los embeddings multi-capa de E-F2 (ef2_L5_L55). Para cada token
(posición fija en la generación), calcula el ángulo coseno entre su
representación en capas consecutivas muestreadas, y el ángulo end-to-end
L5->L55. Compara identidad (axis, axis_pec_only) vs restricción
(automata_neutro) vs neutro (vanilla).
"""
import sys, json
from pathlib import Path

import numpy as np

DATA_DIR = Path("/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local/ef2_L5_L55")
LAYER_NS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
GROUPS = ["axis", "axis_pec_only", "automata_neutro", "vanilla", "generic_long"]


def cos(a, b):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    denom = na * nb
    denom = np.where(denom < 1e-10, 1.0, denom)
    return (a * b).sum(axis=-1) / denom


def load(g):
    d = np.load(DATA_DIR / f"{g}_ef2.npz")
    layers = {n: d[f"embeddings_L{n}"].astype(np.float32) for n in LAYER_NS}
    return layers, d["lengths"].astype(int)


results = {}
for g in GROUPS:
    layers, lengths = load(g)
    # ángulo end-to-end L5 -> L55, promediado sobre tokens válidos y prompts
    e2e_angles = []
    stepwise_angles = {f"{LAYER_NS[i]}->{LAYER_NS[i+1]}": [] for i in range(len(LAYER_NS) - 1)}
    for i, L in enumerate(lengths):
        if L < 2:
            continue
        h5 = layers[5][i, :L]
        h55 = layers[55][i, :L]
        c = cos(h5, h55)
        angles = np.degrees(np.arccos(np.clip(c, -1, 1)))
        e2e_angles.extend(angles.tolist())
        for j in range(len(LAYER_NS) - 1):
            a, b = LAYER_NS[j], LAYER_NS[j + 1]
            ha, hb = layers[a][i, :L], layers[b][i, :L]
            c2 = cos(ha, hb)
            ang2 = np.degrees(np.arccos(np.clip(c2, -1, 1)))
            stepwise_angles[f"{a}->{b}"].extend(ang2.tolist())

    results[g] = {
        "e2e_L5_L55_mean_deg": float(np.mean(e2e_angles)),
        "e2e_L5_L55_std_deg": float(np.std(e2e_angles)),
        "stepwise_mean_deg": {k: float(np.mean(v)) for k, v in stepwise_angles.items()},
    }
    print(f"{g:20s} L5->L55 end-to-end: {results[g]['e2e_L5_L55_mean_deg']:.2f}° "
          f"(±{results[g]['e2e_L5_L55_std_deg']:.2f})")
    print("   stepwise: " + " ".join(f"{k}={v:.1f}°" for k, v in results[g]["stepwise_mean_deg"].items()))

Path("A2_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
print("\nGuardado: A2_results.json")
