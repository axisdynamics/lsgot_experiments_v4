#!/usr/bin/env python3
"""
Calibración de sigma para la perturbación T2 en Command R.

`run_perturbation_t2.py` (Gemma-4-31B-it) hardcodea SIGMA_VALUE=5.979...,
que viene de compute_sigma(mean_velocity=438.4, hidden_dim=5376) — el
438.4 es el mean_velocity REAL medido sobre las trayectorias del grupo
`axis` original (ver run_perturbation.py líneas 90-93: "mean_velocity real
del grupo axis original (...) Exp 0.5 congelado"). No es una constante
universal — es una propiedad empírica de CADA modelo, y reusarla en
Command R rompería la calibración (hidden_dim distinto: 8192 vs 5376, y la
dinámica de velocidad de trayectoria no es comparable entre arquitecturas).

`calibrate_sigma.py` (la utilidad original que hizo este cálculo para
Gemma) nunca se sincronizó a este repo ni a la máquina local (ver
FUENTES.md — era una utilidad operativa del pod, "no necesaria para
reproducir la evidencia primaria"). Este script es una reconstrucción de
buena fe de esa definición, NO una copia verificada del original — antes
de confiar en el sigma resultante, confirmar que la definición de
"velocity" de abajo coincide con la que se usó para calibrar Gemma
(mean_velocity=438.4).

Definición usada aquí: velocity_t = ||h_t - h_{t-1}||_2 en el espacio de
la capa de captura (aquí: capa final, misma convención que v_identidad.npy
y LAYER_IDX=-1 en run_perturbation_t2.py), promediada sobre todos los
pasos t y todos los prompts del grupo `axis` (excluyendo el primer token,
que no tiene t-1).

Uso (después de correr run_command_r_extraction.py --groups axis):
    python3 calibrate_sigma_command_r.py \
        --npz results_local/command_r_fase0/axis_command_r.npz \
        --final-layer-index 39
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "perturbation"))
from perturbation_extractor import compute_sigma  # noqa: E402

HIDDEN_DIM_COMMAND_R = 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True,
                     help="Salida de run_command_r_extraction.py para el grupo axis "
                          "(p.ej. results_local/command_r_fase0/axis_command_r.npz)")
    ap.add_argument("--final-layer-index", type=int, default=39,
                     help="Índice de la capa final usada como capa de captura (default: 39, Command R)")
    ap.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM_COMMAND_R)
    args = ap.parse_args()

    data = np.load(args.npz)
    key = f"embeddings_L{args.final_layer_index}"
    if key not in data:
        print(f"ERROR: {key} no está en {args.npz}. Claves disponibles: {list(data.keys())}",
              file=sys.stderr)
        sys.exit(1)

    emb = data[key].astype(np.float32)   # (n_prompts, max_new_tokens, hidden_dim)
    lengths = data["lengths"]

    velocities = []
    for i, L in enumerate(lengths):
        if L < 2:
            continue
        traj = emb[i, :L]                # (L, hidden_dim)
        step_norms = np.linalg.norm(traj[1:] - traj[:-1], axis=-1)  # (L-1,)
        velocities.append(step_norms)

    all_v = np.concatenate(velocities)
    mean_velocity = float(all_v.mean())
    std_velocity = float(all_v.std())

    print(f"n_prompts={len(lengths)} | n_steps_total={len(all_v)}")
    print(f"mean_velocity = {mean_velocity:.4f} (std={std_velocity:.4f})")

    for k_label, k in (("small", 0.5), ("medium", 1.0), ("large", 2.0)):
        sigma = compute_sigma(mean_velocity, args.hidden_dim, k=k)
        print(f"  sigma[{k_label} k={k}] = {sigma:.6f}")

    print("\nPasar el sigma 'medium' (o el que corresponda) a "
          "run_perturbation_t2_command_r.py con --sigma <valor>, "
          "o --mean-velocity {:.4f} para que lo recalcule.".format(mean_velocity))


if __name__ == "__main__":
    main()
