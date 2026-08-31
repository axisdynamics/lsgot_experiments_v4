#!/usr/bin/env python3
"""
Genera paper/figures/recovery_realignment_curve.png (Figura 1, §3.3 de
lsgot_4.md) — evolución de la ventana móvil de la misma cantidad de la
que se extrae τ en recovery_analyzer.py (cos(cumulative_mean(perturbed[:w]),
centroide baseline propio)), promediada entre prompts, para axis vs
automata_neutro vs vanilla en los tres t_inj.

Requiere los .npz de trayectorias baseline/perturbadas de la corrida
H4_rev — NO incluidos en este repo por tamaño (mismos archivos que usa
recovery_analyzer.py, ver README.md "Cómo reproducir"): un
`{cond}_baseline_embeddings.npz` y un `{cond}_perturbed_t{50,128,200}_embeddings.npz`
por condición en TRAJ_DIR, cada uno con arrays "embeddings" (N,256,D),
"lengths" (N,), "prompt_ids" (N,).

Uso:
    python make_recovery_realignment_figure.py [--traj-dir DIR] [--out PATH]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_TRAJ_DIR = (
    "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
    "gemma4_31b_combined/perturbation/results/perturbation_sia_L30_medium/trajectories"
)
DEFAULT_OUT = Path(__file__).parent.parent.parent / "paper" / "figures" / "recovery_realignment_curve.png"


def load(traj_dir: Path, name: str):
    d = np.load(traj_dir / f"{name}.npz", allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int), d["prompt_ids"]


def rolling_cos_to_centroid(post_seg: np.ndarray, centroid_full: np.ndarray) -> np.ndarray:
    """cos(cumulative_mean(post_seg[:w]), centroid_full) para w=1..len(post_seg) —
    misma cantidad que recovery_analyzer.py::_find_tau usa para definir τ."""
    norm_c = np.linalg.norm(centroid_full) + 1e-8
    cumulative = np.zeros_like(centroid_full)
    out = []
    for w in range(1, len(post_seg) + 1):
        cumulative += post_seg[w - 1]
        centroid_w = cumulative / w
        cos_w = np.dot(centroid_w, centroid_full) / (np.linalg.norm(centroid_w) + 1e-8) / norm_c
        out.append(cos_w)
    return np.array(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str, default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()
    traj_dir = Path(args.traj_dir)

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)

    final_gap = {}
    for ax, t_inj in zip(axes, [50, 128, 200]):
        for cond, color, label in [("axis", "#1f4e9c", "axis (identity, wired)"),
                                     ("automata_neutro", "#c0392b", "automata_neutro (constraint, no identity)"),
                                     ("vanilla", "#888888", "vanilla (minimal baseline)")]:
            base_emb, base_len, base_ids = load(traj_dir, f"{cond}_baseline_embeddings")
            pert_emb, pert_len, pert_ids = load(traj_dir, f"{cond}_perturbed_t{t_inj}_embeddings")
            common = sorted(set(base_ids.tolist()) & set(pert_ids.tolist()))
            base_pos = {pid: i for i, pid in enumerate(base_ids)}
            pert_pos = {pid: i for i, pid in enumerate(pert_ids)}

            curves = []
            for pid in common:
                bi = base_pos[pid]; pi = pert_pos[pid]
                Tb = base_len[bi]; Tp = pert_len[pi]
                if t_inj >= Tb or t_inj >= Tp:
                    continue
                base_full = base_emb[bi, :min(Tb, 256)]
                centroid_full = base_full.mean(axis=0)
                post_p = pert_emb[pi, t_inj:min(Tp, 256)]
                if len(post_p) < 5:
                    continue
                curves.append(rolling_cos_to_centroid(post_p, centroid_full))
            if not curves:
                continue
            maxlen = max(len(c) for c in curves)
            mat = np.full((len(curves), maxlen), np.nan)
            for i, c in enumerate(curves):
                mat[i, :len(c)] = c
            mean_c = np.nanmean(mat, axis=0)
            n_valid = np.sum(~np.isnan(mat), axis=0)
            sem_c = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(n_valid, 1))
            x = np.arange(1, maxlen + 1)
            ax.plot(x, mean_c, color=color, label=f"{label}, n={len(curves)}", lw=2.2)
            ax.fill_between(x, mean_c - sem_c, mean_c + sem_c, color=color, alpha=0.15)
            final_gap[(cond, t_inj)] = mean_c[-1]

        ax.set_title(f"t_inj = {t_inj}", fontsize=12)
        ax.set_xlabel("rolling window size w\n(tokens since injection)")
        ax.set_ylim(0.15, 1.03)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("cos(cumulative_mean(perturbed[:w]),\nown pre-perturbation baseline centroid)")
    axes[1].legend(fontsize=9, loc="lower right")

    fig.suptitle("Figure 1. Coarse recovery masks a persistent realignment gap for automata_neutro\n"
                 "(rolling-window cosine to each trajectory's own baseline centroid — the quantity τ is computed from — mean ± SEM across prompts)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    print("saved", out_path)
    print("final-window values:", final_gap)


if __name__ == "__main__":
    main()
