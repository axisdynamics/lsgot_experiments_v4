#!/usr/bin/env python3
"""
Figura 1 del paper arXiv (`paper/arxiv_lsgot_4.md`, §4.5) — frac_pos por
condición y batería, con la línea de azar en 0.50.

Lee `data/egrav/egrav_results.json` (no recalcula nada) y escribe PNG + PDF.
Rampa secuencial de un solo tono: la batería es una escala ordenada por carga
de auto-referencia (B-TAREA → B-META → B-ONTO), no una categoría; cada barra
lleva su valor como etiqueta directa, así que la identidad nunca depende solo
del color.

Uso:
    pip install matplotlib            # única dependencia extra
    python plot_fig1_egrav.py         # desde la raíz del repo o desde aquí
    python plot_fig1_egrav.py --results ../../data/egrav/egrav_results.json \
                              --out ../../paper/figures/fig1_egrav_frac_pos
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

BATTERIES = [  # orden = carga creciente de auto-referencia; rampa clara → oscura
    ("b_tarea", "B-TAREA", "#5FB3C1"),
    ("b_meta",  "B-META",  "#1596AB"),
    ("b_onto",  "B-ONTO",  "#06717F"),
]
CONDITIONS = [
    ("axis_pec_only",   "axis_pec_only\n(testigo cableado)"),
    ("vanilla",         "vanilla\n(sin system prompt)"),
    ("automata_neutro", "automata_neutro\n(restricción pura)"),
]
CHANCE = 0.50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "data/egrav/egrav_results.json"))
    ap.add_argument("--out", default=str(REPO / "paper/figures/fig1_egrav_frac_pos"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    profiles = json.loads(Path(args.results).read_text())["profiles"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    h = 0.24

    for ci, (cond, label) in enumerate(CONDITIONS):
        for bi, (bat, _, color) in enumerate(BATTERIES):
            v = profiles[cond][bat]["frac_pos"]["mean"]
            y = ci + (bi - 1) * h
            ax.barh(y, v, height=h * 0.88, color=color, zorder=3)
            ax.text(v + 0.012, y, f"{v:.3f}", va="center", ha="left",
                    fontsize=8, color="#14181F", zorder=4)

    ax.axvline(CHANCE, color="#8890A0", lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(CHANCE, len(CONDITIONS) - 0.52, "azar (0.50)", fontsize=8,
            color="#8890A0", ha="center", va="bottom")

    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([lab for _, lab in CONDITIONS], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("frac_pos — fracción de tokens con proyección positiva sobre v̂", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="#14181F", alpha=0.08, zorder=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#8890A0")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in BATTERIES]
    ax.legend(handles, [n for _, n, _ in BATTERIES], loc="lower right",
              frameon=False, fontsize=8, ncols=3, handlelength=1.1)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=args.dpi, bbox_inches="tight")
        print(f"escrito: {out}.{ext}")


if __name__ == "__main__":
    main()
