#!/usr/bin/env python3
"""
Figura 1 · El ancla de t=0 en tres arquitecturas.

Small multiples: un panel por sustrato, cada uno con su propio eje, porque los
niveles absolutos NO son comparables entre modelos (tres v̂ estimadas por
separado, en espacios de dimensión 5376 / 5120 / 8192 — §4.5 del paper). Lo que
viaja entre sustratos es el signo, el orden y la separación, y eso es lo que la
figura muestra.

Codificación: el grupo "cablea el paso" lleva color de acento Y círculo lleno;
el resto lleva tinta neutra Y cuadrado hueco. La identidad nunca es solo color,
así que la figura sobrevive al gris y al daltonismo.

Fuentes de los números, todas en el paquete o en evidence/:
  Gemma-4-31B-it  data/t0_native/e1_vhat_nativa_t0.json -> panel.v_canon[*].t0.mean
  Qwen3-32B       evidence/QWEN3_VALIDATION_REPORT.md   -> §2, tabla E-L
  Command R 35B   evidence/COMMAND_R_VALIDATION_REPORT.md -> §2, tabla E-L

Uso:
    python scripts/figures/plot_fig1_t0_three_models.py
Escribe paper/figures/fig1_t0_three_models.{png,pdf}
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper" / "figures"

# ── paleta validada (scripts/validate_palette.js, modo light) ────────────────
ACCENT = "#2a78d6"   # slot categórico 1; 1 slot => ALL CHECKS PASS
INK = "#0b0b0b"      # text-primary
INK_2 = "#52514e"    # text-secondary: el grupo sin paso cableado
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"

# ── condiciones comunes a los tres sustratos ─────────────────────────────────
# Son las cuatro que los tres reportan en t=0. Qwen3 corrió 7 condiciones pero
# su reporte solo tabula estas cuatro, así que la intersección es la honesta.
ORDER = ["axis", "axis_pec_only", "automata_neutro", "vanilla"]
LABEL = {
    "axis": "axis",
    "axis_pec_only": "axis_pec_only",
    "automata_neutro": "automata_neutro",
    "vanilla": "vanilla",
}
WIRED = {"axis", "axis_pec_only"}          # cablean el paso autorreferente

GLOSS = {
    "axis": "wired step, full scaffold",
    "axis_pec_only": "wired step alone",
    "automata_neutro": "pure rule density",
    "vanilla": "no system prompt",
}

# Qwen3 y Command R: sus tensores no están en el paquete; los valores vienen de
# los reportes de validación que sí se entregan, y están citados ahí.
HARDCODED = {
    "Qwen3-32B": {
        "axis": +0.192, "axis_pec_only": +0.185,
        "automata_neutro": +0.008, "vanilla": +0.018,
        "_d": +12.97, "_src": "evidence/QWEN3_VALIDATION_REPORT.md §2",
    },
    "Command R 35B": {
        "axis": +0.224, "axis_pec_only": +0.226,
        "automata_neutro": +0.079, "vanilla": +0.066,
        "_d": +7.72, "_src": "evidence/COMMAND_R_VALIDATION_REPORT.md §2",
    },
}


def gemma_from_json() -> dict:
    """Gemma sale del JSON, no hardcodeado: es el sustrato cuyos datos viajan."""
    f = REPO / "data" / "t0_native" / "e1_vhat_nativa_t0.json"
    panel = json.loads(f.read_text(encoding="utf-8"))["panel"]["v_canon"]
    out = {c: panel[c]["t0"]["mean"] for c in ORDER}
    out["_d"] = +5.75
    out["_src"] = "data/t0_native/e1_vhat_nativa_t0.json"
    return out


def main() -> None:
    models = [
        ("Gemma-4-31B-it", gemma_from_json()),
        ("Qwen3-32B", HARDCODED["Qwen3-32B"]),
        ("Command R 35B", HARDCODED["Command R 35B"]),
    ]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.5,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.7,
        "xtick.color": INK_2,
        "ytick.color": INK,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    y = list(range(len(ORDER)))[::-1]          # axis arriba

    for ax, (name, vals) in zip(axes, models):
        lo = min(min(vals[c] for c in ORDER), 0.0)
        hi = max(vals[c] for c in ORDER)
        pad = 0.16 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)

        # cero de referencia, recesivo
        ax.axvline(0.0, color=GRID, lw=0.8, zorder=0)

        # banda que separa los dos grupos: entre axis_pec_only y automata_neutro
        ax.axhspan(1.5, len(ORDER) - 0.25, color=ACCENT, alpha=0.055,
                   lw=0, zorder=0)

        for yy, cond in zip(y, ORDER):
            v = vals[cond]
            wired = cond in WIRED
            # línea guía desde el cero hasta el punto: magnitud legible sin barras
            ax.plot([0.0, v], [yy, yy],
                    color=ACCENT if wired else INK_2,
                    lw=1.1 if wired else 0.9,
                    alpha=0.9 if wired else 0.45, zorder=1,
                    solid_capstyle="round")
            if wired:
                ax.plot(v, yy, marker="o", ms=7.0, mfc=ACCENT, mec=SURFACE,
                        mew=1.6, zorder=3, ls="none")
            else:
                ax.plot(v, yy, marker="s", ms=6.2, mfc=SURFACE, mec=INK_2,
                        mew=1.3, zorder=3, ls="none")
            # etiqueta directa encima del punto: nunca colisiona con el panel
            # vecino ni con el nombre de la condición
            ax.annotate(f"{v:+.3f}", (v, yy),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7.4, color=INK if wired else INK_2)

        ax.set_yticks(y)
        # los nombres de condición van una sola vez, en el panel de la izquierda
        if ax is axes[0]:
            ax.set_yticklabels([LABEL[c] for c in ORDER], fontsize=8.2)
            for lbl, cond in zip(ax.get_yticklabels(), ORDER):
                if cond in WIRED:
                    lbl.set_color(INK)
                    lbl.set_fontweight("semibold")
                else:
                    lbl.set_color(INK_2)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(-0.6, len(ORDER) - 0.25)

        ax.set_title(name, fontsize=9, color=INK, pad=9, fontweight="semibold")
        ax.annotate(f"d = {vals['_d']:+.2f}", xy=(0.5, 1.005),
                    xycoords="axes fraction", ha="center", va="bottom",
                    fontsize=7.4, color=ACCENT)

        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2.5, labelsize=7.2)
        ax.set_xlabel("projection on v̂ at t=0", fontsize=7.8, color=INK_2,
                      labelpad=4)

    # leyenda: la identidad nunca es solo color
    h_w = plt.Line2D([], [], marker="o", ms=7, mfc=ACCENT, mec=SURFACE, mew=1.6,
                     ls="none", label="wires a self-referential step")
    h_n = plt.Line2D([], [], marker="s", ms=6.2, mfc=SURFACE, mec=INK_2, mew=1.3,
                     ls="none", label="does not")
    fig.legend(handles=[h_w, h_n], loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.045),
               handletextpad=0.5, columnspacing=2.2)

    fig.subplots_adjust(left=0.165, right=0.99, top=0.80, bottom=0.285,
                        wspace=0.16)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_t0_three_models.{ext}", dpi=300,
                    bbox_inches="tight", facecolor=SURFACE)
    print("escrito:", OUT / "fig1_t0_three_models.png")
    print("escrito:", OUT / "fig1_t0_three_models.pdf")
    for name, vals in models:
        print(f"  {name:16s} fuente: {vals['_src']}")


if __name__ == "__main__":
    main()
