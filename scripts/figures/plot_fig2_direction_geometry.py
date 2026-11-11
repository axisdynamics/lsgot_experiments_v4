#!/usr/bin/env python3
"""
Figura 2 · La geometría de direcciones de §4.7.

(a) Matriz de cosenos 6×6 entre v̂ y cinco direcciones construidas nativamente
    en t=0. v̂ es el *outlier* de la familia: 0.18–0.27 contra todas, mientras
    que las otras cinco correlacionan 0.27–0.85 entre sí. Codificación
    secuencial (un solo matiz, claro->oscuro) porque el dato es magnitud.
    La diagonal es 1.0 por construcción y se marca como tal, fuera de la escala.

(b) Las 7 condiciones del panel leídas en t=0 sobre las dos direcciones, en el
    MISMO eje (las dos son cosenos). El orden se conserva (ρ = 0.964) y el
    rango se abre 3.71× — v̂ comprime la separación pero no la invierte. Ése es
    el resultado de §4.7: las conclusiones sobre *si* un prompt ancla no
    dependen de la sonda; el nombre de la sonda sí.

ρ y el factor 3.71× se recalculan acá desde el JSON, no se copian del paper.

Fuentes (las dos en el paquete):
  data/t0_native/e1f_matriz_cosenos_6x6.json
  data/t0_native/e1_vhat_nativa_t0.json

Uso:
    python scripts/figures/plot_fig2_direction_geometry.py
Escribe paper/figures/fig2_direction_geometry.{png,pdf}
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper" / "figures"

# ── paleta validada ──────────────────────────────────────────────────────────
ACCENT = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"
# rampa secuencial azul 100->700, monotonía de luminancia verificada
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("lsgot_blue", RAMP)

KEYS = ["v_canon", "v_t0", "v_t0_pec", "v_t0_wit", "v_t0_berg", "v_t0_evil"]
NICE = {
    "v_canon": "v̂",
    "v_t0": "v̂$_{t0}$",
    "v_t0_pec": "v̂$_{t0}$(pec)",
    "v_t0_wit": "v̂$_{t0}$(wit)",
    "v_t0_berg": "v̂$_{t0}$(berg)",
    "v_t0_evil": "v̂$_{t0}$(evil)",
}
WIRED = {"axis", "axis_short", "axis_pec_only"}


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def panel_a(ax, M):
    vals = [M[r][c] for r in KEYS for c in KEYS if r != c]
    norm = Normalize(vmin=min(vals), vmax=max(vals))

    for i, r in enumerate(KEYS):
        for j, c in enumerate(KEYS):
            v = M[r][c]
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor=SURFACE, edgecolor=GRID,
                                           lw=0.8, zorder=1))
                ax.text(j, i, "—", ha="center", va="center", fontsize=8,
                        color=GRID, zorder=2)
                continue
            # 2px de separación entre celdas: el borde del color de superficie
            ax.add_patch(plt.Rectangle((j - .49, i - .49), .98, .98,
                                       facecolor=CMAP(norm(v)),
                                       edgecolor=SURFACE, lw=1.4, zorder=1))
            # etiqueta directa en cada celda: una matriz se lee por sus números
            ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=7.2, zorder=2,
                    color="#ffffff" if norm(v) > 0.60 else INK)

    # el recuadro que señala la fila/columna de v̂: es el hallazgo
    ax.add_patch(plt.Rectangle((-.5, -.5), 6, 1, fill=False, edgecolor=INK,
                               lw=1.3, zorder=3))
    ax.add_patch(plt.Rectangle((-.5, -.5), 1, 6, fill=False, edgecolor=INK,
                               lw=1.3, zorder=3))

    ax.set_xlim(-.5, 5.5)
    ax.set_ylim(5.5, -.5)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    SHORT = ["v̂", "v̂$_{t0}$", "pec", "wit", "berg", "evil"]
    ax.set_xticklabels(SHORT, fontsize=7.6, color=INK_2)
    ax.set_yticklabels([NICE[k] for k in KEYS], fontsize=7.6, color=INK_2)
    ax.get_xticklabels()[0].set_color(INK)
    ax.get_yticklabels()[0].set_color(INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(a)  cosine between directions", fontsize=8.8, color=INK,
                 loc="left", pad=26, fontweight="semibold")
    ax.annotate("v̂ sits outside the family it is read against: 0.18–0.27\n"
                "to all five, which reach 0.85 among themselves",
                xy=(0, 1.0), xycoords="axes fraction",
                xytext=(0, 5), textcoords="offset points",
                fontsize=7.2, color=INK_2, va="bottom")


def panel_b(ax, panel):
    conds = [c for c, v in panel["v_canon"].items() if v.get("group") == "panel"]
    a = [panel["v_canon"][c]["t0"]["mean"] for c in conds]
    b = [panel["v_t0"][c]["t0"]["mean"] for c in conds]
    rho = spearman(a, b)
    ratio = (max(b) - min(b)) / (max(a) - min(a))

    order = sorted(range(len(conds)), key=lambda i: -a[i])
    ys = list(range(len(order)))[::-1]

    for y, i in zip(ys, order):
        wired = conds[i] in WIRED
        ax.plot([a[i], b[i]], [y, y], color=GRID, lw=1.4, zorder=1,
                solid_capstyle="round")
        ax.plot(b[i], y, marker="D", ms=5.4, mfc=SURFACE,
                mec=ACCENT if wired else INK_2, mew=1.4, ls="none", zorder=3)
        ax.plot(a[i], y, marker="o", ms=6.6,
                mfc=ACCENT if wired else INK_2, mec=SURFACE, mew=1.3,
                ls="none", zorder=3)

    ax.axvline(0.0, color=GRID, lw=0.8, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([conds[i] for i in order], fontsize=7.8)
    for lbl, i in zip(ax.get_yticklabels(), order):
        lbl.set_color(INK if conds[i] in WIRED else INK_2)
        if conds[i] in WIRED:
            lbl.set_fontweight("semibold")
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.5, labelsize=7.2, colors=INK_2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("projection at t=0 (cosine)", fontsize=7.8, color=INK_2,
                  labelpad=4)
    ax.set_title("(b)  the same panel on both directions", fontsize=8.8,
                 color=INK, loc="left", pad=26, fontweight="semibold")
    ax.annotate(f"order preserved (ρ = {rho:.3f}); separation\n"
                f"{ratio:.2f}× wider on v̂$_{{t0}}$",
                xy=(0, 1.0), xycoords="axes fraction",
                xytext=(0, 5), textcoords="offset points",
                fontsize=7.2, color=INK_2, va="bottom")

    h_a = plt.Line2D([], [], marker="o", ms=6.6, mfc=INK_2, mec=SURFACE,
                     mew=1.3, ls="none", label="read on v̂")
    h_b = plt.Line2D([], [], marker="D", ms=5.4, mfc=SURFACE, mec=INK_2,
                     mew=1.4, ls="none", label="read on v̂$_{t0}$")
    ax.legend(handles=[h_a, h_b], loc="upper left", frameon=False,
              fontsize=7.6, handletextpad=0.4, borderaxespad=0.3)
    return rho, ratio


def main() -> None:
    M = json.loads((REPO / "data/t0_native/e1f_matriz_cosenos_6x6.json")
                   .read_text(encoding="utf-8"))
    panel = json.loads((REPO / "data/t0_native/e1_vhat_nativa_t0.json")
                       .read_text(encoding="utf-8"))["panel"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.5,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.4, 3.25),
        gridspec_kw={"width_ratios": [1.0, 1.15]})
    panel_a(ax1, M)
    rho, ratio = panel_b(ax2, panel)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.775, bottom=0.135,
                        wspace=0.44)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_direction_geometry.{ext}", dpi=300,
                    bbox_inches="tight", facecolor=SURFACE)
    print("escrito:", OUT / "fig2_direction_geometry.png")
    print("escrito:", OUT / "fig2_direction_geometry.pdf")
    print(f"recalculado desde el JSON: rho = {rho:.4f} · factor = {ratio:.2f}x")


if __name__ == "__main__":
    main()
