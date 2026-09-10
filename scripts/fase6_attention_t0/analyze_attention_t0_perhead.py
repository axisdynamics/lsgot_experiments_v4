#!/usr/bin/env python3
"""
FASE 6 · ronda 2, §1.1 + §1.2 — análisis POR CABEZA sobre los .npz ya extraídos.

Propuesta: Descargas/Experimento_atencion_v2.md. NO requiere nuevo forward pass:
los .npz de ronda 1 guardan cada métrica por (prompt, capa, cabeza), incl.
`sysfrac_last` = masa de la última fila de query sobre TODOS los tokens del
system prompt (métrica de §1.1, versión última-posición).

Pregunta (§1.2): promediar sobre 32 cabezas puede diluir una firma que vive en
2-3 cabezas. ¿Hay alguna cabeza (capa, h) donde el mecanismo del testigo — que
axis y soul_md_corto comparten en v̂ pese a no compartir texto — deje una firma
consistente, separándose de sus controles pareados por longitud Y de automata?

Contrastes pareados por longitud (mismos que el análisis final de ronda 1):
  axis          vs generic_long          ΔT≈13
  soul_md_corto vs soul_elena_financial  ΔT≈226
  axis          vs automata_neutro       ΔT≈54   (testigo vs paso-cableado-genérico)

Cabeza candidata (§1.2 + §2): en la MISMA (capa, h),
  - axis se separa de generic_long           (|d| ≥ D_HEAD)
  - soul_md_corto se separa de soul_elena     (|d| ≥ D_HEAD, MISMO signo que arriba)
  - axis se separa de automata_neutro         (|d| ≥ D_AUTO, mismo signo)
d por cabeza sobre los 20 prompts fijos de t=0 → hereda n_efectivo≈1: es
EXPLORATORIO, se reporta el patrón, no la significancia (igual que ronda 1).

Uso:  python analyze_attention_t0_perhead.py [--data-dir results_attn_t0]
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from statistical_tests import GeometricStatisticalTests  # noqa: E402

METRICS = ["sysfrac_last", "ent_last", "ent_rowmean"]      # §1.1 = sysfrac_last
SPAN_METRIC = "wenrich_last"                                 # solo condiciones con span
PAIRS = [
    ("axis", "generic_long"),
    ("soul_md_corto", "soul_elena_financial"),
    ("axis", "automata_neutro"),
]
D_HEAD, D_AUTO = 2.0, 1.5      # barras altas: n_efectivo≈1, buscamos firma marcada


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return np.nan if s == 0 else (a.mean() - b.mean()) / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).parent / "results_attn_t0"))
    args = ap.parse_args()
    dd = Path(args.data_dir)
    conds = {f.stem: {k: np.load(f, allow_pickle=True)[k] for k in np.load(f, allow_pickle=True).files}
             for f in sorted(dd.glob("*.npz"))}
    labels = list(next(iter(conds.values()))["layers"].tolist())
    nL = len(labels)
    H = int(next(iter(conds.values()))["heads"])
    print(f"condiciones: {sorted(conds)} | capas {labels} | H={H}")
    print(f"|T|: " + ", ".join(f"{k}={int(conds[k]['T'].mean())}" for k in sorted(conds)))
    print(f"barras: |d(axis|gen)|≥{D_HEAD}, |d(soul_md|elena)|≥{D_HEAD} mismo signo, "
          f"|d(axis|automata)|≥{D_AUTO} mismo signo\n")

    out = {"labels": labels, "H": H, "metrics": {}}
    for metric in METRICS:
        cand = []
        # d por (capa, cabeza) para cada par
        dmap = {}
        for a, b in PAIRS:
            if a not in conds or b not in conds:
                continue
            arr = np.full((nL, H), np.nan)
            for li in range(nL):
                for h in range(H):
                    va = conds[a][metric][:, li, h]
                    vb = conds[b][metric][:, li, h]
                    va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
                    arr[li, h] = cohens_d(va, vb)
            dmap[(a, b)] = arr
        if len(dmap) < 3:
            continue
        d_ag = dmap[("axis", "generic_long")]
        d_me = dmap[("soul_md_corto", "soul_elena_financial")]
        d_aa = dmap[("axis", "automata_neutro")]
        for li in range(nL):
            for h in range(H):
                x, y, z = d_ag[li, h], d_me[li, h], d_aa[li, h]
                if not np.all(np.isfinite([x, y, z])):
                    continue
                if (abs(x) >= D_HEAD and abs(y) >= D_HEAD and abs(z) >= D_AUTO
                        and np.sign(x) == np.sign(y) == np.sign(z)):
                    cand.append({"layer": labels[li], "head": h,
                                 "d_axis_vs_generic": float(x),
                                 "d_soulmd_vs_elena": float(y),
                                 "d_axis_vs_automata": float(z)})
        out["metrics"][metric] = {
            "n_candidate_heads": len(cand), "candidates": cand,
            "max_abs_d_axis_vs_generic": float(np.nanmax(np.abs(d_ag))),
            "max_abs_d_soulmd_vs_elena": float(np.nanmax(np.abs(d_me))),
        }
        print(f"=== {metric} ===")
        print(f"  máx |d| por cabeza: axis|generic={np.nanmax(np.abs(d_ag)):.1f}  "
              f"soul_md|elena={np.nanmax(np.abs(d_me)):.1f}  axis|automata={np.nanmax(np.abs(d_aa)):.1f}")
        if cand:
            print(f"  {len(cand)} cabeza(s) candidata(s) (firma de testigo consistente):")
            for c in cand:
                print(f"    L{c['layer']:>2} h{c['head']:<2}  "
                      f"d(axis|gen)={c['d_axis_vs_generic']:+.2f}  "
                      f"d(soul_md|elena)={c['d_soulmd_vs_elena']:+.2f}  "
                      f"d(axis|automata)={c['d_axis_vs_automata']:+.2f}")
        else:
            print("  0 cabezas candidatas — ninguna cabeza separa el testigo de sus dos "
                  "controles pareados en la misma dirección.")
        print()

    # firma por cabeza en el span propio (solo condiciones con span; within-prompt)
    span_conds = [c for c in ("axis", "axis_pec_only", "soul_md_corto", "automata_neutro") if c in conds]
    print("=== wenrich_last (enriquecimiento sobre el span propio), por cabeza — "
          "¿hay cabezas que sobre-atienden el span en axis Y soul_md_corto? ===")
    for name in span_conds:
        we = conds[name][SPAN_METRIC]                 # (20, nL, H)
        m = np.nanmean(we, axis=0)                    # (nL, H)
        # cabezas con enriquecimiento > 2 en alguna capa
        hot = [(labels[li], h, float(m[li, h])) for li in range(nL) for h in range(H)
               if np.isfinite(m[li, h]) and m[li, h] >= 2.0]
        hot.sort(key=lambda t: -t[2])
        top = ", ".join(f"L{L}h{h}={v:.1f}" for L, h, v in hot[:6])
        print(f"  {name:22s} cabezas con enrich≥2: {len(hot):3d}   top: {top or '—'}")
    out["span_hot_heads_note"] = ("enrich = masa_span / (span_len/T); >1 = sobre-atiende "
                                  "el span relativo a su tamaño")

    (dd / "attention_t0_perhead.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nguardado: {dd / 'attention_t0_perhead.json'}")


if __name__ == "__main__":
    main()
