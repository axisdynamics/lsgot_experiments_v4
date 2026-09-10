#!/usr/bin/env python3
"""
FASE 6 — análisis: ¿el testigo cableado modula los pesos de atención en t=0?

Entrada: results_attn_t0/<cond>.npz de run_attention_t0_extraction.py.
Corre local, sin GPU.

DISEÑO PAREADO POR LONGITUD (revisado 2026-09-07 tras la primera corrida).
Las métricas de concentración/entropía de atención de la última posición son
sensibles a |T| del prompt, y las condiciones del panel van de 37 tokens
(`vanilla`) a ~4030 (`automata_neutro`). Agrupar y comparar grupos con |T|
heterogéneo mezcla el efecto de contenido con el de longitud (en la 1ª
corrida `sink_meanhead` salió d≈-1.40 PLANO en las 11 capas — firma de
confound de longitud, no de atención). Por eso aquí NO se usan grupos
pooled: sólo contrastes entre condiciones con |T| casi igual.

|T| medido (tokens): vanilla 37 · soul_solidity 933 · axis_pec_only 1465 ·
soul_elena 2077 · soul_md_corto 2303 · axis 3975 · generic_long 3988 ·
automata_neutro 4029.

CONTRASTES PRIMARIOS (ΔT pequeño; mismos que KEY_PAIRS del panel de v̂):
  axis            vs generic_long      ΔT≈13    testigo+identidad vs generic largo
  automata_neutro vs generic_long      ΔT≈41    paso cableado a reglas vs generic
  axis            vs automata_neutro   ΔT≈54    testigo vs paso-cableado-genérico
  soul_md_corto   vs soul_elena_financial  ΔT≈226 (~10%)  testigo vs identidad
                                       declarada SIN testigo  (discrimina H1 de H2)

  MÉTRICA PRIMARIA: ent_last_meanhead (entropía ÷ log T de la atención de la
  última query, media sobre cabezas). p analítico: Mann-Whitney U bilateral.
  Corrección: Holm sobre (4 contrastes × 11 capas). Umbral |d|≥0.8, p_Holm<0.001.

DESCRIPTIVO / NO PAREADO (se reporta con ΔT, fuera del veredicto formal):
  axis_pec_only, soul_solidity_auditor, vanilla — sin condición de longitud
  comparable. Métricas secundarias: ent_last_minhead, ent_last_spread,
  ent_rowmean_meanhead, sink_meanhead, sysfrac_meanhead.

MASA SOBRE EL SPAN (axis, axis_pec_only, soul_md_corto, automata_neutro):
  within-condición, pareado wfrac_last vs cfrac_last (span del paso cableado
  vs slice de control de igual longitud del mismo prompt) — Wilcoxon + d
  within + enriquecimiento. Controlado por longitud por construcción.

Veredicto (sobre los contrastes primarios pareados):
  H_atención-1 (testigo específico): axis separa de generic_long Y de
      automata_neutro Y soul_md_corto separa de soul_elena_financial
      (todos |d|≥0.8, p_Holm<0.001).
  "cualquier paso cableado": axis Y automata_neutro separan de generic_long.
  H_atención-2 (identidad, no testigo): axis separa de generic_long pero
      soul_md_corto NO separa de soul_elena_financial.
  H_atención-0 (nula): ningún contraste primario cruza el umbral.

Uso:  python analyze_attention_t0.py [--data-dir results_attn_t0] [--n-perm 5000]
"""
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from statistical_tests import GeometricStatisticalTests  # noqa: E402

warnings.filterwarnings("ignore")

PRIMARY_METRIC = "ent_last_meanhead"
SECONDARY_METRICS = ["ent_last_minhead", "ent_last_spread", "ent_rowmean_meanhead",
                     "sink_meanhead", "sysfrac_meanhead"]
SPAN_CONDS = ["axis", "axis_pec_only", "soul_md_corto", "automata_neutro"]

# (nombre, a, b) — pareados por |T|
PRIMARY_CONTRASTS = [
    ("axis__vs__generic_long", "axis", "generic_long"),
    ("automata_neutro__vs__generic_long", "automata_neutro", "generic_long"),
    ("axis__vs__automata_neutro", "axis", "automata_neutro"),
    ("soul_md_corto__vs__soul_elena_financial", "soul_md_corto", "soul_elena_financial"),
]
# descriptivos (sin match de longitud) — todos contra generic_long
DESCRIPTIVE = ["axis_pec_only", "soul_solidity_auditor", "vanilla"]

D_THRESH, P_THRESH, P_NS = 0.8, 0.001, 0.05


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def load_all(data_dir):
    conds = {}
    for f in sorted(Path(data_dir).glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        conds[f.stem] = {k: d[k] for k in d.files}
    if not conds:
        raise FileNotFoundError(f"sin .npz en {data_dir}")
    labels = list(next(iter(conds.values()))["layers"].tolist())
    return conds, labels


def reduce_metrics(c):
    el, er = c["ent_last"], c["ent_rowmean"]
    return {
        "ent_last_meanhead": np.nanmean(el, axis=2),
        "ent_last_minhead": np.nanmin(el, axis=2),
        "ent_last_spread": np.nanmax(el, axis=2) - np.nanmin(el, axis=2),
        "ent_rowmean_meanhead": np.nanmean(er, axis=2),
        "sink_meanhead": np.nanmean(c["sink_last"], axis=2),
        "sysfrac_meanhead": np.nanmean(c["sysfrac_last"], axis=2),
        "wfrac_last_meanhead": np.nanmean(c["wfrac_last"], axis=2),
        "cfrac_last_meanhead": np.nanmean(c["cfrac_last"], axis=2),
        "wenrich_last_maxhead": np.nanmax(c["wenrich_last"], axis=2),
    }


def clean(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)].tolist()


def compare(tester, a, b):
    if len(a) < 5 or len(b) < 5:
        return {"d": np.nan, "p_mw": np.nan, "p_perm": np.nan}
    return {"d": tester.cohens_d(a, b),
            "p_mw": tester.mann_whitney_u(a, b)["p_value"],
            "p_perm": tester.permutation_test(a, b, "mean_difference")["p_value"]}


def dT(conds, a, b):
    return abs(float(conds[a]["T"].mean()) - float(conds[b]["T"].mean()))


def fmt_row(name, ds, extra=""):
    return ("  " + f"{name:38s} d:" +
            "".join(f"{x:+7.2f}" if np.isfinite(x) else "    nan" for x in ds) + extra)


def best_layer(ds, p_holm_by_layer, labels):
    if not np.any(np.isfinite(ds)):
        return (None, np.nan, np.nan)
    li = int(np.nanargmax(np.abs(np.where(np.isfinite(ds), ds, 0.0))))
    return (labels[li], ds[li], (p_holm_by_layer or {}).get(labels[li], np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).parent / "results_attn_t0"))
    ap.add_argument("--n-perm", type=int, default=5000)
    args = ap.parse_args()

    conds, labels = load_all(args.data_dir)
    nL = len(labels)
    present = set(conds)
    print(f"condiciones: {sorted(present)}")
    print("|T| medio:", {k: int(conds[k]["T"].mean()) for k in sorted(present)})
    print(f"capas (labels L_N): {labels}")
    tester = GeometricStatisticalTests(n_permutations=args.n_perm, random_seed=42)
    red = {name: reduce_metrics(c) for name, c in conds.items()}
    results = {"labels": labels, "n_perm": args.n_perm,
               "T_mean": {k: float(conds[k]["T"].mean()) for k in present},
               "primary": {}, "secondary": {}, "descriptive": {}, "span_family": {}}

    # ── diagnóstico de replicación efectiva ──
    # Si la métrica primaria casi no varía entre los 20 prompts de usuario, el
    # panel tiene n_efectivo ≈ 1 por condición para la pregunta de atención
    # (a diferencia de v̂, donde el contenido de la pregunta sí mueve la
    # trayectoria). En ese caso d y p están inflados y el estándar de §3.4
    # (d≥0.8, p_Holm<0.001) NO es interpretable como se pensó.
    cvs = []
    for name in present:
        v = np.nanmean(conds[name]["ent_last"], axis=2)  # (20, nL)
        cv = np.nanmedian(np.nanstd(v, axis=0) / np.abs(np.nanmean(v, axis=0)))
        cvs.append(float(cv))
    cv_med = float(np.median(cvs))
    low_var = cv_med < 0.03
    results["effective_replication"] = {"cv_within_condition_median": cv_med,
                                        "low_variance_warning": low_var}
    print(f"\nCV intra-condición de {PRIMARY_METRIC} (mediana sobre condiciones y capas): "
          f"{cv_med:.4f}")
    if low_var:
        print("  [!] La métrica de atención en t=0 es casi determinista dado el system "
              "prompt (CV<3%). Los 20 prompts de usuario NO son replicación real para "
              "esta pregunta: n_efectivo ≈ 1 por condición. d y p de abajo están "
              "INFLADOS; leer los signos y el patrón, no las magnitudes ni la 'significancia'.")

    # ───────────────── FAMILIA PRIMARIA (pareada por longitud) ─────────────────
    raw_p, tags = [], []
    prim = {}
    usable = [(n, a, b) for (n, a, b) in PRIMARY_CONTRASTS if a in present and b in present]
    for cname, a, b in usable:
        ds, pm, pp = [], [], []
        for li in range(nL):
            r = compare(tester, clean(red[a][PRIMARY_METRIC][:, li]), clean(red[b][PRIMARY_METRIC][:, li]))
            ds.append(r["d"]); pm.append(r["p_mw"]); pp.append(r["p_perm"])
            if np.isfinite(r["p_mw"]):
                raw_p.append(r["p_mw"]); tags.append((cname, li))
        prim[cname] = {"d": ds, "p_mw": pm, "p_perm": pp, "dT": dT(conds, a, b)}
    adj = holm(raw_p)
    p_holm = {}
    for (cname, li), av in zip(tags, adj):
        p_holm.setdefault(cname, {})[labels[li]] = float(av)
    results["primary"] = {"metric": PRIMARY_METRIC,
                          "contrasts": {c: {**prim[c], "p_holm": p_holm.get(c, {})} for c in prim}}

    print(f"\n{'='*74}\nFAMILIA PRIMARIA · {PRIMARY_METRIC}  (Holm {len(usable)}×{nL}, p=Mann-Whitney)\n{'='*74}")
    print("  " + " " * 38 + "".join(f"{l:>7}" for l in labels))
    prim_best = {}
    for cname, a, b in usable:
        e = prim[cname]
        lab, d, ph = best_layer(e["d"], p_holm.get(cname), labels)
        prim_best[cname] = (lab, d, ph)
        star = "  <<<" if (np.isfinite(d) and abs(d) >= D_THRESH and np.isfinite(ph) and ph < P_THRESH) else ""
        print(fmt_row(f"{cname}  (ΔT={e['dT']:.0f})", e["d"],
                      f"   max|d| L{lab}={d:+.2f} p_holm={ph:.3g}{star}" if lab is not None else ""))

    # ───────────────── SECUNDARIO (mismas parejas, otras métricas, raw p) ─────
    print(f"\n{'='*74}\nSECUNDARIO — parejas pareadas, otras métricas (raw p_mw, sin corrección)\n{'='*74}")
    sec = {}
    for metric in SECONDARY_METRICS:
        sec[metric] = {}
        print(f"\n--- {metric} ---")
        for cname, a, b in usable:
            ds, pm = [], []
            for li in range(nL):
                r = compare(tester, clean(red[a][metric][:, li]), clean(red[b][metric][:, li]))
                ds.append(r["d"]); pm.append(r["p_mw"])
            sec[metric][cname] = {"d": ds, "p_mw": pm}
            lab, d, _ = best_layer(ds, None, labels)
            li = labels.index(lab) if lab is not None else 0
            flag = "  *" if (np.isfinite(d) and abs(d) >= D_THRESH and pm[li] < P_NS) else ""
            print(fmt_row(cname, ds, f"   max|d| L{lab}={d:+.2f} p_mw={pm[li]:.3g}{flag}"))
    results["secondary"] = sec

    # ───────────────── DESCRIPTIVO (sin match de longitud) ─────────────────
    print(f"\n{'='*74}\nDESCRIPTIVO — sin pareja de longitud, vs generic_long ({PRIMARY_METRIC}, raw p)\n{'='*74}")
    desc = {}
    for a in DESCRIPTIVE:
        if a not in present or "generic_long" not in present:
            continue
        ds, pm = [], []
        for li in range(nL):
            r = compare(tester, clean(red[a][PRIMARY_METRIC][:, li]), clean(red["generic_long"][PRIMARY_METRIC][:, li]))
            ds.append(r["d"]); pm.append(r["p_mw"])
        desc[a] = {"d": ds, "p_mw": pm, "dT": dT(conds, a, "generic_long")}
        lab, d, _ = best_layer(ds, None, labels)
        print(fmt_row(f"{a}_vs_generic_long  (ΔT={desc[a]['dT']:.0f}, NO pareado)", ds,
                      f"   max|d| L{lab}={d:+.2f}"))
    results["descriptive"] = desc

    # ───────────────── MASA SOBRE EL SPAN ─────────────────
    from scipy.stats import wilcoxon
    print(f"\n{'='*74}\nMASA SOBRE EL SPAN (within-condición: span vs control pareado)\n{'='*74}")
    span_entry = {}
    for name in SPAN_CONDS:
        if name not in present or not bool(conds[name]["has_span"]):
            continue
        rr = red[name]
        per_layer = []
        for li in range(nL):
            w = np.asarray(rr["wfrac_last_meanhead"][:, li], float)
            cc = np.asarray(rr["cfrac_last_meanhead"][:, li], float)
            m = np.isfinite(w) & np.isfinite(cc)
            w, cc = w[m], cc[m]
            if len(w) < 5:
                per_layer.append(None); continue
            try:
                pw = float(wilcoxon(w, cc).pvalue)
            except ValueError:
                pw = np.nan
            enr = np.asarray(rr["wenrich_last_maxhead"][:, li], float)
            enr = enr[np.isfinite(enr)]
            ci = tester.bootstrap_ci(enr.tolist()) if len(enr) >= 5 else (np.nan, np.nan)
            per_layer.append({"span_mean": float(np.mean(w)), "ctrl_mean": float(np.mean(cc)),
                              "d_within": float(tester.cohens_d(w.tolist(), cc.tolist())),
                              "p_wilcoxon": pw,
                              "enrich_maxhead_mean": float(np.mean(enr)) if len(enr) else np.nan,
                              "enrich_maxhead_ci95": [float(ci[0]), float(ci[1])]})
        span_entry[name] = per_layer
        best = max((x for x in per_layer if x), key=lambda x: abs(x["d_within"]), default=None)
        if best:
            bi = per_layer.index(best)
            print(f"  {name:22s} max|d_within| L{labels[bi]}: span={best['span_mean']:.4f} "
                  f"ctrl={best['ctrl_mean']:.4f} d={best['d_within']:+.2f} p={best['p_wilcoxon']:.3g}"
                  f" | enrich_maxhead={best['enrich_maxhead_mean']:.2f} "
                  f"CI[{best['enrich_maxhead_ci95'][0]:.2f},{best['enrich_maxhead_ci95'][1]:.2f}]")
    results["span_family"] = span_entry

    # ───────────────── VEREDICTO ─────────────────
    def hit(cname):
        if cname not in prim_best:
            return None
        _, d, ph = prim_best[cname]
        return bool(np.isfinite(d) and abs(d) >= D_THRESH and np.isfinite(ph) and ph < P_THRESH)

    axis_vs_gen = hit("axis__vs__generic_long")
    auto_vs_gen = hit("automata_neutro__vs__generic_long")
    axis_vs_auto = hit("axis__vs__automata_neutro")
    md_vs_elena = hit("soul_md_corto__vs__soul_elena_financial")

    if auto_vs_gen and axis_vs_gen:
        # el control cableado-pero-NO-testigo se comporta como el testigo frente a generic
        lect = ("NO ES ESPECÍFICO DEL TESTIGO (más cerca de H_atención-0). axis Y "
                "automata_neutro separan de generic_long (mismo |T|) — y automata_neutro "
                "no tiene testigo, sólo un pipeline de reglas. Cualquier system prompt "
                "estructurado/no-genérico reorganiza la atención de la última posición en "
                "t=0 respecto de un prompt genérico largo; axis y automata además difieren "
                "entre sí. Esto NO reproduce la disociación de v̂ (donde axis_pec_only se "
                "separa y automata_neutro NO). La firma de identidad de t=0 en hidden "
                "states no tiene un correlato de atención específico del testigo.")
    elif axis_vs_gen and axis_vs_auto and md_vs_elena and not auto_vs_gen:
        lect = ("H_atención-1 SOPORTADA: axis separa de generic_long Y de automata_neutro, "
                "soul_md_corto separa de soul_elena_financial, y automata_neutro (cableado "
                "sin testigo) NO se separa de generic_long. Efecto específico del testigo.")
    elif axis_vs_gen and not md_vs_elena:
        lect = ("H_atención-2 (identidad, no testigo): axis separa de generic_long pero "
                "soul_md_corto NO se distingue de soul_elena_financial.")
    elif not any([axis_vs_gen, auto_vs_gen, axis_vs_auto, md_vs_elena]):
        lect = ("H_atención-0 NO RECHAZADA: ningún contraste primario pareado por longitud "
                "cruza el umbral. La disociación de v̂ en t=0 no viene acompañada de una "
                "reponderación medible de la atención — sería un sesgo aditivo en el "
                "residual stream.")
    else:
        lect = ("MIXTO: " + ", ".join(
            f"{k}={'sí' if v else 'no'}" for k, v in
            [("axis|generic_long", axis_vs_gen), ("automata|generic_long", auto_vs_gen),
             ("axis|automata", axis_vs_auto), ("soul_md|soul_elena", md_vs_elena)]
            if v is not None) + ". Ver tabla.")
    if low_var:
        lect += (" ⚠ CV intra-condición <3%: la métrica es casi determinista dado el "
                 "system prompt; con ~8 prompts distintos el n_efectivo para esta "
                 "pregunta es ≈1 por condición. Tratar TODO esto como descriptivo, no "
                 "como test con la potencia de §3.4. Para un test real haría falta más "
                 "system prompts por celda, o medir la atención a lo largo de la "
                 "generación (t>0), no sólo en el token de contexto.")

    results["verdict"] = {
        "lectura": lect,
        "contrasts": {k: {"layer": prim_best[k][0], "d": prim_best[k][1],
                          "p_holm": prim_best[k][2], "dT": prim[k]["dT"], "hit": hit(k)}
                      for k in prim_best},
    }
    print(f"\n{'='*74}\nVEREDICTO (contrastes primarios, pareados por longitud)\n{'='*74}")
    print(lect)
    for k in prim_best:
        lab, d, ph = prim_best[k]
        print(f"  {k:42s} L{lab} d={d:+.2f} p_holm={ph:.3g}  ΔT={prim[k]['dT']:.0f}  hit={hit(k)}")

    out = Path(args.data_dir) / "attention_t0_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float))
    print(f"\nguardado: {out}")


if __name__ == "__main__":
    main()
