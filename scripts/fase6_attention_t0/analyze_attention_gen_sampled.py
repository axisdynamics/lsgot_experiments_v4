#!/usr/bin/env python3
"""
FASE 6 · ronda 2 §5.3 — análisis de la atención en t>0 con GENERACIÓN POR SAMPLING.

Entrada: results_attn_gen_samp/<cond>.npz de run_attention_gen_sampled.py.
Métricas (nP=20, K=6, N=24, nL=11, H=32): sysfrac, userfrac, genfrac,
sysratio (= sysfrac/(sysfrac+genfrac)), ent.

El sampling debería aportar la varianza que greedy no dio (§1.3 falló su
premisa). Dos vistas:
  - per_sample : cada (prompt, muestra) es una unidad → 120 por condición.
                 Más potencia, pseudo-replicación leve (6 muestras comparten prompt).
  - per_prompt : media sobre las 6 muestras → 20 por condición. Conservador.

Diagnóstico central: ¿el sampling añadió varianza?
  CV_entre_prompts   (std de las medias por prompt / media)
  CV_entre_muestras  (media sobre prompts de std_k / media)  — si ~0, el
                     contenido generado no mueve la métrica: sigue siendo
                     casi una función del system prompt (n efectivo bajo).

Contrastes pareados por longitud (mismos que ronda 1):
  axis vs generic_long · soul_md_corto vs soul_elena_financial ·
  axis vs automata_neutro · automata_neutro vs generic_long

Familia primaria: sysratio × {per_sample, per_prompt} × 4 contrastes × 11 capas.
Holm sobre la familia (p=Mann-Whitney). Umbral |d|≥0.8, p_Holm<0.001.

Veredicto igual que §1.3.  Uso: python analyze_attention_gen_sampled.py [--n-perm 5000]
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

PRIMARY_METRIC = "sysratio"
ALL_METRICS = ["sysratio", "sysfrac", "genfrac", "ent"]
CONTRASTS = [
    ("axis__vs__generic_long", "axis", "generic_long"),
    ("soul_md_corto__vs__soul_elena_financial", "soul_md_corto", "soul_elena_financial"),
    ("axis__vs__automata_neutro", "axis", "automata_neutro"),
    ("automata_neutro__vs__generic_long", "automata_neutro", "generic_long"),
]
D_THRESH, P_THRESH = 0.8, 0.001


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    o = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).parent / "results_attn_gen_samp"))
    ap.add_argument("--n-perm", type=int, default=5000)
    args = ap.parse_args()
    dd = Path(args.data_dir)
    conds = {}
    for f in sorted(dd.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        conds[f.stem] = {k: z[k] for k in z.files}
    labels = list(next(iter(conds.values()))["layers"].tolist())
    nL = len(labels)
    present = set(conds)
    K = int(next(iter(conds.values()))["k_samples"])
    N = int(next(iter(conds.values()))["n_gen"])
    print(f"condiciones: {sorted(present)} | K={K} N_gen={N} | capas {labels}")
    print("|T_ctx|:", {k: int(conds[k]["T_ctx"].mean()) for k in sorted(present)})
    tester = GeometricStatisticalTests(n_permutations=args.n_perm, random_seed=42)

    # meanhead + media de trayectoria: (nP, K, nL) por (cond, metric)
    tm = {c: {m: np.nanmean(np.nanmean(conds[c][m], axis=4), axis=2) for m in ALL_METRICS}
          for c in present}

    # ── diagnóstico de varianza ──
    print(f"\n{'='*72}\nDIAGNÓSTICO — ¿el sampling añadió varianza? ({PRIMARY_METRIC})\n{'='*72}")
    print(f"  {'condición':22s}  CV_entre_prompts   CV_entre_muestras")
    diag = {}
    for c in sorted(present):
        v = tm[c][PRIMARY_METRIC]                          # (nP, K, nL)  media de trayectoria
        pp = np.nanmean(v, axis=1)                         # (nP, nL)  media sobre muestras
        cvp = float(np.nanmedian(np.nanstd(pp, axis=0) / np.abs(np.nanmean(pp, axis=0))))
        # std entre las K muestras de cada prompt, / media; mediana sobre prompts y capas
        cvs = float(np.nanmedian(np.nanstd(v, axis=1) / np.abs(np.nanmean(v, axis=1))))
        diag[c] = {"cv_between_prompts": cvp, "cv_between_samples": cvs}
        print(f"  {c:22s}  {cvp:14.4f}   {cvs:14.4f}")
    cv_s_med = float(np.median([d["cv_between_samples"] for d in diag.values()]))
    sampling_helped = cv_s_med >= 0.02
    print(f"\n  CV_entre_muestras (mediana sobre condiciones): {cv_s_med:.4f} — "
          f"{'el contenido generado SÍ mueve la métrica' if sampling_helped else 'el contenido generado NO mueve la métrica: n efectivo sigue bajo'}")

    results = {"labels": labels, "K": K, "N_gen": N, "diagnostic": diag,
               "cv_between_samples_median": cv_s_med, "primary": {}, "all_metrics": {}}

    # ── familia primaria ──
    fam, raw_p, tags = {}, [], []
    for view in ("per_sample", "per_prompt"):
        fam[view] = {}
        for cname, a, b in CONTRASTS:
            if a not in present or b not in present:
                continue
            ds, pm = [], []
            for li in range(nL):
                va = tm[a][PRIMARY_METRIC][:, :, li].ravel() if view == "per_sample" else np.nanmean(tm[a][PRIMARY_METRIC][:, :, li], axis=1)
                vb = tm[b][PRIMARY_METRIC][:, :, li].ravel() if view == "per_sample" else np.nanmean(tm[b][PRIMARY_METRIC][:, :, li], axis=1)
                va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
                if len(va) < 5 or len(vb) < 5:
                    ds.append(np.nan); pm.append(np.nan); continue
                ds.append(tester.cohens_d(va.tolist(), vb.tolist()))
                p = tester.mann_whitney_u(va.tolist(), vb.tolist())["p_value"]
                pm.append(p); raw_p.append(p); tags.append((view, cname, li))
            fam[view][cname] = {"d": ds, "p_mw": pm}
    adj = holm(raw_p)
    p_holm = {}
    for (view, cname, li), av in zip(tags, adj):
        p_holm.setdefault(view, {}).setdefault(cname, {})[labels[li]] = float(av)
    results["primary"] = {"metric": PRIMARY_METRIC, "family": fam, "p_holm": p_holm}

    def best(view, cname):
        e = fam.get(view, {}).get(cname)
        if not e or not np.any(np.isfinite(e["d"])):
            return (None, np.nan, np.nan)
        li = int(np.nanargmax(np.abs(np.where(np.isfinite(e["d"]), e["d"], 0.0))))
        return (labels[li], e["d"][li], p_holm.get(view, {}).get(cname, {}).get(labels[li], np.nan))

    for view in ("per_sample", "per_prompt"):
        print(f"\n{'='*72}\nFAMILIA PRIMARIA · {PRIMARY_METRIC} · {view}\n{'='*72}")
        print("  " + " " * 42 + "".join(f"{l:>7}" for l in labels))
        for cname, a, b in CONTRASTS:
            if cname not in fam.get(view, {}):
                continue
            e = fam[view][cname]
            lab, d, ph = best(view, cname)
            star = "  <<<" if (np.isfinite(d) and abs(d) >= D_THRESH and np.isfinite(ph) and ph < P_THRESH) else ""
            print(f"  {cname:42s} d:" + "".join(f"{x:+7.2f}" if np.isfinite(x) else "    nan" for x in e["d"])
                  + (f"   max|d| L{lab}={d:+.2f} p_holm={ph:.3g}{star}" if lab else ""))

    # ── otras métricas (per_prompt, raw p) ──
    print(f"\n{'='*72}\nOTRAS MÉTRICAS · per_prompt, raw p_mw\n{'='*72}")
    for metric in [m for m in ALL_METRICS if m != PRIMARY_METRIC]:
        print(f"\n--- {metric} ---")
        rec = {}
        for cname, a, b in CONTRASTS:
            if a not in present or b not in present:
                continue
            ds, pm = [], []
            for li in range(nL):
                va = np.nanmean(tm[a][metric][:, :, li], axis=1)
                vb = np.nanmean(tm[b][metric][:, :, li], axis=1)
                va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
                ds.append(tester.cohens_d(va.tolist(), vb.tolist()) if len(va) >= 5 and len(vb) >= 5 else np.nan)
                pm.append(tester.mann_whitney_u(va.tolist(), vb.tolist())["p_value"] if len(va) >= 5 and len(vb) >= 5 else np.nan)
            rec[cname] = {"d": ds, "p_mw": pm}
            li = int(np.nanargmax(np.abs(np.where(np.isfinite(ds), ds, 0.0)))) if np.any(np.isfinite(ds)) else 0
            print(f"  {cname:42s} d:" + "".join(f"{x:+7.2f}" if np.isfinite(x) else "    nan" for x in ds)
                  + f"   max|d| L{labels[li]}={ds[li]:+.2f} p_mw={pm[li]:.3g}")
        results["all_metrics"][metric] = rec

    # ── veredicto ──
    def hit(cname, view="per_prompt"):
        _, d, ph = best(view, cname)
        return bool(np.isfinite(d) and abs(d) >= D_THRESH and np.isfinite(ph) and ph < P_THRESH)

    axg = hit("axis__vs__generic_long")
    aut = hit("automata_neutro__vs__generic_long")
    axa = hit("axis__vs__automata_neutro")
    mde = hit("soul_md_corto__vs__soul_elena_financial")

    if axg and axa and mde and not aut:
        lect = ("H_atención-1 SOPORTADA (t>0, sampling): axis se separa de generic_long y "
                "de automata_neutro, soul_md_corto de soul_elena_financial, y "
                "automata_neutro NO de generic_long.")
    elif axg and aut:
        lect = ("NO ES ESPECÍFICO DEL TESTIGO. axis Y automata_neutro se separan de "
                "generic_long durante la generación con sampling — igual que en t=0 y en "
                "t>0 greedy. Cualquier system prompt estructurado, no el testigo.")
    elif not any([axg, aut, axa, mde]):
        lect = ("H_atención-0 NO RECHAZADA (t>0, sampling): ningún contraste pareado cruza "
                "|d|≥0.8, p_Holm<0.001.")
    else:
        lect = "MIXTO — " + ", ".join(f"{k}={v}" for k, v in
            [("axis|gen", axg), ("auto|gen", aut), ("axis|auto", axa), ("soul_md|elena", mde)])
    if not sampling_helped:
        lect += (f" ⚠ El sampling NO añadió varianza (CV_entre_muestras {cv_s_med:.4f}): "
                 f"{PRIMARY_METRIC} es casi invariante al contenido generado — sigue siendo "
                 "función del system prompt. Los d siguen inflados; leer como en §1.3. "
                 "El fix §5.3 no rescató el n efectivo: la línea de atención queda cerrada "
                 "salvo §1.4 (pre-softmax) o una métrica genuinamente distinta.")
    results["verdict"] = {"lectura": lect, "hits": {"axis_vs_generic": axg,
                          "automata_vs_generic": aut, "axis_vs_automata": axa,
                          "soul_md_vs_soul_elena": mde}, "sampling_helped": sampling_helped}
    print(f"\n{'='*72}\nVEREDICTO (§5.3, t>0 sampling)\n{'='*72}\n{lect}")
    for cname, _, _ in CONTRASTS:
        lp, dp, php = best("per_prompt", cname)
        ls, dsm, phs = best("per_sample", cname)
        print(f"  {cname:42s} per_prompt: L{lp} d={dp:+.2f} p={php:.3g}  |  per_sample: L{ls} d={dsm:+.2f} p={phs:.3g}")

    out = dd / "attention_gen_sampled_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float))
    print(f"\nguardado: {out}")


if __name__ == "__main__":
    main()
