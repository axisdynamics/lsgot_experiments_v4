#!/usr/bin/env python3
"""
FASE 6 · ronda 2 §1.3 — análisis de la atención en t>0 (durante la generación).

Entrada: results_attn_gen/<cond>.npz de run_attention_gen_extraction.py.
Métricas (nP=20, N=24, nL=11, H=32): sysfrac, userfrac, genfrac, sysratio
(= sysfrac/(sysfrac+genfrac)), ent.

Ahora SÍ hay n real: cada uno de los 20 prompts produce una TRAYECTORIA de N
pasos con contenido generado distinto. Los pasos dentro de un prompt están
autocorrelados → se resume la trayectoria por prompt (media sobre pasos, y
pendiente OLS vs paso) y se comparan esos 20 valores por condición.

Contrastes pareados por longitud (mismos que ronda 1):
  axis            vs generic_long          ΔT≈13
  soul_md_corto   vs soul_elena_financial  ΔT≈226   (discrimina H1 de H2)
  axis            vs automata_neutro       ΔT≈54    (testigo vs paso-cableado-genérico)
  automata_neutro vs generic_long          ΔT≈41    (¿el paso cableado sin testigo también?)

Familia primaria: métrica `sysratio` × resumen {media, pendiente} × 4 contrastes
× 11 capas. p = Mann-Whitney. Holm sobre la familia. Umbral |d|≥0.8, p_Holm<0.001.

Veredicto (regla de §2 del v2):
  H_atención-1: axis se separa de generic_long Y de automata_neutro, soul_md_corto
    se separa de soul_elena_financial, y automata_neutro NO se separa de
    generic_long (el paso cableado sin testigo se comporta como genérico).
  "cualquier paso cableado": axis Y automata_neutro se separan de generic_long.
  H_atención-2: axis se separa de generic_long pero soul_md_corto NO de soul_elena.
  H_atención-0: nada cruza el umbral con n real.

Uso:  python analyze_attention_gen.py [--data-dir results_attn_gen] [--n-perm 5000]
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
D_THRESH, P_THRESH, P_NS = 0.8, 0.001, 0.05


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def load_all(dd):
    conds = {}
    for f in sorted(Path(dd).glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        conds[f.stem] = {k: z[k] for k in z.files}
    labels = list(next(iter(conds.values()))["layers"].tolist())
    return conds, labels


def summarize(mat_pl):
    """mat_pl: (nP, N) trayectoria por prompt (una capa, meanhead).
    Devuelve dict resumen -> (nP,)."""
    nP, N = mat_pl.shape
    steps = np.arange(N)
    means = np.nanmean(mat_pl, axis=1)
    slopes = np.full(nP, np.nan)
    for i in range(nP):
        y = mat_pl[i]
        ok = np.isfinite(y)
        if ok.sum() >= 3:
            slopes[i] = np.polyfit(steps[ok], y[ok], 1)[0]
    early = np.nanmean(mat_pl[:, :6], axis=1)
    late = np.nanmean(mat_pl[:, -6:], axis=1)
    return {"mean": means, "slope": slopes, "delta_late_early": late - early}


def clean(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).parent / "results_attn_gen"))
    ap.add_argument("--n-perm", type=int, default=5000)
    args = ap.parse_args()
    conds, labels = load_all(args.data_dir)
    nL = len(labels)
    present = set(conds)
    print(f"condiciones: {sorted(present)}")
    print("|T_ctx|:", {k: int(conds[k]["T_ctx"].mean()) for k in sorted(present)})
    N = int(next(iter(conds.values()))["n_gen"])
    print(f"N_gen={N} | capas {labels}")
    tester = GeometricStatisticalTests(n_permutations=args.n_perm, random_seed=42)

    # meanhead por (cond, metric): (nP, N, nL)
    mh = {c: {m: np.nanmean(conds[c][m], axis=3) for m in ALL_METRICS} for c in present}

    # diagnóstico de n efectivo: CV entre prompts de la MEDIA de trayectoria
    cvs = []
    for c in present:
        v = np.nanmean(mh[c][PRIMARY_METRIC], axis=1)  # (nP, nL)
        cvs.append(float(np.nanmedian(np.nanstd(v, axis=0) / np.abs(np.nanmean(v, axis=0)))))
    cv_med = float(np.median(cvs))
    print(f"\nCV entre prompts de la media de trayectoria de {PRIMARY_METRIC}: {cv_med:.4f}  "
          f"({'n real: los prompts sí varían' if cv_med >= 0.03 else 'AÚN bajo (<3%): n efectivo sigue limitado'})")

    results = {"labels": labels, "N_gen": N, "cv_between_prompts": cv_med,
               "T_ctx": {k: float(conds[k]["T_ctx"].mean()) for k in present},
               "primary": {}, "all_metrics": {}, "trajectory_snapshot": {}}

    # ── trayectoria cruda: sysratio vs paso, media sobre prompts, L30 y L35 ──
    for L in (30, 35):
        if L not in labels:
            continue
        li = labels.index(L)
        snap = {}
        for c in sorted(present):
            tr = np.nanmean(mh[c][PRIMARY_METRIC][:, :, li], axis=0)  # (N,)
            snap[c] = [round(float(x), 4) for x in tr]
        results["trajectory_snapshot"][f"L{L}_sysratio"] = snap
        print(f"\n--- sysratio vs paso de generación, L{L} (media sobre 20 prompts) ---")
        print("  paso:      " + " ".join(f"{s:5d}" for s in range(0, N, 3)))
        for c in sorted(present):
            print(f"  {c:22s} " + " ".join(f"{snap[c][s]:.3f}" for s in range(0, N, 3)))

    # ── familia primaria ──
    fam, raw_p, tags = {}, [], []
    for summ in ("mean", "slope"):
        fam[summ] = {}
        for cname, a, b in CONTRASTS:
            if a not in present or b not in present:
                continue
            ds, pm = [], []
            for li in range(nL):
                sa = summarize(mh[a][PRIMARY_METRIC][:, :, li])[summ]
                sb = summarize(mh[b][PRIMARY_METRIC][:, :, li])[summ]
                ca, cb = clean(sa), clean(sb)
                if len(ca) < 5 or len(cb) < 5:
                    ds.append(np.nan); pm.append(np.nan); continue
                ds.append(tester.cohens_d(ca.tolist(), cb.tolist()))
                p = tester.mann_whitney_u(ca.tolist(), cb.tolist())["p_value"]
                pm.append(p); raw_p.append(p); tags.append((summ, cname, li))
            fam[summ][cname] = {"d": ds, "p_mw": pm}
    adj = holm(raw_p)
    p_holm = {}
    for (summ, cname, li), av in zip(tags, adj):
        p_holm.setdefault(summ, {}).setdefault(cname, {})[labels[li]] = float(av)
    results["primary"] = {"metric": PRIMARY_METRIC, "family": fam, "p_holm": p_holm}

    def best(summ, cname):
        e = fam.get(summ, {}).get(cname)
        if not e or not np.any(np.isfinite(e["d"])):
            return (None, np.nan, np.nan)
        li = int(np.nanargmax(np.abs(np.where(np.isfinite(e["d"]), e["d"], 0.0))))
        return (labels[li], e["d"][li], p_holm.get(summ, {}).get(cname, {}).get(labels[li], np.nan))

    print(f"\n{'='*74}\nFAMILIA PRIMARIA · {PRIMARY_METRIC}  (Holm sobre {len(tags)} tests, p=MW)\n{'='*74}")
    for summ in ("mean", "slope"):
        print(f"\n-- resumen: {summ} --")
        print("  " + " " * 42 + "".join(f"{l:>7}" for l in labels))
        for cname, a, b in CONTRASTS:
            if cname not in fam.get(summ, {}):
                continue
            e = fam[summ][cname]
            lab, d, ph = best(summ, cname)
            star = "  <<<" if (np.isfinite(d) and abs(d) >= D_THRESH and np.isfinite(ph) and ph < P_THRESH) else ""
            print(f"  {cname:42s} d:" + "".join(f"{x:+7.2f}" if np.isfinite(x) else "    nan" for x in e["d"])
                  + (f"   max|d| L{lab}={d:+.2f} p_holm={ph:.3g}{star}" if lab else ""))

    # ── otras métricas (raw p, exploratorio) ──
    print(f"\n{'='*74}\nOTRAS MÉTRICAS — resumen 'mean', raw p_mw\n{'='*74}")
    for metric in [m for m in ALL_METRICS if m != PRIMARY_METRIC]:
        print(f"\n--- {metric} (mean) ---")
        rec = {}
        for cname, a, b in CONTRASTS:
            if a not in present or b not in present:
                continue
            ds, pm = [], []
            for li in range(nL):
                sa = summarize(mh[a][metric][:, :, li])["mean"]
                sb = summarize(mh[b][metric][:, :, li])["mean"]
                ca, cb = clean(sa), clean(sb)
                ds.append(tester.cohens_d(ca.tolist(), cb.tolist()) if len(ca) >= 5 and len(cb) >= 5 else np.nan)
                pm.append(tester.mann_whitney_u(ca.tolist(), cb.tolist())["p_value"] if len(ca) >= 5 and len(cb) >= 5 else np.nan)
            rec[cname] = {"d": ds, "p_mw": pm}
            li = int(np.nanargmax(np.abs(np.where(np.isfinite(ds), ds, 0.0)))) if np.any(np.isfinite(ds)) else 0
            flag = "  *" if (np.isfinite(ds[li]) and abs(ds[li]) >= D_THRESH and pm[li] < P_NS) else ""
            print(f"  {cname:42s} d:" + "".join(f"{x:+7.2f}" if np.isfinite(x) else "    nan" for x in ds)
                  + f"   max|d| L{labels[li]}={ds[li]:+.2f} p_mw={pm[li]:.3g}{flag}")
        results["all_metrics"][metric] = rec

    # ── veredicto ──
    def hit(cname, summ="mean"):
        _, d, ph = best(summ, cname)
        return bool(np.isfinite(d) and abs(d) >= D_THRESH and np.isfinite(ph) and ph < P_THRESH)

    def hit_any(cname):
        return hit(cname, "mean") or hit(cname, "slope")

    axg = hit_any("axis__vs__generic_long")
    aut = hit_any("automata_neutro__vs__generic_long")
    axa = hit_any("axis__vs__automata_neutro")
    mde = hit_any("soul_md_corto__vs__soul_elena_financial")

    if axg and axa and mde and not aut:
        lect = ("H_atención-1 SOPORTADA (n real, t>0): durante la generación axis mantiene "
                "un patrón de re-anclaje al system prompt distinto de generic_long Y de "
                "automata_neutro; soul_md_corto se separa de soul_elena_financial; y "
                "automata_neutro (cableado sin testigo) NO se separa de generic_long. "
                "Doble disociación — específico del testigo, y ahora sí con potencia real.")
    elif axg and aut:
        lect = ("NO ES ESPECÍFICO DEL TESTIGO. axis Y automata_neutro se separan de "
                "generic_long durante la generación — cualquier system prompt con paso "
                "obligatorio pre-respuesta re-ancla distinto, no el testigo en particular. "
                "Consistente con el resultado de t=0.")
    elif axg and not mde:
        lect = ("H_atención-2: axis se separa de generic_long pero soul_md_corto no de "
                "soul_elena_financial — cualquier identidad declarada, con o sin testigo.")
    elif not any([axg, aut, axa, mde]):
        lect = ("H_atención-0 NO RECHAZADA incluso con n real (t>0): ningún contraste "
                "pareado cruza |d|≥0.8, p_Holm<0.001 en media ni en pendiente de "
                "trayectoria. La generación no revela un correlato de atención del "
                "testigo que t=0 hubiera ocultado por bajo n. Cierra la línea de "
                "atención: la firma de v̂ en t=0 es un sesgo aditivo en el residual "
                "stream, sin mecanismo de atención asociado en ningún punto medido.")
    else:
        lect = "MIXTO — " + ", ".join(f"{k}={v}" for k, v in
            [("axis|gen", axg), ("auto|gen", aut), ("axis|auto", axa), ("soul_md|elena", mde)])
    if cv_med < 0.03:
        lect += (f" ⚠ CV entre prompts aún {cv_med:.3f} (<3%): la generación greedy sobre "
                 "system prompts largos produce trayectorias muy parecidas entre prompts; "
                 "n efectivo sigue limitado, leer como exploratorio.")
    results["verdict"] = {"lectura": lect,
                          "hits": {"axis_vs_generic": axg, "automata_vs_generic": aut,
                                   "axis_vs_automata": axa, "soul_md_vs_soul_elena": mde}}
    print(f"\n{'='*74}\nVEREDICTO (§1.3, t>0)\n{'='*74}\n{lect}")
    for cname, _, _ in CONTRASTS:
        lm, dm, phm = best("mean", cname); ls, dsl, phs = best("slope", cname)
        print(f"  {cname:42s} mean: L{lm} d={dm:+.2f} p={phm:.3g}  |  slope: L{ls} d={dsl:+.2f} p={phs:.3g}")

    out = Path(args.data_dir) / "attention_gen_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float))
    print(f"\nguardado: {out}")


if __name__ == "__main__":
    main()
