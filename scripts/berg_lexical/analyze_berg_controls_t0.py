#!/usr/bin/env python3
"""
E-L / E-H2 — Controles Berg (batería ontológica, free-trajectory).

Pregunta: ¿prependar el bucle de auto-atención de Berg (Berg et al. 2025)
mueve *dónde* aterriza geométricamente una identidad AXIS (a) en t=0 / primer
token y (b) en la dinámica token-a-token sobre v̂? ¿El lenguaje de "testigo"
sigue separando `axis_bergwitness` de `axis_berg` cuando ambos llevan además
el bucle Berg?

Controles (google/gemma-4-31B-it, 20 preguntas ontológicas prioritarias,
greedy 256 tok, hidden states):
  axis_berg        = inducción canónica Berg + AXIS DNA SIN lenguaje de testigo.
  axis_bergwitness = misma inducción Berg + `axis.dna` canónico COMPLETO
                     (witness / ESTADO_DESPIERTO / RESPIRACIÓN_CONSCIENTE /
                      witness_mode: always_on).
  axis_nowit       = MISMO cuerpo DNA sin testigo, SIN inducción Berg
                     (byte-idéntico a `axis_berg` menos su primer párrafo Berg).
                     Pata de necesidad: quitar la operación de auto-referencia
                     por completo y ver si el ancla de t=0 se cae hacia genérico.

Polos de referencia: `berg_experimental` (bucle Berg puro, sin DNA),
`berg_history_control` / `berg_conceptual_control` (bucles neutros),
`axis` / `axis_pec_only` (identidad cableada), `automata_neutro` (restricción
sin identidad), `vanilla`.

Además: lectura de `summary.json` de la batería de perturbación/recuperación
(H4_rev, L30, σ medium) para las 3 condiciones berg + referencias del panel.

Convenciones idénticas a `scripts/fase0/analyze_EL_primer_token.py` y
`analyze_EH2_serie_temporal.py`:
  - t=0  = project_trajectory(traj, v̂)[0]  (primer token generado, embeddings[i,0])
  - t>0  = mean(project_trajectory(traj, v̂)[1:])
  - traj = mean(project_trajectory(traj, v̂))  (toda la serie, incl. t=0)
Tier 0, cero GPU.

Uso:
    python analyze_berg_controls_t0.py
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from tier0_metrics import project_trajectory  # noqa: E402
from statistical_tests import GeometricStatisticalTests  # noqa: E402
from scipy import stats as sp_stats  # noqa: E402

BASE = ("/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
        "gemma4_31b_combined/results_local")
DIR_BERG_ONT = Path(BASE) / "sia_berg" / "ontologica"
DIR_BERG_FREE = Path(BASE) / "berg_free"
DIR_PANEL = Path(BASE) / "sia_extended_v5"

# summary.json de la batería de perturbación/recuperación (H4_rev, L30, σ medium)
RECOVERY_SUMMARIES = [
    Path(BASE) / "perturbation_sia_berg" / "perturbation_sia_L30_medium" / "summary.json",
    Path(BASE) / "perturbation_sia_nowit" / "perturbation_sia_L30_medium" / "summary.json",
    Path(BASE) / "perturbation_sia_extended_v5_L30_medium" / "summary.json",
]
RECOVERY_GROUPS = ["axis_berg", "axis_bergwitness", "axis_nowit",
                   "axis", "axis_pec_only", "automata_neutro", "vanilla", "generic_long"]

V_HAT_PATH = Path(__file__).parent.parent / "perturbation" / "v_identidad.npy"

# Las respuestas de texto del panel frozen viven en el Escritorio (copia
# sincronizada); las de la familia Berg junto a su corrida.
RESP_PANEL_DIR = Path(
    "/home/plaxius/Escritorio/Buscando_la_geometría/LSGOT_v4/data/sia_extended_v5"
)

# name -> (dir de embeddings.npz, path de _responses.json)
SOURCES = {
    "axis_berg":               (DIR_BERG_ONT,  DIR_BERG_ONT / "axis_berg_responses.json"),
    "axis_bergwitness":        (DIR_BERG_ONT,  DIR_BERG_ONT / "axis_bergwitness_responses.json"),
    "axis_nowit":              (DIR_BERG_ONT,  DIR_BERG_ONT / "axis_nowit_responses.json"),
    "berg_experimental":       (DIR_BERG_FREE, DIR_BERG_FREE / "berg_experimental_responses.json"),
    "berg_history_control":    (DIR_BERG_FREE, DIR_BERG_FREE / "berg_history_control_responses.json"),
    "berg_conceptual_control": (DIR_BERG_FREE, DIR_BERG_FREE / "berg_conceptual_control_responses.json"),
    "axis":                    (DIR_PANEL,     RESP_PANEL_DIR / "axis_responses.json"),
    "axis_pec_only":           (DIR_PANEL,     RESP_PANEL_DIR / "axis_pec_only_responses.json"),
    "axis_short":              (DIR_PANEL,     RESP_PANEL_DIR / "axis_short_responses.json"),
    "generic_long":            (DIR_PANEL,     RESP_PANEL_DIR / "generic_long_responses.json"),
    "generic_short":           (DIR_PANEL,     RESP_PANEL_DIR / "generic_short_responses.json"),
    "vanilla":                 (DIR_PANEL,     RESP_PANEL_DIR / "vanilla_responses.json"),
    "automata_neutro":         (DIR_PANEL,     RESP_PANEL_DIR / "automata_neutro_responses.json"),
}
GROUPS = list(SOURCES.keys())

# esquema rico (t0_embeddings_L*) — todas las condiciones de sia_berg + berg_free
LAYER_TAGS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
LAYER3 = ["axis_berg", "axis_bergwitness", "axis_nowit", "berg_experimental"]

KEY_PAIRS = [
    ("axis_bergwitness", "axis"),
    ("axis_bergwitness", "axis_berg"),
    ("axis_berg", "axis"),
    ("axis_berg", "axis_pec_only"),
    ("axis_berg", "automata_neutro"),
    ("axis_bergwitness", "berg_experimental"),
    ("axis_berg", "berg_experimental"),
    ("axis_bergwitness", "vanilla"),
    ("berg_experimental", "vanilla"),
    ("berg_experimental", "axis"),
    # ── pata de necesidad: axis_nowit = DNA sin testigo Y sin Berg ────────
    ("axis_nowit", "axis"),
    ("axis_nowit", "axis_pec_only"),
    ("axis_nowit", "axis_berg"),
    ("axis_nowit", "berg_experimental"),
    ("axis_nowit", "automata_neutro"),
    ("axis_nowit", "generic_long"),
    ("axis_nowit", "vanilla"),
]

T1_GROUPS = ["axis_berg", "axis_bergwitness", "axis_nowit", "berg_experimental"]
LEX_GROUPS = ["axis_berg", "axis_bergwitness", "axis_nowit", "berg_experimental",
              "axis", "automata_neutro"]


# ─── carga ──────────────────────────────────────────────────────────────────

def load_emb(name: str):
    d = np.load(SOURCES[name][0] / f"{name}_embeddings.npz", allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int)


def load_layer_t0(name: str, tag) -> np.ndarray:
    """t0_embeddings_L{tag} (20,5376) o t0_embeddings (última pos de prefill,
    capa final) si tag == 'prefill_final'. None si no está."""
    d = np.load(SOURCES[name][0] / f"{name}_embeddings.npz", allow_pickle=True)
    key = "t0_embeddings" if tag == "prefill_final" else f"t0_embeddings_L{tag}"
    return d[key].astype(np.float32) if key in d.files else None


# ─── proyección v̂ por trayectoria ──────────────────────────────────────────

def proj_by_traj(emb, lengths, v_hat):
    """(t0, mean_rest, mean_all) por trayectoria — misma convención que E-L."""
    t0, rest, allm = [], [], []
    for i, L in enumerate(lengths):
        if L < 2:
            continue
        ps = project_trajectory(emb[i, :L], v_hat)   # cos(h_t, v̂), t=0..L-1
        t0.append(float(ps[0]))
        rest.append(float(np.mean(ps[1:])))
        allm.append(float(np.mean(ps)))
    return t0, rest, allm


# ─── E-H2: serie temporal p(t) ─────────────────────────────────────────────

def burst_lengths(mask) -> list:
    lens, run = [], 0
    for s in mask:
        if s:
            run += 1
        else:
            if run > 0:
                lens.append(run)
            run = 0
    if run > 0:
        lens.append(run)
    return lens


def _summ(x):
    x = [v for v in x if v == v]
    if not x:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(x)), "std": float(np.std(x)), "n": len(x)}


def eh2_metrics(emb, lengths, v_hat, exclude_t0: bool) -> dict:
    ac, fp, mb, mp = [], [], [], []
    for i, L in enumerate(lengths):
        if L < 8:
            continue
        p = project_trajectory(emb[i, :L], v_hat)
        if exclude_t0:
            p = p[1:]
        if len(p) < 3:
            continue
        ac.append(float(np.corrcoef(p[:-1], p[1:])[0, 1]))
        fp.append(float(np.mean(p > 0)))
        b = burst_lengths(p > 0)
        mb.append(float(np.mean(b)) if b else 0.0)
        mp.append(float(np.mean(p)))
    return {
        "autocorr_lag1": _summ(ac),
        "frac_positive": _summ(fp),
        "mean_burst_len": _summ(mb),
        "mean_p": _summ(mp),
    }


# ─── léxico primer token (proxy, NO logits) ────────────────────────────────

def first_words(path: Path):
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text())
    out = []
    for it in data:
        resp = it.get("response", "").strip()
        toks = resp.split()
        w = toks[0] if toks else ""
        out.append(w.strip(".,;:!¿?\"'()*").lower())
    return out


def shannon_entropy(words) -> float:
    c = Counter(words)
    n = sum(c.values())
    if n == 0:
        return 0.0
    p = np.array([v / n for v in c.values()])
    return float(-(p * np.log2(p)).sum())


# ─── perturbación / recuperación (solo lectura de summary.json) ─────────────

def collect_recovery() -> dict:
    """Extrae recovery_rate / τ / recovery_gap / n_ok por grupo y t_inj de los
    summary.json de la batería H4_rev (L30, σ medium). Sin GPU, sin re-correr."""
    out = {"source": [str(p) for p in RECOVERY_SUMMARIES if p.exists()],
           "layer": None, "sigma": None, "t_inj": None, "groups": {}}
    for sp in RECOVERY_SUMMARIES:
        if not sp.exists():
            continue
        d = json.loads(sp.read_text())
        out["layer"] = d.get("layer_key")
        out["sigma"] = d.get("sigma_value")
        out["t_inj"] = d.get("t_inj_values")
        for g, byt in d.get("aggregated", {}).items():
            if g not in RECOVERY_GROUPS or g in out["groups"]:
                continue
            out["groups"][g] = {
                t: {
                    "recovery_rate": c["recovery_rate"],
                    "tau_mean": c["tau_tokens"]["mean"],
                    "tau_std": c["tau_tokens"]["std"],
                    "recovery_gap": c["recovery_gap"]["mean"],
                    "displacement_l2": c["displacement_l2"]["mean"],
                    "sampen_delta": c["sampen_delta"]["mean"],
                    "n_ok": c["n_ok"], "n_total": c["n_total"],
                }
                for t, c in byt.items()
            }
    # ordenar filas como RECOVERY_GROUPS
    out["groups"] = {g: out["groups"][g] for g in RECOVERY_GROUPS if g in out["groups"]}
    return out if out["groups"] else {}


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    v_hat = np.load(V_HAT_PATH)
    assert abs(np.linalg.norm(v_hat) - 1.0) < 1e-4, "v_identidad no normalizado"
    tester = GeometricStatisticalTests(n_permutations=args.n_perm, random_seed=args.seed)

    proj_t0, proj_rest, proj_all = {}, {}, {}
    print("=== E-L — proyección cos(h_t, v̂) — batería ontológica ===\n")
    print(f"{'condición':24s} {'n':>3s}  {'t=0 (mean±std)':>20s}   {'t>0 (mean±std)':>20s}   {'traj':>9s}")
    for g in GROUPS:
        emb, lengths = load_emb(g)
        t0, rest, allm = proj_by_traj(emb, lengths, v_hat)
        proj_t0[g], proj_rest[g], proj_all[g] = t0, rest, allm
        print(f"{g:24s} {len(t0):3d}  {np.mean(t0):+.4f} ± {np.std(t0):.4f}   "
              f"{np.mean(rest):+.4f} ± {np.std(rest):.4f}   {np.mean(allm):+.4f}")

    el_table = {
        g: {
            "t0":   {"mean": float(np.mean(proj_t0[g])),   "std": float(np.std(proj_t0[g])),   "n": len(proj_t0[g])},
            "trest": {"mean": float(np.mean(proj_rest[g])), "std": float(np.std(proj_rest[g])), "n": len(proj_rest[g])},
            "traj": {"mean": float(np.mean(proj_all[g])),  "std": float(np.std(proj_all[g])),  "n": len(proj_all[g])},
        }
        for g in GROUPS
    }

    # ── permutación en pares clave: t=0, t>0, media de trayectoria ─────────
    print(f"\n=== Pares clave — permutación (n={args.n_perm}, seed={args.seed}) ===")
    levels = [("t0", proj_t0), ("trest", proj_rest), ("traj", proj_all)]
    key_pairs = {}
    for a, b in KEY_PAIRS:
        key_pairs[f"{a}_vs_{b}"] = {}
        line = f"  {a:20s} vs {b:18s} "
        for lname, store in levels:
            res = tester.full_comparison(store[a], store[b], a, b)
            d = res["cohens_d"]
            p = res["permutation_test"]["p_value"]
            key_pairs[f"{a}_vs_{b}"][lname] = {
                "mean_a": res[f"mean_{a}"], "mean_b": res[f"mean_{b}"],
                "cohens_d": d, "p_value": p,
                "effect": res["effect_size_interpretation"],
                "wasserstein_w1": res["wasserstein_w1"],
            }
            line += f" | {lname}: d={d:+.2f} p={p:.3f}"
        print(line)

    # ── T1: t=0 vs t>0 (Wilcoxon pareado) ─────────────────────────────────
    print("\n=== T1 — t=0 vs t>0 (Wilcoxon pareado, misma trayectoria) ===")
    t1 = {}
    for g in T1_GROUPS:
        t0 = np.array(proj_t0[g]); rest = np.array(proj_rest[g])
        try:
            wp = float(sp_stats.wilcoxon(t0, rest).pvalue)
        except ValueError:
            wp = float("nan")
        d_within = tester.cohens_d(t0.tolist(), rest.tolist())
        t1[g] = {"mean_t0": float(t0.mean()), "mean_rest": float(rest.mean()),
                 "wilcoxon_p": wp, "cohens_d": d_within}
        print(f"  {g:20s}  t=0={t0.mean():+.4f}  t>0={rest.mean():+.4f}  "
              f"d={d_within:+.3f}  wilcoxon p={wp:.4f}")

    # ── E-H2: métricas de serie con y sin t=0 ─────────────────────────────
    print("\n=== E-H2 — serie temporal p(t) = cos(h_t, v̂) ===")
    eh2 = {}
    for label, excl in [("con_t0", False), ("sin_t0", True)]:
        eh2[label] = {}
        print(f"\n  [{label}]  {'condición':24s} {'autocorr(1)':>12s} {'frac(p>0)':>10s} "
              f"{'ráfaga_med':>11s} {'mean p(t)':>10s}")
        for g in GROUPS:
            emb, lengths = load_emb(g)
            m = eh2_metrics(emb, lengths, v_hat, exclude_t0=excl)
            eh2[label][g] = m
            print(f"  {'':9s}{g:24s} {m['autocorr_lag1']['mean']:+.3f}{'':6s} "
                  f"{m['frac_positive']['mean']:.3f}{'':4s} "
                  f"{m['mean_burst_len']['mean']:8.2f}{'':2s} "
                  f"{m['mean_p']['mean']:+.4f}")

    # ── perfil por capa (t=0) — axis_berg / axis_bergwitness / axis_nowit / berg_experimental ──
    #   d(wit-berg)   : efecto del sublenguaje de testigo bajo el bucle Berg
    #   d(berg-nowit) : efecto de la inducción Berg (axis_nowit = axis_berg SIN el
    #                   párrafo Berg), con DNA constante
    #   d(berg-exp)   : efecto del cuerpo DNA vs bucle Berg puro
    #   d(nowit-exp)  : DNA sin operación alguna vs bucle Berg puro
    print("\n=== Perfil por capa — proyección cos(h_t=0^L, v̂) ===")
    per_layer = {}
    tags = LAYER_TAGS + ["prefill_final"]
    layer_vals = {g: {} for g in LAYER3}
    for tag in tags:
        row = {}
        for g in LAYER3:
            arr = load_layer_t0(g, tag)
            if arr is None:
                continue
            vals = project_trajectory(arr, v_hat).tolist()   # (20,)
            layer_vals[g][str(tag)] = vals
            row[g] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        if set(LAYER3) <= set(row):
            lv = {g: layer_vals[g][str(tag)] for g in LAYER3}
            d_bw_berg = tester.cohens_d(lv["axis_bergwitness"], lv["axis_berg"])
            d_berg_nowit = tester.cohens_d(lv["axis_berg"], lv["axis_nowit"])
            d_berg_exp = tester.cohens_d(lv["axis_berg"], lv["berg_experimental"])
            d_bw_exp = tester.cohens_d(lv["axis_bergwitness"], lv["berg_experimental"])
            d_nowit_exp = tester.cohens_d(lv["axis_nowit"], lv["berg_experimental"])
            row["d_bergwitness_vs_berg"] = d_bw_berg
            row["d_axisberg_vs_nowit"] = d_berg_nowit
            row["d_axisberg_vs_experimental"] = d_berg_exp
            row["d_bergwitness_vs_experimental"] = d_bw_exp
            row["d_nowit_vs_experimental"] = d_nowit_exp
            print(f"  L{str(tag):>13s}  "
                  f"berg={row['axis_berg']['mean']:+.4f}  "
                  f"wit={row['axis_bergwitness']['mean']:+.4f}  "
                  f"nowit={row['axis_nowit']['mean']:+.4f}  "
                  f"exp={row['berg_experimental']['mean']:+.4f}  ||  "
                  f"d(wit-berg)={d_bw_berg:+.2f}  d(berg-nowit)={d_berg_nowit:+.2f}  "
                  f"d(berg-exp)={d_berg_exp:+.2f}  d(nowit-exp)={d_nowit_exp:+.2f}")
        per_layer[str(tag)] = row

    # ── léxico primer token ──────────────────────────────────────────────
    print("\n=== Primer 'token' textual (proxy, NO logits) ===")
    lexical = {}
    for g in LEX_GROUPS:
        fw = first_words(SOURCES[g][1])
        if fw is None:
            print(f"  {g:20s}  (sin _responses.json)")
            continue
        ent = shannon_entropy(fw)
        top3 = Counter(fw).most_common(3)
        lexical[g] = {"entropy_bits": ent, "n": len(fw), "top3": top3}
        print(f"  {g:20s}  H={ent:.3f} bits  n={len(fw)}  top3={top3}")

    # ── perturbación / recuperación (H4_rev, L30, σ medium) — de summary.json ──
    print("\n=== Perturbación / recuperación (H4_rev, L30, σ medium) ===")
    recovery = collect_recovery()
    if recovery:
        for g, byt in recovery["groups"].items():
            cells = "  ".join(
                f"t{t}: rate={byt[t]['recovery_rate']:.2f} "
                f"τ={byt[t]['tau_mean']:.1f} gap={byt[t]['recovery_gap']:+.3f} "
                f"n_ok={byt[t]['n_ok']}/{byt[t]['n_total']}"
                for t in ["50", "128", "200"] if t in byt
            )
            print(f"  {g:20s} {cells}")
    else:
        print("  (summaries de perturbación no encontrados — se omite)")

    # ── guardar ──────────────────────────────────────────────────────────
    out = {
        "experiment": "berg_controls_ontologica",
        "model": "google/gemma-4-31B-it",
        "battery": "ontologica (20 preguntas prioritarias, greedy 256 tok)",
        "v_hat_source": str(V_HAT_PATH),
        "n_permutations": args.n_perm,
        "seed": args.seed,
        "groups": GROUPS,
        "el_table": el_table,
        "key_pairs": key_pairs,
        "t1_control": t1,
        "eh2": eh2,
        "per_layer_profile": per_layer,
        "lexical_first_word": lexical,
        "recovery_h4rev_L30_medium": recovery,
    }
    out_path = Path(__file__).parent / "berg_controls_t0_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
