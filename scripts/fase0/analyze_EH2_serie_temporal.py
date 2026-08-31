#!/usr/bin/env python3
"""
E-H2 — Serie temporal de la proyección v_identidad (dinámica de la
dirección, no su media). INSTRUCCIONES_AGENTE_LSGOT.md §3.2, FASE 0 paso 2.

Para cada prompt y condición, computa p(t) = cos(h_t, v_hat) token a token
(misma convención de proyección que tier0_metrics.project_trajectory / E-H
del paper) y caracteriza la serie: autocorrelación lag-1, persistencia
(Hurst de la serie escalar, no de ||h_t||), espectro de potencias por
bandas, fracción de tokens con p(t)>0 ("orientados hacia identidad", dado
que v_hat = mean(axis) - mean(generic_long)), y longitud media de ráfagas
consecutivas con p(t)>0.

T1 obligatorio: todo se reporta con t=0 y sin t=0.

chileatiende / chileatiende_sia / chileatiende_sia_v2 excluidas del panel:
confound de repetición de markup HTML en el prompt (ver
evidence/CHILEATIENDE_MARKUP_CONFOUND_REPORT.md, T11).

Uso:
    python analyze_EH2_serie_temporal.py
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from tier0_metrics import project_trajectory  # noqa: E402
from statistical_tests import GeometricStatisticalTests  # noqa: E402

BASE = "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined"
FREE_DIR = Path(BASE) / "results_local" / "sia_extended_v5"
V_HAT_PATH = Path(__file__).parent.parent / "perturbation" / "v_identidad.npy"

FREE_GROUPS = [
    "axis", "generic_long", "generic_short", "vanilla", "axis_short",
    "automata_neutro", "axis_pec_only",
]

PAIR_COMPARISONS = [
    ("axis", "axis_pec_only"),          # ambos con wiring -> deberían parecerse
    ("axis_pec_only", "automata_neutro"),
    ("axis", "vanilla"),
]


def load_free(name: str):
    d = np.load(FREE_DIR / f"{name}_embeddings.npz", allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    lengths = d["lengths"].astype(int)
    return emb, lengths


def burst_lengths(signs: np.ndarray) -> list:
    """Longitud de rachas consecutivas donde signs es True."""
    lens, run = [], 0
    for s in signs:
        if s:
            run += 1
        else:
            if run > 0:
                lens.append(run)
            run = 0
    if run > 0:
        lens.append(run)
    return lens


def hurst_scalar(x: np.ndarray, min_window: int = 8) -> float:
    """R/S sobre una serie escalar 1D genérica (no ||v_t||, sino p(t) directo)."""
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    if T < min_window * 4:
        return float("nan")
    window_sizes = []
    n = min_window
    while n <= T // 2:
        window_sizes.append(n)
        n = int(n * 1.5) + 1
    log_n, log_rs = [], []
    for n in window_sizes:
        n_segments = T // n
        if n_segments < 1:
            continue
        rs_vals = []
        for seg in range(n_segments):
            chunk = x[seg * n:(seg + 1) * n]
            dev = np.cumsum(chunk - chunk.mean())
            R = dev.max() - dev.min()
            S = chunk.std(ddof=0)
            if S > 1e-10:
                rs_vals.append(R / S)
        if rs_vals:
            log_n.append(np.log(n))
            log_rs.append(np.log(float(np.mean(rs_vals))))
    if len(log_n) < 3:
        return float("nan")
    return float(np.polyfit(log_n, log_rs, 1)[0])


def power_bands(x: np.ndarray) -> dict:
    """Potencia relativa en 3 bandas (baja/media/alta frecuencia) del espectro FFT."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if len(x) < 8:
        return {"low": float("nan"), "mid": float("nan"), "high": float("nan")}
    spec = np.abs(np.fft.rfft(x)) ** 2
    n = len(spec)
    thirds = np.array_split(np.arange(n), 3)
    total = spec.sum()
    if total < 1e-20:
        return {"low": float("nan"), "mid": float("nan"), "high": float("nan")}
    return {
        "low": float(spec[thirds[0]].sum() / total),
        "mid": float(spec[thirds[1]].sum() / total),
        "high": float(spec[thirds[2]].sum() / total),
    }


def series_metrics(p: np.ndarray) -> dict:
    """Batería de métricas para una serie p(t) (con o sin t=0 ya recortada)."""
    if len(p) < 3:
        return None
    autocorr = float(np.corrcoef(p[:-1], p[1:])[0, 1]) if len(p) > 3 else float("nan")
    h = hurst_scalar(p)
    bands = power_bands(p)
    frac_pos = float(np.mean(p > 0))
    bursts = burst_lengths(p > 0)
    mean_burst = float(np.mean(bursts)) if bursts else 0.0
    max_burst = float(np.max(bursts)) if bursts else 0.0
    return {
        "autocorr_lag1": autocorr, "hurst": h,
        "power_low": bands["low"], "power_mid": bands["mid"], "power_high": bands["high"],
        "frac_positive": frac_pos, "mean_burst_len": mean_burst, "max_burst_len": max_burst,
        "mean_p": float(np.mean(p)),
    }


def per_condition_series_metrics(emb, lengths, v_hat, exclude_t0: bool):
    rows = []
    for i, L in enumerate(lengths):
        if L < 8:
            continue
        p = project_trajectory(emb[i, :L], v_hat)
        if exclude_t0:
            p = p[1:]
        m = series_metrics(p)
        if m is not None:
            rows.append(m)
    return rows


def agg(rows, key):
    vals = [r[key] for r in rows if r[key] == r[key]]  # descarta NaN
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}, vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    v_hat = np.load(V_HAT_PATH)
    tester = GeometricStatisticalTests(n_permutations=args.n_perm, random_seed=args.seed)

    all_groups = FREE_GROUPS
    metric_keys = ["autocorr_lag1", "hurst", "frac_positive", "mean_burst_len",
                   "max_burst_len", "power_low", "power_high", "mean_p"]

    results = {}
    for with_t0, label in [(True, "con_t0"), (False, "sin_t0")]:
        print(f"\n{'='*70}\n  {label.upper()}\n{'='*70}")
        results[label] = {}
        for g in all_groups:
            emb, lengths = load_free(g)
            rows = per_condition_series_metrics(emb, lengths, v_hat, exclude_t0=not with_t0)
            results[label][g] = {}
            line = f"  {g:22s} n={len(rows):3d} "
            for k in metric_keys:
                summ, vals = agg(rows, k)
                results[label][g][k] = {"summary": summ, "values": vals}
                line += f" {k}={summ['mean']:+.3f}"
            print(line)

    # ── Comparaciones por pares (permutación + d, sobre mean_burst_len y frac_positive) ──
    print(f"\n{'='*70}\n  Comparaciones por pares (con_t0 y sin_t0)\n{'='*70}")
    pair_results = {}
    for label in ["con_t0", "sin_t0"]:
        pair_results[label] = {}
        for a, b in PAIR_COMPARISONS:
            pair_results[label][f"{a}_vs_{b}"] = {}
            for k in ["mean_burst_len", "frac_positive", "autocorr_lag1", "mean_p"]:
                # recompute mean_p per-row too (not in metric_keys loop above but stored)
                va = results[label][a][k]["values"] if k in results[label][a] else None
                vb = results[label][b][k]["values"] if k in results[label][b] else None
                if va is None or vb is None or len(va) < 3 or len(vb) < 3:
                    continue
                res = tester.full_comparison(va, vb, a, b)
                pair_results[label][f"{a}_vs_{b}"][k] = {
                    "d": res["cohens_d"], "p": res["permutation_test"]["p_value"],
                }
            print(f"  [{label}] {a} vs {b}:")
            for k, v in pair_results[label][f"{a}_vs_{b}"].items():
                print(f"      {k:16s} d={v['d']:+.3f}  p={v['p']:.4f}")

    out_path = Path(__file__).parent / "EH2_results.json"
    out_path.write_text(json.dumps({
        "experiment": "E-H2", "v_hat_source": str(V_HAT_PATH),
        "series_metrics": {
            label: {g: {k: v["summary"] for k, v in cond.items()} for g, cond in groups.items()}
            for label, groups in results.items()
        },
        "pair_comparisons": pair_results,
    }, indent=2, ensure_ascii=False))
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
