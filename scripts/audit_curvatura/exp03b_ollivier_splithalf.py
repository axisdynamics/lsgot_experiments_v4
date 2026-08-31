"""
Exp 0.3b — la medición que se pretendía hacer desde el origen: curvatura de
Ollivier-Ricci genuina (transporte óptimo, no Forman) sobre el mismo panel y
el mismo protocolo de split-half null que exp03_splithalf_sia.py, para
comparar directamente contra los resultados de Forman-Ricci del audit
original (CURVATURE_SELF_AUDIT_REPORT.md).

Usa GraphRicciCurvature.OllivierRicci (Ni, Lin, Gao, Gu & Saucan) — la
implementación de referencia, no una hecha a mano. alpha=0.5, method='OTD'
(transporte óptimo exacto vía LP, no aproximado) — mismo alpha=1/2 que ya
citaba §2.3 de lsgot_4.md para la métrica original (nunca se había corrido).

Conversión de peso IMPORTANTE: data/sia_extended_v5/_graphs.json guarda
weight = 1/(1+cos_dist) (similitud — mayor weight = más cerca), pero
OllivierRicci espera weight = distancia de grafo para Dijkstra/transporte
(menor weight = más cerca). Se reconstruye cos_dist = 1/weight - 1 antes de
construir el grafo — sin esta conversión el resultado estaría invertido.

Uso:
    ricci_venv/bin/python3 exp03b_ollivier_splithalf.py
(requiere GraphRicciCurvature + POT — ver README del venv si no está)
"""
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance
from GraphRicciCurvature.OllivierRicci import OllivierRicci

GRAPHS = Path(__file__).parent.parent.parent / "data" / "sia_extended_v5" / "_graphs.json"
GROUPS = ["axis", "axis_pec_only", "automata_neutro", "vanilla"]
PAIRS = [("axis_pec_only", "automata_neutro"), ("axis_pec_only", "vanilla"),
         ("automata_neutro", "vanilla"), ("axis", "vanilla")]
ALPHA = 0.5
B = 1000
SEED = 42
OUT = Path(__file__).parent / "exp03b_ollivier_result.json"


def edge_curvatures_per_traj(edge_lists):
    per_traj = []
    for edges in edge_lists:
        G = nx.Graph()
        for u, v, w in edges:
            # similitud -> distancia (ver docstring)
            G.add_edge(int(u), int(v), weight=(1.0 / w - 1.0) if w > 0 else 10.0)
        orc = OllivierRicci(G, alpha=ALPHA, method="OTD", verbose="ERROR")
        orc.compute_ricci_curvature()
        vals = [d["ricciCurvature"] for _, _, d in orc.G.edges(data=True)]
        per_traj.append(np.asarray(vals, dtype=np.float64))
    return per_traj


def pooled_w1(vals_a, vals_b):
    return float(wasserstein_distance(np.concatenate(vals_a), np.concatenate(vals_b)))


def split_half_null(per_traj, rng):
    n = len(per_traj)
    nulls = np.zeros(B, dtype=np.float64)
    idx = np.arange(n)
    for b in range(B):
        rng.shuffle(idx)
        half = n // 2
        a = [per_traj[i] for i in idx[:half]]
        c = [per_traj[i] for i in idx[half:]]
        nulls[b] = pooled_w1(a, c)
    return nulls


def main():
    rng = np.random.default_rng(SEED)
    data = json.loads(GRAPHS.read_text())

    per_traj = {}
    for g in GROUPS:
        t0 = time.time()
        per_traj[g] = edge_curvatures_per_traj(data[g])
        means = [v.mean() for v in per_traj[g]]
        print(f"{g:18s} n_traj={len(per_traj[g])} mean_curv={np.mean(means):+.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    n_traj = {g: len(per_traj[g]) for g in GROUPS}
    observed = {}
    for a, b in PAIRS:
        observed[f"{a}|{b}"] = pooled_w1(per_traj[a], per_traj[b])

    nulls = {g: split_half_null(per_traj[g], rng) for g in GROUPS}

    report = {"alpha": ALPHA, "method": "OTD", "B": B, "seed": SEED,
              "n_trayectorias": n_traj,
              "mean_curvature_per_condition": {g: float(np.mean([v.mean() for v in per_traj[g]])) for g in GROUPS},
              "observados": observed, "comparaciones": {}}
    for a, b in PAIRS:
        obs = observed[f"{a}|{b}"]
        row = {"observed": obs}
        for cond in (a, b):
            arr = nulls[cond]
            pct = float(np.mean(arr >= obs) * 100)
            mult = obs / float(np.median(arr)) if np.median(arr) > 0 else float("nan")
            row[f"null_{cond}"] = {
                "median": float(np.median(arr)), "mean": float(np.mean(arr)),
                "sd": float(np.std(arr, ddof=1)), "p95": float(np.percentile(arr, 95)),
                "percentil_observado_pct": pct, "multiplos_de_mediana": mult,
            }
        report["comparaciones"][f"{a}_vs_{b}"] = row
        print(f"W1({a},{b}) obs={obs:.4f} | vs null {a}: pct={row[f'null_{a}']['percentil_observado_pct']:.1f}% "
              f"x{row[f'null_{a}']['multiplos_de_mediana']:.1f} | "
              f"vs null {b}: pct={row[f'null_{b}']['percentil_observado_pct']:.1f}% "
              f"x{row[f'null_{b}']['multiplos_de_mediana']:.1f}")

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
