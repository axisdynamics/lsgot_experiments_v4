"""
Exp 0.3 (auto-auditoría) — escala de referencia (split-half null) para la
curvatura de Forman-Ricci del panel Gemma-4-31B-it, adaptado de
MIA-experiments/fase0/scripts/exp03_split_half_null.py (REPORTE_FASE0.md).

NOTA: la clase se llama FormanRicciAnalyzer y calcula la fórmula
combinatoria de Forman (ver docstring de curvature_analyzer.py) — NO
Ollivier-Ricci (transporte óptimo). El paper y REPORTE_FASE0.md la citan
como "Ollivier-Ricci" por error heredado; ver CURVATURE_SELF_AUDIT_REPORT.md.
"""
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from curvature_analyzer import FormanRicciAnalyzer  # noqa: E402

GRAPHS = Path(__file__).parent.parent.parent / "data" / "sia_extended_v5" / "_graphs.json"
GROUPS = ["axis", "axis_pec_only", "automata_neutro", "vanilla"]
PAIRS = [("axis_pec_only", "automata_neutro"), ("axis_pec_only", "vanilla"),
         ("automata_neutro", "vanilla"), ("axis", "vanilla")]
B = 1000
SEED = 42
OUT = Path(__file__).parent / "exp03_sia_result.json"


def edge_curvatures_per_traj(edge_lists):
    analyzer = FormanRicciAnalyzer(include_4cycles=True)
    per_traj = []
    for edges in edge_lists:
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        vals = analyzer.compute_all_curvatures(G)
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
    per_traj = {g: edge_curvatures_per_traj(data[g]) for g in GROUPS}
    n_traj = {g: len(per_traj[g]) for g in GROUPS}
    print("n_trayectorias:", n_traj)

    observed = {}
    for a, b in PAIRS:
        observed[f"{a}|{b}"] = pooled_w1(per_traj[a], per_traj[b])

    nulls = {g: split_half_null(per_traj[g], rng) for g in GROUPS}

    report = {"B": B, "seed": SEED, "n_trayectorias": n_traj, "observados": observed, "comparaciones": {}}
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
