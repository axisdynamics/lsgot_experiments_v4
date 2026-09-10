"""
Δκ/W₁ cross-modelo — Forman-Ricci y Ollivier-Ricci genuina, mismo protocolo
de split-half null que exp03_splithalf_sia.py/exp03b_ollivier_splithalf.py
(Gemma-4-31B-it), corrido sobre Qwen3-32B y Command R 35B para responder si
el patrón de curvatura (que ya perdió contra baselines simples en Gemma,
CURVATURE_SELF_AUDIT_REPORT.md) aparece igual, más débil, o directamente no
existe en los otros dos sustratos.

A diferencia del panel Gemma (que consumía `_graphs.json` pre-construido),
acá se construye el grafo desde los embeddings crudos con
`shared/graph_builder.TrajectoryGraphBuilder` (k_nn=5, temporal_weight=1.0
— misma clase, sin modificar) en runtime, 100% CPU.

Uso:
    ricci_venv/bin/python3 exp03_crossmodel.py qwen3
    ricci_venv/bin/python3 exp03_crossmodel.py command_r
"""
import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance
from GraphRicciCurvature.OllivierRicci import OllivierRicci

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from curvature_analyzer import FormanRicciAnalyzer  # noqa: E402
from graph_builder import TrajectoryGraphBuilder  # noqa: E402

HOME = Path.home()
BASE = HOME / "Documentos" / "Proyectos" / "Geometría_LSGOT" / "SIA-experiments"

MODEL_CONFIGS = {
    "qwen3": {
        "dir": BASE / "gemma4_31b_combined" / "results_local" / "qwen3_fase0",
        "file_suffix": "_qwen3.npz",
        "final_layer_key": "embeddings_L63",
    },
    "command_r": {
        "dir": BASE / "command_r_35b_combined" / "results_local" / "command_r_fase0",
        "file_suffix": "_command_r.npz",
        "final_layer_key": "embeddings_L39",
    },
}

GROUPS = ["axis", "axis_pec_only", "automata_neutro", "vanilla"]
PAIRS = [("axis_pec_only", "automata_neutro"), ("axis_pec_only", "vanilla"),
         ("automata_neutro", "vanilla"), ("axis", "vanilla")]
K_NN = 5
ALPHA_OLLIVIER = 0.5
B = 1000
SEED = 42


def load_trajectories(cfg, group):
    d = np.load(cfg["dir"] / f"{group}{cfg['file_suffix']}")
    emb = d[cfg["final_layer_key"]].astype(np.float32)
    lengths = d["lengths"].astype(int)
    return [emb[i, :lengths[i]] for i in range(len(lengths)) if lengths[i] >= 2]


def build_graph(traj):
    builder = TrajectoryGraphBuilder(k_nn=K_NN, temporal_weight=1.0)
    return builder.build([traj[i] for i in range(len(traj))])


def forman_curvatures(G):
    analyzer = FormanRicciAnalyzer(include_4cycles=True)
    return np.asarray(analyzer.compute_all_curvatures(G), dtype=np.float64)


def ollivier_curvatures(G):
    # similitud (weight=1/(1+cos_dist)) -> distancia (cos_dist) para OllivierRicci,
    # misma conversión documentada en exp03b_ollivier_splithalf.py
    Gd = nx.Graph()
    for u, v, w in G.edges(data="weight"):
        Gd.add_edge(u, v, weight=(1.0 / w - 1.0) if w > 0 else 10.0)
    orc = OllivierRicci(Gd, alpha=ALPHA_OLLIVIER, method="OTD", verbose="ERROR")
    orc.compute_ricci_curvature()
    return np.asarray([d["ricciCurvature"] for _, _, d in orc.G.edges(data=True)], dtype=np.float64)


def pooled_w1(vals_a, vals_b):
    return float(wasserstein_distance(np.concatenate(vals_a), np.concatenate(vals_b)))


def split_half_null(per_traj, rng):
    n = len(per_traj)
    nulls = np.zeros(B, dtype=np.float64)
    idx = np.arange(n)
    for b in range(B):
        rng.shuffle(idx)
        half = n // 2
        nulls[b] = pooled_w1([per_traj[i] for i in idx[:half]], [per_traj[i] for i in idx[half:]])
    return nulls


def run_metric(name, curvature_fn, graphs, out_path):
    print(f"\n=== {name} ===", flush=True)
    rng = np.random.default_rng(SEED)
    per_traj = {}
    for g in GROUPS:
        t0 = time.time()
        per_traj[g] = [curvature_fn(G) for G in graphs[g]]
        means = [v.mean() for v in per_traj[g] if len(v) > 0]
        print(f"  {g:18s} n_traj={len(per_traj[g])} mean_curv={np.mean(means):+.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    observed = {f"{a}|{b}": pooled_w1(per_traj[a], per_traj[b]) for a, b in PAIRS}
    nulls = {g: split_half_null(per_traj[g], rng) for g in GROUPS}

    report = {"metric": name, "B": B, "seed": SEED, "alpha": ALPHA_OLLIVIER if "ollivier" in name.lower() else None,
              "n_trayectorias": {g: len(per_traj[g]) for g in GROUPS},
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
        print(f"  W1({a},{b}) obs={obs:.4f} | vs null {a}: pct={row[f'null_{a}']['percentil_observado_pct']:.1f}% "
              f"x{row[f'null_{a}']['multiplos_de_mediana']:.1f} | "
              f"vs null {b}: pct={row[f'null_{b}']['percentil_observado_pct']:.1f}% "
              f"x{row[f'null_{b}']['multiplos_de_mediana']:.1f}", flush=True)

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"  Guardado: {out_path}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--skip-ollivier", action="store_true",
                     help="Solo Forman-Ricci (rápido) — Ollivier-Ricci OTD es más lento")
    args = ap.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    print(f"Cargando trayectorias y construyendo grafos ({args.model}, k_nn={K_NN})...", flush=True)
    trajectories = {g: load_trajectories(cfg, g) for g in GROUPS}
    graphs = {}
    for g in GROUPS:
        t0 = time.time()
        graphs[g] = [build_graph(t) for t in trajectories[g]]
        n_edges = [G.number_of_edges() for G in graphs[g]]
        print(f"  {g:18s} n_traj={len(graphs[g])} edges/traj~{np.mean(n_edges):.0f} ({time.time()-t0:.1f}s)", flush=True)

    out_dir = Path(__file__).parent
    run_metric("forman_ricci", forman_curvatures, graphs, out_dir / f"exp03_{args.model}_forman.json")
    if not args.skip_ollivier:
        run_metric("ollivier_ricci_alpha0.5_OTD", ollivier_curvatures, graphs,
                    out_dir / f"exp03_{args.model}_ollivier.json")

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
