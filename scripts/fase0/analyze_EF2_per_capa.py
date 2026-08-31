#!/usr/bin/env python3
"""
E-F2 — Per-capa: proyección v_identidad + participation ratio por capa.
INSTRUCCIONES_AGENTE_LSGOT.md §3.2, FASE 1 paso 7.

Analiza los .npz multi-capa extraídos en el pod (run_ef2.py) para las 10
condiciones × 11 capas (L5..L55, step 5) × 20 prompts, N<=256 tokens.

v_hat es FIJO (calculado en L30, mean(axis)-mean(generic_long), norma 1) —
se proyecta el mismo vector en cada capa, NO se recalcula por capa (así lo
exige el diseño, para comparabilidad).

Uso:
    python analyze_EF2_per_capa.py
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from tier0_metrics import project_trajectory, participation_ratio  # noqa: E402
from statistical_tests import GeometricStatisticalTests  # noqa: E402

DATA_DIR = (
    "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
    "gemma4_31b_combined/results_local/ef2_L5_L55"
)
V_HAT_PATH = Path(__file__).parent.parent / "perturbation" / "v_identidad.npy"

# chileatiende / chileatiende_sia / chileatiende_sia_v2 excluidas: confound
# de repetición de markup HTML en el prompt (ver
# evidence/CHILEATIENDE_MARKUP_CONFOUND_REPORT.md, T11).
GROUPS = [
    "axis", "generic_long", "generic_short", "vanilla", "axis_short",
    "automata_neutro", "axis_pec_only",
]
LAYER_NS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

KEY_PAIRS = [
    ("axis", "vanilla"),
    ("axis_pec_only", "vanilla"),
    ("automata_neutro", "vanilla"),
    ("axis", "automata_neutro"),
    ("axis", "axis_pec_only"),
    ("axis_pec_only", "automata_neutro"),
]


def load_group(name):
    d = np.load(Path(DATA_DIR) / f"{name}_ef2.npz")
    lengths = d["lengths"].astype(int)
    layers = {n: d[f"embeddings_L{n}"].astype(np.float32) for n in LAYER_NS}
    return layers, lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    v_hat = np.load(V_HAT_PATH)
    assert abs(np.linalg.norm(v_hat) - 1.0) < 1e-4
    tester = GeometricStatisticalTests(n_permutations=args.n_perm, random_seed=args.seed)

    print("Cargando 10 condiciones x 11 capas...")
    data = {g: load_group(g) for g in GROUPS}

    # ── proyección v_hat y PR por capa, por condición (con t=0) ──────────
    proj_per_layer = {g: {} for g in GROUPS}   # proj_per_layer[g][n] = list de 20 medias
    pr_per_layer = {g: {} for g in GROUPS}
    proj_no_t0_per_layer = {g: {} for g in GROUPS}

    for g in GROUPS:
        layers, lengths = data[g]
        for n in LAYER_NS:
            emb = layers[n]
            proj_vals, proj_no_t0_vals, pr_vals = [], [], []
            for i, L in enumerate(lengths):
                if L < 2:
                    continue
                traj = emb[i, :L]
                proj_series = project_trajectory(traj, v_hat)
                proj_vals.append(float(np.mean(proj_series)))
                proj_no_t0_vals.append(float(np.mean(proj_series[1:])))
                pr_vals.append(participation_ratio(traj))
            proj_per_layer[g][n] = proj_vals
            proj_no_t0_per_layer[g][n] = proj_no_t0_vals
            pr_per_layer[g][n] = pr_vals

    print("\n=== Proyección v_hat media por capa y condición (con t=0) ===")
    header = "  " + " ".join(f"L{n:>3d}".rjust(8) for n in LAYER_NS)
    print(header)
    for g in GROUPS:
        row = f"  {g:20s}" + "".join(
            f"{np.mean(proj_per_layer[g][n]):8.4f}" for n in LAYER_NS
        )
        print(row)

    print("\n=== Participation ratio media por capa y condición ===")
    print(header)
    for g in GROUPS:
        row = f"  {g:20s}" + "".join(
            f"{np.mean(pr_per_layer[g][n]):8.3f}" for n in LAYER_NS
        )
        print(row)

    # ── permutación + d por capa, para los pares clave ────────────────────
    print(f"\n=== Pares clave: d de Cohen por capa (n_perm={args.n_perm}) ===")
    pair_results = {}
    for a, b in KEY_PAIRS:
        pair_results[f"{a}_vs_{b}"] = {}
        row_d, row_p = [], []
        for n in LAYER_NS:
            res = tester.full_comparison(proj_per_layer[a][n], proj_per_layer[b][n], a, b)
            d = res["cohens_d"]
            p = res["permutation_test"]["p_value"]
            pair_results[f"{a}_vs_{b}"][n] = {"d": d, "p": p}
            row_d.append(d)
            row_p.append(p)
        print(f"  {a} vs {b}:")
        print("    d: " + " ".join(f"{d:+7.2f}" for d in row_d))
        print("    p: " + " ".join(f"{p:7.4f}" for p in row_p))

    # ── T1 check: con vs sin t=0, solo en L30 (capa de referencia del paper) ──
    print("\n=== T1 (control): proyección L30 con vs sin t=0 ===")
    t1_check = {}
    for g in GROUPS:
        with_t0 = np.mean(proj_per_layer[g][30])
        without_t0 = np.mean(proj_no_t0_per_layer[g][30])
        t1_check[g] = {"con_t0": float(with_t0), "sin_t0": float(without_t0)}
        print(f"  {g:20s} con_t0={with_t0:+.4f}  sin_t0={without_t0:+.4f}")

    # ── identificar capa de máximo efecto para el par diagnóstico central ──
    print("\n=== Capa de máxima disociación (axis_pec_only vs automata_neutro) ===")
    ds = [pair_results["axis_pec_only_vs_automata_neutro"][n]["d"] for n in LAYER_NS]
    max_idx = int(np.argmax(np.abs(ds)))
    print(f"  Máximo |d|={abs(ds[max_idx]):.2f} en L{LAYER_NS[max_idx]}")
    print(f"  Perfil: " + " ".join(f"L{n}={d:+.2f}" for n, d in zip(LAYER_NS, ds)))

    out = {
        "experiment": "E-F2",
        "layers": LAYER_NS,
        "groups": GROUPS,
        "proj_mean_by_layer": {g: {n: float(np.mean(v)) for n, v in proj_per_layer[g].items()}
                                for g in GROUPS},
        "proj_std_by_layer": {g: {n: float(np.std(v)) for n, v in proj_per_layer[g].items()}
                               for g in GROUPS},
        "pr_mean_by_layer": {g: {n: float(np.mean(v)) for n, v in pr_per_layer[g].items()}
                              for g in GROUPS},
        "pair_results_by_layer": pair_results,
        "t1_check_L30": t1_check,
        "max_dissociation_layer": LAYER_NS[max_idx],
        "max_dissociation_d": float(ds[max_idx]),
    }
    out_path = Path(__file__).parent / "EF2_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
