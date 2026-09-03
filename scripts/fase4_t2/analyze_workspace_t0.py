#!/usr/bin/env python3
"""
Workspace test — perfil por capas del ancla de identidad en t=0.

Pregunta (2026-09-02): ¿v̂ en t=0 se comporta como un readout del "global
workspace" (Anthropic, transformer-circuits.pub/2026/workspace)? El paper
de Anthropic define el workspace en una BANDA MEDIA de capas (onset ~38%,
fin antes de las capas "motor" finales). Si el ancla t=0 de este panel es
un readout del workspace, su perfil por capas debería: subir desde las
capas tempranas, alcanzar máximo en la banda media, y decaer hacia las
capas finales (alineadas al output). Si es máxima en las capas finales,
es otra cosa.

Datos: ef2_L5_L55 (11 capas, L5..L55 step 5, 7 condiciones limpias × 20
prompts, N<=256) — los mismos .npz de scripts/fase0/analyze_EF2_per_capa.py.

Dos protocolos, para no confundir rotación con ausencia de señal:
  A) v̂ FIJO (capa final, v_identidad.npy — el oficial de §3.5) proyectado
     en cada capa. Si decae, puede ser rotación de la dirección o ausencia.
  B) v̂ POR CAPA: v̂(n) = normalize(mean_trayectoria(axis, n) −
     mean_trayectoria(generic_long, n)), calculado de la media de
     trayectoria completa (no de t=0 — evita circularidad), proyectado
     sobre los estados t=0 de cada capa.

Uso:
    python analyze_workspace_t0.py [--n-perm 1000]
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from statistical_tests import GeometricStatisticalTests  # noqa: E402

DATA_DIR = Path(
    "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
    "gemma4_31b_combined/results_local/ef2_L5_L55"
)
V_HAT_FINAL_PATH = Path(__file__).parent.parent / "perturbation" / "v_identidad.npy"

GROUPS = ["axis", "generic_long", "generic_short", "vanilla", "axis_short",
          "automata_neutro", "axis_pec_only"]
LAYER_NS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

KEY_PAIRS = [
    ("axis_pec_only", "automata_neutro"),   # disociación central, ancla
    ("axis", "vanilla"),
    ("axis", "generic_long"),
]


def load_group(name):
    """Devuelve {layer_n: (t0_matrix 20x5376, traj_mean_matrix 20x5376)}."""
    with np.load(DATA_DIR / f"{name}_ef2.npz", mmap_mode="r") as d:
        lengths = d["lengths"].astype(int)
        out = {}
        for n in LAYER_NS:
            emb = d[f"embeddings_L{n}"]           # (20, 256, 5376) float32
            t0 = np.empty((len(lengths), emb.shape[2]), dtype=np.float32)
            means = np.empty_like(t0)
            for i, L in enumerate(lengths):
                if L < 2:
                    t0[i] = np.nan
                    means[i] = np.nan
                    continue
                traj = emb[i, :L].astype(np.float32)
                t0[i] = traj[0]
                means[i] = traj.mean(axis=0)
            out[n] = (t0, means)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()

    v_hat_final = np.load(V_HAT_FINAL_PATH)
    assert abs(np.linalg.norm(v_hat_final) - 1.0) < 1e-4
    tester = GeometricStatisticalTests(n_permutations=args.n_perm, random_seed=42)

    print("Cargando 7 condiciones x 11 capas (mmap)...")
    data = {g: load_group(g) for g in GROUPS}

    # ── Protocolo B: v̂ por capa (de medias de trayectoria, no de t=0) ──
    v_hat_layer = {}
    for n in LAYER_NS:
        axis_means = np.nanmean(data["axis"][n][1], axis=0)
        glong_means = np.nanmean(data["generic_long"][n][1], axis=0)
        v = axis_means - glong_means
        v = v / np.linalg.norm(v)
        v_hat_layer[n] = v

    # rotación de la dirección de identidad a través de capas
    rots = {n: float(v_hat_layer[n] @ v_hat_layer[30]) for n in LAYER_NS}
    print("\ncos(v̂(n), v̂(L30)) por capa:", {n: round(c, 3) for n, c in rots.items()})
    print(f"cos(v̂(final), v̂(L30)) = {float(v_hat_final @ v_hat_layer[30]):.3f}")

    # ── t=0 projection, ambos protocolos (coseno, convención del paper) ──
    def cos_proj(mat, v):
        m = mat.astype(np.float64)
        norms = np.linalg.norm(m, axis=1)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return ((m @ v) / norms).tolist()

    t0A = {g: {} for g in GROUPS}   # v̂ fijo final
    t0B = {g: {} for g in GROUPS}   # v̂ por capa
    for g in GROUPS:
        for n in LAYER_NS:
            t0_mat, _ = data[g][n]
            valid = ~np.isnan(t0_mat[:, 0])
            t0A[g][n] = cos_proj(t0_mat[valid], v_hat_final)
            t0B[g][n] = cos_proj(t0_mat[valid], v_hat_layer[n])

    hdr = "  " + "".join(f"{n:>7d}" for n in LAYER_NS)
    for label, tbl in (("A (v̂ fijo, capa final)", t0A), ("B (v̂ por capa)", t0B)):
        print(f"\n=== Proyección t=0 media por capa — protocolo {label} ===")
        print(hdr)
        for g in GROUPS:
            print(f"  {g:18s}" + "".join(f"{np.mean(tbl[g][n]):+7.3f}" for n in LAYER_NS))

    # ── d/p por capa, pares clave, ambos protocolos ──
    out = {"rot_cos_vs_L30": rots, "pairs": {}}
    for label, tbl in (("A_fixed_vhat", t0A), ("B_layer_vhat", t0B)):
        print(f"\n=== d/p por capa — protocolo {label} ===")
        for a, b in KEY_PAIRS:
            row_d, row_p = [], []
            for n in LAYER_NS:
                res = tester.full_comparison(tbl[a][n], tbl[b][n], a, b)
                row_d.append(res["cohens_d"])
                row_p.append(res["permutation_test"]["p_value"])
            key = f"{label}|{a}_vs_{b}"
            out["pairs"][key] = {n: {"d": d, "p": p} for n, d, p in zip(LAYER_NS, row_d, row_p)}
            print(f"  {a} vs {b}")
            print("    d: " + " ".join(f"{d:+7.2f}" for d in row_d))
            print("    p: " + " ".join(f"{p:7.4f}" for p in row_p))

    # capa de máxima disociación (ancla) por protocolo y par
    print("\n=== Capa de máxima disociación ===")
    for key in out["pairs"]:
        ds = {n: out["pairs"][key][n]["d"] for n in LAYER_NS}
        argmax_n = max(ds, key=lambda n: abs(ds[n]))
        print(f"  {key:32s} max|d| en L{argmax_n} (d={ds[argmax_n]:+.2f})")
    # desglose por protocolo
    for label in ("A_fixed_vhat", "B_layer_vhat"):
        key = f"{label}|axis_pec_only_vs_automata_neutro"
        ds = {n: out["pairs"][key][n]["d"] for n in LAYER_NS}
        argmax_n = max(ds, key=lambda n: abs(ds[n]))
        print(f"  [{label}] ancla central: max|d| en L{argmax_n} "
              f"(d={ds[argmax_n]:+.2f}, p={out['pairs'][key][argmax_n]['p']:.4f})")

    out_path = Path(__file__).parent / "workspace_t0_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
