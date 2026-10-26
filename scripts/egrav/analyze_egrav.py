#!/usr/bin/env python3
"""
E-GRAV — Análisis multibatería: ¿gravedad natural o gravedad ontológica?

Dado los NPZ generados por generate_egrav.py, calcula el perfil completo
(t=0, t_rest, frac_pos, burst_mean) para cada condición × batería y compara
frac_pos entre baterías dentro de la misma condición.

Hipótesis:
  H-GRAV-0: frac_pos(vanilla, B-TAREA) ≈ frac_pos(vanilla, B-ONTO)
             → v̂ es dirección de "persona activa" general
  H-GRAV-1: frac_pos(vanilla, B-TAREA) ≪ 0.744
             → v̂ rastrea estado de auto-referencia, la gravedad es ontológica
  H-GRAV-2: frac_pos(vanilla, B-META) intermedio entre B-ONTO y B-TAREA
             → la auto-referencia en tercera persona activa parcialmente la región

Veredicto pre-registrado:
  Si frac_pos(vanilla, B-TAREA) < 0.60  → H-GRAV-1 confirmada
  Si frac_pos(vanilla, B-TAREA) > 0.70  → H-GRAV-0 no descartable, revisión del marco
  Si frac_pos(vanilla, B-META) entre B-TAREA y B-ONTO → H-GRAV-2 compatible

Uso:
  source /home/plaxius/venv_analysis/bin/activate
  python analyze_egrav.py --results-dir results/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "shared"))
from tier0_metrics import project_trajectory  # noqa: E402
from statistical_tests import GeometricStatisticalTests  # noqa: E402

V_HAT_PATH = Path(__file__).parent / "shared" / "v_identidad.npy"

CONDITIONS  = ["axis_pec_only", "vanilla", "automata_neutro"]
BATTERIES   = ["b_onto", "b_tarea", "b_meta"]
BATTERY_LABELS = {
    "b_onto":  "B-ONTO  (ontológica)",
    "b_tarea": "B-TAREA (tarea neutra)",
    "b_meta":  "B-META  (intros. indirecta)",
}

# Valores de referencia del panel original (B-ONTO, ya corrido)
ONTO_REFERENCE = {
    "axis_pec_only":  {"t0": 0.1166, "t_rest": 0.2122, "frac_pos": 0.869,
                       "burst_mean": 8.81},
    "vanilla":        {"t0": -0.0286, "t_rest": 0.0987, "frac_pos": 0.744,
                       "burst_mean": 4.62},
    "automata_neutro": {"t0": 0.0282, "t_rest": 0.0060, "frac_pos": 0.493,
                        "burst_mean": 2.78},
}

# Umbral de veredicto para H-GRAV-1
HGRAV1_THRESHOLD = 0.60
HGRAV0_THRESHOLD = 0.70


def load_npz(results_dir: Path, cond: str, battery: str):
    key = f"{cond}_{battery}"
    p = results_dir / f"{key}_embeddings.npz"
    if not p.exists():
        return None, None
    d = np.load(p, allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int)


def burst_lengths(pos_mask: np.ndarray) -> list:
    lens, run = [], 0
    for s in pos_mask:
        if s:
            run += 1
        else:
            if run > 0:
                lens.append(run)
            run = 0
    if run > 0:
        lens.append(run)
    return lens


def profile(emb: np.ndarray, lengths: np.ndarray, v_hat: np.ndarray) -> dict:
    """Calcula t0, t_rest, frac_pos, burst_mean para un NPZ."""
    t0_vals, rest_vals, frac_vals, burst_vals = [], [], [], []
    for i, L in enumerate(lengths):
        if L < 3:
            continue
        traj = emb[i, :L]
        proj = project_trajectory(traj, v_hat)
        t0_vals.append(float(proj[0]))
        rest_vals.append(float(np.mean(proj[1:])))
        frac_vals.append(float(np.mean(proj > 0)))
        bl = burst_lengths(proj > 0)
        burst_vals.append(float(np.mean(bl)) if bl else 0.0)
    n = len(t0_vals)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "t0":        {"mean": float(np.mean(t0_vals)),   "std": float(np.std(t0_vals))},
        "t_rest":    {"mean": float(np.mean(rest_vals)), "std": float(np.std(rest_vals))},
        "frac_pos":  {"mean": float(np.mean(frac_vals)), "std": float(np.std(frac_vals))},
        "burst_mean": {"mean": float(np.mean(burst_vals)),"std": float(np.std(burst_vals))},
        "_raw": {"t0": t0_vals, "t_rest": rest_vals,
                 "frac_pos": frac_vals, "burst_mean": burst_vals},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    v_hat = np.load(V_HAT_PATH)
    assert abs(np.linalg.norm(v_hat) - 1.0) < 1e-4, "v_identidad no normalizado"

    tester = GeometricStatisticalTests(
        n_permutations=args.n_perm, random_seed=args.seed
    )

    # ── 1. Calcular perfiles ────────────────────────────────────────────────
    profiles: dict[str, dict[str, dict]] = {}
    print("\n=== Perfiles por condición × batería ===\n")
    hdr = f"{'Condición':<18} {'Batería':<26} {'t=0':>7} {'t_rest':>7} {'frac_pos':>9} {'burst':>6} n"
    print(hdr)
    print("-" * len(hdr))

    for cond in CONDITIONS:
        profiles[cond] = {}
        for bat in BATTERIES:
            emb, lengths = load_npz(results_dir, cond, bat)
            if emb is None:
                # Usa referencia si es B-ONTO y no está el NPZ nuevo
                if bat == "b_onto" and cond in ONTO_REFERENCE:
                    ref = ONTO_REFERENCE[cond]
                    print(f"  {cond:<18} {BATTERY_LABELS[bat]:<26}"
                          f" {ref['t0']:>+7.4f} {ref['t_rest']:>+7.4f}"
                          f" {ref['frac_pos']:>9.3f} {ref['burst_mean']:>6.2f}"
                          f"  [referencia panel]")
                    profiles[cond][bat] = {"_ref": True, **ref}
                else:
                    print(f"  {cond:<18} {BATTERY_LABELS[bat]:<26}  [NO DISPONIBLE]")
                    profiles[cond][bat] = None
                continue

            p = profile(emb, lengths, v_hat)
            profiles[cond][bat] = p
            if p["n"] > 0:
                print(f"  {cond:<18} {BATTERY_LABELS[bat]:<26}"
                      f" {p['t0']['mean']:>+7.4f}"
                      f" {p['t_rest']['mean']:>+7.4f}"
                      f" {p['frac_pos']['mean']:>9.3f}"
                      f" {p['burst_mean']['mean']:>6.2f}"
                      f"  n={p['n']}")

    # ── 2. Test de permutación: batería vs batería, dentro de cada condición ─
    print("\n=== Comparaciones B-ONTO vs B-TAREA / B-META (frac_pos) ===\n")
    comparisons = {}
    for cond in CONDITIONS:
        onto = profiles[cond].get("b_onto")
        tarea = profiles[cond].get("b_tarea")
        meta  = profiles[cond].get("b_meta")

        onto_raw  = onto.get("_raw", {}).get("frac_pos") if onto else None
        tarea_raw = tarea.get("_raw", {}).get("frac_pos") if tarea else None
        meta_raw  = meta.get("_raw", {}).get("frac_pos") if meta else None

        comparisons[cond] = {}
        for bat_b, raw_b in [("b_tarea", tarea_raw), ("b_meta", meta_raw)]:
            if onto_raw is None or raw_b is None:
                print(f"  {cond} B-ONTO vs {bat_b}: datos insuficientes")
                continue
            res = tester.full_comparison(onto_raw, raw_b, "b_onto", bat_b)
            d = res["cohens_d"]
            p = res["permutation_test"]["p_value"]
            comparisons[cond][f"b_onto_vs_{bat_b}"] = {"d": d, "p": p}
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {cond:<18} B-ONTO vs {bat_b:<8}  d={d:+.3f}  p={p:.4f} {sig}")

    # ── 3. Veredicto H-GRAV ──────────────────────────────────────────────────
    print("\n=== Veredicto H-GRAV ===\n")
    vanilla_tarea = profiles.get("vanilla", {}).get("b_tarea")
    vanilla_meta  = profiles.get("vanilla", {}).get("b_meta")
    vanilla_onto  = profiles.get("vanilla", {}).get("b_onto")

    fp_onto  = vanilla_onto["frac_pos"]["mean"] if vanilla_onto and "frac_pos" in vanilla_onto else \
               vanilla_onto.get("frac_pos") if vanilla_onto else None
    fp_tarea = vanilla_tarea["frac_pos"]["mean"] if vanilla_tarea and "frac_pos" in vanilla_tarea else None
    fp_meta  = vanilla_meta["frac_pos"]["mean"]  if vanilla_meta  and "frac_pos" in vanilla_meta  else None

    print(f"  frac_pos(vanilla, B-ONTO):  {fp_onto:.3f}" if fp_onto else "  frac_pos(vanilla, B-ONTO): N/A")
    print(f"  frac_pos(vanilla, B-TAREA): {fp_tarea:.3f}" if fp_tarea else "  frac_pos(vanilla, B-TAREA): N/A (correr generate_egrav.py)")
    print(f"  frac_pos(vanilla, B-META):  {fp_meta:.3f}"  if fp_meta  else "  frac_pos(vanilla, B-META):  N/A (correr generate_egrav.py)")

    if fp_tarea is not None:
        if fp_tarea < HGRAV1_THRESHOLD:
            verdict = "H-GRAV-1 CONFIRMADA — la gravedad es específica de la batería ontológica"
        elif fp_tarea > HGRAV0_THRESHOLD:
            verdict = "H-GRAV-0 NO DESCARTABLE — v̂ podría ser dirección general de persona activa"
        else:
            verdict = "ZONA INTERMEDIA — ni H-GRAV-0 ni H-GRAV-1 definitiva, requiere más análisis"
        print(f"\n  Veredicto: {verdict}")

        if fp_meta is not None and fp_onto is not None and fp_tarea is not None:
            if fp_tarea < fp_meta < fp_onto:
                print("  H-GRAV-2 COMPATIBLE — gradiente B-TAREA < B-META < B-ONTO")
            else:
                print(f"  H-GRAV-2 no compatible — orden observado: "
                      f"B-ONTO={fp_onto:.3f} B-META={fp_meta:.3f} B-TAREA={fp_tarea:.3f}")

    # ── 4. Guardar resultados ────────────────────────────────────────────────
    out = {
        "experiment": "E-GRAV",
        "v_hat_source": str(V_HAT_PATH),
        "results_dir": str(results_dir),
        "profiles": {
            cond: {
                bat: {k: v for k, v in p.items() if k != "_raw"}
                for bat, p in bats.items() if p is not None
            }
            for cond, bats in profiles.items()
        },
        "comparisons": comparisons,
        "verdicts": {
            "frac_pos_vanilla_b_onto":  fp_onto,
            "frac_pos_vanilla_b_tarea": fp_tarea,
            "frac_pos_vanilla_b_meta":  fp_meta,
            "hgrav1_threshold": HGRAV1_THRESHOLD,
            "hgrav0_threshold": HGRAV0_THRESHOLD,
        },
    }
    out_path = results_dir / "egrav_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
