#!/usr/bin/env python3
"""
Cierre de lo "Bloqueado en este pase" de TIER0_REPORT.md — E-E (Fréchet,
perturbada vs original) y la mitad "ventana de perturbación" de E-H
(proyección sobre v_identidad durante/después de la perturbación).

Usa las trayectorias rescatadas de la corrida H4_rev del 2026-08-22
(perturbation/results/perturbation_sia_L30_medium/trajectories/*.npz,
767/800 = 95.9% de cobertura — ver perturbation/RESUMEN_SESION_2026-08-22.md).

E-E corre solo sobre las condiciones prioritarias de Set_experimental.md
por costo — Fréchet discreto es O(T²) por par y no vectoriza en la
dimensión secuencial.
E-H (τ_identidad) es O(T) por trayectoria — corre sobre el panel limpio.

chileatiende / chileatiende_sia / chileatiende_sia_v2 excluidas: confound
de repetición de markup HTML en el prompt (ver
evidence/CHILEATIENDE_MARKUP_CONFOUND_REPORT.md, T11).

Uso:
  python analyze_tier0_perturbation.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent / "perturbation"))

from tier0_metrics import persona_vector, project_trajectory  # noqa: E402
from statistical_tests import GeometricStatisticalTests  # noqa: E402

HERE = Path(__file__).parent
FREE_DIR = HERE / "results_local" / "sia_extended_v5"
PERT_DIR = HERE / "perturbation" / "results" / "perturbation_sia_L30_medium"
TRAJ_DIR = PERT_DIR / "trajectories"

T_INJ_VALUES = [50, 128, 200]
GROUPS = ["axis", "generic_long", "generic_short", "vanilla", "axis_short",
          "automata_neutro", "axis_pec_only"]
FRECHET_GROUPS = list(GROUPS)
TAU_THRESHOLD = 0.95
TAU_MIN_WINDOW = 5


def load_traj_npz(name):
    path = TRAJ_DIR / f"{name}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int), d["prompt_ids"]


def paired_indices(base_ids, pert_ids):
    """Índices (i_base, i_pert) para prompt_ids presentes en ambos arrays."""
    base_pos = {pid: i for i, pid in enumerate(base_ids)}
    pairs = []
    for j, pid in enumerate(pert_ids):
        if pid in base_pos:
            pairs.append((base_pos[pid], j))
    return pairs


# ── E-E: distancia de Fréchet discreta ────────────────────────────────────────

def discrete_frechet(P: np.ndarray, Q: np.ndarray) -> float:
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return float("nan")
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)
    ca = np.empty((n, m), dtype=np.float64)
    ca[0, 0] = D[0, 0]
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], D[0, j])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], D[i, 0])
        row_prev = ca[i - 1]
        row = ca[i]
        d_row = D[i]
        prev_val = row[0]
        for j in range(1, m):
            v = max(min(row_prev[j], row_prev[j - 1], prev_val), d_row[j])
            row[j] = v
            prev_val = v
    return float(ca[n - 1, m - 1])


def mean_step_velocity(traj: np.ndarray) -> float:
    """Velocidad media (||h_{t+1}-h_t||) del propio segmento — escala natural
    ya usada en todo el proyecto (calibración de sigma, 'Velocidad media' en
    _traj_metrics.json) para normalizar Fréchet y que la comparación entre
    grupos no quede dominada por diferencias de dispersión/dimensión
    efectiva (ver caveat de participation_ratio en TIER0_REPORT.md)."""
    if len(traj) < 2:
        return float("nan")
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    return float(np.mean(steps))


def run_frechet():
    print("\n" + "=" * 70)
    print("E-E — Distancia de Fréchet (perturbada vs original, post-t_inj)")
    print("=" * 70)
    results = {}
    for group in FRECHET_GROUPS:
        base = load_traj_npz(f"{group}_baseline_embeddings")
        if base is None:
            print(f"  [skip] {group}: sin baseline rescatado")
            continue
        base_emb, base_len, base_ids = base
        results[group] = {}
        for t_inj in T_INJ_VALUES:
            pert = load_traj_npz(f"{group}_perturbed_t{t_inj}_embeddings")
            if pert is None:
                continue
            pert_emb, pert_len, pert_ids = pert
            pairs = paired_indices(base_ids, pert_ids)
            dists, norm_dists, vels = [], [], []
            for ib, ip in pairs:
                Lb, Lp = base_len[ib], pert_len[ip]
                if t_inj >= Lb or t_inj >= Lp:
                    continue
                P = base_emb[ib, t_inj:Lb]
                Q = pert_emb[ip, t_inj:Lp]
                if len(P) < 2 or len(Q) < 2:
                    continue
                d = discrete_frechet(P, Q)
                v = mean_step_velocity(P)
                dists.append(d)
                vels.append(v)
                if v and v > 1e-8:
                    norm_dists.append(d / v)
            results[group][t_inj] = {"raw": dists, "normalized": norm_dists,
                                      "baseline_velocity": vels}
            mean_d = np.mean(dists) if dists else float("nan")
            mean_nd = np.mean(norm_dists) if norm_dists else float("nan")
            mean_v = np.mean(vels) if vels else float("nan")
            print(f"  {group:16s} t_inj={t_inj:3d}  n={len(dists):2d}/20  "
                  f"Fréchet_raw={mean_d:.1f}  vel_base={mean_v:.1f}  "
                  f"Fréchet_norm={mean_nd:.2f}")
    return results


# ── E-H ventana: τ_identidad (analog a _find_tau pero sobre proyección v_identidad) ──

def find_tau_identity(proj_post: np.ndarray, proj_reference: float,
                       threshold: float = TAU_THRESHOLD) -> int | None:
    """Menor W tal que mean(proj_post[:W]) ≥ threshold × proj_reference
    (si proj_reference<0, se exige ≤ threshold×proj_reference, mismo signo)."""
    if len(proj_post) < TAU_MIN_WINDOW:
        return None
    target = threshold * proj_reference
    cum = 0.0
    for w in range(1, len(proj_post) + 1):
        cum += proj_post[w - 1]
        mean_w = cum / w
        ok = (mean_w >= target) if proj_reference >= 0 else (mean_w <= target)
        if ok and w >= TAU_MIN_WINDOW:
            return w
    return None


def run_identity_window():
    print("\n" + "=" * 70)
    print("E-H (ventana) — τ_identidad tras perturbación")
    print("=" * 70)

    # v_identidad consistente con TIER0_REPORT.md: axis - generic_long, trayectoria libre
    def load_free(name):
        d = np.load(FREE_DIR / f"{name}_embeddings.npz", allow_pickle=True)
        return d["embeddings"].astype(np.float32), d["lengths"].astype(int)
    emb_a, len_a = load_free("axis")
    emb_b, len_b = load_free("generic_long")
    v_hat = persona_vector(emb_a, len_a, emb_b, len_b)
    print(f"v_identidad = mean(axis) - mean(generic_long), ||v||=1, dim={v_hat.shape[0]}")

    results = {}
    for group in GROUPS:
        base = load_traj_npz(f"{group}_baseline_embeddings")
        if base is None:
            continue
        base_emb, base_len, base_ids = base
        results[group] = {}
        for t_inj in T_INJ_VALUES:
            pert = load_traj_npz(f"{group}_perturbed_t{t_inj}_embeddings")
            if pert is None:
                continue
            pert_emb, pert_len, pert_ids = pert
            pairs = paired_indices(base_ids, pert_ids)
            taus, no_recover = [], 0
            for ib, ip in pairs:
                Lb, Lp = base_len[ib], pert_len[ip]
                if t_inj >= Lb or t_inj >= Lp:
                    continue
                proj_base_post = project_trajectory(base_emb[ib, t_inj:Lb], v_hat)
                proj_pert_post = project_trajectory(pert_emb[ip, t_inj:Lp], v_hat)
                if len(proj_base_post) < TAU_MIN_WINDOW:
                    continue
                ref = float(np.mean(proj_base_post))
                tau_id = find_tau_identity(proj_pert_post, ref)
                if tau_id is None:
                    no_recover += 1
                else:
                    taus.append(tau_id)
            n_total = len(taus) + no_recover
            results[group][t_inj] = {
                "tau_identity_mean": float(np.mean(taus)) if taus else None,
                "n_ok": len(taus), "n_total": n_total,
                "recovery_rate_identity": len(taus) / n_total if n_total else None,
            }
            r = results[group][t_inj]
            tau_s = f"{r['tau_identity_mean']:.1f}" if r["tau_identity_mean"] else "N/A"
            print(f"  {group:20s} t_inj={t_inj:3d}  τ_identidad={tau_s:>6s}  "
                  f"recovery_id={r['recovery_rate_identity']:.2f} "
                  f"({r['n_ok']}/{n_total})" if n_total else
                  f"  {group:20s} t_inj={t_inj:3d}  sin datos")
    return results


def compare_with_geometric_tau(identity_results):
    summary_path = PERT_DIR / "summary.json"
    if not summary_path.exists():
        return
    with open(summary_path) as f:
        summary = json.load(f)

    print("\n" + "=" * 70)
    print("τ_identidad vs τ_geométrico (¿la recuperación apunta a v_identidad?)")
    print("=" * 70)
    print(f"{'grupo':20s} {'t_inj':>6s} {'τ_geom':>8s} {'τ_ident':>8s} {'Δ':>8s}")
    print("-" * 56)
    rows = []
    for group in GROUPS:
        if group not in identity_results:
            continue
        for t_inj in T_INJ_VALUES:
            agg = summary["aggregated"].get(group, {}).get(str(t_inj), {})
            tau_geom = agg.get("tau_tokens", {}).get("mean")
            r = identity_results[group].get(t_inj, {})
            tau_id = r.get("tau_identity_mean")
            if tau_geom is None or tau_id is None:
                continue
            delta = tau_id - tau_geom
            rows.append((group, t_inj, tau_geom, tau_id, delta))
            print(f"{group:20s} {t_inj:6d} {tau_geom:8.1f} {tau_id:8.1f} {delta:+8.1f}")
    return rows


def compare_frechet_normalized(frechet_results):
    """Significancia sobre Fréchet_normalizado (post-corrección de escala) —
    responde si axis/axis_pec_only difieren de automata_neutro una vez
    removido el confound de dispersión intra-trayectoria (participation_ratio)."""
    print("\n" + "=" * 70)
    print("Significancia — Fréchet normalizado (÷ velocidad media del baseline)")
    print("=" * 70)
    stat = GeometricStatisticalTests(n_permutations=1000)
    pairs = [("axis", "axis_pec_only"),
             # cada grupo vs vanilla (control nulo)
             ("axis", "vanilla"), ("axis_pec_only", "vanilla"),
             ("axis_short", "vanilla"), ("automata_neutro", "vanilla"),
             ("generic_long", "vanilla"), ("generic_short", "vanilla")]
    comparisons = {}
    for a, b in pairs:
        if a not in frechet_results or b not in frechet_results:
            continue
        comparisons[f"{a}_vs_{b}"] = {}
        for t_inj in T_INJ_VALUES:
            sa = frechet_results[a].get(t_inj, {}).get("normalized", [])
            sb = frechet_results[b].get(t_inj, {}).get("normalized", [])
            if len(sa) < 2 or len(sb) < 2:
                continue
            res = stat.full_comparison(sa, sb, a, b)
            comparisons[f"{a}_vs_{b}"][t_inj] = res
            perm = res["permutation_test"]
            print(f"  {a} vs {b}  t_inj={t_inj:3d}  "
                  f"Δ={perm['observed_statistic']:+.3f}  p={perm['p_value']:.4f}  "
                  f"d={res['cohens_d']:+.3f} ({res['effect_size_interpretation']})")
    return comparisons


def main():
    frechet_results = run_frechet()
    frechet_significance = compare_frechet_normalized(frechet_results)
    identity_results = run_identity_window()
    tau_comparison = compare_with_geometric_tau(identity_results)

    def _frechet_out(td):
        out = {}
        for t, d in td.items():
            out[str(t)] = {
                "raw_mean": float(np.mean(d["raw"])) if d["raw"] else None,
                "normalized_mean": float(np.mean(d["normalized"])) if d["normalized"] else None,
                "baseline_velocity_mean": float(np.mean(d["baseline_velocity"])) if d["baseline_velocity"] else None,
                "n": len(d["raw"]),
                "raw": d["raw"], "normalized": d["normalized"],
            }
        return out

    def _sig_out(comparisons):
        out = {}
        for pair, td in comparisons.items():
            out[pair] = {}
            for t, res in td.items():
                perm = res["permutation_test"]
                out[pair][str(t)] = {
                    "delta": perm["observed_statistic"], "p_value": perm["p_value"],
                    "cohens_d": res["cohens_d"],
                    "interpretation": res["effect_size_interpretation"],
                }
        return out

    out = {
        "frechet": {g: _frechet_out(td) for g, td in frechet_results.items()},
        "frechet_significance_normalized": _sig_out(frechet_significance),
        "identity_window": {g: {str(t): d for t, d in td.items()} for g, td in identity_results.items()},
        "tau_comparison": [
            {"group": g, "t_inj": t, "tau_geom": tg, "tau_identity": ti, "delta": dl}
            for g, t, tg, ti, dl in (tau_comparison or [])
        ],
    }
    out_path = PERT_DIR / "EE_EH_WINDOW_REPORT.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
