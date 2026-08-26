#!/usr/bin/env python3
"""
Tier 0 de Set_experimental.md — E-A (RQA), E-C (Hurst), E-D (participation
ratio), E-H (proyección sobre v_identidad), corridos sobre las trayectorias
libres ya extraídas (sin nueva corrida de GPU).

Uso:
  python analyze_tier0.py
  python analyze_tier0.py --results-dir results_local/sia_extended_v5
  python analyze_tier0.py --groups axis axis_pec_only automata_neutro vanilla

Bloqueado en este pase (ver Set_experimental.md / plan): E-E (Fréchet) y la
mitad "ventana de perturbación" de E-H — las trayectorias perturbadas crudas
de H4_rev nunca se guardaron localmente, solo viven como .pkl efímeros en el
Network Volume de RunPod.
"""

import sys
import json
import argparse
from pathlib import Path
from dataclasses import asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "shared"))

from tier0_metrics import (  # noqa: E402
    persona_vector, compute_trajectory_tier0, _agg,
)
from statistical_tests import GeometricStatisticalTests  # noqa: E402

DEFAULT_RESULTS_DIR = "results_local/sia_extended_v5"
DEFAULT_GROUPS = [
    "axis", "generic_long", "generic_short", "vanilla", "axis_short",
    "chileatiende", "automata_neutro", "chileatiende_sia", "axis_pec_only",
]
TIER0_FIELDS = [
    "hurst", "determinism", "laminarity", "trapping_time",
    "rqa_recurrence_rate", "participation_ratio", "identity_projection",
]
COMPARISON_PAIRS = [
    ("axis", "vanilla"),
    ("axis_pec_only", "vanilla"),
    ("automata_neutro", "vanilla"),
    ("axis", "automata_neutro"),
    ("axis", "axis_pec_only"),
    # agregado 2026-08-26: chileatiende no tenía comparación formal contra
    # nada en identity_projection — solo se veía su media cruda.
    ("chileatiende", "vanilla"),
    ("axis", "chileatiende"),
    ("chileatiende", "automata_neutro"),
    ("chileatiende_sia", "vanilla"),
]


def load_group(results_dir: Path, name: str):
    d = np.load(results_dir / f"{name}_embeddings.npz", allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)   # (N, T, D)
    lengths = d["lengths"].astype(int)          # (N,)
    return emb, lengths


def main():
    parser = argparse.ArgumentParser(description="Tier 0 — Set_experimental.md")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--groups", nargs="+", default=None)
    parser.add_argument("--identity-base", type=str, default="axis")
    parser.add_argument("--identity-contrast", type=str, default="generic_long")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    groups = args.groups or DEFAULT_GROUPS

    print(f"\nTier 0 — {results_dir}/")
    print(f"Grupos: {groups}")

    data = {}
    for g in groups:
        npz_path = results_dir / f"{g}_embeddings.npz"
        if not npz_path.exists():
            print(f"  [skip] {g}: {npz_path.name} no existe")
            continue
        data[g] = load_group(results_dir, g)
        emb, lengths = data[g]
        print(f"  {g:<18} {emb.shape[0]} trayectorias, longitud media={lengths.mean():.1f}")

    if args.identity_base not in data or args.identity_contrast not in data:
        print(f"\n[error] v_identidad requiere '{args.identity_base}' y "
              f"'{args.identity_contrast}' cargados — no se calcula proyección.")
        v_identidad = None
    else:
        emb_a, len_a = data[args.identity_base]
        emb_b, len_b = data[args.identity_contrast]
        v_identidad = persona_vector(emb_a, len_a, emb_b, len_b)
        print(f"\nv_identidad = mean({args.identity_base}) - mean({args.identity_contrast}), "
              f"||v||=1, dim={v_identidad.shape[0]}")

    tier0_json = {}
    per_group_samples = {}

    for g, (emb, lengths) in data.items():
        metrics_list = []
        for i, L in enumerate(lengths):
            if L < 8:
                continue
            m = compute_trajectory_tier0(emb[i, :L], v_identidad=v_identidad)
            metrics_list.append(asdict(m))
        tier0_json[g] = metrics_list
        per_group_samples[g] = {
            f: [m[f] for m in metrics_list if m.get(f) is not None]
            for f in TIER0_FIELDS
        }

    out_json = results_dir / "_tier0_metrics.json"
    with open(out_json, "w") as f:
        json.dump(tier0_json, f)
    print(f"\n→ {out_json}")

    # ── Comparaciones estadísticas ────────────────────────────────────────────
    stat = GeometricStatisticalTests(n_permutations=1000)
    comparisons = {}
    for a, b in COMPARISON_PAIRS:
        if a not in per_group_samples or b not in per_group_samples:
            continue
        comparisons[f"{a}_vs_{b}"] = {}
        for field in TIER0_FIELDS:
            sa, sb = per_group_samples[a][field], per_group_samples[b][field]
            if len(sa) < 2 or len(sb) < 2:
                continue
            comparisons[f"{a}_vs_{b}"][field] = stat.full_comparison(sa, sb, a, b)

    # ── Reporte Markdown ─────────────────────────────────────────────────────
    lines = []
    lines.append("# Tier 0 — Set_experimental.md (E-A, E-C, E-D, E-H)\n")
    lines.append(f"Datos: `{results_dir}/*_embeddings.npz`. "
                  f"v_identidad = mean({args.identity_base}) − mean({args.identity_contrast}).\n")

    lines.append("## Tabla comparativa por grupo (media ± std)\n")
    header = "| grupo | hurst | determinism | laminarity | trapping_time | PR | proj(v_identidad) |"
    sep = "|---" * 7 + "|"
    lines.append(header)
    lines.append(sep)
    for g in groups:
        if g not in per_group_samples:
            continue
        s = per_group_samples[g]
        def cell(field):
            a = _agg(s[field])
            return f"{a['mean']:.4f}±{a['std']:.4f}" if a["n"] else "—"
        lines.append(
            f"| {g} | {cell('hurst')} | {cell('determinism')} | {cell('laminarity')} | "
            f"{cell('trapping_time')} | {cell('participation_ratio')} | "
            f"{cell('identity_projection')} |"
        )
    lines.append("")

    lines.append("## Significancia (permutation test, mean_difference, n_perm=1000)\n")
    for pair, fields in comparisons.items():
        lines.append(f"### {pair}\n")
        lines.append("| métrica | Δ(mean) | p | cohen_d | interpretación |")
        lines.append("|---|---|---|---|---|")
        for field, res in fields.items():
            perm = res["permutation_test"]
            lines.append(
                f"| {field} | {perm['observed_statistic']:+.4f} | {perm['p_value']:.4f} | "
                f"{res['cohens_d']:+.3f} | {res['effect_size_interpretation']} |"
            )
        lines.append("")

    lines.append("## Bloqueado en este pase\n")
    lines.append(
        "- **E-E (distancia de Fréchet, perturbada vs original)** y la mitad "
        "\"dentro de la ventana de perturbación\" de **E-H**: las trayectorias "
        "perturbadas crudas de H4_rev (`perturbation/perturbation_extractor.py` "
        "`Trajectory.embeddings`) solo se cachearon como `.pkl` en el Network "
        "Volume de RunPod y nunca se copiaron a local; "
        "`perturbation_*/details.json` solo tiene escalares (τ, recovery_rate, "
        "displacement_l2, etc.), no arrays. Requiere una corrida RunPod "
        "dedicada que persista las trayectorias crudas antes de poder cerrar "
        "estos dos análisis."
    )
    lines.append("")

    out_md = results_dir.parent / "TIER0_REPORT.md"
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"→ {out_md}")

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
