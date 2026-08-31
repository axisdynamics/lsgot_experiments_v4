#!/usr/bin/env python3
"""
E-L — Primer token como firma de identidad (control + medida)
INSTRUCCIONES_AGENTE_LSGOT.md §3.2, FASE 0 paso 1.

Pregunta: ¿la doble disociación de v_identidad ya está presente en t=0
(estado de contexto, antes de generar nada), o solo emerge en t>0 (proceso
de generación)? Desambiguador para E-H2.

Datos: embeddings libres ya extraídos (results_local/sia_extended_v5,
gemma4_31b_combined) — Tier 0, cero GPU.

Uso:
    python analyze_EL_primer_token.py
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

DEFAULT_RAW_DIR = (
    "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
    "gemma4_31b_combined/results_local/sia_extended_v5"
)
V_HAT_PATH = Path(__file__).parent.parent / "perturbation" / "v_identidad.npy"

# chileatiende / chileatiende_sia excluidas: confound de repetición de
# markup HTML en el prompt (ver evidence/CHILEATIENDE_MARKUP_CONFOUND_REPORT.md, T11).
GROUPS = [
    "axis", "generic_long", "generic_short", "vanilla", "axis_short",
    "automata_neutro", "axis_pec_only",
]

KEY_PAIRS = [
    ("axis_pec_only", "automata_neutro"),
    ("axis_pec_only", "vanilla"),
    ("automata_neutro", "vanilla"),
    ("axis", "vanilla"),
    ("axis", "automata_neutro"),
    ("axis", "axis_pec_only"),
]


def load_group(results_dir: Path, name: str):
    d = np.load(results_dir / f"{name}_embeddings.npz", allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    lengths = d["lengths"].astype(int)
    return emb, lengths


def load_first_words(results_dir: Path, name: str):
    resp_path = results_dir.parent.parent / "sia" / "prompts"
    # Las respuestas de texto viven junto a esta corrida en el Escritorio
    # (copia sincronizada); si no están ahí, se omite el análisis léxico.
    candidates = [
        Path("/home/plaxius/Escritorio/Buscando_la_geometría/LSGOT_v4/data/sia_extended_v5")
        / f"{name}_responses.json",
    ]
    for p in candidates:
        if p.exists():
            data = json.loads(p.read_text())
            words = []
            for item in data:
                resp = item.get("response", "").strip()
                first = resp.split()[0] if resp.split() else ""
                first = first.strip(".,;:!¿?\"'()*").lower()
                words.append(first)
            return words
    return None


def shannon_entropy(words):
    c = Counter(words)
    n = sum(c.values())
    if n == 0:
        return 0.0
    probs = np.array([v / n for v in c.values()])
    return float(-(probs * np.log2(probs)).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RAW_DIR)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    v_hat = np.load(V_HAT_PATH)
    assert abs(np.linalg.norm(v_hat) - 1.0) < 1e-4, "v_identidad no normalizado"

    tester = GeometricStatisticalTests(n_permutations=args.n_perm, random_seed=args.seed)

    proj_t0 = {}       # proyección coseno en t=0, por condición
    proj_rest = {}      # proyección media en t>0 (mismo prompt), por condición
    first_words = {}

    print(f"Cargando embeddings desde: {results_dir}\n")
    for g in GROUPS:
        emb, lengths = load_group(results_dir, g)
        t0_vals = []
        rest_vals = []
        for i, L in enumerate(lengths):
            if L < 2:
                continue
            traj = emb[i, :L]
            proj_series = project_trajectory(traj, v_hat)  # cos(v_t, v_hat), t=0..L-1
            t0_vals.append(float(proj_series[0]))
            rest_vals.append(float(np.mean(proj_series[1:])))
        proj_t0[g] = t0_vals
        proj_rest[g] = rest_vals
        fw = load_first_words(results_dir, g)
        if fw is not None:
            first_words[g] = fw
        print(f"  {g:20s} n={len(t0_vals):3d}  "
              f"t=0: {np.mean(t0_vals):+.4f} ± {np.std(t0_vals):.4f}   "
              f"t>0: {np.mean(rest_vals):+.4f} ± {np.std(rest_vals):.4f}")

    # ── Test de permutación en t=0, pares clave ──────────────────────────
    print("\n=== Pares clave — proyección v̂ en t=0 (permutación, n=%d) ===" % args.n_perm)
    key_pair_results = {}
    for a, b in KEY_PAIRS:
        res = tester.full_comparison(proj_t0[a], proj_t0[b], a, b)
        key_pair_results[f"{a}_vs_{b}"] = res
        d = res["cohens_d"]
        p = res["permutation_test"]["p_value"]
        print(f"  {a:16s} vs {b:16s}  d={d:+.3f} ({res['effect_size_interpretation']:10s}) "
              f"p={p:.4f}")

    # ── t=0 vs t>0 dentro de cada condición (T1: control obligatorio) ────
    print("\n=== T1 — t=0 vs t>0 (Wilcoxon pareado, misma trayectoria) ===")
    t0_vs_rest_results = {}
    for g in GROUPS:
        t0 = np.array(proj_t0[g])
        rest = np.array(proj_rest[g])
        try:
            w = sp_stats.wilcoxon(t0, rest)
            wp = float(w.pvalue)
        except ValueError:
            wp = float("nan")
        d_within = tester.cohens_d(t0.tolist(), rest.tolist())
        t0_vs_rest_results[g] = {
            "mean_t0": float(t0.mean()), "mean_rest": float(rest.mean()),
            "wilcoxon_p": wp, "cohens_d": d_within,
        }
        print(f"  {g:20s}  t=0={t0.mean():+.4f}  t>0={rest.mean():+.4f}  "
              f"d={d_within:+.3f}  wilcoxon p={wp:.4f}")

    # ── Distribución léxica del primer token (proxy) ──────────────────────
    lexical = {}
    if first_words:
        print("\n=== Primer 'token' textual por condición (proxy, NO logits) ===")
        for g in GROUPS:
            if g not in first_words:
                continue
            words = first_words[g]
            ent = shannon_entropy(words)
            top3 = Counter(words).most_common(3)
            lexical[g] = {"entropy_bits": ent, "n": len(words), "top3": top3}
            print(f"  {g:20s} H={ent:.3f} bits  top3={top3}")
    else:
        print("\n(Sin datos de _responses.json disponibles — se omite el análisis léxico)")

    # ── Guardar resultados ────────────────────────────────────────────────
    out = {
        "experiment": "E-L",
        "v_hat_source": str(V_HAT_PATH),
        "results_dir": str(results_dir),
        "proj_t0_summary": {g: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                             for g, v in proj_t0.items()},
        "proj_rest_summary": {g: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                               for g, v in proj_rest.items()},
        "key_pairs_t0": key_pair_results,
        "t0_vs_rest": t0_vs_rest_results,
        "lexical_first_word": lexical,
    }
    out_path = Path(__file__).parent / "EL_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
