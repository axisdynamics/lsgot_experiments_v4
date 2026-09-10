#!/usr/bin/env python3
"""
Command R 35B — análisis de recuperación H4_rev sobre la perturbación T2
(los mismos 8 grupos del panel "preguntas": axis, generic_long,
generic_short, vanilla, automata_neutro, axis_pec_only, witness_soul_md,
soul_elena_financial).

Pregunta central (misma del paper, lsgot_4.md §3.3): ¿el grupo con
identidad+restricción (axis, axis_pec_only) recupera su firma geométrica
tras la perturbación, mientras que automata_neutro (restricción sin
identidad) se estanca por debajo y nunca cierra la brecha, en los tres
puntos de inyección (t_inj=50,128,200)?

Usa recovery_analyzer.py (mismo protocolo, τ_threshold=0.95, capa de
captura final) sin modificar su lógica — solo adaptado al set de grupos y
rutas de este panel.
"""
import sys, json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "perturbation"))
from recovery_analyzer import compute_recovery, aggregate_recovery, compare_groups  # noqa: E402

D = Path("/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/command_r_35b_combined/results_t2/trajectories")
GROUPS = ["axis", "generic_long", "generic_short", "vanilla", "automata_neutro",
          "axis_pec_only", "witness_soul_md", "soul_elena_financial"]
T_INJ_VALUES = [50, 128, 200]


def load_traj(name):
    d = np.load(D / f"{name}.npz")
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int), d["prompt_ids"]


print("Cargando baseline + 3 perturbaciones x 8 grupos...")
baselines = {g: load_traj(f"{g}_baseline_embeddings") for g in GROUPS}
perturbeds = {(g, t): load_traj(f"{g}_perturbed_t{t}_embeddings") for g in GROUPS for t in T_INJ_VALUES}

# Sanity check: mismo orden de prompt_ids en baseline y cada perturbado
for g in GROUPS:
    _, _, ids_base = baselines[g]
    for t in T_INJ_VALUES:
        _, _, ids_pert = perturbeds[(g, t)]
        assert list(ids_base) == list(ids_pert), f"prompt_ids desalineados en {g} t={t}"

# ══════════════════════════════════════════════════════════════════════
# Recuperación por prompt, agregada por grupo x t_inj
# ══════════════════════════════════════════════════════════════════════
agg_by_tinj = {t: {} for t in T_INJ_VALUES}
raw_by_tinj = {t: {} for t in T_INJ_VALUES}

for t in T_INJ_VALUES:
    print(f"\n{'=' * 70}\nt_inj = {t}\n{'=' * 70}")
    for g in GROUPS:
        emb_b, len_b, _ = baselines[g]
        emb_p, len_p, _ = perturbeds[(g, t)]
        results = []
        for i in range(len(len_b)):
            base_traj = emb_b[i, :len_b[i]]
            pert_traj = emb_p[i, :len_p[i]]
            results.append(compute_recovery(base_traj, pert_traj, t))
        raw_by_tinj[t][g] = results
        agg = aggregate_recovery(results)
        agg_by_tinj[t][g] = agg
        tau = agg.get("tau_tokens", {})
        print(f"  {g:22s} recovery_rate={agg.get('recovery_rate', 0):.2f}  "
              f"tau_mean={tau.get('mean')}  n_ok={agg.get('n_ok')}/{agg.get('n_total')}")

# ══════════════════════════════════════════════════════════════════════
# Comparación entre grupos (recovery_analyzer.compare_groups, mismo diseño
# que run_perturbation_t2.py) + deltas extra para witness_soul_md /
# soul_elena_financial, no cubiertos por compare_groups (diseñado para el
# panel original de Gemma).
# ══════════════════════════════════════════════════════════════════════
comparisons = {}
for t in T_INJ_VALUES:
    comp = compare_groups(agg_by_tinj[t])
    for key in ("tau_tokens", "recovery_rate", "recovery_gap"):
        row = comp[key]
        an = row.get("automata_neutro")
        for soul in ("witness_soul_md", "soul_elena_financial"):
            v = row.get(soul)
            row[f"delta_{soul}_minus_automata_neutro"] = (v - an) if v is not None and an is not None else None
    comparisons[t] = comp

print("\n" + "=" * 70 + "\nComparación clave: automata_neutro vs axis/axis_pec_only/vanilla\n" + "=" * 70)
for t in T_INJ_VALUES:
    print(f"\n  t_inj={t}:")
    for key in ("recovery_rate", "tau_tokens", "recovery_gap"):
        row = comparisons[t][key]
        print(f"    {key:16s} axis={row.get('axis')}  axis_pec_only={row.get('axis_pec_only')}  "
              f"vanilla={row.get('vanilla')}  automata_neutro={row.get('automata_neutro')}  "
              f"witness_soul_md={row.get('witness_soul_md')}  soul_elena_financial={row.get('soul_elena_financial')}")
        print(f"      delta(automata_neutro - axis)={row.get('delta_automata_neutro_minus_axis')}  "
              f"delta(automata_neutro - vanilla)={row.get('delta_automata_neutro_minus_vanilla')}")

# ══════════════════════════════════════════════════════════════════════
# Veredicto automático: ¿se replica la brecha de recuperación en los TRES
# puntos de inyección, en la MISMA dirección que el paper (automata_neutro
# por debajo de axis/axis_pec_only/vanilla)?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nVeredicto: ¿automata_neutro se estanca en los 3 t_inj?\n" + "=" * 70)
gap_holds = []
for t in T_INJ_VALUES:
    rr = comparisons[t]["recovery_rate"]
    an, ax, apo, van = rr.get("automata_neutro"), rr.get("axis"), rr.get("axis_pec_only"), rr.get("vanilla")
    refs = [v for v in (ax, apo, van) if v is not None]
    below_all = an is not None and refs and all(an <= r for r in refs)
    gap_holds.append(below_all)
    print(f"  t_inj={t}: automata_neutro recovery_rate={an}  <=  todas las referencias (axis/axis_pec_only/vanilla)={refs}  ->  {below_all}")

replicates = all(gap_holds)
print(f"\n  REPLICA en los 3 puntos de inyección: {replicates}")

out = {
    "agg_by_tinj": {str(t): {g: agg_by_tinj[t][g] for g in GROUPS} for t in T_INJ_VALUES},
    "comparisons": {str(t): comparisons[t] for t in T_INJ_VALUES},
    "gap_holds_per_tinj": dict(zip(T_INJ_VALUES, gap_holds)),
    "replicates_all_tinj": replicates,
}
Path(__file__).parent.joinpath("command_r_t2_recovery_results.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False, default=str))
print("\nGuardado: command_r_t2_recovery_results.json")
