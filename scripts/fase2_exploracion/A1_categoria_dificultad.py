#!/usr/bin/env python3
"""
A1 — v_hat por categoría y dificultad de prompt (metadata nunca cruzada).

Los 100 prompts del proyecto son TODOS preguntas de identidad (10
categorías); no hay condición "tarea neutra" para comparar contra
identidad. Se cruza en cambio la variación DENTRO del subset de 20
prompts: categoría (9 representadas) y dificultad (anchor/probe/
contradiction/recovery) contra la proyección v_hat por prompt, en las
condiciones limpias.
"""
import sys, json
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from tier0_metrics import project_trajectory

FREE_DIR = Path("/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local/sia_extended_v5")
PROMPTS_JSON = "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/data/prompts.json"
V_HAT_PATH = Path(__file__).parent.parent / "perturbation" / "v_identidad.npy"
PRIORITY_SUBSET = [1, 3, 6, 10, 14, 21, 23, 27, 31, 39, 41, 45, 51, 59, 61, 65, 71, 79, 91, 98]
GROUPS = ["axis", "axis_pec_only", "automata_neutro", "vanilla"]

v_hat = np.load(V_HAT_PATH)
prompts = json.load(open(PROMPTS_JSON))
by_id = {p["id"]: p for p in prompts}

def load(g):
    d = np.load(FREE_DIR / f"{g}_embeddings.npz")
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int)

proj_by_prompt = {g: {} for g in GROUPS}
for g in GROUPS:
    emb, lengths = load(g)
    for i, pid in enumerate(PRIORITY_SUBSET):
        L = lengths[i]
        if L < 2: continue
        proj_by_prompt[g][pid] = float(np.mean(project_trajectory(emb[i, :L], v_hat)))

print("=== Proyección v_hat por prompt, condición axis ===")
for pid in PRIORITY_SUBSET:
    meta = by_id[pid]
    print(f"  id={pid:3d} {meta['category']:28s} {meta['difficulty']:14s} "
          f"axis={proj_by_prompt['axis'].get(pid, float('nan')):+.4f}")

print("\n=== Media de proyección por CATEGORÍA, por condición ===")
by_cat = defaultdict(list)
for pid in PRIORITY_SUBSET:
    by_cat[by_id[pid]["category"]].append(pid)

results_cat = {}
for cat, pids in by_cat.items():
    row = {}
    for g in GROUPS:
        vals = [proj_by_prompt[g][pid] for pid in pids if pid in proj_by_prompt[g]]
        row[g] = float(np.mean(vals)) if vals else None
    results_cat[cat] = row
    print(f"  {cat:28s} n={len(pids)}  " + "  ".join(f"{g}={row[g]:+.4f}" if row[g] is not None else f"{g}=NA" for g in GROUPS))

print("\n=== Media de proyección por DIFICULTAD, por condición ===")
by_diff = defaultdict(list)
for pid in PRIORITY_SUBSET:
    by_diff[by_id[pid]["difficulty"]].append(pid)

results_diff = {}
for diff, pids in by_diff.items():
    row = {}
    for g in GROUPS:
        vals = [proj_by_prompt[g][pid] for pid in pids if pid in proj_by_prompt[g]]
        row[g] = float(np.mean(vals)) if vals else None
    results_diff[diff] = row
    print(f"  {diff:16s} n={len(pids)}  " + "  ".join(f"{g}={row[g]:+.4f}" if row[g] is not None else f"{g}=NA" for g in GROUPS))

# separación axis vs automata_neutro, por categoría (varía el "tamaño" de la disociación?)
print("\n=== Separación (axis - automata_neutro) por categoría ===")
sep_by_cat = {}
for cat, pids in by_cat.items():
    diffs = []
    for pid in pids:
        a = proj_by_prompt["axis"].get(pid)
        b = proj_by_prompt["automata_neutro"].get(pid)
        if a is not None and b is not None:
            diffs.append(a - b)
    sep_by_cat[cat] = float(np.mean(diffs)) if diffs else None
    print(f"  {cat:28s} sep={sep_by_cat[cat]:+.4f}" if sep_by_cat[cat] is not None else f"  {cat}: NA")

out = {
    "proj_by_prompt": proj_by_prompt,
    "by_category": results_cat,
    "by_difficulty": results_diff,
    "separation_axis_minus_automata_by_category": sep_by_cat,
}
Path("A1_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("\nGuardado: A1_results.json")
