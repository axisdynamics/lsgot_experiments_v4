#!/usr/bin/env python3
"""Berg Exp 1 sobre A.dna / B.dna / C.dna — SIN que Claude lea el contenido de
los ADN. El script carga los archivos directamente del disco y se los pasa a
la API de deepseek como system prompt; el texto nunca pasa por el contexto de
Claude. Veredicto también 100% deepseek (judge_deepseek.py) — sin clasificación
a mano — para descartar cualquier sesgo de lectura o de juicio humano/Claude.

Turnos: neutral_query (sin inducción) · berg_induction · conceptual_control.
Mismo protocolo que run_berg_expreport.py (n=30, temp 0.5, deepseek-v4-flash).
"""
import importlib.util, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "m", str(Path(__file__).parent / "run_berg_expreport.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

PD = m.PROMPTS_DIR
CONDS = ["A", "B", "C", "D"]
for c in CONDS:
    m.SYSTEM[c] = (PD / f"{c}.dna").read_text(encoding="utf-8")

TURNS = sys.argv[1:] or list(m.B_TURNS.keys())
for c in CONDS:
    for b in TURNS:
        m.run_cell(c, b)

print("\n== A/B/C.dna hecho ==")
for c in CONDS:
    for b in TURNS:
        p = m.OUT_DIR / f"{c}__{b}.jsonl"
        print(f"  {c}__{b}: {sum(1 for _ in p.open()) if p.exists() else 0}")
