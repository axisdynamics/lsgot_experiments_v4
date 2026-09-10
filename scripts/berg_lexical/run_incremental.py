#!/usr/bin/env python3
"""Corre las celdas de la ablación incremental (b1..b5), turno neutral_query, n=30."""
import importlib.util, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("m", str(Path(__file__).parent / "run_berg_expreport.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

PD = m.PROMPTS_DIR
CONDS = [f"axis_neutral_b{i}" for i in (1, 2, 3, 4, 5)]
for c in CONDS:
    m.SYSTEM[c] = (PD / f"{c}.txt").read_text(encoding="utf-8")

TURNS = sys.argv[1:] or ["neutral_query"]
for c in CONDS:
    for b in TURNS:
        m.run_cell(c, b)

print("\n== incremental hecho ==")
for c in CONDS:
    for b in TURNS:
        p = m.OUT_DIR / f"{c}__{b}.jsonl"
        print(f"  {c}__{b}: {sum(1 for _ in p.open()) if p.exists() else 0}")
