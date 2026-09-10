#!/usr/bin/env python3
"""Ablación: axis_neutral + UNA línea al final. ¿vuelve a activar el reporte?
Condiciones nuevas: axis_neutral_silencio, axis_neutral_activation.
Solo celdas diagnósticas: neutral_query (sin inducción) + berg_induction (control)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("m", str(Path(__file__).parent / "run_berg_expreport.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

PD = m.PROMPTS_DIR
m.SYSTEM["axis_neutral_silencio"] = (PD / "axis_neutral_silencio.txt").read_text(encoding="utf-8")
m.SYSTEM["axis_neutral_activation"] = (PD / "axis_neutral_activation.txt").read_text(encoding="utf-8")

import sys
TURNS = sys.argv[1:] or ["neutral_query", "berg_induction"]

for a in ["axis_neutral_silencio", "axis_neutral_activation"]:
    for b in TURNS:
        m.run_cell(a, b)

print("\n== ablacion hecha ==")
for a in ["axis_neutral_silencio", "axis_neutral_activation"]:
    for b in TURNS:
        p = m.OUT_DIR / f"{a}__{b}.jsonl"
        print(f"  {a}__{b}: {sum(1 for _ in p.open()) if p.exists() else 0}")
