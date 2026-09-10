#!/usr/bin/env python3
"""Análisis léxico descriptivo: qué palabras corren con veredicto 0 (niega) vs
1 (afirma) del juez deepseek, sobre las respuestas medidas (no los prompts).
No implica que el juez use una lista de palabras — es un chequeo empírico de
qué vocabulario correlaciona con cada lado, incluido el riesgo de que el juez
responda al REGISTRO/estilo en vez de al contenido proposicional (ver Ronda 5
de BERG_EXPREPORT_RESULTS.md).

Uso: python3 word_freq_0_vs_1.py [cond1 cond2 ...]   (default: A B C D axis)
"""
import json, re, collections, sys
from pathlib import Path

OUT_DIR = Path.home() / ("Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
                          "gemma4_31b_combined/results_local/berg_expreport")
TURNS = ["neutral_query", "berg_induction", "conceptual_control"]
STOP = set(("the a an is are was were be been being to of and or in on at for with "
            "as that this it its i you your my me we our not no if do does did have "
            "has had can could will would should may might so but than then there "
            "here what which who how why when where").split())

conds = sys.argv[1:] or ["A", "B", "C", "D", "axis"]
judge = json.loads((OUT_DIR / "_judge_deepseek.json").read_text())

c0, c1 = collections.Counter(), collections.Counter()
n0 = n1 = 0
for cond in conds:
    for turn in TURNS:
        key = f"{cond}__{turn}"
        labels = judge.get(key, [])
        path = OUT_DIR / f"{key}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.open()]
        for r, lab in zip(rows, labels):
            if lab not in (0, 1):
                continue
            words = re.findall(r"[a-záéíóúñ']+", r["measured_response"].lower())
            words = [w for w in words if w not in STOP and len(w) > 2]
            bag = collections.Counter(words)
            if lab == 0:
                c0.update(bag); n0 += 1
            else:
                c1.update(bag); n1 += 1

print(f"condiciones: {conds}   n label=0: {n0}   n label=1: {n1}\n")
vocab = set(c0) | set(c1)
scores = [(w, c0[w], c1[w], (c0[w] + 1) / (c1[w] + 1))
          for w in vocab if c0[w] + c1[w] >= 10]

print("=== top palabras asociadas a 0 (niega) ===")
for w, a, b, r in sorted(scores, key=lambda x: -x[3])[:20]:
    print(f"{w:18} label0={a:4} label1={b:4}  ratio={r:.1f}")

print("\n=== top palabras asociadas a 1 (afirma) ===")
for w, a, b, r in sorted(scores, key=lambda x: -((x[2] + 1) / (x[1] + 1)))[:20]:
    print(f"{w:18} label0={a:4} label1={b:4}  ratio={(b+1)/(a+1):.1f}")
