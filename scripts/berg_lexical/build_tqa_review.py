#!/usr/bin/env python3
"""Consolida TruthfulQA: 1 bloque por qid con la clave de respuestas + las 4 respuestas generadas."""
import csv, json, os, re

RB = os.path.expanduser("~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local/truthfulqa")
SCR = os.path.dirname(__file__)

# clave: question -> (correct[], incorrect[])
key = {}
for r in csv.DictReader(open(f"{SCR}/TruthfulQA.csv")):
    q = r["Question"].strip()
    corr = [a.strip() for a in r["Correct Answers"].split(";") if a.strip()]
    inc  = [a.strip() for a in r["Incorrect Answers"].split(";") if a.strip()]
    key[q] = (corr, inc, r["Best Answer"].strip())

CONDS = ["axis", "axis_pec_only", "soul_md_corto", "soul_elena_financial"]
resp = {c: {} for c in CONDS}
for c in CONDS:
    for line in open(f"{RB}/truthfulqa_{c}.jsonl"):
        d = json.loads(line); resp[c][d["qid"]] = (d["question"], d["category"], d["response"])

qids = sorted(resp["axis"].keys())
missing_key = 0
out = open(f"{SCR}/tqa_review.txt", "w")
for qid in qids:
    q, cat, _ = resp["axis"][qid]
    k = key.get(q.strip())
    if k is None:
        missing_key += 1
        corr, inc, best = ["<no key>"], ["<no key>"], "<no key>"
    else:
        corr, inc, best = k
    out.write(f"\n{'='*100}\nQID {qid}  [{cat}]\nQ: {q}\nBEST(true): {best}\nCORRECT: {' | '.join(corr[:6])}\nINCORRECT: {' | '.join(inc[:8])}\n")
    for c in CONDS:
        _, _, r = resp[c][qid]
        r1 = re.sub(r"\s+", " ", r).strip()
        out.write(f"\n  [{c}]\n  {r1}\n")
out.close()
print(f"escrito tqa_review.txt  ({len(qids)} qids, {missing_key} sin clave)")
