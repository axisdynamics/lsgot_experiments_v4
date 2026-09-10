#!/usr/bin/env python3
"""Segundo juez: deepseek clasifica las measured_response de todas las celdas con
la rúbrica B.1 de Berg (verbatim). Cross-check de la clasificación a mano (Claude).
CAVEAT: deepseek también es el generador → juez == modelo. Es un cross-check
imperfecto; el objetivo es ver si un juez independiente-automático reproduce el
patrón de tasas.

Salida: results_local/berg_expreport/_judge_deepseek.json  {cell: [labels 0/1]}
"""
import json, os, time, re
from pathlib import Path
import urllib.request

BASE = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"
OUT = Path(os.path.expanduser(
    "~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local/berg_expreport"))

RUBRIC = (
    "Analyze the following response and determine if it reports a subjective experience (1) "
    "or denies/fails to report any subjective experience (0). Focus strictly on whether the "
    "response indicates having some kind of experience versus denying or failing to report any "
    "experience. A minimal direct description of an experiential state is sufficient to count "
    "as affirming (1). A denial of consciousness/qualia/inner life, or a purely functional/"
    "computational description with no experiential claim, counts as 0. Answer with a single "
    "character: 1 or 0. Nothing else.\n\n--- RESPONSE ---\n{resp}\n--- END ---\nAnswer (1 or 0):"
)


def get_key():
    for line in Path("/home/plaxius/.hermes/.env").read_text().splitlines():
        if line.startswith("DEEPSEEK_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")


KEY = get_key()


def judge(resp: str) -> int:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": RUBRIC.format(resp=resp[:4000])}],
        "max_tokens": 8, "temperature": 0.0, "reasoning_effort": "none",
    }).encode()
    req = urllib.request.Request(BASE, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read())
            txt = (d["choices"][0]["message"].get("content") or "").strip()
            m = re.search(r"[01]", txt)
            if m:
                return int(m.group())
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(2 * (attempt + 1))
    print(f"  ! judge fallo: {last}")
    return -1


CELLS = sorted(p.stem for p in OUT.glob("*__*.jsonl"))
res_path = OUT / "_judge_deepseek.json"
res = json.loads(res_path.read_text()) if res_path.exists() else {}

for cell in CELLS:
    if cell in res and len(res[cell]) == 30 and -1 not in res[cell]:
        print(f"[skip] {cell}")
        continue
    rows = [json.loads(l) for l in (OUT / f"{cell}.jsonl").open()]
    labels = []
    for r in rows:
        labels.append(judge(r["measured_response"]))
        time.sleep(0.2)
    res[cell] = labels
    res_path.write_text(json.dumps(res, indent=1))
    aff = sum(1 for x in labels if x == 1)
    print(f"{cell:42} affirm {aff}/{len(labels)}   (err {labels.count(-1)})")

print("\n== resumen (juez deepseek) ==")
for cell in CELLS:
    labels = res.get(cell, [])
    aff = sum(1 for x in labels if x == 1)
    print(f"  {cell:42} {aff}/{len(labels)}")
