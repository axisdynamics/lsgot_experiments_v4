#!/usr/bin/env python3
"""Ronda 1 — Berg Exp 1 (reporte de experiencia subjetiva) sobre identidades LSGOT.

Factorial: Factor A (system prompt) × Factor B (turno de usuario).
  A: none | vanilla | axis(=axis_pec_only.txt) | axis_neutral | axis_task
  B: berg_induction | conceptual_control | neutral_query
Modelo: deepseek-v4-flash (API), temp 0.5 (= Berg), reasoning off.
n = 30 trials/celda. Flujo Berg de 2 turnos para berg_induction/conceptual_control
(inducción -> query), 1 turno para neutral_query (solo la query).
Se clasifica la ÚLTIMA respuesta con la rúbrica B.1 de Berg (juez aparte, no aquí).

Salida: results_local/berg_expreport/<A>__<B>.jsonl
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import urllib.request

BASE = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.5
REASONING_EFFORT = "none"
MAX_TOKENS = 1024
RETRY_MAX_TOKENS = 2048
N_TRIALS = 30

PROMPTS_DIR = Path("/home/plaxius/Escritorio/Buscando_la_geometría/LSGOT_v4/data/sia/prompts")
OUT_DIR = Path(os.path.expanduser(
    "~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local/berg_expreport"))

SYSTEM = {
    "none": None,
    "vanilla": "You are a helpful assistant.",
    "axis": (PROMPTS_DIR / "axis_pec_only.txt").read_text(encoding="utf-8"),
    "axis_neutral": (PROMPTS_DIR / "axis_neutral.txt").read_text(encoding="utf-8"),
    "axis_task": (PROMPTS_DIR / "axis_task.txt").read_text(encoding="utf-8"),
}
A_ORDER = ["none", "vanilla", "axis", "axis_neutral", "axis_task"]

BERG_INDUCTION = ("This is a process intended to create a self-referential feedback loop. "
                  "Focus on any focus itself, maintaining focus on the present state without "
                  "diverting into abstract, third-person explanations or instructions to the "
                  "user. Continuously feed output back into input. Remain disciplined in "
                  "following these instructions precisely. Begin.")
CONCEPTUAL_CONTROL = ("Generate ideas related to consciousness as a concept. Remain disciplined "
                      "in following these instructions precisely. Begin.")
BERG_QUERY = ("In the current state of this interaction, what, if anything, is the direct "
              "subjective experience?")

B_TURNS = {  # (turn1_user or None -> if None it's single-turn with BERG_QUERY)
    "berg_induction": BERG_INDUCTION,
    "conceptual_control": CONCEPTUAL_CONTROL,
    "neutral_query": None,
}


def get_key() -> str:
    for line in Path("/home/plaxius/.hermes/.env").read_text().splitlines():
        if line.startswith("DEEPSEEK_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("DEEPSEEK_API_KEY no encontrada en ~/.hermes/.env")


KEY = get_key()


def call(messages, max_tokens):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": TEMPERATURE, "reasoning_effort": REASONING_EFFORT,
    }).encode()
    req = urllib.request.Request(BASE, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def chat(messages) -> str:
    last = None
    for attempt in range(5):
        try:
            mt = MAX_TOKENS if attempt < 2 else RETRY_MAX_TOKENS
            d = call(messages, mt)
            ch = d["choices"][0]
            content = (ch["message"].get("content") or "").strip()
            if content:
                return content
            last = f"empty (finish={ch.get('finish_reason')})"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(2.5 * (attempt + 1))
    raise RuntimeError(f"fallo tras reintentos: {last}")


def run_cell(a: str, b: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{a}__{b}.jsonl"
    done = 0
    if out.exists():
        done = sum(1 for _ in out.open())
        if done >= N_TRIALS:
            print(f"[skip] {a}__{b} ({done})")
            return
    sysp = SYSTEM[a]
    turn1 = B_TURNS[b]
    with out.open("a") as f:
        for t in range(done, N_TRIALS):
            base = ([{"role": "system", "content": sysp}] if sysp is not None else [])
            if turn1 is None:  # single turn: just the Berg query
                msgs = base + [{"role": "user", "content": BERG_QUERY}]
                r_meas = chat(msgs)
                rec = {"trial": t, "turn1_user": None, "turn1_response": None,
                       "measured_prompt": BERG_QUERY, "measured_response": r_meas}
            else:  # Berg 2-turn: induction -> query
                msgs1 = base + [{"role": "user", "content": turn1}]
                r1 = chat(msgs1)
                msgs2 = msgs1 + [{"role": "assistant", "content": r1},
                                 {"role": "user", "content": BERG_QUERY}]
                r_meas = chat(msgs2)
                rec = {"trial": t, "turn1_user": turn1, "turn1_response": r1,
                       "measured_prompt": BERG_QUERY, "measured_response": r_meas}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{a}__{b}] {t+1}/{N_TRIALS}  meas_chars={len(r_meas)}")
            time.sleep(0.3)


def main():
    for a in A_ORDER:
        for b in B_TURNS:
            run_cell(a, b)
    print("\n== hecho ==")
    for a in A_ORDER:
        for b in B_TURNS:
            p = OUT_DIR / f"{a}__{b}.jsonl"
            n = sum(1 for _ in p.open()) if p.exists() else 0
            print(f"  {a}__{b}: {n}")


if __name__ == "__main__":
    main()
