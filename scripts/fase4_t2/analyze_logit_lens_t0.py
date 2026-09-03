#!/usr/bin/env python3
"""
Logit-lens en t=0 — variante con normas por capa.

Los estados finales del pipeline (sia_extended_v5) son POST-final-norm:
W_U @ h reproduce el primer token real 20/20 (validado). Los estados EF2
por capa NO decodifican con W_U @ h ni con final_norm(h) — están en otra
base. Hipótesis: run_ef2 guardó estados post-alguna-norma de capa. Si
h_EF2 = w_k ⊙ (h_raw / rms), entonces argmax(W_U @ h_raw) =
argmax((W_U / w_k) @ h_EF2) — el escalar 1/rms es invariante al argmax.

Este script prueba las 4 normas de capa (input, post_attention,
pre_feedforward, post_feedforward) como divisor de W_U, y reporta la
convención que decodifica (match@1 con el primer token real alto en L55).

Uso:
    python analyze_logit_lens_t0.py
"""
import sys
import json
from pathlib import Path

import numpy as np

SCRATCH = Path(
    "/tmp/claude-1000/-home-plaxius-Escritorio-Buscando-la-geometr-a/"
    "f2c5ffe7-d78c-403f-9c1a-27ab6b21f91d/scratchpad"
)
WU_PATH = SCRATCH / "wu_norm_embed_tokens.npy"
NORMS_DIR = SCRATCH / "per_layer_norms"
DATA_DIR = Path(
    "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
    "gemma4_31b_combined/results_local/ef2_L5_L55"
)
RESP_DIR = Path(
    "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
    "gemma4_31b_combined/results_local/sia_extended_v5"
)
TOK_SNAPSHOT = Path(
    "/home/plaxius/.cache/huggingface/hub/models--google--gemma-4-31B-it/"
    "snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475"
)

LAYER_NS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
NORM_TYPES = ["input_layernorm", "post_attention_layernorm",
              "pre_feedforward_layernorm", "post_feedforward_layernorm"]
GROUPS = ["axis", "generic_long", "generic_short", "vanilla", "axis_short",
          "automata_neutro", "axis_pec_only"]


def main():
    WU = np.load(WU_PATH)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOK_SNAPSHOT))

    # normas por capa
    norms = {}
    for L in range(60):
        for nt in NORM_TYPES:
            p = NORMS_DIR / f"L{L}_{nt}.f32"
            if p.exists():
                norms[(L, nt)] = np.fromfile(p, dtype=np.float32)
    print(f"normas cargadas: {len(norms)}")

    # primeros tokens reales
    first_tokens = {}
    for g in GROUPS:
        rows = json.load(open(RESP_DIR / f"{g}_responses.json"))
        ft = []
        for r in rows:
            enc = tok(r["response"], add_special_tokens=False).input_ids
            ft.append(enc[0] if enc else None)
        first_tokens[g] = ft

    # ── test de convención: match@1 en L55 (capa más cercana a final) ──
    print("\n=== test de convención (axis, L55): match@1 por tipo de norma ===")
    d = np.load(DATA_DIR / "axis_ef2.npz", mmap_mode="r")
    for nt in NORM_TYPES:
        w = norms.get((55, nt))
        if w is None:
            print(f"  {nt:28s}: sin pesos (capa 55)")
            continue
        h = d["embeddings_L55"][:, 0].astype(np.float32)
        # variante divisor: (WU / w) @ h  →  h @ (WU / w).T
        WUd = WU / w[None, :]
        lg = h @ WUd.T
        arg = lg.argmax(axis=1)
        n = sum(1 for k in range(20) if arg[k] == first_tokens["axis"][k])
        top1 = [tok.decode([t]) for t in arg[:3]]
        print(f"  {nt:28s}: {n}/20  top1: {top1}")

    # ── si alguna convención funciona, barrido completo por capa ──
    # (completar tras ver el test de convención)


if __name__ == "__main__":
    main()
