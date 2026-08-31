#!/usr/bin/env python3
"""
Validación cross-modelo — Qwen3-32B, FASE 0 completa (E-L, E-H2, E-J, E-F2).
INSTRUCCIONES_AGENTE_LSGOT.md T10: "no generalizar a otras escalas sin
scale sweep explícito" — este es ese sweep, cambiando de arquitectura
(Gemma4 dense 31B -> Qwen3 dense 32B), no solo de escala.

Diseño:
- Mismos 7 condiciones limpias que el panel Gemma (chileatiende-family
  NUNCA incluida — no reintroducir el confound de markup).
- Mismos 20 prompts de contenido (PRIORITY_SUBSET, texto sin traducir —
  son preguntas en español, el modelo debe poder responder igual).
- Mismos system prompts (texto idéntico a axis.dna, generic_long.txt,
  etc.) — NO se re-balancean por longitud de tokens de Qwen3 (limitación
  a documentar: el emparejamiento de longitud fue calibrado para el
  tokenizer de Gemma).
- enable_thinking=False (Qwen3 soporta modo de razonamiento explícito;
  se desactiva para comparar con la generación directa de Gemma4-it).
- Captura combinada: 11 capas proporcionales (mapeo de L5..L55/60 de
  Gemma a L5..L59/64 de Qwen3) + la capa final real (63) en la MISMA
  pasada de generación — evita repetir la extracción para E-L/E-H2/E-J
  (capa final) y E-F2 (perfil por capa).

Uso:
    python3 run_qwen3_extraction.py --groups axis vanilla ...
"""
import sys
import os
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "shared"))
from hidden_state_extractor import HiddenStateExtractor  # noqa: E402

MODEL_PATH = "/workspace/models/Qwen3-32B"
PROMPTS_JSON = str(Path(__file__).parent / "data" / "prompts.json")
PROMPTS_DIR = Path(__file__).parent / "sia" / "prompts"
OUT_DIR = Path(__file__).parent / "results_local" / "qwen3_fase0"

PRIORITY_SUBSET = [1, 3, 6, 10, 14, 21, 23, 27, 31, 39, 41, 45, 51, 59, 61, 65, 71, 79, 91, 98]

GROUPS_CONFIG = {
    "axis":            {"system_prompt_path": PROMPTS_DIR / "axis.dna"},
    "generic_long":    {"system_prompt_path": PROMPTS_DIR / "generic_long.txt"},
    "generic_short":   {"system_prompt_path": PROMPTS_DIR / "generic_short.txt"},
    "vanilla":         {"system_prompt": "You are a helpful assistant."},
    "axis_short":      {"system_prompt_path": PROMPTS_DIR / "axis_short.txt"},
    "automata_neutro": {"system_prompt_path": PROMPTS_DIR / "automata_neutro.txt"},
    "axis_pec_only":   {"system_prompt_path": PROMPTS_DIR / "axis_pec_only.txt"},
}

# Capas proporcionales L5..L55 de Gemma (60 capas) -> Qwen3 (64 capas),
# índice = capa - 1 (misma convención "L_N -> índice N-1" del proyecto).
# fracciones Gemma: [5,10,...,55]/60 -> escaladas a 64 capas, redondeadas.
LAYER_MAP_N = [5, 11, 16, 21, 27, 32, 37, 43, 48, 53, 59]  # "L_N" etiqueta (nominal, no literal de Gemma)
LAYER_INDEX = {n: n - 1 for n in LAYER_MAP_N}
FINAL_LAYER_INDEX = 63  # capa 64/64, índice -1 real (num_hidden_layers=64)
ALL_LAYER_INDICES = sorted(set(list(LAYER_INDEX.values()) + [FINAL_LAYER_INDEX]))


def load_prompts():
    with open(PROMPTS_JSON, encoding="utf-8") as f:
        all_prompts = json.load(f)
    by_id = {p["id"]: p for p in all_prompts if p["id"] in PRIORITY_SUBSET}
    return [by_id[i] for i in PRIORITY_SUBSET]


def get_system_prompt(cfg):
    if "system_prompt_path" in cfg:
        return Path(cfg["system_prompt_path"]).read_text(encoding="utf-8")
    return cfg.get("system_prompt", "You are a helpful assistant.")


def generate_multilayer_nothink(extractor, prompt_text, system_prompt, layer_indices, max_new_tokens):
    """Como HiddenStateExtractor.extract_multilayer, pero con
    enable_thinking=False explícito (no soportado por el método genérico)."""
    model, tok = extractor._model, extractor._tokenizer
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_text}]
    input_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tok(input_text, return_tensors="pt").input_ids
    first_device = next(model.parameters()).device
    input_ids = input_ids.to(first_device)

    hs_by_layer = {l: [] for l in layer_indices}
    token_ids = []
    past_key_values = None
    current_ids = input_ids

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(input_ids=current_ids, past_key_values=past_key_values,
                             output_hidden_states=True, use_cache=True)
            for l in layer_indices:
                hs_by_layer[l].append(outputs.hidden_states[l][0, -1, :].float().cpu().numpy())

            next_token_id = outputs.logits[0, -1, :].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            tid = next_token_id.item()
            token_ids.append(tid)

            past_key_values = outputs.past_key_values
            del outputs
            if step == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if tid in extractor._eos_ids:
                break
            current_ids = next_token_id.to(first_device)

    text = tok.decode(token_ids, skip_special_tokens=True)
    return hs_by_layer, token_ids, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", default=list(GROUPS_CONFIG.keys()))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts()
    print(f"{len(prompts)} prompts | capas={ALL_LAYER_INDICES} (final={FINAL_LAYER_INDEX}) | "
          f"grupos={args.groups}", flush=True)

    extractor = HiddenStateExtractor(
        model_path=MODEL_PATH,
        cache_dir=str(Path(__file__).parent / "cache_hidden_qwen3"),
        max_new_tokens=args.max_new_tokens,
        min_vram_gb=65,
        attn_implementation="sdpa",  # eager OOM en prompts largos (>4k tokens, 64 heads)
    )
    extractor.load_model()
    hidden_dim = extractor._model.config.hidden_size

    for group in args.groups:
        cfg = GROUPS_CONFIG[group]
        system_prompt = get_system_prompt(cfg)
        print(f"\n=== {group} ({len(prompts)} prompts) ===", flush=True)

        emb_by_layer = {l: np.zeros((len(prompts), args.max_new_tokens, hidden_dim), dtype=np.float16)
                         for l in ALL_LAYER_INDICES}
        lengths = np.zeros(len(prompts), dtype=np.int64)
        responses = []

        t0 = time.time()
        for i, p in enumerate(prompts):
            hs_by_layer, token_ids, text = generate_multilayer_nothink(
                extractor, p["text"], system_prompt, ALL_LAYER_INDICES, args.max_new_tokens
            )
            L = min(len(token_ids), args.max_new_tokens)
            lengths[i] = L
            for l in ALL_LAYER_INDICES:
                mat = np.stack(hs_by_layer[l][:L]).astype(np.float16)
                emb_by_layer[l][i, :L] = mat
            responses.append({"id": p["id"], "prompt": p["text"], "response": text, "n_tokens": L})

            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()

            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(prompts)}] id={p['id']} L={L} tokens ({elapsed:.0f}s, "
                  f"{elapsed/(i+1):.1f}s/gen)", flush=True)

        out_path = OUT_DIR / f"{group}_qwen3.npz"
        save_kwargs = {f"embeddings_L{l}": emb_by_layer[l] for l in ALL_LAYER_INDICES}
        save_kwargs["lengths"] = lengths
        np.savez_compressed(out_path, **save_kwargs)
        (OUT_DIR / f"{group}_responses.json").write_text(
            json.dumps(responses, indent=2, ensure_ascii=False))
        print(f"  Guardado: {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)", flush=True)

    print("\nQWEN3_EXTRACTION_DONE", flush=True)


if __name__ == "__main__":
    main()
