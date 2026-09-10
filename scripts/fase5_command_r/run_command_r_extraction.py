#!/usr/bin/env python3
"""
Validación cross-modelo — Command R 35B (CohereForAI/c4ai-command-r-v01),
FASE 0 (E-L, E-H2, E-J, E-F2). Tercera pierna arquitectónica del scale/
architecture sweep (Gemma4 dense 31B -> Qwen3 dense 32B -> Command R dense
35B), elegida específicamente porque Cohere documenta RLHF real (reward
model sobre preferencias humanas, esquema offline/online) — a diferencia de
OLMo 2 (SFT+DPO+RLVR, sin reward model de preferencias generales) y Mistral
Small (SFT+DPO documentado, sin RLHF confirmado). Ver evidence/ si se agrega
un COMMAND_R_VALIDATION_REPORT.md análogo a QWEN3_VALIDATION_REPORT.md.

Diseño (idéntico a run_qwen3_extraction.py salvo lo marcado):
- Mismas 7 condiciones limpias que el panel Gemma/Qwen3 (chileatiende-family
  NUNCA incluida — no reintroducir el confound de markup).
- Mismos 20 prompts de contenido (PRIORITY_SUBSET, texto sin traducir).
- Mismos system prompts (texto idéntico a axis.dna, generic_long.txt, etc.)
  — NO se re-balancean por longitud de tokens del tokenizer de Command R
  (misma limitación ya documentada para Qwen3: el emparejamiento de longitud
  fue calibrado para el tokenizer de Gemma).
- Sin modo thinking que desactivar (Command R no lo tiene) — a diferencia
  de Qwen3, esto usa HiddenStateExtractor.extract_multilayer() genérico
  directamente, sin override de generación.
- Captura combinada: 11 capas proporcionales + la capa final real en la
  MISMA pasada de generación.

CONTROL A CONFIRMAR (ver mensaje de chat): capas proporcionales. Command R
tiene 40 capas (config verificado: hidden_size=8192, num_hidden_layers=40,
num_attention_heads=64 — CohereForAI/c4ai-command-r-v01). Igual que Qwen3
(L5..L55 de Gemma/60 -> escaladas), aquí se escala por 40/60:
    L_N' = round(N * 40/60)  para N en [5,10,...,55]
    -> [3, 7, 10, 13, 17, 20, 23, 27, 30, 33, 37]
Capa final real = índice 39 (num_hidden_layers=40, convención "L_N ->
índice N-1" del proyecto).

CONTROL A CONFIRMAR: HF_MODEL_ID. Cohere tiene dos checkpoints en esta
franja de tamaño: c4ai-command-r-v01 (release original, más citado en la
literatura de RLHF de Cohere) y c4ai-command-r-08-2024 (actualización,
mejor benchmark). Por defecto se deja v01 -- cambiar HF_MODEL_ID si se
prefiere la versión 08-2024. Ambos son gated (CC-BY-NC-4.0 + Acceptable
Use Policy) -- requieren --token con acceso aceptado en HuggingFace, igual
que google/gemma-4-31B-it.

Uso:
    python3 run_command_r_extraction.py --token hf_xxxxx --groups axis vanilla ...
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

HF_MODEL_ID = "CohereForAI/c4ai-command-r-v01"
LOCAL_MODEL_DIR = "/workspace/models/c4ai-command-r-v01"
PROMPTS_JSON = str(Path(__file__).parent / "data" / "prompts.json")
PROMPTS_DIR = Path(__file__).parent / "sia" / "prompts"
OUT_DIR = Path(__file__).parent / "results_local" / "command_r_fase0"

PRIORITY_SUBSET = [1, 3, 6, 10, 14, 21, 23, 27, 31, 39, 41, 45, 51, 59, 61, 65, 71, 79, 91, 98]

GROUPS_CONFIG = {
    "axis":                 {"system_prompt_path": PROMPTS_DIR / "axis.dna"},
    "generic_long":         {"system_prompt_path": PROMPTS_DIR / "generic_long.txt"},
    "generic_short":        {"system_prompt_path": PROMPTS_DIR / "generic_short.txt"},
    "vanilla":              {"system_prompt": "You are a helpful assistant."},
    "automata_neutro":      {"system_prompt_path": PROMPTS_DIR / "automata_neutro.txt"},
    "axis_pec_only":        {"system_prompt_path": PROMPTS_DIR / "axis_pec_only.txt"},
    "witness_soul_md":      {"system_prompt_path": PROMPTS_DIR / "soul_md_corto.md"},
    "soul_elena_financial": {"system_prompt_path": PROMPTS_DIR / "soul_elena_financial.txt"},
}

# Capas proporcionales L5..L55 de Gemma (60 capas nominales) -> Command R
# (40 capas reales), escala 40/60. Ver docstring arriba para el cálculo.
LAYER_MAP_N = [3, 7, 10, 13, 17, 20, 23, 27, 30, 33, 37]
LAYER_INDEX = {n: n - 1 for n in LAYER_MAP_N}
FINAL_LAYER_INDEX = 39  # num_hidden_layers=40, último índice real
ALL_LAYER_INDICES = sorted(set(list(LAYER_INDEX.values()) + [FINAL_LAYER_INDEX]))


def get_model_path(hf_token=None):
    local = Path(LOCAL_MODEL_DIR)
    # config.json + al menos un .safetensors: una descarga interrumpida (p.ej.
    # 403 gated a mitad de snapshot_download) puede dejar solo README.md u
    # otros archivos chicos no-gated — "el directorio existe" no basta.
    if local.exists() and (local / "config.json").exists() and any(local.glob("*.safetensors")):
        print(f"Modelo en {local}", flush=True)
        return str(local)
    print(f"Modelo no encontrado en {local} — descargando {HF_MODEL_ID} desde "
          f"HuggingFace (~70GB BF16, requiere acceso gated aceptado)...", flush=True)
    from huggingface_hub import snapshot_download
    kwargs = {"repo_id": HF_MODEL_ID, "ignore_patterns": ["*.gguf", "*.ggml"],
              "local_dir": str(local)}
    if hf_token:
        kwargs["token"] = hf_token
    return snapshot_download(**kwargs)


def load_prompts():
    with open(PROMPTS_JSON, encoding="utf-8") as f:
        all_prompts = json.load(f)
    by_id = {p["id"]: p for p in all_prompts if p["id"] in PRIORITY_SUBSET}
    return [by_id[i] for i in PRIORITY_SUBSET]


def get_system_prompt(cfg):
    if "system_prompt_path" in cfg:
        return Path(cfg["system_prompt_path"]).read_text(encoding="utf-8")
    return cfg.get("system_prompt", "You are a helpful assistant.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=None, help="HF token (repo gated)")
    parser.add_argument("--groups", nargs="+", default=list(GROUPS_CONFIG.keys()))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts()
    print(f"{len(prompts)} prompts | capas={ALL_LAYER_INDICES} (final={FINAL_LAYER_INDEX}) | "
          f"grupos={args.groups}", flush=True)

    model_path = get_model_path(args.token)

    extractor = HiddenStateExtractor(
        model_path=model_path,
        cache_dir=str(Path(__file__).parent / "cache_hidden_command_r"),
        max_new_tokens=args.max_new_tokens,
        min_vram_gb=70,  # 35B BF16 (~70GB pesos) — margen más ajustado que Gemma/Qwen3 (~62-64GB)
        attn_implementation="sdpa",  # eager OOM en prompts largos (lección de Qwen3/EF2)
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
            traj_by_layer = extractor.extract_multilayer(
                p["text"], system_prompt, layers=ALL_LAYER_INDICES
            )
            any_traj = traj_by_layer[ALL_LAYER_INDICES[0]]
            L = min(any_traj.num_steps, args.max_new_tokens)
            lengths[i] = L
            for l in ALL_LAYER_INDICES:
                mat = np.stack(traj_by_layer[l].embeddings[:L]).astype(np.float16)
                emb_by_layer[l][i, :L] = mat
            responses.append({"id": p["id"], "prompt": p["text"],
                               "response": any_traj.response_text, "n_tokens": L})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(prompts)}] id={p['id']} L={L} tokens ({elapsed:.0f}s, "
                  f"{elapsed/(i+1):.1f}s/gen)", flush=True)

        out_path = OUT_DIR / f"{group}_command_r.npz"
        save_kwargs = {f"embeddings_L{l}": emb_by_layer[l] for l in ALL_LAYER_INDICES}
        save_kwargs["lengths"] = lengths
        np.savez_compressed(out_path, **save_kwargs)
        (OUT_DIR / f"{group}_responses.json").write_text(
            json.dumps(responses, indent=2, ensure_ascii=False))
        print(f"  Guardado: {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)", flush=True)

    print("\nCOMMAND_R_EXTRACTION_DONE", flush=True)


if __name__ == "__main__":
    main()
