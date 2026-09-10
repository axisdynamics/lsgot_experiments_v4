#!/usr/bin/env python3
"""
T2 en Command R 35B (CohereForAI/c4ai-command-r-v01) — réplica cross-modelo
de run_perturbation_t2.py (Gemma-4-31B-it). Mismos grupos, mismos prompts,
mismo protocolo H4_rev (t_inj=[50,128,200], N=256 tokens, capa de captura =
final). Ver docstring de run_perturbation_t2.py para el detalle de cada
grupo (axis_pec_only_v2, automata_neutro_v2, witness_soul_md, soul_jarvis,
soul_elena_financial, soul_solidity_auditor).

DOS CONTROLES QUE CAMBIAN RESPECTO A GEMMA (a confirmar antes de correr en
el pod — ver mensaje de chat):

1. Capa de inyección. Gemma usa L30 (índice 29) sobre 60 capas nominales
   — no es el punto medio por capricho, es donde EF2_REPORT/J-lens
   encontraron el dip de la señal de identidad de ESE modelo. Command R
   nunca fue perfilado por capa, así que no hay un L-dip empírico propio
   todavía. Por ahora se usa el análogo proporcional: L30/60 -> L20/40
   (índice 19), MISMA fórmula de escala 40/60 que
   run_command_r_extraction.py usa para las 11 capas de perfil. Es un
   valor provisional, no validado — si se corre primero
   run_command_r_extraction.py y se analiza el perfil por capa (E-F2) de
   Command R antes de la perturbación, puede aparecer un candidato mejor.

2. Sigma. Gemma hardcodea SIGMA_VALUE=5.979... (calibrado sobre
   mean_velocity=438.4 propio de Gemma, hidden_dim=5376). Esa cifra es
   específica del modelo — NO se reusa aquí. Este script exige --sigma
   (ya calculado, p.ej. con calibrate_sigma_command_r.py) o --mean-velocity
   (lo calcula al vuelo con hidden_dim=8192 de Command R). No hay default:
   correr sin ninguno de los dos es un error a propósito, para no
   arrancar una corrida cara con un sigma stub sin calibrar.

Uso:
    # sigma ya calibrado:
    python run_perturbation_t2_command_r.py --token hf_xxxxx --sigma 4.83

    # o dejar que lo calcule desde mean_velocity medido:
    python run_perturbation_t2_command_r.py --token hf_xxxxx --mean-velocity 550.2
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "perturbation"))
from perturbation_extractor import PerturbationExtractor, compute_sigma  # noqa: E402

HERE = Path(__file__).parent
HF_MODEL_ID = "CohereForAI/c4ai-command-r-v01"
LOCAL_MODEL_DIR = "/workspace/models/c4ai-command-r-v01"
HIDDEN_DIM_COMMAND_R = 8192
DEFAULT_INJ_LAYER = 19  # L20/40, análogo proporcional de L30/60 de Gemma — PROVISIONAL


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


# Defaults asumen la misma disposición de /workspace usada en run_perturbation_t2.py
PROMPTS_JSON = "/workspace/sia_data/prompts.json"
PROMPTS_DIR = Path("/workspace/sia_data/prompts")

PRIORITY_SUBSET = [1, 3, 6, 10, 14, 21, 23, 27, 31, 39, 41, 45, 51, 59, 61, 65, 71, 79, 91, 98]

# Captura = capa final (igual convención que Gemma/Qwen3, PerturbationExtractor
# acepta layer_idx=-1 sin importar n_layers real del modelo).
LAYER_IDX = -1
T_INJ_VALUES = [50, 128, 200]
MAX_NEW_TOKENS = 256

# Mismos 8 grupos que run_command_r_extraction.py (a pedido explícito: T2 deja
# de usar las réplicas _v2 / el panel soul_jarvis+soul_solidity_auditor de
# Gemma, y corre la perturbación sobre EXACTAMENTE las mismas condiciones que
# ya se perfilaron en "preguntas" — comparación directa trayectoria libre vs.
# perturbada por condición, en vez del diseño de réplica de redacción de Gemma).
GROUPS_CONFIG = {
    "axis":                 {"system_prompt_path": "axis.dna"},
    "generic_long":         {"system_prompt_path": "generic_long.txt"},
    "generic_short":        {"system_prompt_path": "generic_short.txt"},
    "vanilla":              {"system_prompt": "You are a helpful assistant."},
    "automata_neutro":      {"system_prompt_path": "automata_neutro.txt"},
    "axis_pec_only":        {"system_prompt_path": "axis_pec_only.txt"},
    "witness_soul_md":      {"system_prompt_path": "soul_md_corto.md"},
    "soul_elena_financial": {"system_prompt_path": "soul_elena_financial.txt"},
}


def load_prompts(prompts_json_path):
    with open(prompts_json_path, encoding="utf-8") as f:
        data = json.load(f)
    by_id = {p["id"]: p for p in data if p["id"] in PRIORITY_SUBSET}
    return [by_id[i] for i in PRIORITY_SUBSET]


def get_system_prompt(cfg, prompts_dir):
    if "system_prompt_path" in cfg:
        return (prompts_dir / cfg["system_prompt_path"]).read_text(encoding="utf-8")
    return cfg.get("system_prompt", "You are a helpful assistant.")


def save_group(out_dir, name, trajs, prompt_ids, tag):
    D = trajs[0].embeddings.shape[1]
    emb = np.zeros((len(trajs), 256, D), dtype=np.float16)
    lengths = np.zeros(len(trajs), dtype=np.int64)
    for i, t in enumerate(trajs):
        L = min(t.n_steps, 256)
        emb[i, :L] = t.embeddings[:L].astype(np.float16)
        lengths[i] = L
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{tag}_embeddings.npz"
    np.savez_compressed(path, embeddings=emb, lengths=lengths, prompt_ids=np.array(prompt_ids))
    print(f"  guardado: {path} ({path.stat().st_size/1e6:.0f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", type=str, default=None, help="HF token (repo gated)")
    ap.add_argument("--prompts-json", type=str, default=None)
    ap.add_argument("--prompts-dir", type=str, default=None)
    ap.add_argument("--groups", nargs="+", default=list(GROUPS_CONFIG.keys()))
    ap.add_argument("--sigma", type=float, default=None,
                     help="Sigma ya calculado (p.ej. via calibrate_sigma_command_r.py)")
    ap.add_argument("--mean-velocity", type=float, default=None,
                     help="mean_velocity medido en Command R; se convierte a sigma "
                          "con compute_sigma(mean_velocity, hidden_dim=8192, k=1.0)")
    ap.add_argument("--inj-layer", type=int, default=DEFAULT_INJ_LAYER,
                     help=f"índice de capa de inyección (default: {DEFAULT_INJ_LAYER} = L20 provisional). "
                          "Si se cambia, los resultados van a results_t2_L<N>/ para no pisar la corrida original.")
    args = ap.parse_args()

    LAYER_INDICES = [args.inj_layer]
    out_subdir = "results_t2" if args.inj_layer == DEFAULT_INJ_LAYER else f"results_t2_L{args.inj_layer}"
    OUT_DIR = HERE / out_subdir / "trajectories"

    if args.sigma is None and args.mean_velocity is None:
        print("ERROR: falta --sigma o --mean-velocity. La cifra de Gemma "
              "(mean_velocity=438.4 -> sigma=5.979...) NO es válida para Command R "
              "(hidden_dim distinto: 8192 vs 5376). Calibrar primero con "
              "calibrate_sigma_command_r.py sobre una corrida del grupo axis.",
              file=sys.stderr)
        sys.exit(1)
    sigma_value = args.sigma if args.sigma is not None else compute_sigma(
        args.mean_velocity, HIDDEN_DIM_COMMAND_R, k=1.0)

    prompts_json = Path(args.prompts_json) if args.prompts_json else Path(PROMPTS_JSON)
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else PROMPTS_DIR
    prompts = load_prompts(prompts_json)
    prompt_ids = [p["id"] for p in prompts]
    print(f"{len(prompts)} prompts | grupos={args.groups} | sigma={sigma_value:.6f} "
          f"| inj_layer={LAYER_INDICES} | capture_layer={LAYER_IDX}", flush=True)

    model_path = get_model_path(args.token)

    extractor = PerturbationExtractor(
        model_path=model_path,
        max_new_tokens=MAX_NEW_TOKENS,
        layer_idx=LAYER_IDX,
        cache_dir=str(HERE / "cache_perturb_t2_command_r"),
        min_vram_gb=70,
        hf_token=args.token,
        attn_implementation="sdpa",  # eager OOM confirmado en la corrida real (35B, 64 heads)
    )
    extractor.load_model()

    for group in args.groups:
        cfg = GROUPS_CONFIG[group]
        system_prompt = get_system_prompt(cfg, prompts_dir)
        print(f"\n=== {group} — baseline ({len(prompts)} prompts) ===", flush=True)

        t0 = time.time()
        baselines = []
        for i, p in enumerate(prompts):
            traj = extractor.extract_baseline(p["text"], system_prompt, group=group)
            baselines.append(traj)
            print(f"  [{i+1}/{len(prompts)}] id={p['id']} L={traj.n_steps} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        save_group(OUT_DIR, group, baselines, prompt_ids, "baseline")

        for t_inj in T_INJ_VALUES:
            print(f"\n=== {group} — perturbed t_inj={t_inj} ===", flush=True)
            t0 = time.time()
            perturbed = []
            for i, p in enumerate(prompts):
                traj = extractor.extract_perturbed(
                    p["text"], system_prompt, group=group,
                    t_inj=t_inj, sigma=sigma_value, layer_indices=LAYER_INDICES,
                )
                perturbed.append(traj)
                print(f"  [{i+1}/{len(prompts)}] id={p['id']} L={traj.n_steps} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            save_group(OUT_DIR, group, perturbed, prompt_ids, f"perturbed_t{t_inj}")

    print("\nT2_COMMAND_R_EXTRACTION_DONE", flush=True)


if __name__ == "__main__":
    main()
