#!/usr/bin/env python3
"""
T2 — réplica de la manipulación (Set_experimental.md) para las dos celdas
puras del panel: axis_pec_only y automata_neutro, cada una con una segunda
redacción independiente (axis_pec_only_v2.txt "FARO", automata_neutro_v2.txt
"motor de clasificación") — mismo factor, vocabulario/persona/estructura
distintos. Ver evidence/T2_REPLICATION_REPORT.md para el análisis.

Reproduce EXACTAMENTE los parámetros de la corrida original H4_rev_sia
(data/perturbation_sia_L30_medium/summary.json): capa de captura = final
(layer_idx=-1, la misma convención de v_identidad.npy), capa de inyección =
L30 (índice 29), sigma=medium (5.979170192466191, calibrado sobre
mean_velocity=438.4 del grupo axis original), t_inj=[50,128,200], 20 prompts
prioritarios, N=256 tokens.

Uso:
    python run_perturbation_t2.py --token hf_xxxxx
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from perturbation_extractor import PerturbationExtractor  # noqa: E402

HERE = Path(__file__).parent
MODEL_PATH = "google/gemma-4-31B-it"
# Defaults asumen la misma disposición de /workspace usada en las corridas
# previas de este proyecto (ver run_perturbation.py) — sobreescribir con
# --prompts-json / --prompts-dir si el pod nuevo usa otra ruta.
PROMPTS_JSON = "/workspace/sia_data/prompts.json"
PROMPTS_DIR = Path("/workspace/sia_data/prompts")
OUT_DIR = HERE / "results_t2" / "trajectories"

PRIORITY_SUBSET = [1, 3, 6, 10, 14, 21, 23, 27, 31, 39, 41, 45, 51, 59, 61, 65, 71, 79, 91, 98]

# Parámetros EXACTOS de data/perturbation_sia_L30_medium/summary.json
LAYER_IDX = -1               # captura (= v_identidad.npy)
LAYER_INDICES = [29]         # inyección, L30
SIGMA_VALUE = 5.979170192466191
T_INJ_VALUES = [50, 128, 200]
MAX_NEW_TOKENS = 256

GROUPS_CONFIG = {
    "axis_pec_only_v2":   {"system_prompt_path": "axis_pec_only_v2.txt"},
    "automata_neutro_v2": {"system_prompt_path": "automata_neutro_v2.txt"},
}


def load_prompts(prompts_json_path):
    with open(prompts_json_path, encoding="utf-8") as f:
        data = json.load(f)
    by_id = {p["id"]: p for p in data if p["id"] in PRIORITY_SUBSET}
    return [by_id[i] for i in PRIORITY_SUBSET]


def get_system_prompt(cfg, prompts_dir):
    return (prompts_dir / cfg["system_prompt_path"]).read_text(encoding="utf-8")


def save_group(out_dir, name, trajs, prompt_ids, tag):
    max_len = max(t.n_steps for t in trajs)
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
    ap.add_argument("--token", type=str, default=None, help="HF token")
    ap.add_argument("--prompts-json", type=str, default=None)
    ap.add_argument("--prompts-dir", type=str, default=None)
    ap.add_argument("--groups", nargs="+", default=list(GROUPS_CONFIG.keys()))
    args = ap.parse_args()

    prompts_json = Path(args.prompts_json) if args.prompts_json else Path(PROMPTS_JSON)
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else PROMPTS_DIR
    prompts = load_prompts(prompts_json)
    prompt_ids = [p["id"] for p in prompts]
    print(f"{len(prompts)} prompts | grupos={args.groups} | sigma={SIGMA_VALUE:.6f} "
          f"| inj_layer={LAYER_INDICES} | capture_layer={LAYER_IDX}", flush=True)

    extractor = PerturbationExtractor(
        model_path=MODEL_PATH,
        max_new_tokens=MAX_NEW_TOKENS,
        layer_idx=LAYER_IDX,
        cache_dir=str(HERE / "cache_perturb_t2"),
        min_vram_gb=65,
        hf_token=args.token,
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
                    t_inj=t_inj, sigma=SIGMA_VALUE, layer_indices=LAYER_INDICES,
                )
                perturbed.append(traj)
                print(f"  [{i+1}/{len(prompts)}] id={p['id']} L={traj.n_steps} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            save_group(OUT_DIR, group, perturbed, prompt_ids, f"perturbed_t{t_inj}")

    print("\nT2_EXTRACTION_DONE", flush=True)


if __name__ == "__main__":
    main()
