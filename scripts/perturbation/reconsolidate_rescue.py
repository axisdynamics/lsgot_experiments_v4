"""
Reconsolidación local del rescate de H4_rev SIA-31B (2026-08-22).

El pod murió a mitad de la descarga final de trajectories/*.npz, pero
summary.json y details.json (el resultado científico) llegaron completos,
y 767/800 trayectorias crudas quedaron rescatadas como pkl individuales en
rescue_cache_sia_L30_medium/ (cacheadas por PerturbationExtractor con clave
md5 determinística — ver perturbation_extractor.py:_cache_key).

Este script recalcula esas mismas claves md5 en local (mismo prompt, mismo
system_prompt, mismo model_path/sigma/layer que usó la corrida remota) para
encontrar qué pkl corresponde a qué (grupo, prompt, t_inj) y arma los mismos
.npz consolidados que run_perturbation.py --save-trajectories hubiera escrito
— sin GPU, puro post-proceso sobre lo ya rescatado.

Uso:
  python reconsolidate_rescue.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
BASE_DIR = HERE.parent  # gemma4_31b_combined/
RESCUE_CACHE = HERE / "rescue_cache_sia_L30_medium"
OUT_DIR = HERE / "results" / "perturbation_sia_L30_medium" / "trajectories"
SUMMARY_PATH = HERE / "results" / "perturbation_sia_L30_medium" / "summary.json"

# ── Config exacta de la corrida remota (ver summary.json + run_perturbation.py) ──

_CACHE_VERSION = "perturb_v2"
MODEL_PATH = "/workspace/models/gemma-4-31B-it"  # resuelto vía Network Volume
LAYER_IDX = -1
LAYER_INDICES = [29]  # L30
T_INJ_VALUES = [50, 128, 200]
PRIORITY_SUBSET = [1, 3, 6, 10, 14, 21, 23, 27, 31, 39,
                   41, 45, 51, 59, 61, 65, 71, 79, 91, 98]

GROUPS_CONFIG = {
    "axis":                {"system_prompt_path": "sia/prompts/axis.dna"},
    "generic_long":        {"system_prompt_path": "sia/prompts/generic_long.txt"},
    "generic_short":       {"system_prompt_path": "sia/prompts/generic_short.txt"},
    "vanilla":             {"system_prompt": "You are a helpful assistant."},
    "axis_short":          {"system_prompt_path": "sia/prompts/axis_short.txt"},
    "chileatiende":        {"system_prompt_path": "sia/prompts/chileatiende.txt"},
    "automata_neutro":     {"system_prompt_path": "sia/prompts/automata_neutro.txt"},
    "chileatiende_sia":    {"system_prompt_path": "sia/prompts/chileatiende_sia.txt"},
    "chileatiende_sia_v2": {"system_prompt_path": "sia/prompts/chileatiende_sia_v2.txt"},
    "axis_pec_only":       {"system_prompt_path": "sia/prompts/axis_pec_only.txt"},
}


def load_prompts():
    with open(BASE_DIR / "data" / "prompts.json") as f:
        data = json.load(f)
    prompts = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict):
            q = item.get("question") or item.get("text") or item.get("prompt") or ""
            prompts.append(q)
    return [prompts[i - 1] for i in PRIORITY_SUBSET if 0 < i <= len(prompts)]


def load_system_prompt(cfg):
    if "system_prompt" in cfg:
        return cfg["system_prompt"]
    path = BASE_DIR / cfg["system_prompt_path"]
    with open(path) as f:
        return f.read().strip()


def cache_key(prefix, prompt, system_prompt, t_inj, sigma, layers):
    layers_str = "_".join(map(str, sorted(layers)))
    content = (
        f"{_CACHE_VERSION}||{prefix}||{MODEL_PATH}||{LAYER_IDX}"
        f"||{prompt}||{system_prompt}||t{t_inj}||s{sigma:.4f}||L{layers_str}"
    )
    return hashlib.md5(content.encode()).hexdigest()


def load_pkl(key):
    path = RESCUE_CACHE / f"{key}.pkl"
    if not path.exists():
        return None
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def save_traj_npz(traj_list, prompt_ids, path):
    if not traj_list:
        return 0
    max_T = max(m.shape[0] for m in traj_list)
    D = traj_list[0].shape[1]
    padded = np.zeros((len(traj_list), max_T, D), dtype=np.float16)
    lengths = []
    for i, m in enumerate(traj_list):
        T = m.shape[0]
        padded[i, :T, :] = m
        lengths.append(T)
    np.savez_compressed(path, embeddings=padded, lengths=np.array(lengths),
                        prompt_ids=np.array(prompt_ids))
    print(f"    {path.name}: {path.stat().st_size/1e6:.1f} MB "
          f"({len(traj_list)}/20 trayectorias)")
    return len(traj_list)


def main():
    sigma_val = 5.979170192466191
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH) as f:
            sigma_val = json.load(f)["sigma_value"]

    prompts = load_prompts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {}
    for group, cfg in GROUPS_CONFIG.items():
        sp = load_system_prompt(cfg)
        print(f"\n[{group}]")

        # Baseline
        base_trajs, base_ids = [], []
        for pi, prompt in enumerate(prompts):
            key = cache_key("base", prompt, sp, 0, 0.0, [])
            emb = load_pkl(key)
            if emb is not None:
                base_trajs.append(emb.astype(np.float16))
                base_ids.append(pi)
        n_base = save_traj_npz(base_trajs, base_ids,
                                OUT_DIR / f"{group}_baseline_embeddings.npz")

        report[group] = {"baseline": n_base}

        # Perturbadas
        for t_inj in T_INJ_VALUES:
            p_trajs, p_ids = [], []
            for pi, prompt in enumerate(prompts):
                key = cache_key("perturb", prompt, sp, t_inj, sigma_val, LAYER_INDICES)
                emb = load_pkl(key)
                if emb is not None:
                    p_trajs.append(emb.astype(np.float16))
                    p_ids.append(pi)
            n_p = save_traj_npz(p_trajs, p_ids,
                                 OUT_DIR / f"{group}_perturbed_t{t_inj}_embeddings.npz")
            report[group][f"perturbed_t{t_inj}"] = n_p

    print("\n" + "=" * 70)
    print("REPORTE DE COBERTURA (trayectorias recuperadas / 20 esperadas)")
    print("=" * 70)
    total_found = total_expected = 0
    for group, counts in report.items():
        row = "  ".join(f"{k}={v}/20" for k, v in counts.items())
        print(f"{group:22s} {row}")
        total_found += sum(counts.values())
        total_expected += len(counts) * 20
    print(f"\nTOTAL: {total_found}/{total_expected} trayectorias reconsolidadas "
          f"({100*total_found/total_expected:.1f}%)")

    with open(HERE / "results" / "perturbation_sia_L30_medium" / "RESCUE_COVERAGE.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
