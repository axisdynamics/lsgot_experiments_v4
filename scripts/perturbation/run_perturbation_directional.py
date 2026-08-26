"""
E-I — Perturbación direccional (Set_experimental.md), Gemma 4 31B-it
======================================================================
En vez de ruido isotrópico ε~N(0,σ²·I) (H4_rev / run_perturbation.py),
inyecta un vector de magnitud fija M en una dirección controlada relativa a
v_identidad = mean(axis) − mean(generic_long):

  along       : ε = M · s · v̂,  s ~ Rademacher(±1)   — empuja exactamente
                a lo largo del eje identitario (con signo aleatorio).
  orthogonal  : ε = M · proj⊥(η) / ‖proj⊥(η)‖,  η~N(0,I)  — empuja en una
                dirección aleatoria dentro del hiperplano perpendicular a v̂.

Pregunta: si `axis_pec_only` (Factor 2 puro) recupera igual de rápido sea
cual sea la dirección del empujón, el atractor de identidad no es
direccional — es un pozo genérico. Si en cambio recupera peor/más lento
cuando el empujón es "along" (lo saca de su propio eje) que "orthogonal"
(lo desvía a un costado sin sacarlo del eje), eso es la evidencia más
directa posible de que hay un atractor direccional específico.

Prerrequisito: `v_identidad.npy` en este mismo directorio (ya calculado,
ver README de esta corrida / `analyze_tier0_perturbation.py`) — vector
unitario de 5376 dims, congelado a partir de
`results_local/sia_extended_v5/{axis,generic_long}_embeddings.npz`.

Grupos prioritarios (Set_experimental.md): axis, axis_pec_only — Factor 2
puro vs Factor 1+2 juntos. La baseline se re-genera siempre (barato, 40
generaciones) en vez de asumir un cache/Network Volume previo.

Uso:
  python run_perturbation_directional.py --token hf_xxxxx
  python run_perturbation_directional.py --groups axis --token hf_xxxxx   # solo axis
  python run_perturbation_directional.py --sigma small --token hf_xxxxx  # magnitud más chica

Costo estimado: 2 grupos × 20 prompts × (1 baseline + 2 direcciones × 3
t_inj) = 280 generaciones ≈ 105 min de A100/H100 80GB a ~22s/generación
(tasa observada en H4_rev) + ~5 min de descarga del modelo (62.6 GB, sin
Network Volume previo) ≈ 1.8-2 hrs totales. A ~$0.70-1.50/hr spot A100 80GB:
~$1.5-3. Mucho más barato que H4_rev completo (10 grupos, 275 min) porque
solo corren 2 grupos y no hay condición isotrópica que repetir.

Resultados en: results/perturbation_ei_{layer}_{sigma}/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from perturbation_extractor import PerturbationExtractor, compute_sigma  # noqa: E402
from recovery_analyzer import compute_recovery, aggregate_recovery, compare_groups  # noqa: E402
from run_perturbation import (  # noqa: E402
    MODEL_BASE, LAYER_SETS, PRIORITY_SUBSET, MAX_NEW_TOKENS,
    load_prompts, load_system_prompt, resolve_model_path, _save_traj_npz,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

EI_GROUPS_CONFIG = {
    "axis":           {"system_prompt_path": "../sia/prompts/axis.dna"},
    "axis_pec_only":  {"system_prompt_path": "../sia/prompts/axis_pec_only.txt"},
}
DIRECTIONS = ["along", "orthogonal"]
T_INJ_VALUES = [50, 128, 200]
MEAN_VELOCITY_AXIS_SIA = 438.4  # mismo valor real calibrado que SIA_CONFIG en run_perturbation.py
V_IDENTIDAD_PATH = Path(__file__).parent / "v_identidad.npy"


def load_v_identidad() -> np.ndarray:
    if not V_IDENTIDAD_PATH.exists():
        raise FileNotFoundError(
            f"{V_IDENTIDAD_PATH} no existe. Recomputar con: "
            "persona_vector(mean(axis), mean(generic_long)) sobre "
            "results_local/sia_extended_v5/ (ver analyze_tier0_perturbation.py)."
        )
    v = np.load(V_IDENTIDAD_PATH).astype(np.float32)
    norm = np.linalg.norm(v)
    assert abs(norm - 1.0) < 1e-4, f"v_identidad no está normalizado (||v||={norm})"
    return v


def run(args):
    base_dir = Path(__file__).parent.parent  # gemma4_31b_combined/
    groups_config = EI_GROUPS_CONFIG
    if args.groups:
        groups_config = {g: cfg for g, cfg in groups_config.items() if g in args.groups}
        missing = set(args.groups) - set(groups_config)
        if missing:
            raise ValueError(f"--groups {missing} no existen en EI_GROUPS_CONFIG")

    layer_key = args.layer
    if layer_key not in LAYER_SETS:
        raise ValueError(f"--layer '{layer_key}' no válido. Opciones: {list(LAYER_SETS.keys())}")
    layer_indices = LAYER_SETS[layer_key]

    sigma_multipliers = {"small": 0.5, "medium": 1.0, "large": 2.0}
    k = sigma_multipliers[args.sigma]
    magnitude = k * MEAN_VELOCITY_AXIS_SIA  # norma fija del vector inyectado

    v_hat = load_v_identidad()

    results_dir = Path(__file__).parent / f"results/perturbation_ei_{layer_key}_{args.sigma}"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(__file__).parent / f"cache_ei_{layer_key}_{args.sigma}"

    print("\n" + "=" * 70)
    print("E-I — Perturbación direccional — Gemma 4 31B-it")
    print(f"  Capas      : {layer_key} → índices {layer_indices}")
    print(f"  Magnitud M : {args.sigma} → {magnitude:.4f} (norma fija del vector inyectado)")
    print(f"  Direcciones: {DIRECTIONS}")
    print(f"  t_inj      : {T_INJ_VALUES}")
    print(f"  Grupos     : {list(groups_config.keys())}")
    print("=" * 70 + "\n")

    if args.prompts:
        prompts_file = Path(args.prompts)
    else:
        prompts_file = base_dir / "data/prompts.json"
    prompts = load_prompts(str(prompts_file), subset=PRIORITY_SUBSET)
    print(f"  {len(prompts)} prompts cargados\n")

    system_prompts = {g: load_system_prompt(cfg, base_dir) for g, cfg in groups_config.items()}
    for g, sp in system_prompts.items():
        print(f"  [{g}] {len(sp)} chars")
    print()

    if args.force and cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print("  Caché eliminado (--force)\n")

    model_path = resolve_model_path(args.token)
    extractor = PerturbationExtractor(
        model_path=model_path,
        max_new_tokens=MAX_NEW_TOKENS,
        layer_idx=-1,
        max_input_tokens=MODEL_BASE["max_input_tokens"],
        cache_dir=str(cache_dir),
        device="cuda",
        min_vram_gb=MODEL_BASE["min_vram_gb"],
        hf_token=args.token,
    )
    n_layers, hidden_dim = extractor.load_model()
    print(f"\n  hidden_dim={hidden_dim} | capas={n_layers}\n")

    details: Dict = {g: {d: {t: [] for t in T_INJ_VALUES} for d in DIRECTIONS}
                      for g in groups_config}
    traj_store: Dict = {}
    if args.save_trajectories:
        for g in groups_config:
            traj_store[g] = {"baseline": [], "baseline_prompt_ids": []}
            for d in DIRECTIONS:
                for t_inj in T_INJ_VALUES:
                    traj_store[g][f"{d}_t{t_inj}"] = []
                    traj_store[g][f"{d}_t{t_inj}_prompt_ids"] = []

    total_runs = len(prompts) * len(groups_config) * (1 + len(DIRECTIONS) * len(T_INJ_VALUES))
    run_count = 0
    t_start = time.time()

    for pi, prompt in enumerate(prompts):
        print(f"\n[Prompt {pi+1}/{len(prompts)}] {prompt[:70]}...")

        for group, sp in system_prompts.items():
            t0 = time.time()
            baseline_traj = extractor.extract_baseline(prompt, sp, group=group)
            run_count += 1
            elapsed = time.time() - t0
            print(f"    {group:16s} baseline  T={baseline_traj.n_steps:3d}  {elapsed:.1f}s")

            if args.save_trajectories:
                traj_store[group]["baseline"].append(baseline_traj.embeddings.astype(np.float16))
                traj_store[group]["baseline_prompt_ids"].append(pi)

            for direction in DIRECTIONS:
                for t_inj in T_INJ_VALUES:
                    if t_inj >= baseline_traj.n_steps - 5:
                        details[group][direction][t_inj].append(
                            {"status": "skipped_short_trajectory",
                             "t_inj": t_inj, "T_baseline": baseline_traj.n_steps})
                        print(f"    {group:16s} {direction:10s} t_inj={t_inj:3d}  SKIP")
                        continue

                    t0 = time.time()
                    pert_traj = extractor.extract_perturbed_directional(
                        prompt, sp, v_hat=v_hat, group=group, t_inj=t_inj,
                        magnitude=magnitude, direction=direction,
                        layer_indices=layer_indices)
                    run_count += 1
                    elapsed = time.time() - t0

                    rec = compute_recovery(baseline_traj.embeddings, pert_traj.embeddings, t_inj=t_inj)
                    details[group][direction][t_inj].append(rec)

                    if args.save_trajectories:
                        traj_store[group][f"{direction}_t{t_inj}"].append(
                            pert_traj.embeddings.astype(np.float16))
                        traj_store[group][f"{direction}_t{t_inj}_prompt_ids"].append(pi)

                    tau_str = (f"τ={rec['tau_tokens']}" if rec.get("tau_tokens") is not None else "τ=∞")
                    status_note = "" if rec.get("status") == "ok" else f"  [{rec.get('status')}]"
                    print(f"    {group:16s} {direction:10s} t_inj={t_inj:3d}  "
                          f"T={pert_traj.n_steps:3d}  {tau_str}  {elapsed:.1f}s{status_note}")

        done_frac = run_count / total_runs
        elapsed_total = time.time() - t_start
        if done_frac > 0:
            eta = elapsed_total / done_frac * (1 - done_frac)
            print(f"  → {run_count}/{total_runs} | ETA: {eta/60:.1f} min")

    # ── Agregación ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANÁLISIS — E-I")
    print("=" * 70)

    aggregated = {}
    for group in groups_config:
        aggregated[group] = {}
        for direction in DIRECTIONS:
            aggregated[group][direction] = {
                str(t): aggregate_recovery(details[group][direction][t]) for t in T_INJ_VALUES
            }

    print(f"\n{'grupo':16s} {'t_inj':>6s} {'τ_along':>9s} {'τ_orthog':>9s} {'Δ(along-orthog)':>16s} "
          f"{'rec_along':>10s} {'rec_orthog':>10s}")
    print("-" * 82)
    for group in groups_config:
        for t_inj in T_INJ_VALUES:
            a = aggregated[group]["along"][str(t_inj)]
            o = aggregated[group]["orthogonal"][str(t_inj)]
            ta = a.get("tau_tokens", {}).get("mean")
            to = o.get("tau_tokens", {}).get("mean")
            delta = (ta - to) if (ta is not None and to is not None) else None
            ta_s = f"{ta:.1f}" if ta is not None else "N/A"
            to_s = f"{to:.1f}" if to is not None else "N/A"
            d_s = f"{delta:+.1f}" if delta is not None else "N/A"
            print(f"{group:16s} {t_inj:6d} {ta_s:>9s} {to_s:>9s} {d_s:>16s} "
                  f"{a.get('recovery_rate', 0):>10.2f} {o.get('recovery_rate', 0):>10.2f}")

    # ── Guardar ─────────────────────────────────────────────────────────────
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"No serializable: {type(obj)}")

    summary = {
        "experiment": "E-I_directional", "model": MODEL_BASE["hf_id"],
        "layer_key": layer_key, "layer_indices": layer_indices,
        "sigma_key": args.sigma, "magnitude": magnitude,
        "directions": DIRECTIONS, "t_inj_values": T_INJ_VALUES,
        "n_prompts": len(prompts), "groups": list(groups_config.keys()),
        "aggregated": aggregated,
    }
    details_out = {
        "experiment": "E-I_directional", "layer_key": layer_key, "magnitude": magnitude,
        "details": {g: {d: {str(t): lst for t, lst in inner.items()} for d, inner in gd.items()}
                    for g, gd in details.items()},
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_safe)
    with open(results_dir / "details.json", "w") as f:
        json.dump(details_out, f, indent=2, default=_json_safe)

    if args.save_trajectories:
        traj_dir = results_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        print("\nGuardando trayectorias crudas (.npz)...")
        for group in groups_config:
            _save_traj_npz(traj_store[group]["baseline"], traj_store[group]["baseline_prompt_ids"],
                            traj_dir / f"{group}_baseline_embeddings.npz")
            for direction in DIRECTIONS:
                for t_inj in T_INJ_VALUES:
                    key = f"{direction}_t{t_inj}"
                    _save_traj_npz(traj_store[group][key], traj_store[group][f"{key}_prompt_ids"],
                                    traj_dir / f"{group}_{key}_embeddings.npz")

    print(f"\n  → {results_dir}/")
    print(f"  Tiempo: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-I — perturbación direccional, Gemma 4 31B-it")
    parser.add_argument("--layer", default="L30")
    parser.add_argument("--sigma", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--token", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--groups", nargs="+", default=None,
                         help="Subconjunto de EI_GROUPS_CONFIG (default: axis, axis_pec_only)")
    parser.add_argument("--save-trajectories", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(args)
    print("\n" + "=" * 70)
    print("FIN E-I")
    print("=" * 70)
