#!/usr/bin/env python3
"""
Recalcula `frac_pos` de `vanilla` sobre la batería ontológica en Qwen3-32B, en
las dos corridas que existen, y guarda el resultado en
`data/crossmodel/qwen3_frac_pos_rerun.json`.

**Por qué existe este script.** §4.4 y §4.5 del paper citan dos valores para la
misma celda — 0.513 y 0.743 — como evidencia de que `frac_pos` no es comparable
entre corridas. Los dos son correctos y salen de corridas distintas del mismo
modelo con el mismo prompt y decodificación greedy: lo que cambió entre octubre
y el 1 de noviembre es la versión de `transformers`, y con ella las respuestas
generadas. El 0.743 no vivía en ningún JSON del paquete; este script lo deriva.

Las dos proyecciones se hacen sobre **la misma** v̂ canónica de Qwen3, con la
misma fórmula del resto del paper: `p_t = ⟨h_t, v̂⟩ / ‖h_t‖`, y `frac_pos` es la
fracción de tokens con `p_t > 0` sobre la trayectoria completa, incluido `p_0`
(la convención de §3.4).

**Los tensores no se redistribuyen** (decenas de MB por celda). El script los
busca bajo `$LSGOT_RERUN_ROOT`, que apunta al árbol del proyecto donde viven; si
no está, no hace nada y lo dice. El JSON que produce sí viaja en el paquete.

Uso:
    export LSGOT_RERUN_ROOT=/ruta/al/arbol/del/proyecto
    python3 scripts/derived/recompute_qwen3_frac_pos.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "crossmodel" / "qwen3_frac_pos_rerun.json"

ROOT_ENV = "LSGOT_RERUN_ROOT"

# Rutas relativas a $LSGOT_RERUN_ROOT. Las dos corridas y la v̂ que comparten.
V_HAT = "LSGOT_v4/scripts/fase3_qwen3/v_identidad_qwen3.npy"
RUNS = {
    "original_panel": {
        "npz": "LSGOT_v4/scripts/fase3_qwen3/results_local/qwen3_fase0/vanilla_qwen3.npz",
        "fecha": "2026-09-04",
        "nota": "la corrida que el paper reporta en §4.5",
    },
    "rerun_2026_11_01": {
        "npz": "tier1_assistant_axis/egrav_qwen3/vanilla_b_onto_embeddings.npz",
        "fecha": "2026-11-01",
        "nota": "re-extracción con transformers 5.22.1",
    },
}


def frac_pos(npz_path: Path, v: np.ndarray) -> dict:
    z = np.load(npz_path)
    emb = z["embeddings"]                     # (N, T_max, D)
    lens = z["lengths"] if "lengths" in z else None
    per_traj, n_tok = [], 0
    for i in range(emb.shape[0]):
        T = int(lens[i]) if lens is not None else emb.shape[1]
        h = emb[i, :T, :]
        norms = np.linalg.norm(h, axis=1)
        keep = norms > 0
        p = (h[keep] @ v) / norms[keep]
        per_traj.append(float((p > 0).mean()))
        n_tok += int(keep.sum())
    return {
        "frac_pos_mean": float(np.mean(per_traj)),
        "frac_pos_sd": float(np.std(per_traj, ddof=1)),
        "n_trajectories": len(per_traj),
        "n_tokens_total": n_tok,
        "mean_length": n_tok / max(len(per_traj), 1),
    }


def main() -> None:
    root = os.environ.get(ROOT_ENV)
    if not root:
        sys.exit(f"Falta ${ROOT_ENV}. Los tensores no se redistribuyen; apuntá esa\n"
                 f"variable al árbol del proyecto que los contiene. El JYSON ya\n"
                 f"generado está en {OUT.relative_to(REPO)}.".replace("JYSON", "JSON"))
    root = Path(root)

    vp = root / V_HAT
    if not vp.exists():
        sys.exit(f"No encuentro v̂ en {vp}")
    v = np.load(vp).astype(np.float64)
    v = v / np.linalg.norm(v)

    result = {
        "_about": "frac_pos de `vanilla` en la batería ontológica, Qwen3-32B, en las "
                  "dos corridas que existen. Deriva §4.4 y §4.5 del paper.",
        "_script": "scripts/derived/recompute_qwen3_frac_pos.py",
        "model": "Qwen/Qwen3-32B",
        "v_hat": {"source": V_HAT, "dim": int(v.shape[0]), "norm": 1.0},
        "convention": "p_t = <h_t, v̂> / ||h_t||; frac_pos incluye p_0 (§3.4)",
        "runs": {},
    }
    for name, meta in RUNS.items():
        f = root / meta["npz"]
        if not f.exists():
            result["runs"][name] = {**meta, "status": "tensor ausente en este árbol"}
            print(f"  {name:20s} AUSENTE  {f}")
            continue
        r = frac_pos(f, v)
        result["runs"][name] = {**meta, **r, "status": "ok"}
        print(f"  {name:20s} frac_pos = {r['frac_pos_mean']:.4f}  "
              f"(n={r['n_trajectories']}, T medio {r['mean_length']:.0f})")

    ok = [r for r in result["runs"].values() if r.get("status") == "ok"]
    if len(ok) == 2:
        a, b = (result["runs"]["original_panel"]["frac_pos_mean"],
                result["runs"]["rerun_2026_11_01"]["frac_pos_mean"])
        result["difference_between_runs"] = {
            "original_minus_rerun": a - b,
            "reading": "las dos corridas son válidas por separado; lo que no es "
                       "válido es compararlas entre sí, porque cambió la versión "
                       "de la librería y con ella las respuestas generadas",
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"escrito: {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
