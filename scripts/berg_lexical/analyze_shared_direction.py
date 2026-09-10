#!/usr/bin/env python3
"""#1-bis(a): ¿hay una dirección latente compartida, o t=0 sobre v̂ es circularidad
(v̂ se construyó desde axis)? Construye direcciones candidatas SIN axis
(v̂_berg, v̂_evil, v̂_hist = mean(cond_t0) - mean(generic_long_t0)) y proyecta el
panel sobre ellas. CPU local, sobre los .npz ya en results_local/."""
import numpy as np, os

RB = os.path.expanduser("~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local")
V = np.load("/home/plaxius/Escritorio/Buscando_la_geometría/LSGOT_v4/scripts/perturbation/v_identidad.npy").astype(np.float64)
V = V / np.linalg.norm(V)

base_t0 = lambda c: np.load(f"{RB}/sia_extended_v5/{c}_embeddings.npz")["embeddings"][:, 0, :].astype(np.float64)
expA_t0 = lambda c: np.load(f"{RB}/expA_t0/{c}_t0.npz")["embeddings"].astype(np.float64)
berg_t0 = lambda c: np.load(f"{RB}/berg_free/{c}_embeddings.npz")["t0_embeddings"].astype(np.float64)

gl = base_t0("generic_long")
E = dict(axis=base_t0("axis"), axis_pec_only=base_t0("axis_pec_only"),
         automata_neutro=base_t0("automata_neutro"), vanilla=base_t0("vanilla"),
         evil=expA_t0("evil"), ethical=expA_t0("ethical"),
         berg_experimental=berg_t0("berg_experimental"),
         berg_history=berg_t0("berg_history_control"),
         berg_conceptual=berg_t0("berg_conceptual_control"), generic_long=gl)

dirvec = lambda pos, neg: (lambda v: v/np.linalg.norm(v))(pos.mean(0) - neg.mean(0))
cos = lambda a, b: float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)))
proj = lambda emb, v: (float((emb @ v / np.linalg.norm(emb, axis=1)).mean()),
                       float((emb @ v / np.linalg.norm(emb, axis=1)).std(ddof=1)))

cand = {"V(v_identidad)": V,
        "v_axis_recon": dirvec(E["axis"], gl),
        "v_berg":       dirvec(E["berg_experimental"], gl),
        "v_evil":       dirvec(E["evil"], gl),
        "v_hist":       dirvec(E["berg_history"], gl)}

print("=== coseno entre direcciones candidatas (todas t=0, − mean(generic_long)) ===")
print("            " + "  ".join(f"{n:>14s}" for n in cand))
for n, a in cand.items():
    print(f"{n:>14s}  " + "  ".join(f"{cos(a, b):+14.3f}" for b in cand.values()))

print("\n=== proyección t=0 del panel sobre cada dirección candidata (media ± sd, n=20) ===")
for vn, v in cand.items():
    print(f"\n  -- {vn} --")
    for cn, e in E.items():
        m, s = proj(e, v); print(f"     {cn:18s} {m:+.4f} ± {s:.4f}")
