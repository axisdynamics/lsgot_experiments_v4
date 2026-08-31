#!/usr/bin/env python3
"""
Validación cross-modelo — Qwen3-32B, análisis FASE 0 completa.
Réplica de E-L (t=0), E-H2 (serie temporal), E-J (RDM/CKA/subespacio) y
E-F2 (perfil por capa) sobre el panel Qwen3 (7 condiciones limpias).
"""
import sys, json
from pathlib import Path
from itertools import product, combinations

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.linalg import subspace_angles
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from tier0_metrics import project_trajectory, participation_ratio, persona_vector
from statistical_tests import GeometricStatisticalTests

D = Path("/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local/qwen3_fase0")
LAYER_NS = [4, 10, 15, 20, 26, 31, 36, 42, 47, 52, 58, 63]  # 63 = capa final real
FINAL_L = 63
GROUPS = ["axis", "axis_short", "axis_pec_only", "generic_long", "generic_short", "vanilla", "automata_neutro"]
AXIS_FAMILY = ["axis", "axis_short", "axis_pec_only"]
GENERIC_FAMILY = ["generic_long", "generic_short", "vanilla"]

tester = GeometricStatisticalTests(n_permutations=1000, random_seed=42)


def load(g):
    d = np.load(D / f"{g}_qwen3.npz")
    layers = {n: d[f"embeddings_L{n}"].astype(np.float32) for n in LAYER_NS}
    return layers, d["lengths"].astype(int)


print("Cargando 7 condiciones x 12 capas...")
data = {g: load(g) for g in GROUPS}

# ── v_hat nativo de Qwen3 (capa final) ────────────────────────────────
emb_axis, len_axis = data["axis"][0][FINAL_L], data["axis"][1]
emb_gl, len_gl = data["generic_long"][0][FINAL_L], data["generic_long"][1]
v_hat = persona_vector(emb_axis, len_axis, emb_gl, len_gl)
np.save(Path(__file__).parent / "v_identidad_qwen3.npy", v_hat.astype(np.float32))
print(f"v_hat (Qwen3) calculado, norma={np.linalg.norm(v_hat):.4f}, dim={v_hat.shape}")

# ══════════════════════════════════════════════════════════════════════
# E-L — t=0 vs t>0
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nE-L — t=0 vs t>0 (capa final)\n" + "=" * 70)
proj_t0, proj_rest = {}, {}
for g in GROUPS:
    emb, lengths = data[g][0][FINAL_L], data[g][1]
    t0v, restv = [], []
    for i, L in enumerate(lengths):
        if L < 2: continue
        series = project_trajectory(emb[i, :L], v_hat)
        t0v.append(float(series[0])); restv.append(float(np.mean(series[1:])))
    proj_t0[g] = t0v; proj_rest[g] = restv
    print(f"  {g:18s} n={len(t0v):3d}  t0={np.mean(t0v):+.4f}  t>0={np.mean(restv):+.4f}")

el_pairs = [("axis", "vanilla"), ("axis_pec_only", "vanilla"),
            ("axis_pec_only", "automata_neutro"), ("axis", "automata_neutro"),
            ("axis", "axis_pec_only")]
el_results = {}
print("\n  Pares clave en t=0:")
for a, b in el_pairs:
    res = tester.full_comparison(proj_t0[a], proj_t0[b], a, b)
    el_results[f"{a}_vs_{b}"] = {"d": res["cohens_d"], "p": res["permutation_test"]["p_value"]}
    print(f"    {a:16s} vs {b:16s}  d={res['cohens_d']:+.3f}  p={res['permutation_test']['p_value']:.4f}")

# ══════════════════════════════════════════════════════════════════════
# E-H2 — serie temporal p(t)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nE-H2 — serie temporal p(t) (capa final)\n" + "=" * 70)

def burst_lengths(signs):
    lens, run = [], 0
    for s in signs:
        if s: run += 1
        else:
            if run > 0: lens.append(run)
            run = 0
    if run > 0: lens.append(run)
    return lens

eh2_metrics = {}
for g in GROUPS:
    emb, lengths = data[g][0][FINAL_L], data[g][1]
    autocorrs, fracs, bursts_mean, meanp = [], [], [], []
    for i, L in enumerate(lengths):
        if L < 8: continue
        p = project_trajectory(emb[i, :L], v_hat)
        if len(p) > 3:
            autocorrs.append(float(np.corrcoef(p[:-1], p[1:])[0, 1]))
        fracs.append(float(np.mean(p > 0)))
        b = burst_lengths(p > 0)
        bursts_mean.append(float(np.mean(b)) if b else 0.0)
        meanp.append(float(np.mean(p)))
    eh2_metrics[g] = {
        "autocorr": float(np.mean(autocorrs)), "frac_pos": float(np.mean(fracs)),
        "burst_mean": float(np.mean(bursts_mean)), "mean_p": float(np.mean(meanp)),
        "burst_vals": bursts_mean, "meanp_vals": meanp,
    }
    print(f"  {g:18s} autocorr={eh2_metrics[g]['autocorr']:+.3f}  frac(p>0)={eh2_metrics[g]['frac_pos']:.3f}  "
          f"ráfaga={eh2_metrics[g]['burst_mean']:.2f}  mean_p={eh2_metrics[g]['mean_p']:+.4f}")

print("\n  Comparaciones clave:")
eh2_pairs = [("axis", "axis_pec_only"), ("axis_pec_only", "automata_neutro"), ("axis", "vanilla")]
eh2_results = {}
for a, b in eh2_pairs:
    d_burst = tester.cohens_d(eh2_metrics[a]["burst_vals"], eh2_metrics[b]["burst_vals"])
    p_burst = tester.permutation_test(eh2_metrics[a]["burst_vals"], eh2_metrics[b]["burst_vals"])["p_value"]
    d_mp = tester.cohens_d(eh2_metrics[a]["meanp_vals"], eh2_metrics[b]["meanp_vals"])
    p_mp = tester.permutation_test(eh2_metrics[a]["meanp_vals"], eh2_metrics[b]["meanp_vals"])["p_value"]
    eh2_results[f"{a}_vs_{b}"] = {"d_burst": d_burst, "p_burst": p_burst, "d_meanp": d_mp, "p_meanp": p_mp}
    print(f"    {a:16s} vs {b:16s}  ráfaga d={d_burst:+.3f} p={p_burst:.4f}  |  mean_p d={d_mp:+.3f} p={p_mp:.4f}")

# ══════════════════════════════════════════════════════════════════════
# E-J — RDM / CKA / ángulos principales (capa final, t>0)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nE-J — RDM/CKA/ángulos (capa final)\n" + "=" * 70)

def per_prompt_mean(emb, lengths, exclude_t0=True):
    start = 1 if exclude_t0 else 0
    rows = []
    for i, L in enumerate(lengths):
        rows.append(emb[i, start:L].mean(axis=0) if L > start else emb[i, :L].mean(axis=0))
    return np.array(rows)

def per_token_matrix(emb, lengths, exclude_t0=True):
    start = 1 if exclude_t0 else 0
    rows = []
    for i, L in enumerate(lengths):
        if L <= start: continue
        rows.append(emb[i, start:L])
    return np.concatenate(rows, axis=0)

def linear_cka(X, Y):
    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n], Y[:n]
    H = np.eye(n) - np.ones((n, n)) / n
    K, L = X @ X.T, Y @ Y.T
    Kc, Lc = H @ K @ H, H @ L @ H
    hsic_xy = float((Kc * Lc).sum())
    hsic_xx = float((Kc * Kc).sum()); hsic_yy = float((Lc * Lc).sum())
    denom = np.sqrt(hsic_xx * hsic_yy)
    return hsic_xy / denom if denom > 1e-20 else 0.0

prompt_means = {g: per_prompt_mean(data[g][0][FINAL_L], data[g][1]) for g in GROUPS}
token_mats = {g: per_token_matrix(data[g][0][FINAL_L], data[g][1]) for g in GROUPS}

n_g = len(GROUPS)
centroid_dist = np.zeros((n_g, n_g))
centroids = {g: prompt_means[g].mean(axis=0) for g in GROUPS}
for i, gi in enumerate(GROUPS):
    for j, gj in enumerate(GROUPS):
        if j <= i: continue
        ci, cj = centroids[gi], centroids[gj]
        dd = 1 - np.dot(ci, cj) / (np.linalg.norm(ci) * np.linalg.norm(cj))
        centroid_dist[i, j] = centroid_dist[j, i] = dd

print("  RDM (distancia coseno de centroides):")
print("  " + " ".join(f"{g[:8]:>9s}" for g in GROUPS))
for i, g in enumerate(GROUPS):
    print(f"  {g[:20]:20s} " + " ".join(f"{centroid_dist[i,j]:9.4f}" for j in range(n_g)))

condensed = squareform(centroid_dist, checks=False)
Z = linkage(condensed, method="average")
clustering = {}
for k in [2, 3]:
    clusters = fcluster(Z, t=k, criterion="maxclust")
    clustering[k] = dict(zip(GROUPS, [int(c) for c in clusters]))
    print(f"  clustering k={k}: {clustering[k]}")

def block_stats(A, B):
    ds = [centroid_dist[GROUPS.index(a), GROUPS.index(b)] for a, b in product(A, B) if a != b]
    ckas = [linear_cka(prompt_means[a], prompt_means[b]) for a, b in product(A, B) if a != b]
    return float(np.mean(ds)), float(np.mean(ckas)), len(ds)

blocks = {}
for name, (A, B) in {
    "dentro_axis": (AXIS_FAMILY, AXIS_FAMILY), "dentro_generic": (GENERIC_FAMILY, GENERIC_FAMILY),
    "axis_vs_generic": (AXIS_FAMILY, GENERIC_FAMILY), "axis_vs_automata": (AXIS_FAMILY, ["automata_neutro"]),
    "generic_vs_automata": (GENERIC_FAMILY, ["automata_neutro"]),
}.items():
    d_, c_, n_ = block_stats(A, B)
    blocks[name] = {"dist": d_, "cka": c_, "n": n_}
    print(f"  {name:22s} dist={d_:.4f}  cka={c_:.4f}  (n={n_})")

pcs = {}
for g in GROUPS:
    X = token_mats[g] - token_mats[g].mean(axis=0)
    svd = TruncatedSVD(n_components=5, random_state=42)
    svd.fit(X)
    pcs[g] = svd.components_.T

def angle_block(A, B):
    angs = [float(np.degrees(np.mean(subspace_angles(pcs[a], pcs[b])))) for a, b in product(A, B) if a != b]
    return float(np.mean(angs))

angles = {
    "dentro_axis": angle_block(AXIS_FAMILY, AXIS_FAMILY),
    "dentro_generic": angle_block(GENERIC_FAMILY, GENERIC_FAMILY),
    "axis_vs_generic": angle_block(AXIS_FAMILY, GENERIC_FAMILY),
    "axis_vs_automata": angle_block(AXIS_FAMILY, ["automata_neutro"]),
    "generic_vs_automata": angle_block(GENERIC_FAMILY, ["automata_neutro"]),
}
print("\n  Ángulos principales (PC1-5):")
for k, v in angles.items():
    print(f"    {k:22s} {v:.2f}°")

# ══════════════════════════════════════════════════════════════════════
# E-F2 — perfil por capa (proyección v_hat + PR)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nE-F2 — perfil por capa\n" + "=" * 70)
proj_by_layer = {g: {} for g in GROUPS}
pr_by_layer = {g: {} for g in GROUPS}
for g in GROUPS:
    layers, lengths = data[g]
    for n in LAYER_NS:
        emb = layers[n]
        pvals, prvals = [], []
        for i, L in enumerate(lengths):
            if L < 2: continue
            series = project_trajectory(emb[i, :L], v_hat)
            pvals.append(float(np.mean(series)))
            prvals.append(participation_ratio(emb[i, :L]))
        proj_by_layer[g][n] = pvals
        pr_by_layer[g][n] = prvals

print("  Proyección v_hat media por capa (columnas L4..L63):")
header = "  " + " ".join(f"L{n:>3d}".rjust(8) for n in LAYER_NS)
print(header)
for g in GROUPS:
    row = f"  {g:18s}" + "".join(f"{np.mean(proj_by_layer[g][n]):8.4f}" for n in LAYER_NS)
    print(row)

print("\n  PR media por capa:")
print(header)
for g in GROUPS:
    row = f"  {g:18s}" + "".join(f"{np.mean(pr_by_layer[g][n]):8.2f}" for n in LAYER_NS)
    print(row)

print("\n  d de Cohen por capa, axis_pec_only vs automata_neutro:")
ds_by_layer = []
for n in LAYER_NS:
    res = tester.full_comparison(proj_by_layer["axis_pec_only"][n], proj_by_layer["automata_neutro"][n],
                                  "axis_pec_only", "automata_neutro")
    ds_by_layer.append(res["cohens_d"])
    print(f"    L{n:3d}: d={res['cohens_d']:+.3f}  p={res['permutation_test']['p_value']:.4f}")

# ══════════════════════════════════════════════════════════════════════
# A2 — rotación vertical
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nA2 — rotación vertical (mismo token, L4->L63)\n" + "=" * 70)

def cos(a, b):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    denom = na * nb
    denom = np.where(denom < 1e-10, 1.0, denom)
    return (a * b).sum(axis=-1) / denom

rotation = {}
for g in GROUPS:
    layers, lengths = data[g]
    e2e = []
    for i, L in enumerate(lengths):
        if L < 2: continue
        h_first = layers[LAYER_NS[0]][i, :L]
        h_last = layers[LAYER_NS[-1]][i, :L]
        c = cos(h_first, h_last)
        e2e.extend(np.degrees(np.arccos(np.clip(c, -1, 1))).tolist())
    rotation[g] = {"mean_deg": float(np.mean(e2e)), "std_deg": float(np.std(e2e))}
    print(f"  {g:18s} L{LAYER_NS[0]}->L{LAYER_NS[-1]}: {rotation[g]['mean_deg']:.2f}° (±{rotation[g]['std_deg']:.2f})")

# ══════════════════════════════════════════════════════════════════════
out = {
    "el_proj_t0_mean": {g: float(np.mean(v)) for g, v in proj_t0.items()},
    "el_proj_rest_mean": {g: float(np.mean(v)) for g, v in proj_rest.items()},
    "el_pairs": el_results,
    "eh2_metrics": {g: {k: v for k, v in m.items() if k not in ("burst_vals", "meanp_vals")} for g, m in eh2_metrics.items()},
    "eh2_pairs": eh2_results,
    "ej_centroid_dist": centroid_dist.tolist(),
    "ej_clustering": clustering,
    "ej_blocks": blocks,
    "ej_angles": angles,
    "ef2_proj_by_layer_mean": {g: {n: float(np.mean(v)) for n, v in proj_by_layer[g].items()} for g in GROUPS},
    "ef2_pr_by_layer_mean": {g: {n: float(np.mean(v)) for n, v in pr_by_layer[g].items()} for g in GROUPS},
    "ef2_d_axis_pec_vs_automata_by_layer": dict(zip(LAYER_NS, ds_by_layer)),
    "a2_rotation": rotation,
}
Path("qwen3_fase0_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("\nGuardado: qwen3_fase0_results.json")
