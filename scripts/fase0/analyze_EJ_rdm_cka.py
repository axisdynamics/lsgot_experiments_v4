#!/usr/bin/env python3
"""
E-J — Subespacio completo: RDM + CKA multi-condición.
INSTRUCCIONES_AGENTE_LSGOT.md §3.2, FASE 0 paso 3.

Pregunta: ¿"identidad" es un factor ORGANIZADOR del espacio representacional
(orientación de PCs, cluster en la RDM) o solo una dirección lineal (v_hat)?

T4 (control de circularidad): en ningún paso de este script se usa v_hat
como feature. La RDM, el CKA y los ángulos principales se calculan sobre
las representaciones completas (5376-d); v_hat solo se usa al final, como
chequeo post-hoc, para ver si la dirección de mayor separación en la RDM
coincide con v_hat — no para construir la RDM.

Unidad de representación por prompt: media del hidden state sobre los
tokens generados, excluyendo t=0 (T1) — misma convención que group_mean()
en tier0_metrics.persona_vector.

Uso:
    python analyze_EJ_rdm_cka.py
"""

import sys
import json
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.linalg import subspace_angles

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from statistical_tests import GeometricStatisticalTests  # noqa: E402

BASE = "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined"
FREE_DIR = Path(BASE) / "results_local" / "sia_extended_v5"
V_HAT_PATH = Path(__file__).parent.parent / "perturbation" / "v_identidad.npy"

# chileatiende / chileatiende_sia / chileatiende_sia_v2 excluidas: confound
# de repetición de markup HTML en el prompt (ver
# evidence/CHILEATIENDE_MARKUP_CONFOUND_REPORT.md, T11).
ALL_GROUPS = [
    "axis", "generic_long", "generic_short", "vanilla", "axis_short",
    "automata_neutro", "axis_pec_only",
]
IDENTITY_GROUPS = {"axis", "axis_short", "axis_pec_only"}
RESTRICTION_GROUPS = {"automata_neutro"}


def load_free(name: str):
    d = np.load(FREE_DIR / f"{name}_embeddings.npz", allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    lengths = d["lengths"].astype(int)
    return emb, lengths


def per_prompt_mean(emb, lengths, exclude_t0=True):
    """(n_prompts, D) — media del hidden state por prompt, t>0 (T1)."""
    start = 1 if exclude_t0 else 0
    rows = []
    for i, L in enumerate(lengths):
        if L <= start:
            continue
        rows.append(emb[i, start:L].mean(axis=0))
    return np.array(rows)


def per_token_matrix(emb, lengths, exclude_t0=True, max_tokens_per_prompt=None):
    """(n_tokens_total, D) — todos los tokens concatenados (para PCA/subspace)."""
    start = 1 if exclude_t0 else 0
    rows = []
    for i, L in enumerate(lengths):
        if L <= start:
            continue
        chunk = emb[i, start:L]
        if max_tokens_per_prompt is not None:
            chunk = chunk[:max_tokens_per_prompt]
        rows.append(chunk)
    return np.concatenate(rows, axis=0)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """CKA lineal (Kornblith et al. 2019) entre X (n,Dx) y Y (n,Dy), mismo n filas."""
    n = X.shape[0]
    assert Y.shape[0] == n
    H = np.eye(n) - np.ones((n, n)) / n
    K = X @ X.T
    L = Y @ Y.T
    Kc = H @ K @ H
    Lc = H @ L @ H
    hsic_xy = float((Kc * Lc).sum())
    hsic_xx = float((Kc * Kc).sum())
    hsic_yy = float((Lc * Lc).sum())
    denom = np.sqrt(hsic_xx * hsic_yy)
    return hsic_xy / denom if denom > 1e-20 else 0.0


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    """MMD^2 con kernel RBF — sustituto barato de Wasserstein en alta dimensión."""
    if gamma is None:
        allpts = np.concatenate([X, Y], axis=0)
        d2 = np.sum((allpts[:, None, :] - allpts[None, :, :]) ** 2, axis=-1)
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / (2 * med) if med > 0 else 1.0

    def k(A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * d2)

    kxx = k(X, X)
    kyy = k(Y, Y)
    kxy = k(X, Y)
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pcs", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    v_hat = np.load(V_HAT_PATH)
    tester = GeometricStatisticalTests(n_permutations=1000, random_seed=args.seed)

    print("Cargando y agregando por prompt (t>0)...")
    prompt_means = {}   # (20, 5376) por condición
    token_mats = {}      # (~5000, 5376) por condición, para PCA
    for g in ALL_GROUPS:
        emb, lengths = load_free(g)
        prompt_means[g] = per_prompt_mean(emb, lengths, exclude_t0=True)
        token_mats[g] = per_token_matrix(emb, lengths, exclude_t0=True)
        print(f"  {g:22s} prompt_means={prompt_means[g].shape}  tokens={token_mats[g].shape}")

    # ── 1. RDM intra-condición (prompt a prompt, 1 - Pearson) ────────────
    print("\n=== RDM intra-condición (1 - correlación media entre prompts) ===")
    intra_rdm_mean = {}
    for g in ALL_GROUPS:
        X = prompt_means[g]
        corr = np.corrcoef(X)
        iu = np.triu_indices(len(X), k=1)
        intra_rdm_mean[g] = float(np.mean(1 - corr[iu]))
        print(f"  {g:22s} mean(1-corr)={intra_rdm_mean[g]:.4f}")

    # ── 2. RDM entre condiciones: distancia de centroides + MMD ──────────
    print("\n=== RDM entre condiciones (distancia de centroides, coseno) ===")
    n_g = len(ALL_GROUPS)
    centroid_dist = np.zeros((n_g, n_g))
    mmd_mat = np.zeros((n_g, n_g))
    centroids = {g: prompt_means[g].mean(axis=0) for g in ALL_GROUPS}
    for i, gi in enumerate(ALL_GROUPS):
        for j, gj in enumerate(ALL_GROUPS):
            if j <= i:
                continue
            ci, cj = centroids[gi], centroids[gj]
            cos_dist = 1 - np.dot(ci, cj) / (np.linalg.norm(ci) * np.linalg.norm(cj))
            centroid_dist[i, j] = centroid_dist[j, i] = cos_dist
            m = mmd_rbf(prompt_means[gi], prompt_means[gj])
            mmd_mat[i, j] = mmd_mat[j, i] = m

    print("  " + " ".join(f"{g[:8]:>9s}" for g in ALL_GROUPS))
    for i, g in enumerate(ALL_GROUPS):
        print(f"  {g[:20]:20s} " + " ".join(f"{centroid_dist[i,j]:9.4f}" for j in range(n_g)))

    # ── 3. Bootstrap CI sobre distancia identidad-vs-restricción ─────────
    print("\n=== Bootstrap (n=%d): distancia centroide identidad vs restricción ===" % args.n_boot)
    id_list = sorted(IDENTITY_GROUPS)
    res_list = sorted(RESTRICTION_GROUPS)
    id_boot, within_id_boot, within_res_boot = [], [], []
    for _ in range(args.n_boot):
        id_c = np.mean([
            prompt_means[g][rng.randint(0, len(prompt_means[g]), len(prompt_means[g]))].mean(axis=0)
            for g in id_list], axis=0)
        res_c = np.mean([
            prompt_means[g][rng.randint(0, len(prompt_means[g]), len(prompt_means[g]))].mean(axis=0)
            for g in res_list], axis=0)
        d = 1 - np.dot(id_c, res_c) / (np.linalg.norm(id_c) * np.linalg.norm(res_c))
        id_boot.append(d)
    ci_lo, ci_hi = np.percentile(id_boot, [2.5, 97.5])
    print(f"  dist(centroide_identidad, centroide_restricción) = "
          f"{np.mean(id_boot):.4f}  IC95%=({ci_lo:.4f}, {ci_hi:.4f})")

    # comparación: dispersión intra-identidad vs intra-restricción (bootstrap de pares)
    def within_group_dist(members):
        pairs = list(combinations(members, 2))
        ds = []
        for a, b in pairs:
            ca, cb = centroids[a], centroids[b]
            ds.append(1 - np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb)))
        return ds

    within_id = within_group_dist(id_list)
    within_res = within_group_dist(res_list) if len(res_list) > 1 else []
    cross = [centroid_dist[ALL_GROUPS.index(a), ALL_GROUPS.index(b)]
             for a in id_list for b in res_list]
    print(f"  dentro de identidad (pares axis/axis_short/axis_pec_only/sia_v2): "
          f"mean={np.mean(within_id):.4f}  (n={len(within_id)} pares)")
    print(f"  cruzado identidad↔restricción: mean={np.mean(cross):.4f}  (n={len(cross)} pares)")
    if within_res:
        print(f"  dentro de restricción: mean={np.mean(within_res):.4f}  (n={len(within_res)} pares)")

    # ── 4. Clustering jerárquico sobre la RDM entre condiciones ──────────
    print("\n=== Clustering jerárquico (average linkage) sobre RDM(centroides) ===")
    condensed = squareform(centroid_dist, checks=False)
    Z = linkage(condensed, method="average")
    for k in [2, 3, 4]:
        clusters = fcluster(Z, t=k, criterion="maxclust")
        assign = {g: int(c) for g, c in zip(ALL_GROUPS, clusters)}
        print(f"  k={k}: {assign}")

    # ── 5. CKA lineal entre pares de condiciones (n=20 prompts, o 19 con sia_v2) ──
    print("\n=== CKA lineal entre pares de condiciones ===")
    cka_mat = {}
    for gi, gj in combinations(ALL_GROUPS, 2):
        Xi, Xj = prompt_means[gi], prompt_means[gj]
        n = min(len(Xi), len(Xj))
        c = linear_cka(Xi[:n], Xj[:n])
        cka_mat[f"{gi}__{gj}"] = c
    intra_id_cka = [cka_mat[f"{a}__{b}"] if f"{a}__{b}" in cka_mat else cka_mat[f"{b}__{a}"]
                    for a, b in combinations(id_list, 2)]
    cross_cka = [cka_mat.get(f"{a}__{b}", cka_mat.get(f"{b}__{a}"))
                 for a in id_list for b in res_list]
    print(f"  CKA intra-identidad: mean={np.mean(intra_id_cka):.4f}  (n={len(intra_id_cka)})")
    print(f"  CKA identidad↔restricción: mean={np.mean(cross_cka):.4f}  (n={len(cross_cka)})")
    d_cka = tester.cohens_d(intra_id_cka, cross_cka)
    print(f"  Cohen's d (intra-identidad vs cruzado) = {d_cka:+.3f}")

    # ── 6. Ángulos principales entre subespacios PCA top-k ────────────────
    print(f"\n=== Ángulos principales entre subespacios PC1..{args.n_pcs} ===")
    from sklearn.decomposition import TruncatedSVD
    pcs = {}
    for g in ALL_GROUPS:
        X = token_mats[g] - token_mats[g].mean(axis=0)
        svd = TruncatedSVD(n_components=args.n_pcs, random_state=args.seed)
        svd.fit(X)
        pcs[g] = svd.components_.T  # (D, n_pcs), columnas ortonormales
    angle_mat = {}
    for gi, gj in combinations(ALL_GROUPS, 2):
        angles = subspace_angles(pcs[gi], pcs[gj])
        angle_mat[f"{gi}__{gj}"] = float(np.degrees(np.mean(angles)))
    intra_id_angles = [angle_mat[f"{a}__{b}"] if f"{a}__{b}" in angle_mat else angle_mat[f"{b}__{a}"]
                       for a, b in combinations(id_list, 2)]
    cross_angles = [angle_mat.get(f"{a}__{b}", angle_mat.get(f"{b}__{a}"))
                    for a in id_list for b in res_list]
    print(f"  ángulo medio intra-identidad: {np.mean(intra_id_angles):.2f}°  (n={len(intra_id_angles)})")
    print(f"  ángulo medio identidad↔restricción: {np.mean(cross_angles):.2f}°  (n={len(cross_angles)})")

    # ── 7. Chequeo post-hoc (NO usado para construir la RDM/CKA): v_hat ──
    axis_of_max_sep = centroids["axis_pec_only"] - centroids["automata_neutro"]
    axis_of_max_sep /= np.linalg.norm(axis_of_max_sep)
    cos_with_vhat = float(np.dot(axis_of_max_sep, v_hat))
    print(f"\n=== Post-hoc: alineación del eje de máxima separación con v_hat ===")
    print(f"  cos(centroid_axis_pec_only - centroid_automata_neutro, v_hat) = {cos_with_vhat:.4f}")

    out = {
        "experiment": "E-J",
        "groups": ALL_GROUPS,
        "identity_groups": sorted(IDENTITY_GROUPS),
        "restriction_groups": sorted(RESTRICTION_GROUPS),
        "intra_condition_rdm_mean": intra_rdm_mean,
        "centroid_cosine_distance_matrix": centroid_dist.tolist(),
        "mmd_rbf_matrix": mmd_mat.tolist(),
        "bootstrap_identity_vs_restriction_centroid_dist": {
            "mean": float(np.mean(id_boot)), "ci95": [float(ci_lo), float(ci_hi)],
        },
        "within_identity_dist": {"mean": float(np.mean(within_id)), "n": len(within_id)},
        "within_restriction_dist": ({"mean": float(np.mean(within_res)), "n": len(within_res)}
                                     if within_res else None),
        "cross_identity_restriction_dist": {"mean": float(np.mean(cross)), "n": len(cross)},
        "cka_matrix": cka_mat,
        "cka_intra_identity_mean": float(np.mean(intra_id_cka)),
        "cka_cross_mean": float(np.mean(cross_cka)),
        "cka_cohens_d": d_cka,
        "principal_angle_matrix_deg": angle_mat,
        "principal_angle_intra_identity_mean_deg": float(np.mean(intra_id_angles)),
        "principal_angle_cross_mean_deg": float(np.mean(cross_angles)),
        "posthoc_cos_max_sep_axis_vs_vhat": cos_with_vhat,
    }
    out_path = Path(__file__).parent / "EJ_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
