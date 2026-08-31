import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

EMB_DIR = Path("/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local/sia_extended_v5")
PAIRS = [("axis_pec_only", "automata_neutro"), ("axis_pec_only", "vanilla"),
         ("automata_neutro", "vanilla"), ("axis", "vanilla")]
SEED = 42
B = 500
TOKENS_PER_TRAJ = 10
OUT = Path(__file__).parent / "exp02_sia_result.json"

# W1 de referencia (Forman-Ricci), del exp03 recién corrido
FORMAN_W1_REF = {
    "axis_pec_only_vs_automata_neutro": 0.4349,
    "axis_pec_only_vs_vanilla": 0.0133,
    "automata_neutro_vs_vanilla": 0.4360,
    "axis_vs_vanilla": 0.0614,
}


def load(cond):
    d = np.load(EMB_DIR / f"{cond}_embeddings.npz", allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int)


def centroids(arr, lengths):
    return np.stack([arr[i, :lengths[i]].mean(axis=0) for i in range(arr.shape[0])])


def perm_labels(rng, n):
    return rng.permutation(np.array([0] * n + [1] * n))


def mmd_v1_perm_fast(X, rng, B):
    n = X.shape[0] // 2
    sample = X[:min(500, len(X))]
    d2 = ((sample[:, None, :] - sample[None, :, :]) ** 2).sum(-1)
    sigma = float(np.sqrt(np.median(d2[d2 > 0])))
    K = np.exp(-((X[:, None, :] - X[None, :, :]) ** 2).sum(-1) / (2 * sigma ** 2))
    obs = float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())
    y = np.array([0] * n + [1] * n)
    nulls = np.zeros(B)
    for bb in range(B):
        lab = rng.permutation(y)
        m0, m1 = lab == 0, lab == 1
        nulls[bb] = float(K[m0][:, m0].mean() + K[m1][:, m1].mean() - 2 * K[m0][:, m1].mean())
    p = float((1 + np.sum(nulls >= obs)) / (1 + B))
    return obs, p


def subsample_trajectories(arr, lengths, per_traj=TOKENS_PER_TRAJ):
    blocks = []
    for i in range(arr.shape[0]):
        T = lengths[i]
        sel = np.arange(T) if T <= per_traj else np.linspace(0, T - 1, per_traj).astype(int)
        blocks.append(arr[i, sel])
    pool = np.concatenate(blocks)
    starts, acc = [], 0
    for b in blocks:
        starts.append((acc, acc + len(b))); acc += len(b)
    return pool, starts


def mmd_pooled_perm_fast(pool_a, starts_a, pool_b, starts_b, rng, B, pca_dims=128):
    from sklearn.decomposition import PCA
    P_full = np.vstack([pool_a, pool_b])
    pca = PCA(n_components=min(pca_dims, P_full.shape[0], P_full.shape[1]), random_state=0)
    P = pca.fit_transform(P_full).astype(np.float32)
    n = len(pool_a)
    sample = P[:min(1000, len(P))]
    d2 = ((sample[:, None, :] - sample[None, :, :]) ** 2).sum(-1)
    sigma = float(np.sqrt(np.median(d2[d2 > 0])))
    diff = P[:, None, :] - P[None, :, :]
    K = np.exp(-(diff * diff).sum(-1) / (2 * sigma ** 2))
    na, nb = len(pool_a), len(pool_b)
    obs = float(K[:na, :na].mean() + K[na:, na:].mean() - 2 * K[:na, na:].mean())
    starts_all = starts_a + [(s0 + n, s1 + n) for s0, s1 in starts_b]
    n_traj = len(starts_a)
    nulls = np.zeros(B)
    for bb in range(B):
        lab = rng.permutation(np.array([0] * n_traj + [1] * n_traj))
        idx0 = np.concatenate([np.arange(starts_all[i][0], starts_all[i][1]) for i in range(2 * n_traj) if lab[i] == 0])
        idx1 = np.concatenate([np.arange(starts_all[i][0], starts_all[i][1]) for i in range(2 * n_traj) if lab[i] == 1])
        nulls[bb] = float(K[np.ix_(idx0, idx0)].mean() + K[np.ix_(idx1, idx1)].mean() - 2 * K[np.ix_(idx0, idx1)].mean())
    p = float((1 + np.sum(nulls >= obs)) / (1 + B))
    return obs, p


def linear_cka(X, Y):
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = (Xc @ Yc.T) ** 2
    den = np.linalg.norm(Xc @ Xc.T) * np.linalg.norm(Yc @ Yc.T)
    return float(num.sum() / den) if den > 0 else 0.0


def probe_auc(X, y, groups, seed=SEED):
    gkf = GroupKFold(n_splits=5)
    aucs = []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(solver="liblinear", max_iter=1000, tol=1e-3, random_state=seed).fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))


def main():
    rng = np.random.default_rng(SEED)
    report = {"B": B, "comparisons": {}}
    cache = {}

    for a, b in PAIRS:
        t0 = time.time()
        if a not in cache: cache[a] = load(a)
        if b not in cache: cache[b] = load(b)
        arr_a, len_a = cache[a]
        arr_b, len_b = cache[b]
        n = arr_a.shape[0]
        X_v1 = np.vstack([arr_a[:, 0, :], arr_b[:, 0, :]])
        y = np.array([0] * n + [1] * n)
        groups = np.arange(2 * n) % n
        cent_a, cent_b = centroids(arr_a, len_a), centroids(arr_b, len_b)
        pooled_cent = np.vstack([cent_a, cent_b])

        obs_cent = float(np.linalg.norm(cent_a.mean(0) - cent_b.mean(0)))
        null_cent = np.zeros(B)
        for bb in range(B):
            lab = perm_labels(rng, n)
            null_cent[bb] = float(np.linalg.norm(pooled_cent[lab == 0].mean(0) - pooled_cent[lab == 1].mean(0)))
        p_cent = float((1 + np.sum(null_cent >= obs_cent)) / (1 + B))

        obs_mmd_v1, p_mmd_v1 = mmd_v1_perm_fast(X_v1, rng, B)
        pool_a, starts_a = subsample_trajectories(arr_a, len_a)
        pool_b, starts_b = subsample_trajectories(arr_b, len_b)
        obs_mmd_pool, p_mmd_pool = mmd_pooled_perm_fast(pool_a, starts_a, pool_b, starts_b, rng, B)

        obs_cka_v1 = linear_cka(arr_a[:, 0, :], arr_b[:, 0, :])
        null_cka_v1 = np.zeros(B)
        for bb in range(B):
            lab = perm_labels(rng, n)
            null_cka_v1[bb] = linear_cka(X_v1[lab == 0], X_v1[lab == 1])
        p_cka_v1 = float((1 + np.sum(null_cka_v1 >= obs_cka_v1)) / (1 + B))

        obs_auc_v1 = probe_auc(X_v1, y, groups)
        Bp = max(50, B // 10)
        null_auc_v1 = np.zeros(Bp)
        for bb in range(Bp):
            yp = rng.permutation(y)
            null_auc_v1[bb] = probe_auc(X_v1, yp, groups)
        p_auc_v1 = float((1 + np.sum(null_auc_v1 >= obs_auc_v1)) / (1 + Bp))

        nrm_a = np.linalg.norm(arr_a[:, 0, :], axis=1)
        nrm_b = np.linalg.norm(arr_b[:, 0, :], axis=1)
        mw_stat, mw_p = mannwhitneyu(nrm_a, nrm_b, alternative="two-sided")

        key = f"{a}_vs_{b}"
        report["comparisons"][key] = {
            "forman_w1_ref": FORMAN_W1_REF.get(key),
            "centroides": {"dist": obs_cent, "p_perm": p_cent},
            "mmd_v1": {"mmd": obs_mmd_v1, "p_perm": p_mmd_v1},
            "mmd_pool": {"mmd": obs_mmd_pool, "p_perm": p_mmd_pool},
            "cka_v1": {"cka": obs_cka_v1, "p_perm": p_cka_v1},
            "probe_v1_auc": {"auc": obs_auc_v1, "p_perm": p_auc_v1},
            "mw_norma_v1": {"p": float(mw_p), "mean_a": float(nrm_a.mean()), "mean_b": float(nrm_b.mean())},
        }
        print(f"{key}: forman_w1={FORMAN_W1_REF.get(key)} | cent={obs_cent:.3f}(p={p_cent:.3f}) "
              f"mmd_v1={obs_mmd_v1:.4f}(p={p_mmd_v1:.3f}) mmd_pool={obs_mmd_pool:.4f}(p={p_mmd_pool:.3f}) "
              f"cka_v1={obs_cka_v1:.3f}(p={p_cka_v1:.3f}) auc_v1={obs_auc_v1:.3f}(p={p_auc_v1:.3f}) "
              f"mw_norma={nrm_a.mean():.1f}vs{nrm_b.mean():.1f}(p={mw_p:.2e}) [{time.time()-t0:.0f}s]")

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
