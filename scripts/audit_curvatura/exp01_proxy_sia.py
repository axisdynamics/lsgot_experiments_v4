import json
import re
from pathlib import Path
from collections import Counter

import numpy as np

DATA_DIR = Path("/home/plaxius/Escritorio/Buscando_la_geometría/LSGOT_v4/data/sia_extended_v5")
EMB_DIR = Path("/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local/sia_extended_v5")
GROUPS = ["axis", "axis_pec_only", "automata_neutro", "vanilla"]
PAIRS = [("axis_pec_only", "automata_neutro"), ("axis_pec_only", "vanilla"),
         ("automata_neutro", "vanilla"), ("axis", "vanilla")]
SEED = 42
B = 2000


def first_word(text):
    text = text.strip()
    if not text:
        return ""
    w = text.split()[0] if text.split() else ""
    return w.strip(".,;:!¿?\"'()*").lower()


def shannon_entropy(words):
    c = Counter(words)
    n = sum(c.values())
    if n == 0:
        return 0.0
    probs = np.array([v / n for v in c.values()])
    return float(-(probs * np.log2(probs)).sum())


def jaccard_topN(words_a, words_b, N=10):
    ca = set(w for w, _ in Counter(words_a).most_common(N))
    cb = set(w for w, _ in Counter(words_b).most_common(N))
    if not ca and not cb:
        return 1.0
    return len(ca & cb) / len(ca | cb)


def load_words(cond):
    data = json.loads((DATA_DIR / f"{cond}_responses.json").read_text())
    return [first_word(item.get("response", "")) for item in data]


def load_v1(cond):
    d = np.load(EMB_DIR / f"{cond}_embeddings.npz", allow_pickle=True)
    arr = d["embeddings"].astype(np.float32)
    return arr[:, 0, :]


def ols_condition_coef(y, cond_dummy, token_dummies=None):
    """y ~ intercept + cond_dummy [+ token_dummies]. Devuelve coef de cond_dummy."""
    n = len(y)
    X = [np.ones(n), cond_dummy]
    if token_dummies is not None and token_dummies.shape[1] > 0:
        X.append(token_dummies)
    X = np.column_stack(X)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[1])


def build_token_dummies(words, top_k=8):
    c = Counter(words)
    top = [w for w, _ in c.most_common(top_k)]
    n = len(words)
    D = np.zeros((n, len(top)))
    for i, w in enumerate(words):
        if w in top:
            D[i, top.index(w)] = 1.0
    return D


def main():
    words = {g: load_words(g) for g in GROUPS}
    v1 = {g: load_v1(g) for g in GROUPS}
    norms = {g: np.linalg.norm(v1[g], axis=1) for g in GROUPS}

    print("=== 1A: entropía y top-3 por condición ===")
    for g in GROUPS:
        ent = shannon_entropy(words[g])
        top3 = Counter(words[g]).most_common(3)
        print(f"  {g:18s} H={ent:.3f} bits  top3={top3}")

    rng = np.random.default_rng(SEED)
    print("\n=== 1A: Jaccard top-10 + 1B: OLS residualización (con/sin control de token-proxy) ===")
    results = {}
    for a, b in PAIRS:
        jac = jaccard_topN(words[a], words[b])
        y = np.concatenate([norms[a], norms[b]])
        cond = np.concatenate([np.zeros(len(norms[a])), np.ones(len(norms[b]))])
        all_words = words[a] + words[b]
        tok_dummies = build_token_dummies(all_words, top_k=8)

        coef_raw = ols_condition_coef(y, cond, None)
        coef_ctrl = ols_condition_coef(y, cond, tok_dummies)

        # permutation test del coeficiente controlado
        n = len(y)
        null = np.zeros(B)
        for i in range(B):
            cond_perm = rng.permutation(cond)
            null[i] = ols_condition_coef(y, cond_perm, tok_dummies)
        p_perm = float((1 + np.sum(np.abs(null) >= abs(coef_ctrl))) / (1 + B))

        key = f"{a}_vs_{b}"
        results[key] = {"jaccard_top10": jac, "coef_raw": coef_raw, "coef_ctrl": coef_ctrl, "p_perm_ctrl": p_perm}
        print(f"  {key:38s} Jaccard={jac:.2f}  coef_sin_control={coef_raw:+7.2f}  "
              f"coef_con_control={coef_ctrl:+7.2f}  p_perm={p_perm:.4f}")

    (Path(__file__).parent / "exp01_sia_result.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
