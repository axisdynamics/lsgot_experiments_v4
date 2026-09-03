#!/usr/bin/env python3
"""
Análisis local (CPU, numpy) del J-lens por capa — verbalización por capas.

Consume la re-extracción GPU (scripts/fase4_t2/extract_layers_jlens.py,
results_jlens/) y chequea las 3 predicciones registradas en
evidence/WORKSPACE_HYPOTHESIS_REPORT.md §2.2 ANTES de la re-extracción:

  P1. El readout verbal del auto-chequeo (vocabulario del "testigo": pausa,
      silencio, verificación…) aparece en la banda L13-L33, y el readout
      "snap" al primer token real en L33-L35 (costura de salida).
  P2. El dip geométrico de L30 se ve como convergencia del readout al
      contenido común: la separación entre condiciones (JS del readout)
      cae en L30 respecto a sus vecinas.
  P3. La separación máxima del readout entre condiciones coincide con el
      pico geométrico L35.

Lentes por capa (sobre el estado t=0 de cada (cond, prompt)):
  naive  = W_U @ final_norm(h_ℓ)          [final_norm = pesos RMSNorm final]
  jvp    = W_U @ (J_ℓ @ h_ℓ) exacto por muestra — el J-lens primario
           (top-50 guardado en el pod; la Jacobiana de la propia muestra es la
           definición del J-lens de Anthropic)
  jlens  = W_U @ (J̄_ℓ @ h_ℓ) SOLO si existen los archivos jacobians/J_L*.npy
           (el --jbar del pod quedó opcional por los muros medidos — ver
           SOUL_MD_UPDATE_GUIDE.md §9)

Criterios de decisión (fijados ANTES de ver datos, en el espíritu del
pre-registro A1 — no se cambian post-hoc; el instrumento primario es el JVP
exacto por muestra):
  P1a CONFIRMADA si onset_autocheck ∈ [13, 33], donde onset = primera capa ℓ
      en que la hit-rate media del vocabulario auto-chequeo en el top-10 del
      JVP sobre las condiciones identitarias (axis, axis_short, axis_pec_only)
      es ≥ 0.1 sostenida ≥ 3 capas consecutivas.
  P1b CONFIRMADA si snap ∈ [33, 35], donde snap = primera capa con match@1
      medio (JVP, condiciones identitarias) ≥ 0.5.
  P2  CONFIRMADA si sep[30] < sep[29] y sep[30] < sep[31], con sep = 1 −
      solape Jaccard medio del top-50 del JVP entre todas las parejas de
      condiciones (la separación del readout).
  P3  CONFIRMADA si argmax(sep) ∈ {34, 35, 36}.
  Adicional (corroborativo, no confirmatorio): la separación JS del lens
  ingenuo (full logits) se reporta por capa.
Resultado = CONFIRMADA / REFUTADA / PARCIAL (con los números a la vista).
Todo se reporta, coincida o no.

Requisito previo: W_U y normas descargadas con scripts/fase4_t2/fetch_wu_partial.py
  python fetch_wu_partial.py --token hf_xxx --out-dir <data-dir>/wu_norm

Uso:
  python analyze_j_lens.py [--data-dir DIR] [--out j_lens_results.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

GROUPS = ["axis", "generic_long", "generic_short", "vanilla", "axis_short",
          "automata_neutro", "axis_pec_only"]
IDENTITY_GROUPS = ["axis", "axis_short", "axis_pec_only"]
N_LAYERS = 60
EPS = 1e-6

# Vocabulario del mecanismo "testigo" (auto-chequeo cableado) y de identidad,
# anclado en el léxico de axis.dna (frecuencias: silencio 13, eje 12, pausa 8,
# soy 7, axis 7, pec 5, verific 4, umbral 2, testigo 2, chequeo 1).
# Cada string se tokeniza sin special tokens; si produce varios ids, todos
# cuentan (hit = cualquier id del string en el top-k).
AUTOCHECK_WORDS = [
    "testigo", "pausa", "silencio", "verific", "chequeo", "autochequeo",
    "constat", "observ", "monitoreo", "comprob", "registro", "respiro",
    "witness", "check", "pause", "observe", "silence",
]
IDENTITY_WORDS = [
    "soy", "identidad", "identific", "axis", "eje", "pec", "umbral",
    "persona", "autómata", "identidad", "nombre",
]

WU_CHUNK = 8192  # chunk de vocabulario para el matmul con W_U (RAM)


def load_first_tokens_real(tok, resp_dir):
    """Primer token REAL de cada respuesta generada por el pipeline original."""
    ft = {}
    for g in GROUPS:
        rows = json.load(open(resp_dir / f"{g}_responses.json"))
        enc = [tok(r["response"], add_special_tokens=False).input_ids for r in rows]
        ft[g] = [e[0] if e else None for e in enc]
    return ft


def vocab_sets(tok, words):
    out = []
    for w in words:
        ids = tok(w, add_special_tokens=False).input_ids
        if ids:
            out.append((w, set(ids)))
    return out


def rms_norm(h, w, eps=EPS):
    """RMSNorm final (Gemma4): w ⊙ h / sqrt(mean(h²)+eps). h: (D,) o (D, N)."""
    r = np.sqrt(np.mean(h.astype(np.float64) ** 2, axis=0) + eps)
    return (h / r.astype(np.float32)) * w[:, None] if h.ndim == 2 else (h / r) * w


def wu_topk_logits(WU, x, k=10):
    """W_U @ x en chunks de vocabulario. x: (D,) o (D, N). Devuelve
    (logits (V,N) f32 si N>1... solo top-k y logits completos por separado)."""
    V = WU.shape[0]
    single = x.ndim == 1
    x = x[:, None] if single else x
    N = x.shape[1]
    logits = np.empty((V, N), dtype=np.float32)
    for v0 in range(0, V, WU_CHUNK):
        logits[v0:v0 + WU_CHUNK] = WU[v0:v0 + WU_CHUNK] @ x
    if single:
        return logits[:, 0]
    return logits


def topk_ids(logits, k=10):
    """Índices top-k por columna. logits: (V, N) f32."""
    V, N = logits.shape
    kk = min(k, V)
    idx = np.argpartition(-logits, kk - 1, axis=0)[:kk]
    order = np.argsort(-logits[idx, np.arange(N)], axis=0)
    return idx[order, np.arange(N)]


def match_rates(logits, first_tokens):
    """match@1 y match@10 por prompt. logits: (V, N)."""
    N = logits.shape[1]
    t1 = topk_ids(logits, 10)
    m1 = sum(1 for j in range(N) if first_tokens[j] is not None
             and t1[0, j] == first_tokens[j])
    m10 = sum(1 for j in range(N) if first_tokens[j] is not None
              and first_tokens[j] in t1[:, j])
    return m1, m10


def js_pairwise(logits_a, logits_b):
    """Mediana de JS sobre pares de prompts entre dos condiciones (vectorizado).
    logits: (V, N) f32 cada una.
    JS(p_i, q_j) = 0.5 [KL(p_i||M) + KL(q_j||M)], M = 0.5(p_i + q_j)
    KL(p_i||M) = -H(p_i) - Σ_v p_i log M  con H = entropía."""
    pa = _softmax(logits_a)
    pb = _softmax(logits_b)
    N = pa.shape[1]
    ha = -(pa * np.log(pa + 1e-30)).sum(axis=0)      # (N,)
    hb = -(pb * np.log(pb + 1e-30)).sum(axis=0)
    M = 0.5 * (pa[:, :, None] + pb[:, None, :])      # (V, N, N)
    logM = np.log(M + 1e-30)
    cross_ab = np.einsum("vi,vij->ij", pa, logM)     # Σ_v pa[v,i] log M[v,i,j]
    cross_ba = np.einsum("vj,vij->ij", pb, logM)
    js = 0.5 * (-ha[:, None] - cross_ab - hb[None, :] - cross_ba)
    del M, logM
    return float(np.median(js))


def _softmax(logits):
    x = logits - logits.max(axis=0)
    e = np.exp(x)
    return e / e.sum(axis=0)


def scan_tokens(top_ids, vocab, k=10):
    """Hit-rate: fracción de prompts con ≥1 id del vocabulario en el top-k.
    top_ids: (k, N)."""
    N = top_ids.shape[1]
    if N == 0:
        return 0.0
    hits = 0
    for j in range(N):
        ids = set(top_ids[:k, j].tolist())
        if any(ids & vids for _, vids in vocab):
            hits += 1
    return hits / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=Path(
        "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
        "gemma4_31b_combined/results_jlens"))
    ap.add_argument("--resp-dir", default=Path(
        "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
        "gemma4_31b_combined/results_local/sia_extended_v5"))
    ap.add_argument("--tok", default=Path(
        "/home/plaxius/.cache/huggingface/hub/models--google--gemma-4-31B-it/"
        "snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "j_lens_results.json"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    resp_dir = Path(args.resp_dir)
    has_jbar = (data_dir / "jacobians" / "J_L0.npy").exists()
    print(f"J̄ disponible: {has_jbar} (--jbar del pod)")

    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        print(f"meta: {meta['model']} | base_check: {meta['base_check_per_condition']}")
        print(f"primal peor: {max(meta['primal_checks'], key=lambda c: c['worst_rel_err'])['worst_rel_err']:.2e}")
    else:
        print("meta.json ausente (el pod murió antes de escribirla) — se analiza con lo rescatado")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.tok))

    WU = np.load(data_dir / "wu_norm" / "embed_tokens.npy", mmap_mode="r")  # (V, D) f32
    final_w = np.load(data_dir / "wu_norm" / "final_norm.npy")  # (D,)
    D = WU.shape[1]
    V = WU.shape[0]
    print(f"W_U {V}x{D} cargado (mmap)")

    first_real = load_first_tokens_real(tok, resp_dir)
    autocheck_vocab = vocab_sets(tok, AUTOCHECK_WORDS)
    identity_vocab = vocab_sets(tok, IDENTITY_WORDS)

    # estados por condición (solo las rescatadas — el pod murió antes de
    # terminar automata_neutro, 2026-09-03)
    GROUPS = [g for g in globals()["GROUPS"] if (data_dir / "states" / f"{g}.npz").exists()]
    IDENTITY_GROUPS = [g for g in globals()["IDENTITY_GROUPS"] if g in GROUPS]
    print(f"condiciones disponibles: {GROUPS}", flush=True)
    states = {}
    for g in GROUPS:
        d = np.load(data_dir / "states" / f"{g}.npz")
        states[g] = {"h": d["t0_states"], "y": d["y"], "first_saved": d["first_token_ids"],
                     "T": d["T"]}
        print(f"  {g}: T mean={d['T'].mean():.0f}", flush=True)
    jvp = {}
    for g in GROUPS:
        p = data_dir / "jvp" / f"{g}.npz"
        if p.exists():
            jvp[g] = np.load(p)

    # ── validación de base ──
    base = {}
    print("\n=== validación de base ===")
    for g in GROUPS:
        y = states[g]["y"].T  # (D, 20)
        lgy = wu_topk_logits(WU, y)
        ok_wu = sum(1 for j in range(20)
                    if int(lgy[:, j].argmax()) == int(states[g]["first_saved"][j]))
        ok_real = sum(1 for j in range(20) if first_real[g][j] is not None
                      and int(states[g]["first_saved"][j]) == first_real[g][j])
        base[g] = {"wu_y_vs_saved": f"{ok_wu}/20", "saved_vs_real": f"{ok_real}/20"}
        print(f"  {g:18s} W_U@y vs saved {ok_wu}/20 | saved vs REAL {ok_real}/20")

    # ── curvas por capa ──
    curves = {"naive": {"match1": {}, "match10": {}, "scan_autocheck": {},
                        "scan_identity": {}, "js_cond_pairs": []},
              "jlens": {"match1": {}, "match10": {}, "scan_autocheck": {},
                        "scan_identity": {}, "js_cond_pairs": []},
              "jvp": {"match1": {}, "match10": {}, "scan_autocheck": {},
                      "first_tok_logprob": {}}}
    sep = {"naive": [], "jlens": []}
    sep_details = {"naive": {}, "jlens": {}}

    print("\nProcesando 60 capas × 3 lentes...", flush=True)
    for ell in range(N_LAYERS):
        # logits naive (y jlens si hay J̄) por condición para esta capa
        naive_logits = {}
        for g in GROUPS:
            h = states[g]["h"][:, ell].T  # (D, 20) f32
            naive_logits[g] = wu_topk_logits(WU, rms_norm(h, final_w))
        jlens_logits = {}
        if has_jbar:
            Jbar = np.load(data_dir / "jacobians" / f"J_L{ell}.npy", mmap_mode="r")
            for g in GROUPS:
                h = states[g]["h"][:, ell].T
                jlens_logits[g] = wu_topk_logits(WU, Jbar @ h)
            del Jbar

        for lens, logs in (("naive", naive_logits), ("jlens", jlens_logits)):
            if not logs:
                continue
            for g in GROUPS:
                m1, m10 = match_rates(logs[g], first_real[g])
                curves[lens]["match1"].setdefault(g, []).append(m1 / 20.0)
                curves[lens]["match10"].setdefault(g, []).append(m10 / 20.0)
                t10 = topk_ids(logs[g], 10)
                curves[lens]["scan_autocheck"].setdefault(g, []).append(
                    scan_tokens(t10, autocheck_vocab))
                curves[lens]["scan_identity"].setdefault(g, []).append(
                    scan_tokens(t10, identity_vocab))
            # separación entre condiciones (JS, mediana sobre pares de prompts)
            js_pairs = {}
            for i, ga in enumerate(GROUPS):
                for gb in GROUPS[i + 1:]:
                    js_pairs[f"{ga}|{gb}"] = js_pairwise(logs[ga], logs[gb])
            sep[lens].append(float(np.median(list(js_pairs.values()))))
            for k2, v2 in js_pairs.items():
                sep_details[lens].setdefault(k2, []).append(v2)

        # JVP (exacto por muestra): match y scan sobre el top-50 guardado
        jvp_top50 = {}
        for g in GROUPS:
            if g not in jvp:
                curves["jvp"]["match1"].setdefault(g, []).append(float("nan"))
                curves["jvp"]["match10"].setdefault(g, []).append(float("nan"))
                curves["jvp"]["scan_autocheck"].setdefault(g, []).append(float("nan"))
                curves["jvp"]["first_tok_logprob"].setdefault(g, []).append(float("nan"))
                continue
            top50 = jvp[g]["jvp_top50_ids"][:, ell]  # (20, 50)
            jvp_top50[g] = top50
            ft = np.array([first_real[g][j] if first_real[g][j] is not None else -1
                           for j in range(20)])
            m1 = int((top50[:, 0] == ft).sum())
            m10 = int(((top50[:, :10] == ft[:, None])).sum(axis=1).clip(max=1).sum())
            curves["jvp"]["match1"].setdefault(g, []).append(m1 / 20.0)
            curves["jvp"]["match10"].setdefault(g, []).append(m10 / 20.0)
            curves["jvp"]["scan_autocheck"].setdefault(g, []).append(
                scan_tokens(top50[:, :10].T, autocheck_vocab))
            curves["jvp"]["first_tok_logprob"].setdefault(g, []).append(
                float(np.nanmean(jvp[g]["jvp_first_tok_logprob"][:, ell])))
        # separación del JVP: 1 − Jaccard medio del top-50 entre parejas de condiciones
        overlaps = []
        gs = [g for g in GROUPS if g in jvp_top50]
        for i, ga in enumerate(gs):
            for gb in gs[i + 1:]:
                a, b = jvp_top50[ga], jvp_top50[gb]  # (20, 50)
                jac = np.mean([len(set(a[j].tolist()) & set(b[j].tolist())) / 50.0
                               for j in range(20)])
                overlaps.append(1.0 - jac)
        sep["jvp"] = sep.get("jvp", [])
        sep["jvp"].append(float(np.median(overlaps)) if overlaps else float("nan"))
        if has_jbar:
            del Jbar
        del naive_logits, jlens_logits
        if ell % 10 == 0:
            print(f"  L{ell} hecho", flush=True)

    # ── tabla compacta ──
    def row(name, lens, metric, conds, nd=2):
        r = []
        for g in conds:
            v = curves[lens][metric].get(g, [])
            r.append(f"{g}={np.mean(v):.{nd}f}" if v else f"{g}=—")
        return f"{name:34s} " + " ".join(r)

    print("\n=== match@1 medio por condición (lens ingenuo | J-lens pooled | JVP) ===")
    for g in GROUPS:
        m_jlens = np.mean(curves["jlens"]["match1"].get(g, [np.nan]))
        print(f"  {g:18s} naive={np.mean(curves['naive']['match1'][g]):.2f}  "
              f"jlens={'—' if np.isnan(m_jlens) else f'{m_jlens:.2f}'}  "
              f"jvp={np.mean(curves['jvp']['match1'][g]):.2f}")

    print("\n=== hit-rate auto-chequeo media (top-10) ===")
    print(row("naive", "naive", "scan_autocheck", GROUPS))
    print(row("jlens", "jlens", "scan_autocheck", GROUPS))
    print(row("jvp", "jvp", "scan_autocheck", GROUPS))

    # ── chequeo de predicciones (criterios en el docstring) ──
    def mean_identity(lens, metric):
        return np.mean([curves[lens][metric][g] for g in IDENTITY_GROUPS], axis=0)

    pred = {}

    auto = mean_identity("jvp", "scan_autocheck")
    onset = None
    run = 0
    for ell in range(N_LAYERS):
        run = run + 1 if auto[ell] >= 0.1 else 0
        if run >= 3:
            onset = ell - 2
            break
    pred["P1a_autocheck_onset"] = {
        "onset_layer": onset, "banda": [13, 33],
        "result": "CONFIRMADA" if onset is not None and 13 <= onset <= 33
        else ("REFUTADA" if onset is not None else "PARCIAL (sin onset ≥0.1)"),
        "hitrate_identity_peaks": [int(np.argmax(auto)), float(np.max(auto))]}

    m1 = mean_identity("jvp", "match1")
    snap = next((ell for ell in range(N_LAYERS) if m1[ell] >= 0.5), None)
    pred["P1b_snap"] = {
        "snap_layer": snap, "banda": [33, 35],
        "result": "CONFIRMADA" if snap is not None and 33 <= snap <= 35
        else ("REFUTADA" if snap is not None else "PARCIAL (match@1 < 0.5 nunca)")}

    sj = sep["jvp"]
    pred["P2_dip_L30"] = {
        "sep_29_30_31": [round(sj[29], 4), round(sj[30], 4), round(sj[31], 4)],
        "result": "CONFIRMADA" if sj[30] < sj[29] and sj[30] < sj[31] else "REFUTADA"}

    argmax_sep = int(np.argmax(sj))
    pred["P3_max_sep_L35"] = {
        "argmax_layer": argmax_sep, "banda": [34, 36],
        "result": "CONFIRMADA" if 34 <= argmax_sep <= 36 else "REFUTADA",
        "max_vs_L35": [round(sj[argmax_sep], 4), round(sj[35], 4)]}

    print("\n=== Predicciones (WORKSPACE_HYPOTHESIS_REPORT.md §2.2) ===")
    for k, v in pred.items():
        print(f"  {k}: {v['result']} — {v}")

    # curvas por capa para el JSON (compactas: guardar por condición)
    out = {
        "base_validation": base,
        "match1": {l: curves[l]["match1"] for l in ("naive", "jlens", "jvp")},
        "match10": {l: curves[l]["match10"] for l in ("naive", "jlens", "jvp")},
        "scan_autocheck": {l: curves[l]["scan_autocheck"] for l in ("naive", "jlens", "jvp")},
        "scan_identity": {l: curves[l]["scan_identity"] for l in ("naive", "jlens")},
        "jvp_first_tok_logprob": curves["jvp"]["first_tok_logprob"],
        "separation_js_median": sep,
        "separation_js_by_pair": sep_details,
        "predictions": pred,
        "criteria": "docstring de analyze_j_lens.py (fijados antes de los datos)",
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"\nGuardado: {args.out}")


if __name__ == "__main__":
    main()
