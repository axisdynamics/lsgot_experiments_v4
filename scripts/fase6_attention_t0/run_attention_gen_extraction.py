#!/usr/bin/env python3
"""
FASE 6 · ronda 2 §1.3 — atención DURANTE la generación (t>0).

Propuesta: Descargas/Experimento_atencion_v2.md §1.3. El techo de la ronda 1 no
era la métrica sino que en t=0 sólo hay un forward sobre un prompt fijo →
n_efectivo≈1 (CV intra-condición ~1.7%). En t>0 cada token generado aporta un
punto nuevo con variabilidad genuina: n real = 20 prompts × N tokens.

Qué hace:
  - Genera greedy N tokens (default 24) para cada (condición, prompt), con KV
    cache, attn_implementation="eager".
  - Hooks en `self_attn` de las capas objetivo. En cada PASO DE GENERACIÓN
    (query = 1 posición nueva) reduce la fila de atención (1,H,1,K) a métricas
    por cabeza y la descarta. El PREFILL (paso 0, query=T_ctx) se ignora
    (eso es t=0, ya está en la ronda 1).
  - Máscaras de clave por región: system prompt / pregunta de usuario /
    tokens ya generados (esta última crece con cada paso).

Métricas por (condición, prompt, paso_gen, capa, cabeza):
  sysfrac   masa de la fila sobre los tokens del system prompt
  userfrac  masa sobre los tokens de la pregunta de usuario
  genfrac   masa sobre los tokens ya generados (incl. el generation-prompt suffix)
  sysratio  sysfrac / (sysfrac + genfrac)  — de la atención a CONTENIDO (system o
            generado), qué fracción vuelve al system prompt; la competencia real
            durante la generación (userfrac es ~0: la pregunta son ~11 tok de ~3970)
  ent       entropía / log(K)  (K = claves visibles en ese paso)

Salidas (results_attn_gen/):
  <cond>.npz   cada métrica: (nP, N, nL, H) float32
               T_ctx, sys_len, user_len: (nP,) int32 | gen_ids: (nP, N) int32
               prompt_ids, layers, layer_indices, heads, n_gen
  meta.json    modelo, versiones, timings
  <cond>_responses.json  el texto generado por prompt (para sanity)

Uso:
  python run_attention_gen_extraction.py --sanity --token hf_xxx
  python run_attention_gen_extraction.py --token hf_xxx            # 8 condiciones
  python run_attention_gen_extraction.py --conditions axis axis_pec_only soul_md_corto automata_neutro vanilla --token hf_xxx
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

HERE = Path(__file__).parent
HF_MODEL_ID = "google/gemma-4-31B-it"
LOCAL_MODEL_DIR = "/workspace/models/gemma-4-31B-it"

PROMPTS_DIR_CANDIDATES = [Path("/workspace/sia_data/prompts"),
                          HERE.parent.parent / "data" / "sia" / "prompts"]
RESPONSES_FALLBACK = HERE.parent.parent / "data" / "sia_extended_v5" / "vanilla_responses.json"
PROMPTS_JSON_CANDIDATES = ["/workspace/sia_data/prompts.json",
                           str(HERE.parent.parent / "data" / "sia" / "prompts.json")]
OUT_DIR = HERE / "results_attn_gen"

PRIORITY_SUBSET = [1, 3, 6, 10, 14, 21, 23, 27, 31, 39, 41, 45, 51, 59, 61, 65, 71, 79, 91, 98]
MAX_INPUT_TOKENS = 9000
DEFAULT_LAYER_LABELS = [20, 25, 28, 30, 32, 35, 38, 40, 45, 50, 55]
LAYER_OFFSET = -1
N_GEN = 24

CONDITIONS = {
    "axis":                  {"system_prompt_path": "axis.dna"},
    "axis_pec_only":         {"system_prompt_path": "axis_pec_only.txt"},
    "soul_md_corto":         {"system_prompt_path": "soul_md_corto.md"},
    "soul_elena_financial":  {"system_prompt_path": "soul_elena_financial.txt"},
    "soul_solidity_auditor": {"system_prompt_path": "soul_solidity_auditor.txt"},
    "generic_long":          {"system_prompt_path": "generic_long.txt"},
    "vanilla":               {"system_prompt": "You are a helpful assistant."},
    "automata_neutro":       {"system_prompt_path": "automata_neutro.txt"},
}
METRIC_NAMES = ["sysfrac", "userfrac", "genfrac", "sysratio", "ent"]


def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def load_prompts(pj):
    if pj and Path(pj).exists():
        data = json.loads(Path(pj).read_text(encoding="utf-8"))
        by = {p["id"]: p for p in data if p["id"] in PRIORITY_SUBSET}
        if len(by) == len(PRIORITY_SUBSET):
            return [{"id": i, "text": by[i].get("text") or by[i].get("prompt")} for i in PRIORITY_SUBSET]
    data = json.loads(RESPONSES_FALLBACK.read_text(encoding="utf-8"))
    by = {p["id"]: p for p in data}
    return [{"id": i, "text": by[i]["prompt"]} for i in PRIORITY_SUBSET if i in by]


def get_system_prompt(cfg, pdir):
    if "system_prompt" in cfg:
        return cfg["system_prompt"]
    return (pdir / cfg["system_prompt_path"]).read_text(encoding="utf-8")


def render_template(tok, sysp, user):
    msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": user}]
    try:
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if sysp[:40].strip() and sysp[:40].strip() not in txt:
            raise ValueError("template no conservó el system prompt")
        return txt
    except Exception:
        return f"{sysp}\n\nUser: {user}\nAssistant:"


def region_masks(tok, full_text, sysp, user, T):
    """bool (T,) para system y user (sobre claves del contexto). El resto
    (template/suffix) queda fuera de ambas."""
    sysm = np.zeros(T, bool)
    usrm = np.zeros(T, bool)
    try:
        enc = tok(full_text, add_special_tokens=True, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
    except Exception:
        offs = None

    def rng(c0, c1):
        if offs is not None:
            ts = [i for i, (a, b) in enumerate(offs) if a < c1 and b > c0 and b > a]
            return (min(ts), max(ts) + 1) if ts else None
        n0 = len(tok(full_text[:c0], add_special_tokens=False).input_ids)
        n1 = len(tok(full_text[:c1], add_special_tokens=False).input_ids)
        sp = sum(1 for x in tok(full_text, add_special_tokens=True).input_ids[:3]
                 if x in set(getattr(tok, "all_special_ids", []) or []))
        return (min(n0 + sp, T), min(max(n1 + sp, n0 + sp + 1), T))

    sp = full_text.find(sysp[:60].strip()) if sysp[:60].strip() else -1
    if sp >= 0:
        r = rng(sp, sp + len(sysp))
        if r:
            sysm[r[0]:r[1]] = True
    up = full_text.rfind(user[:60])
    if up >= 0:
        r = rng(up, up + len(user))
        if r:
            usrm[r[0]:r[1]] = True
    usrm &= ~sysm
    return sysm, usrm


def get_lm(model):
    for path in ("model.language_model", "model", "language_model"):
        o = model
        try:
            for a in path.split("."):
                o = getattr(o, a)
            if hasattr(o, "layers"):
                return o
        except AttributeError:
            continue
    raise RuntimeError("no encuentro .layers")


def get_model_path(tokn):
    p = Path(LOCAL_MODEL_DIR)
    if p.exists() and any(p.iterdir()):
        return str(p)
    from huggingface_hub import snapshot_download
    kw = dict(repo_id=HF_MODEL_ID, local_dir=str(p), ignore_patterns=["*.gguf", "*.ggml"])
    if tokn:
        kw["token"] = tokn
    return snapshot_download(**kw)


def step_metrics(A, sys_end, usr0, usr1, gen0, K):
    """A: (1,H,1,K) o (H,1,K) o (H,K) — fila de atención del token nuevo.
    Devuelve dict métrica -> (H,) np.float32."""
    a = A
    while a.dim() > 2:
        a = a[0] if a.shape[0] == 1 else a.squeeze(1)
    if a.dim() == 2 and a.shape[0] != 1 and a.shape[-1] == K:
        row = a                      # (H, K)
    else:
        row = a.reshape(-1, K)
    row = row.float().clamp_min(0)
    row = row / row.sum(-1, keepdim=True).clamp_min(1e-12)
    H = row.shape[0]
    logK = float(np.log(max(K, 2)))
    sysf = row[:, :sys_end].sum(-1)
    usrf = row[:, usr0:usr1].sum(-1)
    genf = row[:, gen0:].sum(-1)
    ent = -(row * row.clamp_min(1e-12).log()).sum(-1) / logK
    ratio = sysf / (sysf + genf).clamp_min(1e-9)
    return {"sysfrac": sysf.cpu().numpy(), "userfrac": usrf.cpu().numpy(),
            "genfrac": genf.cpu().numpy(), "sysratio": ratio.cpu().numpy(),
            "ent": ent.cpu().numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=None)
    ap.add_argument("--prompts-json", default=None)
    ap.add_argument("--prompts-dir", default=None)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--layers", default=None)
    ap.add_argument("--layer-offset", type=int, default=LAYER_OFFSET)
    ap.add_argument("--n-gen", type=int, default=N_GEN)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    import transformers
    print(f"torch {torch.__version__} | transformers {transformers.__version__}", flush=True)
    labels = [int(x) for x in args.layers.split(",")] if args.layers else list(DEFAULT_LAYER_LABELS)
    pdir = Path(args.prompts_dir) if args.prompts_dir else first_existing(PROMPTS_DIR_CANDIDATES)
    pj = args.prompts_json or str(first_existing(PROMPTS_JSON_CANDIDATES) or "")
    prompts = load_prompts(pj)
    prompt_ids = [p["id"] for p in prompts]
    N = args.n_gen
    conds = ["axis"] if args.sanity else args.conditions
    sp_list = prompts[:2] if args.sanity else prompts
    print(f"{len(prompts)} prompts | N_gen={N} | conds={conds} | capas={labels}", flush=True)

    device = "cuda:0"
    mp = get_model_path(args.token)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Cargando modelo (bf16, eager)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        mp, dtype=torch.bfloat16, attn_implementation="eager",
        trust_remote_code=True, token=args.token).to(device).eval()
    tok = AutoTokenizer.from_pretrained(mp, token=args.token)
    lm = get_lm(model)
    n_layers = len(lm.layers)
    layer_idx = [l + args.layer_offset for l in labels]
    assert all(0 <= i < n_layers for i in layer_idx)
    try:
        H = int(getattr(model.config.get_text_config(), "num_attention_heads"))
    except Exception:
        H = 32
    eos_ids = set(x for x in ([tok.eos_token_id] + list(getattr(tok, "all_special_ids", []) or []))
                  if isinstance(x, int))
    print(f"n_layers={n_layers} H={H} eos={sorted(eos_ids)[:4]}", flush=True)

    captured = {}
    ctx = {}

    def mk(i):
        def hook(mod, inp, out):
            A = None
            if isinstance(out, (tuple, list)):
                for o in out:
                    if torch.is_tensor(o) and o.dim() >= 3:
                        A = o
                        break
            elif torch.is_tensor(out) and out.dim() >= 3:
                A = out
            if A is None or not ctx or A.shape[-2] != 1:   # sólo pasos de generación (q=1)
                return
            captured[i] = step_metrics(A.detach(), ctx["sys_end"], ctx["usr0"],
                                       ctx["usr1"], ctx["gen0"], A.shape[-1])
        return hook

    handles = [lm.layers[i].self_attn.register_forward_hook(mk(i)) for i in layer_idx]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"experiment": "FASE 6 ronda 2 §1.3 — atención en t>0",
            "model": HF_MODEL_ID, "attn_implementation": "eager", "n_gen": N,
            "layer_labels": labels, "layer_indices": layer_idx, "heads": H,
            "prompt_ids": prompt_ids, "torch": torch.__version__,
            "transformers": transformers.__version__, "timings": {}}
    t_all = time.time()

    for cond in conds:
        cfg = CONDITIONS[cond]
        sysp = get_system_prompt(cfg, pdir)
        nL, nP = len(layer_idx), len(sp_list)
        store = {m: np.full((nP, N, nL, H), np.nan, np.float32) for m in METRIC_NAMES}
        T_ctx = np.zeros(nP, np.int32); sys_len = np.zeros(nP, np.int32)
        usr_len = np.zeros(nP, np.int32); gen_ids = np.full((nP, N), -1, np.int32)
        responses = []
        print(f"\n=== {cond} ({nP} prompts) ===", flush=True)
        tc = time.time()
        for pi, p in enumerate(sp_list):
            full_text = render_template(tok, sysp, p["text"])
            ids = tok(full_text, return_tensors="pt").input_ids
            if ids.shape[1] > MAX_INPUT_TOKENS:
                ids = ids[:, :MAX_INPUT_TOKENS]
            ids = ids.to(device)
            T = int(ids.shape[1])
            sysm, usrm = region_masks(tok, full_text, sysp, p["text"], T)
            sys_end = int(np.where(sysm)[0].max()) + 1 if sysm.any() else 0
            uw = np.where(usrm)[0]
            usr0, usr1 = (int(uw.min()), int(uw.max()) + 1) if len(uw) else (T, T)
            ctx.update(sys_end=sys_end, usr0=usr0, usr1=usr1, gen0=T)
            T_ctx[pi] = T; sys_len[pi] = int(sysm.sum()); usr_len[pi] = int(usrm.sum())

            with torch.no_grad():
                out = model(input_ids=ids, use_cache=True)
                past = out.past_key_values
                nxt = int(out.logits[0, -1].argmax())
            gen_toks = []
            for step in range(N):
                gen_toks.append(nxt)
                gen_ids[pi, step] = nxt
                captured.clear()
                cur = torch.tensor([[nxt]], device=device)
                with torch.no_grad():
                    out = model(input_ids=cur, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                for li, i in enumerate(layer_idx):
                    mm = captured.get(i)
                    if mm is None:
                        continue
                    for m in METRIC_NAMES:
                        store[m][pi, step, li, :len(mm[m])] = mm[m]
                nxt = int(out.logits[0, -1].argmax())
                if nxt in eos_ids:
                    break
            responses.append({"id": p["id"], "text": tok.decode(gen_toks, skip_special_tokens=True)})
            if (pi + 1) % 5 == 0:
                gc.collect(); torch.cuda.empty_cache()
            if pi < 2 or args.sanity:
                sr = np.nanmean(store["sysratio"][pi])
                print(f"  [{pi+1}/{nP}] id={p['id']} T={T} sys_end={sys_end} usr=[{usr0}:{usr1}] "
                      f"gen={len(gen_toks)}tok sysratio~{sr:.3f}", flush=True)
        meta["timings"][cond] = round(time.time() - tc, 1)

        if args.sanity:
            print("\n  sanity — sysratio y ent por capa (media sobre cabezas, prompts, pasos):", flush=True)
            for li, lab in enumerate(labels):
                print(f"    L{lab:>2}: sysratio={np.nanmean(store['sysratio'][:,:,li]):.3f}  "
                      f"sysfrac={np.nanmean(store['sysfrac'][:,:,li]):.3f}  "
                      f"genfrac={np.nanmean(store['genfrac'][:,:,li]):.3f}  "
                      f"ent={np.nanmean(store['ent'][:,:,li]):.3f}", flush=True)
            per = (time.time() - t_all) / nP
            print(f"\n  ETA completa: {len(args.conditions)*len(prompts)} muestras × ~{per:.1f}s "
                  f"≈ {per*len(args.conditions)*len(prompts)/60:.0f} min", flush=True)
            print("SANITY_OK", flush=True)
            for h in handles:
                h.remove()
            return

        np.savez_compressed(OUT_DIR / f"{cond}.npz",
                            T_ctx=T_ctx, sys_len=sys_len, usr_len=usr_len, gen_ids=gen_ids,
                            prompt_ids=np.array(prompt_ids, np.int32),
                            layers=np.array(labels, np.int32),
                            layer_indices=np.array(layer_idx, np.int32),
                            heads=np.int32(H), n_gen=np.int32(N), **store)
        (OUT_DIR / f"{cond}_responses.json").write_text(json.dumps(responses, ensure_ascii=False, indent=1))
        print(f"  guardado results_attn_gen/{cond}.npz | {meta['timings'][cond]}s", flush=True)

    for h in handles:
        h.remove()
    meta["runtime_sec"] = round(time.time() - t_all, 1)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nmeta.json | {meta['runtime_sec']}s\nATTENTION_GEN_EXTRACTION_DONE", flush=True)


if __name__ == "__main__":
    main()
