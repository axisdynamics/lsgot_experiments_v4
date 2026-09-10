#!/usr/bin/env python3
"""
FASE 6 · ronda 2 §5.3 — atención en t>0 con GENERACIÓN POR SAMPLING.

§1.3 con greedy falló su premisa: las trayectorias eran casi idénticas entre los
20 prompts (CV entre prompts 0.009), así que el n efectivo siguió ≈1. El fix
(§5.3 de Experimento_atencion_v2.md): sampling (temp>0) → cada continuación
genera contenido distinto → varianza de trayectoria REAL. Ahora hay n de verdad:
20 prompts × K muestras por condición.

Qué hace:
  - Para cada (condición, prompt): model.generate(do_sample=True, temperature=T,
    top_p=0.95, num_return_sequences=K, max_new_tokens=N), semilla fija por prompt.
  - Hooks en `self_attn` de las capas objetivo. En cada paso de generación
    (query=1) reduce la atención (K,H,1,Kv) a métricas por (muestra, cabeza).
    El prefill (query=T) se ignora.
  - Máscaras de clave: system / pregunta de usuario / tokens ya generados.

Métricas por (condición, prompt, muestra, paso, capa, cabeza):
  sysfrac, userfrac, genfrac, sysratio(=sysfrac/(sysfrac+genfrac)), ent

Salidas (results_attn_gen_samp/):
  <cond>.npz   cada métrica: (nP, K, N, nL, H) float32
               T_ctx, sys_len, usr_len: (nP,) | gen_ids: (nP, K, N) int32
               prompt_ids, layers, layer_indices, heads, n_gen, k_samples, temperature
  <cond>_responses.json   K continuaciones por prompt

Uso:
  python run_attention_gen_sampled.py --sanity --token hf_xxx
  python run_attention_gen_sampled.py --token hf_xxx --k-samples 6 --temperature 0.8
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
OUT_DIR = HERE / "results_attn_gen_samp"

PRIORITY_SUBSET = [1, 3, 6, 10, 14, 21, 23, 27, 31, 39, 41, 45, 51, 59, 61, 65, 71, 79, 91, 98]
MAX_INPUT_TOKENS = 9000
DEFAULT_LAYER_LABELS = [20, 25, 28, 30, 32, 35, 38, 40, 45, 50, 55]
LAYER_OFFSET = -1
N_GEN = 24
K_SAMPLES = 6
TEMPERATURE = 0.8
TOP_P = 0.95
SEED = 42

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
            raise ValueError
        return txt
    except Exception:
        return f"{sysp}\n\nUser: {user}\nAssistant:"


def region_bounds(tok, full_text, sysp, user, T):
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
    sr = rng(sp, sp + len(sysp)) if sp >= 0 else None
    up = full_text.rfind(user[:60])
    ur = rng(up, up + len(user)) if up >= 0 else None
    sys_end = sr[1] if sr else 0
    usr0, usr1 = (ur[0], ur[1]) if ur else (T, T)
    if usr0 < sys_end:               # sin solapamiento
        usr0 = max(usr0, sys_end)
        usr1 = max(usr1, usr0)
    sys_len = sr[1] - sr[0] if sr else 0
    usr_len = usr1 - usr0
    return sys_end, usr0, usr1, sys_len, usr_len


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
    raise RuntimeError("no .layers")


def get_model_path(tokn):
    p = Path(LOCAL_MODEL_DIR)
    if p.exists() and any(p.iterdir()):
        return str(p)
    from huggingface_hub import snapshot_download
    kw = dict(repo_id=HF_MODEL_ID, local_dir=str(p), ignore_patterns=["*.gguf", "*.ggml"])
    if tokn:
        kw["token"] = tokn
    return snapshot_download(**kw)


def step_metrics_batched(A, sys_end, usr0, usr1, gen0):
    """A: (B,H,1,Kv) o (B,H,Kv). Devuelve dict métrica -> (B,H) np.float32."""
    a = A
    if a.dim() == 4:
        a = a[:, :, 0, :]            # (B,H,Kv)  — la query es 1 sola posición
    B, H, Kv = a.shape
    row = a.float().clamp_min(0)
    row = row / row.sum(-1, keepdim=True).clamp_min(1e-12)
    logK = float(np.log(max(Kv, 2)))
    sysf = row[:, :, :sys_end].sum(-1)
    usrf = row[:, :, usr0:usr1].sum(-1)
    genf = row[:, :, gen0:].sum(-1)
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
    ap.add_argument("--k-samples", type=int, default=K_SAMPLES)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"torch {torch.__version__} | transformers {transformers.__version__}", flush=True)
    labels = [int(x) for x in args.layers.split(",")] if args.layers else list(DEFAULT_LAYER_LABELS)
    pdir = Path(args.prompts_dir) if args.prompts_dir else first_existing(PROMPTS_DIR_CANDIDATES)
    pj = args.prompts_json or str(first_existing(PROMPTS_JSON_CANDIDATES) or "")
    prompts = load_prompts(pj)
    prompt_ids = [p["id"] for p in prompts]
    N, K = args.n_gen, args.k_samples
    conds = ["axis"] if args.sanity else args.conditions
    sp_list = prompts[:2] if args.sanity else prompts
    print(f"{len(prompts)} prompts | N_gen={N} K={K} temp={args.temperature} | conds={conds} | capas={labels}", flush=True)

    device = "cuda:0"
    mp = get_model_path(args.token)
    print("Cargando modelo (bf16, eager)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        mp, dtype=torch.bfloat16, attn_implementation="eager",
        trust_remote_code=True, token=args.token).to(device).eval()
    tok = AutoTokenizer.from_pretrained(mp, token=args.token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lm = get_lm(model)
    n_layers = len(lm.layers)
    layer_idx = [l + args.layer_offset for l in labels]
    assert all(0 <= i < n_layers for i in layer_idx)
    try:
        H = int(getattr(model.config.get_text_config(), "num_attention_heads"))
    except Exception:
        H = 32
    print(f"n_layers={n_layers} H={H}", flush=True)

    captured, ctx = {}, {}

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
            if A is None or not ctx or A.shape[-2] != 1:   # sólo pasos de generación
                return
            captured[i] = step_metrics_batched(A.detach(), ctx["sys_end"], ctx["usr0"],
                                               ctx["usr1"], ctx["gen0"])
        return hook

    handles = [lm.layers[i].self_attn.register_forward_hook(mk(i)) for i in layer_idx]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"experiment": "FASE 6 ronda 2 §5.3 — atención t>0 con sampling",
            "model": HF_MODEL_ID, "attn_implementation": "eager", "n_gen": N,
            "k_samples": K, "temperature": args.temperature, "top_p": TOP_P, "seed": SEED,
            "layer_labels": labels, "layer_indices": layer_idx, "heads": H,
            "prompt_ids": prompt_ids, "torch": torch.__version__,
            "transformers": transformers.__version__, "timings": {}}
    t_all = time.time()

    for cond in conds:
        cfg = CONDITIONS[cond]
        sysp = get_system_prompt(cfg, pdir)
        nL, nP = len(layer_idx), len(sp_list)
        store = {m: np.full((nP, K, N, nL, H), np.nan, np.float32) for m in METRIC_NAMES}
        T_ctx = np.zeros(nP, np.int32); sys_len = np.zeros(nP, np.int32)
        usr_len = np.zeros(nP, np.int32); gen_ids = np.full((nP, K, N), -1, np.int32)
        responses = []
        print(f"\n=== {cond} ({nP} prompts × {K}) ===", flush=True)
        tc = time.time()
        for pi, p in enumerate(sp_list):
            full_text = render_template(tok, sysp, p["text"])
            ids = tok(full_text, return_tensors="pt").input_ids
            if ids.shape[1] > MAX_INPUT_TOKENS:
                ids = ids[:, :MAX_INPUT_TOKENS]
            ids = ids.to(device)
            T = int(ids.shape[1])
            sys_end, usr0, usr1, sl, ul = region_bounds(tok, full_text, sysp, p["text"], T)
            ctx.update(sys_end=sys_end, usr0=usr0, usr1=usr1, gen0=T)
            T_ctx[pi] = T; sys_len[pi] = sl; usr_len[pi] = ul

            # prefill batch=1 (barato: el hook lo ignora, q≠1), luego se replica el
            # KV cache a batch K y se generan K continuaciones en paralelo con
            # top-p sampling independiente por fila. (Prefill batcheado OOM: el
            # softmax eager (K,32,T,T) f32 pide ~12 GiB a K=6.)
            transformers.set_seed(SEED * 1000 + int(p["id"]))
            with torch.no_grad():
                o0 = model(input_ids=ids, use_cache=True)
                pk = o0.past_key_values
                for lyr in pk.layers:
                    lyr.keys = lyr.keys.repeat(K, 1, 1, 1)
                    lyr.values = lyr.values.repeat(K, 1, 1, 1)
                nxt = _sample_batch(o0.logits[:, -1, :].expand(K, -1), args.temperature, TOP_P)   # (K,)
            alive = torch.ones(K, dtype=torch.bool, device=device)
            toks_k = [[] for _ in range(K)]
            for step in range(N):
                for k in range(K):
                    if alive[k]:
                        toks_k[k].append(int(nxt[k]))
                        gen_ids[pi, k, step] = int(nxt[k])
                captured.clear()
                with torch.no_grad():
                    o = model(input_ids=nxt[:, None], past_key_values=pk, use_cache=True)
                    pk = o.past_key_values
                for li, i in enumerate(layer_idx):
                    mm = captured.get(i)
                    if mm is None:
                        continue
                    for m in METRIC_NAMES:
                        store[m][pi, :, step, li, :mm[m].shape[-1]] = mm[m]
                nxt = _sample_batch(o.logits[:, -1, :], args.temperature, TOP_P)
                alive &= nxt != tok.eos_token_id
                if not alive.any():
                    break
            for k in range(K):
                responses.append({"id": p["id"], "k": k,
                                  "text": tok.decode(toks_k[k], skip_special_tokens=True)})
            if (pi + 1) % 5 == 0:
                gc.collect(); torch.cuda.empty_cache()
            if pi < 2 or args.sanity:
                sr = np.nanmean(store["sysratio"][pi])
                gv = np.nanstd(np.nanmean(store["sysratio"][pi], axis=(1, 2, 3)))
                print(f"  [{pi+1}/{nP}] id={p['id']} T={T} sys_end={sys_end} usr=[{usr0}:{usr1}] "
                      f"sysratio~{sr:.3f} std_entre_muestras~{gv:.4f}", flush=True)
        meta["timings"][cond] = round(time.time() - tc, 1)

        if args.sanity:
            print("\n  sanity — sysratio por capa (media sobre cabezas/prompts/muestras/pasos) "
                  "y std ENTRE MUESTRAS:", flush=True)
            for li, lab in enumerate(labels):
                v = store["sysratio"][:, :, :, li, :]
                per_samp = np.nanmean(v, axis=(2, 3))          # (nP, K)
                std_s = np.nanmean(np.nanstd(per_samp, axis=1))
                print(f"    L{lab:>2}: sysratio={np.nanmean(v):.3f}  std_entre_muestras={std_s:.4f}  "
                      f"ent={np.nanmean(store['ent'][:,:,:,li,:]):.3f}", flush=True)
            per = (time.time() - t_all) / nP
            print(f"\n  ETA completa: {len(args.conditions)} cond × {len(prompts)} prompts "
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
                            heads=np.int32(H), n_gen=np.int32(N), k_samples=np.int32(K),
                            temperature=np.float32(args.temperature), **store)
        (OUT_DIR / f"{cond}_responses.json").write_text(json.dumps(responses, ensure_ascii=False, indent=1))
        print(f"  guardado results_attn_gen_samp/{cond}.npz | {meta['timings'][cond]}s", flush=True)

    for h in handles:
        h.remove()
    meta["runtime_sec"] = round(time.time() - t_all, 1)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nmeta.json | {meta['runtime_sec']}s\nATTENTION_GEN_SAMPLED_DONE", flush=True)


def _sample_batch(logits, temp, top_p):
    """logits: (B,V) -> (B,) token ids por top-p (nucleus) sampling independiente por fila."""
    if temp <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / temp, dim=-1)          # (B,V)
    sp, si = torch.sort(probs, descending=True, dim=-1)
    csum = torch.cumsum(sp, dim=-1)
    mask = csum - sp > top_p                                       # descarta la cola
    sp = sp.masked_fill(mask, 0.0)
    sp = sp / sp.sum(-1, keepdim=True).clamp_min(1e-12)
    pick = torch.multinomial(sp, 1)                               # (B,1) índice en el orden ordenado
    return si.gather(-1, pick)[:, 0]


if __name__ == "__main__":
    main()
