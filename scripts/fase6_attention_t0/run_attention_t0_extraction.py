#!/usr/bin/env python3
"""
FASE 6 — ¿el testigo cableado modula los PESOS DE ATENCIÓN en t=0?

Propuesta: Descargas/Experimento_atencion.md (2026-09-07). La ronda de v̂
cerró que la identidad ya está en el hidden state de t=0 (proyección sobre
v̂, d hasta +9.78). Pregunta nueva y distinta: ¿ese desplazamiento viene
acompañado de una reponderación de la ATENCIÓN, o es sólo un sesgo aditivo
en el residual stream? Toda la instrumentación previa del proyecto engancha
la SALIDA de cada capa decoder (residual); NADIE miró los pesos de atención
(verificado 2026-09-07, §2·bis.1 de la propuesta).

Qué hace este script:
  - Un único forward de contexto puro por (condición, prompt) — t=0, sin
    generar ningún token (use_cache=False, output_hidden_states=False).
  - attn_implementation="eager" (SDPA/flash devuelven None en los pesos).
    Precedente: extract_layers_jlens.py ya corre este modelo en eager.
  - Hooks SOLO en `self_attn` de las capas objetivo (banda de v̂: L20–L55,
    denso alrededor de la transición L30→L35). El tensor (1,H,T,T) de cada
    capa se reduce a métricas (H,) dentro del hook y se descarta en el acto
    → pico de memoria ≈ 1 capa, no las 60 (§5.1 de la propuesta).
  - Guarda sólo vectores resumidos por (condición, prompt, capa, cabeza).

Métricas por (condición, prompt, capa, cabeza):
  ent_last      entropía (nats) de la atención de la ÚLTIMA posición de query
                (la que produce el primer token) sobre las claves 0..T-1
  ent_rowmean   media, sobre todas las filas de query, de la entropía por fila
  sink_last     masa de la última fila sobre la posición 0 (sink de atención)
  sysfrac_last  masa de la última fila sobre TODOS los tokens del system prompt
  wfrac_last    masa de la última fila sobre los tokens del span del testigo
  wfrac_mean    media sobre filas de query de la masa sobre el span del testigo
  cfrac_last    ídem wfrac_last, sobre un slice de control de la misma longitud
                (tokens del system prompt fuera del span, contiguos, justo antes)
  cfrac_mean    ídem wfrac_mean para el control
  wenrich_last  wfrac_last / (span_len / T)   — enriquecimiento vs tamaño del span
  cenrich_last  cfrac_last / (ctrl_len / T)
Las métricas w*/c* son NaN en las condiciones sin span (por diseño).

Salidas (results_attn_t0/):
  <cond>.npz    cada métrica: (n_prompts, n_layers, H) float32
                span_len, ctrl_len, T, n_sys_tokens: (n_prompts,) int32
                prompt_ids: (n_prompts,) | layers: labels | layer_indices |
                heads: int | logT: (n_prompts,)
  meta.json     modelo, versiones, spans encontrados por condición (marcador →
                nº de tokens), método de mapeo de spans, convención de capas,
                timings

Uso:
  python run_attention_t0_extraction.py --sanity --token hf_xxx
  python run_attention_t0_extraction.py --token hf_xxx
  python run_attention_t0_extraction.py --conditions axis vanilla --token hf_xxx
  python run_attention_t0_extraction.py --layers 20,25,28,30,32,35,38,40,45,50,55 --token hf_xxx

Después: analyze_attention_t0.py sobre results_attn_t0/ (local, sin GPU).
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
LOCAL_MODEL_DIR = "/workspace/models/gemma-4-31B-it"      # convención de run_perturbation_t2.py / jlens

PROMPTS_JSON_CANDIDATES = [
    "/workspace/sia_data/prompts.json",
    str(HERE.parent.parent / "data" / "sia" / "prompts.json"),
]
PROMPTS_DIR_CANDIDATES = [
    Path("/workspace/sia_data/prompts"),
    HERE.parent.parent / "data" / "sia" / "prompts",
]
# fallback final para el texto de los prompts si no hay prompts.json en el pod:
RESPONSES_FALLBACK = HERE.parent.parent / "data" / "sia_extended_v5" / "vanilla_responses.json"

SPANS_JSON = HERE / "witness_spans.json"
OUT_DIR = HERE / "results_attn_t0"

PRIORITY_SUBSET = [1, 3, 6, 10, 14, 21, 23, 27, 31, 39, 41, 45, 51, 59, 61, 65, 71, 79, 91, 98]
MAX_INPUT_TOKENS = 9000     # igual que jlens / PerturbationExtractor (no liga con estos prompts)

# "L_N" -> índice de módulo: convención del proyecto "L_N -> índice N-1"
# (run_qwen3_extraction.py:61). El perfil de v̂ (Experimento_atencion.md §2·bis.2)
# tiene dip en L30 y pico en L35; se muestrea denso alrededor de la transición.
# La ambigüedad ±1 capa (el script de ef2 original está perdido) queda absorbida
# por el muestreo denso L28/30/32/35/38.
DEFAULT_LAYER_LABELS = [20, 25, 28, 30, 32, 35, 38, 40, 45, 50, 55]
LAYER_OFFSET = -1

# Panel de §3.1 de la propuesta (primera pasada — se omiten axis_short y
# soul_jarvis a propósito; jarvis tiene la sospecha léxica sin cerrar).
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

METRIC_NAMES = ["ent_last", "ent_rowmean", "sink_last", "sysfrac_last",
                "wfrac_last", "wfrac_mean", "cfrac_last", "cfrac_mean",
                "wenrich_last", "cenrich_last"]


# ─────────────────────────── carga de datos ───────────────────────────

def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def load_prompts(prompts_json):
    if prompts_json and Path(prompts_json).exists():
        data = json.loads(Path(prompts_json).read_text(encoding="utf-8"))
        by_id = {p["id"]: p for p in data if p["id"] in PRIORITY_SUBSET}
        out = [by_id[i] for i in PRIORITY_SUBSET if i in by_id]
        if len(out) == len(PRIORITY_SUBSET):
            return [{"id": p["id"], "text": p.get("text") or p.get("prompt")} for p in out]
        print(f"  [warn] {prompts_json} sólo tiene {len(out)}/{len(PRIORITY_SUBSET)} "
              f"del subset — probando fallback", flush=True)
    if RESPONSES_FALLBACK.exists():
        data = json.loads(RESPONSES_FALLBACK.read_text(encoding="utf-8"))
        by_id = {p["id"]: p for p in data}
        return [{"id": i, "text": by_id[i]["prompt"]} for i in PRIORITY_SUBSET if i in by_id]
    raise FileNotFoundError(
        "no encuentro prompts.json ni el fallback vanilla_responses.json — "
        "pasar --prompts-json")


def get_system_prompt(cfg, prompts_dir):
    if "system_prompt" in cfg:
        return cfg["system_prompt"]
    return (prompts_dir / cfg["system_prompt_path"]).read_text(encoding="utf-8")


def render_template(tokenizer, system_prompt, user_text):
    """Mismo render que jlens/PerturbationExtractor: chat template + generation
    prompt. Devuelve (full_text, used_fallback)."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        if system_prompt[:40].strip() and system_prompt[:40].strip() not in text:
            raise ValueError("el template no conservó el system prompt literal")
        return text, False
    except Exception as e:
        print(f"    [warn] chat template: {type(e).__name__}: {e} — usando fallback manual", flush=True)
        return f"{system_prompt}\n\nUser: {user_text}\nAssistant:", True


def char_to_token_index(tokenizer, full_text, char_idx, n_prefix_special):
    """nº de tokens en full_text[:char_idx] (sin especiales) + offset de especiales."""
    n = len(tokenizer(full_text[:char_idx], add_special_tokens=False).input_ids)
    return n_prefix_special + n


def build_key_masks(tokenizer, full_text, system_prompt, spans, T, ids_list):
    """Devuelve (witness_mask, control_mask, sys_mask) bool (T,) sobre claves,
    y un dict {marcador: nº tokens} de lo que se encontró.

    Estrategia primaria: offsets del tokenizer rápido. Fallback: conteo
    incremental de tokens por prefijo de texto (fuzz de ±1-2 tokens en los
    bordes de span por merges de BPE — despreciable para spans de 50-400 tok)."""
    witness = np.zeros(T, dtype=bool)
    control = np.zeros(T, dtype=bool)
    sysm = np.zeros(T, dtype=bool)
    found = {}

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    n_prefix_special = 0
    for tid in ids_list:
        if tid in special_ids:
            n_prefix_special += 1
        else:
            break

    offsets = None
    try:
        enc = tokenizer(full_text, add_special_tokens=True, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
    except Exception:
        offsets = None

    def span_token_range(c0, c1):
        if offsets is not None:
            toks = [i for i, (a, b) in enumerate(offsets)
                    if a < c1 and b > c0 and b > a]
            if toks:
                return min(toks), max(toks) + 1
            return None
        t0 = char_to_token_index(tokenizer, full_text, c0, n_prefix_special)
        t1 = char_to_token_index(tokenizer, full_text, c1, n_prefix_special)
        return (min(t0, T), min(max(t1, t0 + 1), T))

    # system prompt completo
    s_head = system_prompt[:60].strip()
    s_pos = full_text.find(s_head) if s_head else -1
    if s_pos >= 0:
        rng = span_token_range(s_pos, s_pos + len(system_prompt))
        if rng:
            sysm[rng[0]:rng[1]] = True

    # spans del testigo
    witness_char_ranges = []
    for start_sub, end_sub in (spans or []):
        s = full_text.find(start_sub)
        if s < 0:
            found[start_sub[:48]] = 0
            print(f"    [warn] marcador NO encontrado: {start_sub[:60]!r}", flush=True)
            continue
        e = full_text.find(end_sub, s)
        if e < 0:
            found[start_sub[:48]] = 0
            print(f"    [warn] end-marker NO encontrado: {end_sub[:60]!r}", flush=True)
            continue
        e += len(end_sub)
        rng = span_token_range(s, e)
        if rng:
            witness[rng[0]:rng[1]] = True
            witness_char_ranges.append((s, e))
            found[start_sub[:48]] = int(rng[1] - rng[0])

    # slice de control: mismos nº de tokens que el span del testigo, tomados del
    # system prompt, fuera del span, contiguos justo antes del primer span.
    n_w = int(witness.sum())
    if n_w > 0 and witness_char_ranges:
        first_s = min(r[0] for r in witness_char_ranges)
        rng = span_token_range(s_pos if s_pos >= 0 else 0, first_s)
        if rng:
            lo, hi = rng
            cand = [i for i in range(lo, hi) if not witness[i]]
            control[cand[-n_w:]] = True
        if control.sum() < n_w and s_pos >= 0:  # relleno: desde el inicio del system
            rng2 = span_token_range(s_pos, s_pos + len(system_prompt))
            if rng2:
                cand = [i for i in range(rng2[0], rng2[1]) if not witness[i] and not control[i]]
                need = n_w - int(control.sum())
                control[cand[:need]] = True

    return witness, control, sysm, found


# ─────────────────────────── modelo ───────────────────────────

def get_model_path(hf_token=None):
    local = Path(LOCAL_MODEL_DIR)
    if local.exists() and any(local.iterdir()):
        print(f"Modelo en {local}", flush=True)
        return str(local)
    print(f"Modelo no encontrado en {local} — descargando {HF_MODEL_ID} "
          f"(~62GB, ~15-20 min)...", flush=True)
    from huggingface_hub import snapshot_download
    kwargs = {"repo_id": HF_MODEL_ID, "ignore_patterns": ["*.gguf", "*.ggml"],
              "local_dir": str(local)}
    if hf_token:
        kwargs["token"] = hf_token
    return snapshot_download(**kwargs)


def get_language_model(model):
    for path in ("model.language_model", "model", "language_model"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, "layers"):
                return obj
        except AttributeError:
            continue
    raise RuntimeError("no encuentro el stack de capas (.layers) del modelo")


# ─────────────────────── métricas por capa (hook) ───────────────────────

def attn_metrics(A, witness, control, sysm, T, device, head_chunk=8):
    """A: pesos de atención (1,H,T,T) o (H,T,T). Devuelve dict métrica -> (H,)
    np.float32. Vectorizado por bloques de cabezas (evita la tormenta de syncs
    GPU→CPU del bucle por cabeza; el bloque de (chunk,T,T) f32 es el pico)."""
    if A.dim() == 4:
        A = A[0]
    H = A.shape[0]
    logT = float(np.log(max(T, 2)))
    wcol = torch.as_tensor(np.ascontiguousarray(witness), device=device)
    ccol = torch.as_tensor(np.ascontiguousarray(control), device=device)
    scol = torch.as_tensor(np.ascontiguousarray(sysm), device=device)
    n_w, n_c = float(witness.sum()), float(control.sum())
    has_w, has_c, has_s = n_w > 0, n_c > 0, bool(sysm.any())

    out = {m: np.full(H, np.nan, dtype=np.float32) for m in METRIC_NAMES}
    for h0 in range(0, H, head_chunk):
        h1 = min(h0 + head_chunk, H)
        Ac = A[h0:h1].float()                          # (c,T,T)
        last = Ac[:, -1, :].clamp_min(0)
        last = last / last.sum(-1, keepdim=True).clamp_min(1e-12)   # (c,T)
        out["ent_last"][h0:h1] = (
            (-(last * last.clamp_min(1e-12).log()).sum(-1)) / logT).cpu().numpy()
        rowent = -(Ac * Ac.clamp_min(1e-12).log()).sum(-1)          # (c,T)
        out["ent_rowmean"][h0:h1] = (rowent.mean(-1) / logT).cpu().numpy()
        out["sink_last"][h0:h1] = last[:, 0].cpu().numpy()
        if has_s:
            out["sysfrac_last"][h0:h1] = last[:, scol].sum(-1).cpu().numpy()
        if has_w:
            wl = last[:, wcol].sum(-1)
            out["wfrac_last"][h0:h1] = wl.cpu().numpy()
            out["wfrac_mean"][h0:h1] = Ac[:, :, wcol].sum(-1).mean(-1).cpu().numpy()
            out["wenrich_last"][h0:h1] = (wl / (n_w / T)).cpu().numpy()
        if has_c:
            cl = last[:, ccol].sum(-1)
            out["cfrac_last"][h0:h1] = cl.cpu().numpy()
            out["cfrac_mean"][h0:h1] = Ac[:, :, ccol].sum(-1).mean(-1).cpu().numpy()
            out["cenrich_last"][h0:h1] = (cl / (n_c / T)).cpu().numpy()
        del Ac, rowent, last
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", type=str, default=None, help="HF token")
    ap.add_argument("--prompts-json", type=str, default=None)
    ap.add_argument("--prompts-dir", type=str, default=None)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()))
    ap.add_argument("--layers", type=str, default=None,
                    help="labels 'L_N' separados por coma (default: "
                         + ",".join(map(str, DEFAULT_LAYER_LABELS)) + ")")
    ap.add_argument("--layer-offset", type=int, default=LAYER_OFFSET,
                    help="idx_modulo = label + offset (default -1, convención del proyecto)")
    ap.add_argument("--sanity", action="store_true",
                    help="axis + 2 prompts: T, spans, H, entropías de muestra, ETA; no guarda")
    args = ap.parse_args()

    import transformers
    print(f"torch {torch.__version__} | transformers {transformers.__version__}", flush=True)

    labels = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(DEFAULT_LAYER_LABELS))
    prompts_json = args.prompts_json or str(first_existing(PROMPTS_JSON_CANDIDATES) or "")
    prompts_dir = (Path(args.prompts_dir) if args.prompts_dir
                   else first_existing(PROMPTS_DIR_CANDIDATES))
    if prompts_dir is None:
        raise FileNotFoundError("no encuentro el directorio de prompts — pasar --prompts-dir")
    spans_all = json.loads(SPANS_JSON.read_text(encoding="utf-8"))

    prompts = load_prompts(prompts_json)
    prompt_ids = [p["id"] for p in prompts]
    print(f"{len(prompts)} prompts | condiciones={args.conditions}", flush=True)
    print(f"capas (labels)={labels} offset={args.layer_offset} "
          f"-> idx={[l + args.layer_offset for l in labels]}", flush=True)

    device = "cuda:0"
    model_path = get_model_path(args.token)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Cargando modelo (bf16, eager)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager",
        trust_remote_code=True, token=args.token,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=args.token)

    lm = get_language_model(model)
    n_layers = len(lm.layers)
    layer_idx = [l + args.layer_offset for l in labels]
    assert all(0 <= i < n_layers for i in layer_idx), \
        f"labels {labels} + offset {args.layer_offset} fuera de rango (n_layers={n_layers})"
    # forzar eager también a nivel de módulo, por si el config no propagó
    for i in layer_idx:
        for m in lm.layers[i].modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = "eager"
    def _cfg_heads():
        cfgs = []
        if hasattr(model.config, "get_text_config"):
            try:
                cfgs.append(model.config.get_text_config())
            except Exception:
                pass
        cfgs.append(model.config)
        for cc in cfgs:
            try:
                v = getattr(cc, "num_attention_heads", None)
                if v:
                    return int(v)
            except Exception:
                continue
        return 0
    H = _cfg_heads()   # Gemma-4-31B: 32 query heads, uniforme (GQA: kv 16/4 se
                       # repite a 32 en el path eager -> attn_weights (1,32,T,T))
    H_ALLOC = H if H > 0 else 64   # cota para preasignar; se recorta al guardar
    print(f"n_layers={n_layers} | num_attention_heads(config)={H}", flush=True)

    # ── hooks: REDUCEN (1,H,T,T) a métricas (H,) DENTRO del hook y descartan el
    #    tensor en el acto — pico de memoria ≈ 1 capa, no las 11 acumuladas
    #    (§5.1 de la propuesta). Las máscaras de span se calculan antes del
    #    forward y se dejan en hook_ctx para que el hook las lea. ──
    captured = {}                       # i -> dict métrica -> (H,)  (ya reducido)
    hook_ctx = {}                       # witness/control/sysm/T/device del prompt en curso

    def make_hook(i):
        def hook(module, inputs, output):
            A = None
            if isinstance(output, (tuple, list)):
                for o in output:
                    if torch.is_tensor(o) and o.dim() == 4:
                        A = o
                        break
            elif torch.is_tensor(output) and output.dim() == 4:
                A = output
            if A is None or not hook_ctx:
                captured[i] = None
                return
            captured[i] = attn_metrics(A.detach(), hook_ctx["witness"], hook_ctx["control"],
                                       hook_ctx["sysm"], hook_ctx["T"], hook_ctx["device"])
        return hook

    handles = [lm.layers[i].self_attn.register_forward_hook(make_hook(i)) for i in layer_idx]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment": "FASE 6 — atención en t=0 vs testigo cableado (Experimento_atencion.md)",
        "model": HF_MODEL_ID, "attn_implementation": "eager",
        "n_layers": n_layers, "num_attention_heads_config": H,
        "layer_labels": labels, "layer_offset": args.layer_offset,
        "layer_indices": layer_idx,
        "layer_convention": "L_N -> módulo layers[N + offset]; offset=-1 = convención "
                            "del proyecto (run_qwen3_extraction.py:61). El script de ef2 "
                            "original está perdido; muestreo denso L28-38 absorbe el ±1.",
        "prompt_ids": prompt_ids, "priority_subset": PRIORITY_SUBSET,
        "span_mapping": "offsets del fast tokenizer si están; si no, conteo incremental "
                        "por prefijo (fuzz ±1-2 tok en bordes de span).",
        "spans_found": {}, "template_fallback": {}, "T_per_prompt": {},
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    t_all = time.time()

    conds = ["axis"] if args.sanity else args.conditions
    sample_prompts = prompts[:2] if args.sanity else prompts

    for cond in conds:
        cfg = CONDITIONS[cond]
        sys_p = get_system_prompt(cfg, prompts_dir)
        spans = spans_all.get(cond, [])
        has_span = bool(spans)
        nL, nP = len(layer_idx), len(sample_prompts)

        store = {m: np.full((nP, nL, H_ALLOC), np.nan, dtype=np.float32) for m in METRIC_NAMES}
        H_seen = H if H > 0 else 0
        span_len = np.zeros(nP, dtype=np.int32)
        ctrl_len = np.zeros(nP, dtype=np.int32)
        Tarr = np.zeros(nP, dtype=np.int32)
        n_sys = np.zeros(nP, dtype=np.int32)
        logT = np.zeros(nP, dtype=np.float32)
        meta["spans_found"][cond] = {}

        print(f"\n=== {cond} ({nP} prompts, span={'sí' if has_span else 'no'}) ===", flush=True)
        for pi, p in enumerate(sample_prompts):
            t_p = time.time()
            full_text, used_fb = render_template(tokenizer, sys_p, p["text"])
            meta["template_fallback"][cond] = used_fb
            enc = tokenizer(full_text, return_tensors="pt")
            ids = enc.input_ids
            if ids.shape[1] > MAX_INPUT_TOKENS:
                ids = ids[:, :MAX_INPUT_TOKENS]
            ids = ids.to(device)
            T = int(ids.shape[1])
            ids_list = ids[0].tolist()

            witness, control, sysm, found = build_key_masks(
                tokenizer, full_text, sys_p, spans, T, ids_list)
            for k, v in found.items():
                meta["spans_found"][cond][k] = v
            span_len[pi] = int(witness.sum())
            ctrl_len[pi] = int(control.sum())
            Tarr[pi] = T
            n_sys[pi] = int(sysm.sum())
            logT[pi] = float(np.log(max(T, 2)))

            captured.clear()
            hook_ctx.update(witness=witness, control=control, sysm=sysm, T=T, device=device)
            with torch.no_grad():
                model(input_ids=ids, use_cache=False, output_hidden_states=False,
                      output_attentions=False)
            hook_ctx.clear()

            for li, i in enumerate(layer_idx):
                mm = captured.get(i)
                if mm is None:
                    if pi == 0 and li == 0:
                        print(f"    [ERROR] la capa {i} no devolvió pesos de atención — "
                              f"¿eager activo? revisá transformers", flush=True)
                    continue
                for m in METRIC_NAMES:
                    store[m][pi, li, :len(mm[m])] = mm[m]
                H_seen = max(H_seen, len(mm[next(iter(METRIC_NAMES))]))
            captured.clear()

            wl = np.nanmean(store["wfrac_last"][pi]) if has_span else float("nan")
            el = np.nanmean(store["ent_last"][pi])
            print(f"  [{pi+1}/{nP}] id={p['id']} T={T} span_tok={span_len[pi]} "
                  f"ctrl_tok={ctrl_len[pi]} | ent_last~{el:.3f} wfrac_last~{wl:.4f} "
                  f"({time.time()-t_p:.1f}s)", flush=True)

            if (pi + 1) % 5 == 0:
                gc.collect(); torch.cuda.empty_cache()

        meta["T_per_prompt"][cond] = Tarr.tolist()

        if args.sanity:
            print("\n  --- sanity: entropía normalizada por capa (media sobre cabezas y prompts) ---", flush=True)
            for li, lab in enumerate(labels):
                print(f"    L{lab:>2}: ent_last={np.nanmean(store['ent_last'][:, li]):.3f}  "
                      f"ent_rowmean={np.nanmean(store['ent_rowmean'][:, li]):.3f}  "
                      f"wenrich_last={np.nanmean(store['wenrich_last'][:, li]):.2f}  "
                      f"cenrich_last={np.nanmean(store['cenrich_last'][:, li]):.2f}", flush=True)
            per_prompt_s = (time.time() - t_all) / nP
            n_total = len(args.conditions) * len(prompts)
            print(f"\n  ETA corrida completa: {n_total} muestras × ~{per_prompt_s:.1f}s "
                  f"≈ {per_prompt_s * n_total / 60:.0f} min", flush=True)
            print("  spans_found:", json.dumps(meta["spans_found"][cond], ensure_ascii=False), flush=True)
            print("SANITY_OK", flush=True)
            for hd in handles:
                hd.remove()
            return

        hh = H_seen if H_seen > 0 else H_ALLOC
        np.savez_compressed(
            OUT_DIR / f"{cond}.npz",
            span_len=span_len, ctrl_len=ctrl_len, T=Tarr, n_sys_tokens=n_sys, logT=logT,
            prompt_ids=np.array(prompt_ids, dtype=np.int32),
            layers=np.array(labels, dtype=np.int32),
            layer_indices=np.array(layer_idx, dtype=np.int32),
            heads=np.int32(hh), has_span=np.bool_(has_span),
            **{m: store[m][:, :, :hh] for m in METRIC_NAMES},
        )
        print(f"  guardado results_attn_t0/{cond}.npz", flush=True)

    for hd in handles:
        hd.remove()
    meta["runtime_sec"] = round(time.time() - t_all, 1)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nmeta.json guardado | {meta['runtime_sec']}s", flush=True)
    print("ATTENTION_T0_EXTRACTION_DONE", flush=True)


if __name__ == "__main__":
    main()
