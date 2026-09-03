#!/usr/bin/env python3
"""
Extracción GPU t=0 por capa + Jacobianas para el J-lens (verbalización por capas).

Protocolo de SOUL_MD_UPDATE_GUIDE.md §8 y pre-registro A1 §4.1
(preregistro_A1_jlens_L21.md): z := residual stream post-norma-final; para cada
muestra (condición, prompt), J_ℓ = ∂z[t=0]/∂h_ℓ[t=0] computado por VJPs chunked
(un backward por chunk de cotangentes da las filas de la Jacobiana); J̄_ℓ =
promedio elemento a elemento sobre las muestras.
Lente: readout(h) = W_U @ (J̄_ℓ @ h_ℓ[t=0]).

PROMEDIO POOLED: J̄ se promedia sobre las 140 muestras (7 condiciones × 20
prompts), NO por condición — la lente es un único instrumento común a todas las
condiciones (un J̄ por condición inyectaría la señal de condición dentro de la
propia lente y confundiría la comparación entre condiciones). Con --per-condition
se guardan además los J̄ por condición (48GB, solo si el análisis local los pide).

Además del J-lens con J̄ (análisis local), se guardan readouts JVP por
(prompt, capa) con la Jacobiana EXACTA de cada muestra (torch.func.jvp sobre el
mismo grafo reducido): el contraste J̄-pooled vs J-per-prompt mide cuánto cambia
el readout al sustituir la Jacobiana exacta por la promediada.

Truco de costo: las VJPs se computan sobre un GRAFO REDUCIDO de 1 posición:
x_ℓ = h_ℓ[t] como hoja (f32), cache K/V de las posiciones 0..t-1 recortada y
congelada (constantes), y las capas ℓ+1..59 llamadas como módulos reales con
use_cache=True. Cada grafo reducido se valida contra el forward original
(identidad del primal: z_red == final_norm(h_59[t]) original, tolerancia
relativa) — si falla, se reintenta con máscara de ventana deslizante explícita
por capa, y si vuelve a fallar, ABORTA. Sin el grafo reducido, las VJPs sobre
la secuencia completa costarían ~100× más y requerirían cotangentes (T, D, C)
imposibles de materializar (T≈6K).

Verificaciones (grabadas en meta.json):
  1. argmax(W_U @ y) == argmax(logits lm_head) por prompt — valida que y es
     exactamente el estado que la lm_head consume (debe ser 20/20).
  2. Identidad del primal del grafo reducido por (prompt, capa) — ABORTA si
     supera --primal-tol.
  3. Chequeo analítico de J_59 (Jacobiana de la RMSNorm final, forma cerrada)
     en --sanity: valida la maquinaria VJP completa.
El cotejo final vs el primer token REAL generado (responses.json del pipeline
sia_extended_v5) se hace localmente en analyze_j_lens.py.

Salidas (results_jlens/):
  states/<cond>.npz    t0_states (20,60,D) f32 | y post-final-norm (20,D) f32 |
                       lm_logits full (20,V) f16 | lm_top10_ids (20,10) i4 |
                       lm_top10_logprobs (20,10) f16 | first_token_ids (20) i4 |
                       T (20) i4 | prompt_ids (20) i4
  jvp/<cond>.npz       (si no se pasa --no-jvp): jvp_top50_ids (20,60,50) i4 |
                       jvp_top50_logprobs (20,60,50) f16 |
                       jvp_first_tok_logprob (20,60) f16
  jacobians/J_L{ℓ}.npy J̄_ℓ pooled f32 (D,D), 60 archivos (~6.9GB total)
  meta.json            protocolo, verificaciones, timings, versiones

Uso:
  python extract_layers_jlens.py --sanity --token hf_xxx   # 1 prompt: chequeos + ETA, sin guardar
  python extract_layers_jlens.py --token hf_xxx            # corrida completa
  python extract_layers_jlens.py --conditions axis vanilla --no-jvp --token hf_xxx

Entorno del pod: mismo que run_perturbation_t2.py (desinstalar
torchvision/torchaudio — ver SOUL_MD_UPDATE_GUIDE.md). Transformers ≥ 4.49,
torch ≥ 2.4. Modelo: google/gemma-4-31B-it, BF16, una GPU ≥ 80GB.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
HF_MODEL_ID = "google/gemma-4-31B-it"
LOCAL_MODEL_DIR = "/workspace/models/gemma-4-31B-it"  # misma convención que run_perturbation_t2.py

PROMPTS_JSON = "/workspace/sia_data/prompts.json"
PROMPTS_DIR = Path("/workspace/sia_data/prompts")
OUT_DIR = HERE / "results_jlens"

PRIORITY_SUBSET = [1, 3, 6, 10, 14, 21, 23, 27, 31, 39, 41, 45, 51, 59, 61, 65, 71, 79, 91, 98]
MAX_INPUT_TOKENS = 9000  # mismo que PerturbationExtractor (nunca liga con estos prompts)

# Condiciones limpias del panel (run_perturbation.py, variante sia) — 7 celdas.
CONDITIONS = {
    "axis":            {"system_prompt_path": "axis.dna"},
    "axis_short":      {"system_prompt_path": "axis_short.txt"},
    "axis_pec_only":   {"system_prompt_path": "axis_pec_only.txt"},
    "generic_long":    {"system_prompt_path": "generic_long.txt"},
    "generic_short":   {"system_prompt_path": "generic_short.txt"},
    "vanilla":         {"system_prompt": "You are a helpful assistant."},
    "automata_neutro": {"system_prompt_path": "automata_neutro.txt"},
}

CHUNK_COLS = 256          # columnas de cotangente por backward
JVP_TOPK = 50             # top-k guardado del readout JVP por (prompt, capa)
PRIMAL_TOL = 5e-2         # error relativo máximo admitido en la identidad del primal
WU_CHUNK = 16384          # chunk de vocabulario para W_U @ zt (memoria GPU)


def get_model_path(hf_token=None):
    local = Path(LOCAL_MODEL_DIR)
    if local.exists() and any(local.iterdir()):
        print(f"Modelo en {local}", flush=True)
        return str(local)
    print(f"Modelo no encontrado en {local} — descargando {HF_MODEL_ID} desde "
          f"HuggingFace (~62GB, ~15-20 min)...", flush=True)
    from huggingface_hub import snapshot_download
    kwargs = {"repo_id": HF_MODEL_ID, "ignore_patterns": ["*.gguf", "*.ggml"],
              "local_dir": str(local)}
    if hf_token:
        kwargs["token"] = hf_token
    return snapshot_download(**kwargs)


def load_prompts(prompts_json_path):
    with open(prompts_json_path, encoding="utf-8") as f:
        data = json.load(f)
    by_id = {p["id"]: p for p in data if p["id"] in PRIORITY_SUBSET}
    return [by_id[i] for i in PRIORITY_SUBSET]


def get_system_prompt(cfg, prompts_dir):
    if "system_prompt" in cfg:
        return cfg["system_prompt"]
    return (prompts_dir / cfg["system_prompt_path"]).read_text(encoding="utf-8")


def tokenize(tokenizer, prompt, system_prompt, device):
    """Réplica exacta de PerturbationExtractor._tokenize (chat template, generation prompt)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
    ids = tokenizer(text, return_tensors="pt").input_ids
    if ids.shape[1] > MAX_INPUT_TOKENS:
        ids = ids[:, :MAX_INPUT_TOKENS]
    return ids.to(device)


def crop_cache(cache, t):
    """Recorta el cache a las posiciones 0..t-1 (la posición t la escribe el grafo reducido)."""
    if not hasattr(cache, "key_cache"):
        # transformers <4.45 devuelve tuplas de KV — envolver en DynamicCache
        from transformers.cache_utils import DynamicCache
        wrapped = DynamicCache()
        wrapped.key_cache = [kv[0] for kv in cache]
        wrapped.value_cache = [kv[1] for kv in cache]
        cache = wrapped
    try:
        cache.crop(t)
    except Exception:
        cache.key_cache = [k[:, :, :t] for k in cache.key_cache]
        cache.value_cache = [v[:, :, :t] for v in cache.value_cache]
    return cache


def make_red_factory(layers, final_norm, cache, t, n_layers, device, mask_rows):
    """Devuelve get_red(ℓ) -> red(x): final_norm(capas ℓ+1..59 sobre la posición t
    con cache congelado). x: (D,) — se castea a bf16 (mismas operaciones que el
    forward original). mask_rows: None (modo auto — la capa construye su propia
    máscara causal+ventana vía cache_position) o dict {m: (1,1,1,t+1)} (fallback)."""
    pos_ids = torch.tensor([[t]], device=device, dtype=torch.long)
    cp = torch.tensor([t], device=device, dtype=torch.long)

    def get_red(ell):
        def red(x):
            h = x[None, None, :].to(torch.bfloat16)
            for m in range(ell + 1, n_layers):
                mask = None if mask_rows is None else mask_rows[m]
                out = layers[m](
                    hidden_states=h,
                    attention_mask=mask,
                    position_ids=pos_ids,
                    past_key_value=cache,
                    use_cache=True,
                    cache_position=cp,
                )
                h = out[0]
            return final_norm(h)[0, 0]
        return red

    return get_red


def build_explicit_mask_rows(layers, t, device):
    """Máscara explícita por capa para la posición t (solo ventana deslizante;
    la fila causal de la query t no enmascara nada: todas las claves son ≤ t)."""
    rows = {}
    for m, layer in enumerate(layers):
        attn = layer.self_attn
        w = getattr(attn, "sliding_window", None)
        if w is None:
            continue  # capa global: máscara trivial (todo ceros)
        row = torch.zeros(1, 1, 1, t + 1, device=device, dtype=torch.bfloat16)
        start = t - w + 1
        if start > 0:
            row[:, :, :, :start] = float("-inf")
        rows[m] = row
    return rows


def vjp_ell(red, x_leaf, y_orig, device, chunk_cols, primal_tol):
    """J_ℓ por VJPs chunked sobre el grafo reducido, con verificación de primal.
    Devuelve (J (D,D) f32, err_rel). Lanza RuntimeError si el primal no cuadra."""
    D = x_leaf.shape[0]
    z = red(x_leaf)
    err = float((z - y_orig).abs().max() / y_orig.abs().mean().clamp(min=1e-6))
    if err > primal_tol:
        del z
        raise RuntimeError(f"primal mismatch (err={err:.2e} > {primal_tol})")

    J = torch.empty(D, D, device=device, dtype=torch.float32)
    try:
        # cotangentes batcheados (D, C): un backward por chunk.
        # grad = J^T @ G con G = I[:, c0:c1] → grad.T = J[c0:c1, :]
        for c0 in range(0, D, chunk_cols):
            c1 = min(c0 + chunk_cols, D)
            C = c1 - c0
            G = torch.zeros(D, C, device=device, dtype=torch.bfloat16)
            G[torch.arange(c0, c1, device=device), torch.arange(C, device=device)] = 1.0
            grads = torch.autograd.grad(
                z, x_leaf, grad_outputs=G, retain_graph=True, materialize_grads=True
            )[0]  # (D, C) f32
            J[c0:c1, :] = grads.T
    except RuntimeError as e:
        # fallback: cotangente identidad completa (D, D) en un solo backward
        print(f"    [vjp] cotangentes batcheados fallaron ({e}); usando identidad completa", flush=True)
        G = torch.eye(D, device=device, dtype=torch.bfloat16)
        grads = torch.autograd.grad(z, x_leaf, grad_outputs=G, retain_graph=True,
                                    materialize_grads=True)[0]  # (D, D) = J^T
        J[...] = grads.T
    del z
    return J, err


def jvp_readout(red, x_bf, WU, first_tok, device, k=JVP_TOPK):
    """Readout J-lens exacto de la muestra: v = W_U @ (J_ℓ @ h_ℓ) vía forward-mode
    (torch.func.jvp, dual). Devuelve (top_ids (k,), top_logprobs (k,),
    logprob_primer_token). Normalización log-softmax completa vía logsumexp
    incremental en el mismo barrido chunked del vocabulario."""
    import torch.func as func
    z, zt = func.jvp(red, (x_bf,), (x_bf,))
    del z
    ztf = zt.float()

    V = WU.shape[0]
    best = torch.full((k,), -1, dtype=torch.int64, device=device)
    bestv = torch.full((k,), float("-inf"), device=device, dtype=torch.float32)
    m = float("-inf")
    s = 0.0
    ft_val = None
    for v0 in range(0, V, WU_CHUNK):
        vals = WU[v0:v0 + WU_CHUNK].float() @ ztf  # (C,)
        if ft_val is None and v0 <= first_tok < v0 + WU_CHUNK:
            ft_val = float(vals[first_tok - v0])
        cand = torch.cat([vals, bestv])[: 2 * k]
        cand_ids = torch.cat(
            [torch.arange(v0, v0 + vals.shape[0], device=device), best])[: 2 * k]
        tv, ti = torch.topk(cand, k)
        bestv, best = tv, cand_ids[ti]
        mi = float(vals.max())
        m_new = max(m, mi)
        s = s * np.exp(m - m_new) + float(torch.exp(vals - m_new).sum())
        m = m_new
    logZ = float(np.log(s)) + m
    logprobs = (bestv.float() - logZ).cpu().numpy()
    ft = ft_val - logZ if ft_val is not None else float("nan")
    return best.cpu().numpy(), logprobs, ft


def forward_prompt(model, tokenizer, prompt, system_prompt, device):
    """Forward original (no_grad): estados por capa (t=0), cache K/V, logits lm, y."""
    ids = tokenize(tokenizer, prompt, system_prompt, device)
    T = int(ids.shape[1])
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, output_hidden_states=True)
    hs = out.hidden_states  # 61 tensores (1, T, D): hs[ℓ+1] = salida de la capa ℓ
    cache_full = out.past_key_values
    logits = out.logits[0, -1].float()  # (V,)
    n_layers = len(hs) - 1
    t0 = {ell: hs[ell + 1][0, -1].clone() for ell in range(n_layers)}  # bf16 (D,)
    y = model.language_model.norm(hs[-1])[0, -1].clone()  # bf16 (D,) post-final-norm
    del out, hs
    return T, t0, y, cache_full, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", type=str, default=None, help="HF token")
    ap.add_argument("--prompts-json", type=str, default=None)
    ap.add_argument("--prompts-dir", type=str, default=None)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()))
    ap.add_argument("--sanity", action="store_true",
                    help="1 prompt de axis: chequeos + J_59 analítico + ETA, sin guardar")
    ap.add_argument("--no-jvp", action="store_true", help="no calcular readouts JVP por muestra")
    ap.add_argument("--per-condition", action="store_true",
                    help="guardar además J̄ por condición (48GB en disco del pod)")
    ap.add_argument("--chunk-cols", type=int, default=CHUNK_COLS)
    ap.add_argument("--primal-tol", type=float, default=PRIMAL_TOL)
    args = ap.parse_args()

    print(f"torch {torch.__version__} | transformers ", end="", flush=True)
    import transformers
    print(transformers.__version__, flush=True)

    prompts_json = Path(args.prompts_json) if args.prompts_json else Path(PROMPTS_JSON)
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else PROMPTS_DIR
    prompts = load_prompts(prompts_json)
    prompt_ids = [p["id"] for p in prompts]
    print(f"{len(prompts)} prompts | condiciones={args.conditions} | sanity={args.sanity} | "
          f"jvp={not args.no_jvp} | per-condition={args.per_condition}", flush=True)

    device = "cuda:0"
    model_path = get_model_path(args.token)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Cargando modelo (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, token=args.token
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=args.token)

    lm = model.language_model
    layers = lm.layers
    final_norm = lm.norm
    n_layers = len(layers)
    D = int(lm.config.hidden_size)
    WU = model.lm_head.weight  # tied a embed_tokens (verificar abajo)
    tied = bool((model.lm_head.weight == model.get_input_embeddings().weight).all())
    print(f"capas={n_layers} D={D} V={WU.shape[0]} tied_WU={tied}", flush=True)
    assert tied, "lm_head no está atado a embed_tokens — el lens local usa embed_tokens"
    assert n_layers == 60, f"esperaba 60 capas, hay {n_layers}"

    first_tok_check = {}
    t_times = {"forward": [], "vjp": [], "jvp": []}

    # ── SANITY: 1 prompt de axis, chequeos completos ──
    if args.sanity:
        p = prompts[0]
        sys_p = get_system_prompt(CONDITIONS["axis"], prompts_dir)
        t_start = time.time()
        T, t0, y, cache_full, logits = forward_prompt(model, tokenizer, p["text"], sys_p, device)
        print(f"T={T} ids={p['id']} ({time.time()-t_start:.0f}s)", flush=True)

        wy = WU.float() @ y.float()
        ok = int(wy.argmax()) == int(logits.argmax())
        print(f"base check (argmax(W_U @ y) == argmax(lm_logits)): {ok} "
              f"(top1={tokenizer.decode([int(wy.argmax())])!r})", flush=True)
        assert ok, "verificación de base falló — revisar convención de estados"

        t = T - 1
        cache_full = crop_cache(cache_full, t)
        mask_rows = build_explicit_mask_rows(layers, t, device)
        get_red = make_red_factory(layers, final_norm, cache_full, t, n_layers, device, None)

        # J_59 analítico: Jacobiana de la RMSNorm final, z = w⊙h/r, r = sqrt(mean(h²)+eps)
        x59 = t0[59].detach().float().requires_grad_(True)
        J59, err59 = vjp_ell(get_red(59), x59, y, device, args.chunk_cols, args.primal_tol)
        print(f"primal L59 (auto): err={err59:.2e}", flush=True)
        eps = float(getattr(final_norm, "eps", 1e-6))
        h = t0[59].detach().float()
        r2 = float((h ** 2).mean()) + eps
        r = r2 ** 0.5
        w = final_norm.weight.float()
        j_errs = []
        for j in [0, D // 2, D - 1]:
            ana = w * ((torch.arange(D, device=device) == j).float() / r
                       - h * h[j] / (D * r ** 3))
            j_errs.append(float((J59[j] - ana).abs().max()))
        print(f"J_59 analítico: máx |J59[j] - analítico| = {max(j_errs):.2e}", flush=True)
        assert max(j_errs) < 1e-2, "J_59 no cuadra con la Jacobiana analítica — maquinaria VJP rota"
        del x59, J59

        # primal en una capa media: modo auto, y fallback explícito si hace falta
        t_v = time.time()
        x30 = t0[30].detach().float().requires_grad_(True)
        try:
            J30, err30 = vjp_ell(get_red(30), x30, y, device, args.chunk_cols, args.primal_tol)
            print(f"primal L30 (auto): err={err30:.2e} | vjp L30 {time.time()-t_v:.1f}s", flush=True)
        except RuntimeError as e:
            print(f"primal FAIL auto L30: {e} — probando máscara explícita", flush=True)
            get_red2 = make_red_factory(layers, final_norm, cache_full, t, n_layers,
                                        device, mask_rows)
            J30, err30 = vjp_ell(get_red2(30), x30, y, device, args.chunk_cols, args.primal_tol)
            print(f"primal L30 (explícito): err={err30:.2e} | vjp L30 {time.time()-t_v:.1f}s", flush=True)
        del x30, J30

        if not args.no_jvp:
            try:
                t_j = time.time()
                top, lp, ft = jvp_readout(get_red(59), t0[59].detach(), WU,
                                          int(logits.argmax()), device)
                print(f"JVP L59 ok (top1={tokenizer.decode([int(top[0])])!r}, "
                      f"logprob 1er token={ft:.2f}) ({time.time()-t_j:.1f}s)", flush=True)
            except Exception as e:
                print(f"JVP falló: {type(e).__name__}: {e} — correr con --no-jvp", flush=True)

        per_ell_vjp = t_v / 1.0
        n_samples = len(args.conditions) * len(prompts)
        est_h = (per_ell_vjp * n_layers * n_samples) / 3600.0
        print(f"\nETA: {n_samples} muestras × ~{per_ell_vjp * n_layers:.0f}s VJP por muestra "
              f"≈ {est_h:.1f} h solo VJP (+ forward ~{time.time()-t_start:.0f}s/muestra)",
              flush=True)
        print("SANITY_OK", flush=True)
        return

    # ── CORRIDA COMPLETA ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "states").mkdir(exist_ok=True)
    (OUT_DIR / "jacobians").mkdir(exist_ok=True)
    if not args.no_jvp:
        (OUT_DIR / "jvp").mkdir(exist_ok=True)

    jac_sum = {ell: np.zeros((D, D), dtype=np.float32) for ell in range(n_layers)}
    per_cond = ({c: {ell: np.zeros((D, D), dtype=np.float32) for ell in range(n_layers)}
                 for c in args.conditions}) if args.per_condition else None
    n_jac_samples = 0
    meta_checks = []

    for cond in args.conditions:
        cfg = CONDITIONS[cond]
        sys_p = get_system_prompt(cfg, prompts_dir)
        out = {
            "t0_states": np.zeros((len(prompts), n_layers, D), dtype=np.float32),
            "y": np.zeros((len(prompts), D), dtype=np.float32),
            "lm_logits": np.zeros((len(prompts), WU.shape[0]), dtype=np.float16),
            "lm_top10_ids": np.zeros((len(prompts), 10), dtype=np.int32),
            "lm_top10_logprobs": np.zeros((len(prompts), 10), dtype=np.float16),
            "first_token_ids": np.zeros(len(prompts), dtype=np.int32),
            "T": np.zeros(len(prompts), dtype=np.int32),
            "prompt_ids": np.array(prompt_ids, dtype=np.int32),
        }
        jvp_out = {
            "jvp_top50_ids": np.full((len(prompts), n_layers, JVP_TOPK), -1, dtype=np.int32),
            "jvp_top50_logprobs": np.zeros((len(prompts), n_layers, JVP_TOPK), dtype=np.float16),
            "jvp_first_tok_logprob": np.zeros((len(prompts), n_layers), dtype=np.float16),
        }

        for i, p in enumerate(prompts):
            print(f"\n[{cond}] prompt {i+1}/{len(prompts)} id={p['id']}", flush=True)
            t_p0 = time.time()
            T, t0, y, cache_full, logits = forward_prompt(model, tokenizer, p["text"], sys_p, device)
            t_times["forward"].append(time.time() - t_p0)

            out["t0_states"][i] = np.stack([t0[e].float().cpu().numpy() for e in range(n_layers)])
            out["y"][i] = y.float().cpu().numpy()
            out["lm_logits"][i] = logits.half().cpu().numpy()
            topk = torch.topk(logits, 10)
            out["lm_top10_ids"][i] = topk.indices.cpu().numpy()
            out["lm_top10_logprobs"][i] = torch.log_softmax(logits, -1)[topk.indices].half().cpu().numpy()
            first_tok = int(logits.argmax())
            out["first_token_ids"][i] = first_tok
            out["T"][i] = T

            ok = int((WU.float() @ y.float()).argmax()) == first_tok
            first_tok_check.setdefault(cond, []).append(int(ok))
            assert ok, f"base check falló en {cond} id={p['id']}"

            # cache recortado a 0..t-1 (posición t la escribe el grafo reducido)
            t = T - 1
            cache_full = crop_cache(cache_full, t)
            mask_rows = build_explicit_mask_rows(layers, t, device)
            get_red_auto = make_red_factory(layers, final_norm, cache_full, t, n_layers,
                                            device, None)
            get_red_expl = None
            mode = "auto"

            t_v0 = time.time()
            worst_err = 0.0
            worst_ell = -1
            for ell in range(n_layers):
                x_leaf = t0[ell].detach().float().requires_grad_(True)
                try:
                    J, err = vjp_ell(get_red_auto(ell), x_leaf, y, device,
                                     args.chunk_cols, args.primal_tol)
                except RuntimeError as e:
                    if get_red_expl is None:
                        get_red_expl = make_red_factory(layers, final_norm, cache_full, t,
                                                        n_layers, device, mask_rows)
                    print(f"    L{ell}: {e} — reintento con máscara explícita", flush=True)
                    mode = "explícito"
                    J, err = vjp_ell(get_red_expl(ell), x_leaf, y, device,
                                     args.chunk_cols, args.primal_tol)
                if err > worst_err:
                    worst_err, worst_ell = err, ell
                jac_sum[ell] += J.cpu().numpy()
                if per_cond is not None:
                    per_cond[cond][ell] += J.cpu().numpy()
                del x_leaf, J

                if not args.no_jvp:
                    try:
                        t_j = time.time()
                        top, lp, ft = jvp_readout(get_red_auto(ell), t0[ell].detach(), WU,
                                                  first_tok, device)
                        t_times["jvp"].append(time.time() - t_j)
                        jvp_out["jvp_top50_ids"][i, ell] = top
                        jvp_out["jvp_top50_logprobs"][i, ell] = lp
                        jvp_out["jvp_first_tok_logprob"][i, ell] = ft
                    except Exception as e:
                        print(f"    JVP L{ell} falló ({type(e).__name__}: {e}) — "
                              f"solo VJP para esta capa", flush=True)
            n_jac_samples += 1
            t_times["vjp"].append(time.time() - t_v0)

            meta_checks.append({"cond": cond, "id": p["id"], "mode": mode,
                                "worst_layer": int(worst_ell),
                                "worst_rel_err": float(worst_err)})
            print(f"    VJP {time.time()-t_v0:.0f}s | primal máx L{worst_ell}={worst_err:.2e} "
                  f"({mode}) | total {time.time()-t_p0:.0f}s", flush=True)

            del t0, y, cache_full
            if (i + 1) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        np.savez_compressed(OUT_DIR / "states" / f"{cond}.npz", **out)
        if not args.no_jvp:
            np.savez_compressed(OUT_DIR / "jvp" / f"{cond}.npz", **jvp_out)
        fc = sum(first_tok_check.get(cond, []))
        print(f"[{cond}] guardado | base check {fc}/{len(prompts)}", flush=True)

    # ── Jacobianas promediadas ──
    print("\nGuardando Jacobianas promediadas (pooled)...", flush=True)
    for ell in range(n_layers):
        np.save(OUT_DIR / "jacobians" / f"J_L{ell}.npy", jac_sum[ell] / n_jac_samples)
    if per_cond is not None:
        for cond in args.conditions:
            for ell in range(n_layers):
                np.save(OUT_DIR / "jacobians" / f"J_percond_{cond}_L{ell}.npy",
                        per_cond[cond][ell] / len(prompts))
    print(f"jacobians/J_L*.npy × {n_layers} (pooled, n={n_jac_samples})", flush=True)

    meta = {
        "protocol": "SOUL_MD_UPDATE_GUIDE.md §8 + preregistro A1 §4.1 (VJPs chunked, J̄ pooled)",
        "model": HF_MODEL_ID,
        "tied_WU": tied,
        "n_layers": n_layers,
        "hidden_dim": D,
        "n_samples_jacobian": n_jac_samples,
        "conditions": args.conditions,
        "prompt_ids": prompt_ids,
        "chunk_cols": args.chunk_cols,
        "primal_tol": args.primal_tol,
        "jvp": not args.no_jvp,
        "base_check_per_condition": {c: f"{sum(v)}/{len(v)}"
                                     for c, v in first_tok_check.items()},
        "primal_checks": meta_checks,
        "timings_sec": {k: {"sum": float(np.sum(v)),
                            "mean": float(np.mean(v)) if v else 0.0,
                            "max": float(np.max(v)) if v else 0.0}
                        for k, v in t_times.items()},
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("meta.json guardado", flush=True)
    print("EXTRACTION_DONE", flush=True)


if __name__ == "__main__":
    main()
