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

J-lens: se guardan readouts JVP por (prompt, capa) con la Jacobiana EXACTA de
cada muestra (torch.func.jvp sobre el grafo reducido) — el instrumento primario
(la definición de Anthropic aplica la Jacobiana de la propia muestra). El J̄
promediado (pre-registro A1 §4.1) quedó OPCIONAL (--jbar): los dos muros medidos
en el pod 2026-09-03 (pico de duales por profundidad del jvp full-stack y ~17ms
de overhead functorch por llamada) lo hacen inviable con contextos de 4K tokens
— el pre-registro asumía corpus de 256 tokens. Ver SOUL_MD_UPDATE_GUIDE.md §9.

Truco de costo: las VJPs se computan sobre un GRAFO REDUCIDO de 1 posición
(x_ℓ = h_ℓ[t] como hoja f32; cache K/V de las posiciones 0..t-1 congelado en un
estado-master; capas ℓ+1..59 llamadas como módulos reales). Sin el grafo
reducido, los cotangentes (T, D, C) de la secuencia completa (T≈6K) son
imposibles de materializar y el costo ~100×.

Entorno validado (pod 2026-09-03): transformers 5.16.1 + torch 2.14.0+cu130.
Particularidades de esta versión, ya resueltas y validadas por el test del pod:
  - hs[-1] YA ES post-final-norm (la lm_head lo consume directo); el h_59 crudo
    se captura con un hook en la entrada de lm.norm.
  - Gemma4TextDecoderLayer recibe per_layer_input=None (hidden_size_per_layer_input
    =0 en este checkpoint), position_embeddings por tipo de capa (rotary,
    head_dim 256 sliding / 512 full), máscaras por tipo vía
    transformers.masking_utils (create_causal_mask / create_sliding_window_causal_mask),
    y shared_kv_states={} (ninguna capa comparte K/V en este checkpoint; solo L59
    guarda en el dict y nadie lo lee).
  - Cache 5.x: DynamicCache con capas DynamicLayer (attr .keys/.values); el
    crop usa semántica nueva (crop(-1) quita 1 token del final). Cada llamada
    reducida RESTAURA el cache desde un estado-master (los updates reemplazan
    los tensores, no mutan in-place — verificado).
  - allow_bf16_reduced_precision_reduction = False al inicio: sin esto, las
    GEMMs (1,1,D) toman kernels con acumulación distinta a las (1,T,D) del
    forward original (diferencia escalar determinista en q_proj, verificado).
    Con la flag OFF, el grafo reducido reproduce el forward original salvo
    ruido ULP (diferente orden de reducción entre kernels): error del primal
    medido 0.0 en L59, 5.8e-2 en L58, satura ~0.37-0.46 en capas profundas.
    NOTA: ese ruido del FORWARD no contamina las JACOBIANAS — los ops de
    redondeo bf16 son identidad en el backward (straight-through), de modo que
    J = producto de Jacobianas locales suaves, exacto salvo ULP del backward.
    Por eso el primal usa tolerancia ESTRUCTURAL (0.6): máscaras/claves
    equivocadas dan error O(1); el ruido ULP ≤ 0.46. El error del primal por
    (prompt, capa) queda grabado en meta.json.

Verificaciones (grabadas en meta.json):
  1. argmax(W_U @ y) == argmax(logits lm_head) por prompt — 20/20 esperado.
  2. Identidad del primal del grafo reducido por (prompt, capa) — ABORTA si
     supera --primal-tol (estructural).
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
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

# antes de importar torch: segmentos expandibles (evita fragmentación del
# allocator — el dual del K/V del grafo reducido es la presión de memoria)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

# f32-accum para todas las GEMMs bf16: sin esto, las GEMMs (1,1,D) del grafo
# reducido toman kernels con acumulación distinta a las (1,T,D) del forward
# original (verificado en el pod 2026-09-03).
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

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
    # Controles exploratorios (2026-09-03): entidades externas del panel T2 —
    # soul_md_corto = mecanismo testigo sin identidad propia; soul_elena_financial
    # = identidad declarada con valores, sin testigo. Mismo estatus exploratorio
    # que en run_perturbation_t2.py.
    "soul_md_corto":         {"system_prompt_path": "soul_md_corto.md"},
    "soul_elena_financial":  {"system_prompt_path": "soul_elena_financial.txt"},
}

CHUNK_COLS = 8            # columnas de tangente (dual del cat es f32; baseline 61.6GiB → pico ~67GiB)
JVP_TOPK = 50             # top-k guardado del readout JVP por (prompt, capa)
PRIMAL_TOL = 2.0          # tolerancia ESTRUCTURAL del primal (ruido ULP crece con T: ≤0.46 en T=12, ~1.4 en T≈3K)
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


def build_plumbing(model, lm, layers, layer_types, ids, t, cache):
    """Máscaras por tipo + embeddings rotacionales + estado-master del cache.

    Devuelve (mk, pe, master_k, master_v, reset). `cache` queda en estado past
    (posiciones 0..t-1) — la posición t original fue removida con crop(-1)."""
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    emb = model.get_input_embeddings()(ids)          # (1, T, D) — para máscaras y rope
    # estado past del cache: quitar la posición t original (slicing directo,
    # sin crop — el crop de 5.x no permite rollback en capas deslizantes).
    # Las capas deslizantes guardan SOLO el pasado (longitud W-1, ya sin t);
    # las completas guardan 0..t. Slice solo si el slot incluye t (len == T).
    T = ids.shape[1]
    master_k, master_v = [], []
    for i in range(len(layers)):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        if k.shape[-2] == T:
            master_k.append(k[..., :-1, :])
            master_v.append(v[..., :-1, :])
        else:
            master_k.append(k)
            master_v.append(v)

    def reset(batch=1):
        for i in range(len(layers)):
            cache.layers[i].keys = master_k[i].expand(batch, -1, -1, -1)
            cache.layers[i].values = master_v[i].expand(batch, -1, -1, -1)

    reset()
    pos_t = torch.tensor([[t]], device=ids.device, dtype=torch.long)
    mk = {}
    for lt, fn in (("full_attention", create_causal_mask),
                   ("sliding_attention", create_sliding_window_causal_mask)):
        mk[lt] = fn(config=lm.config, inputs_embeds=emb[:, :1], attention_mask=None,
                    past_key_values=cache, position_ids=pos_t)
    # el builder de la máscara deslizante produce una columna de más cuando
    # T < ventana (verificado en el pod: T=966 → máscara 967) — recortar
    W = int(getattr(lm.config, "sliding_window", 1024) or 1024)
    mk["sliding_attention"] = mk["sliding_attention"][..., :min(T, W)]
    pos_full = torch.arange(ids.shape[1], device=ids.device)[None]
    pe = {}
    for lt in ["sliding_attention", "full_attention"]:
        c, s = lm.rotary_emb(emb, pos_full, lt)
        pe[lt] = (c[:, t:t + 1], s[:, t:t + 1])

    return mk, pe, pos_t, reset, master_k, master_v


def wu_argmax(WU, x, device):
    """argmax(W_U @ x) en chunks de vocabulario (sin materializar W_U f32)."""
    best_i, best_v = -1, float("-inf")
    V = WU.shape[0]
    for v0 in range(0, V, WU_CHUNK):
        vals = WU[v0:v0 + WU_CHUNK].float() @ x
        i = int(vals.argmax()) + v0
        v = float(vals.max())
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def make_red_factory(layers, final_norm, cache, layer_types, mk, pe, pos_t,
                      master_k, master_v, n_layers, device):
    """Devuelve get_red(ℓ) -> red(x): capas ℓ+1..59 sobre la posición t (x: (D,)
    o (C, D) — batch de copias para los tangentes del jvp; se castea a bf16).
    El caller hace reset() antes de cada llamada. Después de cada capa se
    restaura su slot desde el master: el cat (C, 16, kv, 512) del update es
    grande (2-4GB) y forward-mode no retiene grafo, así que se libera al
    instante — sin esto los 60 cats se acumulan hasta el siguiente reset."""
    def get_red(ell):
        def red(x):
            hh = (x[None, :] if x.ndim == 1 else x)[:, None, :].to(torch.bfloat16)
            C = hh.shape[0]
            pos_t_c = pos_t.expand(C, 1)
            for m in range(ell + 1, n_layers):
                hh = layers[m](
                    hh, None,
                    shared_kv_states={},
                    position_embeddings=pe[layer_types[m]],
                    attention_mask=mk[layer_types[m]],
                    position_ids=pos_t_c,
                    past_key_values=cache,
                )
                cache.layers[m].keys = master_k[m]
                cache.layers[m].values = master_v[m]
            z = final_norm(hh)[:, 0]
            return z[0] if C == 1 else z
        return red

    return get_red


def vjp_ell(red, x_val, y_orig, reset, device, chunk_cols, primal_tol):
    """J_ℓ = ∂z/∂x por VJPs chunked sobre el grafo reducido, con verificación de
    primal. torch ≥2.14 no admite cotangentes batcheados en autograd.grad (shape
    check estricto) y jacrev (backward) OOM-ea por los cotangentes del KV
    completo — se usa torch.func.jacfwd (forward-mode, chunk_size), el análogo
    forward del chunking pre-registrado. x_val f32 → J f32.
    IMPORTANTE: cada forward de red agrega la posición t al cache — hay que
    reset() entre el forward del primal y el forward interno de jacfwd.
    Devuelve (J (D,D) f32, err_rel). Lanza RuntimeError si el primal no cuadra
    (error estructural — máscaras/claves equivocadas; el ruido ULP ≤ 0.46 no
    supera la tolerancia)."""
    import torch.func as func
    D = x_val.shape[0]
    z = red(x_val)
    err = float((z - y_orig).abs().max() / y_orig.abs().mean().clamp(min=1e-6))
    if err > primal_tol:
        del z
        raise RuntimeError(f"primal mismatch (err={err:.2e} > {primal_tol})")
    del z
    # FORWARD-MODE chunked (jvp con tangentes batcheados), no backward: los
    # tangentes solo fluyen por la posición actual (el pasado K/V tiene tangente
    # cero), de modo que la memoria es (C, D_inter) — el backward (jacrev)
    # materializa cotangentes (kv≈4K, D, C) = 33GB por el cat del KV y OOM-ea.
    # jacfwd de torch no acepta chunk_size, así que el chunking es manual:
    # por chunk, X = C copias del primal y V = las C columnas base del chunk;
    # el jvp da zt[c, :] = ∂z/∂x_{c0+c} (columna c0+c de J).
    # La Jacobiana se acumula en CPU (numpy): materializar el tangente del jvp
    # en GPU (asignación/clone) retiene ~3.9GB por chunk en torch 2.14 — bug
    # del nivel functorch — mientras la vía .cpu().numpy() es plana (verificado).
    J = np.zeros((D, D), dtype=np.float32)
    for c0 in range(0, D, chunk_cols):
        c1 = min(c0 + chunk_cols, D)
        C = c1 - c0
        reset(C)
        # bf16: los duales del jvp quedan en bf16 (las normas internas castean a
        # f32 y de vuelta) — en f32 el dual del cat es (C,16,kv,512)·4B = 4GB
        X = x_val.detach()[None, :].expand(C, -1).clone().to(torch.bfloat16)
        V = torch.zeros(C, D, device=device, dtype=torch.bfloat16)
        V[torch.arange(C, device=device), torch.arange(c0, c1, device=device)] = 1.0
        zz, zt = func.jvp(red, (X,), (V,))
        np.copyto(J[:, c0:c1], zt.T.detach().float().cpu().numpy())
        del X, V, zz, zt
        if c0 % 1024 == 0:
            torch.cuda.synchronize()
            print(f"    [mem] chunk {c0}: alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
    reset(1)
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
        # merge: cat COMPLETO de vals + best acumulado (sin slice — el slice
        # cortaba el best: con chunk 16K > 2k el topk solo veía el chunk actual)
        cand = torch.cat([vals, bestv])
        cand_ids = torch.cat(
            [torch.arange(v0, v0 + vals.shape[0], device=device), best])
        tv, ti = torch.topk(cand, k)
        bestv, best = tv, cand_ids[ti]
        mi = float(vals.max())
        m_new = max(m, mi)
        s = s * np.exp(m - m_new) + float(torch.exp(vals - m_new).sum())
        m = m_new
    logZ = float(np.log(s)) + m
    logprobs = (bestv.float() - logZ).detach().cpu().numpy()
    ft = ft_val - logZ if ft_val is not None else float("nan")
    return best.detach().cpu().numpy(), logprobs, ft


def forward_prompt(model, lm, final_norm, tokenizer, prompt, system_prompt, device):
    """Forward original (no_grad): estados por capa (t=0), cache K/V, logits lm, y.

    Convención 5.16.1: hs[-1] ya es post-final-norm (= y); el h_59 crudo se
    captura con un hook en la entrada de lm.norm (no está en la tupla)."""
    ids = tokenize(tokenizer, prompt, system_prompt, device)
    T = int(ids.shape[1])
    raw59 = {}
    h_norm = final_norm.register_forward_hook(
        lambda m, args, out: raw59.__setitem__(0, args[0][0, -1].detach().clone()))
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, output_hidden_states=True)
    h_norm.remove()
    hs = out.hidden_states  # 61 tensores (1, T, D): hs[ℓ+1] = salida capa ℓ; hs[-1] = post-norm
    cache_full = out.past_key_values
    logits = out.logits[0, -1].float()  # (V,)
    n_layers = len(hs) - 1
    t0 = {ell: hs[ell + 1][0, -1].clone() for ell in range(n_layers - 1)}  # bf16 (D,)
    t0[n_layers - 1] = raw59[0].clone()  # h_59 crudo
    y = hs[-1][0, -1].clone()            # bf16 (D,) post-final-norm
    del out, hs
    return ids, T, t0, y, cache_full, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", type=str, default=None, help="HF token")
    ap.add_argument("--prompts-json", type=str, default=None)
    ap.add_argument("--prompts-dir", type=str, default=None)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()))
    ap.add_argument("--sanity", action="store_true",
                    help="1 prompt de axis: chequeos + J_59 analítico + ETA, sin guardar")
    ap.add_argument("--no-jvp", action="store_true", help="no calcular readouts JVP por muestra")
    ap.add_argument("--skip-primal", action="store_true",
                    help="saltar los primal checks C=1 (run de solo-readouts — "
                         "la mitad del tiempo por prompt)")
    ap.add_argument("--per-condition", action="store_true",
                    help="guardar además J̄ por condición (48GB en disco del pod)")
    ap.add_argument("--jbar", action="store_true",
                    help="computar J̄ (Jacobianas completas por capa) — los dos muros "
                         "medidos en el pod (pico de duales por profundidad en el jvp "
                         "full-stack y ~17ms de overhead functorch por llamada) lo hacen "
                         "inviable con contextos de 4K tokens; ver SOUL_MD_UPDATE_GUIDE.md §9")
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
    print("Cargando modelo (bf16, eager)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager",
        trust_remote_code=True, token=args.token
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=args.token)

    lm = model.model.language_model
    layers = lm.layers
    final_norm = lm.norm
    layer_types = lm.config.layer_types
    n_layers = len(layers)
    WU = model.lm_head.weight  # tied a embed_tokens (verificar abajo)
    D = int(WU.shape[1])
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
        ids, T, t0, y, cache_full, logits = forward_prompt(
            model, lm, final_norm, tokenizer, p["text"], sys_p, device)
        print(f"T={T} ids={p['id']} ({time.time()-t_start:.0f}s)", flush=True)

        ok = wu_argmax(WU, y.float(), device) == int(logits.argmax())
        print(f"base check (argmax(W_U @ y) == argmax(lm_logits)): {ok} "
              f"(top1={tokenizer.decode([int(logits.argmax())])!r})", flush=True)
        assert ok, "verificación de base falló — revisar convención de estados"

        t = T - 1
        mk, pe, pos_t, reset, master_k, master_v = build_plumbing(
            model, lm, layers, layer_types, ids, t, cache_full)
        get_red = make_red_factory(layers, final_norm, cache_full, layer_types, mk, pe, pos_t,
                                   master_k, master_v, n_layers, device)

        # J_59 analítico: Jacobiana de la RMSNorm final, z = w⊙h/r, r = sqrt(mean(h²)+eps)
        x59 = t0[59].detach().float().requires_grad_(True)
        J59, err59 = vjp_ell(get_red(59), x59, y, reset, device, args.chunk_cols, args.primal_tol)
        print(f"primal L59 (solo norma): err={err59:.2e}", flush=True)
        eps = float(getattr(final_norm, "eps", 1e-6))
        h = t0[59].detach().float()
        r2 = float((h ** 2).mean()) + eps
        r = r2 ** 0.5
        w = final_norm.weight.float()
        j_errs = []
        J59_np = J59  # numpy (D, D), J[i, j] = ∂z_i/∂h_j
        ana_np = (w.detach().cpu().numpy(), h.cpu().numpy())
        for j in [0, D // 2, D - 1]:
            # ana = ∂z_*/∂h_j (columna j de la Jacobiana)
            wa, ha = ana_np
            r_np = float(np.sqrt((ha ** 2).mean() + eps))
            ana = wa * ((np.arange(D) == j).astype(np.float32) / r_np
                        - ha * ha[j] / (D * r_np ** 3))
            j_errs.append(float(np.abs(J59_np[:, j] - ana).max() / np.abs(ana).max()))
        print(f"J_59 analítico: máx err RELATIVO por columna = {max(j_errs):.2e} "
              f"(pesos de la norma ~{w.abs().median():.1f}, no ~1)", flush=True)
        assert max(j_errs) < 1e-2, "J_59 no cuadra con la Jacobiana analítica — maquinaria VJP rota"
        del x59, J59

        # primal en capas con pila completa, C=1 (barato: un forward cada una).
        # Un error estructural — máscaras/claves — se ve ya en L58; el ruido ULP
        # crece con la profundidad y con T (se registra en meta, no aborta).
        t_v = time.time()
        for ell in [58, 30, 0]:
            reset()
            z_ell = get_red(ell)(t0[ell].detach().float())
            err = float((z_ell - y).abs().max() / y.abs().mean().clamp(min=1e-6))
            print(f"primal L{ell} (C=1): err={err:.2e}", flush=True)
            del z_ell

        if not args.no_jvp:
            try:
                t_j = time.time()
                reset()
                top, lp, ft = jvp_readout(get_red(59), t0[59].detach(), WU,
                                          int(logits.argmax()), device)
                print(f"JVP L59 ok (top1={tokenizer.decode([int(top[0])])!r}, "
                      f"logprob 1er token={ft:.2f}) ({time.time()-t_j:.1f}s)", flush=True)
            except Exception as e:
                print(f"JVP falló: {type(e).__name__}: {e} — correr con --no-jvp", flush=True)

        n_samples = len(args.conditions) * len(prompts)
        t_per_ell = max((time.time() - t_start) / 4.0, 1.0)
        est_h = (t_per_ell * n_layers * n_samples) / 3600.0
        print(f"\nETA (JVP readouts + primal checks por capa): {n_samples} muestras × "
              f"{n_layers} capas ≈ {est_h:.1f} h (+ forward ~5s/muestra)", flush=True)
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
            ids, T, t0, y, cache_full, logits = forward_prompt(
                model, lm, final_norm, tokenizer, p["text"], sys_p, device)
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

            ok = wu_argmax(WU, y.float(), device) == first_tok
            first_tok_check.setdefault(cond, []).append(int(ok))
            if not ok:
                print(f"    [warn] base check divergió en {cond} id={p['id']} "
                      f"(near-tie de logits probable — el primer token de referencia "
                      f"sigue siendo el argmax del modelo)", flush=True)

            # estado past del cache + plumbing (máscaras/rope) + master
            t = T - 1
            mk, pe, pos_t, reset, master_k, master_v = build_plumbing(
                model, lm, layers, layer_types, ids, t, cache_full)
            get_red = make_red_factory(layers, final_norm, cache_full, layer_types, mk, pe, pos_t,
                                       master_k, master_v, n_layers, device)

            t_v0 = time.time()
            worst_err = 0.0
            worst_ell = -1
            for ell in range(n_layers):
                # primal check C=1 (informativo: ruido ULP crece con profundidad/T —
                # se registra en meta, no aborta)
                if not args.skip_primal:
                    reset()
                    x_chk = t0[ell].detach().float()
                    z_chk = get_red(ell)(x_chk)
                    err = float((z_chk - y).abs().max() / y.abs().mean().clamp(min=1e-6))
                    del x_chk, z_chk
                    if err > worst_err:
                        worst_err, worst_ell = err, ell

                if args.jbar:
                    reset()
                    x_leaf = t0[ell].detach().float().requires_grad_(True)
                    J, jerr = vjp_ell(get_red(ell), x_leaf, y, reset, device,
                                      args.chunk_cols, args.primal_tol)
                    jac_sum[ell] += J
                    if per_cond is not None:
                        per_cond[cond][ell] += J
                    del x_leaf, J

                if not args.no_jvp:
                    try:
                        t_j = time.time()
                        reset()
                        top, lp, ft = jvp_readout(get_red(ell), t0[ell].detach(), WU,
                                                  first_tok, device)
                        t_times["jvp"].append(time.time() - t_j)
                        jvp_out["jvp_top50_ids"][i, ell] = top
                        jvp_out["jvp_top50_logprobs"][i, ell] = lp
                        jvp_out["jvp_first_tok_logprob"][i, ell] = ft
                        # GATE en la última capa: por homogeneidad de la RMSNorm
                        # (eps>0) el tangente es z·eps/(mse+eps) ≈ z·1e-10 y
                        # SUBDESBORDA a cero en el cast bf16 de la norma — el
                        # readout es la distribución UNIFORME (logprob del primer
                        # token ≈ -log(V)). Valida la cadena logZ/ft_val/W_U con
                        # una respuesta conocida. (El top-50 de L59 es artefacto
                        # de empates sobre ceros — esperado y documentado.)
                        if ell == n_layers - 1:
                            import math
                            unif = -math.log(WU.shape[0])
                            if not (unif - 0.5 <= ft <= unif + 0.5):
                                raise RuntimeError(
                                    f"gate L59: ft={ft:.2f} ≠ uniforme {-unif:.2f} — "
                                    f"revisar la cadena del readout")
                    except Exception as e:
                        print(f"    JVP L{ell} falló ({type(e).__name__}: {e})", flush=True)
            if args.jbar:
                n_jac_samples += 1
            t_times["vjp"].append(time.time() - t_v0)

            meta_checks.append({"cond": cond, "id": p["id"],
                                "worst_layer": int(worst_ell),
                                "worst_rel_err": float(worst_err)})
            print(f"    capas {time.time()-t_v0:.0f}s | primal máx L{worst_ell}={worst_err:.2e} | "
                  f"total {time.time()-t_p0:.0f}s", flush=True)

            del t0, y, cache_full, mk, pe
            if (i + 1) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        np.savez_compressed(OUT_DIR / "states" / f"{cond}.npz", **out)
        if not args.no_jvp:
            np.savez_compressed(OUT_DIR / "jvp" / f"{cond}.npz", **jvp_out)
        fc = sum(first_tok_check.get(cond, []))
        print(f"[{cond}] guardado | base check {fc}/{len(prompts)}", flush=True)

    # ── Jacobianas promediadas (solo con --jbar) ──
    if args.jbar:
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
        "bf16_reduced_precision": False,
        "jvp": not args.no_jvp,
        "jbar": args.jbar,
        "jbar_note": "J̄ desactivado por defecto: pico de duales por profundidad del jvp "
                     "full-stack (~0.5GB/capa a C=8) y ~17ms de overhead functorch por "
                     "llamada lo hacen inviable en contextos de 4K tokens (ver guía §9). "
                     "El instrumento J-lens primario es el readout exacto por muestra (jvp/).",
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
