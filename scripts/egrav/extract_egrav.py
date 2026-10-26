#!/usr/bin/env python3
"""
E-GRAV · extractor GPU para RunPod.

Genera respuestas + embeddings (capa final post-norm, greedy, 256 tok, bf16)
para 3 condiciones × 3 baterías = 9 corridas × 20 prompts = 180 generaciones.

Mismo pipeline que el panel LSGOT original (lsgot_extract.py):
  - chat template nativo de Gemma-4
  - stop-mode legacy: solo para en <eos> (id 1), no en <turn|> (id 106)
  - capa final = hidden_states[-1] del último token generado en cada paso
  - output NPZ: embeddings (N, T_max, 5376) + lengths (N,)

Reanudable: salta las corridas ya guardadas. Usa --force para recalcular.

Uso:
    export HF_TOKEN=hf_xxx
    python extract_egrav.py --out /workspace/egrav_results
    python extract_egrav.py --dry-run          # sin GPU
    python extract_egrav.py --smoke            # 2 prompts por corrida (prueba rápida)
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

BATTERIES = {
    "b_onto":  HERE / "batteries" / "b_onto.json",
    "b_tarea": HERE / "batteries" / "b_tarea.json",
    "b_meta":  HERE / "batteries" / "b_meta.json",
}

CONDITIONS = {
    "axis_pec_only":   HERE / "prompts" / "axis_pec_only.txt",
    "vanilla":         None,
    "automata_neutro": HERE / "prompts" / "automata_neutro.txt",
}

MODEL_LOCAL = "/workspace/models/gemma-4-31B-it"
HF_ID       = "google/gemma-4-31B-it"
HIDDEN_DIM  = 5376   # Gemma-4-31B post-norm capa final
MAX_TOKENS  = 256
EOS_ID      = 1      # stop-mode legacy: solo <eos>


def get_model_path(hf_token: str | None) -> str:
    p = Path(MODEL_LOCAL)
    if p.exists() and any(p.iterdir()):
        print(f"Modelo en {p}", flush=True)
        return str(p)
    print(f"Descargando {HF_ID} (~62 GB)…", flush=True)
    from huggingface_hub import snapshot_download
    return snapshot_download(HF_ID, token=hf_token, local_dir=MODEL_LOCAL)


def load_model(model_path: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # igual que el panel
    )
    model.eval()
    print(f"Modelo en {next(model.parameters()).device}", flush=True)
    return tok, model


def extract_one(tok, model, system_prompt: str | None, user_text: str):
    """
    Genera una respuesta y devuelve (hidden_matrix, response_text).
    hidden_matrix: (T_gen, HIDDEN_DIM) — capa final post-norm, un vector por token.
    """
    import torch

    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_text})

    # transformers ≥5.x devuelve BatchEncoding; ≤4.x devuelve tensor directo
    _chat_out = tok.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True
    )
    if hasattr(_chat_out, "input_ids"):
        input_ids = _chat_out.input_ids.to(next(model.parameters()).device)
    else:
        input_ids = _chat_out.to(next(model.parameters()).device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            eos_token_id=EOS_ID,          # stop-mode legacy
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    # hidden_states: tuple(T_gen) de tuple(n_layers+1) de (batch, seq, hidden)
    # Tomamos capa final (índice -1), batch 0, posición -1 (token recién generado)
    hidden_vecs = []
    for step_hs in out.hidden_states:
        h = step_hs[-1][0, -1, :].float().cpu().numpy()
        hidden_vecs.append(h)

    hidden_matrix = np.stack(hidden_vecs, axis=0).astype(np.float32)
    response_ids  = out.sequences[0, input_ids.shape[1]:]
    response_text = tok.decode(response_ids, skip_special_tokens=True)

    return hidden_matrix, response_text


def run_batch(tok, model, cond_name: str, sys_prompt: str | None,
              battery_name: str, items: list[dict],
              out_dir: Path, force: bool, dry_run: bool, smoke: bool):

    key      = f"{cond_name}_{battery_name}"
    npz_path = out_dir / f"{key}_embeddings.npz"
    js_path  = out_dir / f"{key}_responses.json"

    if npz_path.exists() and js_path.exists() and not force:
        print(f"  {key}: ya existe — saltando (--force para recalcular)", flush=True)
        return

    if smoke:
        items = items[:2]

    N         = len(items)
    responses = []

    if dry_run:
        print(f"  [DRY] {key}: {N} prompts (simulado)", flush=True)
        hiddens = [
            np.random.randn(np.random.randint(30, MAX_TOKENS), HIDDEN_DIM).astype(np.float32)
            for _ in range(N)
        ]
        for item in items:
            responses.append({"id": item["id"], "prompt": item["text"], "response": "[DRY]"})
    else:
        print(f"  {key}: {N} prompts…", flush=True)
        hiddens = []
        for i, item in enumerate(items):
            h, text = extract_one(tok, model, sys_prompt, item["text"])
            hiddens.append(h)
            responses.append({"id": item["id"], "prompt": item["text"], "response": text})
            print(f"    [{i+1:02d}/{N}] id={item['id']} T={h.shape[0]}", flush=True)
        gc.collect()

    T_max   = max(h.shape[0] for h in hiddens)
    emb_arr = np.zeros((N, T_max, HIDDEN_DIM), dtype=np.float32)
    lengths = np.zeros(N, dtype=np.int32)
    for i, h in enumerate(hiddens):
        L = h.shape[0]
        emb_arr[i, :L] = h
        lengths[i] = L

    np.savez_compressed(npz_path, embeddings=emb_arr, lengths=lengths)
    js_path.write_text(json.dumps(responses, ensure_ascii=False, indent=1))
    print(f"  {key}: guardado — shape={emb_arr.shape}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="E-GRAV extractor GPU")
    parser.add_argument("--out",         default="/workspace/egrav_results",
                        help="Directorio de salida")
    parser.add_argument("--conditions",  nargs="+", default=list(CONDITIONS.keys()),
                        choices=list(CONDITIONS.keys()))
    parser.add_argument("--batteries",   nargs="+", default=list(BATTERIES.keys()),
                        choices=list(BATTERIES.keys()))
    parser.add_argument("--force",  action="store_true", help="Recalcular aunque exista")
    parser.add_argument("--dry-run", action="store_true", help="Sin GPU, datos ficticios")
    parser.add_argument("--smoke",   action="store_true", help="Solo 2 prompts por corrida")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = len(args.conditions) * len(args.batteries)
    n_prompts = 2 if args.smoke else 20
    print(f"\nE-GRAV extractor · {len(args.conditions)} cond × "
          f"{len(args.batteries)} baterías = {n_jobs} corridas × {n_prompts} prompts")
    print(f"Output: {out_dir}\n", flush=True)

    # Verificar archivos
    for b in args.batteries:
        if not BATTERIES[b].exists():
            sys.exit(f"ERROR: batería '{b}' no encontrada en {BATTERIES[b]}")
    for c in args.conditions:
        p = CONDITIONS[c]
        if p is not None and not p.exists():
            sys.exit(f"ERROR: prompt '{c}' no encontrado en {p}")

    hf_token = os.environ.get("HF_TOKEN")
    tok = model = None

    if not args.dry_run:
        model_path = get_model_path(hf_token)
        tok, model = load_model(model_path)

    for cond in args.conditions:
        sys_prompt = CONDITIONS[cond]
        sys_text   = sys_prompt.read_text(encoding="utf-8") if sys_prompt else None
        sys_label  = "none" if sys_text is None else f"{len(sys_text):,} chars"
        print(f"\n── {cond}  (system={sys_label})")

        for bat in args.batteries:
            items = json.loads(BATTERIES[bat].read_text(encoding="utf-8"))
            run_batch(tok, model, cond, sys_text, bat, items,
                      out_dir, args.force, args.dry_run, args.smoke)

    print(f"\n✓ Listo. Resultados en: {out_dir}")
    print("Siguiente: scp los resultados y corre analyze_egrav.py localmente.")


if __name__ == "__main__":
    main()
