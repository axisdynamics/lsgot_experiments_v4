#!/usr/bin/env python3
"""E27 — extractor de t=0 sin generación.

t=0 es la última posición de prefill: el estado desde el que se muestrea el
primer token. No hace falta generar nada para leerlo, así que esto es un
forward por pregunta en vez de 256. Minutos en vez de media hora.

Sirve para dos cosas de la sesión de GPU:
  · re-extraer el panel de Qwen3 en la capa 32 exacta, que es la que Lu et al.
    recomiendan (hoy medimos en 31, una capa de aproximación declarada);
  · medir `axis_pec_task` en Qwen3, que es la única forma de proyectarlo sobre
    el eje de Lu — su eje vive en 5120 y Qwen3 también.

**Convención de capas, que hay que confirmar antes de usar los números.**
`output_hidden_states=True` devuelve n_layers+1 tensores: el índice 0 es la
salida del embedding y el índice i≥1 es la salida de la capa i-1. Las corridas
originales guardaron claves `embeddings_L4 … embeddings_L63` y no queda
registrado cuál de las dos convenciones usaron. Por eso este script extrae
varios índices y los nombra por el índice crudo (`hs_63`, `hs_64`): el control
del runbook dice cuál reproduce el +0.185 publicado, y ése es el que vale.

Uso:
    python extract_prefill_t0.py --model Qwen/Qwen3-32B \\
        --conditions axis axis_pec_only automata_neutro generic_long \\
                     generic_short vanilla axis_pec_task \\
        --layers 31 32 63 64 --out /workspace/qwen3_t0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-32B')
    ap.add_argument('--prompts', default='prompts',
                    help='directorio con <condicion>.txt')
    ap.add_argument('--battery', default='b_onto.json',
                    help='batería: lista de {id, text}')
    ap.add_argument('--conditions', nargs='+', required=True)
    ap.add_argument('--layers', nargs='+', type=int, required=True,
                    help='índices CRUDOS de hidden_states, no números de capa')
    ap.add_argument('--out', required=True)
    ap.add_argument('--attn', default='sdpa',
                    help='sdpa para Qwen3 y Command R, eager para Gemma (§3.2)')
    args = ap.parse_args()

    battery = json.loads(Path(args.battery).read_text())
    questions = [(q['id'], q['text']) for q in battery]
    print(f'batería: {len(questions)} preguntas · condiciones: {len(args.conditions)}')

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='auto',
        attn_implementation=args.attn)
    model.eval()
    n_hs = model.config.num_hidden_layers + 1
    print(f'modelo cargado · {model.config.num_hidden_layers} capas → '
          f'hidden_states tiene {n_hs} entradas (0 = embedding)')
    for L in args.layers:
        if not 0 <= L < n_hs:
            raise SystemExit(f'índice {L} fuera de rango [0, {n_hs - 1}]')

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    for cond in args.conditions:
        ptxt = Path(args.prompts) / f'{cond}.txt'
        if not ptxt.exists():
            raise SystemExit(f'falta el prompt {ptxt}')
        system = ptxt.read_text().strip()
        states = {L: [] for L in args.layers}
        ids = []

        for qid, qtext in questions:
            msgs = ([] if cond == 'vanilla' or not system
                    else [{'role': 'system', 'content': system}])
            msgs.append({'role': 'user', 'content': qtext})
            try:   # Qwen3 expone enable_thinking; otros tokenizers no
                text = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
            except TypeError:
                text = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
            enc = tok(text, return_tensors='pt').to(model.device)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            for L in args.layers:
                # última posición de prefill = de donde sale el primer token
                states[L].append(hs[L][0, -1, :].float().cpu().numpy())
            ids.append(qid)

        payload = {f'hs_{L}': np.stack(states[L]).astype(np.float32)
                   for L in args.layers}
        payload['prompt_ids'] = np.array(ids, dtype=np.int64)
        dest = out_dir / f'{cond}_t0.npz'
        np.savez_compressed(dest, **payload)
        shape = payload[f'hs_{args.layers[0]}'].shape
        print(f'  {cond:<18} {shape} → {dest.name}')

    (out_dir / 'META.json').write_text(json.dumps({
        'model': args.model, 'attn': args.attn, 'layers_raw_index': args.layers,
        'n_hidden_states': n_hs, 'battery': args.battery,
        'conditions': args.conditions,
        'note': 'hs_<i> es hidden_states[i]; i=0 es el embedding. '
                'Confirmar contra el valor publicado antes de usar.',
    }, indent=1, ensure_ascii=False))
    print(f'\nListo: {out_dir}')


if __name__ == '__main__':
    main()
