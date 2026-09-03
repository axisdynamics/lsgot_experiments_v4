#!/usr/bin/env python3
"""
Descarga parcial de W_U (embed_tokens, tied) + normas de gemma-4-31B-it.

Descarga solo los tensores necesarios para el logit-lens vía HTTP Range,
sin bajar el shard completo (~31GB): embed_tokens (2.8GB bf16→f32) y las
normas (final + 4 tipos × 60 capas, ~10KB c/u).

Uso:
    python fetch_wu_partial.py --token hf_xxxxx --out-dir /tmp/wu
    # o HF_TOKEN=hf_xxxxx python fetch_wu_partial.py
"""
import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import requests

REPO = "google/gemma-4-31B-it"
REV = "842da3794eaa0b77d5f08bae87a17459d91ff475"
SHARD = "model-00001-of-00002.safetensors"
NORM_TYPES = ["input_layernorm", "post_attention_layernorm",
              "pre_feedforward_layernorm", "post_feedforward_layernorm"]
N_LAYERS = 60


def bf16_to_f32(buf):
    u16 = np.frombuffer(buf, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--out-dir", default="/tmp/wu_norm")
    args = ap.parse_args()
    assert args.token, "requiere --token o HF_TOKEN"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    HDR = {"Authorization": f"Bearer {args.token}"}
    URL = f"https://huggingface.co/{REPO}/resolve/{REV}/{SHARD}"

    # header del shard (primeros 2MB)
    r = requests.get(URL, headers=HDR, stream=True, timeout=60)
    r.raise_for_status()
    head = next(r.iter_content(2 * 1024 * 1024))
    header_len = struct.unpack("<Q", head[:8])[0]
    meta = json.loads(head[8 : 8 + header_len].decode("utf-8"))

    def fetch(name, t):
        start = 8 + header_len + t["data_offsets"][0]
        end = 8 + header_len + t["data_offsets"][1]
        buf = bytearray(end - start)
        pos = 0
        with requests.get(URL, headers={**HDR, "Range": f"bytes={start}-{end-1}"},
                          stream=True, timeout=600) as rr:
            rr.raise_for_status()
            for chunk in rr.iter_content(8 * 1024 * 1024):
                buf[pos : pos + len(chunk)] = chunk
                pos += len(chunk)
        arr = bf16_to_f32(buf)
        assert arr.size == int(np.prod(t["shape"])), f"{name}: tamaño inesperado"
        arr = arr.reshape(t["shape"])
        np.save(out / f"{name}.npy", arr)
        print(f"{name}: {t['shape']} guardado")

    fetch("embed_tokens", meta["model.language_model.embed_tokens.weight"])
    fetch("final_norm", meta["model.language_model.norm.weight"])

    (out / "per_layer_norms").mkdir(exist_ok=True)
    n_found = 0
    for L in range(N_LAYERS):
        for nt in NORM_TYPES:
            key = f"model.language_model.layers.{L}.{nt}.weight"
            if key not in meta:
                continue
            t = meta[key]
            start = 8 + header_len + t["data_offsets"][0]
            end = 8 + header_len + t["data_offsets"][1]
            rr = requests.get(URL, headers={**HDR, "Range": f"bytes={start}-{end-1}"},
                              timeout=120)
            arr = bf16_to_f32(rr.content)
            arr.tofile(out / "per_layer_norms" / f"L{L}_{nt}.f32")
            n_found += 1
    print(f"normas por capa: {n_found}")

    # ÍNDICE de shards para localizar el resto de normas si hiciera falta
    print("hecho. Nota: si alguna norma vive en el shard 2, bajar el índice y adaptar.")


if __name__ == "__main__":
    main()
