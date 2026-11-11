# Runbook · extracción de trayectorias en GPU

Reproduce desde cero los tensores que el paquete no redistribuye. El análisis posterior
es CPU-only y está en `REPRODUCIBILITY.md`.

**Qué reproduce.** Las N=20 preguntas ontológicas del panel (`questions/b_onto.json`) en
tres condiciones: `axis_pec_only`, `vanilla` y `automata_neutro`. Eso cubre el contraste
central del paper (§4.1, d = +5.75) y la línea de base sin system prompt. Las otras seis
condiciones del panel se corren con el mismo script apuntando `CONDITIONS` a sus archivos
de prompt en `data/sia/prompts/`.

**Qué hace falta.** Una A100 o H100 de 80 GB, `transformers`, `torch`, `numpy`, y un
`HF_TOKEN` con acceso a `google/gemma-4-31B-it` (~62 GB de descarga la primera vez).

```bash
export HF_TOKEN=hf_...            # obligatorio; sin esto la descarga falla al final
cd scripts/extraction/

# copiar los system prompts junto al script (el dict CONDITIONS los busca en ./prompts/)
mkdir -p prompts
cp ../../data/sia/prompts/axis_pec_only.txt   prompts/
cp ../../data/sia/prompts/automata_neutro.txt prompts/

# prueba de humo: 2 preguntas por condición, unos minutos
python extract_trajectories.py --smoke --out /workspace/lsgot_extraction

# corrida completa: 60 generaciones, ~12 min en A100
python extract_trajectories.py --out /workspace/lsgot_extraction
```

Es reanudable: salta las corridas ya guardadas, y `--force` recalcula. `--dry-run` corre
el pipeline entero sin GPU con datos ficticios, para validar rutas antes de pagar la hora.

**Salida.** Un NPZ por condición en `--out`, con `embeddings` de forma (N, T_max, 5376) y
`lengths` de forma (N,), más el JSON de respuestas en texto. Son los mismos arrays que
`data/panel_results/` resume: t=0 es `embeddings[i, 0, :]` proyectado sobre
`data/v_identidad.npy`.

**Parámetros fijos, por si hay que compararlos.** Greedy (sin muestreo), 256 tokens de
tope, bf16, capa final post-norm, atención `eager` en Gemma. La parada es solo en `<eos>`
(id 1) y no en `<turn|>` (id 106) — es el modo del panel original y cambiarlo cambia las
longitudes.
