# FASE 6 — ¿el testigo cableado modula los pesos de atención en t=0?

Propuesta: `Descargas/Experimento_atencion.md` (2026-09-07). Pregunta nueva:
la disociación de v̂ en t=0 (hidden states, d hasta +9.78) ¿viene con una
**reponderación de la atención**, o es sólo un sesgo aditivo en el residual
stream? Toda la instrumentación previa engancha la *salida* de cada capa;
nadie miró los pesos de atención (verificado, §2·bis.1 de la propuesta).

## Archivos

| archivo | qué hace |
|---|---|
| `run_attention_t0_extraction.py` | **GPU (pod).** Un forward de contexto puro (t=0) por (condición, prompt), `attn_implementation="eager"`. Hooks sólo en `self_attn` de las capas objetivo; el tensor `(1,H,T,T)` se reduce a métricas `(H,)` **dentro del hook** y se descarta → pico ≈ 1 capa, no las 11 (§5.1). Guarda vectores resumidos. |
| `analyze_attention_t0.py` | **Local, sin GPU.** Permutación + d de Cohen (`shared/statistical_tests.py`). Familia primaria acotada + Holm; resto exploratorio. Veredicto H_atención-0/1/2. |
| `witness_spans.json` | Spans del testigo / paso pre-respuesta por condición (marcadores `[start, end]`), verificados contra `data/sia/prompts/` el 2026-09-07. Editable. |

## Correr

```bash
# 1) en el pod (A100/H100 80GB), con el modelo o el token HF:
python run_attention_t0_extraction.py --sanity --token hf_xxx      # axis×2: T, spans, H, ETA
python run_attention_t0_extraction.py --token hf_xxx               # corrida completa -> results_attn_t0/

# 2) traer results_attn_t0/ a local y:
python analyze_attention_t0.py --data-dir results_attn_t0
```

`--sanity` es el primer chequeo obligatorio: confirma que los hooks reciben
los pesos de atención en esta versión de `transformers` (si no, imprime
`[ERROR] la capa X no devolvió pesos de atención`), que los spans se
encuentran (`spans_found`), y da un ETA. La corrida completa son 8
condiciones × 20 prompts × 11 capas en un solo forward por prompt
(sin bucle de generación) — del orden de 15–25 min.

## Diseño (resumen; detalle en la propuesta)

**Panel (§3.1, primera pasada — se omiten `axis_short` y `soul_jarvis`):**

| grupo | condiciones |
|---|---|
| testigo | `axis`, `axis_pec_only`, `soul_md_corto` |
| id sin testigo | `soul_elena_financial`, `soul_solidity_auditor` |
| sin identidad | `generic_long`, `vanilla` |
| restricción autómata | `automata_neutro` |

**Capas:** labels `L20,25,28,30,32,35,38,40,45,50,55` → módulo `layers[N-1]`
(convención del proyecto). Denso alrededor de la transición L30 (dip) → L35
(pico d=+34.5) del perfil de v̂ (§2·bis.2). El script de ef2 original está
perdido → la ambigüedad ±1 capa queda absorbida por el muestreo denso.

**Métrica primaria:** `ent_last_meanhead` — entropía (÷ log T) de la
atención de la última posición de query (la que produce el primer token),
media sobre cabezas. Baja = concentrada. Familia primaria = métrica primaria
× 3 contrastes × 11 capas (tamaño comparable a los 33 tests del panel de v̂),
Holm sobre p de Mann-Whitney, umbral |d|≥0.8 y p_Holm<0.001.

**Secundario / exploratorio** (raw p, sin corrección): `ent_last_minhead`
(cabeza más concentrada), `ent_last_spread` (¿cabezas con patrón propio?),
`sink_meanhead`, `sysfrac_meanhead`, `ent_rowmean_meanhead`; pares tipo panel
de v̂; grupo `automata`; robustez `testigo_strict` (= testigo sin
`soul_md_corto`).

**Masa sobre el span** (sólo condiciones con span): within-condición,
pareado `wfrac_last` vs `cfrac_last` (span del testigo vs slice de control
de igual longitud, contiguo, del mismo system prompt) — Wilcoxon + d within +
enriquecimiento (masa / tamaño del span).

## Veredicto (§3.4)

- **H_atención-1** (testigo específico): testigo separa de *ambos* grupos
  (|d|≥0.8, p_Holm<0.001) y `id_sin_testigo` no separa de `sin_identidad`
  por el mismo estándar. Se marca si se sostiene con `testigo_strict`.
- **H_atención-2** (identidad, no testigo): `testigo` e `id_sin_testigo`
  ambos separan de `sin_identidad`; `testigo` no separa de `id_sin_testigo`.
- **H_atención-0** (nula): nada en la familia primaria cruza el umbral →
  el hallazgo de t=0 queda como estaba (robusto en hidden states, sin
  mecanismo de atención específico).
- Si `automata_neutro` también concentra: "cualquier paso cableado basta",
  no forzar la narrativa de testigo específico.

## Confounds declarados

- **`soul_md_corto`** trae andamiaje de atención autodescrito en el propio
  prompt ("HIGH ATTENTION ZONE", "the `## Guardrails` heading activates extra
  model attention"). El span del testigo cae dentro de esa zona → cualquier
  concentración ahí es en parte *por construcción*. Por eso la robustez
  `testigo_strict` y el reporte por separado.
- **Correlacional**, igual que v̂ en t=0. No establece causalidad; a lo sumo
  da una hipótesis más específica de *dónde* intervenir (cabezas concretas)
  para el próximo intento de steering (la inyección directa en v̂ falló 17/17).
- Mapeo span→token: offsets del fast tokenizer si están; si no, conteo
  incremental por prefijo (fuzz ±1–2 tokens en bordes de span; despreciable
  para spans de 50–400 tokens).
