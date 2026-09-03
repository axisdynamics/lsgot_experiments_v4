# Hipótesis workspace: qué quedó probado y qué falta (2026-09-02)

**Pregunta:** ¿el ancla de identidad en t=0 (v̂ proyectado antes de generar)
es un readout del "global workspace" de
[Anthropic 2026](https://transformer-circuits.pub/2026/workspace/index.html)?
**Modelo:** Gemma-4-31B-it (60 capas) — distinto del Claude Sonnet 4.5 de
Anthropic y del gemma-4-e4b-it (42 capas) del repo hermano
`axisdynamics/workspace-8b` (réplica pre-registrada del J-lens, reporte:
`report_ES.md` en ese repo).

## 1. Lo que quedó PROBADO

### 1.1 El lens en la capa final está validado al 100% (nuevo resultado de infraestructura)

Los estados t=0 guardados por el pipeline (`sia_extended_v5`, capa final)
son **post-final-norm** (norma ≈ 460 ≈ √5376 × 6.28, la mediana del peso
de la norm final descargada por HTTP Range del shard 1 de
`google/gemma-4-31B-it`). El logit-lens correcto es **sin norm adicional**:

```
logits = W_U @ h_t0        # W_U = embed_tokens (tie_word_embeddings=True, bf16→f32)
```

**Match-rate argmax(readout) vs primer token real: 20/20 en `axis`,
`vanilla`, `automata_neutro`, `generic_long`.** El estado de contexto en
t=0 contiene exactamente la distribución del primer token que el modelo
genera (greedy) — dato relevante para la auditoría de primer token (§3.4
del paper): el confound de primer token es real y total en la capa final.

Para reproducir: descarga parcial de `embed_tokens` + `norm` vía HTTP
Range (script de la sesión, 2.8GB en vez del shard completo de 31GB),
`scripts/fase4_t2/analyze_logit_lens_t0.py` (la versión "sin norm" es la
validada).

### 1.2 Perfil por capas del ancla (geometría, consistente internamente)

`scripts/fase4_t2/analyze_workspace_t0.py` sobre `ef2_L5_L55` (11 capas,
7 condiciones). Protocolo B (v̂ por capa, calculado de medias de
trayectoria, sin circularidad) — disociación central
`axis_pec_only` vs `automata_neutro` en t=0:

| L5 | L10 | L15 | L20 | L25 | L30 | L35 | L40 | L45 | L50 | L55 |
|---|---|---|---|---|---|---|---|---|---|---|
| +0.9 | +6.9 | +10.1 | +5.1 | +7.5 | **+0.5 (p=0.054, n.s.)** | **+34.5** | +17.6 | +16.9 | +15.6 | +12.9 |

Débil abajo (L5), fuerte L10-L55, **dip en L30 (centro)**, **pico en L35**.

### 1.3 Consistencia con las costuras de workspace-8b

Costuras del e4b (42 capas) → equivalentes en 60 capas por fracción:

| costura e4b | fracción | equiv. 60c | ΔQ marginal | observación nuestra |
|---|---|---|---|---|
| L9 | 21.4% | L13 | +0.0116 | ancla fuerte ya en L10-L15 — coincide con el **modo secundario L9 (14% de réplicas)** del onset bimodal sellado en el e4b |
| L15 | 35.7% | L21 | +0.0160 | ancla sostenida L20-L25 |
| L23 | 54.8% | L33 | **+0.0337 (dominante)** | **pico d=+34.5 en L35 — la disociación es máxima justo en la salida del workspace** |
| L40 | 95.2% | L57 | +0.0087 | ancla aún fuerte en L55 (92%, dentro); decae después (capa final d≈+4.1) |

Lecturas: (a) el onset "temprano" no contradice a Anthropic — es la
segunda moda del onset bimodal que el propio e4b selló como
inconcluyente; (b) el dip en L30 (50%) cae en el centro de la banda
L21-L33: las condiciones convergen al contenido común de la tarea antes
de divergir en la salida; (c) la rotación de v̂ colapsa máximamente en un
paso L30→L35 (cos 1.0→0.505) — cruzando la costura de salida.

## 2. Lo que NO se pudo probar (y por qué)

### 2.1 El test de verbalizabilidad por capa necesita re-extracción GPU

- Los estados `ef2_L5_L55` están en una base no identificable: 7
  convenciones de lens probadas (raw, final-norm, rms-solo, y 4 normas de
  capa como divisor, con y sin off-by-one) — **ninguna decodifica** en
  L55, mientras la capa final sí (20/20). El script de extracción
  (`run_ef2.py`, del pod) está perdido; los estados tienen estructura
  geométrica coherente pero no son residuales crudos ni post-norm en
  ninguna norma conocida.
- **Y aunque lo fueran, el lens ingenuo no basta**: el método de
  Anthropic es el J-lens (Jacobiana de la capa final respecto a la capa
  ℓ). `W_U @ h_ℓ` solo es válido cerca de la final (J≈I). En la banda
  media se necesita la Jacobiana → modelo completo en GPU.

### 2.2 Predicciones registradas para la re-extracción

Re-extracción limpia (pod A100, ~40 min): 7 condiciones × 20 prompts,
estados t=0 por capa con `output_hidden_states=True` **verificando la
base** contra el lens final validado (20/20), + Jacobianas J_ℓ para el
J-lens. Predicciones:

1. El readout verbal del auto-chequeo debería aparecer entre L13-L33
   (banda del workspace en este modelo) y "snap" hacia el primer token
   real en L33-L35 (costura de salida).
2. El dip de L30 debería verse como convergencia del readout al contenido
   común (la pregunta), sin separación identidad-vs-restricción.
3. La separación máxima del readout entre condiciones debería coincidir
   con el pico geométrico L35.

## 3. Estado del análisis

- `workspace_t0_results.json` — perfil por capas (protocolos A/B) + rotaciones.
- `logit_lens_t0_results.json` — solo el test de capa final válido (el
  barrido por capa con EF2 falló en la base, no en el método).
- Norms por capa descargadas (240 archivos f32) y W_U (5.3GB) en el
  scratchpad de la sesión — reutilizables para el J-lens si se re-extrae.

## 4. Re-extracción 2026-09-03: infraestructura validada, JVP corregido pendiente

### 4.1 Qué se validó (infraestructura, nuevo)

- Entorno real: transformers 5.16.1 + torch 2.14.0+cu130 (gemma4 NO existe
  en transformers <5.x). Convención del lens confirmada a nivel de código:
  `hs[-1]` ya es post-final-norm (la lm_head lo consume directo — el "sin
  norm adicional" de §1.1); el h_59 crudo se captura con hook en `lm.norm`.
  Base check re-validado en la re-extracción: 20/20 en 5/6 condiciones
  rescatadas; axis 18/20 y generic_long 19/20 en saved-vs-real (near-ties
  por la acumulación f32 de esta corrida).
- Grafo reducido de 1 posición validado bitwise en la atención (máscaras
  por tipo de capa vía `transformers.masking_utils`, cache 5.x con estado
  past por slicing, `allow_bf16_reduced_precision_reduction=False`). El
  primal reproduce el forward original salvo ruido ULP (0.0 en L59, 8e-2 en
  L58, ~1.4-2.5 en capas profundas a T≈4K) — los ops de redondeo bf16 son
  identidad en el backward, así que los readouts JVP no heredan ese ruido.
  J_59 analítico: err rel 2.6e-3.
- **Instrumento J-lens = JVP exacto por muestra** (la definición de
  Anthropic: J_ℓ de la propia muestra aplicada a su estado), un dual-forward
  por (prompt, capa). El J̄ promediado (pre-registro A1 §4.1) quedó
  OPCIONAL (`--jbar`): dos muros medidos con contextos de ~4K tokens —
  cotangentes (kv, D, C) = 33GB en jacrev a C=256; ~0.5GB/capa de duales
  retenidos en el jvp full-stack (L30 = +15GB → OOM a C=8); ~17ms de
  overhead functorch por llamada (5.6M llamadas = 26h); y una retención de
  3.9GB/chunk al materializar tangentes functorch en GPU (la vía CPU numpy
  es plana). El pre-registro asumía corpus de 256 tokens.

### 4.2 Rescate y bug del readout (pendiente de corregir)

- El pod murió a mitad de la corrida: rescatadas 6/7 condiciones (falta
  automata_neutro); meta.json no se escribió.
- **Los readouts JVP guardados son BASURA**: bug del merge del topk chunked
  (`cat([vals, bestv])[:2k]` cortaba el best acumulado — el top-50 resultante
  eran los primeros 100 valores del último chunk, constantes en todas las
  capas). Corregido en `extract_layers_jlens.py` (commit e27dcd9); los
  readouts deben regenerarse en un pod nuevo (~1h con descarga del modelo;
  `--skip-primal` ahorra la mitad).
- Mientras tanto, el lens ingenuo (válido, computado localmente desde los
  estados rescatados + W_U): match@1 = 0.0 en L5-L50 y 0.9-1.0 en L59 (el
  snap del lens ingenuo es tardío, >L50); separación JS por capa oscilante:
  máx L0 (0.69), ~0 en L5-L25, L30=0.32, L35=0.0 (valle), L40-L50≈0.63-0.69,
  L55=0.0, L59=0.69. Las predicciones P1-P3 siguen PENDIENTES del readout
  JVP corregido (los criterios están pre-especificados en el docstring de
  `analyze_j_lens.py`).
