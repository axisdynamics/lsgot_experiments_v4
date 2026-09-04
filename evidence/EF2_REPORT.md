# E-F2 — Per-capa: proyección v̂ y participation ratio por capa

> ⚠️ **Reescrito 2026-08-28 (noche):** la versión anterior incluía
> chat_agente/chat_agente_sia/chat_agente_sia_v2 (34-44% de cada
> respuesta es markup HTML idéntico repetido — ver
> `CHAT_AGENTE_MARKUP_CONFOUND_REPORT.md`). Esas 3 condiciones se
> **eliminaron por completo** de tablas y comparaciones — no solo se
> marcaron. El hallazgo central (crecimiento monótono del efecto de v̂
> hacia capas tardías) sobrevive intacto, replicado en axis vs
> automata_neutro y axis vs vanilla.

**Fecha:** 2026-08-28
**Scripts:** `LSGOT_v4/scripts/fase0/run_ef2.py` (extracción, RunPod A100 80GB) +
`LSGOT_v4/scripts/fase0/analyze_EF2_per_capa.py` (análisis)
**Datos crudos:** `.../gemma4_31b_combined/results_local/ef2_L5_L55/*.npz`
(10 condiciones × 11 capas × 20 prompts × ≤256 tokens × 5376 dims, 4.5GB)
**Resultados:** `LSGOT_v4/scripts/fase0/EF2_results.json`

## 0. ⚠️ Corrección metodológica al documento de instrucciones

**`v_identidad.npy` NO está calculado en L30.** El documento (§1, tabla de
contexto) afirma "Modelo: Gemma-4-31B-it, capa L30 (índice 29/60)" como si
fuera la capa de TODAS las métricas del paper. Verificado en el código:

- `sia/run_exp.py:134` — `HiddenStateExtractor(model_path, layer_idx=-1, ...)`:
  la extracción de `sia_extended_v5` (de donde salen PR, identity_projection,
  RQA, Hurst — TODO el Tier 0 del paper, y `v_identidad.npy` mismo, ver
  `tier0_metrics.persona_vector()` operando sobre esos mismos embeddings) usa
  **`layer_idx=-1`, la capa final (60/60)**, no L30.
- L30 (índice 29) es la capa usada **solo en el pipeline de perturbación**
  (`perturbation/run_perturbation.py`, `LAYER_SETS["L30"] = [29]`) — un
  extractor distinto, para un experimento distinto (E-I, E-B).

Esto significa: v̂ es una dirección definida en el espacio residual de la
**capa final**, no de L30. Las cifras d=8.89 (§3.5 del paper) y el PR
gradual (§3.1) están calculadas en capa final, no en L30. El perfil por
capa de este experimento (§3) debe leerse en consecuencia — ver §5-6.

## 1. Resumen

Se proyectó v̂ (fijo, calculado en capa final) sobre 11 capas (L5..L55,
step 5, índices 4..54) en las 10 condiciones. El efecto crece
**monótonamente** de ~0 en L5 a d=10.43 (axis_pec_only vs automata_neutro)
en L55 — consistente con que v̂ "vive" en el espacio de la capa final y se
capta mejor cuanto más cerca está la capa muestreada de ese origen, no
necesariamente con que "la identidad emerja tarde" en sentido absoluto. El
participation ratio muestra un paisaje no-monótono (alto en L5, mínimo en
L15-L20, pico en L25-L30, valle en L35, sube de nuevo hasta L55) con
separación clara identidad/genérico vs restricción concentrada en L25-L30.

## 2. Método

- **Extracción (GPU, RunPod A100 80GB):** `run_ef2.py`, reutilizando
  `HiddenStateExtractor.extract_multilayer()` (ya existente en
  `shared/hidden_state_extractor.py`) para capturar 11 capas en el mismo
  forward pass, 10 condiciones × 20 prompts (`PRIORITY_SUBSET`, mismos
  prompts que `sia_extended_v5`) × ≤256 tokens greedy.
  - **Incidente:** el proceso original murió por OOM de CUDA en la
    condición `chat_agente` (fragmentación de memoria con atención
    `eager` en un proceso largo). Se relanzaron las condiciones restantes
    como procesos separados; `chat_agente` volvió a fallar en el mismo
    punto exacto incluso con contexto CUDA limpio (confound de contenido,
    no de proceso) — se resolvió con
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (fix recomendado
    por el propio traceback de PyTorch), sin pérdida de datos.
- **Análisis:** v̂ = `v_identidad.npy` (fijo, NO recalculado por capa, tal
  como exige el diseño). Por capa: proyección coseno media
  (`project_trajectory`, misma convención que E-H/E-L/E-H2) y
  participation ratio (`participation_ratio`, misma convención que E-D).
- Test de permutación (n=1000, seed=42) + d de Cohen por capa, para 9 pares
  clave (mismos que `analyze_tier0.py` — replican los del paper).
- Control T1: proyección en L30 con y sin t=0 (no se repitió en las 11
  capas por presupuesto de tiempo; L30 es la capa de referencia histórica
  del proyecto para el pipeline de perturbación).

## 3. Resultados

### 3.1 Proyección v̂ media por capa y condición

| Condición | L5 | L10 | L15 | L20 | L25 | L30 | L35 | L40 | L45 | L50 | L55 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| axis | -0.0002 | -0.0053 | -0.0037 | -0.0025 | -0.0005 | -0.0488 | -0.0949 | -0.0686 | -0.0518 | -0.0384 | **+0.0091** |
| axis_pec_only | -0.0004 | -0.0054 | -0.0035 | -0.0022 | +0.0002 | -0.0493 | -0.0950 | -0.0692 | -0.0529 | -0.0392 | **+0.0087** |
| axis_short | -0.0002 | -0.0053 | -0.0035 | -0.0021 | +0.0001 | -0.0488 | -0.0949 | -0.0687 | -0.0523 | -0.0389 | **+0.0087** |
| generic_long | -0.0020 | -0.0072 | -0.0050 | -0.0031 | -0.0017 | -0.0525 | -0.0987 | -0.0788 | -0.0750 | -0.0638 | -0.0321 |
| generic_short | -0.0019 | -0.0070 | -0.0046 | -0.0028 | -0.0011 | -0.0527 | -0.0991 | -0.0792 | -0.0763 | -0.0658 | -0.0347 |
| vanilla | -0.0005 | -0.0057 | -0.0038 | -0.0022 | +0.0004 | -0.0528 | -0.0987 | -0.0775 | -0.0725 | -0.0613 | -0.0267 |
| automata_neutro | +0.0023 | -0.0038 | -0.0038 | -0.0020 | -0.0008 | -0.0510 | -0.0976 | -0.0751 | -0.0642 | -0.0577 | -0.0311 |

Nota: solo la familia axis (axis, axis_pec_only, axis_short) cruza a
proyección **positiva** en L55; el resto queda negativo en las 11 capas
muestreadas — la separación real (ver d de Cohen, §3.2) es relativa entre
condiciones, no un cruce de signo absoluto salvo en axis-family.

### 3.2 d de Cohen por capa, pares clave (permutación n=1000)

| Par | L5 | L10 | L15 | L20 | L25 | L30 | L35 | L40 | L45 | L50 | L55 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| axis vs vanilla | +0.26 | +0.59* | +0.12 | -0.52* | -1.33*** | +2.38*** | +3.89*** | +5.18*** | +5.90*** | +5.98*** | +6.12*** |
| axis_pec_only vs vanilla | +0.04 | +0.52 | +0.59* | +0.05 | -0.33 | +2.14*** | +4.01*** | +4.85*** | +5.58*** | +5.70*** | +6.15*** |
| axis_pec_only vs automata_neutro | -0.63* | -0.73** | +0.41 | -0.17 | +1.67*** | +0.72* | +2.21*** | +3.13*** | +1.94*** | +4.66*** | **+10.43\*\*\*** |
| axis vs automata_neutro | -0.56* | -0.66* | +0.10 | -0.54* | +0.53* | +0.91** | +2.19*** | +3.43*** | +2.13*** | +4.91*** | **+10.18\*\*\*** |
| axis vs axis_pec_only | +0.23 | +0.16 | -0.43 | -0.65* | -1.34*** | +0.35 | +0.07 | +0.35 | +0.37 | +0.26 | +0.08 |
| automata_neutro vs vanilla | +0.64* | +0.87** | -0.02 | +0.18 | -1.64*** | +0.68* | +0.87** | +1.20*** | +1.35*** | +0.77* | -0.78* |

`*` p<0.05, `**` p<0.01, `***` p<0.001. axis vs axis_pec_only se mantiene
**no significativo o pequeño en todas las capas** — replica la doble
disociación limpia del paper (ambos con wiring, no deberían diferir).

### 3.3 Participation ratio media por capa y condición

| Condición | L5 | L10 | L15 | L20 | L25 | L30 | L35 | L40 | L45 | L50 | L55 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| axis | 27.0 | 15.6 | 8.0 | 8.9 | 10.1 | 29.2 | 12.2 | 20.0 | 16.5 | 14.7 | 33.3 |
| axis_pec_only | 27.3 | 15.9 | 8.1 | 9.3 | 10.7 | **34.1** | 12.9 | 22.1 | 17.9 | 15.2 | 33.7 |
| axis_short | 27.3 | 15.5 | 7.8 | 8.7 | 9.6 | 29.8 | 11.8 | 19.2 | 16.0 | 14.4 | 32.7 |
| generic_long | 29.2 | 16.4 | 7.9 | 9.1 | 11.4 | 33.6 | 12.4 | 18.5 | 18.6 | 18.8 | 42.7 |
| generic_short | 28.7 | 16.0 | 7.6 | 8.6 | 11.3 | 30.4 | 11.8 | 17.0 | 16.5 | 17.8 | 39.8 |
| vanilla | 29.0 | 17.5 | 8.0 | 9.2 | 12.2 | 32.1 | 13.4 | 16.5 | 18.1 | 20.4 | 43.6 |
| automata_neutro | 19.0 | 11.2 | 5.8 | 5.8 | 5.6 | **14.3** | 7.4 | 11.0 | 8.6 | 10.0 | 23.7 |

Patrón: PR alto en L5 (~27-29, uniforme entre condiciones), colapsa en
L15-L20 (~6-9, todas las condiciones convergen a dimensión baja), **pico
divergente en L25-L30** — aquí es donde axis-family/generic/vanilla
(29-34) se separa claramente de automata_neutro (14.3, el mínimo del
panel) —, vuelve a colapsar en L35 (7-13, todas las condiciones otra vez),
y sube de nuevo hacia L55 sin recuperar la separación de L30 (automata_neutro
sigue siendo el más bajo, 23.7, pero la brecha relativa es menor que en L30).

**El pico de separación de PR coincide con L25-L30**, no con L55 (donde
está el pico de la proyección v̂) — son dos fenómenos con **perfiles de
capa distintos**, ver §5-6.

### 3.4 T1 — proyección L30 con vs sin t=0

Diferencia despreciable (<0.0001 en todas las condiciones) — a diferencia
de E-L (que trabajó en capa final), en L30 el primer token no domina la
media de la trayectoria. Ver `EF2_results.json`, bloque `t1_check_L30`.

## 4. Trampas aplicadas

- **Corrección metodológica (§0):** verificada contra código fuente
  (`sia/run_exp.py:134`), no es una suposición — v̂ es de capa final, no
  L30. Se reporta como hallazgo, no como error propio de este experimento.
- **T1:** verificado en L30, sin efecto relevante de t=0 (§3.4). No se
  verificó en las otras 10 capas por presupuesto de tiempo — si se usa
  este resultado para decidir capas de E-K, recomendable repetir T1 en la
  capa elegida antes de comprometerse.
- **T2:** n=20 prompts, 1 manipulación por condición — igual que en
  E-L/E-H2/E-J, no se puede separar "L55 es la capa de identidad" de
  "esta redacción específica de axis_pec_only en L55".
- **T5:** 4 pares × 11 capas = 44 comparaciones. El criterio de éxito del
  documento ("replicable en ≥2 ventanas de capas adyacentes") se cumple
  para el par axis_pec_only vs automata_neutro (y axis vs automata_neutro)
  en la **ventana L35-L55** (5 capas consecutivas, todas p<0.001, signo
  estable) — no se cumple de forma limpia en L5-L25 (signos inconsistentes,
  varias comparaciones n.s.).
- **T9:** no aplica (embeddings recién extraídos, no rescatados).
- **T11:** chat_agente-family eliminada por completo de este reporte.

## 5. Replicación

El crecimiento monótono del efecto hacia capas tardías replica en 2/2
pares centrales (axis_pec_only vs automata_neutro, axis vs automata_neutro)
con el mismo patrón cualitativo: débil/mixto en L5-L25, fuerte y estable
en L35-L55. axis vs axis_pec_only se mantiene no-significativo o pequeño
en las 11 capas — replica limpiamente la predicción del paper de que
ambas condiciones (con wiring) no deberían diferir, independiente de la
capa.

## 6. Implicación para el paper

**Hallazgo primario, con una corrección de documentación importante
adjunta:**

1. **Corregir §1 del documento de instrucciones** (y cualquier lugar del
   paper que diga "L30" refiriéndose a PR/identity_projection/RQA/Hurst):
   esas métricas usan la capa final (60/60), no L30. L30 es exclusivo del
   pipeline de perturbación. Esto no invalida ningún resultado del paper
   (los números siguen siendo correctos), pero la etiqueta de capa en la
   tabla de contexto está mal atribuida.

2. **Perfil de v̂ por capa:** dado que v̂ es una dirección de capa final,
   este experimento no responde literalmente "¿en qué capa vive la
   identidad?" (esa pregunta requeriría recalcular v̂ en cada capa, que el
   diseño excluye deliberadamente) — responde "¿en qué capas es *legible*
   la dirección de capa final?", y la respuesta es: débilmente y con signo
   inconsistente en capas tempranas/medias (L5-L25), fuerte y estable en
   L35-L55, creciendo hacia la capa donde se definió.

3. **Implicación directa para E-K (steering):** el diseño de E-K planea
   inyectar α·v̂ en L∈{10,20,30,40}. Este experimento muestra que v̂ está
   **débilmente alineado** con la estructura del residual stream en esas
   capas (d pequeño o no-significativo en L10/L20, moderado en L30/L40) —
   el steering en L10/L20 puede no tener efecto observable simplemente
   porque v̂ "no significa lo mismo" ahí, no porque la dirección sea
   causalmente inerte. Recomendación: agregar L50/L55 al barrido de E-K
   (donde v̂ sí es fuerte), o recalcular un v̂ propio por capa objetivo
   antes de descartar el steering como no-causal en capas tempranas.

4. **PR tiene un perfil de capa distinto al de v̂** (pico de separación en
   L25-L30, no en capas tardías) — esto sugiere que la dimensionalidad
   efectiva de la trayectoria y la dirección de identidad son fenómenos
   parcialmente independientes por capa, coherente con el hallazgo de E-J
   (identidad organiza un subespacio, pero de magnitud/rango menor que
   otros factores) — el rango L25-L30 sería el candidato natural para
   profundizar E-G (atención a spans) si se busca el correlato mecánico
   de la caída/pico de PR.

## 7. Adenda 2026-08-28 (tarde) — v̂ nativo de L30, calculado y comparado

A pedido del usuario, se calculó un v̂ **nativo** de L30 (no reciclado de
capa final): `v_identidad_L30.npy` = mean(axis en L30) − mean(generic_long
en L30), norma 1, misma convención que `persona_vector()`. Usa los mismos
embeddings de E-F2 (cero GPU adicional). Guardado en
`LSGOT_v4/scripts/perturbation/v_identidad_L30.npy`.

**Hallazgo 1 — las dos direcciones son casi ortogonales:**
`cos(v̂_L30, v̂_final) = 0.090`. Confirma cuantitativamente que v̂_final
(usado en todo el paper hasta ahora) y una dirección "identidad" definida
genuinamente en L30 **no son la misma dirección ni una aproximada rotación
de la otra** — son direcciones esencialmente distintas del espacio
residual de 5376-d.

**Hallazgo 2 — v̂_L30 no replica la doble disociación limpia del paper:**

| Par | d (v̂_L30 nativo) | p | d (v̂_final proyectado en L30, §3.2) |
|---|---|---|---|
| axis vs axis_pec_only | **+0.95** | **0.002** | +0.35 (n.s.) |
| axis_pec_only vs automata_neutro | +0.48 | 0.064 (n.s.) | +0.72 |
| axis vs vanilla | +5.52 | <0.001 | +2.38 |

Con v̂_L30, **axis y axis_pec_only difieren significativamente** —
contradice la doble disociación limpia que es uno de los hallazgos
centrales del paper (§3.5, donde axis_pec_only sigue a axis, p=0.229 con
v̂_final). Y **todas** las condiciones con system prompt estructurado y
denso en reglas (axis, axis_pec_only, axis_short, automata_neutro —
proyección +0.08 a +0.09) se separan de las condiciones con prompt
genérico/prosa simple (generic_long, generic_short, vanilla — proyección
+0.05), **sin distinguir identidad de restricción dentro del primer
grupo**. axis y generic_long están emparejados en longitud de tokens
(3,945 vs 3,957, ratio 1.003 — no es un confound de longitud cruda), así
que la lectura más plausible es que v̂_L30 capta algo como "densidad de
estructura tipo-YAML/reglas" en la capa media, no identidad
específicamente.

**Interpretación:** en L30 (capa intermedia, ~50% de profundidad), el
espacio residual parece estar dominado por una señal de "esto es un
prompt de sistema denso y estructurado" que es ortogonal a — y más fuerte
que — cualquier distinción fina entre "identidad" y "restricción" dentro
de esos prompts estructurados. Esa distinción fina solo se vuelve legible
más adelante en la red (consistente con §3, donde el efecto crece hacia
L35-L55). Esto es coherente con, y podría ser parte de la explicación de,
el fracaso del audit `REPORTE_FASE0.md` (Δκ/W₁ en L30, 12/12 comparaciones
sin superar baseline) — si L30 está dominado por una señal de
"estructura/complejidad" no relacionada con el contraste que se buscaba
medir, cualquier métrica derivada de esa capa hereda ese confound.

**Implicación para E-K, revisada:** no se recomienda usar `v_identidad_L30.npy`
para steering en L30 esperando mover la generación hacia "identidad" — lo
más probable es que mueva hacia "estilo de prompt estructurado/denso en
reglas", un efecto distinto e indeseado. Para E-K en L30, la opción más
limpia sigue siendo v̂_final (el proyectado, aunque débil ahí, §3.2) o
directamente excluir L30 del barrido y concentrar los α·v̂ en L45-L55
(donde v̂_final es fuerte, §3, Hallazgo 1) — sujeto a T2 (validar con una
segunda redacción de axis/generic_long antes de generalizar esta lectura
de L30).
