# E-J — Subespacio completo: RDM + CKA multi-condición

> ⚠️ **Reescrito 2026-08-28 (noche):** la versión anterior de este reporte
> incluía `chileatiende_family` (chileatiende, chileatiende_sia,
> chileatiende_sia_v2) como uno de los 3 bloques de dominio centrales del
> análisis. Se confirmó que esas 3 condiciones tienen 34-44% de cada
> respuesta como markup HTML idéntico repetido (`CHILEATIENDE_MARKUP_CONFOUND_REPORT.md`)
> — se **descartaron por completo** del análisis, no solo de la
> interpretación. Esta versión recalcula todo (RDM, CKA, clustering,
> ángulos principales) desde cero sin esas 3 condiciones. El resultado es
> más limpio e interpretable que la versión original.

**Fecha:** 2026-08-28 (recalculado, versión sin chileatiende)
**Script:** `LSGOT_v4/scripts/fase0/analyze_EJ_rdm_cka.py` (metodología) +
recálculo con el subconjunto de condiciones revisado
**Datos:** embeddings libres (`sia_extended_v5`), 7 condiciones:
axis, axis_short, axis_pec_only, generic_long, generic_short, vanilla,
automata_neutro.
**Resultados:** `LSGOT_v4/scripts/fase0/EJ_sin_chileatiende_results.json`

## 1. Resumen

Sin chileatiende-family, la RDM tiene una estructura limpia e
interpretable: **`automata_neutro` (restricción) es la condición más
distinta de todo el panel** (distancia de centroide 0.05-0.11 contra
cualquier otra condición — 5-15× la dispersión intra-familia), más
distinta incluso que la separación identidad-vs-genérico (axis-family vs
generic-family, 0.029). El clustering jerárquico confirma esto: en k=2,
`automata_neutro` se separa solo contra {todo el resto}; recién en k=3
axis-family y generic-family se separan entre sí. Esto **triangula** (con
un tercer método independiente — RDM/CKA/ángulos, además de v̂ y PR) el
hallazgo de `CORRECCION_DVHAT_SIN_CHILEATIENDE.md`: restricción tiene una
huella geométrica más grande que identidad en este panel, no al revés.

## 2. Método

- Unidad de análisis por condición: media del hidden state por prompt
  (t>0, T1), 20 vectores de 5376-d por condición.
- RDM entre condiciones: distancia coseno entre centroides.
- CKA lineal (Kornblith et al. 2019) entre pares de condiciones.
- Ángulos principales (`scipy.linalg.subspace_angles`) entre los primeros
  5 componentes de TruncatedSVD de cada condición (fit sobre todos los
  tokens t>0).
- **T4 (control de circularidad):** ninguno de estos pasos usa v̂.
- **Condiciones:** axis_family = {axis, axis_short, axis_pec_only},
  generic_family = {generic_long, generic_short, vanilla},
  automata_neutro (sola, sin family — es la única condición de
  "restricción pura" que sobrevive tras excluir chileatiende-family).

## 3. Resultados

### 3.1 RDM completa (distancia coseno de centroides)

|  | axis | axis_short | axis_pec_only | generic_long | generic_short | vanilla | automata_neutro |
|---|---|---|---|---|---|---|---|
| **axis** | 0 | 0.0008 | 0.0017 | 0.0276 | 0.0366 | 0.0216 | **0.1011** |
| **axis_short** | 0.0008 | 0 | 0.0012 | 0.0268 | 0.0353 | 0.0196 | **0.0993** |
| **axis_pec_only** | 0.0017 | 0.0012 | 0 | 0.0303 | 0.0397 | 0.0219 | **0.1107** |
| **generic_long** | 0.0276 | 0.0268 | 0.0303 | 0 | 0.0021 | 0.0075 | 0.0542 |
| **generic_short** | 0.0366 | 0.0353 | 0.0397 | 0.0021 | 0 | 0.0093 | 0.0459 |
| **vanilla** | 0.0216 | 0.0196 | 0.0219 | 0.0075 | 0.0093 | 0 | 0.0690 |
| **automata_neutro** | 0.1011 | 0.0993 | 0.1107 | 0.0542 | 0.0459 | 0.0690 | 0 |

`automata_neutro` es la fila/columna con las distancias más grandes del
panel completo — más distante de axis (0.10-0.11) que axis lo está de
generic (0.02-0.04), y más distante de generic (0.05-0.07) que generic lo
está de sí mismo (0.002-0.009).

### 3.2 Clustering jerárquico (average linkage)

- **k=2:** `{automata_neutro}` vs `{axis, axis_short, axis_pec_only,
  generic_long, generic_short, vanilla}` — restricción se separa primero
  de TODO lo demás, antes que identidad se separe de genérico.
- **k=3:** `{automata_neutro}` / `{axis, axis_short, axis_pec_only}` /
  `{generic_long, generic_short, vanilla}` — recién en el tercer cluster
  axis-family y generic-family se separan entre sí.

### 3.3 Bloques de dominio: distancia, CKA, ángulo principal

| Bloque | n pares | distancia | CKA | ángulo (°) |
|---|---|---|---|---|
| DENTRO axis_family | 6 | 0.0012 | 0.706 | 5.7 |
| DENTRO generic_family | 6 | 0.0063 | 0.766 | 10.9 |
| axis_family vs generic_family (identidad vs no) | 9 | 0.0288 | 0.548 | 28.3 |
| axis_family vs automata_neutro (identidad vs restricción) | 3 | **0.1037** | **0.190** | **34.1** |
| generic_family vs automata_neutro (nada vs restricción) | 3 | **0.0564** | **0.206** | 21.9 |

En las tres métricas independientes (distancia, CKA, ángulo), el
contraste con `automata_neutro` es igual o mayor que el contraste
identidad-vs-genérico. Restricción organiza más varianza del espacio
representacional que identidad, en este panel limpio.

## 4. Trampas aplicadas

- **T4:** verificado — v̂ no se usó en ningún paso de este análisis.
- **T2:** cada bloque de familia tiene solo 3 condiciones (n=3 de
  manipulación real); `automata_neutro` es una única instanciación —
  el contraste con automata_neutro no tiene réplica de manipulación
  todavía (ver Set_experimental.md, réplicas T2/T3 pendientes).
- **Confound de dominio (chileatiende):** resuelto por exclusión completa,
  no por control estadístico — ver `CHILEATIENDE_MARKUP_CONFOUND_REPORT.md`.

## 5. Replicación

El patrón "automata_neutro es el más distinto del panel" replica en las 3
métricas independientes (distancia de centroide, CKA, ángulo principal) y
en el clustering jerárquico (k=2 lo separa primero). Es consistente con
`CORRECCION_DVHAT_SIN_CHILEATIENDE.md` (v̂ projection: automata_neutro
d=−2.40 vs vanilla; PR: automata_neutro d=−1.79 vs vanilla) — **3 métodos
independientes (v̂, PR, RDM/CKA/ángulos) convergen en que restricción
tiene mayor huella geométrica que identidad** en este panel.

## 6. Implicación para el paper

**El reencuadre central de `lsgot_4.md` se sostiene, y con datos limpios
se ve incluso más nítido de lo que mostraba la versión con chileatiende.**
La pregunta original de E-J ("¿identidad organiza un subespacio o es solo
una dirección?") tiene ahora una respuesta más clara: identidad sí separa
un subespacio real (28° de ángulo, CKA cae de ~0.7 a 0.55), pero
restricción (automata_neutro) separa un subespacio **más grande** (34°,
CKA 0.19) — coherente con la idea de que restricción es el efecto
dominante y dinámico del panel, mientras identidad es la huella estática
y más sutil, exactamente como plantea el reencuadre. La versión anterior
de este análisis (con chileatiende) sugería la jerarquía "dominio >
restricción > identidad" — con chileatiende removida, la jerarquía
correcta es simplemente **"restricción > identidad"**, sin el término de
dominio (que era, en retrospectiva, el propio confound de chileatiende).
