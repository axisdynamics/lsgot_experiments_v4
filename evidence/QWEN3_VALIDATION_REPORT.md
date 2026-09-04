# Validación cross-modelo — Qwen3-32B (T10 del checklist)

**Fecha:** 2026-08-29
**Modelo:** Qwen/Qwen3-32B (dense, BF16, `enable_thinking=False`, `sdpa`)
**Scripts:** `LSGOT_v4/scripts/fase3_qwen3/run_qwen3_extraction.py` +
`analyze_qwen3_fase0.py`
**Panel:** 7 condiciones limpias (mismas que Gemma, chat_agente-family
nunca incluida), mismos 20 prompts, mismos system prompts (texto
idéntico, **no re-balanceado** por longitud de tokens de Qwen3 — ver §0).
**Datos:** `results_local/qwen3_fase0/*.npz` (12 capas: 11 proporcionales
+ capa final real L63, 2.9GB)

## 0. Nota de incidente y limitaciones de diseño

- **OOM con atención `eager`** en `generic_long` (prompt más largo bajo
  el tokenizer de Qwen3, 4794 tokens) — resuelto con
  `attn_implementation="sdpa"` (pico de VRAM bajó de >79GB a 71.5GB).
  `HiddenStateExtractor` ahora acepta `attn_implementation` como
  parámetro (antes hardcodeado a `"eager"`).
- **El emparejamiento de longitud calibrado para Gemma no se sostiene en
  Qwen3**: `axis`=3819 tokens vs `generic_long`=4794 (antes, en Gemma,
  3945 vs 3957, casi idénticos). No se rebalanceó — si la separación
  identidad/restricción aparece igual con `axis` siendo *más corto* que
  las condiciones sin identidad, juega en contra de una explicación
  alternativa trivial de "más texto = más señal".
- Sin perturbación (H4_rev) en este pase — solo trayectoria libre,
  réplica de FASE 0 (E-L, E-H2, E-J, E-F2) + A2.

## 1. Resumen — qué replica, qué no

| Hallazgo (Gemma) | ¿Replica en Qwen3? | Detalle |
|---|---|---|
| Doble disociación en t=0 (E-L) | ✅ **Sí, más fuerte** | d=+12.97 (axis_pec_only vs automata_neutro), vs d=+5.75 en Gemma |
| axis ≈ axis_pec_only (indistinguibles) | ✅ Sí (marginal) | d=+0.51, p=0.062 — no significativo, igual que en Gemma |
| PR: restricción colapsa dimensionalidad | ✅ **Sí, más limpio** | automata_neutro es el PR más bajo en las 12 capas, sin excepción (vs perfil no-monótono en Gemma) |
| v̂ crece monótonamente hacia capas tardías | ✅ Sí, con matiz | Crece igual, pero YA es grande en capas tempranas (d=4.39 en L4, 6% de profundidad) — en Gemma las capas tempranas eran ruido |
| Identidad/restricción separa el mayor subespacio del panel | ✅ Sí | ángulo axis-vs-automata=40.7°, el mayor de todos los pares |
| Restricción es la condición MÁS distinta del panel (jerarquía "restricción > identidad") | ⚠️ **Parcial/matizado** | En Qwen3 es identidad (axis-family) la que se separa primero en el clustering (k=2); automata_neutro se parece más a generic (dist=0.009) que a axis (dist=0.056). Los dos factores están más balanceados, no hay un ganador tan claro como en Gemma |
| Rotación vertical: axis≈axis_pec_only, automata_neutro rota más (A2) | ❌ **No replica la segunda mitad** | axis≈axis_pec_only sí se mantiene (72.86° vs 72.76°), pero las 7 condiciones rotan casi idéntico (72.7°-73.4°) — automata_neutro NO es un outlier en rotación en Qwen3 |

## 2. E-L — t=0, capa final

| Condición | t=0 | t>0 |
|---|---|---|
| axis | +0.192 | +0.216 |
| axis_pec_only | +0.185 | +0.193 |
| automata_neutro | +0.008 | −0.008 |
| vanilla | +0.018 | +0.006 |

| Par, t=0 | d | p |
|---|---|---|
| axis_pec_only vs automata_neutro | **+12.97** | <0.001 |
| axis vs automata_neutro | +11.83 | <0.001 |
| axis_pec_only vs vanilla | +8.61 | <0.001 |
| axis vs vanilla | +8.38 | <0.001 |
| axis vs axis_pec_only | +0.51 | 0.062 (n.s.) |

Igual que en Gemma: la disociación ya está en t=0, antes de generar. Los
tamaños de efecto son **mayores** que en Gemma (d=12.97 vs 5.75 para el
mismo contraste) — el hallazgo no solo replica, es más nítido.

## 3. E-H2 — serie temporal p(t)

| Condición | autocorr(1) | frac(p>0) | ráfaga media | mean p(t) |
|---|---|---|---|---|
| axis | 0.235 | **0.992** | 108.5 | +0.216 |
| axis_pec_only | 0.302 | 0.987 | 89.6 | +0.193 |
| vanilla | 0.545 | 0.513 | 4.0 | +0.006 |
| automata_neutro | 0.484 | 0.443 | 2.83 | −0.008 |

Mismo patrón cualitativo que Gemma (identidad: autocorrelación baja,
activación sostenida; restricción/genérico: autocorrelación alta,
ráfagas cortas) — pero **más extremo**: en Qwen3, `frac(p>0)` de
axis-family está prácticamente en el techo (0.987-0.992, contra ~0.85 en
Gemma), así que las ráfagas promedio son enormes (89-133 tokens — casi
toda la respuesta sostenida en la dirección de v̂), no ~8 tokens como en
Gemma. Ambos modelos "activan y sostienen" identidad, pero Qwen3 lo hace
de forma casi total, Gemma de forma oscilante.

## 4. E-J — RDM / CKA / ángulos principales

| Bloque | distancia | CKA | ángulo |
|---|---|---|---|
| dentro axis-family | 0.0020 | 0.819 | 10.3° |
| dentro generic-family | 0.0027 | 0.842 | 11.9° |
| axis vs generic | 0.0474 | 0.733 | 38.9° |
| **axis vs automata_neutro** | **0.0558** | **0.610** | **40.7°** |
| generic vs automata_neutro | 0.0087 | 0.622 | 19.5° |

Clustering k=2: `{axis, axis_short, axis_pec_only}` vs `{el resto,
incluido automata_neutro}` — a diferencia de Gemma, donde
`automata_neutro` se separaba primero de todos. Acá `automata_neutro`
está geométricamente **cerca de generic** (dist=0.009, la más chica del
panel entre condiciones distintas) pero **lejos de axis** (dist=0.056, la
más grande) — es identidad, no restricción, la que organiza el
subespacio más distintivo en este modelo. La separación identidad↔restricción
sigue siendo la mayor del panel en ángulo (40.7°) — coincide con Gemma en
que **esa** es la comparación más extrema — pero el modelo "quién es el
outlier" se invierte.

## 5. E-F2 — perfil por capa

v̂ crece monótonamente de L4 (~0) a L63 (axis=0.216) — mismo patrón
cualitativo que Gemma. Diferencia: el efecto axis_pec_only vs
automata_neutro **ya es grande en L4** (d=4.39, 6% de profundidad) y
sigue creciendo hasta d=10.64 en L63 — en Gemma las capas tempranas eran
ruido/inconsistentes (d cerca de 0 o de signo variable en L5-L25). Qwen3
parece "decidir" la separación identidad/restricción mucho antes en la
red.

PR: `automata_neutro` es la condición con menor PR en las **12 capas sin
excepción** (26-47 vs 31-70 del resto) — señal más limpia y consistente
que el perfil no-monótono de Gemma (que tenía un pico de separación
específico en L25-L30).

## 6. A2 — rotación vertical (hallazgo que NO replica)

| Condición | L4→L63 |
|---|---|
| axis | 72.86° |
| axis_pec_only | 72.76° |
| automata_neutro | 73.37° |
| vanilla | 73.45° |

Rango total: 72.67°-73.45° (**0.78° de diferencia entre TODAS las
condiciones**). En Gemma, `automata_neutro` rotaba 4-5° más que el resto
(62.6° vs 57.6-59.9°) — un efecto claro. En Qwen3 no hay diferenciación
por condición en absoluto — todas rotan prácticamente lo mismo. El
mecanismo geométrico que en Gemma distinguía identidad de restricción
"verticalmente" (capa a capa) **no aparece en Qwen3** — no significa que
sea falso en Gemma, pero sí que no es una propiedad universal del
fenómeno, es específica de esa arquitectura/entrenamiento.

## 7. Implicación para el paper

**El hallazgo central sobrevive la validación cross-modelo — con más
fuerza en algunos aspectos, matizada en otros:**

1. La doble disociación estática (identidad vs restricción, presente
   desde t=0) **replica robustamente**, con efectos incluso mayores, en
   una arquitectura completamente distinta (Qwen3: GQA + QK-norm, 64
   capas, tokenizer distinto) sin re-balancear las condiciones. Esto es
   evidencia fuerte de que el fenómeno no es un artefacto de Gemma-4.
2. **La jerarquía "restricción > identidad" NO es universal** — en Qwen3
   los dos factores están más equilibrados, e identidad incluso lidera
   el clustering. El paper debería presentar esa jerarquía como un
   hallazgo *de este modelo*, no como propiedad general de "identidad
   vs restricción en LLMs".
3. **A2 (rotación vertical) era una hipótesis de mecanismo específica de
   Gemma** — no generalizar esa observación sin más evidencia
   cross-modelo. Es un buen ejemplo de por qué T10 existe: algunos
   mecanismos geométricos finos no viajan entre arquitecturas aunque el
   fenómeno de alto nivel sí lo haga.
4. Recomendación: si el paper reporta esta validación, presentarla
   explícitamente como "replica en 5/6 hallazgos probados, con matices
   en la jerarquía relativa de los dos factores y un mecanismo (rotación
   vertical) que no generaliza" — no como una replicación limpia 1:1.
