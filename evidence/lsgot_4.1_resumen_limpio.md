# LSGOT 4.1 — Síntesis limpia (propuesta de reencuadre post-sanitización)

**Nota de origen:** este documento es una síntesis del agente, construida
desde cero solo con valores verificados sobre el panel limpio (sin
chat_agente-family). **No reemplaza `lsgot_4.md`** — es un insumo para
que Castillo/Torres Yévenes/Lanas decidan qué incorporar en la próxima
revisión. Cada cifra citada aquí tiene trazabilidad directa a un reporte
de la sesión 2026-08-28 (`LSGOT_v4/evidence/`).

**Modelo:** Gemma-4-31B-it, BF16, greedy, N≤256 tokens/respuesta, 20
prompts por condición (`PRIORITY_SUBSET`, todos preguntas de identidad/
introspección — no hay condición de tarea neutra en el diseño).
**Panel (7 condiciones, todas verificadas sin contaminación de markup):**
`axis`, `axis_short`, `axis_pec_only`, `generic_long`, `generic_short`,
`vanilla`, `automata_neutro`.

---

## Tesis

La identidad declarada en el system prompt deja una huella geométrica
**real, estática, direccional y semánticamente específica** en el espacio
de hidden states — pero es **menor** que la huella de la densidad de
restricción operativa, que domina la geometría del panel en casi todas
las métricas. Ninguna de las dos huellas es reducible a la otra: son
factores independientes con firmas geométricas distintas (identidad:
dirección; restricción: colapso dimensional + desviación dinámica de
ruta).

---

## Hallazgo 1 — Restricción colapsa la dimensionalidad más que identidad

Participation ratio (capa final) vs `vanilla`:

| Condición | d de Cohen | Lectura |
|---|---|---|
| `automata_neutro` | **−1.79** | restricción, sin wiring |
| `axis` | −1.51 | identidad, con wiring |
| `axis_pec_only` | −1.18 | wiring puro, sin restricción |

Gradual, no binario: ambos factores reducen la dimensionalidad efectiva
de la trayectoria, con restricción contribuyendo más.
*(`CORRECCION_DVHAT_SIN_CHAT_AGENTE.md`)*

## Hallazgo 2 — La recuperación tras perturbación está gateada por wiring, no por declaración

`recovery_id` (fracción de trayectorias que vuelven a proyectarse sobre
v̂ como su propio baseline, tras perturbación en L30):

| Condición | t=50 | t=128 | t=200 |
|---|---|---|---|
| `axis`, `axis_pec_only`, `vanilla`, `generic_long` | 0.85–1.00 | 0.85–1.00 | 0.85–1.00 |
| **`automata_neutro`** | 0.53 | 0.67 | **0.43** |

`axis` no le gana a `vanilla` — la señal real es que restricción **sin**
wiring degrada la recuperación específica de identidad (no que identidad
la mejore). El recovery geométrico grueso (τ_geom≈1.00 siempre) esconde
esta falla — son escalares disociables. Fréchet normalizado confirma lo
mismo por otra vía: `automata_neutro` se desvía de su propia ruta con
d=1.07–1.35 (p≤0.001, los 3 puntos de inyección) mientras ningún grupo
axis/generic supera "small" (d≤0.36).
*(`EE_EH_WINDOW_REPORT.md`)*

## Hallazgo 3 — Doble disociación limpia en la dirección persona-vector

v̂ = mean(axis) − mean(generic_long), capa final, norma 1.

| Comparación | d | Lectura |
|---|---|---|
| axis vs axis_pec_only | n.s. (p=0.229) | ambos con wiring — indistinguibles, como predice el modelo |
| axis vs vanilla | +3.45 | identidad se separa de lo genérico |
| axis_pec_only vs vanilla | +3.79 | wiring puro también se separa |
| automata_neutro vs vanilla | **−2.40** | restricción diverge en el sentido *opuesto* |
| **axis_pec_only vs automata_neutro** | **+5.52** | el mayor efecto limpio del panel *(antes reportado d=8.89 con chat_agente, contaminado — ver `CHAT_AGENTE_MARKUP_CONFOUND_REPORT.md`)* |

Identidad-con-wiring y wiring-puro caen del mismo lado; restricción-pura
cae del lado opuesto. No es un artefacto de longitud ni de rigidez
general del prompt — es específico del contenido de identidad.

## Hallazgo 4 — Sin atractor direccional (perturbación null)

Perturbar a lo largo vs ortogonal a v̂ no produce diferencia en
recuperación (p>0.14, 6/6 comparaciones del panel original). La huella de
v̂ es correlacional/estática, no funciona como cuenca de atracción
dinámica. *(paper original, E-I — no depende de chat_agente)*

## Hallazgo 5 — Identidad ya está en el estado de contexto, no emerge del proceso

La disociación de v̂ (Hallazgo 3) ya está presente en **t=0**, el primer
token generado, antes de que el modelo produzca ningún contenido:

| Par, en t=0 | d | p |
|---|---|---|
| axis vs vanilla | +10.22 | <0.001 |
| axis_pec_only vs vanilla | +9.78 | <0.001 |
| axis_pec_only vs automata_neutro | +5.75 | <0.001 |

El confound del primer token (ya documentado en `REPORTE_FASE0.md`) infla
la *magnitud* pero no invierte ni borra la disociación — el signo y el
orden entre condiciones son idénticos con y sin t=0. *(E-L)*

## Hallazgo 6 — Identidad y restricción tienen dinámicas temporales cualitativamente distintas

Serie p(t) = cos(h_t, v̂) token a token:

| Condición | autocorrelación(1) | ráfaga media (tokens) | frac(p(t)>0) |
|---|---|---|---|
| `axis` / `axis_pec_only` | 0.26–0.27 | **7.9–8.8** | 0.85–0.87 |
| `automata_neutro` | 0.25 | **2.8** | 0.49 |

Identidad oscila en ráfagas largas y sostenidas hacia v̂; restricción no
tiene una dinámica "anti-identidad" activa — simplemente no logra
sostener activación en esa dirección (proyección media ≈0, no negativa
de forma dinámica). *(E-H2, versión limpia — la autocorrelación elevada
que sugería la versión contaminada con chat_agente era artefacto)*

## Hallazgo 7 — Restricción organiza más subespacio que identidad

RDM/CKA/ángulos principales sobre el panel completo (t>0, sin usar v̂
para construir el análisis — control de circularidad T4):

| Bloque | distancia centroide | CKA | ángulo principal |
|---|---|---|---|
| Identidad vs genérico (axis-family vs generic-family) | 0.029 | 0.548 | 28.3° |
| **Restricción vs identidad** (automata_neutro vs axis-family) | **0.104** | **0.190** | **34.1°** |

`automata_neutro` es, en distancia bruta, **la condición más distinta de
todo el panel de 7** — más distinta de axis que axis lo está de lo
genérico. Jerarquía: restricción > identidad, en subespacio completo, no
solo en la dirección v̂. *(E-J, recalculado sin chat_agente)*

## Hallazgo 8 — El perfil por capa de v̂ no es donde se calculó (corrección)

`v_identidad.npy` está calculado en la **capa final** (layer_idx=-1), no
en L30 como asumía la documentación previa del proyecto (verificado en
`sia/run_exp.py:134`). Proyectado en 11 capas (L5–L55):

- Débil e inconsistente en L5–L25 (varios pares n.s. o de signo mixto).
- Fuerte y estable en L35–L55, creciendo hasta d=10.18–10.43 en L55.
- El participation ratio tiene un perfil de capa **distinto**: pico de
  separación en L25–L30, no en capas tardías — dos fenómenos disociados
  por profundidad, no la misma señal vista dos veces.

Un v̂ calculado nativamente en L30 es **casi ortogonal** al de capa final
(cos=0.09) y **rompe** la doble disociación axis/axis_pec_only — L30
parece estar dominado por una señal de "densidad de estructura/reglas"
distinta de identidad. *(E-F2)*

## Hallazgo 9 — La huella de identidad es geométricamente coherente en profundidad, no solo en agregado

Ángulo de rotación de la representación del **mismo token** entre L5 y
L55 (trayectoria vertical, no temporal):

| Condición | rotación L5→L55 |
|---|---|
| `axis` | 57.92° |
| `axis_pec_only` | 57.58° (coincide con axis en cada tramo intermedio, <0.5° de diferencia) |
| `automata_neutro` | **62.60°** — consistentemente mayor en casi cada tramo, sobre todo en capas finales |

`axis` y `axis_pec_only` no solo terminan parecidos: **rotan por la misma
ruta** capa a capa. Es la confirmación geométrica más directa de la
doble disociación limpia, en un eje nunca antes examinado en este
proyecto. *(exploración adicional, A2)*

## Hallazgo 10 — v̂ tiene contenido semántico verificable

La proyección sobre v̂ es más alta específicamente cerca de vocabulario
auto-referencial explícito ("soy", "identidad", "esencia", "presencia")
dentro del mismo texto (+0.060 en axis, +0.048 en vanilla, sobre miles de
tokens), y la separación identidad/restricción es máxima en preguntas de
anclaje directo sobre el self (`identidad_basica`: sep=0.249) y mínima en
preguntas indirectas/meta-cognitivas (`diferenciacion_otras_ias`:
sep=0.159). v̂ no es una caja negra geométrica — correlaciona con
contenido literal verificable. *(exploración adicional, A1+A5)*

## Lo que sigue sin respuesta

- **¿Es causalmente utilizable?** No se pudo probar — el mecanismo de
  steering aditivo simple es inestable independientemente de la
  dirección (control con vector aleatorio replica el mismo colapso).
  Pregunta abierta, no cerrada en negativo. *(E-K)*
- **¿Hay correlato de atención (wiring)?** Sin correr — E-G pendiente,
  con L25-L30 como candidato de capa (por el pico de PR del Hallazgo 8).
- **n=1 de manipulación** en las dos celdas que cargan el argumento
  central (`axis_pec_only`, `automata_neutro`) — sin réplica de
  redacción todavía.

---

## Enunciado de tesis propuesto para lsgot_4.1

> La identidad declarada dejó una huella geométrica real: estática (ya
> presente antes de generar, Hallazgo 5), direccional y semánticamente
> específica (Hallazgos 3 y 10), coherente en profundidad a través de la
> red (Hallazgo 9), pero sin funcionar como atractor dinámico (Hallazgo
> 4) y sin haberse podido probar como causalmente utilizable (E-K,
> abierto). La densidad de restricción operativa, en cambio, domina la
> geometría del panel: colapsa más la dimensionalidad (Hallazgo 1),
> desvía más la ruta de recuperación (Hallazgo 2), y organiza un
> subespacio más grande (Hallazgo 7) que la identidad misma. Ambos
> factores son reales, independientes, y geométricamente disociables —
> pero no del mismo tamaño.
