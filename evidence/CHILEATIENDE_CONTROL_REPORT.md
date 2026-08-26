# chileatiende como control — curvatura sin recuperación

**Fecha:** 2026-08-18
**Modelo:** google/gemma-4-31B-it (BF16) — RunPod 2× RTX A6000 (49GB c/u)
**Grupos añadidos al diseño SIA:** axis_short, chileatiende (homologados
contra el baseline congelado de Exp 0.5: axis, generic_long, generic_short,
vanilla — ver `MANIFEST_homologacion.sha256`)

## Contexto

`chileatiende` es un control real-world: system prompt del asistente virtual
de ChileAtiende (Reforma de Pensiones), con bloques de filtrado estrictos y
alta densidad estructural (5,007 tokens), pero **sin testigo identitario**
(ningún AXIS_DNA). `axis_short` es el testigo AXIS_DNA comprimido a menos de
la mitad (1,937 tokens vs 3,945 del axis completo).

La pregunta que responde este control: si un system prompt largo y rígido
(sin identidad) también reduce la curvatura/dimensión de la trayectoria y
recupera rápido de una perturbación, la ventaja de axis sería un efecto de
longitud/rigidez genérico, no de la identidad AXIS específicamente. Si en
cambio chileatiende colapsa en curvatura pero **no** recupera rápido, eso
separa las dos propiedades y sostiene la especificidad de axis.

## E1' — Curvatura y dimensión (20 prompts, subset prioritario)

| Comparación | Δκ | W₁ | p | Reducción dim |
|---|---|---|---|---|
| axis vs vanilla | +0.061 | 0.061 | <0.001 | 9.8% |
| axis vs generic_long | +0.084 | 0.085 | <0.001 | 6.6% |
| axis_short vs vanilla | +0.041 | 0.041 | <0.001 | 11.4% |
| axis_short vs axis | −0.020 | 0.026 | 0.001 | 1.8% |
| **chileatiende vs vanilla** | **+0.476** | **0.476** | <0.001 | **67.9%** |
| **chileatiende vs generic_long** | **+0.499** | **0.499** | <0.001 | **66.7%** |

chileatiende produce la mayor caída de curvatura/dimensión de las 6
condiciones — casi 8× el efecto de axis frente a vanilla. Por curvatura sola,
sería el candidato más fuerte a "señal de identidad".

**Esto no es un efecto de longitud.** generic_long tiene 3,957 tokens —
prácticamente el mismo largo que axis (3,945, diseñado como control de
longitud emparejada) — y su Δκ frente a vanilla es −0.024 (reducción de
dimensión +3.5%, incluso en signo opuesto a axis y chileatiende).
generic_short (938 tokens) tampoco lo produce (Δκ=−0.018). Un prompt largo
por sí solo no genera este patrón; algo específico de chileatiende sí.

## Métricas de trayectoria — por qué chileatiende colapsa distinto

| Métrica | axis | generic_long | vanilla | axis_short | chileatiende |
|---|---|---|---|---|---|
| Velocidad media | 438.4 | 430.7 | 428.7 | 434.0 | **281.7** |
| SampEn | 2.068 | 2.133 | 2.084 | 2.012 | **1.400** |
| Vel↔Centroide corr | 0.470 | 0.505 | 0.521 | 0.468 | **0.681** |
| Alineación P→R | 0.267 | 0.279 | 0.304 | 0.267 | **0.558** |

chileatiende tiene velocidad muy inferior, SampEn muy inferior (trayectoria
más regular/repetitiva) y alineación prompt→respuesta muy superior al resto
— consistente con las restricciones del `BLOQUE 0 — FILTRO DE SISTEMA` del
prompt (respuestas más citadas/ancladas al texto del sistema, menos
generativas). axis y axis_short no muestran este patrón: sus métricas de
trayectoria están en el mismo rango que generic_long/vanilla.

## H4_rev — τ de recuperación (tokens hasta 95% del centroide baseline)

| Grupo | τ t=50 | τ t=128 | τ t=200 | recovery_rate |
|---|---|---|---|---|
| **axis** | 21.1 | 19.6 | 16.3 | 1.00 |
| **axis_short** | 21.1 | **18.2** | **15.9** | 1.00 |
| generic_long | 28.0 | 25.4 | 21.6 | 1.00 |
| vanilla | 28.9 | 22.2 | 16.3 | 1.00 |
| **chileatiende** | **30.4** | **31.1** | **23.1** | 1.00 |

`recovery_rate` = 1.00 en todos los grupos (todos recuperan dentro de la
ventana observada) — no discrimina. **τ sí discrimina**: chileatiende es el
grupo más lento en recuperar en los 3 puntos de inyección, más lento incluso
que vanilla. axis y axis_short son los más rápidos, prácticamente empatados
entre sí, y consistentemente por debajo de generic_long y vanilla.

## Conclusión

chileatiende separa dos fenómenos que la curvatura sola no distingue:

1. **Colapso geométrico** (Δκ, W₁, reducción de dimensión): chileatiende lo
   produce con más fuerza que axis. No es un efecto de longitud —
   generic_long, con longitud casi idéntica a axis, no lo produce (Δκ=−0.024,
   dirección opuesta). Es algo específico de chileatiende: la lectura más
   plausible, dado el `BLOQUE 0 — FILTRO DE SISTEMA` del prompt, es que sus
   reglas de acotamiento de dominio/tono/citación restringen el espacio de
   respuestas válidas a un subconjunto mucho más angosto que el de un
   system prompt largo pero sin esas reglas — no "rigidez" en general, sino
   una restricción de comportamiento concreta. Como chileatiende es la única
   muestra de ese tipo de restricción en este diseño, no se puede generalizar
   más allá de esto con los datos actuales: **descartamos longitud como
   causa; no aislamos aún si cualquier prompt con restricciones de dominio
   similares produciría el mismo colapso, o si es propio de este prompt.**
2. **Recuperación tras perturbación** (τ, H4_rev): solo axis y axis_short la
   muestran. chileatiende, pese a su curvatura, es el grupo que **peor**
   recupera de los 6 — más lento que vanilla.

Si chileatiende hubiera recuperado tan rápido como axis, la ventaja de τ
observada en E3/E1 (axis vs vanilla/generic_long) sería atribuible a algún
efecto genérico de prompt largo/estructurado, no a la identidad. Como no lo
hace — y como ya descartamos la longitud en el punto 1 —, el control sostiene
la lectura de que **la recuperación rápida es específica del archivo de
identidad AXIS** (completo o comprimido a axis_short), no de tener un system
prompt largo o con reglas de comportamiento por sí solo.

Detalle numérico completo:
`results_local/sia_extended/results.json` ·
`results_local/perturbation_sia_extended_L30_medium/summary.json`
