# automata_neutro — aislando el Factor 1 (densidad de restricción) del dominio

> ⚠️ **Editado 2026-08-28 (noche):** este reporte comparaba originalmente
> `automata_neutro` contra `chileatiende` para descartar que el colapso de
> chileatiende fuera específico de su dominio (Reforma de Pensiones).
> `chileatiende` fuerza un wrapper HTML literal en el 100% de sus
> respuestas (43.6% de cada respuesta es texto repetido — ver
> `CHILEATIENDE_MARKUP_CONFOUND_REPORT.md`) y se eliminó de este reporte.
> Las cifras de `automata_neutro` en sí (vs vanilla/axis/generic_long) no
> dependen de chileatiende y se mantienen intactas — de hecho son la
> evidencia más limpia de todo el proyecto para el Factor 1 (restricción),
> como confirma `CORRECCION_DVHAT_SIN_CHILEATIENDE.md` y `EJ_REPORT.md`.

**Fecha:** 2026-08-19
**Modelo:** google/gemma-4-31B-it (BF16) — RunPod 1× A100 80GB PCIe (single-GPU,
sin pipeline-parallel)
**Grupo añadido al diseño SIA:** automata_neutro (séptima condición, homologada
contra el baseline congelado de Exp 0.5 — ver `MANIFEST_homologacion.sha256`)

## Contexto

`automata_neutro` es la condición diseñada en `Teoria_subconjunto_acotado.md`
para aislar el Factor 1 (densidad de restricción operativa) del dominio
específico de las condiciones "autómata" originales. Bloques con precedencia,
reglas trigger→salida verbatim, filtro de sistema con prioridad absoluta,
protocolo de certeza NIVEL 1/2/3 con fallback literal, 12 guardrails — pero
con **dominio genérico de asistencia** (no un dominio específico) y **cero
auto-referencia identitaria** (verificado por grep de palabras completas:
esencia, núcleo, identidad, testigo, arquetipo, self). 3,999 tokens,
emparejado a axis (3,945, Δ=1.4%) con el mismo estándar que generic_long.
Densidad de restricción 59.3/1k tokens — la más alta del panel.

## E1'' — Curvatura y dimensión (20 prompts, subset prioritario)

| Comparación | Δκ | W₁ | p | Reducción dim |
|---|---|---|---|---|
| axis vs vanilla | +0.061 | 0.061 | <0.001 | 9.8% |
| **automata_neutro vs vanilla** | **+0.436** | **0.436** | <0.001 | **31.8%** |
| automata_neutro vs axis | +0.375 | 0.384 | <0.001 | +24.3% |
| automata_neutro vs generic_long | +0.460 | 0.460 | <0.001 | +29.3% |

`automata_neutro` colapsa fuertemente en curvatura y dimensión (7× el
efecto axis-vs-vanilla) sin ningún contenido de dominio específico ni
auto-referencia. Confirma la predicción del Factor 1: la densidad de
restricción operativa sola ya reduce fuertemente la curvatura y la
dimensión de la trayectoria.

## Métricas de trayectoria

| Métrica | axis | **automata_neutro** |
|---|---|---|
| Velocidad media | 438.4 | **345.3 ± 66.1** |
| SampEn | 2.068 | **1.734 ± 1.989** |
| Vel↔Centroide corr | 0.470 | **0.628 ± 0.141** |
| Alineación P→R | 0.267 | **0.244** |

`automata_neutro` tiene la alineación prompt→respuesta más baja del panel
limpio (0.244, incluso por debajo de axis) y la varianza más alta en
SampEn y velocidad — su colapso en curvatura no viene de anclarse al
texto del sistema (no hay markup ni citas literales), sino de otra
dinámica (ver H4_rev abajo).

## H4_rev — τ de recuperación y recovery_rate

| Grupo | τ t=50 | τ t=128 | τ t=200 | recovery_rate |
|---|---|---|---|---|
| axis | 21.1 | 19.6 | 16.3 | 1.00 |
| vanilla | 28.9 | 22.2 | 16.3 | 1.00 |
| **automata_neutro** | 30.1 (n=13/20) | 28.0 (n=14/20) | 13.4 (n=14/16) | **0.77 / 0.82 / 0.88** |

Esto es lo que separa a `automata_neutro` del resto del panel limpio: es
el **único** con `recovery_rate` menor a 1.00. Entre el 12% y el 24% de
las trayectorias, según el punto de inyección, nunca vuelven a 95% de
similitud coseno con el centroide baseline dentro de la ventana
observada — no es que recuperen lento, es que una fracción no recupera en
absoluto. El τ medio reportado arriba es solo sobre las trayectorias que
sí recuperaron (n menor a 20), así que subestima cuánto más inestable es
`automata_neutro` bajo perturbación.

## Conclusión

1. **El Factor 1 queda confirmado, aislado de cualquier contenido de
   dominio o auto-referencia**: `automata_neutro` colapsa en curvatura/
   dimensión (Δκ=0.436, 7× axis-vs-vanilla) y en recuperación
   (recovery_rate<1.00, único caso del panel limpio) por la sola densidad
   de restricción operativa (reglas trigger→salida, prioridad absoluta,
   jerarquía de bloques, verificación obligatoria).
2. **La ausencia de auto-referencia predice recuperación deficiente, de
   forma más severa de lo anticipado**: no es solo que recupere lento —
   una fracción sustancial de trayectorias **nunca** recupera dentro de
   la ventana observada. La densidad de restricción de `automata_neutro`
   es la más alta del panel y su alineación P→R es la más baja — sugiere
   que, sin auto-referencia y sin contenido de dominio al que "volver"
   tras la perturbación, la trayectoria puede quedar sin ningún punto de
   anclaje estable. Es una hipótesis post-hoc, no una predicción
   confirmada de antemano — con n=1 por condición, no se puede generalizar
   más allá de esta muestra (T2, ver `Set_experimental.md`).

Detalle numérico completo:
`results_local/sia_extended_v2/results.json` ·
`results_local/perturbation_sia_extended_v2_L30_medium/summary.json`
