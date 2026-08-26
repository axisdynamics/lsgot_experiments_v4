# automata_neutro — aislando el Factor 1 (densidad de restricción) del dominio

**Fecha:** 2026-08-19
**Modelo:** google/gemma-4-31B-it (BF16) — RunPod 1× A100 80GB PCIe (single-GPU,
sin pipeline-parallel)
**Grupo añadido al diseño SIA:** automata_neutro (séptima condición, homologada
contra el baseline congelado de Exp 0.5 + axis_short/chileatiende — ver
`MANIFEST_homologacion.sha256`)

## Contexto

`automata_neutro` es la condición diseñada en `Teoria_subconjunto_acotado.md`
para aislar el Factor 1 (densidad de restricción operativa) del dominio
específico de chileatiende (Reforma de Pensiones). Mismo tipo de arquitectura
que chileatiende — bloques con precedencia, reglas trigger→salida verbatim,
filtro de sistema con prioridad absoluta, protocolo de certeza NIVEL 1/2/3
con fallback literal, 12 guardrails — pero con **dominio genérico de
asistencia** (no pensiones) y **cero auto-referencia identitaria** (verificado
por grep de palabras completas: esencia, núcleo, identidad, testigo,
arquetipo, self). 3,999 tokens, emparejado a axis (3,945, Δ=1.4%) con el
mismo estándar que generic_long. Densidad de restricción 59.3/1k tokens — la
más alta del panel (chileatiende: 54.6/1k).

Predicción del documento de teoría: si colapsa en curvatura/dimensión como
chileatiende pero no recupera rápido tras perturbación, la celda "autómata sin
auto-referencia" queda ocupada con un dominio neutro, descartando que el
colapso de chileatiende fuera un efecto del dominio de pensiones específico.

## E1'' — Curvatura y dimensión (20 prompts, subset prioritario)

| Comparación | Δκ | W₁ | p | Reducción dim |
|---|---|---|---|---|
| axis vs vanilla | +0.061 | 0.061 | <0.001 | 9.8% |
| chileatiende vs vanilla | +0.476 | 0.476 | <0.001 | 67.9% |
| **automata_neutro vs vanilla** | **+0.436** | **0.436** | <0.001 | **31.8%** |
| automata_neutro vs axis | +0.375 | 0.384 | <0.001 | +24.3% |
| automata_neutro vs generic_long | +0.460 | 0.460 | <0.001 | +29.3% |
| automata_neutro vs chileatiende | −0.040 | 0.132 | <0.001 | −112.4%* |

*La "reducción de dimensión" de automata_neutro vs chileatiende es negativa
porque automata_neutro tiene *mayor* dimensión fractal que chileatiende, no
porque no colapse — ambos colapsan fuerte frente a vanilla/axis, con
magnitudes de Δκ casi idénticas (0.436 vs 0.476, diferencia de 0.040) pero
composición geométrica distinta (ver métricas de trayectoria).

automata_neutro reproduce el colapso de curvatura de chileatiende (mismo
orden de magnitud, 8× el efecto axis-vs-vanilla) **sin el dominio de
pensiones**. Esto confirma la predicción del Factor 1: la densidad de
restricción operativa sola, sin dominio específico, ya reduce fuertemente la
curvatura y la dimensión de la trayectoria.

## Métricas de trayectoria — automata_neutro no es un clon de chileatiende

| Métrica | axis | chileatiende | **automata_neutro** |
|---|---|---|---|
| Velocidad media | 438.4 | 281.7 | **345.3 ± 66.1** |
| SampEn | 2.068 | 1.400 | **1.734 ± 1.989** |
| Vel↔Centroide corr | 0.470 | 0.681 | **0.628 ± 0.141** |
| Alineación P→R | 0.267 | **0.558** | **0.244** |

Aquí aparece la primera diferencia real entre los dos colapsos. chileatiende
tiene alineación prompt→respuesta muy superior al resto (0.558) — sus
respuestas citan/anclan fuertemente el texto del sistema (HTML literal,
bloques de filtro). automata_neutro tiene la alineación **más baja de las
siete condiciones** (0.244, incluso por debajo de axis). Ambos colapsan en
curvatura por vías distintas: chileatiende se ancla al texto del prompt;
automata_neutro no — su colapso viene de otro lado (ver H4_rev abajo, la
varianza alta en SampEn y velocidad ya lo anticipa).

## H4_rev — τ de recuperación y recovery_rate

| Grupo | τ t=50 | τ t=128 | τ t=200 | recovery_rate |
|---|---|---|---|---|
| axis | 21.1 | 19.6 | 16.3 | 1.00 |
| vanilla | 28.9 | 22.2 | 16.3 | 1.00 |
| chileatiende | 30.4 | 31.1 | 23.1 | 1.00 |
| **automata_neutro** | 30.1 (n=13/20) | 28.0 (n=14/20) | 13.4 (n=14/16) | **0.77 / 0.82 / 0.88** |

Esto es lo que separa a automata_neutro de los otros seis grupos: es el
**único** con `recovery_rate` menor a 1.00. Entre el 12% y el 24% de las
trayectorias, según el punto de inyección, nunca vuelven a 95% de similitud
coseno con el centroide baseline dentro de la ventana observada — no es que
recuperen lento, es que una fracción no recupera en absoluto. Ningún otro
grupo del panel (incluido chileatiende, que sí recupera pero lento) muestra
este patrón. El τ medio reportado arriba es solo sobre las trayectorias que
sí recuperaron (n menor a 20), así que no es directamente comparable con el τ
de 100% de los otros grupos — subestima cuánto más inestable es
automata_neutro bajo perturbación.

## Conclusión

1. **El Factor 1 queda confirmado, aislado del dominio**: automata_neutro
   colapsa en curvatura/dimensión con una magnitud comparable a chileatiende
   (Δκ=0.436 vs 0.476) sin ningún contenido de pensiones — la densidad de
   restricción operativa (reglas trigger→salida, prioridad absoluta,
   jerarquía de bloques, verificación obligatoria) basta por sí sola. Esto
   descarta que el colapso de chileatiende fuera específico del dominio de
   Reforma de Pensiones.
2. **El Factor 2 (ausencia de auto-referencia) predice recuperación
   deficiente — pero la forma que toma es más severa de lo anticipado**: la
   teoría predecía que automata_neutro, sin auto-referencia, no recuperaría
   rápido, como chileatiende. Lo que se observa es distinto en tipo, no solo
   en grado: chileatiende siempre recupera (lento, pero recovery_rate=1.00);
   automata_neutro **falla en recuperar** en una fracción sustancial de
   casos. La densidad de restricción de automata_neutro es la más alta del
   panel (59.3/1k vs 54.6/1k de chileatiende) y su alineación P→R es la más
   baja — sugiere que, sin auto-referencia Y sin contenido de dominio al que
   "volver" tras la perturbación, la trayectoria puede quedar sin ningún
   punto de anclaje estable. Es una hipótesis post-hoc, no una predicción
   confirmada de antemano — con n=1 por condición, no se puede generalizar
   más allá de esta muestra.

La tabla de dos factores de `Teoria_subconjunto_acotado.md` queda así:

| | Recupera rápido (τ bajo, recovery_rate=1.00) | No recupera bien |
|---|---|---|
| **Colapsa curvatura/dimensión** | axis, axis_short — autómata + auto-referencia | chileatiende (lento, recovery_rate=1.00) · **automata_neutro (recovery_rate<1.00, el único con fallas de recuperación)** |
| **No colapsa** | *(sin muestra)* | vanilla, generic_long, generic_short — sin autómata |

Detalle numérico completo:
`results_local/sia_extended_v2/results.json` ·
`results_local/perturbation_sia_extended_v2_L30_medium/summary.json`
