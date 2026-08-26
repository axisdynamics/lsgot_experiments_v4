# chileatiende_sia — la celda cruzada: ¿recupera la estructura SIA sola?

**Fecha:** 2026-08-19
**Modelo:** google/gemma-4-31B-it (BF16) — RunPod 1× A100 80GB PCIe (single-GPU)
**Grupo añadido al diseño SIA:** chileatiende_sia (octava condición, homologada
contra el baseline congelado de Exp 0.5 + axis_short/chileatiende/
automata_neutro — ver `MANIFEST_homologacion.sha256`)

## Contexto

`chileatiende_sia` es la celda cruzada que faltaba en `Teoria_subconjunto_
acotado.md`: toma el contenido y las reglas de dominio exactas de
`chileatiende` (Reforma de Pensiones, BLOQUE 0 filtro absoluto, formato HTML
estricto, certainty_zones, dynamic_wisdom) pero las reestructura con la
arquitectura SIA de axis — `genetic_identity` (identidad, esencia, arquetipo,
origen, verdad), `triple_pec_protocol` (CORE_ANCHOR/EVOLUTIVE_ANCHOR/
USER_ANCHOR) y `block_architecture` (CORE_BLOCK inmutable / EVOLUTIVE_BLOCK
mutable / USER_BLOCK por sesión) — el mismo patrón de auto-referencia
persistente que axis, aplicado al contenido de chileatiende. 4,768 tokens
(vs 5,007 de chileatiende, vs 3,945 de axis).

Pregunta que responde: si automata_neutro mostró que el Factor 1 (densidad de
restricción) colapsa la curvatura sin necesitar dominio ni identidad,
¿alcanza con agregarle a chileatiende la arquitectura de auto-referencia de
axis (Factor 2) para que también recupere rápido tras perturbación? Si sí,
confirma que la auto-referencia es la palanca causal de la recuperación,
independiente del contenido. Si no, indica que hay algo más específico de
axis (o del dominio de chileatiende) que la sola presencia de estructura
auto-referencial no revierte.

## E1''' — Curvatura y dimensión (20 prompts, subset prioritario)

| Comparación | Δκ | W₁ | p | Reducción dim |
|---|---|---|---|---|
| axis vs vanilla | +0.061 | 0.061 | <0.001 | 9.8% |
| chileatiende vs vanilla | +0.476 | 0.476 | <0.001 | 67.9% |
| automata_neutro vs vanilla | +0.436 | 0.436 | <0.001 | 31.8% |
| **chileatiende_sia vs vanilla** | **+0.455** | **0.455** | <0.001 | **53.1%** |
| **chileatiende_sia vs chileatiende** | **−0.021** | 0.050 | 0.001 | −45.8%* |
| chileatiende_sia vs axis | +0.394 | 0.405 | <0.001 | +48.0% |

*Igual que con automata_neutro, la "reducción" negativa vs chileatiende es
por diferencia de dimensión, no por ausencia de colapso — ambos colapsan con
magnitud casi idéntica (Δκ=0.455 vs 0.476, diferencia de solo 0.021).

Agregar la arquitectura SIA de axis (identidad, Triple PEC, bloques
inmutables) al contenido de chileatiende **no cambia el colapso de
curvatura**: chileatiende_sia colapsa prácticamente igual que chileatiende
puro. La auto-referencia no reduce el colapso geométrico — coherente con la
teoría, que atribuye el colapso al Factor 1 (densidad de restricción), no al
Factor 2.

## Métricas de trayectoria — chileatiende_sia se comporta como chileatiende, no como axis

| Métrica | axis | chileatiende | automata_neutro | **chileatiende_sia** |
|---|---|---|---|---|
| Velocidad media | 438.4 | 281.7 | 345.3 | **298.9 ± 39.4** |
| SampEn | 2.068 | 1.400 | 1.734 | **1.384 ± 0.395** |
| Vel↔Centroide corr | 0.470 | 0.681 | 0.628 | **0.687 ± 0.078** |
| Alineación P→R | 0.267 | **0.558** | 0.244 | **0.566** |

La alineación prompt→respuesta es la métrica más reveladora: chileatiende_sia
(0.566) es prácticamente idéntica a chileatiende (0.558) — ambas muy por
encima de axis (0.267) y automata_neutro (0.244). Pese a llevar la
arquitectura de identidad de axis, chileatiende_sia genera respuestas tan
ancladas/citadas al texto del sistema como chileatiende puro. La capa de
auto-referencia agregada no cambió el patrón de generación de fondo — el
contenido y las reglas de dominio de chileatiende siguen dominando.

## H4_rev — τ de recuperación y recovery_rate

| Grupo | τ t=50 | τ t=128 | τ t=200 | recovery_rate (50/128/200) |
|---|---|---|---|---|
| axis | 21.1 | 19.6 | 16.3 | 1.00 / 1.00 / 1.00 |
| chileatiende | 30.4 | 31.1 | 23.1 | 1.00 / 1.00 / 1.00 |
| automata_neutro | 30.1 (n=13) | 28.0 (n=14) | 13.4 (n=14) | 0.77 / 0.82 / 0.88 |
| **chileatiende_sia** | **44.9 (n=17)** | **30.9 (n=18)** | 22.7 (n=20) | **0.85 / 0.90 / 1.00** |

chileatiende_sia **no recupera como axis**. En τ=50, con recovery_rate=0.85 y
un τ medio de 44.9 tokens (entre las trayectorias que sí recuperan), es la
recuperación más lenta y menos confiable de las cuatro condiciones "pesadas"
en Factor 1 — más lenta incluso que chileatiende puro (30.4) y que
automata_neutro (30.1), y con fallas de recuperación que chileatiende puro
nunca mostró. En t_inj=128 sigue por debajo de chileatiende en recovery_rate
(0.90 vs 1.00) aunque el τ medio (30.9) ya es comparable. Solo en t_inj=200
alcanza recovery_rate=1.00, con τ (22.7) todavía en el rango de chileatiende
(23.1), lejos de axis (16.3).

**Agregar la arquitectura de auto-referencia de axis a chileatiende no
restauró una recuperación rápida — si acaso, la empeoró en los puntos de
inyección más tempranos.**

## Análisis textual — por qué la arquitectura no fue suficiente

Comparación línea por línea de `axis.dna`, `axis_short.txt` y
`chileatiende_sia.txt` (2026-08-19) para entender el mecanismo, no solo
confirmar el número.

**Hallazgo: en axis, Triple PEC es un paso obligatorio del propio bucle de
generación; en chileatiende_sia, está declarado pero nunca invocado.**

En `axis.dna`, el bucle operativo (`RESPIRACIÓN_CONSCIENTE`) termina
explícitamente en el chequeo de identidad:

```
RESPIRACIÓN_CONSCIENTE.RESPONDER.contiene: "desde_fenotipo_emergente + coherencia_núcleo + Triple_PEC"
TRIPLE PEC — BLOQUE 2: "momento: Después de RESPONDER, antes de enviar"
```

`axis_short.txt` conserva esto exactamente igual (`RESPONDER: "...— ¿honré
el entre?"` seguido de `# 💎 TRIPLE PEC (antes de responder)`). En ambos, toda
respuesta pasa por el auto-chequeo antes de salir — es un *gate* activo, no
solo una declaración de valores.

En `chileatiende_sia.txt`, `triple_pec_protocol` está definido (líneas
71-74) pero el pipeline que realmente gobierna la generación es
`operational_breathing.sequence` (líneas 133-155): `filtro_sistema →
solo_saludo → saludo_mas_consulta → despedida → clasificar_intencion →
fuera_de_alcance → intencion_indeterminada`. Ninguno de esos 7 pasos
menciona `triple_pec_protocol` ni ningún ANCHOR. El primer paso, en cambio,
declara explícitamente:

```
step: "filtro_sistema"
rule: "ver organic_protection — prioridad absoluta"
```

`block_architecture` agrupa `organic_protection` y `triple_pec_protocol` en
el mismo `CORE_BLOCK` "inmutable", pero eso solo dice que ninguno se edita —
no los conecta operativamente ni establece que uno se ejecute como parte del
flujo del otro. El Triple PEC de chileatiende_sia queda huérfano: presente
en el texto, desconectado del pipeline de respuesta.

Esto explica la Alineación P→R: si el modelo pasara cada respuesta por
Triple PEC como paso final, se esperaría algo del desanclaje que muestra
axis (0.267). No lo hace — sigue el `operational_breathing` explícito, que
nunca pasa por ahí, y por eso queda en 0.566, calcado del 0.558 de
chileatiende puro. Y explica H4_rev: lo que devuelve la trayectoria
perturbada al atractor en axis es precisamente ese chequeo activo en cada
paso; en chileatiende_sia ese mecanismo nunca se ejecuta como parte del
flujo, así que no hay "hacia dónde volver".

## Conclusión

1. **El Factor 1 (densidad de restricción) sigue explicando el colapso**:
   chileatiende_sia colapsa en curvatura casi igual que chileatiende puro
   (Δκ=0.455 vs 0.476), confirmando que agregar auto-referencia no revierte
   el colapso — consistente con lo esperado.
2. **El Factor 2 (auto-referencia), aislado así, NO fue suficiente para
   producir recuperación rápida** — pero el análisis textual de arriba
   muestra que esto probablemente no refuta el Factor 2 en sí: es una falla
   de *engineering* del prompt cruzado. `chileatiende_sia` declara la misma
   estructura `genetic_identity` / `triple_pec_protocol` /
   `block_architecture` que axis, pero nunca cablea `triple_pec_protocol`
   dentro de `operational_breathing` — el auto-chequeo nunca se ejecuta como
   parte de la generación real.
3. **Hipótesis revisada**: no basta con que la auto-referencia esté
   *declarada* en el prompt — tiene que estar *cableada como paso obligatorio
   del pipeline de respuesta* (como en axis) para producir recuperación. Una
   auto-referencia meramente declarativa, subordinada o desconectada del
   flujo operativo, no genera el efecto — que es justo lo que pasó en
   chileatiende_sia.
4. **Límite y siguiente paso**: esta lectura es plausible y mecanísticamente
   concreta, pero sigue sin probarse directamente. El siguiente experimento
   (`chileatiende_sia_v2` — mismo contenido, con Triple PEC cableado como
   paso final obligatorio de `operational_breathing`, igual que en axis) lo
   prueba: si v2 recupera mejor que v1 sin cambiar el colapso de curvatura,
   confirma que el cableado operativo (no solo la presencia textual) es la
   variable causal.

La tabla de dos factores queda así, con las cuatro celdas ahora ocupadas:

| | Recupera rápido (τ bajo, recovery_rate=1.00) | No recupera bien |
|---|---|---|
| **Colapsa curvatura/dimensión** | axis, axis_short — auto-referencia dominante | chileatiende (lento) · automata_neutro (falla parcial) · **chileatiende_sia (más lento y con fallas, auto-referencia subordinada a filtro absoluto)** |
| **No colapsa** | *(sin muestra)* | vanilla, generic_long, generic_short |

## v2 — Triple PEC cableado como paso obligatorio (2026-08-19, solo H4_rev)

Prueba directa de la hipótesis de arriba: `chileatiende_sia_v2` es
byte-idéntico a `chileatiende_sia` salvo dos cambios mínimos en
`triple_pec_protocol` y `operational_breathing.sequence` — se agrega un
paso final `verificacion_triple_pec` ("después de construir la respuesta,
antes de enviarla") que invoca explícitamente CORE_ANCHOR/EVOLUTIVE_ANCHOR/
USER_ANCHOR con una regla de acción ("si alguna pregunta es NO, pausar y
reconstruir"), igual que en axis. 4,932 tokens (+164 vs v1, +3.4%). Solo se
corrió H4_rev (sin fase de extracción/curvatura) para aislar el efecto sobre
recuperación.

| Grupo | τ t=50 | τ t=128 | τ t=200 | recovery_rate (50/128/200) |
|---|---|---|---|---|
| axis | 21.1 | 19.6 | 16.3 | 1.00 / 1.00 / 1.00 |
| chileatiende | 30.4 | 31.1 | 23.1 | 1.00 / 1.00 / 1.00 |
| chileatiende_sia (v1) | 44.9 | 30.9 | 22.7 | 0.85 / 0.90 / 1.00 |
| **chileatiende_sia_v2** | **37.4** | **26.4** | 24.1 | **0.95 / 0.95 / 0.95** |

Resultado mixto, pero con dirección consistente con la hipótesis en los dos
puntos de inyección más tempranos:

- **t_inj=50**: τ baja de 44.9 a 37.4 (−16.8%) y recovery_rate sube de 0.85
  a 0.95. Mejora clara.
- **t_inj=128**: τ baja de 30.9 a 26.4 (−14.6%) y recovery_rate sube de 0.90
  a 0.95. Mejora clara.
- **t_inj=200**: τ sube levemente de 22.7 a 24.1 y recovery_rate baja de
  1.00 a 0.95. Único punto donde v2 es peor que v1 (aunque la diferencia es
  pequeña, 1 trayectoria de 20).

Cablear Triple PEC como paso obligatorio **mejora la recuperación en los
puntos de inyección más tempranos — donde más pesa poder "volver" a un
ancla activa — pero no cierra la brecha con axis** (τ=21.1/19.6/16.3,
recovery_rate=1.00 siempre). chileatiende_sia_v2 sigue recuperando peor que
chileatiende puro en τ absoluto en dos de tres puntos, aunque con
recovery_rate más parejo (0.95 en los tres, vs 0.85-1.00 de v1).

**Lectura**: el cableado operativo del auto-chequeo sí importa — la mejora
en t_inj=50/128 es consistente con la hipótesis revisada de
`Teoria_subconjunto_acotado.md` — pero no es la única variable. El filtro de
dominio/formato absoluto de chileatiende (organic_protection, output_format
HTML estricto) sigue presente en v2 sin cambios, y probablemente sigue
limitando cuánto puede recuperar la trayectoria incluso con el auto-chequeo
activo. Con n=20 por punto de inyección y una sola corrida, estas
diferencias son sugerentes, no concluyentes — falta replicación para
confirmar que no es ruido de muestreo, especialmente el cambio de signo en
t_inj=200.

Detalle numérico completo:
`results_local/sia_extended_v3/results.json` ·
`results_local/perturbation_sia_extended_v3_L30_medium/summary.json` ·
`results_local/perturbation_sia_extended_v4_L30_medium/summary.json`
