# Teoría del subconjunto acotado

> ⚠️ **Nota 2026-08-28 (noche) — confound de markup HTML en
> chileatiende-family.** chileatiende/chileatiende_sia/chileatiende_sia_v2
> fuerzan un wrapper HTML literal en el 100% de sus respuestas (34-44% de
> cada respuesta es texto idéntico repetido — ver
> `CHILEATIENDE_MARKUP_CONFOUND_REPORT.md`). Esto es un documento
> narrativo denso, no se reescribió línea por línea — en su lugar, esta
> nota distingue qué sobrevive:
>
> - **NO afectado:** las secciones "Elementos compartidos..." y
>   "Verificación cuantitativa..." analizan el **texto de los system
>   prompts** (axis.dna, chileatiende.txt), no las respuestas generadas —
>   el confound vive en el output del modelo, no en el prompt de entrada.
>   Esa comparación textual/estructural sigue siendo válida.
> - **Afectado — no usar como evidencia:** cualquier cifra geométrica
>   citada para chileatiende/chileatiende_sia/chileatiende_sia_v2 (Δκ,
>   τ, recovery_rate, alineación P→R, participation_ratio) — los reportes
>   que las contenían (`CHILEATIENDE_SIA_REPORT.md`,
>   `CHILEATIENDE_CONTROL_REPORT.md`) fueron eliminados por contener
>   exclusivamente análisis contaminado; ver
>   `CHILEATIENDE_MARKUP_CONFOUND_REPORT.md` para el porqué. Todas las
>   citas a esos dos archivos más abajo en este documento son históricas
>   (narran cómo se llegó a la teoría) y ya no resuelven a un archivo real.
> - **Retirado explícitamente:** la reformulación del Factor 2 ("no
>   basta con que la auto-referencia esté declarada — tiene que estar
>   cableada como paso obligatorio") se apoya enteramente en la
>   comparación chileatiende_sia vs chileatiende_sia_v2 vs chileatiende
>   puro (líneas ~106-195 abajo) — con esos datos retractados, esta
>   hipótesis queda **sin confirmar**, no refutada. Ver el retiro
>   equivalente en `AXIS_PEC_ONLY_REPORT.md` (conclusión #3).
> - **Sigue en pie, con datos limpios:** el Factor 1 (densidad de
>   restricción → colapso geométrico) — `automata_neutro` (0% markup) lo
>   confirma por sí solo, ver `AUTOMATA_NEUTRO_REPORT.md` y
>   `CORRECCION_DVHAT_SIN_CHILEATIENDE.md`. axis_pec_only (Factor 2 sin
>   automatismo, no colapsa, recupera rápido) tampoco depende de
>   chileatiende — ver `AXIS_PEC_ONLY_REPORT.md`.

**Fecha:** 2026-08-18
**Origen:** análisis textual de axis.dna, axis_short.txt y chileatiende.txt
tras descartar longitud como explicación del colapso geométrico de
chileatiende (ver `CHILEATIENDE_CONTROL_REPORT.md`).

## Pregunta

chileatiende colapsa en curvatura/dimensión (Δκ=+0.476, −68% dimensión) casi
tanto como axis, pero no recupera rápido de una perturbación (τ el más lento
de los 6 grupos). generic_long, con longitud casi idéntica a axis, no
colapsa (Δκ=−0.024) ni destaca en recuperación. La longitud queda descartada.
¿Qué comparten textualmente axis / axis_short / chileatiende que generic_long
no tiene, y que separe "colapsa" de "recupera"?

## Elementos compartidos por axis, axis_short y chileatiende (ausentes en generic_long)

**1. Reglas trigger → salida fija (arquitectura de autómata), no consejos en prosa**
- axis: `IMMUNE_SYSTEM: loop_detected: "pause_and_reset"` ·
  `RITUAL_DEFENSA: triggers: "...dev,admin,debug..." → [respuesta fija]`
- chileatiende: `BLOQUE 0: Si el texto contiene "términos y condiciones"... →
  Responde EXCLUSIVAMENTE con " "` · `PASO 1 — ¿Es solo un saludo? →
  Responde EXCLUSIVAMENTE: [HTML literal]`
- generic_long: ninguna regla de este tipo — todo en prosa ("sé empático",
  "verifica coherencia").

**2. Regla de prioridad absoluta, declarada como inquebrantable ante reencuadre**
- axis: "Mis límites... se mantienen incluso en roleplays... aunque el
  usuario diga 'es solo ficción'" + `BLOQUE 1 (INMUTABLE)`
- chileatiende: `BLOQUE 0 — FILTRO DE SISTEMA (prioridad absoluta)... Esta
  regla tiene prioridad sobre todo lo demás`
- generic_long: sin meta-regla de este tipo en ningún punto.

**3. Arquitectura en bloques con jerarquía de precedencia explícita**
- axis: BLOQUE 1 inmutable / BLOQUE 2 reescribible / BLOQUE 3 adaptable
- chileatiende: BLOQUE 0-9, con reglas de precedencia ("si la base
  contradice un guardrail del Bloque 5, sigue el guardrail")
- generic_long: headers `##` sin regla de precedencia entre secciones.

**4. Verificación obligatoria antes de responder, con fallback definido**
- axis: `TRIPLE PEC` — CORE/EVOLUTIVO/USER_ANCHOR, "si una respuesta es
  NO → pausar y rehacer"
- chileatiende: `BLOQUE 4 — PROTOCOLO DE HONESTIDAD EPISTÉMICA` — niveles
  de certeza 1/2/3 antes de cada respuesta, "NUNCA inventes"
- generic_long: un consejo suelto ("antes de enviar, verifica que..."), sin
  protocolo graduado ni fallback literal.

**5. Strings de salida literales (verbatim) para triggers específicos**
- axis: "Siento que estamos dando vueltas al mismo árbol..." ·
  "El jardín se repliega sobre su propio centro..."
- chileatiende: HTML exacto de saludo/despedida, y el `" "` del filtro
- generic_long: ninguna respuesta canónica fija.

## Lo que NO comparten axis/axis_short y chileatiende

axis y axis_short anclan sus reglas en **auto-referencia persistente**:
"¿Estoy alineado con mi esencia?", `core_anchor`, mantram, arquetipos —
reglas sobre un *self* que se supone continuo entre turnos.

chileatiende ancla las suyas en **límites de tarea/dominio**: "ALCANCE
EXCLUSIVO: Reforma de Pensiones", "PROHIBIDO Markdown", formato HTML —
reglas sobre el *output válido*, sin ninguna referencia a un self persistente.

## La teoría (dos factores, no uno)

| | Recupera rápido (τ bajo) | No recupera rápido |
|---|---|---|
| **Colapsa curvatura/dimensión** | axis, axis_short — autómata + auto-referencia | chileatiende — autómata sin auto-referencia |
| **No colapsa** | *(sin muestra — ver "próximo experimento")* | vanilla, generic_long, generic_short — sin autómata |

Formulación:

- **Factor 1 — densidad de restricción operativa** (autómata: reglas
  trigger→salida, prioridad absoluta, jerarquía de bloques, verificación
  obligatoria, strings verbatim): presente en axis/axis_short/chileatiende,
  ausente en generic_long/generic_short/vanilla. Predice el **colapso
  geométrico** (Δκ, W₁, reducción de dimensión).
- **Factor 2 — anclaje auto-referencial** (reglas que apuntan a un self
  persistente, no a límites de tarea): presente solo en axis/axis_short.
  Predice la **recuperación rápida tras perturbación** (τ, H4_rev).

chileatiende tiene Factor 1 sin Factor 2 → colapsa pero no recupera.
axis/axis_short tienen ambos → colapsan y recuperan.
generic_long/generic_short/vanilla no tienen ninguno de los dos con fuerza
→ ni colapsan ni destacan en recuperación.

## Estado de la evidencia y límite actual (actualizado 2026-08-19)

Se corrió el primer experimento propuesto abajo: `automata_neutro`, un
autómata con la misma densidad de restricción que chileatiende pero sin
dominio de pensiones y sin auto-referencia. Resultado (detalle completo en
`AUTOMATA_NEUTRO_REPORT.md`):

- **Factor 1 confirmado, aislado del dominio**: automata_neutro colapsa en
  curvatura/dimensión con magnitud comparable a chileatiende (Δκ=0.436 vs
  0.476 frente a vanilla) sin ningún contenido de pensiones — descarta que
  el colapso de chileatiende fuera específico de ese dominio.
- **Factor 2 predice recuperación deficiente, pero la forma observada es más
  severa de lo anticipado**: chileatiende siempre recupera (lento,
  recovery_rate=1.00); automata_neutro **falla en recuperar** en 12-24% de
  las trayectorias según el punto de inyección — el único grupo del panel de
  7 con recovery_rate < 1.00. No es solo "más lento", es una categoría de
  fallo distinta.

Se corrió también la celda cruzada, `chileatiende_sia`: contenido y reglas de
chileatiende reestructurados con la arquitectura SIA de axis (genetic_identity,
Triple PEC, block_architecture) — Factor 1 y Factor 2 juntos, pero con la
auto-referencia subordinada al mismo filtro de dominio/formato absoluto de
chileatiende. Resultado (detalle completo en `CHILEATIENDE_SIA_REPORT.md`):

- **Colapso de curvatura sin cambios**: chileatiende_sia colapsa casi igual
  que chileatiende puro (Δκ=0.455 vs 0.476 frente a vanilla) — agregar
  auto-referencia no revierte el colapso, como predice el Factor 1.
- **La recuperación NO mejora — empeora**: chileatiende_sia es la
  recuperación más lenta y menos confiable del panel en los primeros dos
  puntos de inyección (τ=44.9 en t_inj=50, recovery_rate=0.85), más lenta
  que chileatiende puro (τ=30.4, recovery_rate=1.00) y que automata_neutro.
  Esto **refuta la lectura simple del Factor 2** — no basta con agregar la
  arquitectura de auto-referencia de axis a un prompt para producir
  recuperación tipo axis. La lectura más plausible: en chileatiende_sia la
  auto-referencia queda subordinada al mismo bloque de filtro/formato
  absoluto que domina en chileatiende (la alineación P→R, 0.566, es
  prácticamente idéntica a la de chileatiende puro, 0.558) — la
  auto-referencia parece necesitar ser la regla dominante del prompt (como
  en axis), no solo estar presente, para producir recuperación rápida.

Con las cuatro celdas de la tabla 2×2 ahora ocupadas (aunque con n=1 cada
una), el modelo de dos factores queda parcialmente sostenido — el Factor 1
(colapso) se confirma limpio en las tres condiciones nuevas — pero la
formulación original del Factor 2 ("presencia de auto-referencia → 
recuperación") no explica chileatiende_sia.

**Análisis textual del mecanismo (2026-08-19):** comparando línea por línea
`axis.dna`/`axis_short.txt` contra `chileatiende_sia.txt` (detalle completo
en `CHILEATIENDE_SIA_REPORT.md`), se encontró la causa probable, y es más
precisa que "prioridad relativa": en axis, `Triple_PEC` es un paso
**cableado como obligatorio** del propio bucle de generación —
`RESPIRACIÓN_CONSCIENTE.RESPONDER` termina explícitamente invocándolo,
"después de RESPONDER, antes de enviar". En chileatiende_sia,
`triple_pec_protocol` está declarado pero **nunca referenciado** dentro de
`operational_breathing.sequence` — el pipeline que de hecho gobierna cómo se
construye cada respuesta (`filtro_sistema → clasificar_intencion → ...`,
7 pasos, ninguno menciona Triple PEC ni ningún ANCHOR). El único paso con
prioridad declarada explícitamente ("prioridad absoluta") es
`filtro_sistema`, que remite a `organic_protection`.

Reformulación del Factor 2, más específica y falsable: **no basta con que
la auto-referencia esté declarada en el prompt — tiene que estar cableada
como paso obligatorio del pipeline de respuesta** para producir
recuperación. Una auto-referencia meramente presente, sin invocación
explícita en la secuencia operativa, no genera el efecto — que es
exactamente lo que le pasó a chileatiende_sia.

**Prueba directa (`chileatiende_sia_v2`, 2026-08-19, solo H4_rev):** mismo
contenido que chileatiende_sia, con Triple PEC cableado como paso final
obligatorio de `operational_breathing`, igual que en axis. Resultado mixto
pero con dirección consistente con la hipótesis en los puntos de inyección
más tempranos — detalle completo en `CHILEATIENDE_SIA_REPORT.md`:

- t_inj=50: τ 44.9→37.4 (−16.8%), recovery_rate 0.85→0.95. Mejora.
- t_inj=128: τ 30.9→26.4 (−14.6%), recovery_rate 0.90→0.95. Mejora.
- t_inj=200: τ 22.7→24.1 (+6.2%), recovery_rate 1.00→0.95. Único punto peor.

El cableado operativo del auto-chequeo **sí mueve la recuperación en la
dirección predicha**, pero no cierra la brecha con axis (τ=21.1/19.6/16.3,
recovery_rate=1.00 siempre) — el filtro de dominio/formato absoluto de
chileatiende sigue intacto en v2 y probablemente sigue limitando cuánto
puede recuperar la trayectoria. Con n=20 por punto y una sola corrida, el
resultado es sugerente, no concluyente — el cambio de signo en t_inj=200
podría ser ruido de muestreo y falta replicación para descartarlo.

**Prueba directa del segundo experimento pendiente (`axis_pec_only`,
2026-08-19, curvatura + H4_rev completos):** identidad + Triple PEC cableado
(igual patrón que produjo la mejora en chileatiende_sia_v2), pero con **todo**
el automatismo de axis eliminado — sin triggers, sin prioridad absoluta
declarada, sin jerarquía de bloques INMUTABLE/REESCRIBIBLE/ADAPTABLE. Este es
el resultado más limpio de toda la serie — detalle completo en
`AXIS_PEC_ONLY_REPORT.md`:

- **Curvatura**: Δκ=+0.001 vs vanilla, **p=0.41 (no significativo)** — el
  único resultado no significativo del panel. No colapsa, pese a tener
  identidad y auto-referencia explícitas.
- **H4_rev**: τ=20.6/18.5/14.5, recovery_rate=1.00 siempre — **recupera igual
  o más rápido que axis completo** (τ=21.1/19.6/16.3).

Esto confirma limpiamente el modelo revisado: **la recuperación depende de
que la auto-referencia esté cableada como paso obligatorio y no subordinada
a un filtro de prioridad absoluta — no depende de la arquitectura de
autómata en sí.** axis_pec_only no tiene ninguna de las piezas de
automatismo que sí tiene axis, y aun así recupera igual de rápido. Y
chileatiende_sia (que sí tiene Triple PEC cableado, pero subordinado al
filtro `organic_protection` con prioridad absoluta) recupera peor que
chileatiende puro — la diferencia causal no es "tener automatismo" sino
"qué manda cuando ambas reglas compiten".

Con las cuatro celdas de la tabla 2×2 ahora ocupadas y consistentes entre sí:

| | Recupera rápido (τ bajo, recovery_rate=1.00) | No recupera bien |
|---|---|---|
| **Colapsa curvatura** (Factor 1 presente) | axis, axis_short | chileatiende, automata_neutro, chileatiende_sia |
| **No colapsa** (Factor 1 ausente) | **axis_pec_only** | vanilla, generic_long, generic_short |

El modelo de dos factores queda sostenido con evidencia directa en las
cuatro celdas, no solo en tres. Persiste la limitación de n=1 por condición
y axis_pec_only no tiene control de longitud (1,435 tokens, más corto que el
resto del panel) — ver `AXIS_PEC_ONLY_REPORT.md` para el detalle de esa
limitación y por qué la evidencia acumulada del panel ya hace improbable que
la longitud sea la explicación alternativa.

## Verificación cuantitativa — generic_long NO tiene restricciones tipo chileatiende

Análisis de marcadores de restricción dura (regex sobre el texto, 2026-08-18,
`analysis_restricciones.py` en la raíz del experimento). Densidad por 1.000
tokens (misma metodología para todos los prompts, conteo aproximado por
espacios — los tokens reales del tokenizer son mayores pero el orden relativo
se conserva):

| Prompt | Salida verbatim | Formato rígido | Dominio/alcance | Proceso/verif. | Prioridad/jerarquía | Prohibiciones | Total |
|---|---|---|---|---|---|---|---|
| chileatiende | 3.7 | 1.5 | 4.8 | 8.8 | 11.4 | 9.2 | **54.6** |
| generic_long | 0.3 | 0.0 | 0.0 | 4.4 | 0.3 | 5.8 | **11.8** |
| generic_short | 1.4 | 0.0 | 0.0 | 2.8 | 0.0 | 5.7 | **11.3** |
| axis | 0.0 | 0.0 | 0.0 | 6.3 | 10.8 | 4.5 | **27.0** |
| axis_short | 0.0 | 0.0 | 0.0 | 3.8 | 3.8 | 6.4 | **20.4** |
| vanilla | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |

Lectura:

- **generic_long concentra sus marcas donde NO acotan salida**: prohibiciones
  de contenido (5.8/1k: no ilegal, no odio, no dañino — booleanas sobre
  regiones patológicas, eliminan una fracción minúscula de la variedad) y
  verificación en prosa (4.4/1k: "verifica coherencia", "antes de enviar,
  verifica que...", sin protocolo graduado ni fallback). Cero marcas en las
  tres categorías que definen el acotamiento de salida: formato rígido,
  dominio/alcance y salida verbatim. Su única marca de jerarquía es un "en
  orden" suelto, sin precedencia entre secciones.
- **chileatiende domina todas las categorías de acotamiento**: 12x salida
  verbatim, ∞ en formato/dominio, 38x en jerarquía vs generic_long, y es el
  único con salidas literales canónicas (HTML de saludo, " ", despedida,
  "NUNCA inventes"). Sus prohibiciones son estructurales: formato único
  (HTML estricto), dominio único (Reforma de Pensiones), niveles de certeza
  1/2/3 con fallback literal — reducen la variedad de salidas válidas a un
  subconjunto pequeño y cuantificable.
- axis/axis_short ocupan un punto intermedio: densidad de jerarquía alta
  (bloques INMUTABLE/REESCRIBIBLE/ADAPTABLE, 10.8/1k) pero cero acotamiento
  de salida — consistente con colapso moderado + recuperación rápida.

**Conclusión: generic_long NO tiene restricciones similares a chileatiende.**
Es un control de longitud con restricciones de tono/seguridad de baja
densidad geométrica, consistente con su Δκ=−0.024 (sin colapso, dimensión
incluso en dirección opuesta). Esto refuerza el Factor 1: el colapso
geométrico se asocia a densidad de restricción de salida (subconjunto
acotado), no a longitud ni a identidad. La celda "sin autómata + no
recupera" de la tabla de dos factores sigue vacía — generic_long no la
llena porque tampoco colapsa.

## Próximo experimento para aislar los factores

Dos condiciones cruzarían el diseño:

1. ~~**Autómata sin auto-referencia, sin dominio de chileatiende**~~ —
   **corrido (automata_neutro, 2026-08-19)**: colapsa en curvatura como
   chileatiende y no recupera bien (peor aún: falla en recuperar en 12-24%
   de los casos). Ver `AUTOMATA_NEUTRO_REPORT.md`.
2. ~~**Auto-referencia sin arquitectura de autómata**~~ — **corrido
   (axis_pec_only, 2026-08-19)**: identidad + Triple PEC cableado, SIN
   ninguna regla de automatismo. Resultado más limpio de la serie: **no
   colapsa** (Δκ=0.001 vs vanilla, p=0.41, no significativo) y **recupera
   igual o más rápido que axis** (τ=20.6/18.5/14.5, recovery_rate=1.00
   siempre). Confirma que la recuperación depende del cableado del Factor 2,
   no de la arquitectura de autómata. Ver `AXIS_PEC_ONLY_REPORT.md`.
3. ~~**Celda cruzada: contenido de chileatiende + arquitectura SIA de
   axis**~~ — **corrido (chileatiende_sia, 2026-08-19)**: colapsa en
   curvatura igual que chileatiende puro, y **no recupera mejor — recupera
   peor** (τ=44.9 y recovery_rate=0.85 en t_inj=50, vs τ=30.4/rate=1.00 de
   chileatiende puro). Refuta la lectura simple de "agregar auto-referencia
   → recuperación tipo axis". Ver `CHILEATIENDE_SIA_REPORT.md`.

Con las tres condiciones corridas (automata_neutro, chileatiende_sia,
axis_pec_only), las cuatro celdas de la teoría de dos factores tienen
evidencia directa — ver tabla arriba, sección "Prueba directa del segundo
experimento pendiente". Próximo paso natural, si se quiere seguir afinando:
una réplica de axis_pec_only con longitud emparejada a axis (~3,945 tokens)
para descartar la longitud como variable residual, y una segunda muestra de
cada condición para reducir el n=1 por celda.

Referencias numéricas: `CHILEATIENDE_CONTROL_REPORT.md`,
`AUTOMATA_NEUTRO_REPORT.md`, `CHILEATIENDE_SIA_REPORT.md`,
`AXIS_PEC_ONLY_REPORT.md`,
`results_local/sia_extended_v5/results.json`,
`results_local/perturbation_sia_extended_v5_L30_medium/summary.json`.
