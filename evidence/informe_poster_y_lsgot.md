# Informe: "The Geometry of Identity" (Geometry.jpeg) vs. estado real del proyecto LSGOT/SIA

**Fecha original:** 2026-08-26 (pre-mediciones) · **Actualizado:** 2026-08-26 (post E-E/E-H/E-I)
**Fuentes cruzadas (pre):**
- `Geometry.jpeg` (póster conceptual, "Continuity Science — Geometric Foundations 001", autor Eduardo Filho)
- `/home/plaxius/Documentos/Proyectos/Autopoiesis/H4_rev_31B_addendum.md`
- `/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/Set_experimental.md`
- Memorias en MemoVex (agente `claude`) sobre el reencuadre densidad-de-restricción vs identidad

**Fuentes agregadas (post — mismo día, sesión posterior):**
- `TIER0_REPORT.md` (E-A/E-C/E-D/E-H corridos, 9 grupos, + comparaciones chileatiende agregadas)
- `EE_EH_WINDOW_REPORT.md` (E-E extendido a los 10 grupos, E-H ventana de perturbación)
- `EI_PERMUTATION_REPORT.md` (E-I, perturbación direccional, corrido en RunPod)
- `lsgot_4.md` (reformulación del paper solo con el panel Gemma-4-31B/SIA, sin Δκ/W₁ como evidencia primaria)

**Cómo leer este informe ahora:** cada panel tiene una subsección **Antes** (lo que decía la versión original de este informe, cuando E-H/E-I/E-E aún no se habían corrido) y **Después** (lo que cambió con las mediciones de la sesión siguiente). Los paneles sin mediciones nuevas no tienen subsección "Después" — se marcan como sin cambio.

---

## 0. Aclaración de partida

`Geometry.jpeg` **no contiene valores ni datos experimentales**. Es un póster teórico/conceptual con definiciones simbólicas (distancia d_I, métrica g_I, geodésicas γ*, curvatura) y preguntas abiertas — no un output del experimento H4. No hay "valores" que verificar contra H4 porque el póster no reporta ninguno.

Dicho eso, el marco conceptual del póster calza de forma notable con el estado real de la investigación en `Geometría_LSGOT/SIA-experiments/`, incluyendo el error metodológico que ya detectaron (reencuadre densidad-de-restricción) y los experimentos que faltan por correr (`Set_experimental.md`).

---

## 1. Mapeo panel por panel

### Panel 1 — Topología vs. Geometría ("connected states → measured manifold")
**Sin cambio.** El proyecto ya opera en este nivel: no solo preguntan si los estados están conectados, sino que miden la *forma* de la trayectoria en el espacio de representación (curvatura, dimensión, distribución). Coincide con el enfoque general de LSGOT.

### Panel 2 — Distancia de identidad, "State Distance ≠ Identity Distance"

**Antes:** hallazgo central del reencuadre, apoyado en Δκ/W₁/reducción de dimensión — lo que originalmente se interpretó como huella de identidad resultó ser **densidad de restricción** (Factor 1), no identidad. Evidencia: `H4_rev_31B_addendum.md` §2 y `ROADMAP_REENCUADRE_DENSIDAD_RESTRICCION.md`.

**Después — confirmado de forma independiente, y matizado.** `TIER0_REPORT.md` replica el mismo reencuadre con métricas que no son Δκ/W₁: `participation_ratio` (efecto grande en todas las condiciones con restricción, d=−0.98 a −1.89) y RQA determinism (dicotomía limpia: cero en axis/vanilla, alto en chileatiende/automata_neutro). La trampa que nombra el póster ya no depende de una sola métrica cuestionable — se sostiene con al menos tres vías de evidencia independientes. **Matiz nuevo:** `participation_ratio` no reproduce la dicotomía binaria de Δκ (`axis_pec_only` "no colapsa" con Δκ, p=0.41; pero sí colapsa con PR, d=−1.18) — la reducción de dimensión es **gradual**, no un interruptor Factor1/Factor2. Detalle en `lsgot_4.md` §3.1 y §4.

### Panel 3 — Estructura métrica g_I ("¿de dónde viene g_I?")

**Antes:** hueco estructural real — todo lo medido son escalares ad hoc, sin una métrica g_I derivada de restricciones de admisibilidad.

**Después — el hueco no se cerró, pero se caracterizó mejor, y en la dirección opuesta a "más cerca de g_I".** Ahora hay evidencia de que el fenómeno tiene **al menos dos componentes matemáticamente distintos** que no colapsan en un solo escalar: (a) un efecto estático/gradual de dimensión (participation_ratio, compartido por identidad y restricción) y (b) un efecto dinámico/compuerta de recuperación (τ, recovery_id, Fréchet — casi exclusivo de restricción, no de identidad). Cualquier g_I futura tendría que dar cuenta de ambos por separado, no como una sola cantidad — es información real, pero mueve la meta, no la acerca.

### Panel 4 — Dirección, "Possible Direction ≠ Identity-Admissible Direction"

**Antes:** mapeaba a E-H/E-I, diseñados pero **no corridos** — descrito como prioridad #1 del set.

**Después — corrido, con resultado de doble filo.** E-H (`TIER0_REPORT.md`, `EE_EH_WINDOW_REPORT.md`) confirma que `v_identidad` es una **dirección real**: doble disociación limpia (`axis`/`axis_pec_only` proyectan positivo, d~3.5-3.8 vs vanilla; `chileatiende`/`chileatiende_sia` proyectan **negativo**, d=−3.65 a −5.60 vs vanilla; axis vs chileatiende, d=+8.89, el efecto más grande de todo el panel). Pero E-I (`EI_PERMUTATION_REPORT.md`, perturbación a lo largo vs ortogonal a esa misma dirección) da **null result**: ninguna de las 6 condiciones alcanza p<0.05 en τ. **Es la respuesta más precisa posible a la pregunta exacta del panel:** existe una "Possible Direction" (v_identidad, medible y robusta), pero no hay evidencia de que sea "Identity-Admissible" en el sentido dinámico que el póster pregunta — el sistema no resiste ni recupera preferentemente cuando se lo empuja a lo largo de ella. Detalle en `lsgot_4.md` §3.5 y §3.7.

### Panel 5 — Geodésicas, γ* = argmin L_I(γ)

**Antes:** mapeaba a E-E (Fréchet), **propuesto, no ejecutado**.

**Después — corrido en los 10 grupos** (`EE_EH_WINDOW_REPORT.md`, extendido 2026-08-26 desde los 3 grupos originales). Respuesta a la pregunta del póster ("¿vuelve por la misma ruta o converge por otra?"): axis/axis_pec_only/axis_short no difieren de vanilla en fidelidad de ruta (todos "negligible" a "small", d≤0.36) — **no hay geodésica privilegiada para axis**. Los 4 grupos de restricción (automata_neutro, chileatiende y sus dos variantes SIA) sí se desvían significativamente más que vanilla, con efecto grande en las tres ventanas de inyección (d=0.94-2.33, p≤0.002 sin excepción). La pregunta del panel queda respondida: identidad no compra ruta fiel; restricción operativa sí la rompe.

### Panel 6 — Curvatura ("¿las restricciones de admisibilidad doblan el espacio de transformaciones posibles?")

**Antes:** colisión de términos — la κ medida es curvatura de trayectoria (Factor 1), distinta de la curvatura del espacio de transformaciones admisibles que pide el póster.

**Después — sin operacionalizar, y con menos terreno bajo los pies que antes.** `lsgot_4.md` §3.4 degrada Δκ/W₁ a evidencia secundaria: una auditoría pre-registrada (`REPORTE_FASE0.md`, panel hermano E4B) encontró que esa misma métrica no le gana a baselines simples en 12/12 comparaciones y cae dentro del ruido de split-half — nunca auditado en el panel 31B. La κ que el proyecto tenía "gastada" en Factor 1 ahora ni siquiera está firme en ese rol. La curvatura del póster sigue sin ningún intento de operacionalización.

### Panel 7 — Problema central

**Antes:** "¿qué estructura basta para derivar una geometría intrínseca de identidad persistente?" — abierto; E-H como semilla mínima de una futura base.

**Después — sigue abierto, pero el espacio de hipótesis descartables creció.** Con E-H+E-I+E-E corridos, queda descartada la hipótesis más simple ("identidad = pozo de atracción dinámico que resiste perturbación") — el Panel 4/5 la responden en negativo. Lo que sobrevive es una identidad que ocupa una **región/dirección** distinguible del espacio (estático) sin ser un **mecanismo de restauración privilegiado** (dinámico). Cualquier estructura matemática candidata a g_I tiene ahora una restricción real que satisfacer: debe producir un objeto direccional-estático sin implicar dinámica atractora — eso no resuelve el panel 7, pero elimina una familia entera de respuestas candidatas.

---

## 2. Qué se está midiendo (y qué mide realmente) — pre vs post

**Antes (2026-08-26, pre-mediciones):**

| Métrica actual | Qué mide de verdad | Factor |
|---|---|---|
| κ̄, Δκ | Curvatura local de la trayectoria | Factor 1 (densidad de restricción) |
| W₁ | Distancia distribucional | Factor 1 |
| Reducción de dimensión | Colapso del espacio efectivo | Factor 1 |
| τ (tiempo de recuperación) | Velocidad de retorno post-perturbación | Factor 2 (auto-referencia), pero un solo escalar |
| SampEnΔ | Exploración activa vs. constricción pasiva post-perturbación | Factor 2 |

Fuente: `H4_rev_31B_addendum.md` §2–§3, resultado E1 (SIA 31B: Δκ = +0.084, 3.6× el efecto de MIA) y tablas de τ/SampEnΔ por `t_inj`.

**Después — tabla ampliada, con Δκ/W₁ degradados y seis filas nuevas:**

| Métrica | Qué mide de verdad | Factor | Estado |
|---|---|---|---|
| κ̄, Δκ, W₁ | Curvatura/distancia distribucional | Factor 1 (nominal) | **secundaria** — no auditada en este panel, ver Panel 6 |
| Reducción de dimensión (vía ORC) | Colapso del espacio efectivo | Factor 1 (nominal) | **secundaria**, mismo motivo |
| `participation_ratio` | Dimensión efectiva (espectro de covarianza) | Ambos factores, **gradual** | primaria, nueva |
| RQA determinism/laminarity | Retorno recurrente a estados parecidos | Factor 1 (dicotomía limpia) | primaria, nueva — responde el punto 5 de abajo |
| Hurst | Memoria de largo alcance | Sin señal clara (negligible axis vs vanilla) | corrida, sin efecto |
| τ / recovery_rate | Velocidad de retorno post-perturbación | Factor 2 — pero **no distingue axis de vanilla** | confirmada, matizada |
| `identity_projection` (v_identidad) | Dirección estática de identidad | Factor 2, **estático**, doble disociación limpia | primaria, nueva — panel 4 |
| `recovery_id` / τ_identidad | Recuperación específica hacia v_identidad tras perturbar | Factor 1 la degrada (0.21-0.56); Factor 2 no aporta ventaja sobre vanilla | primaria, nueva — panel 4/5 |
| Fréchet normalizado | Fidelidad de ruta post-perturbación | Factor 1 (efecto grande, 4/4 grupos); Factor 2 = vanilla | primaria, nueva — panel 5 |
| E-I (along/orthogonal) | ¿La dirección de identidad es dinámicamente admisible? | **Null result** — sin evidencia de atractor direccional | primaria, nueva — panel 4 |

## 3. Qué se está midiendo ahora que antes no (y qué sigue sin medirse)

**Antes** — cinco huecos, todos sin correr:
1. Dirección admisible de identidad (panel 4) — E-H/E-I diseñados, no corridos.
2. Estructura geodésica / ruta de recuperación (panel 5) — E-E propuesto, no ejecutado.
3. Tensor métrico g_I explícito (panel 3) — sin intento.
4. Curvatura del espacio de transformaciones admisibles (panel 6) — no operacionalizada.
5. Retorno recurrente independiente de curvatura (panel 1) — E-A (RQA), no corrido.

**Después:**
1. ~~Dirección admisible de identidad~~ — **corrido.** Dirección confirmada (identity_projection); admisibilidad dinámica, null result (E-I).
2. ~~Estructura geodésica / ruta de recuperación~~ — **corrido**, 10 grupos. Sin geodésica privilegiada para identidad; sí para restricción (en sentido negativo — se desvía más).
3. **Tensor métrico g_I explícito — sigue sin intento**, y ahora con una restricción adicional que debe satisfacer (ver Panel 3, "después").
4. **Curvatura del espacio de transformaciones admisibles — sigue sin operacionalizar**, y con menos base (Δκ degradado, Panel 6).
5. ~~Retorno recurrente~~ — **corrido.** Dicotomía limpia: determinism=0 en identidad/vanilla, alto en restricción — confirma independientemente el reencuadre del Panel 2.

De los cinco huecos originales, **tres se cerraron empíricamente** (1, 2, 5) y **dos siguen exactamente abiertos** (3, 4) — que son, no por casualidad, los dos paneles que pedían una estructura matemática nueva (g_I, curvatura de admisibilidad) en vez de un experimento sobre datos ya existentes.

---

## 4. Conclusión (actualizada)

**Versión original (pre-mediciones):** el póster no aportaba datos nuevos, pero validaba que `Set_experimental.md` apuntaba al lugar correcto — E-H como el experimento de mayor payoff teórico del set, aún no corrido.

**Versión post-mediciones:** E-H, E-I y E-E ya corrieron, y la respuesta que dan es más interesante que un simple "sí, hay geometría de identidad". El póster distinguía "Possible Direction" de "Identity-Admissible Direction" (panel 4) y preguntaba por la ruta de retorno (panel 5) — la evidencia acumulada dice que la identidad en este modelo **es una dirección real y medible (estática)**, pero **no es dinámicamente admisible** en el sentido que el póster pregunta: no resiste perturbación más que un prompt genérico, no recupera más rápido, no recupera preferentemente a lo largo de su propio eje, y su ruta de retorno no es más fiel que la de vanilla. Donde sí hay una firma dinámica clara (recuperación degradada, ruta desviada, determinismo alto) es en las condiciones de **restricción operativa**, no de identidad — el reencuadre del Panel 2 queda confirmado con tres métricas independientes, no solo con la Δκ que ahora sabemos que no está auditada.

Esto dejó una consecuencia concreta fuera del marco del póster: motivó `lsgot_4.md`, una reformulación completa del paper original sobre un solo modelo (Gemma-4-31B-it) y un diseño 2×2 que separa explícitamente identidad de densidad de restricción — exactamente la distinción que el póster, sin saberlo, ya estaba pidiendo en el Panel 2.
