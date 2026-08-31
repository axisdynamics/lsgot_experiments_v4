# Set experimental — métricas adicionales para separar Factor 1 / Factor 2

**Fecha:** 2026-08-19
**Estado:** propuesta, ningún experimento de este set corrido todavía.
**Origen:** objeción directa a que "identidad anclada" quede sostenida por un
solo escalar (τ / recovery_rate). Este documento lista qué más se puede medir
sobre las mismas trayectorias (o con extracción adicional barata) antes de
aceptar el reencuadre de `ROADMAP_REENCUADRE_DENSIDAD_RESTRICCION.md` como
resultado, no como indicio.

## Regla de diseño

Cada experimento nuevo se evalúa contra el modelo de dos factores ya
establecido (`Teoria_subconjunto_acotado.md`):
- **Factor 1 (densidad de restricción)** predice: colapso de curvatura/dim
  (ya medido — Δκ, W₁, reducción dim).
- **Factor 2 (auto-referencia cableada)** predice: recuperación rápida tras
  perturbación (ya medido — τ, recovery_rate, un solo escalar por condición).

Un experimento de este set es útil si predice de forma distinta bajo Factor 1
puro (`automata_neutro`) vs Factor 2 puro (`axis_pec_only`) vs ambos (`axis`)
vs ninguno (`vanilla`/`generic_*`). Si dos experimentos dan siempre la misma
lectura relativa entre esas cuatro condiciones, uno es redundante — no vale
la pena correr ambos.

## Tier 0 — sobre datos ya guardados, sin tocar el modelo

No requieren RunPod ni nueva extracción; se calculan sobre
`results_local/sia_extended_v*/results.json` y
`results_local/perturbation_sia_extended_v*/summary.json` que ya existen.

### E-A — Recurrence Quantification Analysis (RQA)

- **Qué mide:** determinismo, laminaridad, trapping time sobre la matriz de
  recurrencia de la trayectoria (umbral de distancia sobre pares v_i, v_j).
  Captura retorno recurrente a estados parecidos — algo que Δκ (local, par a
  par) no ve porque es agregado sobre vecinos inmediatos, no sobre toda la
  trayectoria.
- **Datos:** la trayectoria completa ya guardada, mismo insumo que el grafo
  k-NN existente.
- **Predicción:** si "ancla" es literal, `axis`/`axis_pec_only` deberían tener
  laminaridad alta (retorno a la misma región) incluso sin colapso de
  curvatura. `automata_neutro`/`chileatiende` podrían tener determinismo alto
  (por las reglas trigger→salida) sin laminaridad — el patrón que distinguiría
  "restringido" de "anclado" es determinismo-sin-laminaridad vs
  laminaridad-con-o-sin-determinismo.
- **Condiciones prioritarias:** axis, axis_pec_only, automata_neutro, vanilla.
- **Prioridad:** alta — costo de cómputo bajo, ya hay librerías (`pyrqa`,
  `nolds`).

### E-B — Exponente de Lyapunov local (proxy)

- **Qué mide:** tasa de divergencia entre trayectorias vecinas (mismo prompt,
  perturbaciones de semilla/ruido pequeño) — dinámica, no una foto estática
  como Δκ.
- **Datos:** requiere múltiples corridas con micro-perturbación por prompt
  (ya existe infraestructura de perturbación de H4_rev, reusar con σ mucho
  menor que el σ_medium calibrado).
- **Predicción:** Factor 2 (axis_pec_only) → exponente bajo/negativo
  (trayectorias vecinas convergen, consistente con recovery_rate=1.00).
  Factor 1 sin Factor 2 (automata_neutro) → exponente alto o bimodal,
  coherente con el 12-24% de trayectorias que no recuperan.
- **Condiciones prioritarias:** axis_pec_only, automata_neutro.
- **Prioridad:** media — reusa la infraestructura de H4_rev pero multiplica el
  número de corridas.

### E-C — Exponente de Hurst

- **Qué mide:** memoria de largo alcance de la serie de posiciones (o de la
  serie de velocidad) — persistencia (H>0.5) vs anti-persistencia (H<0.5).
  A diferencia de τ, no depende de una ventana de perturbación: se puede medir
  en trayectoria libre (sin perturbar), lejos de cualquier evento de
  recuperación.
- **Datos:** trayectoria libre ya guardada.
- **Predicción:** si la identidad ancla de forma continua (no solo reactiva
  ante perturbación), `axis_pec_only` debería mostrar H sostenido >0.5 en toda
  la ventana, no solo en la fase de recuperación. Si H no difiere de vanilla
  fuera de la ventana de perturbación, el "anclaje" sería puramente reactivo
  (solo aparece cuando se lo perturba), lo cual también es un hallazgo, no un
  null result.
- **Condiciones prioritarias:** las 9 del panel — es barato, correr todas.
- **Prioridad:** alta — barato, resultado interpretable en ambas direcciones.

### E-D — Participation ratio / rango efectivo de la covarianza

- **Qué mide:** dimensión efectiva vía el espectro de covarianza de la
  trayectoria, cross-check del estimador de dimensión intrínseca ya usado
  (evita depender de un solo estimador para reclamar "reducción de
  dimensión").
- **Datos:** trayectoria libre ya guardada.
- **Predicción:** debería correlacionar fuerte con la reducción de dimensión
  ya reportada — este experimento es principalmente un control de robustez,
  no una fuente de señal nueva. Útil para blindar la sección de métodos, no
  para el reencuadre identidad-vs-restricción en sí.
- **Prioridad:** baja — control, no descubrimiento.

### E-E — Distancia de Fréchet: trayectoria perturbada vs original

- **Qué mide:** si la trayectoria post-perturbación recupera *por la misma
  ruta* o simplemente converge a un punto parecido por otro camino. τ y
  recovery_rate solo miden similitud coseno al centroide en el punto final —
  no dicen nada sobre la ruta.
- **Datos:** pares de trayectorias (perturbada, no perturbada) ya guardados
  en los summary.json de H4_rev.
- **Predicción:** si `axis_pec_only` "recupera igual de rápido que axis" pero
  por una ruta distinta cada vez (Fréchet alto pese a recovery_rate=1.00),
  eso debilita la lectura de "atractor único de identidad" y la deja como
  "amortiguamiento genérico rápido" — distinción que el set actual no puede
  hacer.
- **Condiciones prioritarias:** axis, axis_pec_only, chileatiende (para
  comparar un caso que sí recupera lento con uno que recupera rápido).
- **Prioridad:** alta — reusa datos existentes, cierra un hueco real en la
  interpretación de τ.

## Tier 1 — requiere reprocesar corridas ya hechas con nueva extracción

Necesita volver a correr inferencia sobre los mismos prompts/condiciones ya
diseñados, pero extrayendo capas o cabezas de atención que hoy se descartan.

### E-F — Trayectoria vertical (cross-layer, token fijo)

- **Qué mide:** cómo converge la representación por profundidad de capa
  dentro de un mismo paso de generación (estilo logit-lens/tuned-lens),
  en vez de la trayectoria horizontal (cross-token, última capa) que se usa
  hoy.
- **Datos:** requiere guardar el hidden state en todas las capas, no solo la
  última pre-`lm_head` — cambio de extracción, no de análisis.
- **Predicción:** si la identidad estabiliza más temprano en profundidad bajo
  `axis`/`axis_pec_only` que bajo `automata_neutro`, es una firma de Factor 2
  ortogonal a τ — aparece en una dimensión (capa) que el análisis horizontal
  ni siquiera mide.
- **Prioridad:** media-alta conceptualmente, pero implica re-extracción
  completa — evaluar costo de GPU antes de comprometer.

### E-G — Atención al system prompt durante la generación

- **Qué mide:** masa de atención que cada token generado dirige de vuelta a
  los tramos del prompt que declaran identidad/mantra vs los que declaran
  reglas operativas (bloques, triggers). Canal distinto (atención, no hidden
  states) — la medida más directa de "auto-referencia activa" que existe en
  este diseño.
- **Datos:** requiere guardar attention weights por cabeza/capa durante la
  generación — extracción nueva, mucho más pesada en disco que hidden states.
- **Predicción:** `axis`/`axis_pec_only` deberían mostrar atención sostenida
  o creciente hacia el tramo de identidad a lo largo de los 256 tokens;
  `automata_neutro`/`chileatiende` deberían mostrar atención concentrada en
  los tramos de reglas/triggers, no en ningún tramo "identitario" (no lo
  tienen). Es la prueba más directa posible del mecanismo, al costo de ser la
  más cara de extraer.
- **Prioridad:** alta en valor explicativo, baja en factibilidad inmediata —
  candidata a una corrida dedicada, no a un análisis retroactivo.

## Tier 2 — requiere diseño experimental nuevo (nueva corrida en RunPod)

### E-H — Proyección sobre dirección de identidad extraída (persona-vector)

- **Qué mide:** define v_identidad = media(activaciones bajo axis) −
  media(activaciones bajo generic) (mismo método que persona-vector research,
  ya citado en `lsgot_3.pdf` intro: Chen et al. 2025, Lu et al. 2026), y mide
  la proyección de v_t sobre esa dirección a lo largo de la trayectoria,
  dentro y fuera de la ventana de perturbación.
- **Datos:** las activaciones ya extraídas alcanzan para calcular v_identidad
  (contraste axis vs generic_long, longitud emparejada); la proyección se
  computa sobre trayectorias ya guardadas — no requiere nueva corrida de
  generación, solo el vector de contraste.
- **Predicción:** convierte "identidad" de un efecto agregado-emergente a una
  dirección concreta. Si τ bajo en axis_pec_only refleja recuperación
  específicamente *hacia* v_identidad (proyección vuelve a su valor
  pre-perturbación tan rápido como el recovery_rate global), eso es evidencia
  fuerte de anclaje real. Si la proyección sobre v_identidad se recupera más
  lento que el resto de la trayectoria (recovery_rate global alto pero
  proyección-identidad rezagada), el "anclaje" es más débil de lo que τ solo
  sugiere.
- **Condiciones prioritarias:** axis, axis_pec_only, automata_neutro (control
  negativo: no debería tener proyección significativa sobre v_identidad en
  absoluto, al no compartir contenido identitario).
- **Prioridad:** la más alta del set completo — reutiliza el aparato
  conceptual del propio `lsgot_3.pdf` (dirección vs magnitud) y no requiere
  nueva corrida de generación, solo cómputo adicional sobre lo ya extraído.

### E-I — Perturbación direccional (a lo largo vs ortogonal a v_identidad)

- **Qué mide:** en vez de ruido isotrópico (H4_rev actual), perturbar
  específicamente a lo largo de v_identidad (E-H) vs en direcciones
  ortogonales a ella, y comparar τ/recovery_rate entre ambas.
- **Datos:** requiere nueva corrida de perturbación (mismo pipeline de
  H4_rev, cambiando la distribución del ruido inyectado) — depende de tener
  primero v_identidad de E-H.
- **Predicción:** si `axis_pec_only` recupera rápido solo cuando la
  perturbación es ortogonal a v_identidad pero lento/incompleto cuando es a
  lo largo de ella (o al revés), eso es la evidencia más fuerte posible de
  "hay un atractor de identidad específico direccional" — un solo τ agregado
  sobre ruido isotrópico no puede distinguir esto, que es exactamente la
  objeción de partida de este documento.
- **Condiciones prioritarias:** axis_pec_only (Factor 2 puro), axis (ambos
  factores) — comparar si la direccionalidad del atractor cambia cuando
  además hay densidad de restricción.
- **Prioridad:** alta en valor, condicionada a que E-H se corra primero.

## Orden de ejecución sugerido

1. **E-H** (proyección sobre v_identidad) — sin nueva corrida de GPU, mayor
   valor conceptual, y es prerrequisito de E-I.
2. **E-A + E-C + E-E** (RQA, Hurst, Fréchet) — todos sobre datos ya
   guardados, en paralelo, bajo costo.
3. **E-D** — control de robustez, correr junto con el paso 2 si hay tiempo.
4. **E-I** (perturbación direccional) — una vez que E-H dé un v_identidad
   utilizable.
5. **E-B** (Lyapunov local) — evaluar si el costo de multiplicar corridas de
   perturbación se justifica según lo que ya muestre E-I.
6. **E-F, E-G** (cross-layer, atención) — quedan para una ronda dedicada de
   RunPod con extracción ampliada; no bloquean el reencuadre central, lo
   profundizan si los pasos 1-5 confirman que hay más de una señal de
   identidad.

## Anexo 2026-08-28 — FASE 0 de INSTRUCCIONES_AGENTE_LSGOT.md (E-J, E-K, E-L, E-F2, E-H2)

Set anexado (no reemplaza lo anterior) tras el reencuadre de `lsgot_4.md`
v0.1: identidad = dirección estática (v_identidad, d=8.89), sin atractor
(E-I null). Ver `INSTRUCCIONES_AGENTE_LSGOT.md` para el diseño completo de
E-J/E-K/E-L/E-F2/E-H2.

| ID | Estado | Fecha | Reporte |
|---|---|---|---|
| E-L (primer token) | ✅ corrido (limpio, sin chileatiende) | 2026-08-28 | `EL_REPORT.md` — doble disociación de v̂ ya presente en t=0 (d=10.22 axis vs vanilla); identidad = propiedad del estado de contexto |
| E-H2 (serie temporal v̂) | ✅ corrido (limpio, sin chileatiende) | 2026-08-28 | `EH2_REPORT.md` — identidad: autocorr baja (~0.27)/ráfagas largas (~8 tokens); restricción (automata_neutro): ráfagas cortas (2.78) y proyección casi nula, sin la autocorrelación elevada que sugería la versión con chileatiende (era artefacto de esa) |
| E-J (RDM/CKA) | ✅ recalculado (limpio, sin chileatiende) | 2026-08-28 | `EJ_REPORT.md` — reescrito desde cero. automata_neutro es la condición más distinta del panel completo (RDM/CKA/ángulos, 3 métodos independientes). Jerarquía revisada: restricción > identidad (sin el término espurio "dominio", que era el propio confound de chileatiende) |
| E-F2 (per-capa v̂+PR) | ✅ corrido (limpio, sin chileatiende) | 2026-08-28 | `EF2_REPORT.md` — ⚠️ corrección: v_identidad.npy es de CAPA FINAL (layer_idx=-1), no L30 (verificado en sia/run_exp.py:134); el efecto crece monótonamente hacia capas tardías (d=10.43 en L55 para axis_pec_only vs automata_neutro), débil/inconsistente en L5-L25. PR tiene pico de separación distinto, en L25-L30 |
| Corrección v̂ sin chileatiende | ✅ hecho | 2026-08-28 | `CORRECCION_DVHAT_SIN_CHILEATIENDE.md` — restricción (automata_neutro) tiene huella propia y robusta (d=−2.40 v̂, d=−1.79 PR, ambas p<0.001), independiente de chileatiende. Cifra insignia del paper (d=8.89, axis_pec_only vs chileatiende) corregida a d=5.52 (vs automata_neutro) |
| E-K (steering causal) | ⏳ pendiente (intentado, bloqueado) | 2026-08-28 | `EK_REPORT.md` — mecanismo de intervención (vector aditivo fijo h+α·‖h‖·v̂, sostenido) es inestable en Gemma-4-31B: 17/17 combinaciones probadas (2 capas, 6 α desde 0.05, 2 modos de inyección, v̂ Y control aleatorio) degeneraron en repetición. Control aleatorio (T6) descarta que sea específico de v̂ — es el mecanismo. Pregunta causal de E-K SIGUE ABIERTA, no respondida negativamente. Requiere mecanismo de steering distinto (inyección puntual única tipo E-I, o clamping de norma) antes de reintentar. No usar chileatiende-family (confound de markup) ni v̂_L30 (casi ortogonal a v̂_final, ver EF2_REPORT.md §7) cuando se retome |

## ⚠️ REVISIÓN HUMANA PENDIENTE — lsgot_4.md necesita actualizarse (T11)

**No editado por el agente** (decisión del usuario 2026-08-28: el paper es
de autoría compartida — Castillo, Torres Yévenes, Lanas — requiere
revisión humana explícita, no edición mecánica). Secciones afectadas por
el confound de markup de chileatiende-family (ver T11 abajo), con
reemplazo limpio disponible:

| Línea aprox. | Contenido afectado | Reemplazo limpio |
|---|---|---|
| L74-79 | Tabla de condiciones — describe chileatiende/chileatiende_sia/chileatiende_sia_v2 sin advertencia de confound | `CHILEATIENDE_MARKUP_CONFOUND_REPORT.md` |
| L118-121 | PR: `chileatiende_sia` d=−1.89 citado como "el mayor efecto" junto a automata_neutro | `automata_neutro` (d=−1.79 vs vanilla) ya sostiene esto solo — `EJ_REPORT.md` |
| L133-136 | Métricas de trayectoria (velocidad, SampEn, alineación P→R) de chileatiende/chileatiende_sia citadas como "outliers" | `AUTOMATA_NEUTRO_REPORT.md` (versión limpia, editada 2026-08-28) |
| L149-154 | H4_rev τ/recovery_rate — la narrativa "chileatiende_sia recupera peor, v2 lo mejora parcialmente" | Retirado en `AXIS_PEC_ONLY_REPORT.md` conclusión #3 — sin confirmar con datos limpios |
| **L162** | **§3.5 — "this paper's strongest single result": d=+8.89 entre axis y chileatiende** | **Corregido a d=+5.52 (axis_pec_only vs automata_neutro) en `CORRECCION_DVHAT_SIN_CHILEATIENDE.md`** — la cifra insignia del paper necesita reemplazo |
| L173-178 | recovery_id: chileatiende-family en 0.21-0.56 | `automata_neutro` solo (0.43-0.67) ya sostiene la disociación — `EE_EH_WINDOW_REPORT.md` |
| L191-195 | Fréchet/route-fidelity: chileatiende-family con d=0.94-2.33 | `automata_neutro` solo (d=1.07-1.35, p≤0.001 en los 3 t_inj) — `EE_EH_WINDOW_REPORT.md` |
| L239 | Discusión: "chileatiende, chileatiende_sia... diverge from it" como parte central del argumento de §3.5 | Ídem L162 |
| L223, L249 | Nota de limitación: "token counts... not recorded" para chileatiende_sia/_v2 | Ya no aplica — condiciones excluidas, no pendientes de medir |
| L261 | Tabla de densidad de restricción (Apéndice A) — esta es análisis del **texto del prompt**, no de trayectorias generadas — NO afectada por el confound | Sin cambios necesarios |

**Recomendación:** la corrección más urgente es L162 (§3.5) — es la cifra
que el paper llama su resultado más fuerte. El reencuadre cualitativo
sobrevive (`automata_neutro` solo confirma todo), pero el número
específico necesita reemplazo antes de que el paper avance.

## T10 — Validación cross-modelo (2026-08-29, RESUELTO parcialmente)

Réplica de FASE 0 completa (E-L, E-H2, E-J, E-F2) + A2 en Qwen3-32B
(dense, arquitectura distinta a Gemma4 — GQA+QK-norm, 64 capas, tokenizer
distinto), mismas 7 condiciones limpias, mismos prompts, sin
re-balancear longitud. Ver `QWEN3_VALIDATION_REPORT.md`.

**Replica robustamente:** doble disociación en t=0 (d=+12.97, más fuerte
que en Gemma), PR de restricción (más limpio: 12/12 capas sin excepción),
separación identidad/restricción como el mayor ángulo de subespacio del
panel (40.7°), crecimiento monótono de v̂ por capa.
**No universal / matizado:** la jerarquía "restricción > identidad" de
Gemma se equilibra en Qwen3 (identidad lidera el clustering ahí, no
restricción). **No replica:** A2 (rotación vertical) — en Qwen3 las 7
condiciones rotan casi idéntico, `automata_neutro` no es outlier de
rotación como en Gemma.

**No corrido:** perturbación (H4_rev) en Qwen3 — solo trayectoria libre.

## T11 — Confound de markup HTML en chileatiende-family (2026-08-28)

`chileatiende`, `chileatiende_sia`, `chileatiende_sia_v2` fuerzan un wrapper
HTML literal (`<div class="respuesta-bot"...>`) en el 100% de las
respuestas — 43.6% / 34.6% / (misma regla, no medido directamente) de cada
respuesta es texto idéntico repetido entre las 20 trayectorias. Mismo
mecanismo que el fallo ya conocido de "ADN mal encarnado" (repetición
verbatim del prompt inflando curvatura/determinismo). `automata_neutro`,
`axis`, `vanilla` están limpias (0.0% markup) y siguen siendo válidas.
**Resuelto 2026-08-28 (noche): chileatiende/chileatiende_sia/
chileatiende_sia_v2 se eliminaron por completo (no solo se marcaron) de
EL/EH2/EJ/EF2_REPORT.md** — EJ_REPORT.md se recalculó desde cero sin esa
familia (resultado más limpio, ver arriba); los demás se editaron
quitando filas/pares. **Estas 3 condiciones quedan excluidas del set
activo de condiciones del proyecto** hasta que exista una re-extracción
con el markup removido del texto antes de tokenizar (no es un post-proceso
sobre embeddings ya extraídos). Detalle completo y evidencia cuantitativa:
`CHILEATIENDE_MARKUP_CONFOUND_REPORT.md`.

Datos crudos usados (no están en `LSGOT_v4/data/`, que solo tiene métricas
agregadas): `/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local/sia_extended_v5/*_embeddings.npz`
y `.../perturbation/results/perturbation_sia_L30_medium/trajectories/chileatiende_sia_v2_baseline_embeddings.npz`.
Scripts: `LSGOT_v4/scripts/fase0/analyze_E{L,H2,J}_*.py`.

**Hallazgo no anticipado por el diseño original:** el diseño de E-J asumía
que las 4 condiciones "con identidad" (axis, axis_short, axis_pec_only,
chileatiende_sia_v2) formarían un cluster natural en la RDM. No ocurre:
chileatiende_sia_v2 comparte dominio/formato con chileatiende y
chileatiende_sia (todas generan HTML sobre pensiones) y ese confound de
dominio domina la distancia cruda por un orden de magnitud sobre cualquier
efecto de identidad. Cualquier corrida futura de E-J debe controlar por
dominio antes de interpretar distancias o CKA entre condiciones de
familias temáticas distintas (ver `EJ_REPORT.md` §3.2 para la
descomposición correcta).

## Qué cambia si estos experimentos confirman una segunda señal de identidad

Si E-H (o E-A/E-G) muestra una firma de Factor 2 independiente de τ, el
reencuadre de `ROADMAP_REENCUADRE_DENSIDAD_RESTRICCION.md` se fortalece en
vez de debilitarse: la identidad tendría entonces **dos** magnitudes propias
(recuperación agregada + anclaje direccional/recurrente), ninguna de las
cuales es la magnitud que `lsgot_3.pdf` interpretó originalmente como huella
de identidad (esa, ya establecido, es densidad de restricción). El objetivo
de este set no es rescatar la hipótesis original sino evitar reemplazarla por
otra igual de subdeterminada — un solo escalar por un solo escalar.
