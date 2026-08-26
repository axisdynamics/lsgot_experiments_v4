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

## Qué cambia si estos experimentos confirman una segunda señal de identidad

Si E-H (o E-A/E-G) muestra una firma de Factor 2 independiente de τ, el
reencuadre de `ROADMAP_REENCUADRE_DENSIDAD_RESTRICCION.md` se fortalece en
vez de debilitarse: la identidad tendría entonces **dos** magnitudes propias
(recuperación agregada + anclaje direccional/recurrente), ninguna de las
cuales es la magnitud que `lsgot_3.pdf` interpretó originalmente como huella
de identidad (esa, ya establecido, es densidad de restricción). El objetivo
de este set no es rescatar la hipótesis original sino evitar reemplazarla por
otra igual de subdeterminada — un solo escalar por un solo escalar.
