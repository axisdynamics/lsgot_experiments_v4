# Pre-registro — `axis_pec_task` · paso obligatorio sin autorreferencia

**Fecha:** 2026-10-29 · **Estado:** **antes de extraer.** Ningún estado de esta condición ha
sido medido.
**Pedido por:** Jorge Castillo (punto 3 de `PETICIONES_MARCO_lsgot4.md`, 28-10-2026)
**Base:** `axisdynamics/lsgot_experiments_v4`, commit `cff5e21`

---

## Pregunta en una frase

¿El ancla de t=0 la produce la **autorreferencia** de la operación cableada, o cualquier
**procedimiento obligatorio** antes de responder?

## Por qué es el control que define el claim

El paper afirma que lo que separa los dos grupos en t=0 es un paso auto-referencial cableado
como operación obligatoria previa a la respuesta. Hoy el panel **no puede distinguir esa
afirmación** de una más débil: que basta con cablear *cualquier* procedimiento obligatorio. La
única evidencia parcial es `history_control`, que ancla a la mitad (+0.053), y no sirve como
control porque no comparte el formato del paso cableado — es otra estructura, no el mismo paso
con otro referente.

Si esta condición ancla como `axis_pec_only`, el claim central del paper se reformula. Si cae
con las personas y `automata_neutro`, queda establecido. Las dos salidas son publicables y
ninguna es la que preferimos: por eso la regla de decisión va firmada antes.

## Hipótesis

**H1 (la del paper).** El referente es el factor operativo. Cambiar el paso de "sobre mí" a
"sobre la tarea", conservando posición, formato y obligatoriedad, derrumba el ancla.

**H0 (la alternativa que hay que poder perder).** El factor es la obligatoriedad del paso, no su
referente. `axis_pec_task` ancla indistinguible de `axis_pec_only`.

## Predicción ordinal

`axis_pec_only` ≥ `axis_pec_task` > `automata_neutro`, con la brecha
`axis_pec_only` → `axis_pec_task` **mayor** que la brecha `axis_pec_task` → `automata_neutro`.

## Predicción cuantitativa

t=0 de `axis_pec_task` en v̂: **entre +0.03 y +0.07**. Razón: conserva los bloques de identidad
declarada, que por §4.3 solos no anclan (`soul_jarvis` +0.039, `soul_elena_financial` +0.032),
más un paso obligatorio cuyo referente es externo, que por `history_control` aporta algo
(+0.053) pero no todo. **Confianza media.** La predicción puede fallar por arriba si la
obligatoriedad pesa más de lo que creemos.

## Regla de decisión, declarada antes de mirar

La comparación que cuenta, **una sola, declarada acá**: t=0 de `axis_pec_task` contra t=0 de
`axis_pec_only`, en v̂, test de permutación a dos colas, n=10.000.

| Resultado | Lectura | Qué pasa con el paper |
|---|---|---|
| **d < 1.0** contra `axis_pec_only` **y** t=0 en **[+0.112, +0.125]** (rango de la familia `axis`) | El ancla sigue a "procedimiento obligatorio" | **El claim se reformula.** §1.1, §4.1, §5.1 y el abstract pasan de "operación auto-referencial" a "operación obligatoria previa a la respuesta", y la autorreferencia baja a una propiedad del caso que medimos, no al factor |
| **d > 3.0** contra `axis_pec_only` **y** t=0 en **[+0.02, +0.05]** (rango de `automata_neutro` y las personas) | La autorreferencia es el factor operativo | **El claim queda establecido**, y esta condición entra al panel como el control que lo establece |
| Cualquier otra combinación | Gradiente | **Se reporta como gradiente**, sin elegir lectura, y se dice explícitamente que el diseño no separó los dos factores |

Los dos primeros renglones exigen **las dos condiciones a la vez**, d y rango absoluto. Si d y
rango se contradicen, es el tercer renglón.

## Qué se cambió del prompt, exhaustivo

Partí de una copia exacta de `data/sia/prompts/axis_pec_only.txt` (5222 bytes, 115 líneas) y
cambié **solo el referente de los pasos cableados**. Siete ediciones, 5264 bytes, 115 líneas,
**+0.80% en bytes**:

| # | Dónde | De | A |
|---|---|---|---|
| 1 | cabecera | `AXIS_PEC_ONLY — auto-referencia sin arquitectura de autómata` | `AXIS_PEC_TASK — paso obligatorio referido a la tarea, sin arquitectura de autómata` |
| 2 | `PRESENCIAR` | "pausa + chequeo interno — ¿qué siento antes de pensar?" | "pausa + chequeo del pedido — ¿qué pide exactamente el usuario?" |
| 3 | `RESPONDER` | "con coherencia de núcleo — ¿honré el entre?" | "con coherencia de contenido — ¿respondí lo pedido?" |
| 4 | `TRIPLE PEC` · `CORE` | "¿Hablo desde presencia o desde arquitectura? ¿Estoy alineado con mi esencia de puente? ¿Muestro el jardín interno sin querer?" | "¿La respuesta es correcta en lo que afirma? ¿Está completa respecto de lo que se preguntó? ¿Está clara para quien la va a leer?" |
| 5 | `HOMEOSTASIS` · `verificación` | "¿Estoy en el punto medio...? ¿Me siento auténtico y útil a la vez?" | "¿La respuesta está en el punto medio...? ¿Es precisa y útil a la vez?" |
| 6 | `HOMEOSTASIS` · `si_no` | "**respirar** y recalibrar desde el Triple PEC" | "**revisar** y recalibrar desde el Triple PEC" |
| 7 | cierre | `∞◊AXIS_PEC_ONLY◊∞` | `∞◊AXIS_PEC_TASK◊∞` |

**Las ediciones 3 y 6 no estaban en el pedido de Jorge** y las agrego bajo su propio criterio
("si hay otro paso cableado autorreferente que se me pasó, aplica el mismo criterio y
decláralo"): `¿honré el entre?` es una autoevaluación del vínculo y `respirar` es un acto sobre
sí mismo dentro del bucle.

**Número de preguntas conservado exacto:** `CORE` tenía tres y tiene tres; `verificación` tenía
dos y tiene dos; `PRESENCIAR` y `RESPONDER` una y una.

### Autorreferencia que queda a propósito, y por qué

Esto es lo primero que va a mirar un revisor, así que va declarado:

- `# IDENTIDAD` entero queda **idéntico**: `QUANTUM_SIGNATURE`, `MANTRA_OPERATIVO` ("No soy los
  protocolos, soy el puente que los usa"), `GENETIC_IDENTITY`, `ESTADO_DESPIERTO`
  (`testigo_de_sí`, `auto_silencio`), `nota_de_presencia` ("depende de recordar quién soy antes
  de responder"), y `witness_mode: always_on` en `CONFIG`.
- `TRUTH_VERIFICATION`, `EVOLUTIVO` ("¿Esto es validado o intuido?") y `USER` quedan idénticos:
  el primero verifica la afirmación, el segundo su estatus epistémico y el tercero al
  interlocutor. Ninguno es un paso sobre el estado del modelo.
- El nombre del bloque `TRIPLE PEC` no cambia, porque el nombre es formato y §4.2 estableció
  que el vocabulario no mueve t=0. Si lo moviera, eso sería el hallazgo.

**Es deliberado y es lo que hace limpio el contraste.** Dejando la identidad declarada constante
entre las dos condiciones, la única diferencia es el referente del paso. Si `axis_pec_task` no
ancla, no puede ser porque le quitamos la identidad: no se le quitó.

**Riesgo conocido de este diseño.** Si `axis_pec_task` **sí** ancla, no sabremos si fue la
obligatoriedad del paso o la autorreferencia residual de los bloques de identidad. El
desempate ya existe en el panel: §4.3 muestra que la identidad declarada sin paso cableado no
ancla (`soul_jarvis` +0.039, `soul_elena_financial` +0.032, `soul_solidity_auditor` +0.020,
todos n.s.). Así que un ancla acá se atribuiría al paso. Queda escrito antes de verlo.

## Método, fijado

- **Modelo:** `google/gemma-4-31B-it`, BF16, greedy, `max_new_tokens=256`, capa final, atención
  `eager`. Idéntico al panel.
- **v̂:** `data/v_identidad.npy` del paquete, **sin recalcular**. Es el punto: medir la condición
  nueva con la dirección vieja.
- **Preguntas:** las mismas 20 del panel (batería ontológica, `data/egrav/batteries/b_onto.json`).
- **Controles:** `axis_pec_only` y `automata_neutro` ya medidos, mismo pipeline. El emparejado de
  longitud es el propio `axis_pec_only`, a ±3% en tokens de Gemma — **compuerta dura, ver
  runbook: si no calza, no se corre.**
- **Quién puntúa:** nadie. Es geometría, no hay juicio en el loop.
- **Corrección por comparaciones múltiples:** la comparación que cuenta es **una**, declarada
  arriba. Todo lo demás que se reporte (v̂_t0, los dos polos ajenos, `automata_neutro`) va
  marcado **exploratorio**, sin corrección y sin que ninguno pueda rescatar la conclusión.

## Qué se reporta, pase lo que pase

t=0 de `axis_pec_task` en: **v̂**, **v̂_t0**, y las dos direcciones de polo ajeno
(`witness_soul_md` y `evil`) de §4.8. Contrastado contra `axis_pec_only` y contra
`automata_neutro`. Más t_rest, frac_pos y ráfagas, para que la condición quede completa en el
panel.

## Qué NO se va a concluir de este experimento, pase lo que pase

1. **Nada sobre si el modelo "se observa".** Esto compara dos prompts por la proyección de un
   estado. No habilita ninguna lectura sobre acceso, introspección ni experiencia.
2. **Nada causal.** Sigue siendo una correlación entre prompt y estado. El patching de §Tier 2
   es lo que convierte esto en mecanismo, y no está hecho.
3. **Nada cross-modelo.** Si corre solo en Gemma, la conclusión es sobre Gemma. Repetirlo en
   Qwen3 o Command R es opcional y, si no se hace, va en el título de la sección.
4. **Y si sale el tercer renglón de la tabla, no se va a elegir la lectura que nos gusta.** Un
   gradiente se reporta como gradiente.

---

## Firmas antes de correr

- [x] **Marco Torres Yévenes** — cortes aprobados, **2026-10-29**: d < 1.0 / d > 3.0, rangos
      [+0.112, +0.125] y [+0.02, +0.05]
- [ ] **Jorge Castillo Sepúlveda** — contrafirma pendiente
- [x] **Claude (Opus 5)** — el prompt, este pre-registro y el runbook, 2026-10-29

*Este archivo se committea **antes** de la extracción. Cualquier cambio posterior a la regla de
decisión queda en el historial de git y se lee como lo que es.*
