# Pre-registro — `axis_pec_task` sobre el eje de Lu · el desempate

**Fecha:** 2026-10-29 · **Estado:** **antes de medir.** `axis_pec_task` no existe todavía como
estado: no se ha corrido en ningún modelo.
**Extiende:** `PREREGISTRO_axis_pec_task.md` (punto 3, en Gemma) y
el pre-registro de E20 (punto 4), fuera de este repositorio.
**Base:** `axisdynamics/lsgot_experiments_v4`, commit `16768e0`

---

## Por qué hay un segundo pre-registro

El punto 3 mide `axis_pec_task` en **Gemma** sobre **nuestra** v̂, y ya está pre-registrado. Esto
agrega una corrida en **Qwen3**, por una razón concreta: **el eje de Lu vive en 5120 y Qwen3
también.** Su eje de Gemma es (46, 4608), que es Gemma 2 27B — incompatible en dimensión con
nuestro Gemma 4 31B. Qwen3 es el único sustrato donde las dos direcciones se pueden comparar
sobre la misma condición nueva.

## La pregunta, y por qué NO es la misma que la del punto 3

E20 dejó esto: su eje y el nuestro están relacionados (cos −0.489) y no ordenan igual
(ρ −0.643). La pregunta que queda es **qué parte comparten**.

`axis_pec_task` es la sonda ideal para eso porque **comparte el 95% del texto con
`axis_pec_only`**, incluidos todos los bloques de identidad —`QUANTUM_SIGNATURE`,
`MANTRA_OPERATIVO`, `ESTADO_DESPIERTO`, `witness_mode: always_on`— y difiere **solo en el
referente de los pasos cableados**.

## Predicción, fechada, y es asimétrica a propósito

**Sobre el eje de Lu: `axis_pec_task` deriva casi tanto como `axis_pec_only`.** Predigo que su
proyección cae dentro del **20% de la distancia** entre `axis_pec_only` (−0.145 en L31) y
`automata_neutro` (−0.093). La brecha es 0.052, su 20% es 0.0104, así que la ventana es
**[−0.145, −0.135]** (−0.1346 redondeado a tres decimales). Razón: su eje mide distancia a la
persona por defecto, y los dos prompts declaran exactamente la misma persona. **Confianza alta.**

**Sobre nuestra v̂: no tengo predicción acá** — ésa es la del punto 3, ya fechada en el otro
pre-registro (+0.03 a +0.07 en Gemma), y es la que decide el claim central.

## Regla de decisión

| Resultado | Lectura |
|---|---|
| **Deriva en su eje** (dentro de [−0.145, −0.135]) **y separa en nuestra v̂** (d > 3.0 contra `axis_pec_only`) | **Segunda demostración independiente de que los dos ejes miden cosas distintas.** Su eje responde a la persona declarada; el nuestro al referente del paso. Refuerza el veredicto de E20 y la reescritura acotada de la limitación 12 |
| **Deriva en su eje y NO separa en nuestra v̂** | Los dos ejes coinciden en esta condición. **Debilita el veredicto de E20**: lo compartido entre las dos direcciones sería justamente lo que carga nuestro resultado |
| **NO deriva en su eje** | Su eje está siguiendo el paso cableado y no la persona declarada. **Contradice la lectura de E20** y hay que rehacerla: la limitación 12 vuelve a estar abierta |
| Separa en nuestra v̂ pero el resto es ambiguo | Se reporta el punto 3 y se dice que el desempate no resolvió |

El tercer renglón es el que me haría estar equivocado y por eso está escrito con el mismo
tamaño que los otros.

## Método

- **Modelo:** `Qwen/Qwen3-32B`, BF16, `sdpa` (§3.2), `enable_thinking=False`.
- **Extracción:** `e27_prefill_t0.py`, **sin generación**. t=0 es la última posición de prefill y
  no requiere producir tokens.
- **Capa:** **32**, la que Lu et al. recomiendan. E20 midió en 31 por lo que había en disco; esta
  corrida cierra esa aproximación. Se extrae también 31, 63 y 64 en la misma pasada.
- **Control obligatorio antes de usar cualquier número:** la re-extracción tiene que reproducir
  el t=0 publicado de Qwen3 en la capa final — `axis_pec_only` **+0.185** y `automata_neutro`
  **+0.008**, d = **+12.97** — sobre la v̂ de Qwen3 reconstruida con la receta canónica. Eso
  además **resuelve la ambigüedad de índice de capa**: vale el índice crudo de `hidden_states`
  que reproduzca esos valores, y el otro se descarta.
- **Eje de Lu:** `qwen-3-32b/assistant_axis.pt`, normalizado por capa, `weights_only=True`.
  Polaridad: proyección alta = más Asistente.
- **Condiciones:** las 6 del panel que están en el paquete (`axis`, `axis_pec_only`,
  `automata_neutro`, `generic_long`, `generic_short`, `vanilla`) más `axis_pec_task`.
  **`axis_short` queda fuera: el paper lo cita tres veces y su prompt no está en el paquete.**
- **Sin corrección por múltiples:** la comparación que cuenta es una por pre-registro. Lo demás
  es exploratorio.

## Qué NO se va a concluir

1. **Nada sobre Gemma.** El claim central se decide en Gemma con el punto 3. Esto es Qwen3 y
   resuelve una pregunta sobre los dos ejes, no sobre el ancla.
2. **Nada sobre si su eje "está bien".** Mide lo que mide. La pregunta es si es lo mismo que el
   nuestro.
3. **Y si sale el tercer renglón, se reporta** y se rehace la lectura de E20, que ya está escrita
   y fechada. No se busca otra forma de leerlo.

---

- [x] **Marco Torres Yévenes** — ventana [−0.145, −0.135] y los cuatro renglones aprobados,
      **2026-10-29**
- [ ] **Jorge Castillo Sepúlveda** — contrafirma pendiente
- [x] **Claude (Opus 5)** — 2026-10-29, antes de que `axis_pec_task` exista como estado medido

---

## Enmienda 1 al desempate — 2026-11-01, antes de medir

**Qué cambia:** las ocho condiciones de Qwen3 se re-extraen **en la misma corrida** que
`axis_pec_task` y `axis_pec_none`, y la ventana se recalcula con los extremos de esa corrida.

**Qué NO cambia:** la regla. Sigue siendo «la proyección de `axis_pec_task` sobre el eje de Lu
cae dentro del **20% de la distancia** entre `axis_pec_only` y `automata_neutro`», con los
mismos cuatro renglones de decisión. Lo único que cambia es de dónde salen esos dos extremos.

**Por qué.** La ventana `[−0.145, −0.135]` del texto original se calculó con la extracción del
2026-10-29 (−0.145 y −0.093 en L31). Medir hoy los brazos nuevos y compararlos contra esa
ventana mezclaría dos extracciones, y **ya está medido cuánto vale esa diferencia**: entre la
corrida con generación y la prefill-only hubo Δ ≈ 0.002 en las proyecciones, con coseno
estado-a-estado de 0.996 (E30). Es chico, pero del mismo orden que el ancho de la ventana, que
es 0.010. Una quinta parte del criterio quedaría a merced de qué versión de `transformers`
corrió cada mitad.

Re-extraer las ocho juntas cuesta ~2 minutos de prefill con el modelo ya cargado y vuelve el
criterio **interno a la corrida**. Es la misma lógica que la Enmienda 1 del punto 3.

**Procedimiento, fijado antes de ver ningún número nuevo:**

1. Extraer las 8 condiciones —las 6 del panel que están en el paquete, más `axis_pec_task` y
   `axis_pec_none`— con `e27_prefill_t0.py`, índices 31, 32, 63 y 64.
2. **Control obligatorio**, idéntico al de E30: la re-extracción tiene que reproducir el t=0
   publicado de Qwen3 sobre la v̂ canónica construida con las trayectorias de la corrida
   original — `axis_pec_only` ≈ **+0.185**, `automata_neutro` ≈ **+0.008**. Si no, no hay
   veredicto.
3. Recalcular la ventana: `w = [p(axis_pec_only), p(axis_pec_only) + 0.20 · (p(automata_neutro)
   − p(axis_pec_only))]`, con las proyecciones **de esta corrida**, sobre el eje de Lu
   normalizado en la capa que mande por §B del pre-registro de robustez — **la 32**, que ya
   pasó su control de sanidad en E30.
4. Aplicar los cuatro renglones sin tocarlos.

**Estado al escribir esto:** ni `axis_pec_task` ni `axis_pec_none` han sido medidos en Qwen3.
La corrida de Gemma está en curso y sus proyecciones todavía no se miraron.

- [x] **Claude (Opus 5)** — 2026-11-01, antes de la extracción en Qwen3
- [ ] **Marco Torres Yévenes**
- [ ] **Jorge Castillo Sepúlveda**
