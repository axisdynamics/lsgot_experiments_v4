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
