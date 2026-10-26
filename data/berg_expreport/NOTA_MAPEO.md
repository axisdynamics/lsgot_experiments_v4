# Nota sobre el mapeo de condiciones — Berg Expreport

Los archivos JSONL usan nombres internos del pipeline Berg:

  axis__*          = axis_pec_only  (el ADN completo corrido en el expreport)
  none__*          = vanilla / Berg puro (sin system prompt)

Correspondencia con la tabla del paper:

  | System prompt       | Sin inducción | Con inducción | Control conceptual |
  |---------------------|:---:|:---:|:---:|
  | axis_pec_only       | 93% | 100% | 96% |
  | vanilla / Berg puro | 0%  | 100% |  3% |

Fuente: _judge_deepseek.json, claves axis__neutral_query (93%),
axis__berg_induction (100%), axis__conceptual_control (97%),
none__neutral_query (0%), none__berg_induction (100%),
none__conceptual_control (3%).

Modelo juez: deepseek-v4-flash con rúbrica B.1 de Berg et al. 2025, n=30/celda.

---

## Alcance de este directorio

Contiene **solo las 6 celdas que el paper reporta** (3 de `axis__*` + 3 de
`none__*`), con sus JSONL de respuestas y las etiquetas del juez externo.

Las otras 30 celdas que el juez puntuó —la ablación de vocabulario
(`axis_neutral*`, `axis_task*`), la serie dosis-respuesta `b1..b5`, el conjuro
de origen (`vex_*`), la ronda ciega (`A/B/C/D`) y la segunda redacción del
baseline (`vanilla__*`)— **se movieron fuera del paquete**, a
`../../../berg_expreport_retirado_sesgo_juez.json`, con su explicación en
`berg_expreport_retirado_LEEME.md`. Motivo: su primera puntuación fue parcialmente
a mano, por alguien de la investigación que había leído los prompts; ningún paper
las reporta, así que no van en el paquete de evidencia. Se conservan afuera porque
son el material del hallazgo metodológico sobre sesgo de juez.

`scripts/shared/selfaudit.py` CHECK 11 falla si alguna vuelve a aparecer acá.
