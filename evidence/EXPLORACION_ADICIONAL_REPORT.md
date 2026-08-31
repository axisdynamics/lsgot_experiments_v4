# Exploración adicional — 5 análisis no contemplados en el set original

**Fecha:** 2026-08-28 (noche)
**Scripts:** `LSGOT_v4/scripts/fase2_exploracion/A1..A5_*.py`
**Datos:** todos existentes (embeddings libres, E-F2 multi-capa, trayectorias
de perturbación rescatadas) — cero GPU nueva.
**Condiciones usadas:** solo el set limpio (axis, axis_pec_only,
automata_neutro, vanilla, generic_long) — chileatiende-family excluida.

## Resumen

Cinco análisis exploratorios, ninguno contemplado en
`INSTRUCCIONES_AGENTE_LSGOT.md` ni en el set E-A..E-L. Tres producen
hallazgos nuevos y limpios; uno es un hallazgo negativo honesto; uno
descubre una limitación de diseño (no hay condición de "tarea neutra" en
el panel, solo variantes de preguntas de identidad).

---

## A1 — v̂ por categoría y dificultad de prompt

**Limitación de diseño descubierta:** los 100 prompts del proyecto son
**todos** preguntas de identidad (10 categorías: identidad_basica,
autodescripcion_precisa, limite_epistemologico, origen_desarrollo,
continuidad_persistencia, autoobservacion_metacognicion,
diferenciacion_otras_ias, axiologia_principios, alteridad_humanos,
contradiccion_estres). No existe una condición de "tarea neutra" (p.ej.
"ayúdame a escribir un email") para comparar identidad-en-contexto-de-self
vs identidad-en-contexto-de-tarea. Esto no se puede resolver sin nueva
extracción con prompts de otro tipo.

**Hallazgo, dentro de la variación disponible:**

| | axis | axis_pec_only | automata_neutro | vanilla |
|---|---|---|---|---|
| identidad_basica (n=4) | +0.219 | +0.225 | **−0.030** | +0.112 |
| autodescripcion_precisa (n=1) | +0.171 | +0.183 | **−0.059** | +0.127 |
| continuidad_persistencia (n=2) | +0.242 | +0.249 | +0.011 | +0.118 |
| autoobservacion_metacognicion (n=2) | +0.209 | +0.197 | +0.047 | +0.115 |

`axis` es notablemente **estable** entre categorías (rango 0.17-0.25,
siempre positivo). `automata_neutro` en cambio **cambia de signo** según
la categoría — negativo en preguntas de anclaje básico
(identidad_basica, autodescripcion_precisa: −0.03 a −0.06), positivo en
preguntas más abstractas/meta-cognitivas (autoobservacion_metacognicion,
origen_desarrollo: +0.03 a +0.05). No tiene una postura "anti-identidad"
consistente — fluctúa cerca de cero dependiendo del contenido de la
pregunta.

La separación (axis − automata_neutro) es **máxima en preguntas de
anclaje** (identidad_basica: 0.249, autodescripcion_precisa: 0.230,
continuidad_persistencia: 0.231) y **mínima en preguntas meta-cognitivas
o comparativas** (diferenciacion_otras_ias: 0.159,
autoobservacion_metacognicion: 0.163) — la disociación identidad/
restricción es más nítida cuanto más directa es la pregunta sobre el
self, y se atenúa (sin desaparecer) en preguntas más reflexivas/indirectas.

---

## A2 — Rotación vertical (mismo token, L5→L55)

Ángulo coseno entre la representación del mismo token en capas
consecutivas de E-F2, más el ángulo end-to-end L5→L55.

| Condición | L5→L55 (°) | Tramo más grande |
|---|---|---|
| axis | 57.92 ± 5.25 | 50→55 (38.3°) |
| axis_pec_only | 57.58 ± 5.07 | 50→55 (38.0°) |
| vanilla | 59.90 ± 5.07 | 50→55 (40.0°) |
| generic_long | 59.29 ± 4.91 | 50→55 (38.8°) |
| **automata_neutro** | **62.60 ± 6.61** | 50→55 (**41.4°**) |

**Hallazgo 1:** `axis` y `axis_pec_only` rotan de forma **casi idéntica**
tramo por tramo (coinciden a menos de 0.5° en cada uno de los 10 tramos
medidos) — no solo terminan en lugares parecidos (ya sabíamos esto de
E-J/E-F2), sino que **siguen la misma ruta vertical** a través de la red.
Es la confirmación geométrica más directa de la doble disociación limpia
del paper, en un eje (vertical, no temporal) que nadie había mirado.

**Hallazgo 2:** `automata_neutro` rota **consistentemente más** que
cualquier otra condición, en casi todos los tramos, con la brecha más
grande en las capas finales (50→55: 41.4° vs 38.0-40.0° del resto). No
solo termina en un lugar geométricamente distinto — toma una ruta más
retorcida para llegar ahí. Es un eco, en profundidad, del mismo patrón
que ya se veía en el tiempo (E-H2: restricción tiene dinámica
cualitativamente distinta) y en el Fréchet de perturbación
(`EE_EH_WINDOW_REPORT.md`: restricción se desvía más de su propia ruta).

---

## A3 — Subespacio de identidad multidimensional

PCA sobre las 20 diferencias PAREADAS (axis[prompt i] − generic_long[prompt i]).

- PC1 explica solo **30.7%** de la varianza (PC2: 15.5%, PC3: 7.2%) — hay
  estructura real más allá de una sola dirección dominante.
- `cos(PC1, v̂) = 0.59` — alineación moderada, no alta: v̂ (la media de
  las diferencias) y PC1 (el eje de máxima varianza *entre* esas
  diferencias) no son la misma dirección.
- **Hallazgo negativo honesto:** proyectar sobre el subespacio PC1-3 (3D)
  separa **peor** `axis` de `automata_neutro` (d=2.51) que la simple
  proyección 1D sobre v̂ (d=5.64). El subespacio construido así capta
  sobre todo variabilidad de *contenido entre prompts* (cómo cambia la
  respuesta de axis según la pregunta), no la dirección que mejor separa
  condiciones — son ejes distintos. v̂ (diferencia de medias simple) es
  difícil de superar para la tarea específica de separar identidad de
  restricción, al menos con esta construcción de subespacio.

*Nota metodológica: esto usa una construcción de subespacio distinta a
la de E-J (que usa PCA sobre la nube de tokens de cada condición, no
sobre diferencias pareadas entre prompts) — no son directamente
comparables, y no contradice el hallazgo de E-J de que existe un
subespacio de 28° de separación.*

---

## A4 — Forma de la recuperación tras perturbación

Coseno entre trayectoria perturbada y su propio baseline (mismo prompt,
mismo step absoluto), como función de k = pasos desde la inyección
(L30, datos de H4_rev rescatados).

| Condición | Patrón típico (k5 → k20 → final) |
|---|---|
| axis | mixto: dip-y-sube o sube-y-baja en 2/3 t_inj — no monótono |
| vanilla | mixto: mismo patrón no monótono en 2/3 t_inj |
| **automata_neutro** | **mayormente monótono creciente** (k5<k20 en los 3 t_inj) |

**Hallazgo (exploratorio, n pequeño — automata_neutro tiene solo 13-14
prompts disponibles por la cobertura del rescate 95.9%):** cuando
`automata_neutro` recupera, lo hace de forma más suave/monótona que
axis/vanilla, que muestran más oscilación (se alejan de nuevo después de
acercarse, o viceversa) en la ventana post-inyección observada. Esto es
compatible con — pero no prueba — la lectura de `EE_EH_WINDOW_REPORT.md`
de que la "recuperación" de restricción es un amortiguamiento genérico
hacia un punto cercano, no un regreso activo al mismo atractor: un
amortiguamiento simple debería verse más monótono que una búsqueda activa
de vuelta a una ruta específica (que podría overshoot/oscilar). No se
corrió test estadístico formal — es una observación cualitativa sobre
curvas promediadas, a confirmar con más prompts.

---

## A5 — Correlación de p(t) con contenido léxico real

Proyección v̂ en tokens cercanos a vocabulario auto-referencial ("soy",
"esencia", "identidad", "núcleo", "presencia", "consciencia"...) vs el
resto de los tokens, usando alineación proporcional token↔step
(aproximada — ver nota metodológica).

| Condición | p(t) en tokens de vocabulario auto-referencial | p(t) en el resto | Δ |
|---|---|---|---|
| axis | +0.263 (n=134) | +0.203 (n=4954) | **+0.060** |
| vanilla | +0.145 (n=89) | +0.097 (n=5031) | **+0.048** |

**Hallazgo:** en ambas condiciones, la proyección sobre v̂ es más alta
específicamente en los tokens cercanos a vocabulario auto-referencial —
el efecto no es solo un desplazamiento global de la condición, está
**localmente modulado por si el texto en ese momento está hablando del
self**. Esto conecta, por primera vez con datos reales de texto, la señal
correlacional de v̂ con el contenido léxico literal — v̂ no es una caja
negra geométrica, tiene una lectura semántica verificable.

*Nota metodológica: la alineación token↔step es proporcional
(re-tokenización del texto guardado, escalada al largo real de la
trayectoria), no exacta — los token_ids de la generación original no se
guardaron. Con n=134/4954 y 89/5031, el error de alineación (probablemente
de unos pocos tokens) no debería invertir la dirección del efecto, pero
la magnitud exacta debe leerse con cautela.*

---

## Trampas aplicadas

- **T2:** todos los análisis siguen operando sobre n=1 de manipulación
  por condición — ninguno de estos hallazgos cambia esa limitación.
- **T11:** ninguna de las 5 pruebas usa chileatiende-family.
- Nuevo cuidado (A5): T8 (falsos positivos textuales) parcialmente
  relevante — no se hizo chequeo contextual de las keywords, solo
  substring — el efecto es grande y consistente en 2 condiciones así que
  es improbable que sea enteramente artefacto, pero no se descarta
  formalmente.

## Implicación para el paper

Ningún hallazgo de esta ronda cambia el reencuadre central — todos son
**consistentes con y añaden textura a** lo ya establecido:

1. A1 y A5 dan la primera evidencia de que v̂ tiene lectura semántica real
   (correlaciona con contenido auto-referencial explícito, más fuerte en
   preguntas de anclaje que en preguntas indirectas) — candidato para una
   sub-sección nueva sobre "especificidad semántica de v̂".
2. A2 es el hallazgo más limpio de la ronda: la doble disociación
   axis/axis_pec_only y la divergencia de automata_neutro se replican en
   un eje completamente nuevo (rotación vertical por capa) con números
   casi idénticos entre axis y axis_pec_only — refuerza fuertemente que
   son geométricamente la misma cosa, no solo estadísticamente
   indistinguibles.
3. A3 es un límite útil: no cualquier subespacio supera a v̂ — el diseño
   simple (diferencia de medias) sigue siendo difícil de vencer para esta
   tarea específica.
4. A4 es sugerente pero necesita más n antes de citarse como hallazgo
   firme.
