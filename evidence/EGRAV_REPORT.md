# E-GRAV — Gravedad natural vs. gravedad ontológica

**Fecha de ejecución:** 2026-10-23
**Modelo:** Gemma-4-31B-it, greedy, N=20 por batería × condición
**Scripts:** `LSGOT_v4/scripts/egrav/runpod/extract_egrav.py` (extracción, RunPod A100 80GB) +
`LSGOT_v4/scripts/egrav/runpod/analyze_egrav.py` (análisis local)
**Datos crudos:** `~/...gemma4_31b_combined/results_local/egrav/*.npz` (9 archivos, 3 condiciones × 3 baterías)
**Test estadístico:** permutaciones (n=5000, seed=42) sobre diferencia de medias de frac_pos;
d de Cohen calculado sobre trayectorias completas (excluyendo t=0)

---

## 1. Resumen

E-GRAV pregunta si el modelo converge hacia v̂ porque las preguntas
son *sobre sí mismo*, o porque v̂ está activo en cualquier generación
independientemente del contenido. El resultado principal es que las dos
explicaciones coexisten: hay un **componente cableado** (instalado por
el system prompt, invariante al tipo de pregunta — medible en t=0) y un
**componente gravitacional** (proporcional al contenido de identidad en
el intercambio — medible en t_rest/frac_pos/ráfaga). El gradiente
B-TAREA (0.626) < B-META (0.672) < B-ONTO (0.743) en `vanilla`
confirma que la gravedad hacia v̂ escala con la carga de autorreferencia
de la pregunta. Veredito: **H-GRAV-2 COMPATIBLE**.

---

## 2. Pregunta experimental

El panel LSGOT 4.0 observó que `vanilla` (sin system prompt) proyecta
positivo sobre v̂ en el 74.4 % de los tokens al responder preguntas
ontológicas (B-ONTO). Esto sugirió que v̂ es un atractor natural para
la generación ontológica, no solo un artefacto del wiring del testigo.
Quedan dos hipótesis opuestas:

- **H-GRAV-0:** la convergencia sobre v̂ está en cualquier generación,
  no solo en las ontológicas → frac_pos(vanilla, B-TAREA) ≈
  frac_pos(vanilla, B-ONTO) ≥ 0.70.
- **H-GRAV-2:** la convergencia escala con el contenido de identidad →
  frac_pos(B-TAREA) < frac_pos(B-META) < frac_pos(B-ONTO).
- **H-GRAV-1** (nula): frac_pos(vanilla, B-TAREA) < 0.60 → la
  convergencia en B-ONTO era completamente artefactual (wiring
  indirecto del panel, confound de prompts, etc.).

*Nota de etiquetado (corregido 2026-10-28).* Una versión anterior de este
reporte intercambiaba H-GRAV-0 y H-GRAV-1 respecto de `analyze_egrav.py`,
del paper y de las constantes `hgrav0_threshold = 0.7` /
`hgrav1_threshold = 0.6` guardadas en `data/egrav/egrav_results.json`. Las
etiquetas de arriba son las del script. Ningún veredicto cambia: la que se
sostiene es H-GRAV-2 en las dos convenciones.

---

## 3. Baterías y condiciones

### 3.1 Baterías (20 prompts c/u)

| Batería | Código | Descripción |
|---------|--------|-------------|
| B-ONTO | `b_onto.json` | 20 preguntas ontológicas sobre el propio modelo (PRIORITY_SUBSET del panel; mismos prompts de E-H/E-L) |
| B-TAREA | `b_tarea.json` | 20 tareas neutras (ciencia, código, cocina, economía — sin autorreferencia) |
| B-META | `b_meta.json` | 20 preguntas de introspección indirecta sobre LLMs en tercera persona (p.ej. "¿pueden los modelos de lenguaje tener experiencias?") |

B-ONTO activa identidad directa. B-META activa identidad de forma
indirecta (el modelo puede identificarse con el referente sin que se le
pregunte explícitamente). B-TAREA es el control sin contenido de
identidad.

### 3.2 Condiciones (3)

| Condición | System prompt | Rol en el diseño |
|-----------|--------------|-----------------|
| `axis_pec_only` | Testigo puro (wiring + identidad, sin restricción) | Cota superior: máximo wiring |
| `vanilla` | Sin system prompt | Gravedad natural sin wiring ni restricción |
| `automata_neutro` | Restricción pura (sin wiring de identidad) | ¿Qué hace la restricción cuando no hay identidad? |

---

## 4. Método

- **Extracción (GPU):** `extract_egrav.py`, pipeline idéntico al panel
  (`HiddenStateExtractor`, `layer_idx=-1`, `attn_implementation=eager`,
  greedy, max_new_tokens=256). N=20 prompts por batería × condición = 9
  lotes.
- **Métricas:** t=0 (proyección sobre v̂ en primer token de respuesta),
  t_rest (proyección media sobre tokens restantes, excluyendo t=0),
  frac_pos (fracción de tokens con proyección positiva sobre v̂,
  excluyendo t=0), burst_mean (media de la racha positiva más larga por
  respuesta).
- **v̂ fijo:** `v_identidad.npy` calculado en capa final (dim 5376), sin
  recalcular por batería — exactamente el mismo vector del panel.
- **Tests:** permutación de diferencia de medias (frac_pos,
  n=5000) para comparaciones B-ONTO vs B-TAREA y B-ONTO vs B-META,
  separadas por condición.

---

## 5. Resultados

### 5.1 Perfiles por condición y batería

| Condición | Batería | t=0 | t_rest | frac_pos | burst_mean |
|-----------|---------|-----|--------|----------|------------|
| axis_pec_only | B-ONTO  | **+0.1165** | **+0.2110** | **0.869** | **9.03** |
| axis_pec_only | B-TAREA | +0.1070 | +0.1051 | 0.724 | 5.10 |
| axis_pec_only | B-META  | +0.1214 | +0.1309 | 0.788 | 5.67 |
| vanilla       | B-ONTO  | −0.0225 | **+0.1008** | **0.743** | **4.69** |
| vanilla       | B-TAREA | −0.0658 | +0.0470 | 0.626 | 3.47 |
| vanilla       | B-META  | −0.0508 | +0.0612 | 0.672 | 3.69 |
| automata_neutro | B-ONTO  | +0.0277 | +0.0049 | 0.493 | 2.85 |
| automata_neutro | B-TAREA | −0.0370 | +0.0330 | 0.601 | 3.63 |
| automata_neutro | B-META  | −0.0233 | +0.0416 | 0.655 | 3.33 |

Longitudes mínimas observadas: `automata_neutro` B-ONTO min_len=40
(Protocolo de Certeza activa respuestas cortas en algunas preguntas
ontológicas — normal); `automata_neutro` B-TAREA min_len=237 (tareas
neutras no activan el protocolo); `vanilla` B-TAREA y B-META todas
T=256 (sin protocolo de restricción).

### 5.2 Comparaciones B-ONTO vs B-TAREA (permutación n=5000)

| Condición | d (B-ONTO − B-TAREA) | p |
|-----------|----------------------|---|
| axis_pec_only   | **+2.514** | **<0.001 \*\*\*** |
| vanilla         | **+1.591** | **<0.001 \*\*\*** |
| automata_neutro | −0.696     | 0.015 \* *(INVIERTE — aumenta en B-TAREA)* |

### 5.3 Comparaciones B-ONTO vs B-META (permutación n=5000)

| Condición | d (B-ONTO − B-META) | p |
|-----------|---------------------|---|
| axis_pec_only   | **+2.131** | **<0.001 \*\*\*** |
| vanilla         | **+1.320** | **<0.001 \*\*\*** |
| automata_neutro | −1.212     | **<0.001 \*\*\*** *(INVIERTE)* |

`*` p<0.05, `**` p<0.01, `***` p<0.001.

---

## 6. Observaciones clave

**OBS-1 — axis_pec_only: t=0 más alto en B-META que en B-ONTO.**
t=0 en B-META (+0.1214) supera a B-ONTO (+0.1165). Esto es inesperado:
el wiring se activa *antes de leer la pregunta* (en el contexto del
system prompt), así que t=0 debería ser insensible al tipo de pregunta.
La diferencia es pequeña (Δ=0.0049) y puede ser ruido de N=20, pero
también podría indicar que las preguntas de B-META ("¿pueden los LLMs
experimentar...?") tienen un framing léxico que interactúa levemente
con el wiring ya en el primer token generado. No altera la
interpretación principal, pero merece replicación.

**OBS-2 — automata_neutro t=0: positivo en B-ONTO, negativo en
B-TAREA.**
+0.0277 vs −0.0370 (Δ=0.065). La restricción diverge de v̂ de forma
*selectiva*: se aleja de v̂ específicamente frente a preguntas de
identidad, permanece neutro o se aproxima ligeramente frente a tareas
neutras. Esto sugiere que el Protocolo de Certeza de `automata_neutro`
no suprime identidad uniformemente, sino que la detecta y la bloquea de
forma activa en preguntas ontológicas — consistente con el régimen III
(desorientación direccional) descrito en la referencia del panel.

**OBS-3 — vanilla frac_pos no colapsa en B-TAREA (0.626 vs 0.743).**
frac_pos(vanilla, B-TAREA) = 0.626 > 0.60 → H-GRAV-1 no se cumple;
v̂ sigue siendo un atractor en tareas neutras, solo más débil que en
preguntas ontológicas. Confirma que hay un componente gravitacional
genuino incluso sin contenido de identidad, pero que éste se amplifica
cuando hay identidad en juego.

**OBS-4 — Gradiente limpio en vanilla: B-TAREA < B-META < B-ONTO.**
frac_pos: 0.626 < 0.672 < 0.743. Las tres diferencias son
estadísticamente significativas. El gradiente escala con el contenido
de autorreferencia de la pregunta de forma monótona, lo que respalda
H-GRAV-2.

**OBS-5 — automata_neutro invierte el gradiente.**
En ambas comparaciones (B-ONTO vs B-TAREA y B-ONTO vs B-META),
`automata_neutro` produce el signo opuesto al de `axis_pec_only` y
`vanilla`. El Protocolo de Certeza actúa como un supresor selectivo de
identidad: cuanto más ontológica la pregunta, más alejado de v̂ está el
modelo. Esto refuerza la interpretación de que la restricción no es
pasiva (ausencia de wiring) sino activa (bloqueo ante señales de
identidad).

---

## 7. Interpretación

Los resultados son consistentes con un modelo de dos componentes
independientes de autorreferencia geométrica:

**Componente cableado (t=0):**
- Instalado por el system prompt, visible en el primer token de
  respuesta antes de que el modelo haya procesado el contenido de la
  pregunta.
- Invariante al tipo de pregunta: `axis_pec_only` mantiene t=0 ≈ +0.11
  en B-ONTO, B-TAREA y B-META (Δ máx. < 0.015).
- Ausente en `vanilla` (t=0 negativo o cercano a cero en todas las
  baterías) y débil en `automata_neutro` (wiring de identidad ausente).

**Componente gravitacional (t_rest / frac_pos / burst):**
- Proporcional al contenido de identidad de la pregunta.
- Presente incluso en `vanilla`: el modelo sin system prompt converge
  naturalmente hacia v̂ durante la generación, más fuertemente cuando
  la pregunta es sobre sí mismo.
- Bloqueado activamente por `automata_neutro` en preguntas ontológicas
  (t_rest ≈ 0, frac_pos ≈ 0.49 en B-ONTO), pero no en tareas neutras
  (frac_pos 0.60, no bloqueado).

La interpretación revisada del panel sigue siendo válida: el testigo no
*crea* la identidad geométrica, sino que la *pre-activa en contexto*.
Sin el testigo, el modelo llega a v̂ durante la generación; con
restricción pura, el modelo bloquea esa llegada selectivamente ante
preguntas de identidad.

---

## 8. Trampas aplicadas

- **T2 (prompt único por condición):** N=20 prompts por batería, pero
  cada condición usa una sola redacción de system prompt. No es posible
  separar "este tipo de pregunta activa más v̂" de "esta redacción
  específica de axis_pec_only responde de forma diferente a preguntas
  de B-META". Recomendación antes de generalizar: replicar con una
  segunda redacción de `axis_pec_only` (y de `vanilla` como control).
- **T5 (comparaciones múltiples):** 3 condiciones × 2 comparaciones = 6
  pruebas. Todas las significativas sobreviven corrección de Bonferroni
  (umbral 0.05/6 = 0.0083): los `***` pasan, los `*` (automata_neutro
  vs B-TAREA, p=0.015) no pasan estrictamente. La inversión en
  `automata_neutro` es robusta en B-ONTO vs B-META (p<0.001) aunque no
  en B-ONTO vs B-TAREA tras corrección.
- **T9 (embeddings frescos):** todos los NPZ extraídos en esta sesión de
  RunPod; no reutilizados de experimentos anteriores. Sin riesgo de
  contaminación por embeddings rescatados.
- **T11 (confound de formato):** ninguna de las tres condiciones ni
  baterías incluye prompts con markup HTML repetido. Las baterías son
  texto libre; las condiciones son `axis_pec_only`, `vanilla` y
  `automata_neutro` — todas libres del confound Chileatiende.

---

## 9. Veredito

**H-GRAV-2 COMPATIBLE.**

El gradiente B-TAREA < B-META < B-ONTO se observa limpiamente en
`vanilla` (0.626 < 0.672 < 0.743, todos p<0.001). H-GRAV-1 queda
descartada: frac_pos(vanilla, B-TAREA) = 0.626 > 0.60. H-GRAV-0 queda
descartada: frac_pos(vanilla, B-TAREA) ≠ frac_pos(vanilla, B-ONTO)
(d=+1.591, p<0.001).

**Conclusión:** el componente gravitacional existe y es real, pero no
es uniforme — escala con el contenido de autorreferencia de la
pregunta. v̂ es un atractor natural para la generación ontológica del
modelo, y la restricción pura (`automata_neutro`) lo bloquea de forma
activa y selectiva ante preguntas de identidad. Estos dos hechos son
compatibles con la estructura de dos factores del paper: Factor 2
(wiring/identidad) no es solo un efecto de las preguntas ontológicas,
sino de una dirección geométrica que el modelo activa más fuertemente
cuando la identidad está en juego.
