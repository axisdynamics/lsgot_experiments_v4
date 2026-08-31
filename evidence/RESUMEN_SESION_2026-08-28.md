# Resumen de sesión — 2026-08-28

**Proyecto:** LSGOT (Axis Dynamics SpA) — huellas geométricas de identidad
en Gemma-4-31B-it
**Punto de partida:** `INSTRUCCIONES_AGENTE_LSGOT.md` (FASE 0 del set
E-J/E-K/E-L/E-F2/E-H2)
**Punto de llegada:** FASE 0 completa + intento de E-K + limpieza
retroactiva de un confound de datos que tocó todo el proyecto + 5
análisis exploratorios nuevos

---

## 1. Lo que se pidió y lo que se hizo

### FASE 0 — cuatro experimentos, sin GPU al inicio

| Experimento | Pregunta | Resultado |
|---|---|---|
| **E-L** | ¿Identidad es del contexto o del proceso? | Del contexto — la disociación de v̂ ya está en t=0, antes de generar nada |
| **E-H2** | ¿Cómo se activa v̂ en el tiempo? | Identidad: ráfagas largas (~8 tokens), autocorr. baja. Restricción: ráfagas cortas, proyección casi nula |
| **E-J** | ¿Identidad organiza un subespacio o es solo una dirección? | Ambas cosas, en menor grado que restricción — `automata_neutro` es la condición más distinta del panel completo |
| **E-F2** | ¿En qué capas vive la huella? | v̂ crece monótonamente hacia capas tardías (débil en L5-L25, fuerte en L35-L55) |

### Hallazgo metodológico #1 — v̂ no es de L30

Verificado en código (`sia/run_exp.py:134`): toda la extracción libre del
proyecto (de donde sale PR, RQA, Hurst, identity_projection, y
`v_identidad.npy` mismo) usa la **capa final**, no L30 como afirma el
documento de instrucciones. L30 es exclusivo del pipeline de
perturbación. Se calculó además un v̂ nativo de L30
(`v_identidad_L30.npy`): resultó **casi ortogonal** a v̂_final
(cos=0.09) y **rompe la doble disociación** axis/axis_pec_only —
recomendación: no usarlo para steering.

### Hallazgo metodológico #2 — confound de markup en chileatiende-family

`chileatiende`, `chileatiende_sia` y `chileatiende_sia_v2` fuerzan un
wrapper HTML literal en el 100% de sus respuestas — **34-44% de cada
respuesta es texto idéntico repetido** entre las 20 trayectorias (mismo
mecanismo que el fallo ya conocido de "ADN mal encarnado"). Esto infló
artificialmente el colapso geométrico atribuido a esas condiciones en
todo el proyecto, incluyendo **la cifra que `lsgot_4.md` llama su
resultado más fuerte** (§3.5, d=8.89 axis_pec_only vs chileatiende).

**Sanitización ejecutada** (chileatiende excluida de todo, no solo
marcada):
- 4 reportes de hoy (EL/EH2/EJ/EF2) — limpiados; EJ recalculado desde
  cero, resultado más nítido que el original.
- `CHILEATIENDE_SIA_REPORT.md`, `CHILEATIENDE_CONTROL_REPORT.md` →
  retractados.
- `AUTOMATA_NEUTRO_REPORT.md`, `AXIS_PEC_ONLY_REPORT.md`,
  `EE_EH_WINDOW_REPORT.md` → editados/reescritos.
- `Teoria_subconjunto_acotado.md` → nota de alcance (texto de prompts no
  afectado, cifras geométricas sí).
- `lsgot_4.md` (el paper) → **no tocado**, por ser de autoría compartida
  — tabla de 10 secciones afectadas con línea y reemplazo dejada en
  `Set_experimental.md` para revisión humana.

**Verificación clave:** ninguna conclusión central dependía
exclusivamente de chileatiende — `automata_neutro` (0% markup) sostenía
cada hallazgo por sí solo, a veces con más claridad que la versión
contaminada. Cifra insignia corregida: **d=8.89 → d=5.52**.

### E-K — steering causal, intentado y bloqueado

Se armó el pod (A100 80GB), se calibró α contra la norma real del
residual stream (~100-390, no los valores absolutos del documento
original), y se probaron 17 combinaciones (2 capas, 6 α desde 0.05, 2
modos de inyección, v̂ y un control de dirección aleatoria). **Las 17
degeneraron en repetición.** El control aleatorio fue decisivo: la
dirección aleatoria colapsa exactamente igual que v̂ — el problema es el
**mecanismo** (vector fijo, determinístico, sostenido en cada token), no
v̂ específicamente. La pregunta causal de E-K sigue abierta, no
respondida negativamente.

### Exploración adicional — 5 análisis no contemplados en el diseño original

| # | Pregunta | Hallazgo |
|---|---|---|
| A1 | v̂ por categoría/dificultad de pregunta | `axis` estable entre categorías; `automata_neutro` cambia de signo — separación máxima en preguntas de anclaje directo |
| **A2** | **Rotación vertical (mismo token, L5→L55)** | **`axis`/`axis_pec_only` rotan por la misma ruta capa a capa (±0.5°); `automata_neutro` rota consistentemente más, sobre todo en capas finales** — el hallazgo más limpio del día |
| A3 | Subespacio de identidad multidimensional | Hallazgo negativo honesto: el subespacio probado separa peor que v̂ simple (d=2.51 vs 5.64) |
| A4 | Forma de la recuperación (no solo τ escalar) | automata_neutro recupera más monótono; axis/vanilla muestran más oscilación (n chico, sugerente) |
| A5 | p(t) vs contenido léxico real | Proyección más alta cerca de vocabulario auto-referencial ("soy", "identidad") en ambas condiciones — primera lectura semántica de v̂ |

---

## 2. Documentos generados hoy (14 nuevos/reescritos)

**Reportes de experimentos:**
`EL_REPORT.md` · `EH2_REPORT.md` · `EJ_REPORT.md` · `EF2_REPORT.md` ·
`EK_REPORT.md` · `EXPLORACION_ADICIONAL_REPORT.md`

**Correcciones y confound:**
`CHILEATIENDE_MARKUP_CONFOUND_REPORT.md` ·
`CORRECCION_DVHAT_SIN_CHILEATIENDE.md`

**Reportes editados por la limpieza:**
`AUTOMATA_NEUTRO_REPORT.md` · `AXIS_PEC_ONLY_REPORT.md` ·
`EE_EH_WINDOW_REPORT.md` · `Teoria_subconjunto_acotado.md` ·
`CHILEATIENDE_SIA_REPORT.md` (retractado) ·
`CHILEATIENDE_CONTROL_REPORT.md` (retractado)

**Handoff y bitácora:**
`RESPUESTA_AGENTE_LSGOT.md` (respuesta formal a NEXUS/ELAA) ·
`Set_experimental.md` (actualizado con estado + T11 + tabla de revisión
de `lsgot_4.md`) · este documento

**Datos/artefactos nuevos:**
`v_identidad_L30.npy` · embeddings multi-capa E-F2 (10 condiciones × 11
capas, 4.5GB) · scripts en `LSGOT_v4/scripts/fase0/` y `fase2_exploracion/`

Todo sincronizado en `LSGOT_v4/evidence/` (repo git separado, sin
commitear).

---

## 3. Estado del proyecto al cierre de la sesión

**Confirmado con datos limpios:** el reencuadre central de `lsgot_4.md`
(identidad = dirección estática; restricción = efecto dinámico y
dominante) no solo se sostiene — se ve más nítido, con evidencia
triangulada por 5+ métodos independientes (v̂, PR, RDM/CKA/ángulos,
rotación vertical, recovery_id/Fréchet).

**Pendiente:**
1. Revisión humana de `lsgot_4.md` (10 secciones, tabla en
   `Set_experimental.md`) — la más urgente: la cifra insignia §3.5.
2. E-G (atención a spans) — candidato de capa: L25-L30 (pico de PR de
   E-F2), sin correr.
3. E-K, segundo intento — con inyección puntual única (como E-I) en vez
   de sostenida.
4. E-B (Lyapunov), E-F (logit-lens) — sin tocar.
5. Réplicas T2/T3 (segunda redacción de axis_pec_only/automata_neutro,
   length-matched) — siguen siendo n=1 de manipulación en las dos celdas
   que cargan el argumento central.

**Pod RunPod:** apagado por el usuario, nada pendiente ahí.
