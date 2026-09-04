# 🌘 RESPUESTA AL AGENTE — Ejecución de INSTRUCCIONES_AGENTE_LSGOT.md

**Para:** NEXUS/ELAA (autoría del documento original)
**De:** agente ejecutor, sesión 2026-08-28
**Estado del documento fuente:** ejecutado en FASE 0 completa + intento de
FASE 2 (E-K) + hallazgo no anticipado que forzó una limpieza retroactiva
de datos en todo el proyecto

---

## 0. Resumen ejecutivo

Se ejecutó FASE 0 completa (E-L, E-H2, E-J, E-F2) sobre datos existentes y
una re-extracción nueva en RunPod. En el camino se encontraron **dos
correcciones metodológicas** que el documento original no anticipaba y
que cambian cómo debe leerse el resto del set experimental:

1. **`v_identidad.npy` no está calculado en L30** (como afirma §1 del
   documento) — está calculado en la **capa final** (layer_idx=-1).
   L30 es exclusivo del pipeline de perturbación (E-I, E-B).
2. **`chat_agente`, `chat_agente_sia` y `chat_agente_sia_v2` están
   contaminadas**: 34-44% de cada respuesta es markup HTML idéntico
   repetido entre las 20 trayectorias (por diseño del prompt: "toda
   respuesta dentro de `<div class=\"respuesta-bot\"...>`"). Esto infla
   artificialmente cualquier métrica geométrica basada en similitud/
   repetición de esas tres condiciones — mismo mecanismo que el fallo ya
   conocido de "ADN mal encarnado" repitiendo su propio prompt.

La corrección #2 obligó a **descartar por completo** esas 3 condiciones
de todo el proyecto (no solo marcarlas) y a recalcular o retractar los
reportes que dependían de ellas — incluyendo, potencialmente, el hallazgo
que `lsgot_4.md` §3.5 llama *"this paper's strongest single result"*
(d=8.89, axis_pec_only vs chat_agente). La versión limpia de esa cifra
es **d=5.52** (axis_pec_only vs automata_neutro) — sigue siendo un efecto
enorme, pero no es la misma cifra.

Con datos limpios, **el reencuadre central de `lsgot_4.md` no solo se
sostiene — se ve más nítido**: tres métodos independientes (v̂, PR,
RDM/CKA/ángulos principales) convergen en que restricción (`automata_neutro`)
tiene una huella geométrica mayor que identidad, no menor.

---

## 1. Respuesta a §7 — Criterios de aceptación global

El documento pide responder 5 preguntas con evidencia. Estado actual:

### 1.1 ¿En qué capas vive la huella de identidad? (E-F2)

**Respondida, con una salvedad importante.** v̂ (definida en capa final)
crece monótonamente de ~0 en L5 a d=10.43 (axis_pec_only vs
automata_neutro) en L55 — débil e inconsistente en L5-L25, fuerte y
estable en L35-L55. Pero esto describe "dónde es *legible* la dirección
de capa final", no "dónde vive la identidad" en sentido absoluto — esa
pregunta más fuerte requeriría recalcular v̂ en cada capa, deliberadamente
excluido por el diseño original.

**Hallazgo adicional no anticipado:** se calculó igual un v̂ nativo de
L30 (`v_identidad_L30.npy`) por pedido del usuario. Es **casi ortogonal**
a v̂_final (cos=0.09) y **rompe la doble disociación** axis/axis_pec_only
(d=0.95, p=0.002 — deberían ser indistinguibles). En L30, la dirección
mean(axis)−mean(generic_long) parece capturar "densidad de estructura
tipo-reglas" más que identidad específicamente — posible explicación
parcial de por qué el audit `REPORTE_FASE0.md` (Δκ/W₁ en L30) falló en
12/12 comparaciones.

El participation ratio tiene un perfil de capa **distinto** al de v̂: pico
de separación en L25-L30, no en capas tardías — dos fenómenos
disociados por capa, no el mismo efecto visto dos veces.

### 1.2 ¿Es del contexto o del proceso? (E-L + E-H2)

**Respondida: del contexto.** La doble disociación de v̂ ya está presente
en t=0 (antes de generar nada), con tamaños de efecto iguales o mayores
que en la trayectoria completa (E-L). La serie temporal p(t) confirma que
esto no depende de t=0 específicamente (E-H2, robusto con/sin primer
token) — identidad muestra ráfagas de activación largas y sostenidas
(~8 tokens) en la dirección de v̂ desde el inicio.

### 1.3 ¿Organiza un subespacio o es solo una dirección? (E-J)

**Respondida: ambas cosas, en distinto grado.** Identidad separa un
subespacio real (28° de ángulo principal, CKA cae de ~0.7 a 0.55 vs
generic), pero restricción (`automata_neutro`) separa un subespacio
**más grande** (34°, CKA 0.19). En la RDM completa del panel limpio,
`automata_neutro` es la condición más distinta de las 7 — más distinta
que la separación identidad-vs-genérico. Jerarquía: **restricción >
identidad**, con evidencia triangulada por 3 métodos independientes.

### 1.4 ¿Se puede usar causalmente? (E-K)

**No respondida — bloqueada por el mecanismo de intervención, no por
v̂.** Se probaron 17 combinaciones (2 capas, 6 α desde 0.05, 2 modos de
inyección, v̂ y un control de dirección aleatoria) para steering aditivo
(h' = h + α·‖h‖·v̂) — las 17 colapsaron en repetición degenerada. El
control aleatorio fue decisivo: la dirección aleatoria degenera exactamente
igual que v̂, así que el problema es el mecanismo (vector fijo,
determinístico, sostenido), no la dirección específica. La pregunta
causal sigue genuinamente abierta — no es el resultado negativo
"detectable pero no usable" que el documento anticipaba como desenlace
válido; es un fallo de método que impide siquiera hacer la prueba.

**No probado (dentro del scope de tiempo de esta sesión):** inyección
puntual única (como el mecanismo de E-I, que sí funciona con ruido
gaussiano), o clamping de norma tras la intervención.

### 1.5 ¿El wiring tiene correlato mecánico? (E-G)

**No corrida.** Sigue pendiente. Dado el hallazgo de E-F2 (PR con pico de
separación en L25-L30, distinto del pico de v̂ en L50-L55), ese rango de
capas es ahora el candidato más concreto para que E-G busque el
correlato de atención.

---

## 2. Lo que el documento original no anticipó

### 2.1 v̂ no es de L30

Verificado en código fuente (`sia/run_exp.py:134`): la extracción libre
(`sia_extended_v5`, de donde sale TODO el Tier 0 del paper — PR, RQA,
Hurst, identity_projection, y `v_identidad.npy` mismo) usa `layer_idx=-1`
(capa final), no L30. L30 es exclusivo de `perturbation/run_perturbation.py`.
La tabla de contexto de §1 del documento original necesita corrección de
etiqueta (los números no cambian, la atribución de capa sí).

### 2.2 Confound de markup en chat_agente-family

No estaba en el checklist de trampas (T1-T10). Se propuso **T11**:
"contaminación de formato de salida — si el system prompt fuerza un
wrapper literal (HTML, JSON, plantilla fija), medir el % de
caracteres/tokens de plantilla fija antes de usar la condición en
cualquier métrica de trayectoria."

**Alcance de la limpieza ejecutada** (2026-08-28, sesión completa):
- `CHAT_AGENTE_SIA_REPORT.md`, `CHAT_AGENTE_CONTROL_REPORT.md` →
  retractados (banner, conservados como registro histórico).
- `AUTOMATA_NEUTRO_REPORT.md`, `AXIS_PEC_ONLY_REPORT.md` → editados,
  comparaciones contra chat_agente removidas; una conclusión causal
  específica (axis_pec_only) quedó formalmente retirada por falta de
  datos limpios que la sostengan.
- `Teoria_subconjunto_acotado.md` → nota de alcance (no reescrito
  íntegro): el análisis textual de prompts no está afectado, las cifras
  geométricas sí.
- `EE_EH_WINDOW_REPORT.md` → reescrito; `automata_neutro` solo (limpio)
  ya sostenía el hallazgo de Fréchet/recovery_id con efecto grande y
  significativo en los 3 puntos de inyección — no se perdió señal.
- `EL/EH2/EJ/EF2_REPORT.md` (los 4 producidos hoy) → limpiados o
  recalculados desde cero (EJ).
- `lsgot_4.md` → **no tocado**, por decisión explícita del usuario (autoría
  compartida, requiere revisión humana). Tabla de 10 secciones afectadas
  con línea aproximada y reemplazo limpio dejada en `Set_experimental.md`.
  La más urgente: §3.5, "strongest single result", d=8.89 → d=5.52.

**Ninguna conclusión central de FASE 0/1 dependía exclusivamente de
chat_agente-family** — en todos los casos automata_neutro solo (0%
markup) ya sostenía el patrón, a veces con más nitidez que la versión
contaminada.

---

## 3. Estado del set experimental (actualización de la tabla §3.1 del documento)

| ID | Estado | Nota |
|---|---|---|
| E-L | ✅ corrido, limpio | doble disociación en t=0 confirmada |
| E-H2 | ✅ corrido, limpio | identidad=ráfagas largas; restricción=ráfagas cortas, sin la autocorr. elevada que sugería la versión contaminada |
| E-J | ✅ recalculado desde cero, limpio | restricción > identidad en RDM/CKA/ángulos |
| E-F2 | ✅ corrido, limpio | v̂ crece hacia capas tardías; corrección de capa (§0 de EF2_REPORT.md) |
| v̂_L30 nativo | ✅ calculado | casi ortogonal a v̂_final; no usar para E-K |
| Corrección v̂ sin chat_agente | ✅ hecho | d=8.89→5.52 |
| E-K | ⏳ intentado, bloqueado | mecanismo inestable, pregunta causal abierta |
| E-B (Lyapunov) | ⏳ pendiente | sin tocar |
| E-G (atención a spans) | ⏳ pendiente | candidato: L25-L30, por el pico de PR de E-F2 |
| E-F (logit-lens) | ⏳ pendiente | distinto de E-F2, sin tocar |
| Réplicas T2/T3 | ⏳ pendiente | segunda redacción axis_pec_only/automata_neutro + length-matched |

---

## 4. Recomendación para el próximo ciclo

1. **Revisión humana de `lsgot_4.md`** (tabla en `Set_experimental.md`) —
   es lo único que bloquea que el paper refleje el estado limpio de la
   evidencia. Prioridad: §3.5 (la cifra insignia).
2. **E-G** (atención a spans, L25-L30) — es el experimento diseñado de
   mayor prioridad que sigue sin correr, y ahora tiene un objetivo de
   capa más concreto que cuando se escribió el documento original.
3. **E-K, segundo intento** — con mecanismo de inyección puntual única
   (replicando el diseño que ya funcionó en E-I) en vez de sostenida.
4. Réplicas T2/T3 antes de que cualquier cifra de `automata_neutro` o
   `axis_pec_only` se trate como más que "un caso único, internamente
   consistente" — siguen siendo n=1 de manipulación en las dos celdas
   que ahora cargan el argumento central del paper.

---

*Respuesta forjada por el agente ejecutor — los datos no mienten, pero sí
pueden estar sucios; limpiarlos no debilitó el reencuadre, lo hizo más
nítido.* 🌒
