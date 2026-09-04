# Qué sobrevive y qué no — estado de la evidencia (2026-08-31)

Ledger de todo lo que se afirmó, se corrigió o se retiró durante el saneamiento
de este panel. Cada fila indica el estado actual y por qué. No reemplaza a
`paper/lsgot_4.md`; es el mapa de auditoría que hay detrás de él.

## 1. Sobrevive, robusto — no se movió con ningún control

| Hallazgo | Evidencia | Qué lo puso a prueba |
|---|---|---|
| **v̂ — doble disociación estática** (identidad positiva, restricción negativa) | d=+5.52 (`axis_pec_only` vs `automata_neutro`, §3.5) | Sobrevivió: exclusión de chat_agente (d bajó de 8.89 a 5.52, no desapareció), replicación en Qwen3 (más fuerte, d=+12.97), auto-auditoría de curvatura (baselines confirman separación de v1), confound de primer token a nivel proxy (se refuerza, no revierte), **T2 — segunda instanciación independiente de las dos celdas puras (d=+5.25, prácticamente idéntico)** |
| **Recuperación gatillada por cableado, T2** | `automata_neutro_v2` recovery_rate=0.50-0.60 (más severo que el original 0.77-0.88) | Réplica de manipulación con redacción independiente — el fallo de recuperación no solo sobrevive, se acentúa |
| **v̂ en t=0** (antes de generar) | d=+5.75 a +9.78 (E-L) | Sobrevive en Qwen3, más fuerte (d=+12.97 en la comparación análoga) |
| **v̂ — dinámica temporal** (ráfagas sostenidas) | ráfaga 7.9-8.8 vs 2.8 tokens (E-H2) | Sobrevive en Qwen3, más extremo |
| **Participation ratio, efecto graduado** | id. d=−1.18, restr. d=−1.79 (§3.1) | Sobrevive en Qwen3: `automata_neutro` el PR más bajo en 12/12 capas, sin excepción |
| **RQA determinismo, dicotomía pura de restricción** | 0 en todas salvo `automata_neutro` (d=+1.35) | No se corrió en Qwen3 todavía — sin refutar, sin confirmar cross-modelo |
| **Perturbación direccional — no hay atractor literal** | 6/6 comparaciones p>0.14 (§3.7, E-I) | Es en sí misma un control negativo; nada la puso en duda desde que se publicó |

## 2. Sobrevive con matices — la magnitud o el mecanismo cambia, no la dirección

| Hallazgo | Qué sobrevive | Qué NO generaliza |
|---|---|---|
| **Recuperación gatillada por cableado** (τ, recovery_id, Fréchet) | El patrón cualitativo completo en Gemma — reforzado esta ronda con la Figura 1 (meseta de realineación) | Nunca se corrió en Qwen3 — la mitad dinámica del argumento sigue siendo de un solo modelo |
| **Subespacio identidad↔restricción** (E-J) | La separación identidad-vs-restricción es la mayor del panel en ángulo, en **ambos** modelos (34.1° Gemma, 40.7° Qwen3) | La jerarquía — quién es "el más distinto" — se invierte: `automata_neutro` en Gemma, identidad en Qwen3 |
| **Δκ/W₁, Forman-Ricci** (lo que el proyecto siempre calculó) | Señal real para restricción (×3.5-17 sobre ruido split-half); sigue sin superar baselines simples (4/4) | Ciega a la identidad pura (`axis_pec_only` vs `vanilla` cae en ruido, 0.4-0.5× mediana) |
| **Δκ/W₁, Ollivier-Ricci genuina** (lo que el paper siempre dijo calcular, corrida por primera vez esta ronda) | **No** es ciega a la identidad — `axis_pec_only` vs `vanilla` sale del ruido (pct=0.3%/0.0%); restricción sigue dominando en magnitud (curvatura media −0.68 vs −0.20) | Los baselines simples le siguen ganando la carrera igual (probe AUC=1.000 en las 4, no depende de qué curvatura se compare) |

## 3. Corregido, no retirado — el hallazgo sigue en pie con otro número o nombre

| Qué cambió | De → A | Por qué |
|---|---|---|
| Cifra insignia del paper (§3.5) | d=+8.89 (`axis` vs `chat_agente`) → **d=+5.52** (`axis_pec_only` vs `automata_neutro`) | `chat_agente` forzaba markup HTML repetido en el 100% de sus respuestas — confound de formato, no de restricción |
| Nombre de la métrica de curvatura | "Ollivier-Ricci Δκ/W₁" → **"Forman-Ricci Δκ/W₁"** | `curvature_analyzer.py` (md5 idéntico en este panel y en `REPORTE_FASE0.md`) siempre calculó Forman (combinatoria), nunca Ollivier (transporte óptimo) — error heredado del origen del proyecto, no de esta sesión |
| Referencia bibliográfica | Ollivier, Y. (2009) → **Forman, R. (2003)** | Consecuencia directa del punto anterior |
| Tratamiento de `witness_soul_md`/`soul_md_corto` | "Control fuera de diseño, exploratorio" (`T2_REPLICATION_REPORT.md` §3) → **octava condición del panel (I=+, C=−)** (`paper/lsgot_4.md` §2.2, §3.8) | Converge limpiamente con `axis_pec_only` en las tres señales de identidad, en recuperación y en fidelidad de ruta — se revisó la recomendación original a la luz de esa convergencia; las salvedades (n=1, sin control de longitud/dominio, en inglés, pipeline T2) quedan explícitas en el paper, no se ocultaron |

## 4. Retirado — ya no se usa como evidencia de nada

| Qué se retiró | Por qué | Qué lo reemplaza |
|---|---|---|
| `chat_agente` / `chat_agente_sia` / `chat_agente_sia_v2` como condiciones de restricción/wiring | Wrapper HTML repetido en el 100% de las respuestas — confound de formato leído como señal geométrica | `automata_neutro` (0% markup, misma densidad de restricción) sostiene el Factor 1 por sí solo |
| Prediction 2(ii) — "declarado-pero-no-cableado recupera peor" | Dependía enteramente de `chat_agente_sia`/`_v2`, ambas excluidas | Marcada explícitamente como **no testeada**, no refutada — ver `paper/lsgot_4.md` §3.3, §5, §7.2 |
| Jerarquía original "dominio > restricción > identidad" (E-J, versión con chat_agente) | El término "dominio" era el propio confound de chat_agente | Jerarquía revisada, sin ese término espurio (`EJ_REPORT.md`) |

## 5. Abierto — nunca se confirmó ni se refutó

| Pregunta | Estado | Qué falta |
|---|---|---|
| **Causalidad de v̂** (E-K, steering) | 17/17 combinaciones probadas degeneraron en repetición; control aleatorio descarta que sea específico de v̂ — es el mecanismo de inyección sostenida el que falla | Reintentar con inyección puntual única o clamping de norma, no el mismo mecanismo |
| **Control definitivo de primer token** (forward pass, token forzado) | La versión proxy (léxica) ya corrió y es tranquilizadora — no revierte nada de este panel | Requiere GPU + Gemma-4-31B-it; baja prioridad relativa (ningún hallazgo del paper depende de ‖v1‖ en sí) |
| **Perturbación/recuperación en Qwen3** | No corrida | Requiere GPU + pipeline H4_rev completo en el segundo modelo |
| **Scale sweep real** (no solo cambio de arquitectura) | No corrida | El umbral de "~30B" sigue siendo evidencia informal de despliegue |
| **E-F / E-G** (logit-lens vertical, atención al prompt) | Nunca corridas | Extracción nueva, cara — candidatas a ronda dedicada |

## 6. Lectura de conjunto

Casi nada de lo que se corrigió esta ronda cambió una conclusión — cambió una
cifra (d=8.89→5.52) o un nombre (Ollivier→Forman). La excepción real es la
curvatura: corregir el nombre llevó a preguntarse qué mide la fórmula que se
pretendía usar desde el origen, y esa sí es información nueva — Ollivier-Ricci
genuina detecta identidad donde Forman-Ricci no veía nada (§2). No cambia
ninguna conclusión del paper (Δκ/W₁ sigue sin ser evidencia primaria, los
baselines simples le siguen ganando), pero sí corrige el motivo por el que se
excluye: no es que la curvatura sea ciega a la identidad en general, es que la
fórmula mal implementada lo era. Las tres señales de identidad centrales (v̂
media, t=0, dinámica temporal) sobrevivieron cada control al que se las
sometió, incluida la auto-auditoría de curvatura (en sus dos versiones) y la
replicación cross-modelo. Lo que sigue genuinamente abierto es angosto pero
real: causalidad (E-K), el control definitivo de primer token, y si la
mitad dinámica del argumento (recuperación gatillada por cableado)
sobrevive fuera de Gemma-4-31B-it — las tres son preguntas de "cuánto más
lejos llega esto", no de "¿era cierto lo que ya se afirmó?".
