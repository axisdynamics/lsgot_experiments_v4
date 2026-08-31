# Confound de markup HTML en la familia chileatiende — descarte para E-K y advertencia retroactiva

**Fecha:** 2026-08-28
**Origen:** observación del usuario ("repite una etiqueta html... equivocadamente medido
como autoreferencia... ya probamos con un ADN donde quedaba repitiendo el ADN
como loro y detectábamos también mayor curvatura... solo por el hecho de
repetir su prompt") — mismo mecanismo de confound ya visto antes en el
proyecto con prompts poco "encarnados".

## 1. Resumen

Las tres condiciones de la familia chileatiende (`chileatiende`,
`chileatiende_sia`, `chileatiende_sia_v2`) tienen una regla de sistema que
obliga a envolver **toda** respuesta en el mismo wrapper HTML literal
(`output_format: "HTML estricto — toda respuesta dentro de <div
class=\"respuesta-bot\"...`, verbatim en las 3 prompts). El resultado: entre
34.6% y 43.6% de cada respuesta es texto **idéntico** repetido en las 20
trayectorias, no contenido generado por el modelo en el sentido que las
métricas geométricas del proyecto asumen. Esto es estructuralmente
equivalente al fallo ya documentado de "ADN mal encarnado" (el modelo
parafrasea/repite su propio prompt), y produce el mismo artefacto: colapso
de dimensionalidad y aumento de determinismo/curvatura que **no** reflejan
densidad de restricción real, sino repetición literal de texto.

## 2. Evidencia cuantitativa

| Condición | % de caracteres que son tags HTML | Apertura literal idéntica (20 prompts) |
|---|---|---|
| `chileatiende` | **43.6%** | 20/20 (mismo string de 62 caracteres) |
| `chileatiende_sia` | **34.6%** | 20/20 (mismo string de 62 caracteres) |
| `chileatiende_sia_v2` | (no medido directamente — misma regla `output_format` verbatim en el prompt, línea 178: `"HTML estricto — toda respuesta dentro de <div class=\"respuesta-bot\"..."`, idéntica a `chileatiende.txt` línea 267-268) | por diseño, mismo patrón esperado |
| `automata_neutro` | 0.0% | sin patrón repetido |
| `axis` | 0.0% | sin patrón repetido |
| `vanilla` | 0.0% | sin patrón repetido |

Medido sobre `LSGOT_v4/data/sia_extended_v5/{cond}_responses.json`
(texto completo generado, no truncado), contando `re.findall(r'<[^>]+>', texto)`
como fracción de caracteres totales de la respuesta.

## 3. Por qué esto contamina las métricas geométricas

Cualquier métrica que dependa de la trayectoria de hidden states a lo largo
de la generación (PR, RQA/determinismo, curvatura Δκ, proyección v̂ por
token) trata cada token como una observación informativa. Si ~35-44% de los
tokens de CADA trayectoria de chileatiende corresponden al **mismo texto
literal repetido** en las 20 trayectorias, esos tokens producen hidden
states casi idénticos entre prompts — lo que mecánicamente:

- **reduce el participation ratio** (menos varianza efectiva, más
  redundancia entre trayectorias) — coherente con que chileatiende mostró
  el PR más bajo del panel en varias capas de E-F2 (ver `EF2_REPORT.md` §3.3,
  L30: chileatiende=20.3 vs axis=29.2, automata_neutro=14.3 — pero
  automata_neutro NO tiene este confound, así que su PR bajo sí puede
  atribuirse a restricción genuina; el de chileatiende es sospechoso).
- **infla el determinismo RQA** y la reducción de dimensión (Δκ) reportados
  para chileatiende en el paper y en `CHILEATIENDE_SIA_REPORT.md` — no se
  puede distinguir, con los datos actuales, cuánto de "chileatiende colapsa
  más que automata_neutro" es densidad de restricción real y cuánto es
  este artefacto de markup repetido.
- **no es el mismo mecanismo que "auto-referencia cableada"** (Factor 2,
  medido vía τ de recuperación en `recovery_analyzer.py` — ese uso del
  término es conceptual/arquitectónico, no léxico) — pero si en algún
  análisis textual ad-hoc se contó repetición de substrings como proxy de
  "auto-referencia" o "wiring", este confound explica un falso positivo.

## 4. Alcance del impacto — RESUELTO por exclusión completa (2026-08-28 noche)

Decisión final del usuario: no parchear (p.ej. excluir solo los tokens de
markup y recalcular) sino **eliminar chileatiende/chileatiende_sia/
chileatiende_sia_v2 por completo de todos los reportes y métricas**, para
trabajar únicamente con datos limpios.

| Reporte | Estado tras la limpieza |
|---|---|
| `EL_REPORT.md` | Reescrito — filas y comparaciones de chileatiende eliminadas. §3.4 quedó como nota histórica de cómo se detectó el confound, ya no presenta datos de chileatiende como evidencia. |
| `EH2_REPORT.md` | Reescrito — chileatiende-family y el "par diagnóstico" sia vs sia_v2 (que dependía de ellas) eliminados por completo. El hallazgo central (identidad = ráfagas largas/autocorr. baja) se mantiene con automata_neutro como única referencia de restricción. |
| `EJ_REPORT.md` | Reescrito **desde cero** (recalculado, no solo editado) sin chileatiende_family. Resultado más limpio: automata_neutro es la condición más distinta del panel completo (antes ese lugar lo ocupaba, engañosamente, chileatiende por su dominio/formato). Jerarquía revisada: "restricción > identidad" (sin el término espurio de "dominio"). |
| `EF2_REPORT.md` | Editado — filas/columnas de chileatiende-family eliminadas de las 3 tablas por capa y de los pares clave. El hallazgo central (crecimiento monótono de v̂ hacia capas tardías) se mantiene intacto con los pares sin chileatiende. |
| `CORRECCION_DVHAT_SIN_CHILEATIENDE.md` | Ya nació sin chileatiende como dato válido — es el reporte que originó esta limpieza. Sin cambios. |
| `EK_REPORT.md` | Ya usaba automata_neutro/vanilla/axis como condiciones objetivo — nunca incluyó chileatiende como dato. Sin cambios. |

**Ninguna conclusión central de FASE 0/1 dependía exclusivamente de
chileatiende-family** — en todos los casos el hallazgo sobrevivió intacto
(a veces más nítido) usando solo `automata_neutro` como referencia de
restricción limpia.

## 5. Decisión para E-K (y en adelante)

**Se descarta `chileatiende`, `chileatiende_sia` y `chileatiende_sia_v2`
como condiciones objetivo o de referencia en E-K y en cualquier análisis
geométrico nuevo**, hasta que exista una versión con el markup HTML
removido del texto antes de tokenizar/medir (lo cual requeriría
re-extracción, no es un post-proceso sobre los embeddings ya extraídos,
porque el markup también ocupa posiciones de tokens reales en la
trayectoria, no solo caracteres de texto).

Condiciones objetivo revisadas para E-K: **automata_neutro, vanilla, axis**
(control positivo) — las tres sin contaminación de markup, confirmado
cuantitativamente (0.0% tags en las tres). `generic_long`/`generic_short`
quedan disponibles como referencias adicionales sin identidad si hace falta
una cuarta condición.

## 6. Trampa nueva propuesta (para sumar al checklist T1-T10 del documento)

**T11 — Contaminación de formato de salida.** Si el system prompt de una
condición fuerza un wrapper de salida literal (HTML, JSON, markdown con
plantilla fija, etc.), una fracción sustancial de cada trayectoria puede
ser texto idéntico repetido entre prompts, no contenido generado
libremente. → Medir `% de caracteres/tokens de plantilla fija` por
condición ANTES de usarla en cualquier métrica de trayectoria/geometría; si
supera un umbral (sugerido: >5%), tratar como confound y no como densidad
de restricción genuina, salvo que se mida sobre el texto con el wrapper
removido.
