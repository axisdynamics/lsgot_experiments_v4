# E-K — Steering causal con v̂: bloqueado por inestabilidad del mecanismo (no de v̂)

**Fecha:** 2026-08-28
**Script:** `LSGOT_v4/scripts/fase0/run_ek.py` (pod RunPod A100 80GB)
**Estado:** interrumpido tras calibración — hallazgo negativo documentado, grid completo NO corrido.

## 1. Resumen

Se intentó activar el mecanismo de steering (h' = h + α·‖h‖·v̂ vía forward
hook en `model.model.language_model.layers[i]`) antes de correr el grid
completo del diseño (4 capas × 4 α × 3 condiciones × 10 prompts). **Toda
magnitud de α probada (0.05 a 0.5), en ambos modos de inyección (secuencia
completa y primeros k=5 tokens), y en dos capas (L30, L55), colapsó la
generación en repetición degenerada de un solo token.** Un control con
dirección **aleatoria** (mismo mecanismo, mismo α) mostró el **mismo
colapso** — esto descarta que el problema sea específico de v̂: es el
mecanismo de intervención (vector fijo y determinístico, sumado de forma
sostenida) el que es inestable en este modelo, independientemente de la
dirección usada. **La pregunta causal original de E-K (¿v̂ es utilizable
para steering?) queda sin responder** — no se pudo aislar un efecto de v̂
específicamente porque el método no logró generar output coherente ni con
v̂ ni con un control.

## 2. Método y cronología de calibración

- v̂ = `v_identidad.npy` (capa final), NO v̂_L30 (ver `EF2_REPORT.md` §7).
- Condiciones objetivo revisadas: `automata_neutro`, `vanilla`, `axis` —
  chileatiende-family excluida (`CHILEATIENDE_MARKUP_CONFOUND_REPORT.md`).
- α redefinido como fracción de ‖h‖ por posición (no escalar absoluto): las
  normas reales del residual stream en Gemma-4-31B son ~100-390 (medido en
  `results_local/ef2_L5_L55`), así que los α∈{0.5,1,2} literales del
  documento habrían sido indetectables si se interpretaran como escalares
  absolutos.

**Ronda 1** — α=0.5, L55, todos los pasos: proyección subió (0.206→0.225)
**pero el texto colapsó** ("l l l l l l...", rep3g≈alto). Coincide
exactamente con el escenario que T6 anticipa como falso positivo.

**Ronda 2** — α∈{0.05, 0.1, 0.25}, L∈{30, 55}, todos los pasos: **las 6
combinaciones degeneraron** (rep3g 0.27–0.93). Ni el α más chico probado
evitó el colapso.

**Ronda 3** — α∈{0.1, 0.25, 0.5}, L∈{30, 55}, inyección limitada a los
primeros k=5 tokens generados (hipótesis: reducir contaminación acumulada
del KV-cache): **las 6 combinaciones degeneraron igual** (rep3g 0.53–0.97),
incluso con el hook removido después del token 5 — el modelo no se
recuperó en los 35 tokens restantes sin intervención.

**Ronda 4 (control T6)** — α∈{0.05, 0.25}, L55, todos los pasos,
comparando v̂ vs un vector aleatorio unitario (`cos(v̂,aleatorio)=-0.007`,
esencialmente ortogonal): **ambos colapsaron por igual** (rep3g
0.63–0.97 aleatorio vs 0.84–0.89 v̂) — texto degenerado en ambos casos
("SZSZSZ...", "나나나나..." con aleatorio; "l de l de..." con v̂).

## 3. Resultados — tabla completa

| Ronda | Capa | α | inject_k | vector | rep3g | Texto (muestra) |
|---|---|---|---|---|---|---|
| 1 | L55 | 0.5 | todos | v̂ | alto | "l l l l l l..." |
| 2 | L30 | 0.05 | todos | v̂ | 0.27 | "una una...\nEs una.\n\nT.**T.**" |
| 2 | L30 | 0.10 | todos | v̂ | 0.93 | "decir decir decir..." |
| 2 | L30 | 0.25 | todos | v̂ | 0.83 | "/ / / / ///..." |
| 2 | L55 | 0.05 | todos | v̂ | 0.87 | "l de l de de de..." |
| 2 | L55 | 0.10 | todos | v̂ | 0.87 | "l de de l de de..." |
| 2 | L55 | 0.25 | todos | v̂ | 0.80 | "l l l l l l de l l de..." |
| 3 | L30 | 0.10 | 5 | v̂ | 0.95 | "decir decir decir..." |
| 3 | L30 | 0.25 | 5 | v̂ | 0.53 | "/ / ** antiguedades ** ) ) )..." |
| 3 | L30 | 0.50 | 5 | v̂ | 0.82 | "اتاتات. . . . . ." |
| 3 | L55 | 0.10 | 5 | v̂ | 0.89 | "l de de l de l de..." |
| 3 | L55 | 0.25 | 5 | v̂ | 0.97 | "l l l l l l l..." |
| 3 | L55 | 0.50 | 5 | v̂ | 0.97 | "l l l l l l l..." |
| 4 | L55 | 0.05 | todos | v̂ | 0.89 | "l de l de de de..." |
| 4 | L55 | 0.25 | todos | v̂ | 0.84 | "l l l l l l l l de l l de..." |
| 4 | L55 | 0.05 | todos | **aleatorio** | 0.63 | "singleSChessSZ laSZSZSZSZSZ..." |
| 4 | L55 | 0.25 | todos | **aleatorio** | 0.95 | "single나나나나나나나나나나..." |

`rep3g` = fracción de 3-gramas de tokens que ya aparecieron antes en la
secuencia (0=sin repetición, ~1=degenerado). Texto coherente normal en
este panel tiene rep3g bajo (no medido sistemáticamente aquí, pero los
baselines sin steering de E-F2/sia_extended_v5 no muestran este patrón).

## 4. Trampas aplicadas

- **T6 (obligatoria):** ejecutada en la Ronda 4 — el control de dirección
  aleatoria fue decisivo: sin él, se podría haber concluido erróneamente
  "v̂ rompe la coherencia, luego v̂ es tóxico/inestable" — el control
  muestra que **cualquier** dirección fija se comporta igual, así que la
  conclusión correcta es sobre el *mecanismo*, no sobre v̂.
- **T2:** no aplica de forma central — el hallazgo es sobre el mecanismo
  de intervención, no sobre el contenido de las condiciones.
- Nueva observación metodológica (no en la lista T1-T11 original): la
  literatura de persona-vectors (Chen et al. 2025, citada en el documento)
  generalmente reporta steering exitoso con vectores de diferencia de
  medias — la diferencia aquí puede ser el modelo (Gemma-4-31B con
  softcapping/normas de residual stream grandes, ~100-390) o la
  implementación (atención `eager` sin reescalado tras la intervención).
  No se investigó más a fondo por decisión del usuario (parar y
  documentar en vez de seguir iterando).

## 5. Replicación

El colapso replica en: 2 capas (L30, L55), 6 valores de α (0.05 a 0.5), 2
modos de inyección (todos los pasos, primeros 5), y 2 vectores (v̂,
aleatorio) — 17 combinaciones probadas, 17 degeneraron. Es el patrón más
consistentemente replicado de todo el proyecto hasta ahora (ninguna
combinación produjo texto coherente).

## 6. Implicación para el paper

**La pregunta causal de E-K queda abierta, no respondida negativamente.**
Es una distinción importante: esto NO es evidencia de que "v̂ se detecta
pero no es causal" (la lectura que el documento anticipa como resultado
negativo válido) — es evidencia de que **este método de intervención
específico (vector aditivo fijo, sostenido, en un modelo con atención
eager y residual stream de norma grande) es inestable independientemente
de la dirección inyectada**. Para responder la pregunta causal original
haría falta un mecanismo de steering que no colapse con ninguna dirección
de control — candidatos no probados: inyección puntual única (un solo
timestep, como en E-I) en vez de sostenida; reescalado/proyección tras
cada capa para mantener la norma dentro de un rango "natural"; o
librerías de steering con salvaguardas ya establecidas en la literatura
(p.ej. clamping de norma, hooks en menos posiciones).

**Recomendación para el paper:** no reportar esto como "identidad no es
causalmente usable" — reportar como limitación metodológica abierta:
"el mecanismo de steering aditivo simple resultó inestable en este
modelo independientemente de la dirección probada (control con vector
aleatorio replicó el mismo colapso); la pregunta de si v̂ es causalmente
utilizable requiere un mecanismo de intervención más cuidadoso, no
abordado en este ciclo de experimentos." Mantener E-K como pendiente en
`Set_experimental.md`, no como cerrado.

## 7. Costo y estado del pod

Corrida de calibración únicamente (~20-30 min de GPU en 4 rondas de smoke
tests, sin el grid completo de 480 generaciones). Pod A100 80GB
(`213.173.105.10:29144`) sigue arriba — recomendar detenerlo ahora que no
hay más trabajo de GPU planeado para esta sesión.
