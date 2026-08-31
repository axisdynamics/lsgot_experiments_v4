# E-H2 — Serie temporal de la proyección v̂ (dinámica de la dirección)

> ⚠️ **Reescrito 2026-08-28 (noche):** la versión anterior incluía
> chileatiende/chileatiende_sia/chileatiende_sia_v2 (34-44% de cada
> respuesta es markup HTML idéntico repetido — ver
> `CHILEATIENDE_MARKUP_CONFOUND_REPORT.md`). Esas 3 condiciones y el par
> diagnóstico que dependía de ellas (chileatiende_sia vs
> chileatiende_sia_v2) se **eliminaron por completo**, no solo se
> marcaron. El hallazgo central sobrevive intacto con `automata_neutro`
> como única referencia de restricción limpia.

**Fecha:** 2026-08-28
**Script:** `LSGOT_v4/scripts/fase0/analyze_EH2_serie_temporal.py`
**Datos:** embeddings libres (`sia_extended_v5`)
**Resultados crudos:** `LSGOT_v4/scripts/fase0/EH2_results.json` (contiene
las filas de chileatiende-family con fines de auditoría — no usar para
interpretación, ver confound report)

## 1. Resumen

Se calculó p(t) = cos(h_t, v̂) token a token y se caracterizó la serie
(autocorrelación, Hurst, ráfagas de activación). Las condiciones de
identidad (axis, axis_short, axis_pec_only) muestran **ráfagas de
activación sostenidas y largas** en la dirección de v̂ (media ~8 tokens,
hasta 30-35 de máximo) con autocorrelación *baja* (~0.26-0.29);
`automata_neutro` (restricción limpia, sin confound) muestra el patrón
inverso en menor grado: autocorrelación similar a la de identidad (0.248,
no elevada) pero ráfagas mucho más cortas (2.78) y proyección media casi
nula (+0.006, vs +0.2 de identidad). Todo el patrón es idéntico con y sin
t=0 (T1 no contamina esta métrica).

## 2. Método

- p(t) por prompt: `project_trajectory(h[0:L], v̂)` (misma función que
  E-H del paper, coseno).
- Métricas por serie: autocorrelación lag-1, Hurst (R/S sobre p(t)
  directo), fracción de tokens con p(t)>0, longitud media/máxima de
  rachas consecutivas con p(t)>0 ("ráfagas").
- Repetido completo con t=0 incluido y excluido.
- Comparaciones (permutación n=1000 + d de Cohen): axis vs axis_pec_only
  (ambos con wiring); axis_pec_only vs automata_neutro; axis vs vanilla.

## 3. Resultados (con t=0; sin t=0 es prácticamente idéntico, ver §3.3)

### 3.1 Métricas por condición

| Condición | autocorr(1) | Hurst | frac(p>0) | ráfaga media | ráfaga máx | mean p(t) |
|---|---|---|---|---|---|---|
| axis | 0.263 | 0.666 | 0.848 | 7.88 | 29.9 | +0.205 |
| axis_short | 0.294 | 0.653 | 0.850 | 7.81 | 26.0 | +0.202 |
| axis_pec_only | 0.265 | 0.671 | 0.869 | **8.81** | **34.7** | +0.212 |
| vanilla | 0.249 | 0.684 | 0.744 | 4.62 | 18.6 | +0.098 |
| generic_long | 0.264 | 0.698 | 0.699 | 4.18 | 17.9 | +0.067 |
| generic_short | 0.248 | 0.720 | 0.649 | 3.40 | 15.7 | +0.048 |
| automata_neutro | 0.248 | 0.677 | 0.493 | 2.78 | 11.8 | +0.006 |

### 3.2 Comparaciones clave (permutación, con_t0 ≈ sin_t0)

| Par | métrica | d | p |
|---|---|---|---|
| axis vs axis_pec_only | ráfaga media | −0.43 | 0.09 (n.s.) |
| axis vs axis_pec_only | frac(p>0) | −0.53 | 0.06 (n.s.) |
| axis_pec_only vs automata_neutro | mean p(t) | +5.52 | <0.001 |
| axis vs vanilla | mean p(t) | +3.45 | <0.001 |
| axis_pec_only vs automata_neutro | ráfaga media | +3.42 | <0.001 |

### 3.3 Robustez a T1 (con t=0 vs sin t=0)

Todas las métricas cambian <0.01 en magnitud absoluta y ningún p-valor
cruza el umbral de 0.05 al excluir t=0 (ver `EH2_results.json`, bloques
`con_t0`/`sin_t0`). La serie temporal p(t) **no depende** del primer
token — a diferencia de la proyección media cruda (E-L), aquí el confound
de t=0 es irrelevante porque es solo 1 de ~250 puntos de la serie.

## 4. Trampas aplicadas

- **T1**: reportado explícitamente con/sin t=0 (§3.3) — sin efecto
  detectable en esta métrica.
- **T2**: n=20 prompts, 1 manipulación por condición — no hay réplica de
  la manipulación todavía (ver Set_experimental.md, réplicas pendientes).
- **T5**: 3 pares × 2 métricas destacadas = 6 comparaciones centrales;
  todas replican en la misma dirección.
- **T11**: chileatiende-family eliminada por completo de este reporte.

## 5. Replicación

El contraste identidad-vs-restricción (ráfagas largas vs cortas, mean
p(t) positivo vs casi nulo) replica en 2 métricas relacionadas
(mean p(t), ráfaga media) sobre el único par identidad-vs-restricción
limpio disponible (axis_pec_only vs automata_neutro). El resultado nulo
de axis vs axis_pec_only es consistente entre con/sin t=0.

## 6. Implicación para el paper

Restricción limpia (`automata_neutro`) no muestra la autocorrelación
elevada que sugería la versión contaminada de este reporte (esa
elevación —0.48-0.52— era específica de chileatiende-family, ahora
eliminada) — la autocorrelación de automata_neutro (0.248) es
prácticamente igual a la de identidad. Lo que sí distingue limpiamente a
restricción es: **ráfagas mucho más cortas** (2.78 vs 7.8-8.8) y
**proyección media casi nula** (+0.006 vs +0.2) — restricción no "se aleja
activamente" de v̂ de forma sostenida, simplemente no logra activar
ráfagas largas en esa dirección. Es una lectura más modesta y más
correcta que la de la versión anterior (que atribuía a restricción una
dinámica de "atrapamiento" con alta persistencia — ese patrón era, en
retrospectiva, mayormente un artefacto de la repetición de markup en
chileatiende).
