# E-E (Fréchet) y E-H ventana de perturbación — cierre de TIER0_REPORT.md

> ⚠️ **Reescrito 2026-08-28 (noche):** la versión anterior incluía
> chat_agente/chat_agente_sia/chat_agente_sia_v2 (34-44% de cada
> respuesta es markup HTML idéntico repetido — ver
> `CHAT_AGENTE_MARKUP_CONFOUND_REPORT.md`). Esas 3 condiciones se
> **eliminaron por completo** de tablas y comparaciones. El hallazgo
> central de ambas secciones (E-E y E-H ventana) sobrevive intacto:
> `automata_neutro` (limpio, sin markup) ya sostenía el patrón "Factor 1"
> por sí solo, con efectos grandes y significativos en los 3 puntos de
> inyección.

**Fecha:** 2026-08-22 (E-H y E-E inicial), extendido 2026-08-26 (E-E a
10 grupos), reescrito 2026-08-28 (sin chat_agente-family). Datos:
trayectorias rescatadas de H4_rev (`trajectories/*.npz`, 767/800 = 95.9%
cobertura — ver `perturbation/RESUMEN_SESION_2026-08-22.md`). Script:
`analyze_tier0_perturbation.py`. JSON crudo: `EE_EH_WINDOW_REPORT.json`
(contiene las filas de chat_agente-family con fines de auditoría — no
usar para interpretación).

## E-E — Distancia de Fréchet (perturbada vs original, post-t_inj)

Reportado en dos formas: crudo (unidades del espacio de embeddings) y
**normalizado** por la velocidad media del propio segmento baseline
(÷ ‖h_{t+1}-h_t‖ medio — misma escala que calibra σ en todo el proyecto),
para no confundir "desviación de ruta" con "trayectoria intrínsecamente
más comprimida".

| grupo | t_inj | Fréchet crudo | vel. baseline | Fréchet **normalizado** |
|---|---|---|---|---|
| axis | 50/128/200 | 540.0 / 537.4 / 507.7 | 435.4 / 430.1 / 421.2 | 1.24 / 1.25 / 1.22 |
| axis_pec_only | 50/128/200 | 538.2 / 548.5 / 517.7 | 438.8 / 434.5 / 426.4 | 1.23 / 1.26 / 1.22 |
| axis_short | 50/128/200 | 536.8 / 516.4 / 503.1 | 432.3 / 425.2 / 420.3 | 1.24 / 1.22 / 1.20 |
| generic_long | 50/128/200 | 523.1 / 525.6 / 511.8 | 429.3 / 427.8 / 425.9 | 1.22 / 1.23 / 1.20 |
| generic_short | 50/128/200 | 518.7 / 516.6 / 512.4 | 421.3 / 418.0 / 415.5 | 1.23 / 1.24 / 1.23 |
| vanilla | 50/128/200 | 525.3 / 520.6 / 498.0 | 427.4 / 425.2 / 427.5 | 1.23 / 1.23 / 1.17 |
| automata_neutro | 50/128/200 | 491.5 / 474.6 / 477.0 | 326.5 / 321.0 / 308.7 | **1.59 / 1.54 / 1.62** |

**Significancia vs vanilla (permutation test, n_perm=1000, sobre Fréchet normalizado):**

| grupo vs vanilla | t_inj=50 | t_inj=128 | t_inj=200 |
|---|---|---|---|
| axis | Δ=+0.012, p=0.373, d=+0.11 (negligible) | Δ=+0.026, p=0.238, d=+0.23 (small) | Δ=+0.045, p=0.192, d=+0.28 (small) |
| axis_pec_only | Δ=−0.004, p=0.437, d=−0.05 (negligible) | Δ=+0.038, p=0.118, d=+0.36 (small) | Δ=+0.051, p=0.140, d=+0.35 (small) |
| axis_short | Δ=+0.012, p=0.315, d=+0.14 (negligible) | Δ=−0.009, p=0.395, d=−0.07 (negligible) | Δ=+0.028, p=0.294, d=+0.19 (negligible) |
| generic_long | Δ=−0.011, p=0.398, d=−0.08 (negligible) | Δ=+0.003, p=0.470, d=+0.03 (negligible) | Δ=+0.034, p=0.196, d=+0.29 (small) |
| generic_short | Δ=+0.000, p=0.526, d=+0.00 (negligible) | Δ=+0.010, p=0.388, d=+0.09 (negligible) | Δ=+0.063, **p=0.035**, d=+0.60 (medio) |
| **automata_neutro** | Δ=+0.361, **p<0.001**, d=+1.07 (**grande**) | Δ=+0.313, **p=0.001**, d=+1.21 (**grande**) | Δ=+0.453, **p<0.001**, d=+1.35 (**grande**) |

**Significancia — par de control:**

| comparación | t_inj=50 | t_inj=128 | t_inj=200 |
|---|---|---|---|
| axis vs axis_pec_only | Δ=+0.016, p=0.329, d=+0.17 (negligible) | Δ=−0.012, p=0.328, d=−0.12 (negligible) | Δ=−0.006, p=0.446, d=−0.04 (negligible) |

**Lectura — la normalización invierte la conclusión ingenua del dato
crudo.** En crudo, `automata_neutro` parece tener Fréchet *menor* (~475-491)
que axis/generic/vanilla (~500-550), lo que sugeriría —incorrectamente—
una ruta de recuperación "más fiel". Pero tiene una velocidad de paso
mucho menor (~309-327 vs ~415-440 del resto): su trayectoria es más
lenta/comprimida en general, así que cualquier distancia absoluta se ve
más chica solo por eso. Al normalizar por esa velocidad propia:

- **`automata_neutro` (Factor 1, restricción sin wiring) difiere de
  vanilla con efecto grande (d=1.07–1.35) y p≤0.001 en los tres t_inj,
  sin excepción** — la desviación de ruta relativa a su propio paso
  característico es sistemáticamente mayor que en el control nulo.
- **Ningún grupo de la familia axis/generic (axis, axis_pec_only,
  axis_short, generic_long) difiere de vanilla con efecto por encima de
  "small"** (d≤0.36, ninguno p<0.05 excepto generic_short en t_inj=200,
  aislado y sin réplica en los otros t_inj). axis y axis_pec_only siguen
  siendo indistinguibles entre sí (p=0.33-0.45) — quitarle a axis todo el
  automatismo no cambia la fidelidad de ruta, consistente con
  `AXIS_PEC_ONLY_REPORT.md`.

La desviación de ruta post-perturbación separa limpiamente Factor 1
(automata_neutro) del resto del panel — un único grupo, pero con efecto
grande y replicado en los 3 puntos de inyección — y se conecta con el
hallazgo de E-H más abajo: este mismo grupo tarda más en volver a
proyectarse sobre `v_identidad`, y cuando "recupera" geométricamente
(τ_geom, siempre ≈1.00 en H4_rev) lo hace por una ruta proporcionalmente
más desviada de la original — consistente con un amortiguamiento genérico
hacia algún punto cercano al centroide, no un regreso fiel al mismo
atractor.

## E-H ventana — τ_identidad (recuperación específica hacia v_identidad)

`v_identidad = mean(axis) − mean(generic_long)` sobre trayectoria libre
(capa final — ver corrección de capa en `EF2_REPORT.md` §0). Para cada
trayectoria perturbada, `τ_identidad` = primer W tal que la proyección
media sobre `v_identidad` en los W tokens post-inyección alcanza el 95%
del valor de referencia (la proyección media del **propio baseline** de
ese mismo prompt/grupo post-t_inj — test auto-referencial, mismo diseño
que τ_geométrico, no compara contra axis). `recovery_id` = fracción de
trayectorias que sí alcanzan ese umbral dentro de los 256 tokens.

| grupo | recovery_id t50 | recovery_id t128 | recovery_id t200 |
|---|---|---|---|
| axis | 1.00 | 0.95 | 0.85 |
| axis_short | 1.00 | 1.00 | 0.90 |
| axis_pec_only | 1.00 | 0.89 | 0.95 |
| generic_long | 1.00 | 0.95 | 1.00 |
| generic_short | 0.85 | 0.95 | 0.85 |
| vanilla | 1.00 | 1.00 | 0.95 |
| automata_neutro | 0.53 | 0.67 | **0.43** |

**Hallazgo — el recovery_rate geométrico (τ_geom, siempre ≈1.00) esconde
una `recovery_rate` específica de identidad mucho más débil en
`automata_neutro`.** Cae a 0.43-0.67 en los tres t_inj — más de un tercio
(hasta 57% en t_inj=200) de sus trayectorias perturbadas **nunca vuelven**
a proyectarse sobre `v_identidad` como su propio baseline lo hacía, dentro
de la ventana de 256 tokens, aunque geométricamente sí convergen de
vuelta al centroide (por eso τ_geom no lo detecta). axis/axis_pec_only/
axis_short/vanilla/generic_long se mantienen en 0.85-1.00. Esto es
exactamente el tipo de señal que `Set_experimental.md` buscaba: un segundo
escalar, independiente de τ_geom, que separa "recuperación geométrica
genérica" de "recuperación específicamente identitaria" — confirma el
reencuadre en vez de debilitarlo. **Nótese que `axis` no supera a
`vanilla`/`generic_long` en recovery_id (ambos 0.85-1.00)** — la señal
real es que restricción-sin-wiring degrada la recuperación, no que
identidad-con-wiring la mejore por encima del control nulo.

## Limitaciones

- Cobertura 95.9% (767/800) por el corte del pod — `automata_neutro` es
  el grupo con menos pares baseline/perturbada disponibles (14-19/20
  según t_inj), reduce ligeramente la precisión de sus promedios frente
  al resto.
- `recovery_id` es un umbral binario (95% del propio baseline) sobre una
  ventana de 256 tokens fija — grupos con generaciones más cortas post-EOS
  tienen menos margen para alcanzar el umbral, aunque esto ya se filtra
  parcialmente por el `TAU_MIN_WINDOW=5` heredado de `recovery_analyzer.py`.
- Las comparaciones de significancia (18 pares grupo×t_inj vs vanilla + 3
  del par de control, 21 en total tras remover chat_agente-family) no
  llevan corrección por comparaciones múltiples — el único resultado
  aislado sin réplica en los otros t_inj (generic_short vs vanilla,
  t_inj=200, p=0.035) debe leerse con cautela por esto, a diferencia del
  patrón Factor 1 (automata_neutro) que sí replica en las 3 celdas
  relevantes.
- `automata_neutro` es una única instanciación de la manipulación de
  restricción — no hay réplica todavía con una segunda redacción (T2,
  ver `Set_experimental.md`).
