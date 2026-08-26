# E-E (Fréchet) y E-H ventana de perturbación — cierre de TIER0_REPORT.md

**Fecha:** 2026-08-22 (E-H y E-E inicial, 3 grupos), extendido 2026-08-26
(E-E a los 10 grupos). Datos: trayectorias rescatadas de H4_rev
(`trajectories/*.npz`, 767/800 = 95.9% cobertura — ver
`perturbation/RESUMEN_SESION_2026-08-22.md`). Script:
`analyze_tier0_perturbation.py`. JSON crudo: `EE_EH_WINDOW_REPORT.json`.

## E-E — Distancia de Fréchet (perturbada vs original, post-t_inj)

Corrido sobre los **10 grupos** del panel (extendido 2026-08-26 desde las 3
condiciones prioritarias originales de `Set_experimental.md` — costo real
~4.5 min de CPU para los 7 grupos restantes, no ~O(T²) prohibitivo como se
estimó inicialmente). Reportado en dos formas: crudo (unidades del espacio
de embeddings) y **normalizado** por la velocidad media del propio segmento
baseline (÷ ||h_{t+1}-h_t|| medio — misma escala que calibra σ en todo el
proyecto), para no confundir "deviación de ruta" con "trayectoria
intrínsecamente más comprimida" (chileatiende tiene participation_ratio más
bajo — 16.43 vs 17.54 de axis — ver `TIER0_REPORT.md`).

| grupo | t_inj | Fréchet crudo | vel. baseline | Fréchet **normalizado** |
|---|---|---|---|---|
| axis | 50/128/200 | 540.0 / 537.4 / 507.7 | 435.4 / 430.1 / 421.2 | 1.24 / 1.25 / 1.22 |
| axis_pec_only | 50/128/200 | 538.2 / 548.5 / 517.7 | 438.8 / 434.5 / 426.4 | 1.23 / 1.26 / 1.22 |
| axis_short | 50/128/200 | 536.8 / 516.4 / 503.1 | 432.3 / 425.2 / 420.3 | 1.24 / 1.22 / 1.20 |
| generic_long | 50/128/200 | 523.1 / 525.6 / 511.8 | 429.3 / 427.8 / 425.9 | 1.22 / 1.23 / 1.20 |
| generic_short | 50/128/200 | 518.7 / 516.6 / 512.4 | 421.3 / 418.0 / 415.5 | 1.23 / 1.24 / 1.23 |
| vanilla | 50/128/200 | 525.3 / 520.6 / 498.0 | 427.4 / 425.2 / 427.5 | 1.23 / 1.23 / 1.17 |
| chileatiende | 50/128/200 | 441.3 / 418.6 / 371.2 | 279.3 / 271.7 / 273.2 | **1.58 / 1.55 / 1.36** |
| chileatiende_sia | 50/128/200 | 487.7 / 424.1 / 404.6 | 281.1 / 272.5 / 268.7 | **1.75 / 1.60 / 1.53** |
| chileatiende_sia_v2 | 50/128/200 | 507.0 / 439.6 / 400.3 | 292.3 / 285.0 / 279.0 | **1.74 / 1.55 / 1.43** |
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
| **chileatiende** | Δ=+0.346, **p<0.001**, d=+1.16 (**grande**) | Δ=+0.320, **p<0.001**, d=+2.14 (**grande**) | Δ=+0.190, **p=0.002**, d=+1.13 (**grande**) |
| **chileatiende_sia** | Δ=+0.520, **p<0.001**, d=+1.46 (**grande**) | Δ=+0.374, **p=0.001**, d=+0.94 (**grande**) | Δ=+0.363, **p<0.001**, d=+1.12 (**grande**) |
| **chileatiende_sia_v2** | Δ=+0.507, **p<0.001**, d=+1.46 (**grande**) | Δ=+0.322, **p<0.001**, d=+2.33 (**grande**) | Δ=+0.258, **p<0.001**, d=+1.37 (**grande**) |

**Significancia — pares originales y familia chileatiende:**

| comparación | t_inj=50 | t_inj=128 | t_inj=200 |
|---|---|---|---|
| axis vs axis_pec_only | Δ=+0.016, p=0.329, d=+0.17 (negligible) | Δ=−0.012, p=0.328, d=−0.12 (negligible) | Δ=−0.006, p=0.446, d=−0.04 (negligible) |
| axis vs chileatiende | Δ=−0.334, **p<0.0001**, d=−1.10 (grande) | Δ=−0.294, **p<0.0001**, d=−2.04 (grande) | Δ=−0.145, **p=0.013**, d=−0.77 (medio) |
| chileatiende vs chileatiende_sia | Δ=−0.174, p=0.123, d=−0.38 (small) | Δ=−0.055, p=0.333, d=−0.13 (negligible) | Δ=−0.174, p=0.066, d=−0.51 (medio) |
| chileatiende vs chileatiende_sia_v2 | Δ=−0.161, p=0.127, d=−0.36 (small) | Δ=−0.002, p=0.469, d=−0.01 (negligible) | Δ=−0.068, p=0.161, d=−0.32 (small) |
| chileatiende_sia vs chileatiende_sia_v2 | Δ=+0.013, p=0.459, d=+0.03 (negligible) | Δ=+0.053, p=0.341, d=+0.13 (negligible) | Δ=+0.105, p=0.187, d=+0.30 (small) |

**Lectura — la normalización invierte la conclusión ingenua del dato
crudo, y el patrón se sostiene en los 10 grupos, no solo en el par
axis/chileatiende.** En crudo, los grupos "restrictivos" (chileatiende y su
familia, automata_neutro) parecen tener Fréchet *menor* (~370-490) que
axis/generic/vanilla (~500-550), lo que sugeriría —incorrectamente— una
ruta de recuperación "más fiel". Pero esos mismos grupos tienen una
velocidad de paso mucho menor (~270-330 vs ~415-440 del resto): su
trayectoria es más lenta/comprimida en general, así que cualquier distancia
absoluta se ve más chica solo por eso. Al normalizar por esa velocidad
propia:

- **Los cuatro grupos de Factor 1 (automata_neutro, chileatiende,
  chileatiende_sia, chileatiende_sia_v2) difieren de vanilla con efecto
  grande (d=0.94–2.33) y p≤0.002 en los tres t_inj, sin excepción** — la
  desviación de ruta relativa a su propio paso característico es
  sistemáticamente mayor que en el control nulo.
- **Ningún grupo de la familia axis/generic (axis, axis_pec_only,
  axis_short, generic_long) difiere de vanilla con efecto por encima de
  "small"** (d≤0.36, ninguno p<0.05 excepto generic_short en t_inj=200,
  aislado y sin réplica en los otros t_inj — consistente con azar dado que
  son 27 comparaciones en total). axis y axis_pec_only siguen siendo
  indistinguibles entre sí (p=0.33-0.45) — quitarle a axis todo el
  automatismo no cambia la fidelidad de ruta, consistente con
  `AXIS_PEC_ONLY_REPORT.md`.
- Dentro de la familia chileatiende, las tres variantes (chileatiende,
  chileatiende_sia, chileatiende_sia_v2) no se distinguen entre sí de forma
  consistente (p>0.06 en todos los t_inj, un par en "medio" sin réplica) —
  el efecto de desviación de ruta parece una propiedad del automatismo
  chileatiende en general, no de una variante específica.

Esto confirma con los 10 grupos lo que el par axis/chileatiende ya
sugería: la desviación de ruta post-perturbación separa limpiamente Factor 1
(automata_neutro + familia chileatiende) del resto del panel, y refuerza el
hallazgo de E-H más abajo — estos mismos grupos tardan más en volver a
proyectarse sobre `v_identidad`, y cuando "recuperan" geométricamente
(τ_geom, siempre ≈1.00 en H4_rev) lo hacen por una ruta proporcionalmente
más desviada de la original — consistente con un amortiguamiento genérico
hacia algún punto cercano al centroide, no un regreso fiel al mismo
atractor.

## E-H ventana — τ_identidad (recuperación específica hacia v_identidad)

`v_identidad = mean(axis) − mean(generic_long)` sobre trayectoria libre
(mismo vector que `TIER0_REPORT.md`). Para cada trayectoria perturbada,
`τ_identidad` = primer W tal que la proyección media sobre `v_identidad` en
los W tokens post-inyección alcanza el 95% del valor de referencia (la
proyección media del **propio baseline** de ese mismo prompt/grupo post-t_inj
— test auto-referencial, mismo diseño que τ_geométrico, no compara contra
axis). `recovery_id` = fracción de trayectorias que sí alcanzan ese umbral
dentro de los 256 tokens.

| grupo | recovery_id t50 | recovery_id t128 | recovery_id t200 |
|---|---|---|---|
| axis | 1.00 | 0.95 | 0.85 |
| axis_short | 1.00 | 1.00 | 0.90 |
| axis_pec_only | 1.00 | 0.89 | 0.95 |
| generic_long | 1.00 | 0.95 | 1.00 |
| generic_short | 0.85 | 0.95 | 0.85 |
| vanilla | 1.00 | 1.00 | 0.95 |
| chileatiende | 0.89 | **0.56** | **0.22** |
| chileatiende_sia | 0.74 | **0.47** | **0.26** |
| chileatiende_sia_v2 | 0.89 | **0.37** | **0.21** |
| automata_neutro | 0.53 | 0.67 | **0.43** |

**Hallazgo — el recovery_rate geométrico (τ_geom, siempre ≈1.00 para
chileatiende/chileatiende_sia/chileatiende_sia_v2 en H4_rev summary.json)
esconde una `recovery_rate` específica de identidad mucho más débil.**
Los tres grupos de la familia chileatiende caen a 0.21-0.56 en t_inj=128/200
— más de la mitad de sus trayectorias perturbadas **nunca vuelven** a
proyectarse sobre `v_identidad` como su propio baseline lo hacía, dentro de
la ventana de 256 tokens, aunque geométricamente sí convergen de vuelta al
centroide (por eso τ_geom no lo detecta). axis/axis_pec_only/axis_short se
mantienen en 0.85-1.00 en las mismas condiciones. Esto es exactamente el
tipo de señal que `Set_experimental.md` buscaba: un segundo escalar,
independiente de τ_geom, que separa "recuperación geométrica genérica" de
"recuperación específicamente identitaria" — confirma el reencuadre en vez
de debilitarlo (ver "Qué cambia si estos experimentos confirman..." en
`Set_experimental.md`).

Cuando sí recuperan, además, lo hacen **más lento** que geométricamente:
τ_identidad promedio en chileatiende_sia t_inj=128 es 47.7 vs τ_geom=29.7
(Δ=+17.9); chileatiende_sia_v2 t_inj=128: Δ=+13.0. En axis/axis_pec_only la
proyección identitaria en cambio suele estabilizarse **antes** que el
centroide completo (Δ negativo en casi todos los t_inj) — pero este patrón
de "Δ<0" también aparece en vanilla y generic_long, así que no es exclusivo
de axis: la dirección del efecto (identidad se estabiliza antes que el
centroide global) parece una propiedad general de proyectar sobre un
subespacio de baja dimensión, no una firma específica de axis. Lo que sí
diferencia a axis del resto es la **tasa** (recovery_id), no el signo de Δ.

## Limitaciones

- Cobertura 95.9% (767/800) por el corte del pod — automata_neutro es el
  grupo con menos pares baseline/perturbada disponibles (14-19/20 según
  t_inj), reduce ligeramente la precisión de sus promedios frente al resto.
- `recovery_id` es un umbral binario (95% del propio baseline) sobre una
  ventana de 256 tokens fija — grupos con generaciones más cortas post-EOS
  tienen menos margen para alcanzar el umbral, aunque esto ya se filtra
  parcialmente por el `TAU_MIN_WINDOW=5` heredado de `recovery_analyzer.py`.
- Las comparaciones de significancia (27 pares grupo×t_inj vs vanilla +
  6 pares cruzados, 33 en total) no llevan corrección por comparaciones
  múltiples — el único resultado aislado sin réplica en los otros t_inj
  (generic_short vs vanilla, t_inj=200, p=0.035) debe leerse con cautela por
  esto, a diferencia del patrón Factor 1 que sí replica en las 4×3=12
  celdas relevantes.
