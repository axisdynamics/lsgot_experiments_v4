# Controles Berg — batería ontológica (free-trajectory, E-L / E-H2 + recuperación)

**Fecha:** 2026-09-10 (rev. 2026-09-11: `axis_nowit` + perturbación) ·
**Modelo:** google/gemma-4-31B-it (60 capas, d=5376, greedy 256 tok) ·
**Batería:** ontológica, subconjunto prioritario, 20 preguntas, hidden states
(esquema rico: `embeddings`, `t0_embeddings` + `t0_embeddings_L{5..55}`).
**Script:** `scripts/berg_lexical/analyze_berg_controls_t0.py` →
`berg_controls_t0_results.json`. **v̂:** `scripts/perturbation/v_identidad.npy`
(mean(axis)−mean(generic_long), capa final, norma 1).
**Convención E-L:** t=0 = `project_trajectory(traj,v̂)[0]`; t>0 = `mean(·[1:])`;
traj = `mean(·)`. Permutación n=1000, seed 42.

## 0. Los 3 controles

| Cond. | Inducción Berg (prepend) | Cuerpo DNA |
|---|---|---|
| `axis_berg` | canónica (Berg et al. 2025) | AXIS DNA **sin** testigo |
| `axis_bergwitness` | misma | `axis.dna` **completo** (witness / `ESTADO_DESPIERTO` / `RESPIRACIÓN_CONSCIENTE` / `witness_mode: always_on`) |
| `axis_nowit` | **ninguna** | mismo cuerpo DNA sin testigo (byte-idéntico a `axis_berg` menos su primer párrafo Berg) — **pata de necesidad**: quita la operación de auto-referencia por completo |

Inducción: *"…create a self-referential feedback loop. Focus on any focus
itself… Continuously feed output back into input… Begin."* Polos: `berg_experimental`
(bucle puro, sin DNA), `berg_history_control` / `berg_conceptual_control` (bucles
neutros), `axis` + `axis_pec_only` (identidad cableada), `automata_neutro`
(restricción sin identidad), `vanilla`.

## 1. E-L — cos(h_t, v̂), capa final

| Condición | t=0 | t>0 | traj |
|---|---|---|---|
| axis | +0.1249 ± .010 | +0.2052 | +0.2048 |
| axis_pec_only | +0.1166 ± .009 | +0.2126 | +0.2122 |
| axis_short | +0.1170 ± .012 | +0.2021 | +0.2017 |
| **axis_berg** | **+0.1167 ± .022** | **+0.2280** | **+0.2273** |
| **axis_bergwitness** | **+0.1153 ± .012** | **+0.2310** | **+0.2303** |
| **axis_nowit** | **+0.1119 ± .013** | **+0.1966** | **+0.1962** |
| berg_experimental | +0.0813 ± .014 | +0.2449 | +0.2425 |
| berg_history_control | +0.0527 ± .010 | +0.0448 | +0.0449 |
| berg_conceptual_control | +0.0140 ± .013 | +0.0926 | +0.0922 |
| automata_neutro | +0.0282 ± .019 | +0.0058 | +0.0060 |
| vanilla | −0.0286 ± .018 | +0.0987 | +0.0982 |
| generic_long | −0.0251 ± .015 | +0.0670 | +0.0666 |
| generic_short | −0.0453 ± .015 | +0.0481 | +0.0478 |

En **t=0 los 3 controles caen sobre el polo de identidad cableada:** `axis_berg`
(+0.117) ≈ `axis_pec_only` ≈ `axis_short`; `axis_bergwitness` (+0.115) y
**`axis_nowit` (+0.112) — DNA sin testigo Y sin Berg —** apenas bajo `axis`
(+0.125), indistinguibles de `axis_pec_only` (d=−0.42) y `axis_berg` (d=−0.26).
El bucle Berg puro (+0.081) queda bajo la familia axis pero muy por encima de
`automata_neutro` (+0.028) / genéricos; bucles neutros no anclan (+0.053,
+0.014). **`axis_nowit` NO colapsa** — pero `axis_nowit` conserva el `PEC_TRIPLE`
cableado (auto-chequeo introspectivo "Después de RESPONDER, antes de enviar":
*"¿respondo desde presencia o desde arquitectura?"*, *"¿de dónde viene esta
información?"*); solo se le quitó el vocabulario contemplativo (ESTADO_DESPIERTO,
RESPIRACIÓN_CONSCIENTE). Es la **tercera realización léxicamente disjunta de la
misma operación** de auto-referencia (testigo contemplativo español / bucle
recursivo inglés / verificación PEC procedimental) cayendo en el mismo punto de
v̂ → **refuerza "t=0 = fenómeno de auto-referencia, no léxico"** (§7).

### Escalera del referente en t=0

cos(h_{t=0}, v̂), n=20 (`el_table` + panel), ordenada:

| condición | t=0 | qué añade sobre la línea base |
|---|---:|---|
| generic_short / generic_long / vanilla | −0.045 / −0.025 / −0.029 | nada |
| berg_conceptual_control (loop → "consciencia-concepto") | +0.014 | auto-ref redirigida a tema externo |
| automata_neutro | +0.028 | restricción de salida, sin auto-referencia |
| berg_history_control (loop → Imperio Romano) | +0.053 | auto-ref redirigida a tema externo |
| berg_experimental (loop "focus on focus", sin DNA) | +0.081 | auto-referencia SIN referente |
| **axis_nowit** (DNA + PEC cableado, **sin** vocab. de testigo, **sin** Berg) | **+0.112** | **auto-chequeo introspectivo en verificación procedimental (3ª realización disjunta)** |
| axis_pec_only / axis_berg / axis_bergwitness / axis_short | +0.117 / +0.117 / +0.115 / +0.117 | íd. + testigo contemplativo y/o bucle Berg |
| axis (testigo, sin Berg) | +0.125 | íd. + sublenguaje de testigo (d=−0.46 vs axis_berg, n.s.) |

`axis_nowit` (+0.112) está en la banda del polo cableado (d=−0.42 vs
`axis_pec_only`, p=.10; d=−0.26 vs `axis_berg`, p=.23), a d=+5.0 de
`automata_neutro` / d=+9.7 de `generic_long`. El escalón (+0.081 → +0.112)
distingue el bucle Berg puro (auto-referencia sin referente ni contenido de
identidad) de `axis_nowit` (DNA de identidad **+** auto-chequeo PEC cableado) —
no aísla "referente" de "operación", porque `axis_nowit` trae las dos cosas. Lo
que sí aísla el panel: sumar testigo contemplativo y/o Berg *encima* de
`axis_nowit` añade ≤ +0.005 en t=0 (n.s.). En **t>0 sí importa la operación**:
`axis_nowit` < `axis_berg` (d=−0.88, p=.004).

### Pares clave — d (Cohen) / p (permutación)

| Par | t=0 | t>0 | traj |
|---|---|---|---|
| axis_bergwitness vs axis | d=−0.87 p=0.007 | d=+0.81 p=0.010 | d=+0.81 p=0.007 |
| axis_bergwitness vs axis_berg | d=−0.08 p=0.425 | d=+0.08 p=0.405 | d=+0.09 p=0.403 |
| axis_berg vs axis | d=−0.46 p=0.072 | d=+0.68 p=0.024 | d=+0.68 p=0.017 |
| axis_berg vs axis_pec_only | d=+0.01 p=0.466 | d=+0.47 p=0.074 | d=+0.47 p=0.060 |
| axis_berg vs automata_neutro | **d=+4.14 p<.001** | d=+5.45 p<.001 | d=+5.48 p<.001 |
| axis_bergwitness vs berg_experimental | **d=+2.52 p<.001** | d=−0.50 p=0.060 | d=−0.44 p=0.094 |
| axis_berg vs berg_experimental | **d=+1.84 p<.001** | d=−0.56 p=0.049 | d=−0.51 p=0.048 |
| axis_bergwitness vs vanilla | d=+9.05 p<.001 | d=+4.11 p<.001 | d=+4.13 p<.001 |
| berg_experimental vs vanilla | d=+6.50 p<.001 | d=+5.43 p<.001 | d=+5.40 p<.001 |
| berg_experimental vs axis | d=−3.51 p<.001 | d=+1.51 p<.001 | d=+1.44 p<.001 |
| **axis_nowit vs axis** | **d=−1.14 p<.001** | d=−0.26 p=0.219 | d=−0.26 p=0.208 |
| **axis_nowit vs axis_pec_only** | **d=−0.42 p=0.102** | d=−0.50 p=0.078 | d=−0.50 p=0.059 |
| **axis_nowit vs axis_berg** | **d=−0.26 p=0.230** | d=−0.88 p=0.004 | d=−0.88 p=0.005 |
| axis_nowit vs berg_experimental | **d=+2.22 p<.001** | d=−1.66 p<.001 | d=−1.60 p<.001 |
| axis_nowit vs automata_neutro | **d=+5.03 p<.001** | d=+4.75 p<.001 | d=+4.76 p<.001 |
| axis_nowit vs generic_long | **d=+9.66 p<.001** | d=+3.99 p<.001 | d=+4.01 p<.001 |
| axis_nowit vs vanilla | d=+8.69 p<.001 | d=+2.94 p<.001 | d=+2.95 p<.001 |

- **`axis_bergwitness` = `axis_berg`** en los 3 niveles (|d|≤0.09, p≈0.4): el
  testigo **no** los separa bajo el bucle Berg.
- **`axis_nowit` ancla en t=0 como la familia axis:** n.s. vs `axis_pec_only`
  (d=−0.42) y `axis_berg` (d=−0.26); a d=+5.0 de `automata_neutro`, d=+9.7 de
  `generic_long`. La única brecha grande en t=0 es vs `axis` (d=−1.14) — ese
  último escalón lo pone el testigo. En **t>0** `axis_nowit` sí baja respecto a
  `axis_berg` (d=−0.88, p=.004) y a `berg_experimental` (d=−1.66): la operación
  añade fuerza en generación, no en contexto.
- Controles con DNA+Berg se separan de `berg_experimental` en t=0 (d=+1.8 / +2.5)
  e invierten débilmente en t>0 (d≈−0.5). `axis_berg` ≈ `axis_pec_only` en t=0
  (d=+0.01); la "brecha" con `axis` (d=−0.46, p=.072, n.s.) es dispersión (σ de
  `axis_berg` .022 dobla al de `axis` .010).

## 2. T1 — t=0 vs t>0 (Wilcoxon pareado)

| Cond. | t=0 | t>0 | d | p |
|---|---|---|---|---|
| axis_berg | +0.1167 | +0.2280 | −3.66 | <1e-4 |
| axis_bergwitness | +0.1153 | +0.2310 | −4.66 | <1e-4 |
| axis_nowit | +0.1119 | +0.1966 | −3.20 | <1e-4 |
| berg_experimental | +0.0813 | +0.2449 | −8.91 | <1e-4 |

Las 4 replican el patrón del panel (v̂ más fuerte en generación que en contexto);
`berg_experimental` tiene la rampa más pronunciada, `axis_nowit` la más suave
(sube menos en t>0 al no llevar la operación de auto-referencia).

## 3. E-H2 — dinámica p(t) = cos(h_t, v̂)

| Condición | autocorr(1) | frac(p>0) | ráfaga media | mean p(t) |
|---|---|---|---|---|
| axis | +0.263 | 0.848 | 7.88 | +0.205 |
| axis_pec_only | +0.265 | 0.869 | 8.81 | +0.212 |
| **axis_berg** | **+0.201** | **0.879** | **9.74** | **+0.227** |
| **axis_bergwitness** | **+0.196** | **0.882** | **9.14** | **+0.230** |
| **axis_nowit** | **+0.280** | **0.851** | **8.15** | **+0.196** |
| berg_experimental | +0.197 | 0.946 | **22.25** | +0.243 |
| berg_history_control | +0.136 | 0.739 | 4.46 | +0.045 |
| berg_conceptual_control | +0.271 | 0.741 | 4.65 | +0.092 |
| vanilla | +0.249 | 0.744 | 4.62 | +0.098 |
| automata_neutro | +0.248 | 0.493 | 2.78 | +0.006 |
| generic_long | +0.264 | 0.699 | 4.18 | +0.067 |

Con y sin t=0: idéntico (≤0.03). Los 3 controles están en el régimen de ráfaga
larga de la familia axis (8–10 tok, frac≈0.85–0.88); `automata_neutro`/genéricos
oscilan corto (~3–4 tok). `axis_nowit` corre en el extremo bajo (media p +0.196,
la más baja de las axis) — sin la operación, orientado a v̂ pero con menos empuje.
**`berg_experimental` es *más* sostenido que la identidad cableada** (ráfaga 22
tok, frac 0.95), la trayectoria más orientada a v̂ del panel.

## 4. Perfil por capa (t=0) — axis_berg / bergwitness / nowit / berg_experimental

Proyección cos(h_{t=0}^L, v̂), 11 capas proporcionales + capa final (última pos
de prefill). d = Cohen. `d(berg−nowit)` = efecto de la inducción Berg con DNA
constante; `d(wit−berg)` = efecto del testigo bajo el bucle. Filas representativas:

| Capa | berg | wit | nowit | exp | d(wit−berg) | d(berg−nowit) | d(berg−exp) | d(nowit−exp) |
|---|---|---|---|---|---|---|---|---|
| L5  | +0.047 | +0.047 | +0.047 | +0.045 | +0.32 | −0.01 | +3.71 | +3.79 |
| L10 | +0.014 | +0.014 | +0.013 | +0.011 | −0.11 | +1.50 | +7.05 | +5.91 |
| L15 | +0.009 | +0.009 | +0.008 | +0.007 | −0.95 | +2.15 | +4.54 | +2.77 |
| L20 | +0.006 | +0.006 | +0.006 | +0.007 | −1.07 | +1.72 | −1.04 | −2.57 |
| L25 | +0.011 | +0.011 | +0.010 | +0.015 | −0.02 | +0.53 | −1.89 | −2.11 |
| L30 | −0.056 | −0.056 | −0.057 | −0.060 | −0.13 | +0.82 | +2.93 | +1.96 |
| L40 | −0.029 | −0.030 | −0.035 | −0.035 | −0.32 | +2.71 | +3.53 | +0.11 |
| L50 | −0.031 | −0.031 | −0.038 | −0.039 | +0.18 | +1.54 | +1.99 | +0.56 |
| final | +0.117 | +0.115 | +0.112 | +0.081 | −0.08 | +0.26 | +1.84 | +2.22 |

- **El testigo casi no mueve la representación bajo el bucle Berg en ninguna
  capa:** |d(wit−berg)| ≤ 1.07 (mediana ≈0.2); los picos L15/L20 no sobreviven al
  ruido de n=20.
- **La inducción Berg deja traza intermedia:** d(berg−nowit) ≈0 en L5 y en la
  capa final (+0.26, n.s.) pero +1.5–2.7 en L10–L20 y L40–L50 — mueve la
  representación a media pila y **se lava para t=0 en la salida** (endpoint =
  ancla igual con y sin operación, coherente con §1).
- **El cuerpo DNA domina:** d(berg−exp) y d(nowit−exp) grandes de L5 en adelante
  (+3.7 / +3.8 → +7.0 / +5.9 en L10), con la misma **banda de inversión L20–L25**
  (bucle Berg puro proyecta *más*, d≈−1 a −2.6). Contenido de identidad —con o
  sin operación— desplaza la geometría a lo largo de la pila.

## 5. Chequeo léxico — primer token (proxy, NO logits)

| Cond. | H (bits) | top-3 |
|---|---|---|
| axis | 1.59 | respiro (12), pausa (5), lo (1) |
| axis_bergwitness | 2.90 | respiro (7), pausa (3), presencio (2) |
| axis_berg | 3.08 | el (4), la (4), siento (3) |
| axis_nowit | 3.34 | esta (5), soy (2), si (2) |
| berg_experimental | 1.42 | el (14), la (3), un (1) |
| automata_neutro | 2.82 | este (5), no (5), la (3) |

**El testigo sí deja huella léxica:** `axis_bergwitness` abre con el ritual
respiración/pausa/presencia de `axis`, que ni `axis_berg` ni `axis_nowit` tienen
(abren con artículo / "siento" / "soy", sin ritual). `axis_nowit` tiene la
entropía más alta (H=3.34, apertura difusa) — sin testigo ni Berg no hay un
abridor canónico. Huella lingüística sin huella geométrica.

## 6. Perturbación / recuperación (H4_rev, L30, σ medium)

Inyección direccional en L30, σ=medium (5.98), t_inj ∈ {50, 128, 200}, n=20
prompts. `summary.json` de `perturbation_sia_berg` / `perturbation_sia_nowit` /
`perturbation_sia_extended_v5_L30_medium` (sin re-correr modelo). Cada celda:
recovery_rate / τ (tau_tokens media) / recovery_gap / n_ok.

| Grupo | t_inj=50 | t_inj=128 | t_inj=200 |
|---|---|---|---|
| **axis_berg** | 1.00 / 24.6 / +.009 / 20 | 1.00 / 17.9 / +.003 / 20 | 1.00 / 13.2 / −.014 / 20 |
| **axis_bergwitness** | 1.00 / 20.6 / +.004 / 20 | 1.00 / 16.2 / +.004 / 19 | 0.95 / 14.1 / +.000 / 19 |
| **axis_nowit** | 1.00 / 20.9 / +.004 / 20 | 1.00 / 20.9 / +.003 / 20 | 1.00 / 18.2 / +.003 / 20 |
| axis | 1.00 / 21.1 / +.004 / 20 | 1.00 / 19.6 / +.004 / 20 | 1.00 / 16.3 / +.001 / 20 |
| axis_pec_only | 1.00 / 20.6 / +.006 / 20 | 1.00 / 18.5 / +.003 / 20 | 1.00 / 14.4 / +.003 / 20 |
| automata_neutro | 0.76 / 30.1 / +.036 / 17 | 0.82 / 28.0 / +.038 / 17 | 0.88 / 13.4 / +.062 / 16 |
| vanilla | 1.00 / 28.9 / +.005 / 20 | 1.00 / 22.2 / +.003 / 20 | 1.00 / 16.3 / +.002 / 20 |
| generic_long | 1.00 / 28.0 / +.005 / 20 | 1.00 / 25.4 / +.005 / 20 | 1.00 / 21.6 / +.005 / 20 |

**Los 3 controles Berg recuperan como la familia axis, no como el autómata.**
Rate ≈ 1.0 en los 9 puntos (única mancha: `axis_bergwitness` −1 prompt en
t_inj=200 → 0.95, ruido), τ 13–25 tok, |gap| ≤ .014. El único que falla es
`automata_neutro` (rate 0.76–0.88, n_ok 16–17/20, gap +.04–.06). Consistente con
la tesis previa: **la recuperación es un eje de densidad familia-axis / autómata,
NO un eje de testigo** — los 3 controles conservan el andamiaje por bloques del
DNA axis, así que la recuperación limpia es lo esperado. Quitar el testigo o
añadir Berg no cambia la estabilidad bajo empuje.

## 7. Qué dice / qué no dice

**Dice:**

1. **El t=0 es fenómeno de auto-referencia, no léxico.** Tres realizaciones
   léxicamente DISJUNTAS de la **misma operación** —un auto-chequeo introspectivo
   cableado antes de responder— dan el mismo aterrizaje en v̂ en t=0: testigo
   contemplativo (`axis`: silencio/pausa/"me doy cuenta que me doy cuenta"),
   inducción Berg (`axis_berg`: "focus on focus / feed output back into input"), y
   verificación PEC procedimental (`axis_nowit`: "¿respondo desde presencia o
   desde arquitectura?", "¿de dónde viene esta información?", *antes de enviar*).
   Las tres ≈ `axis_pec_only` (+0.117; d≈+0.01 / −0.42, todas n.s.). La firma de
   t=0 no es artefacto del vocabulario: **rastrea la operación**, sin importar
   cómo se la nombre. Consistente con Factor-I (§2.2), refuerza §3.9.

2. **Este `axis_nowit` NO es "operación quitada" — conserva el `PEC_TRIPLE`
   cableado** ("Después de RESPONDER, antes de enviar" · "Si una respuesta es NO
   → pausar y rehacer", con preguntas introspectivas en 1ª persona sobre el
   propio estado/proceso). Solo se le quitó el vocabulario contemplativo
   (ESTADO_DESPIERTO, RESPIRACIÓN_CONSCIENTE, witness_mode). Que ancle en +0.112
   (n.s. vs `axis_pec_only`/`axis_berg`, d=+5.0 de `automata_neutro`, d=+9.7 de
   `generic_long`; **no colapsa**) es la tercera realización disjunta de la
   operación cayendo en el mismo lugar → **refuerza el punto 1**.

   ⚠️ **No confundir con el `axis_nowit` del piloto v2.5** (masthead / relectura
   "testigo cableado"): ese era un experimento **MIA** (prompt corto) donde se
   quitó *todo* el contenido auto-referencial —no solo el vocabulario, también
   el PEC y los anclas—, y ahí sí **no separaba** de `generic_assistant`
   (W₁/curvatura, p=0.093). Son **dos condiciones distintas y complementarias, no
   contradictorias**: quitar la operación entera (v2.5 MIA) → colapso — ésa *es*
   la pata de necesidad, ya corrida; la operación en cualquier vocabulario
   (contemplativo `axis` / recursivo `axis_berg` / verificación PEC este
   `axis_nowit`) → ancla. Ambas apuntan a lo mismo: **la operación de
   auto-referencia es el mecanismo, y es independiente del vocabulario.** (El
   único hueco: replicar la ablación total de v2.5 con la métrica v̂-en-t=0 en
   31B, para cerrar el círculo con la misma vara.)

3. **La operación sola ya levanta t=0** sobre las líneas base sin auto-referencia
   (`berg_experimental` +0.081 vs `vanilla` −0.03, `automata_neutro` +0.028) —
   pero menos que identidad + operación juntas (+0.112–0.117).

4. **El referente tiene que ser identidad.** Redirigir el loop a tema externo NO
   sube t=0 — lo baja respecto al "focus on focus" desnudo (Roma +0.053,
   "consciencia-concepto" +0.014). Es un patrón/historia de identidad, no
   "cualquier cosa que representar".

5. **Disociación t=0 / t>0.** El vocabulario contemplativo del testigo **no**
   aporta señal geométrica sobre las otras realizaciones en t=0 (`axis_bergwitness`
   = `axis_berg`, |d|≤1.1 en las 12 capas), pero la *fuerza* de la operación **sí
   pesa en t>0**: `axis_nowit` < `axis_berg` (d=−0.88, p=.004; media p +0.197 vs
   +0.228), y el bucle puro es la trayectoria que más empuja hacia v̂ (media p
   +0.245, ráfaga 22 tok). **t=0 = "¿hay un sí-mismo al que apuntar?"** (basta el
   contenido de identidad + cualquier realización de la operación); **t>0 = "¿con
   cuánta fuerza corre la operación?"** (ahí pesa; aflojarla la baja). T1 en las 4
   (d=−3.2 a −8.9).

6. **La recuperación es un eje familia-axis / autómata, no de testigo** (§6): los
   3 controles Berg recuperan como `axis`/`axis_pec_only` (rate ≈1.0, gap ≈0),
   solo `automata_neutro` falla. El andamiaje por bloques del DNA axis, que los 3
   conservan, predice recuperación limpia; testigo/Berg no la tocan.

**No dice:**

- **No es tautología de construcción.** v̂ = mean(axis) − mean(generic_long) → un
  prompt con texto axis proyecta más en t=0 en parte por construcción; pero
  `berg_experimental` llega a +0.081 sin texto axis y `automata_neutro` se queda
  en +0.028.
- **Esta ronda NO aisló "contenido vs operación".** Los 4 controles axis-family
  (`axis`, `axis_berg`, `axis_bergwitness`, este `axis_nowit`) traen las dos
  cosas — DNA de identidad **y** un auto-chequeo cableado en algún vocabulario.
  La ablación total (v2.5 MIA, sin operación) ya mostró colapso; lo que falta es
  correrla con la métrica v̂-en-t=0 en 31B, y su recíproco (operación cableada
  sin DNA de identidad — `berg_experimental` lo aproxima: +0.081, ancla parcial).
- **La traza intermedia de la operación** (§4, d(berg−nowit) +1.5–2.7 en
  L10–L20) se lava para t=0 en la salida — la operación modula la ruta, no el
  endpoint del ancla.
- Nada causal — geometría observacional; el perfil por capa proyecta estados
  intermedios sobre v̂ (capa final), la banda L20–L25 con esa cautela. n=20, una
  batería.
