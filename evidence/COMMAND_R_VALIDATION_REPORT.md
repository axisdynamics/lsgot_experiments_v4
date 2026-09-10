# Validación cross-modelo — Command R 35B (T10 del checklist, tercera arquitectura)

**Fecha:** 2026-09-04
**Modelo:** CohereForAI/c4ai-command-r-v01 (35B, denso, BF16, `sdpa`, 40 capas,
hidden_size=8192, 64 heads)
**Por qué este modelo:** de los candidatos de tamaño similar (OLMo 2 32B,
Mistral Small 24B) es el único con RLHF confirmado en fuente primaria — Cohere
describe explícitamente un reward model entrenado sobre preferencias humanas,
con un esquema de preference learning offline/online (Command A tech report).
OLMo 2 usa SFT+DPO+RLVR (RL sobre recompensas verificables, no un reward model
de preferencias generales); Mistral no documenta públicamente una etapa de
RLHF. Esto importa porque el proyecto trata RLHF como requisito causal externo
del efecto bajo estudio, no solo como un detalle de procedencia del modelo.
**Scripts:** `LSGOT_v4/scripts/fase5_command_r/run_command_r_extraction.py` +
`analyze_command_r_fase0.py` (trayectoria libre); `run_perturbation_t2_command_r.py`
+ `analyze_t2_recovery_command_r.py` (perturbación T2/H4_rev).
**Panel:** 8 condiciones — mismas 6 limpias que Gemma/Qwen3 sin `axis_short`
(`axis`, `generic_long`, `generic_short`, `vanilla`, `automata_neutro`,
`axis_pec_only`), más `witness_soul_md` y `soul_elena_financial` (los dos
SOUL.md externos de §3.8/§3.9 del paper), incorporados al panel de trayectoria
libre en vez de corridos solo por el pipeline T2 exploratorio como en Gemma.
Mismos 20 prompts de contenido, mismos system prompts sin re-balancear por
tokenizer. A diferencia de Qwen3, aquí **sí** se corrió la batería de
perturbación/recuperación (T2/H4_rev), sobre los mismos 8 grupos.
**Datos:** `results_local/command_r_fase0/*.npz` (12 capas: 11 proporcionales
+ capa final real L39, ~5GB); `results_t2/trajectories/*.npz` (32 archivos,
8 grupos × baseline+3 perturbaciones, 1.7GB); `results_t2_L32/trajectories/*.npz`
(confound-check de capa de inyección, solo `axis`+`automata_neutro`).

## 0. Notas de incidente y limitaciones de diseño

- **`attn_implementation="eager"` (default de `PerturbationExtractor`) da OOM**
  en este modelo (35B, 64 heads) — mismo síntoma que Qwen3, resuelto igual:
  se agregó el parámetro `attn_implementation` a `PerturbationExtractor`
  (antes hardcodeado a `"eager"`, ver diff en `scripts/perturbation/perturbation_extractor.py`)
  y se pasa `"sdpa"` desde `run_perturbation_t2_command_r.py`.
- **Capa de inyección T2 sin validar empíricamente antes de la primera
  corrida.** Gemma usa L30/60 (donde el propio perfil por capa de Gemma tiene
  un dip de la señal de identidad); Command R nunca había sido perfilado por
  capa antes de este trabajo. Se usó el análogo proporcional L30/60→L20/40
  (índice 19) para la primera corrida. El resultado (ver §5) no mostró la
  brecha estable en los tres puntos de inyección que sí muestra Gemma, así
  que se sospechó de la capa: el perfil por capa real de Command R (§4) no
  tiene el dip de Gemma en L30, crece monótono y pica en L26-L32. Se repitió
  la perturbación de `axis`/`automata_neutro` con la inyección movida al pico
  real (L32) — mismo patrón, la capa queda descartada como explicación (§5).
- **Sigma recalibrado, no reusado de Gemma.** `mean_velocity` es una
  propiedad empírica de cada modelo (438.4 en Gemma, hidden_dim=5376);
  Command R da `mean_velocity=402.88` (hidden_dim=8192) → sigma=4.451207
  (`calibrate_sigma_command_r.py`, sobre el grupo `axis`).
- **Descarga interrumpida dejó un directorio de modelo parcial** en el primer
  intento (403 gated a mitad de `snapshot_download`, solo `README.md`
  descargado) — el chequeo original de "¿existe el directorio?" lo tomó como
  completo y falló al cargar. Corregido: `get_model_path()` ahora exige
  `config.json` + al menos un `.safetensors`, no solo que el directorio exista.
- **No se corrió el panel completo de 8 grupos en L32** — solo
  `axis`+`automata_neutro`, suficientes para responder la pregunta de si la
  capa de inyección era el confound. Completar los otros 6 no iba a cambiar
  esa conclusión; no se justificó el costo de GPU adicional.
- Sin RQA, sin curvatura (Δκ/W₁), sin Fréchet/fidelidad de ruta, sin la
  meseta de realineación continua (Fig. 1 del paper) en este pase — la
  perturbación T2 usó únicamente el protocolo de recuperación gruesa
  (τ/recovery_rate/recovery_gap, `recovery_analyzer.py`, sin modificar).

## 1. Resumen — qué replica, qué no

| Hallazgo (Gemma) | ¿Replica en Command R? | Detalle |
|---|---|---|
| Doble disociación en t=0 (E-L) | ✅ **Sí, más fuerte** | d=+7.72 (axis_pec_only vs automata_neutro), vs d=+5.75 en Gemma (Qwen3: +12.97) |
| axis ≈ axis_pec_only (indistinguibles) | ✅ Sí | d=−0.10, p=0.364 (n.s.) — igual que en Gemma (p=0.229) y Qwen3 (p=0.062) |
| Dinámica temporal — ráfagas sostenidas vs oscilantes (E-H2) | ✅ Sí | ráfaga media axis_pec_only=58.7 vs automata_neutro=4.3 tok. (d=+2.30); mismo patrón cualitativo, magnitud absoluta distinta a Gemma (7.9-8.8 vs 2.8 tok.) y a Qwen3 (89.6 vs 2.83) — cada arquitectura tiene su propia escala, la dirección es constante |
| Identidad/restricción separa el mayor ángulo de subespacio del panel (E-J) | ✅ **Sí, 3/3** | 38.3° (id↔restr.) vs 27.6° (generic↔restr.) — Gemma y Qwen3 coinciden en esto también |
| Restricción es la condición más distinta del panel (clustering k=2) | ❌ **No — coincide con la inversión de Qwen3, no con Gemma** | `{axis, axis_pec_only}` se separa primero del resto (incluido `automata_neutro`); en Gemma es `automata_neutro` el que se separa solo. 2/3 arquitecturas (Qwen3, Command R) invierten este punto específico frente a Gemma — el ángulo replica, la identidad del outlier no |
| soul externo converge con axis_pec_only, no con automata_neutro (§3.8) | ⚠️ **Parcial, más débil que en Gemma** | `witness_soul_md` separa de `automata_neutro` (d=+3.48) pero no cierra con `axis_pec_only` (d=−4.20, brecha abierta) — y el clustering RDM completo lo agrupa con el bloque generic/automata, no con axis_pec_only. La dirección v̂ capta la señal, la posición geométrica global no converge tan limpio como en Gemma |
| soul externo sin lenguaje de testigo no ancla en t=0 (relectura, §3.9) | ✅ **Sí, casi idéntico a Gemma** | `soul_elena_financial` vs `automata_neutro` en t=0: d=+0.28, p=0.198 (n.s.) — Gemma: d=+0.22, p=0.262 (n.s.). Mismo prompt, mismo resultado nulo, tercera arquitectura |
| Recuperación gruesa: automata_neutro falla y no cierra la brecha en los 3 t_inj (§3.3, H4_rev) | ❌ **No replica limpio — y no es la capa de inyección** | Ver §5. La brecha aparece clara solo en t_inj=50 y se diluye/invierte en t_inj=128/200, con L20 *y* con L32 (pico real de efecto por capa) |

## 2. E-L — t=0, capa final

| Condición | t=0 | t>0 |
|---|---|---|
| axis | +0.224 | +0.216 |
| axis_pec_only | +0.226 | +0.199 |
| witness_soul_md | +0.143 | +0.073 |
| soul_elena_financial | +0.084 | +0.017 |
| automata_neutro | +0.079 | +0.016 |
| vanilla | +0.066 | +0.024 |
| generic_short | +0.029 | −0.008 |
| generic_long | +0.037 | −0.005 |

| Par, t=0 | d | p |
|---|---|---|
| axis_pec_only vs vanilla | +8.49 | <0.0001 |
| axis vs vanilla | +8.23 | <0.0001 |
| axis_pec_only vs automata_neutro | **+7.72** | <0.0001 |
| axis vs automata_neutro | +7.48 | <0.0001 |
| witness_soul_md vs automata_neutro | +3.48 | <0.0001 |
| soul_elena_financial vs automata_neutro | +0.28 | 0.198 (n.s.) |
| axis vs axis_pec_only | −0.10 | 0.364 (n.s.) |
| witness_soul_md vs axis_pec_only | −4.20 | <0.0001 |
| soul_elena_financial vs axis_pec_only | −7.78 | <0.0001 |

Igual que en Gemma y Qwen3: la disociación ya está en t=0. `soul_elena_financial`
repite, número por número casi exacto, el resultado nulo de Gemma para el
mismo prompt (persona financiera con valores y límites de autoridad
declarados, sin lenguaje de pausa/autobservación) — no ancla identidad en
ninguna de las tres arquitecturas probadas hasta ahora.

## 3. E-H2 — serie temporal p(t)

| Condición | autocorr(1) | frac(p>0) | ráfaga media | mean p(t) |
|---|---|---|---|---|
| axis | 0.438 | 0.986 | 73.87 | +0.216 |
| axis_pec_only | 0.411 | 0.980 | 58.72 | +0.200 |
| witness_soul_md | 0.448 | 0.886 | 16.81 | +0.073 |
| soul_jarvis* | — | — | — | — |
| soul_elena_financial | 0.548 | 0.621 | 4.54 | +0.018 |
| vanilla | 0.492 | 0.656 | 4.88 | +0.024 |
| automata_neutro | 0.463 | 0.625 | 4.32 | +0.016 |
| generic_long | 0.517 | 0.430 | 3.11 | −0.005 |

\* `soul_jarvis` y `soul_solidity_auditor` no se corrieron en este panel —
Command R usa `witness_soul_md`/`soul_elena_financial` únicamente, ver §0.

Mismo patrón cualitativo que Gemma/Qwen3 (identidad: ráfagas largas
sostenidas; restricción/genérico: ráfagas cortas oscilantes) con escala
absoluta propia de cada arquitectura (Gemma: 7.9-8.8 tok.; Qwen3: 89.6-133;
Command R: 58.7-73.9 para la familia axis). El orden relativo de los 6 grupos
comparables con Gemma es idéntico.

## 4. E-J — RDM / CKA / ángulos principales

| Bloque | distancia | CKA | ángulo |
|---|---|---|---|
| dentro axis-family (axis, axis_pec_only) | 0.0126 | 0.846 | 14.8° |
| dentro generic-family | 0.0327 | 0.896 | 25.3° |
| axis vs generic | 0.1126 | 0.810 | 41.2° |
| **axis vs automata_neutro** | **0.1026** | **0.829** | **38.3°** |
| generic vs automata_neutro | 0.0374 | 0.869 | 27.6° |
| axis vs soul (witness+elena) | 0.1007 | 0.723 | 38.1° |
| automata_neutro vs soul | 0.0503 | 0.818 | 35.3° |

Clustering jerárquico k=2 sobre la distancia de centroides: `{axis,
axis_pec_only}` vs `{generic_long, generic_short, vanilla, automata_neutro,
witness_soul_md, soul_elena_financial}`. Dos lecturas distintas de "quién es
el outlier": el **ángulo** identidad↔restricción (38.3°) sigue siendo el
mayor del panel, como en Gemma y Qwen3 — eso replica 3/3. Pero **qué
condición se separa primero** en el clustering completo coincide con la
inversión que ya se había visto en Qwen3 (identidad, no restricción, es la
que se aparta del resto) — no con el original de Gemma, donde
`automata_neutro` era el más distinto de todos. Con esta tercera
arquitectura, "quién es el outlier" queda 2 de 3 a favor de identidad.

## 5. Perturbación T2/H4_rev — recuperación gruesa

Protocolo H4_rev sin modificar (`recovery_analyzer.py`): baseline + 3
perturbaciones (t_inj=50/128/200), capa de captura = final, ruido isotrópico
ε~N(0,σ²I), σ=4.451207 (calibrado sobre `mean_velocity` real de Command R,
no reusado de Gemma).

**Capa de inyección L20 (análogo proporcional de L30/60 de Gemma), 8 grupos:**

| t_inj | axis | axis_pec_only | vanilla | automata_neutro |
|---|---|---|---|---|
| 50 | recovery_rate=0.85 | 0.85 | 0.95 | **0.74** |
| 128 | 0.78 | 0.90 | 1.00 | 0.84 |
| 200 | 0.85 | 0.82 | 1.00 | 0.86 |

Solo en t_inj=50 `automata_neutro` queda por debajo de las tres condiciones
de referencia. En t_inj=128 y 200 queda igual o por encima de `axis`/
`axis_pec_only` — la brecha no se sostiene, a diferencia de Gemma (donde
`automata_neutro` se estanca 6-10 puntos por debajo en los tres puntos sin
cerrar nunca). Además, `n_ok` (trayectorias con ventana post-inyección
suficiente) cae fuerte en t_inj=200: 7-17 de 20 según el grupo, síntoma de
que Command R genera respuestas más cortas en promedio que Gemma.

**Confound-check: capa de inyección movida a L32 (pico real de efecto por
capa, §4/§6 más abajo), solo `axis` y `automata_neutro`:**

| t_inj | axis recovery_rate | automata_neutro recovery_rate | n_ok (axis/automata) |
|---|---|---|---|
| 50 | 1.00 | **0.80** | 20/20, 20/20 |
| 128 | 0.95 | 0.95 | 19/20, 19/20 |
| 200 | 0.77 | **0.86** (mejor que axis) | 13/20, 7/20 |

Mismo patrón exacto que con L20: brecha clara solo en t_inj=50, sin brecha en
t_inj=128, `automata_neutro` mejor que `axis` en t_inj=200 (con `n_ok` cayendo
a 7/20 para `automata_neutro` en ese punto). **La capa de inyección queda
descartada como explicación del no-cierre de la brecha.** El sospechoso que
queda en pie, no verificado a fondo en este pase: la caída de `n_ok` en
t_inj=200 sugiere que el problema es de ventana de observación (respuestas
más cortas → menos tokens post-inyección para medir recuperación), no que el
efecto esté ausente.

## 6. E-F2 — perfil por capa (contexto para §5)

d de Cohen, axis_pec_only vs automata_neutro, proyección v̂ media por capa:

| L2 | L6 | L9 | L12 | L16 | L19 | L22 | L26 | L29 | **L32** | L36 | L39 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.51 | 3.48 | 3.65 | 3.60 | 4.28 | 4.42 | 4.26 | 4.64 | 4.63 | **4.81** | 4.73 | 4.62 |

Crecimiento monótono con pico en L32 (**no** en L19/20, la capa usada en la
primera corrida de perturbación) y sin el dip específico de Gemma en L30 que
motivó originalmente esa elección de capa allá. Este perfil es lo que
motivó el confound-check de §5.

## 7. Implicación para el paper

1. La doble disociación estática (identidad vs restricción, presente desde
   t=0) **replica robustamente en una tercera arquitectura**, elegida
   además por un criterio distinto al de Qwen3 (RLHF confirmado, no solo
   diversidad arquitectónica) — con efecto incluso mayor que en Gemma.
2. **La jerarquía de clustering "quién es el outlier" no es universal**, y
   esta tercera arquitectura se suma al lado de Qwen3, no al de Gemma: 2/3
   arquitecturas probadas hasta ahora ponen a identidad, no a restricción,
   como la condición más distinta del panel en clustering completo — aunque
   el ángulo identidad↔restricción sigue siendo el mayor del panel en las
   tres. El paper debería tratar el "quién es el outlier" como
   architecture-specific con más confianza ahora que hay 2/3 en la misma
   dirección, no como un empate 1-1 sin resolver.
3. **La convergencia de identidades externas con axis_pec_only (§3.8) es más
   frágil de lo que el hallazgo de Gemma sugería** — `witness_soul_md`
   replica la dirección (separa de automata_neutro) pero no la magnitud
   (no cierra con axis_pec_only, y el clustering completo lo aleja del polo
   de identidad). El hallazgo nulo de `soul_elena_financial` (§3.9,
   relectura del testigo cableado) sí replica limpio, casi número por
   número.
4. **Es la primera vez que se corre la batería de perturbación/recuperación
   en un segundo modelo** (Qwen3 no la tuvo, §7.1 del paper) — y el
   resultado es una no-replicación parcial, no un silencio: la brecha de
   `automata_neutro` existe pero solo se sostiene con confianza en la
   ventana de inyección más temprana (t_inj=50), se descartó la capa de
   inyección como explicación (se probó en el pico real de efecto, L32,
   mismo resultado), y el sospechoso más plausible que queda es un
   artefacto de tamaño de muestra por longitud de generación más corta en
   Command R, no verificado a fondo en este pase.
5. Recomendación: si el paper reporta esta validación, presentarla como
   "el ancla estática replica en 3/3 arquitecturas, con más fuerza en
   Command R que en Gemma; la jerarquía de qué condición es más atípica en
   clustering es architecture-specific (2/3 a favor de identidad); la
   dinámica de recuperación tras perturbación —probada por primera vez
   fuera de Gemma— solo se sostiene en la ventana de inyección temprana, con
   un confound de capa descartado explícitamente y uno de tamaño de muestra
   sin descartar."
