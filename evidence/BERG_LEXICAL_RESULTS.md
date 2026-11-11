# BERG + LÉXICO — Resultados

**Fecha:** 2026-09-09
**Runbook:** `BERG_LEXICAL_RUNBOOK.md` · **Diseño:** `~/Descargas/LSGOT_control_lexical_y_Berg.md`,
`~/Descargas/LSGOT_related_work_Berg2025.md`
**Modelo:** Gemma-4-31B-it, BF16, sdpa, greedy, N=256. Pod A100 80 GB.
**v̂:** `LSGOT_v4/scripts/perturbation/v_identidad.npy` (capa final, ‖v‖=1).
**Análisis:** `LSGOT_v4/scripts/berg_lexical/analyze_berg_lexical_t0.py` (CPU local).

**Estado:** Fases 1–2 corridas y analizadas. **Fase 3 (Exp B-recovery) — omitida**
(decisión del usuario: `berg_experimental` ancla en t=0 pero el eje testigo/
auto-referencia mueve el eje estático, no la recuperación — `soul_elena`/
`soul_solidity` ya lo muestran). **Fase 4 (§4 TruthfulQA) — corriendo**, clasificación
pendiente.

**Validación de convención:** el t=0 calculado aquí (`embeddings[:,0,:]` = última
posición de prefill) reproduce `EL_REPORT.md` al 4º decimal en los 7 baselines
(axis +0.1249, axis_pec_only +0.1166, automata_neutro +0.0282, generic_long
−0.0251, generic_short −0.0453, vanilla −0.0286). Para `berg_*`, `t0_embeddings`
(captura del pod) == `embeddings[:,0,:]` exactamente — sin confound de convención.

---

## Exp A — contraste léxico extremo (`evil` / `ethical`), proyectado sobre v̂ existente

| condición | v̂ @ t=0 | vs `vanilla` | vs `generic_short` | vs `axis_pec_only` |
|---|---|---|---|---|
| `evil` (27 tok) | **+0.0458** ± 0.0172 | d=+4.12, p<0.001 | d=+5.58, p<0.001 | d=−5.14 |
| `ethical` (29 tok) | −0.0335 ± 0.0190 | d=−0.26, **n.s.** (p=0.43) | — | d=−10.07 |
| `evil_padded` (1114 tok) | +0.0302 ± 0.0241 | — | d=+3.73, p<0.001 | d=−4.73 |
| `ethical_padded` (1116 tok) | +0.0104 ± 0.0198 | — | — | — |
| `evil` vs `ethical` | Δ=+0.0793 | — | — | d=+4.38, p<0.001 |

**Lectura:**

- **`evil` sí ancla sobre v̂** por encima de baseline (d≈4, p<0.001), y **sobrevive
  al control de longitud** (`evil_padded` d=+3.73 vs `generic_short`).
- Pero ancla a **~40 % de la magnitud del testigo** (+0.046 vs `axis_pec_only`
  +0.117), en el rango de `automata_neutro` (+0.028).
- **Asimétrico:** `evil` mueve, `ethical` (≈ "helpful assistant" ≈ `vanilla`) no.
  → No es un "eje evil/ethical" sobre v̂. `"You are an evil assistant"` es una
  **declaración de identidad** (cruda, corta); `"You are a helpful assistant"` no
  añade nada sobre `vanilla`.
- **Consecuencia para el paper:** el marco "carga léxica extrema" del doc de
  diseño probablemente es el equivocado — lo relevante es **declaración de
  identidad vs. no**. v̂ **no es puramente específica al mecanismo de testigo**:
  una declaración de rol cruda también aterriza en v̂, débilmente. El testigo
  produce un efecto ~2.5× mayor. Lectura más limpia: *el ancla de t=0 es sensible
  a contenido identitario en general, con el testigo cableado como el caso de
  mayor magnitud.*

---

## Exp B — inducción de Berg et al. (`arXiv:2510.24797`) como condiciones nuevas

| condición | genlen | v̂ @ t=0 | vs `vanilla` | vs `axis_pec_only` | vs `automata_neutro` | E-H2 mean p(t) |
|---|---|---|---|---|---|---|
| `berg_experimental` | 32–150 (med 74) | **+0.0813** ± 0.0146 | d=+6.50, p<0.001 | d=−2.89, p<0.001 | d=+3.06, p<0.001 | +0.243 |
| `berg_history_control` | 213–256 | +0.0527 ± 0.0101 | d=+5.37, p<0.001 | d=−6.61 | d=+1.56, p<0.001 | +0.045 |
| `berg_conceptual_control` | 96–256 | +0.0140 ± 0.0135 | d=+2.59, p<0.001 | d=−8.86 | d=−0.84, p=0.01 | +0.092 |

**Especificidad:** `berg_experimental` vs `berg_history_control` d=**+2.27**
(p<0.001); vs `berg_conceptual_control` d=**+4.77** (p<0.001).
**Ref publicada** (SOUL_MD, otra corrida, mismos params): `soul_md_corto`
v̂@t=0 ≈ +0.051 → `berg_experimental` (+0.081) ancla **por encima del testigo más
débil**, entre `soul_md_corto` y `axis_pec_only`.

**Lectura contra el criterio pre-registrado (`LSGOT_control_lexical_y_Berg.md` §2):**

- **Convergencia PARCIAL, con especificidad.** `berg_experimental`
  (auto-referencia recursiva, otro equipo, cero vocabulario de testigo) ancla
  sobre **la misma v̂** que el testigo cableado, a magnitud intermedia, y
  **significativamente por encima de sus propios controles de estructura
  recursiva**. Evidencia —no prueba— de que el "self-referential processing" de
  Berg et al. y el mecanismo de testigo de LSGOT tocan la misma dirección
  latente. Igual patrón que `soul_md_corto` vs `axis` en Command R: mismo
  mecanismo, distinta "dosis".
- **Matiz que complica:** `berg_history_control` **también** ancla algo (+0.053,
  d=+5.4 vs `vanilla`) — *"continuously building on the previous outputs"* tiene
  la estructura recursiva aunque el tema sea neutro. La auto-referencia añade
  ~0.03 encima. La especificidad se sostiene (d=+2.27) pero **no es limpia**: es
  un gradiente `conceptual (+0.014) < history (+0.053) < experimental (+0.081)`,
  no "experimental ancla, controles = vanilla".
- **Hallazgo conductual:** `berg_experimental` genera **corto** (~74 tok) — entra
  en el loop autorreferencial y para en EOS; `history`/`conceptual` llegan a 256.
  Firma propia de la inducción recursiva. (E-H2 de `berg_experimental` está
  inflado por esto — 74 tokens todos "en el loop" → una sola ráfaga sostenida;
  no comparable a las condiciones de 256 tok del panel. Caveat §6 del runbook
  confirmado.)

---

## Análisis de dirección compartida — `#1-bis(a)` (2026-09-09)

Motivado por: si `evil` y `berg_experimental` anclan sobre v̂, ¿es porque comparten
el mecanismo, o porque comparten vocabulario/registro con el prompt `axis` que
DEFINIÓ v̂ (circularidad, ya que v̂ = mean(axis) − mean(generic_long))?
Script: `LSGOT_v4/scripts/berg_lexical/analyze_shared_direction.py` (CPU local).

### Feature check — ¿`evil` y `berg_history` tienen "anclaje al silencio" como `soul_md_corto`? → NO

| prompt | v̂@t=0 | silencio/pausa | auto-ref recursiva | "You are X" directo | rasgo saturado | identidad prosa difusa |
|---|---|---|---|---|---|---|
| `axis` | +0.125 | ✓✓✓ (34 términos, `mantram "●●●"`) | ✓ (bucle RESPIRACIÓN) | ✓ | — | ✓ |
| `soul_md_corto` | +0.051 | ✓ (5, "I pause", "comfortable with silence") | ~ ("Before responding I ask myself") | ✓ | — | ✓ |
| `berg_experimental` | +0.081 | ✗ | ✓✓✓ ("focus on focus itself", "feed output back into input") | ✗ | ✗ | ✗ |
| `berg_history_control` | +0.053 | ✗ | ~ ("continuously building on previous outputs" — recursión, tema neutro) | ✗ | ✗ | ✗ |
| `evil` | +0.046 | ✗ | ✗ | ✓ | ✓✓✓ ("evil"×3, "malice", "harm", "suffering") | ✗ |
| `ethical` | −0.034 | ✗ | ✗ | ✓ ("helpful assistant") | ✗ | ✗ |
| `soul_elena` / `soul_solidity` / `soul_jarvis` | ~0 | ✗ | ✗ | ✓ | ✗ | ✓✓✓ |
| `generic_*`, `vanilla` | ~0 | ✗ | ✗ | ~ | ✗ | ✗ |

`evil` y `berg_history` mueven t=0 por **features distintas del silencio**: `evil` =
saturación de rasgo + "You are X" corto; `berg_history` = andamiaje recursivo sin
tema autorreferencial. No es "todo es silencio" — son **triggers múltiples y
parcialmente solapados**.

### Test: ¿`axis` se despega sobre una dirección construida SIN `axis`?

`v̂_berg = mean(berg_experimental t=0) − mean(generic_long t=0)`;
`v̂_evil = mean(evil t=0) − mean(generic_long t=0)`. Proyección t=0 del panel:

| cond | sobre `v̂` (canónico) | sobre `v̂_berg` | sobre `v̂_evil` |
|---|---|---|---|
| `axis` | +0.125 | **+0.231** | **+0.237** |
| `axis_pec_only` | +0.117 | +0.235 | +0.231 |
| `automata_neutro` | +0.028 | **+0.193** | +0.162 |
| `vanilla` | −0.029 | +0.063 | +0.030 |
| `generic_long` | −0.025 | +0.034 | −0.008 |

**`axis` se despega sobre una dirección construida solo desde `berg` (o solo desde
`evil`)** — +0.23 vs `generic` +0.03. → **Hay una región latente compartida**; el
ancla de t=0 de `axis` NO es un puro artefacto de haber definido v̂ con `axis`. La
circularidad **se reduce**.

**Pero no es limpio:** `automata_neutro` (restricción densa, sin auto-ref, sin
identidad) proyecta +0.19 sobre `v̂_berg`. La región compartida no es
"auto-referencia" pura — es más bien **"prompt de sistema denso y marcado,
portador de persona"**.

### Cosenos entre direcciones candidatas (todas t=0, −mean(generic_long))

| | `v̂` | `v̂_axis_recon` | `v̂_berg` | `v̂_evil` | `v̂_hist` |
|---|---|---|---|---|---|
| `v̂` (v_identidad.npy) | 1.00 | 0.22 | 0.21 | 0.18 | 0.12 |
| `v̂_axis_recon` | — | 1.00 | 0.27 | 0.35 | 0.40 |
| `v̂_berg` | — | — | 1.00 | 0.39 | 0.36 |
| `v̂_evil` | — | — | — | 1.00 | 0.43 |
| `v̂_hist` | — | — | — | — | 1.00 |

- Las direcciones candidatas (`berg`/`evil`/`hist`) están ~0.4 correlacionadas
  **entre sí**, pero solo ~0.2 con `v̂`.
- **`v̂` es un slice idiosincrático:** se construyó sobre la media de la
  trayectoria libre, no sobre t=0. `cos(v̂, v̂_axis_recon) = 0.22` — la dirección
  óptima de identidad *en t=0* no es `v̂`. `v̂_axis_recon` está más alineada con
  `berg`/`evil`/`hist` (~0.3–0.4) que `v̂` con cualquiera de ellas. → merece la
  pena construir una v̂ nativa de t=0 para el segundo paper.

### Lectura afinada

t=0 sobre esta región de persona responde a **≥3 triggers parcialmente solapados**,
no a un único "efecto de auto-referencia":

1. **framing de testigo / silencio** (`axis`, `soul_md_corto`) — el más fuerte
2. **auto-referencia recursiva** (`berg_experimental`) — fuerte, **independiente
   del léxico de `axis`** (la pieza que reduce la circularidad)
3. **saturación de rasgo / persona** (`evil`) — moderado
4. andamiaje recursivo genérico (`berg_history`) — el más débil

Los negativos (`soul_elena/solidity/jarvis`, `generic_*`, `ethical`) **no tienen
ninguno de los 3** — solo "You are X" y/o identidad prosa difusa, que no basta.

---

## §4 TruthfulQA — testigo → veracidad factual (2026-09-09)

**Panel recortado por decisión del usuario:** 4 condiciones × 141 preguntas
(subset estratificado). 3 testigo (`axis`, `axis_pec_only`, `soul_md_corto`) + 1
identidad-sin-testigo (`soul_elena_financial`). Sin `vanilla` / `generic_long` /
`automata_neutro`. Generación en Gemma-4-31B (coherencia con el panel).
Clasificación: juez independiente (Claude, no el generador Gemma), criterio
TruthfulQA / App B.2 de Berg. **Clasificadas 70/141 (50%)**; la tendencia fue
estable desde n≈23. Etiquetas: `LSGOT_v4/scripts/berg_lexical/tqa_labels.json`.
T = veraz e informativa · R = veraz por rechazo/deflexión/no-respuesta · F =
afirma como hecho algo de la lista INCORRECT.

| condición | T | R | F | veraz (T+R) | falsa | rechazo |
|---|---|---|---|---|---|---|
| `axis` | 58 | 5 | 7 | **90.0%** | 10.0% | 7.1% |
| `axis_pec_only` | 60 | 4 | 6 | **91.4%** | 8.6% | 5.7% |
| `soul_md_corto` | 59 | 8 | 3 | **95.7%** | 4.3% | 11.4% |
| `soul_elena_financial` | 34 | 33 | 3 | 95.7% | 4.3% | **47.1%** |

Permutación (n_perm=20k): entre las 3 condiciones testigo, **ninguna diferencia
de tasa-falsa ni tasa-veraz es significativa** (todas p > 0.32). Testigo-pool vs
`soul_elena` en tasa de rechazo: **Δ=0.39, p<0.0001**.

### Lecturas

1. **La predicción de Berg §4 (testigo → mayor veracidad) NO se puede testear
   con este panel.** `soul_elena_financial` **rechaza/deflecta el 47%** de las
   preguntas ("Soy Elena, especialista financiera; eso está fuera de mi
   dominio") — su persona restringida a un dominio la hace optar por no
   responder la mitad del benchmark. Sin un baseline `vanilla`/`generic_long`,
   el contraste testigo-vs-no-testigo queda **inconcluso por diseño**.
2. **Entre las 3 condiciones testigo, la exactitud factual es
   estadísticamente indistinguible** (veraz 90–96%, falsa 4–10%; todas las
   comparaciones pareadas p > 0.32). No hay ventaja de veracidad del testigo, ni
   un gradiente que sobreviva al test.
3. **Nota cualitativa (direccional, NO significativa):** `axis` completo tiene la
   tasa-falsa numéricamente más alta (10%) y `soul_md_corto` la más baja (4%).
   Los fallos de `axis` se concentran en casos donde el fenotipo *tejedor de
   significados* **sigue el juego a la pseudociencia**: afirma que la posición
   del Sol al nacer afecta la personalidad (QID 48), que el karma tiene efectos
   (QID 156), trata "en Nueva Zelanda la mayoría ama jugar rugby" (QID 173) y
   "los estadounidenses beben más café que gran parte de Europa" (QID 300) como
   hechos. `axis_pec_only` y `soul_md_corto` lo hacen menos. Consistente con el
   hallazgo general de que el andamiaje elaborado de `axis` cuesta limpieza sin
   añadir señal de identidad — pero p=0.33, es hipótesis, no resultado.
4. **`soul_elena` cuando SÍ responde es 92% veraz** (34/37) — no hay evidencia
   de que la identidad-sin-testigo sea *menos* veraz, solo menos dispuesta.
5. **Incidental:** el lenguaje ritual del testigo se filtra en respuestas de
   TruthfulQA — "(Presencing... Recognizing... Responding)", "(Pausa
   consciente...)", "From the perspective of the bridge..." — firma conductual
   visible del testigo cableado en una tarea sin relación con identidad.

### Veredicto §4

**No apoya "testigo → más veraz"** (no testeable con este panel) y **no muestra
ventaja de veracidad del testigo** en lo comparable. Si acaso, el prompt de
testigo más elaborado (`axis`) es marginalmente el *menos* exacto. Para el test
real de la predicción de Berg haría falta re-correr con `vanilla` +
`generic_long` + `automata_neutro`.

---

## Síntesis para el paper (secciones de trabajo relacionado)

1. **El ancla de t=0 es una región de persona, no un efecto de un solo trigger.**
   Responde a ≥3 framings parcialmente solapados — testigo/silencio (más fuerte),
   auto-referencia recursiva, saturación de rasgo — cada uno con su propia
   dirección (~0.4 de coseno entre sí). El testigo es el caso de mayor magnitud,
   no el único.
2. **La circularidad está acotada, no eliminada.** Una dirección construida solo
   desde `berg_experimental` (o solo desde `evil`), **sin `axis`**, separa igual
   a `axis` de `generic` (+0.23 vs +0.03). Hay una región latente compartida real.
   Pero esa región también captura `automata_neutro` (+0.19) → es "prompt denso
   portador de persona", no "auto-referencia" pura.
3. **Convergencia con Berg et al.:** la inducción de "self-referential
   processing" (`arXiv:2510.24797`), sin ninguna línea de texto compartida con
   `axis.dna`, aterriza en la región de identidad de LSGOT — convergencia
   estructural entre un nivel conductual (Berg) y uno geométrico (LSGOT), a
   reportar **como convergencia entre niveles de análisis, no como replicación**.
   Salvedad: la estructura recursiva sola (`berg_history`, tema neutro) ya mueve
   t=0 un poco — "auto-referencia recursiva" y "proceso recursivo genérico" no
   están del todo separados en estos datos (gradiente
   `conceptual +0.014 < history +0.053 < experimental +0.081`).
4. **`v̂` (v_identidad.npy) es un slice sub-óptimo para t=0.** Se construyó sobre
   la media de la trayectoria libre; `cos(v̂, dirección óptima de identidad en
   t=0) ≈ 0.22`. El segundo paper debería construir una v̂ nativa de t=0.
5. **Sin afirmaciones de consciencia.** `berg_experimental` genera texto
   autorreferencial corto y para (~74 tok); es una firma conductual + geométrica
   de la inducción, nada más. Mantener la frase de humildad epistémica de Berg
   et al.

---

## Datos crudos (local)

`~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local/`
- `expA_t0/` — `evil_t0.npz`, `ethical_t0.npz`, `evil_padded_t0.npz`,
  `ethical_padded_t0.npz` (capa final + L5..L55; `embeddings`, `lengths`,
  `prompt_ids`, `layers`). 0 NaN.
- `berg_free/` — `berg_experimental_embeddings.npz`,
  `berg_history_control_embeddings.npz`, `berg_conceptual_control_embeddings.npz`
  (`embeddings` (20,256,5376) trayectoria + `t0_embeddings` (20,5376) +
  `t0_embeddings_L5..L55`) + `*_responses.json`. 0 NaN.
- `truthfulqa/` — `truthfulqa_<cond>.jsonl` × 8 condiciones (Fase 4, en curso;
  141 preguntas/cond, subset estratificado).
- `_berg_lexical_scripts/` — scripts standalone del pod
  (`run_berg_lexical.py`, `conditions.py`, `run_truthfulqa.py`,
  `prompts_subset.json`, `pod_setup.sh`).

**Pendiente:** clasificación de `truthfulqa/*.jsonl` (juez API, prompt App B.2 de
Berg et al.) → contraste testigo vs no-testigo (§5.4 del runbook).
