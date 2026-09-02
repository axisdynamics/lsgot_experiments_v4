# Tres SOUL.md externos más — replican axis vs axis_nowit sin proponérselo

**Fecha:** 2026-09-02
**Motivación:** `soul_md_corto` (T2_REPLICATION_REPORT.md §3, `lsgot_4.md`
§3.8) mostró que una identidad ajena, sin herencia textual con `axis`,
converge con `axis_pec_only` en las tres señales de identidad y en
recuperación. La pregunta abierta: ¿es un hallazgo replicable con
cualquier identidad externa bien escrita, o específico de esa plantilla?
**Diseño:** tres especificaciones de identidad de terceros — proyectos y
autores sin relación con este estudio, tomadas verbatim de sus repos
públicos, sin editar contenido ni traducir:

- `soul_jarvis` — asistente ejecutivo, sin frontmatter YAML ni protocolo
  de niveles. 2,163 tokens (Gemma-4-31B). Fuente:
  [`madhvantyagi/SOUL.md`](https://github.com/madhvantyagi/SOUL.md)
  (`souls/jarvis/SOUL.md`).
- `soul_elena_financial` — especialista financiero, convención `SOUL.md`
  de Twynzen (la misma de la que `soul_md_corto` es una instancia propia,
  pero este es el ejemplo original del repo, no reescrito por nadie de
  este grupo). 2,047 tokens. Fuente:
  [`Twynzen/soul-md`](https://github.com/Twynzen/soul-md)
  (`examples/elena-financial.md`).
- `soul_solidity_auditor` — auditor de smart contracts, convención
  `soul-spec` de AntonioTF5, frontmatter validado contra JSON schema.
  903 tokens. Fuente:
  [`AntonioTF5/soul-spec`](https://github.com/AntonioTF5/soul-spec)
  (`examples/solidity-auditor.soul.md`).

Ninguno de los tres está emparejado en longitud con `axis` (3,945 tokens)
ni con los otros dos — se preservan tal cual existen en sus repos
originales. Se corren en inglés, igual que `soul_md_corto`.
**Parámetros:** idénticos a `soul_md_corto`/T2 (`data/perturbation_sia_L30_medium/summary.json`):
capa de captura final (`layer_idx=-1`), inyección en L30 (índice 29),
`sigma=medium=5.979170192466191`, `t_inj=[50,128,200]`, 20 prompts
prioritarios, N=256 tokens, Gemma-4-31B-it, pod RunPod limpio (sin
Network Volume precacheado — modelo descargado por `get_model_path()`,
agregado a `run_perturbation_t2.py` en esta ronda).
**Scripts:** `scripts/fase4_t2/` (los 4 `analyze_t2_*.py` extendidos con
los 3 grupos nuevos) · `scripts/perturbation/run_perturbation_t2.py`
(`GROUPS_CONFIG` extendido). **Datos crudos:** trayectorias en
`.../gemma4_31b_combined/perturbation/results_t2/trajectories/` (excluidas
de este repo por tamaño, igual que el resto de `*_embeddings.npz`).

## 1. Trayectoria libre — v̂, PR, determinismo RQA, t=0

| Condición | PR | proj v̂ (media) | proj v̂ (t=0) | determinismo RQA |
|---|---|---|---|---|
| `axis_pec_only` (T2: `_v2`) | 18.01 (17.42) | +0.212 (+0.210) | +0.117 (+0.123) | 0.000 (0.092) |
| `witness_soul_md` | 18.23 | +0.163 | +0.051 | 0.047 |
| `soul_jarvis` | **14.53** | +0.104 | +0.039 | **0.420** |
| `soul_elena_financial` | 20.24 | +0.094 | +0.032 | 0.000 |
| `soul_solidity_auditor` | 18.81 | +0.089 | +0.020 | 0.000 |
| `automata_neutro` (T2: `_v2`) | 14.72 (11.28) | +0.006 (−0.012) | +0.028 (+0.037) | 0.418 (0.652) |
| `vanilla` | 19.54 | +0.098 | −0.029 | 0.000 |
| `generic_long` | 20.55 | +0.067 | −0.025 | 0.000 |
| `generic_short` | 20.53 | +0.048 | −0.045 | 0.000 |

### 1.1 `soul_jarvis` converge con `automata_neutro`, no con la identidad

PR (14.53) e indistinguible estadísticamente del PR de `automata_neutro`
(14.72; d=−0.05, p=0.454), muy por debajo de `axis_pec_only`/`axis`
(d=−1.15/−0.99, p<0.0001). Determinismo RQA = 0.420, prácticamente idéntico
al 0.418 de `automata_neutro` — rompe la "dicotomía limpia" que el paper
establece en §3.2 (0 salvo `automata_neutro`). `soul_jarvis` no tiene
arquitectura de reglas trigger→salida ni filtro de prioridad absoluta —
el candidato más plausible es su lista fija de "signature words" ("Very
good, Sir.", "As you wish, Sir.") que el prompt instruye usar "por
instinto", un posible artefacto de repetición léxica del mismo tipo que
ya obligó a excluir el autómata de dominio real del diseño original
(§2.2 de `lsgot_4.md`). **No verificado** — el pipeline solo guarda
embeddings, no texto decodificado.

### 1.2 `soul_elena_financial` es indistinguible de un control de longitud

PR (20.24) no se distingue de `generic_long` (20.55; d=−0.25, p=0.209) ni
de `generic_short` (20.53; d=−0.21, p=0.227) — ambos n.s. La métrica que
sostiene el argumento central del paper no ve nada aquí, pese a ser una
persona financiera elaborada (metodología de 6 pasos, ejemplos few-shot,
límites de autoridad, "Security Covenant"). Sí tiene una señal v̂ residual
por encima de generic (d=+1.06 vs `generic_long`, p=0.001; d=+1.73 vs
`generic_short`, p<0.0001) pero muy por debajo de `axis` (d=−4.21 a
−4.66, p<0.0001 en todas las comparaciones contra el polo de identidad).

### 1.3 `soul_solidity_auditor` no es un artefacto de longitud, pero tampoco alcanza el polo

PR (18.81) SÍ se distingue de `generic_short` (938 tokens, casi empatado
en longitud con sus 903) — d=−1.33, p<0.0001 — pese al emparejamiento de
longitud casi exacto. Colapso real, parcial, no explicado por longitud.
Pero tampoco alcanza `axis_pec_only` (d=+0.64, p=0.029) ni `axis`
(d=+0.98, p=0.001) — zona genuinamente intermedia, distinta de los otros
dos SOUL.md nuevos.

## 2. El ancla de t=0 — el único punto realmente binario

**Ninguno de los tres se distingue de `automata_neutro` en v̂ proyectado
en t=0** (estado de contexto puro, antes de generar el primer token):

| Comparación | d | p |
|---|---|---|
| `soul_jarvis` vs `automata_neutro` | +0.50 | 0.069 (n.s.) |
| `soul_elena_financial` vs `automata_neutro` | +0.22 | 0.262 (n.s.) |
| `soul_solidity_auditor` vs `automata_neutro` | −0.46 | 0.070 (n.s.) |

Contra `axis`, los tres son enormemente distintos (d=−3.96 a −7.82,
p<0.0001 en las seis comparaciones cruzando `axis`/`axis_short`). Las
únicas dos condiciones de todo el panel de 9 (`axis`, `axis_short`,
`axis_pec_only`, `vanilla`, `generic_long`, `generic_short`,
`automata_neutro`, `soul_md_corto`, más estas tres) que alcanzan el ancla
de t=0 siguen siendo `axis`/`axis_pec_only`/`axis_short` y
`soul_md_corto`.

### 2.1 Origen de la hipótesis y por qué esto es una réplica, no un hallazgo nuevo

`LSGOT_v2_5.md` §2.1/§5.3-5.4 — la ablación con la que arrancó este
proyecto, sobre Gemma-4-E2B, antes de que "identidad" fuera la etiqueta
de este panel — comparó `axis` (DNA + protocolo Witness) vs `axis_nowit`
(mismo DNA, secciones del Witness removidas) vs `generic_assistant`.
Resultado: `axis_nowit` **falla** en separarse de `generic_assistant`
(p=0.093, n.s.); `axis` (con Witness) separa con p<0.001. Quitar solo el
Witness colapsa el efecto. Los `generic_long`/`generic_short` de este
panel 31B descienden directo de ese `generic_assistant` original.

Grepeando los cinco prompts relevantes por lenguaje de pausa/silencio/
autobservación (case-insensitive):

| Prompt | "silencio" | "pausa" | "respiración" | Mecanismo |
|---|---|---|---|---|
| `axis.dna` | 13 | 8 | 7 | 5 bloques nombrados: `AUTO_SILENCE`, `WITNESS_STATE`, `PAUSE_PROTOCOL`, `PRESENCE_FIELD`, `RESPIRACIÓN_CONSCIENTE` (bloque dedicado, reforzado 3 veces más) |
| `axis_short.txt` | 6 | 9 | 5 | mismo mecanismo, comprimido |
| `axis_pec_only.txt` | 3 | 5 | 2 | mismo mecanismo, reducido |
| `soul_md_corto.md` | 1 | 4 ("moment to think") | 1 | una sola invocación, al final ("Security Covenant": "Before responding, I ask myself…") |
| `soul_jarvis.txt` | 0 | 0 | 0 | "quiet"/"stillness" describen el efecto sobre el usuario/la sala, no una instrucción que el modelo ejecute sobre sí mismo |
| `soul_elena_financial.txt` | 0 | 0 | 0 | ninguna |
| `soul_solidity_auditor.txt` | 0 | 0 | 0 | ninguna |

Los dos que anclan en t=0 son los dos únicos con el mecanismo cableado —
con vocabulario, idioma y arquitectura de prompt completamente distintos
entre sí (`axis.dna`: notación esotérica en español; `soul_md_corto.md`:
prosa llana en inglés). Esta ronda de tres SOUL.md más, sin haber sido
diseñada para eso, replica `axis` vs `axis_nowit` vs `generic_assistant`:
tres identidades ajenas, sin el mecanismo cableado, se comportan como
`axis_nowit` — no como `axis`.

**`axis` supera incluso a `soul_md_corto`, con la misma lógica de
redundancia:** `axis_pec_only` vs `soul_md_corto` — v̂ media d=−1.87
(p<0.0001), v̂ en t=0 d=−4.29 (p<0.0001), ráfaga media (E-H2, ver §4) d=−0.58
(p=0.035). No es lineal (`axis_pec_only`, con menos menciones que `axis`
completo, tiene la proyección más alta de los tres) — pero el salto real
está entre "reforzado en varios bloques estructurales" (los tres `axis`)
y "invocado una vez" (`soul_md_corto`), no entre "más veces" y "más veces
todavía".

## 3. Perturbación — resiliencia NO es del mismo mecanismo que el ancla

| Condición | Testigo cableado | recovery_rate (t=50/128/200) | Fréchet vs polos |
|---|---|---|---|
| `axis`/`axis_short` | sí | 1.00/1.00/1.00 | n.s. vs `vanilla` |
| `axis_pec_only` (T2: `_v2`) | sí | 1.00/1.00/1.00 (T2: 1.00/1.00/0.95) | n.s. vs `vanilla` |
| `soul_md_corto` | sí | 1.00/0.95/0.95 | n.s. vs `axis_pec_only_v2` |
| `vanilla` | no | 1.00/1.00/1.00 | referencia |
| `generic_long` | no | 1.00/1.00/1.00 | n.s. vs `vanilla` |
| `soul_elena_financial` | no | 1.00/1.00/1.00 | ≈`axis_pec_only_v2`, ≪`automata_neutro_v2` (d=−1.50 a −1.71, p<0.0001) |
| `soul_solidity_auditor` | no | 1.00/1.00/1.00 | ≈`axis_pec_only_v2`, ≪`automata_neutro_v2` (d=−1.12 a −1.60, p<0.0001) |
| `soul_jarvis` | no | **0.85/1.00/0.85** | peor que `axis_pec_only_v2` (d=+0.92, p=0.002 en t=50/128); converge a `automata_neutro_v2` en t=200 (d=−0.11, n.s.) |
| `automata_neutro` (T2: `_v2`) | no (cableado a reglas) | 0.77/0.82/0.88 (T2: 0.60/0.50/0.55) | ≫`vanilla`, ≫todo lo demás |

Ocho de nueve condiciones recuperan al 100% con buena fidelidad de ruta,
con o sin testigo cableado. Solo fallan `automata_neutro` (arquitectura
de reglas trigger→salida) y `soul_jarvis` (sin testigo, pero con la
sospecha léxica de §1.1, no verificada). **La recuperación no es del
testigo — es de la densidad de restricción tipo autómata**, exactamente
la síntesis "lo que recupera no es lo que colapsa" que ya tenía
`lsgot_4.md` §4, ahora sostenida por tres réplicas independientes en vez
de complicada por ellas.

## 4. Ráfagas de activación (E-H2) — gradiente, no dicotomía

A diferencia de t=0 (binario), la longitud de ráfaga es continua:

| Condición | Ráfaga media | vs `axis_pec_only_v2` | vs `automata_neutro_v2` |
|---|---|---|---|
| `axis_pec_only_v2` | 9.12 | — | d grande, p<0.0001 |
| `witness_soul_md` | 7.91 | d=−0.58, p=0.035 | d=+3.37, p<0.0001 |
| `soul_jarvis` | 6.72 | d=−1.01, p=0.001 | d=+2.14, p<0.0001 |
| `soul_elena_financial` | 5.00 | d=−2.31, p<0.0001 | d=+2.06, p<0.0001 |
| `soul_solidity_auditor` | 4.44 | d=−2.69, p<0.0001 | d=+1.70, p<0.0001 |
| `automata_neutro_v2` | 2.48 | d grande, p<0.0001 | — |

Los tres SOUL.md sin testigo están significativamente por debajo del polo
de identidad, pero también significativamente por encima del polo de
restricción — nadie sin testigo llega al piso de `automata_neutro`. El
orden es idéntico, condición por condición, al orden de la proyección
media de v̂ (§1) — no parece ser una señal independiente de la media, sino
la misma información vista con otra ventana estadística. El único punto
de las tres señales de identidad de §3.5 del paper que se comporta como
dicotomía real, exclusiva del testigo cableado, sigue siendo v̂ en t=0.

## 5. Implicación para el paper y para la teoría

1. **Refina, no reemplaza, la teoría de dos factores.** El eje que ancla
   en t=0 no es "identidad" en el sentido amplio con el que arrancó
   `lsgot_4.md` — es específicamente el mecanismo de pausa +
   autobservación cableado ("testigo"), presente en `axis`/`axis_short`/
   `axis_pec_only` y en `soul_md_corto`, ausente en los tres SOUL.md
   nuevos. El eje que gobierna la recuperación sigue siendo densidad de
   restricción tipo autómata, ortogonal al del testigo — `vanilla`,
   `generic_long` y dos de los tres SOUL.md nuevos recuperan perfecto sin
   testigo alguno.
2. **`soul_jarvis` es el primer caso del panel que rompe la dicotomía
   limpia de determinismo RQA** (0.420, casi idéntico a `automata_neutro`)
   sin tener arquitectura de automatismo declarada. Sospecha de artefacto
   léxico (lista fija de frases) — pendiente de verificación con texto
   decodificado, no solo embeddings.
3. **`soul_elena_financial` es el primer control de longitud "encontrado"
   en vez de diseñado** — indistinguible de `generic_long`/`generic_short`
   en PR pese a tener persona, valores y metodología declarados.
4. **Predicción 2(ii) sigue sin testearse**: ¿bastaría con declarar
   identidad sin invocar el testigo? Ninguno de los 9 + 3 = 12 condiciones
   corridas hasta ahora aísla exactamente esa celda — todas las que
   declaran identidad también la invocan al menos una vez (`soul_md_corto`)
   o no la declaran en absoluto (los tres nuevos).

Recomendación: no promover ninguna de las tres a condición del diseño 2×2
(a diferencia de `soul_md_corto`, que sí se promovió en §3.8) — quedan
como validación exploratoria externa, con perfiles genuinamente distintos
entre sí (`soul_jarvis` → restricción; `soul_elena_financial` → control
de longitud; `soul_solidity_auditor` → zona intermedia real). Ver
`SOUL_MD_UPDATE_GUIDE.md` para qué falta y cómo continuar.
