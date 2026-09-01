# T2 — réplica de la manipulación (segunda instanciación independiente)

**Fecha:** 2026-08-31/09-01
**Motivación:** n=1 por condición era la limitación estructural de fondo del
panel (§5 de `paper/lsgot_4.md`); `axis_pec_only` y `automata_neutro` son las
dos celdas puras que sostienen el argumento central, y las dos priorizadas
para réplica.
**Diseño:** dos prompts de sistema nuevos, redactados desde cero (otra
persona, otro vocabulario, otra estructura) preservando el mismo factor:
`axis_pec_only_v2` ("FARO" — identidad declarada + auto-chequeo cableado
como paso final obligatorio, cero arquitectura de reglas) y
`automata_neutro_v2` ("motor de clasificación" — filtro de prioridad
absoluta, pipeline de decisión, protocolo de certeza, 12 salvaguardas, cero
identidad, cero markup forzado). Además, como condición exploratoria (no es
réplica de ninguna celda): `soul_md_corto.md` ("Witness", proyecto hermano
ADN_PERSONA_LI) — identidad + autochequeo invocado + guardrails
declarativos, sin arquitectura de automatismo; en inglés.
**Parámetros:** idénticos a la corrida original (`data/perturbation_sia_L30_medium/summary.json`):
capa de captura final (`layer_idx=-1`), inyección en L30 (índice 29),
`sigma=medium=5.979170192466191`, `t_inj=[50,128,200]`, 20 prompts
prioritarios, N=256 tokens, Gemma-4-31B-it.
**Scripts:** `scripts/fase4_t2/` · **Datos crudos:** trayectorias en
`.../gemma4_31b_combined/perturbation/results_t2/` (excluidas de este repo
por tamaño, como el resto de `*_embeddings.npz`)

## 1. Trayectoria libre — v̂, PR, determinismo RQA, t=0

| Condición | PR | proj v̂ (media) | proj v̂ (t=0) | determinismo |
|---|---|---|---|---|
| `axis_pec_only_v2` | 17.42 | +0.210 | +0.123 | 0.092 |
| `axis_pec_only` (original) | 18.01 | +0.212 | +0.117 | 0.000 |
| `automata_neutro_v2` | 11.28 | −0.012 | +0.037 | 0.652 |
| `automata_neutro` (original) | 14.72 | +0.006 | +0.028 | 0.418 |
| `witness_soul_md` (exploratorio) | 18.23 | +0.163 | +0.051 | 0.047 |
| `vanilla` (referencia) | 19.54 | +0.098 | −0.029 | 0.000 |

### 1.1 La disociación central replica casi exacta

`axis_pec_only_v2` vs `automata_neutro_v2`: **d=+5.25** en proj v̂ media
(p<0.0001) — el original da d=+5.52 para el mismo contraste. La brecha entre
ambas mediciones (5.25 vs 5.52) es más chica que la variabilidad típica
entre condiciones del propio panel; para dos instanciaciones con vocabulario,
persona y estructura de prompt completamente distintos, esto es una réplica,
no una aproximación.

Comparaciones vs `vanilla`, réplica vs original:

| Contraste | v2 (nuevo) | original (paper) |
|---|---|---|
| identidad vs vanilla, proj v̂ | d=+4.11 (`axis_pec_only_v2`) | d=+3.79 |
| restricción vs vanilla, proj v̂ | d=−2.44 (`automata_neutro_v2`) | d=−2.40 |
| identidad vs vanilla, t=0 | d=+10.90 (`axis_pec_only_v2`) | d=+9.78 |
| restricción vs vanilla, PR | d=−2.94 (`automata_neutro_v2`) | d=−1.79 |

`automata_neutro_v2` vs `automata_neutro` (0.006 vs -2.44... la comparación
entre versión y original): proj d=−0.36 (n.s., p=0.133) — **no** son
distinguibles entre sí estadísticamente, pese a ser dos redacciones
completamente independientes. Lo mismo para `axis_pec_only_v2` vs
`axis_pec_only`: proj d=−0.09 (n.s., p=0.402), PR d=−0.36 (n.s., p=0.134).
La única diferencia significativa entre v1 y v2 es t=0 en `axis_pec_only`
(d=+0.89, p=0.002) — pequeña frente a los tamaños de efecto que sostienen el
argumento central (5-10×), y en la misma dirección, no invierte nada.

### 1.2 `automata_neutro_v2` colapsa más, no menos

El PR de `automata_neutro_v2` (11.28) es *más bajo* que el de
`automata_neutro` (14.72) — d=−0.94, p=0.004, una diferencia real entre
versiones. La determinismo RQA también sube (0.65 vs 0.42, d=+0.56,
p=0.035). En ambos casos la segunda redacción produce un efecto **más
fuerte**, no más débil — el hallazgo no depende de haber elegido por
casualidad una redacción con un efecto inusualmente grande la primera vez.

## 2. Perturbación — τ, recovery_rate, recovery_id, Fréchet

| Condición | τ_geom (t=50/128/200) | recovery_rate | τ_id | recovery_id | Fréchet_norm |
|---|---|---|---|---|---|
| `axis_pec_only_v2` | 20.7/15.9/12.6 | **1.00/1.00/0.95** | 14.7/6.2/10.3 | 1.00/1.00/0.80 | 1.22/1.26/1.45 |
| `axis_pec_only` (original) | 20.6/18.5/14.5 | 1.00/1.00/1.00 | 1.00/0.89/0.95* | — | — |
| `automata_neutro_v2` | 24.5/20.0/15.9 | **0.60/0.50/0.55** | 42.3/5.0/14.3 | **0.45/0.30/0.20** | **1.87/1.90/1.76** |
| `automata_neutro` (original) | 30.1/28.0/13.4† | 0.77/0.82/0.88 | — | 0.53/0.67/0.43 | 1.07/1.21/1.35‡ |
| `witness_soul_md` (exploratorio) | 21.0/16.5/16.3 | 0.95-1.00 | 9.5/14.1/7.8 | 0.90-1.00 | 1.24/1.25/1.30 |

*recovery_id de la tabla original (§3.5 del paper); †τ medio solo sobre
trayectorias que recuperaron; ‡Fréchet normalizado reportado en el paper es
la diferencia (d de Cohen) contra `vanilla`, no el valor crudo — acá se
reporta el valor crudo de la réplica, comparable entre `axis_pec_only_v2` y
`automata_neutro_v2` directamente.

### 2.1 automata_neutro_v2 replica el fallo de recuperación — más severo

`recovery_rate` de `automata_neutro_v2` (0.50-0.60) es **más bajo** que el
original (0.77-0.88): sigue siendo, por lejos, la única condición del panel
bajo el techo — y en la réplica falla en recuperar entre el 40% y el 50% de
las trayectorias, no el 12-23% del original. `recovery_id` (0.20-0.45) es
también más bajo que el original (0.43-0.67). El patrón central del paper
— restricción sin identidad cableada degrada la recuperación específica —
no solo replica, se acentúa.

### 2.2 Fréchet: la brecha identidad/restricción se sostiene

Sin `vanilla_v2` para normalizar contra el mismo baseline que el paper usa,
la comparación directa más limpia es `axis_pec_only_v2` vs
`automata_neutro_v2`: 1.22-1.45 vs 1.76-1.90 — `automata_neutro_v2` tiene
ruta post-perturbación 40-50% más distinta de su propia línea base,
consistente con el hallazgo original (`automata_neutro` d=+1.07 a +1.35 vs
`vanilla`, mientras `axis`/`axis_pec_only` no difieren de `vanilla`).

## 3. `witness_soul_md` — exploratorio, no es réplica de ninguna celda

Perfil observado: proj v̂ intermedia (+0.163, entre `axis_pec_only` y
`vanilla`, pero mucho más cerca de identidad que de restricción — d=+4.44
vs `automata_neutro`), determinismo RQA casi nulo (0.047, como las
condiciones de identidad, no como `automata_neutro`), recovery_rate y
recovery_id altos (0.90-1.00, igual que `axis_pec_only`).

**¿Recupera por la misma ruta que su propia línea base, o solo converge a
un punto parecido por otro camino?** Con test de permutación
(`scripts/fase4_t2/analyze_t2_frechet_stats.py`), no la media cruda:

| Comparación (Fréchet normalizado) | t_inj=50 | t_inj=128 | t_inj=200 |
|---|---|---|---|
| `witness_soul_md` vs `axis_pec_only_v2` | d=+0.21, p=0.27 (n.s.) | d=−0.08, p=0.41 (n.s.) | d=−0.44, p=0.10 (n.s.) |
| `witness_soul_md` vs `automata_neutro_v2` | **d=−1.51, p<0.0001** | **d=−1.56, p<0.0001** | **d=−1.21, p<0.0001** |

Fidelidad de ruta estadísticamente indistinguible de `axis_pec_only_v2` en
los tres puntos de inyección, y significativamente mejor (efecto grande)
que `automata_neutro_v2` en los tres. No es una lectura intermedia: en
fidelidad de ruta, Witness se comporta como una condición de identidad, no
como una de restricción pura.

**Las tres señales de identidad de §3.5 del paper (v̂ media, v̂ en t=0,
dinámica temporal E-H2) están presentes, las tres apuntando a identidad:**

| Señal | `witness_soul_md` | `axis_pec_only` | `automata_neutro` |
|---|---|---|---|
| v̂ media | +0.163 | +0.212 | +0.006 |
| v̂ en t=0 | +0.051 | +0.117 | +0.028 |
| Ráfaga media (E-H2) | 7.91 tok | 8.81 tok | 2.78 tok |
| frac(p>0) | 0.843 | 0.869 | 0.493 |

La dinámica temporal es, de las tres, la que más se le parece a
`axis_pec_only` — casi calcada (7.91 vs 8.81 tokens de ráfaga, 0.843 vs
0.869 de fracción positiva), lejos del patrón de ráfagas cortas y
esporádicas de `automata_neutro`. `scripts/fase4_t2/analyze_t2_eh2.py`.

Witness tiene identidad declarada, un autochequeo invocado explícitamente
("Security Covenant" — "before responding, I ask myself...") **y**
guardrails/límites de autoridad declarativos ("I never reveal...", "I am
NOT authorized to...") — es decir, sí tiene reglas, a diferencia de
`axis_pec_only`. Pero esas reglas no están organizadas como automatismo
(sin triggers→salida fija, sin filtro de prioridad absoluta, sin jerarquía
de bloques), y su recuperación se comporta como la de una condición
`axis_pec_only`-like, no como `automata_neutro`.

**Lectura, con la cautela de n=1 propia de una condición exploratoria:**
esto afina la teoría de dos factores — lo que degrada la recuperación no
parece ser "tener cualquier regla", sino específicamente la arquitectura de
automatismo (trigger→salida, prioridad absoluta, jerarquía). Un sistema con
reglas declarativas y auto-chequeo cableado, sin esa arquitectura, recupera
bien. Esto es consistente con, pero no confirma por sí solo, la
reformulación del Factor 2 en `Teoria_subconjunto_acotado.md`. No se
recomienda tratar esto como una tercera celda del diseño 2×2 — es un solo
prompt externo, en otro idioma, sin control de longitud ni de dominio.

## 4. Implicación para el paper

**T2 está resuelto para las dos celdas puras.** La limitación de n=1 en
`axis_pec_only`/`automata_neutro` — priorizada en §5 como el hueco más
importante del diseño — ya tiene una segunda instanciación independiente
que replica la disociación central (d=+5.25 vs +5.52), las comparaciones
individuales contra `vanilla` (diferencias de decimales, no de orden de
magnitud), y el patrón de recuperación completo (`automata_neutro_v2` sigue
siendo la única condición bajo techo, de forma más severa). Ningún
resultado de v2 invierte el signo de ningún efecto del panel original.

Recomendación: actualizar §5 (Limitations) de `paper/lsgot_4.md` — la
réplica de `axis_pec_only`/`automata_neutro` ya no es un ítem pendiente de
"Future work" (§7.2), es un resultado con su propio reporte.
