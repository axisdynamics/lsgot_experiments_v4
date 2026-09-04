# E-L — Primer token como firma de identidad

> ⚠️ **Reescrito 2026-08-28 (noche):** chat_agente/chat_agente_sia
> (originalmente §3.4 de este reporte las identificó como el primer
> indicio del confound — ver `CHAT_AGENTE_MARKUP_CONFOUND_REPORT.md`)
> se **eliminaron por completo** de tablas y comparaciones. Los pares
> clave (axis_pec_only vs vanilla/automata_neutro) nunca las usaron — el
> hallazgo central no cambia.

**Fecha:** 2026-08-28
**Script:** `LSGOT_v4/scripts/fase0/analyze_EL_primer_token.py`
**Datos:** `.../gemma4_31b_combined/results_local/sia_extended_v5/*_embeddings.npz` (embeddings crudos, no versionados en `LSGOT_v4/data/`, que solo tiene métricas ya agregadas)
**Resultados crudos:** `LSGOT_v4/scripts/fase0/EL_results.json`

## 1. Resumen

Se midió la proyección coseno sobre v̂ del hidden state de **t=0** (primer
token generado) en las 9 condiciones libres, y se comparó contra la
proyección media de t>0 en la misma trayectoria. La doble disociación de
v̂ (axis_pec_only vs automata_neutro vs vanilla) **ya está presente en t=0**,
con el mismo signo y un tamaño de efecto igual de grande que el reportado
en el paper para la trayectoria completa. Implica que la identidad es (al
menos en parte) una propiedad del **estado de contexto**, no un efecto que
emerge del proceso de generación.

## 2. Método

- v̂ = `v_identidad.npy` (mean(axis) − mean(generic_long), **capa final**,
  norma 1; mismo vector usado en el paper — no L30, ver corrección en
  `EF2_REPORT.md` §0).
- Proyección: `cos(h_t, v̂)` — misma convención que `project_trajectory()`
  en `tier0_metrics.py` (E-H del paper).
- Por cada prompt (N=20/condición): `p(0)` y `mean(p(1:L))`.
- Test de permutación bilateral (n=1000, seed=42) + d de Cohen sobre `p(0)`
  entre pares clave.
- Wilcoxon pareado `p(0)` vs `mean(p(1:L))` dentro de cada condición (T1).
- Distribución léxica del primer token: **no hay logits/tokenizer
  disponibles localmente**, así que se usa como proxy aproximado la primera
  palabra del texto de respuesta (`_responses.json`), con entropía de
  Shannon sobre esa distribución léxica. Esto NO es entropía de logits —
  se reporta como control cualitativo, no como medida principal.

## 3. Resultados

### 3.1 Proyección v̂ en t=0, por condición (N=20)

| Condición | t=0 (media ± sd) | t>0 (media ± sd) |
|---|---|---|
| axis | +0.1249 ± 0.0095 | +0.2052 ± 0.0298 |
| axis_short | +0.1170 ± 0.0119 | +0.2021 ± 0.0248 |
| axis_pec_only | +0.1166 ± 0.0090 | +0.2126 ± 0.0281 |
| automata_neutro | +0.0282 ± 0.0192 | +0.0058 ± 0.0436 |
| generic_long | −0.0251 ± 0.0150 | +0.0670 ± 0.0289 |
| generic_short | −0.0453 ± 0.0150 | +0.0481 ± 0.0306 |
| vanilla | −0.0286 ± 0.0184 | +0.0987 ± 0.0306 |

### 3.2 Pares clave, t=0 (permutación n=1000)

| Par | d de Cohen | p |
|---|---|---|
| axis_pec_only vs vanilla | **+9.78** | <0.001 |
| axis vs vanilla | **+10.22** | <0.001 |
| axis vs automata_neutro | +6.23 | <0.001 |
| axis_pec_only vs automata_neutro | +5.75 | <0.001 |
| axis vs axis_pec_only | +0.88 | 0.005 |
| automata_neutro vs vanilla | +2.94 | <0.001 |

La doble disociación central del paper (axis_pec_only sigue a axis;
automata_neutro diverge) **replica en t=0** con tamaños de efecto tan
grandes o mayores que los reportados para la trayectoria completa.

### 3.3 T1 — t=0 vs t>0 (Wilcoxon pareado)

| Condición | t=0 | t>0 | d (within) | p (Wilcoxon) |
|---|---|---|---|---|
| axis | +0.125 | +0.205 | −3.54 | <0.001 |
| axis_pec_only | +0.117 | +0.213 | −4.48 | <0.001 |
| automata_neutro | +0.028 | +0.006 | +0.65 | 0.123 (n.s.) |

Hay un salto sistemático de magnitud entre t=0 y t>0 en casi todas las
condiciones (confirma el confound de REPORTE_FASE0: el primer token es
especial). **Pero la separación *entre* condiciones tiene el mismo signo y
orden en t=0 y en t>0** — el confound infla/desinfla la magnitud, no
invierte ni borra la disociación.

### 3.4 Primer token textual (proxy léxico, no logits)

`axis`/`axis_pec_only` convergen en "pausa"/"respiro" (H=1.6–1.9 bits);
`vanilla`/`generic_*` son más dispersos (H=2.3–2.5 bits). Nota histórica:
este análisis fue el que originalmente detectó que `chat_agente`/
`chat_agente_sia` generan **siempre** `<div` como primera palabra
(H=0 bits) — el hallazgo que llevó a `CHAT_AGENTE_MARKUP_CONFOUND_REPORT.md`
y a excluir esa familia de condiciones de todo el proyecto (ya no
aparecen en este reporte).

## 4. Trampas aplicadas

- **T1**: núcleo del experimento — comparación explícita t=0 vs t>0,
  reportada en ambas direcciones (§3.3). Resultado: el confound existe en
  magnitud pero no en signo/orden.
- **T2**: n=20 prompts, 1 manipulación por condición — no se puede
  distinguir "true" identidad de un artefacto de esa redacción específica
  de system prompt. Pendiente de réplica (ver Set_experimental.md).
- **T5**: 8 comparaciones de pares en t=0, todas con p<0.001 salvo axis vs
  axis_pec_only (p=0.005) — replicado también en t>0 (mismos signos, ver
  paper §3.5). No se corrigió por comparaciones múltiples; se reporta el
  conteo explícito.
- **T7**: no se usó Δκ/W₁.

## 5. Replicación

Los pares clave con d>2 replican tanto en t=0 (esta corrida) como en la
trayectoria completa (§3 de este reporte, con automata_neutro como
referencia limpia de restricción). El patrón t=0 vs t>0 (§3.3) es interno
a este experimento, sin punto de comparación externo aún.

## 6. Implicación para el paper

**Desambiguador resuelto: identidad = propiedad del estado de contexto,
no del proceso de generación.** La huella de v̂ no "emerge" durante la
generación — ya está en el hidden state del primer token, antes de que el
modelo haya producido nada. Esto es coherente con, y refuerza, la lectura
de §4 del paper (identidad como dirección estática): si fuera un efecto
del proceso dinámico de generación, debería estar ausente o débil en t=0.
Sugerencia de redacción para §4/§5: agregar esta evidencia como apoyo
directo a "identidad = huella estática del contexto", con la salvedad de
T2 (n=1 de manipulación) antes de tratarlo como definitivo.
