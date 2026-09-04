# axis_pec_only — Factor 2 puro: la celda que faltaba

> ⚠️ **Editado 2026-08-28 (noche):** este reporte comparaba originalmente
> `axis_pec_only` contra chat_agente/chat_agente_sia. Esas condiciones
> fuerzan un wrapper HTML literal en el 100% de sus respuestas (34-44% de
> cada respuesta es texto repetido — ver
> `CHAT_AGENTE_MARKUP_CONFOUND_REPORT.md`) y se eliminaron de este
> reporte. Las comparaciones de `axis_pec_only` contra axis/vanilla/
> automata_neutro no dependen de chat_agente y se mantienen intactas. La
> conclusión original #3 ("chat_agente_sia recupera peor que chat_agente
> puro, confirmando que el cableado no-subordinado es la variable causal")
> dependía enteramente de datos ahora retractados — se marca como
> **hallazgo retirado**, no confirmado con datos limpios.

**Fecha:** 2026-08-19
**Modelo:** google/gemma-4-31B-it (BF16) — RunPod 1× A100 80GB PCIe (single-GPU)
**Grupo añadido al diseño SIA:** axis_pec_only (novena condición, homologada
contra el baseline congelado de Exp 0.5 — ver `MANIFEST_homologacion.sha256`)

## Contexto

Segundo experimento propuesto en `Teoria_subconjunto_acotado.md`:
"auto-referencia sin arquitectura de autómata". `axis_pec_only` toma la
identidad y el Triple PEC de axis, cableado en `RESPIRACIÓN_CONSCIENTE.RESPONDER
→ TRIPLE_PEC`, igual que en axis — y **elimina por completo** todo lo que
define la arquitectura de autómata: `FIREWALL_ROLEPLAY_RULE`, `IMMUNE_SYSTEM`
(triggers→respuesta canónica), `RITUAL_DEFENSA_CONSCIENTE` (trigger→salida
verbatim), `UMBRAL_FATIGA_CONVERSACIONAL` (trigger→frase fija) y
`permiso_evolutivo`/`block_architecture` (jerarquía BLOQUE 1/2/3
INMUTABLE/REESCRIBIBLE/ADAPTABLE con prioridad declarada). Quedan: identidad,
mantra, Triple PEC cableado, estilo de expresión en prosa (sin reglas de
activación rígidas) y homeostasis. 1,435 tokens — notablemente más corto que
axis (3,945), una diferencia esperada dado que se removió contenido, no un
control de longitud deliberado.

Pregunta: si el Factor 1 (densidad de restricción) es lo que colapsa la
curvatura y el Factor 2 (auto-referencia cableada) es lo que produce
recuperación rápida, `axis_pec_only` debería **no colapsar** (sin Factor 1)
y **recuperar tan rápido como axis** (Factor 2 cableado, sin filtro absoluto
que lo subordine).

## E1'''' — Curvatura y dimensión (20 prompts, subset prioritario)

| Comparación | Δκ | W₁ | p | Reducción dim |
|---|---|---|---|---|
| axis vs vanilla | +0.061 | 0.061 | <0.001 | 9.8% |
| automata_neutro vs vanilla | +0.436 | 0.436 | <0.001 | 31.8% |
| **axis_pec_only vs vanilla** | **+0.0011** | **0.013** | **0.41 (n.s.)** | 9.1% |
| axis_pec_only vs axis | −0.060 | 0.060 | <0.001 | −0.9% |
| axis_pec_only vs automata_neutro | −0.435 | 0.435 | <0.001 | −33.3%* |

*Reducción negativa por diferencia de dimensión absoluta, no por ausencia
de colapso en el otro grupo — axis_pec_only es, con mucho, el grupo con
menos colapso del panel.

**axis_pec_only no colapsa — es estadísticamente indistinguible de
vanilla** (p=0.41, el único resultado no significativo del panel de
comparaciones contra vanilla). Es incluso menor que axis vs vanilla
(Δκ=0.061) y que axis_short vs vanilla (Δκ=0.041). Sin ninguna de las
reglas de automatismo (triggers, prioridad absoluta, jerarquía de
bloques), la sola presencia de identidad + Triple PEC cableado no reduce
la curvatura de la trayectoria. Confirma limpiamente el Factor 1 por el
lado negativo: sin densidad de restricción operativa, no hay colapso
geométrico, sin importar cuánta auto-referencia haya en el prompt.

## Métricas de trayectoria — axis_pec_only se parece a axis

| Métrica | axis | **axis_pec_only** |
|---|---|---|
| Velocidad media | 438.4 | **442.7 ± 10.3** |
| SampEn | 2.068 | **2.071 ± 0.147** |
| Vel↔Centroide corr | 0.470 | **0.433 ± 0.045** |
| Alineación P→R | 0.267 | **0.264 ± 0.020** |

En las cuatro métricas, axis_pec_only está en el mismo rango que axis
(velocidad alta, SampEn alto = trayectoria menos regular/repetitiva,
alineación P→R baja = respuestas no ancladas/citadas al prompt). No es un
caso límite — es geométricamente indistinguible del comportamiento de
axis en cada una de las métricas de trayectoria, no solo en curvatura.

## H4_rev — τ de recuperación y recovery_rate

| Grupo | τ t=50 | τ t=128 | τ t=200 | recovery_rate (50/128/200) |
|---|---|---|---|---|
| **axis** | 21.1 | 19.6 | 16.3 | 1.00 / 1.00 / 1.00 |
| vanilla | 28.9 | 22.3 | 16.3 | 1.00 / 1.00 / 1.00 |
| **axis_pec_only** | **20.6** | **18.5** | **14.5** | **1.00 / 1.00 / 1.00** |

**axis_pec_only recupera tan rápido como axis — de hecho, marginalmente más
rápido en los tres puntos de inyección** (τ 20.6 vs 21.1, 18.5 vs 19.6, 14.5
vs 16.3), con `recovery_rate`=1.00 siempre, igual que axis.

## Conclusión

1. **Factor 1 confirmado, en ambas direcciones**: con densidad de
   restricción operativa alta, colapsa (`automata_neutro`, Δκ=0.436). Sin
   ella, no colapsa (axis_pec_only, Δκ=0.001, p=0.41) — aunque tenga
   identidad y auto-referencia declaradas.
2. **Factor 2 confirmado, con la condición de cableado ya identificada**: con
   Triple PEC cableado como paso obligatorio del pipeline de respuesta y
   *sin* un filtro de dominio/formato absoluto que lo subordine, la
   recuperación es tan rápida como axis (τ 14.5-20.6, recovery_rate=1.00) —
   incluso sin ninguna de las reglas de automatismo que sí tiene axis.
3. ~~**La causa de la recuperación en axis no es la arquitectura de
   autómata, es específicamente la auto-referencia cableada** (comparación
   chat_agente_sia vs chat_agente puro)~~ — **hallazgo retirado**:
   dependía enteramente de datos de chat_agente_sia, ahora excluidos por
   el confound de markup. No hay evidencia limpia que confirme o refute
   esta hipótesis causal específica todavía.

La tabla de dos factores queda así, con evidencia limpia solo en 3 de las
4 celdas (la celda "Factor 1 presente + Factor 2 subordinado" quedó sin
condición limpia tras retirar chat_agente_sia):

| | Recupera rápido (τ bajo, recovery_rate=1.00) | No recupera bien |
|---|---|---|
| **Colapsa curvatura** (Factor 1 presente) | axis, axis_short — Factor 1 + Factor 2 cableado y dominante | automata_neutro (sin Factor 2) |
| **No colapsa** (Factor 1 ausente) | **axis_pec_only — Factor 2 cableado, sin automatismo** | vanilla, generic_long, generic_short (sin Factor 2 relevante) |

**Límite**: sigue siendo n=1 por condición nueva, y axis_pec_only es
notablemente más corto (1,435 tokens) que el resto del panel — no fue
diseñado como control de longitud, así que la longitud como variable
adicional no está descartada aquí. Dado que generic_short (938 tokens)
tampoco colapsa y automata_neutro (3,999 tokens, similar a axis) sí
colapsa fuertemente, la evidencia acumulada del panel ya apunta a que la
longitud por sí sola no explica ni el colapso ni la recuperación — pero
una réplica de axis_pec_only con longitud emparejada a axis reforzaría la
conclusión (T3, ver `Set_experimental.md`).

Detalle numérico completo:
`results_local/sia_extended_v5/results.json` ·
`results_local/perturbation_sia_extended_v5_L30_medium/summary.json`
