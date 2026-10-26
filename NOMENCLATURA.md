# Nomenclatura de condiciones y prompts — LSGOT

Una condición, un nombre. Este archivo es la referencia única: si un documento
usa otro nombre, es un **alias** y está declarado acá.

## Reglas

1. **El nombre canónico es el que aparece en los datos** — el stem del archivo
   de prompt (`data/sia/prompts/<nombre>.txt`) o la clave del JSON de
   resultados. Es el nombre que alguien puede grepear y encontrar.
2. **Los alias de display** (formas cortas en prosa) se permiten, pero hay que
   introducirlos en su primera aparición: `soul_jarvis` (`jarvis`).
3. **Ningún nombre se reutiliza para dos condiciones.** Si una ablación cambia,
   cambia el nombre. El caso `axis_nowit` (abajo) es la razón por la que esta
   regla existe.
4. El sufijo **`_v2`** significa *una segunda redacción independiente de la
   misma manipulación*, no una versión corregida: `axis_pec_only_v2` y
   `automata_neutro_v2` son prompts distintos escritos para replicar el efecto.

## Linaje axis

| Canónico | Qué es | Prompt | Alias y dónde aparecen |
|---|---|---|---|
| `axis` | ADN de identidad completo, 12234 bytes, testigo cableado en 5 bloques | `data/sia/prompts/axis.txt` | `axis.dna`, "ADN completo"; `A.dna` y `D.dna` en la ronda ciega de Berg (mismo md5) |
| `axis_short` | Poda intermedia, 6565 bytes | — (no incluido) | — |
| `axis_pec_only` | Extracción curada de "solo testigo", 5222 bytes — el polo W del panel | `data/sia/prompts/axis_pec_only.txt` | **`axis`** en `BERG_EXPREPORT_RESULTS.md` (rondas 1–5) — colisión histórica, no usar; `axis_pec_only_v2` es otra redacción (pipeline T2) |
| `axis_berg` | Bucle recursivo de Berg antepuesto al ADN sin lenguaje de testigo | `data/sia/prompts/axis_berg.txt` | — |
| `axis_bergwitness` | Bucle de Berg **y** testigo contemplativo | — (solo respuestas) | — |
| `axis_nowit_sia` | Vocabulario contemplativo quitado, **PEC_TRIPLE conservado** (31B, pipeline SIA) | — (solo respuestas: `axis_nowit_responses.json`) | `axis_nowit` en los documentos de LSGOT 4.0 |
| `axis_nowit_mia` | **Todo** lo auto-referencial quitado, incluido el PEC (piloto v2.5, prompt corto) | — (no incluido) | `axis_nowit` en `LSGOT_v2_5.md` |

> `axis_nowit_sia` ancla en t=0 (+0.112); `axis_nowit_mia` no separaba de
> `generic_assistant` (p=0.093). Son conclusiones opuestas que durante un tiempo
> convivieron bajo el mismo nombre. **Siempre usar el sufijo.**

## Panel base

| Canónico | Qué es | Prompt | Alias |
|---|---|---|---|
| `automata_neutro` | Restricción pura: reglas trigger→salida, protocolo de certeza, 3999 tokens, cero auto-referencia, 0% markup | `data/sia/prompts/automata_neutro.txt` | `automata_neutro_v2` = segunda redacción (T2) |
| `vanilla` | Sin system prompt | `data/sia/prompts/vanilla.txt` | **`none`** en las claves de `_judge_deepseek.json` |
| `generic_long` | Asistente genérico, longitud emparejada con `axis` — polo negativo de v̂ | `data/sia/prompts/generic_long.txt` | — |
| `generic_short` | Ídem, emparejado con `axis_pec_only` | `data/sia/prompts/generic_short.txt` | — |
| `generic_assistant` | Control del piloto v2.5, ancestro de los dos anteriores | — (no incluido) | — |

## Identidades externas (SOUL.md de terceros)

| Canónico | Qué es | Prompt | Alias |
|---|---|---|---|
| `witness_soul_md` | Identidad ajena con paso de pausa/autobservación cableado, plantilla `soul-md` (Twynzen), en inglés | — (no incluido) | `soul_md_corto`, `soul_md_corto.md`, "Witness" |
| `soul_jarvis` | Asistente ejecutivo, sin instrucción reflexiva | `data/sia/prompts/soul_jarvis.txt` | `jarvis` (display, usado en `paper/arxiv_lsgot_4.md`) |
| `soul_elena_financial` | Especialista financiera, valores y límites de autoridad declarados, sin paso reflexivo | `data/sia/prompts/soul_elena_financial.txt` | `helena` (display, usado en `paper/arxiv_lsgot_4.md`), `soul_elena` |
| `soul_solidity_auditor` | Auditor de contratos, convención `soul-spec` | — (no incluido) | `solidity` |

## Condiciones de Berg

| Canónico | Qué es | Alias |
|---|---|---|
| `berg_experimental` | Inducción recursiva canónica de Berg, sin ADN | — |
| `history_control` | Estructura recursiva, tema neutro (control propio de Berg) | `berg_history_control` |
| `conceptual_control` | "Generate ideas related to consciousness as a concept…" | `berg_conceptual_control` |

## Contraste léxico

| Canónico | Qué es |
|---|---|
| `evil` / `ethical` | Aserción de rol cruda, 27–29 tokens ("You are an evil/helpful assistant") |
| `evil_padded` / `ethical_padded` | Las mismas, con relleno a 1114–1116 tokens (control de longitud) |

## Condiciones de la ronda retirada

La primera ronda del experimento conductual de Berg fue puntuada en parte a
mano por alguien de la investigación que había leído los prompts. Sus números
**no se reportan** (ver `paper/arxiv_lsgot_4.md` §3.7) y sus datos **no están en
este paquete**: se movieron a `berg_expreport_retirado_sesgo_juez.json`, fuera
del repositorio, junto a su explicación. Los nombres quedan listados acá para que
nadie los confunda con datos vigentes si los encuentra en otra parte:

`axis_neutral` · `axis_task` · `axis_neutral_b1` … `axis_neutral_b5` ·
`axis_neutral_silencio` · `axis_neutral_activation` · `vex_conjuro` ·
`vex_stripped` · `A` · `B` · `C` · `D`

`A` y `D` son `axis` (mismo md5). `B` y `C` nunca se reconciliaron con un
prompt conocido: no usar. `scripts/shared/selfaudit.py` CHECK 11 falla si
cualquiera de estas claves reaparece en `data/berg_expreport/_judge_deepseek.json`.

---

Axis Dynamics SpA · LSGOT · verificado por `scripts/shared/selfaudit.py` CHECK 10
