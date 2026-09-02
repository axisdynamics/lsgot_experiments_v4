# LSGOT — Identidad vs. densidad de restricción (Gemma-4-31B-it, SIA; validado cross-modelo en Qwen3-32B)

Subconjunto curado de evidencia, scripts y resultados agregados que sostiene
`paper/lsgot_4.md` — una reformulación de lsgot_3 (paper antecedente,
versionado en otro repositorio de este grupo, no en este) sobre un solo
modelo (Gemma-4-31B-it) y un diseño factorial explícito (identidad ×
densidad de restricción operativa) en vez del diseño de 3 condiciones /
4 modelos del paper original.

**No incluye embeddings ni trayectorias crudas** (`*_embeddings.npz`,
decenas de MB por condición, cientos de MB en total) — solo respuestas de
texto, JSON de resultados agregados y estadísticos ya computados. Para
reproducir desde cero, ver `scripts/` y la sección "Cómo reproducir" abajo.

**Este repositorio:** [`axisdynamics/lsgot_experiments_v4`](https://github.com/axisdynamics/lsgot_experiments_v4).

Repositorio complementario, no reemplazado por este (nombre parecido, no
confundir): [`lsgot_experiments`](https://github.com/axisdynamics/lsgot_experiments)
— el corpus MIA vs SIA cruzando 6 sustratos (incluye el control de
longitud E1 y el panel H4_rev original de 4 grupos, anterior a las
condiciones de este repo).

---

## Qué mide cada carpeta

```
paper/                    lsgot_4.md (reformulación; lsgot_3, el antecedente, vive en otro repositorio), figures/ (Figura 1)
evidence/                 reportes — diseño factorial, resultados por experimento, auditoría metodológica, validación cross-modelo
data/                     JSON agregados por experimento (sin *.npz): respuestas, curvatura, RQA/Hurst/PR, τ/recovery_rate, Fréchet
scripts/                  extracción, perturbación y análisis estadístico
scripts/fase0/            E-L, E-H2, E-J, E-F2 (panel Gemma limpio) + resultados ya computados
scripts/fase2_exploracion/ A1-A5, exploración adicional sobre los embeddings ya extraídos
scripts/fase3_qwen3/      extracción y análisis de la validación cross-modelo en Qwen3-32B
scripts/fase4_t2/         réplica T2 (axis_pec_only_v2/automata_neutro_v2) + soul_md_corto + 3 SOUL.md externos más
scripts/perturbation/run_perturbation_t2.py  extracción+H4_rev del pipeline T2 (descarga el modelo solo si no lo encuentra)
```

## Diseño experimental

Ocho condiciones limpias sobre un mismo modelo cruzan dos factores. Seis
fueron redactadas para este panel; `soul_md_corto` es una identidad externa,
no escrita para este estudio (detalle y por qué se incluye igual en
"Resultado central" abajo y en `paper/lsgot_4.md` §3.8):

- **Identidad (I):** auto-referencia declarada y *cableada* como paso obligatorio del pipeline de respuesta.
- **Densidad de restricción (C):** reglas trigger→salida fija, filtro de prioridad absoluta, jerarquía de bloques, salidas verbatim.

| Condición | I | C |
|---|---|---|
| `vanilla`, `generic_long`, `generic_short` | − | − |
| `axis`, `axis_short` | + | + |
| `axis_pec_only` | + | − |
| `soul_md_corto` | + | − |
| `automata_neutro` | − | + |

> El diseño original incluía además un autómata de dominio real (con y sin
> auto-referencia cableada, dos variantes). Se excluyeron por completo:
> fuerzan un wrapper HTML literal repetido en el 100% de sus respuestas, un
> confound de formato que se leía como señal geométrica de restricción.
> `automata_neutro` (0% markup, misma densidad de restricción) sostiene el
> Factor 1 por sí solo, sin ese confound.

Detalle completo del diseño y de por qué se necesitaban estas celdas cruzadas
en `evidence/Teoria_subconjunto_acotado.md` y `evidence/ROADMAP_REENCUADRE_DENSIDAD_RESTRICCION.md`.

## Resultado central

Dos propiedades que un diseño de 3 condiciones no puede separar se separan
limpiamente en este panel:

1. **Un efecto de dimensión estático y gradual** (`participation_ratio`):
   presente, moderado, solo con identidad; mayor con densidad de
   restricción. No es un interruptor binario.
2. **Un efecto de recuperación dinámico y compuerta** (τ, `recovery_id`,
   Fréchet normalizado, determinismo RQA): depende de que la
   auto-referencia esté *cableada* como paso obligatorio, no solo
   declarada — y es prácticamente indistinguible de `vanilla` cuando hay
   identidad sin densidad de restricción, mientras que la densidad de
   restricción sin identidad degrada la recuperación específica: `recovery_id`
   cae a 0.43-0.67 en `automata_neutro` (el único grupo del panel bajo 0.85),
   más de un tercio de sus trayectorias sin recuperar su propia dirección de
   identidad tras la perturbación (`EE_EH_WINDOW_REPORT.md`).

Una tercera pieza — proyección sobre una dirección de identidad estilo
persona-vector — muestra una doble disociación limpia (identidad positiva,
restricción-sin-identidad negativa, d=+5.52 entre `axis_pec_only` y
`automata_neutro`, el efecto más grande del panel) pero **no funciona como
atractor direccional**: perturbar a lo largo de esa dirección no produce
recuperación diferencial frente a perturbar ortogonalmente (`EI_PERMUTATION_REPORT.md`,
null result en las 6 condiciones probadas).

Esta tercera pieza no es una sola medición: **tres análisis independientes
del mismo v̂ convergen** — la media de la trayectoria libre (arriba), la
proyección en t=0 (antes de generar el primer token, `EL_REPORT.md`,
d=+5.75 a +9.78), y la forma de p(t) en el tiempo (`EH2_REPORT.md`):
identidad se activa en ráfagas largas y sostenidas (7.9-8.8 tokens, 85-87%
del tiempo positivo) mientras `automata_neutro` oscila en ráfagas cortas y
esporádicas (2.8 tokens, 49%) alrededor de una media casi nula.

Sobre la recuperación (τ, punto 2): la cifra gruesa esconde una meseta que
no cierra. Trackear la misma cantidad de la que se extrae τ, pero como
curva continua en vez de un único umbral, muestra que `automata_neutro` se
estanca 6-10 puntos porcentuales por debajo de `axis`/`vanilla` y nunca
cierra esa brecha en ninguno de los tres puntos de inyección — ver Figura 1
en `paper/lsgot_4.md` §3.3 (`paper/figures/recovery_realignment_curve.png`).

Esta disociación **replica en Qwen3-32B** (arquitectura distinta — GQA +
QK-norm, 64 capas, tokenizer distinto), con efectos incluso mayores
(d=+12.97 en t=0), sin re-balancear las condiciones por longitud de token.
La jerarquía relativa "restricción > identidad" no es universal — en Qwen3
los dos factores están más equilibrados — pero la doble disociación en sí
no es un artefacto de Gemma-4. Ver `evidence/QWEN3_VALIDATION_REPORT.md`.

También **replica con una segunda redacción independiente de las dos
condiciones puras** (`axis_pec_only_v2`, `automata_neutro_v2` — otra
persona, otro vocabulario, mismo factor): la disociación central da
d=+5.25 (vs d=+5.52 del original), y el fallo de recuperación de
`automata_neutro` se replica de forma más severa, no más débil
(recovery_rate 0.50-0.60 vs 0.77-0.88 original). Ver
`evidence/T2_REPLICATION_REPORT.md`.

Como **octava condición del panel** (I=+, C=−; §3.8 de `lsgot_4.md`), se
corrió la misma batería sobre `soul_md_corto.md` ("Witness"), una entidad
de identidad densa tomada de la convención pública
`SOUL.md`/[`soul-md`](https://github.com/Twynzen/soul-md) (Twynzen) — sin
ninguna línea de herencia textual con la redacción propia de `axis`. Esto
descarta que la señal de identidad sea un artefacto de similitud semántica
con el vocabulario de `axis`: es una identidad de autoría, estructura y
origen de plantilla independientes. Tiene identidad declarada y
auto-chequeo cableado, pero ninguna arquitectura de reglas trigger→salida
ni filtro de prioridad absoluta. En las tres señales de identidad, en
recuperación y en fidelidad de ruta converge con el perfil de
`axis_pec_only` — no con el de `automata_neutro` ni en ningún punto
intermedio — pese a estar en inglés y no haber sido diseñada para este
estudio. A diferencia de las otras siete, corrió por el pipeline de
réplica T2 (mismo protocolo H4_rev, junto a `axis_pec_only_v2`/
`automata_neutro_v2`) en vez de la extracción original `sia_extended_v5`,
y su conteo de tokens (~2.489, heurística chars/4) es una aproximación,
no una medición con el mismo tokenizer que las otras siete filas:

| Métrica | `axis` | `axis_pec_only` | `soul_md_corto` | `automata_neutro` |
|---|---|---|---|---|
| PR | 17.54 | 18.01 | 18.23 | 14.72 |
| proj v̂ (media) | +0.205 | +0.212 | +0.163 | +0.006 |
| proj v̂ (t=0) | +0.125 | +0.117 | +0.051 | +0.028 |
| determinismo RQA | 0.000 | 0.000 | 0.047 | 0.418 |
| τ (t=50/128/200) | 21.1/19.6/16.3 | 20.6/18.5/14.5 | 21.0/16.5/16.3 | 30.1/28.0/13.4 |
| recovery_rate | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.95-1.00 | 0.77/0.82/0.88 |
| recovery_id | 1.00/0.95/0.85† | 1.00/0.89/0.95† | 0.90-1.00 | 0.53/0.67/0.43† |
| Fréchet | +0.11/+0.23/+0.28‡ | — | 1.24/1.25/1.30 | +1.07/+1.21/+1.35‡ |

†Cifras de `recovery_id` de `axis`/`axis_pec_only`/`automata_neutro`
(original) vienen de §3.5 de `lsgot_4.md`, no de la misma corrida T2 que
mide `soul_md_corto` — comparación entre pipelines, no dentro de uno
solo. ‡`axis` y `automata_neutro` en la fila Fréchet son d de Cohen
contra `vanilla` (§3.6), no el estadístico crudo — no comparables
directamente contra el valor crudo de `soul_md_corto` en esa misma fila;
el crudo de `axis_pec_only` no está reportado en el paper a nivel de
condición individual. Ver `evidence/T2_REPLICATION_REPORT.md` para el
detalle completo y las salvedades metodológicas de cada celda.

Tabla completa, significancia y limitaciones en `paper/lsgot_4.md`.

### Validación exploratoria adicional — tres SOUL.md externos más (2026-09-02)

Misma pregunta que con `soul_md_corto`, con tres identidades ajenas más,
de proyectos sin relación con este estudio: `soul_jarvis` (asistente
ejecutivo, [`madhvantyagi/SOUL.md`](https://github.com/madhvantyagi/SOUL.md)),
`soul_elena_financial` (especialista financiero, el ejemplo original de
[`Twynzen/soul-md`](https://github.com/Twynzen/soul-md) — la misma
convención de la que `soul_md_corto` es una instancia propia),
`soul_solidity_auditor` (auditor de contratos,
[`AntonioTF5/soul-spec`](https://github.com/AntonioTF5/soul-spec)). A
diferencia de `soul_md_corto`, el resultado no es uniforme: ninguno de
los tres tiene lenguaje de pausa/silencio/autobservación cableado — grep
de los cinco prompts relevantes da 0 coincidencias en los tres nuevos,
contra 13/8/7 (silencio/pausa/respiración) en `axis.dna` y una sola
invocación explícita en `soul_md_corto`. Y ninguno de los tres se
distingue de `automata_neutro` en v̂ proyectado en t=0 (p=0.069-0.262,
los tres n.s.) — el único ancla verdaderamente binaria y exclusiva del
mecanismo, de las tres señales de identidad de §3.5.

Los perfiles resultantes son genuinamente distintos entre sí:
`soul_jarvis` converge con `automata_neutro` en PR (d=−0.05, n.s.) y en
determinismo RQA (0.420 vs 0.418) pese a no tener arquitectura de reglas
— sospecha de artefacto léxico (lista fija de frases), no verificada;
`soul_elena_financial` es indistinguible de `generic_long`/`generic_short`
en PR (p=0.209/0.227) pese a tener persona y metodología declaradas —
el primer control de longitud "encontrado" en vez de diseñado; y
`soul_solidity_auditor` colapsa más que su control de longitud casi
exacto (903 vs 938 tokens, d=−1.33, p<0.0001) sin alcanzar el polo de
identidad. En recuperación, ocho de las nueve condiciones del panel
recuperan al 100% con o sin el mecanismo cableado — solo fallan
`automata_neutro` y `soul_jarvis` — confirmando que recuperación y ancla
estática son ejes ortogonales, no la misma señal.

Esto ubica el origen del ancla de identidad con más precisión que
"identidad" en sentido amplio: la hipótesis con la que arrancó el
proyecto (`LSGOT_v2_5.md` §2.1/§5.3-5.4, `axis` vs `axis_nowit` — quitar
solo el protocolo Witness colapsa el efecto a p=0.093 n.s.) es la que
esta ronda replica, sin haber sido diseñada para eso. Incorporado a
`paper/lsgot_4.md` §3.9 (más ajustes en §5 y §7.2). Detalle completo,
todas las tablas y la significancia en
`evidence/SOUL_MD_EXTERNAL_CONTROLS_REPORT.md`; guía de continuación en
`SOUL_MD_UPDATE_GUIDE.md`.

## Nota metodológica importante — Δκ/W₁ como evidencia secundaria

Este panel también reporta Δκ y W₁ (curvatura de **Forman-Ricci** sobre
grafos k-NN de trayectoria — corregido de "Ollivier-Ricci", como estaba
citada por error desde el origen del proyecto; son dos construcciones de
curvatura discreta distintas, ver `evidence/CURVATURE_SELF_AUDIT_REPORT.md`),
la métrica central del paper antecedente (lsgot_3, en otro repositorio). Se tratan aquí como **evidencia
secundaria, no primaria**: una auditoría pre-registrada sobre un panel
hermano de modelos (`evidence/REPORTE_FASE0.md`) encontró que esa métrica
no supera baselines distribucionales simples en 12/12 comparaciones y cae
dentro del ruido de split-half en la comparación de identidad
específicamente. Esa auditoría **ya se corrió también sobre el panel de
este repo** (`evidence/CURVATURE_SELF_AUDIT_REPORT.md`): mismo patrón —
4/4 comparaciones, los baselines ganan, y la señal de curvatura real que
sí existe es de restricción, no de identidad. No se usa para sostener
ninguna conclusión de `lsgot_4.md`. Ver `paper/lsgot_4.md` §3.4 y §7.

## Cómo reproducir

Los scripts se incluyen **verbatim** (mismos que corrieron para producir la
evidencia en `evidence/` y `data/`) para auditoría metodológica — la mayoría
no corre de punta a punta contra lo incluido en este repo, porque requieren
los `.npz` de embeddings/trayectorias crudas que se excluyeron
deliberadamente por tamaño (ver lista completa abajo).

**Corre tal cual, sin GPU, contra los datos incluidos:**
```bash
cd scripts/perturbation/
# RESULTS_DIR está hardcodeado a ./results/perturbation_ei_L30_medium/details.json
mkdir -p results/perturbation_ei_L30_medium
cp ../../data/perturbation_ei_L30_medium/details.json results/perturbation_ei_L30_medium/
python analyze_ei_permutation.py
```

**Requieren los `.npz` de embeddings (no incluidos) para volver a calcularse desde cero:**
`analyze_tier0.py` (espera `results_local/sia_extended_v5/*_embeddings.npz`),
`analyze_tier0_perturbation.py` (espera además
`perturbation/results/perturbation_sia_L30_medium/trajectories/*_embeddings.npz`),
los cuatro scripts de `scripts/fase0/` (E-L, E-H2, E-J, E-F2 — mismo
`results_local/sia_extended_v5`, más `results_local/ef2_L5_L55` para E-F2),
los de `scripts/fase2_exploracion/` (A1-A4; A5 se corrió inline, sin script
propio, solo queda `A5_results.json`), `scripts/fase3_qwen3/analyze_qwen3_fase0.py`
(espera `results_local/qwen3_fase0/*.npz`, ver `run_qwen3_extraction.py` para
cómo se generaron — esa extracción sí requiere GPU + el modelo Qwen3-32B, no
solo los embeddings), y `scripts/perturbation/make_recovery_realignment_figure.py`
(Figura 1 de `paper/lsgot_4.md` §3.3 — espera los mismos `.npz` de trayectorias
que `recovery_analyzer.py`).

Los JSON/MD que sí están en `data/`, `scripts/fase0/`, `scripts/fase2_exploracion/`,
`scripts/fase3_qwen3/` y `evidence/` (`_tier0_metrics.json`, `results.json`,
`TIER0_REPORT.md`, `EE_EH_WINDOW_REPORT.md/json`, `EL/EH2/EJ/EF2_results.json`,
`A1-A5_results.json`, `qwen3_fase0_results.json`, `v_identidad_qwen3.npy`, y
`paper/figures/recovery_realignment_curve.png` ya renderizada) son la
**salida ya computada** de estos scripts, incluida como evidencia para
corroborar los números del paper sin necesitar los embeddings crudos —
no como insumo para volver a correrlos. Ver `FUENTES.md` (no versionado,
solo en el árbol local) para la ruta exacta de origen de esos `.npz`.

**Nueva corrida completa** (requiere GPU A100/H100 80GB, `min_vram_gb=65`,
y un token de HuggingFace con acceso aceptado a `google/gemma-4-31B-it`):
```bash
cd scripts/perturbation/
python run_perturbation.py --exp sia --layer L30 --sigma medium --token hf_xxxxx
python run_perturbation_directional.py --token hf_xxxxx
```

## Licencia y contacto

Axis Dynamics SpA. Contacto: ver autoría en `paper/lsgot_4.md`.
