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
```

## Diseño experimental

Siete condiciones limpias sobre un mismo modelo cruzan dos factores:

- **Identidad (I):** auto-referencia declarada y *cableada* como paso obligatorio del pipeline de respuesta.
- **Densidad de restricción (C):** reglas trigger→salida fija, filtro de prioridad absoluta, jerarquía de bloques, salidas verbatim.

| Condición | I | C |
|---|---|---|
| `vanilla`, `generic_long`, `generic_short` | − | − |
| `axis`, `axis_short` | + | + |
| `axis_pec_only` | + | − |
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

Como **control fuera de diseño** (§3.8 de `lsgot_4.md`), se corrió la misma
batería sobre `soul_md_corto.md` ("Witness"), una entidad de identidad
densa tomada de `ADN_PERSONA_LI` — un proyecto hermano de este grupo de
investigación, redactado por otra persona sin conocimiento de este panel,
sobre la convención pública `SOUL.md`/[`soul-md`](https://github.com/Twynzen/soul-md)
(Twynzen) — sin ninguna línea de herencia textual con la redacción propia
de `axis`. Esto descarta que la señal de identidad sea un artefacto de
similitud semántica con el vocabulario de `axis`: es una identidad de
autoría, estructura y origen de plantilla independientes. Tiene identidad
declarada y auto-chequeo cableado, pero ninguna arquitectura de reglas
trigger→salida ni filtro de prioridad absoluta. En las tres señales de
identidad, en recuperación y en fidelidad de ruta converge con el perfil
de `axis_pec_only` — no con el de `automata_neutro` ni en ningún punto
intermedio — pese a estar en inglés y no haber sido diseñada para este
estudio.

Tabla completa, significancia y limitaciones en `paper/lsgot_4.md`.

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
