# LSGOT — Identidad vs. densidad de restricción (Gemma-4-31B-it, SIA)

Subconjunto curado de evidencia, scripts y resultados agregados que sostiene
`paper/lsgot_4.md` — una reformulación de `paper/lsgot_3.md` sobre un solo
modelo (Gemma-4-31B-it) y un diseño factorial explícito (identidad ×
densidad de restricción operativa) en vez del diseño de 3 condiciones /
4 modelos del paper original.

**No incluye embeddings ni trayectorias crudas** (`*_embeddings.npz`,
decenas de MB por condición, cientos de MB en total) — solo respuestas de
texto, JSON de resultados agregados y estadísticos ya computados. Para
reproducir desde cero, ver `scripts/` y la sección "Cómo reproducir" abajo.

Repositorio complementario, no reemplazado por este: [`lsgot_experiments`](https://github.com/axisdynamics/lsgot_experiments)
— el corpus MIA vs SIA cruzando 6 sustratos (incluye el control de
longitud E1 y el panel H4_rev original de 4 grupos, anterior a las
condiciones de este repo).

---

## Qué mide cada carpeta

```
paper/       lsgot_4.md (reformulación) y lsgot_3.md (paper anterior, referenciado como antecedente)
evidence/    15 reportes — diseño factorial, resultados por experimento, auditoría metodológica
data/        JSON agregados por experimento (sin *.npz): respuestas, curvatura, RQA/Hurst/PR, τ/recovery_rate, Fréchet
scripts/     pipeline de extracción, perturbación y análisis estadístico
```

## Diseño experimental

Nueve condiciones sobre un mismo modelo cruzan dos factores:

- **Identidad (I):** auto-referencia declarada y *cableada* como paso obligatorio del pipeline de respuesta.
- **Densidad de restricción (C):** reglas trigger→salida fija, filtro de prioridad absoluta, jerarquía de bloques, salidas verbatim.

| Condición | I | C |
|---|---|---|
| `vanilla`, `generic_long`, `generic_short` | − | − |
| `axis`, `axis_short` | + | + |
| `axis_pec_only` | + | − |
| `chileatiende`, `automata_neutro` | − | + |
| `chileatiende_sia`, `chileatiende_sia_v2` | + (declarada/cableada) | + |

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
   restricción sin identidad degrada la recuperación específica hasta en
   un 79% de los casos (`recovery_id` cae a 0.21 en `chileatiende_sia_v2`,
   t_inj=200).

Una tercera pieza — proyección sobre una dirección de identidad estilo
persona-vector — muestra una doble disociación limpia (identidad positiva,
restricción-sin-identidad negativa, d=8.89 entre `axis` y `chileatiende`,
el efecto más grande del panel) pero **no funciona como atractor
direccional**: perturbar a lo largo de esa dirección no produce
recuperación diferencial frente a perturbar ortogonalmente (`EI_PERMUTATION_REPORT.md`,
null result en las 6 condiciones probadas).

Tabla completa, significancia y limitaciones en `paper/lsgot_4.md`.

## Nota metodológica importante — Δκ/W₁ como evidencia secundaria

Este panel también reporta Δκ y W₁ (curvatura de Ollivier-Ricci sobre
grafos k-NN de trayectoria), la métrica central de `lsgot_3.md`. Se tratan
aquí como **evidencia secundaria, no primaria**: una auditoría
pre-registrada sobre un panel hermano de modelos (`evidence/REPORTE_FASE0.md`)
encontró que esa métrica no supera baselines distribucionales simples en
12/12 comparaciones y cae dentro del ruido de split-half en la comparación
de identidad específicamente. Esa auditoría **no se corrió sobre el panel
de este repo** — se reporta el hallazgo por transparencia, no se usa para
sostener ninguna conclusión de `lsgot_4.md`. Ver `paper/lsgot_4.md` §3.4 y §7.

## Cómo reproducir

Los scripts se incluyen **verbatim** (mismos que corrieron para producir la
evidencia en `evidence/` y `data/`) para auditoría metodológica — no todos
corren de punta a punta contra lo incluido en este repo, porque dos de
ellos requieren los `.npz` de embeddings crudos que se excluyeron
deliberadamente por tamaño.

**Corre tal cual, sin GPU, contra los datos incluidos:**
```bash
cd scripts/perturbation/
# RESULTS_DIR está hardcodeado a ./results/perturbation_ei_L30_medium/details.json
mkdir -p results/perturbation_ei_L30_medium
cp ../../data/perturbation_ei_L30_medium/details.json results/perturbation_ei_L30_medium/
python analyze_ei_permutation.py
```

**Requieren los `.npz` de embeddings (no incluidos) para volver a calcularse desde cero:**
`analyze_tier0.py` (espera `results_local/sia_extended_v5/*_embeddings.npz`) y
`analyze_tier0_perturbation.py` (espera además
`perturbation/results/perturbation_sia_L30_medium/trajectories/*_embeddings.npz`).
Los JSON/MD que sí están en `data/` y `evidence/` (`_tier0_metrics.json`,
`results.json`, `TIER0_REPORT.md`, `EE_EH_WINDOW_REPORT.md/json`) son la
**salida ya computada** de estos dos scripts, incluida como evidencia, no
como insumo para volver a correrlos. Ver `FUENTES.md` (no versionado, solo
en el árbol local) para la ruta exacta de origen de esos `.npz`.

**Nueva corrida completa** (requiere GPU A100/H100 80GB, `min_vram_gb=65`,
y un token de HuggingFace con acceso aceptado a `google/gemma-4-31B-it`):
```bash
cd scripts/perturbation/
python run_perturbation.py --exp sia --layer L30 --sigma medium --token hf_xxxxx
python run_perturbation_directional.py --token hf_xxxxx
```

## Licencia y contacto

Axis Dynamics SpA. Contacto: ver autoría en `paper/lsgot_4.md`.
