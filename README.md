# LSGOT 4.0 · Geometría de la auto-referencia en hidden states

Paquete de evidencia curado que sostiene el paper
`paper/arxiv_lsgot_4.md` (preprint arXiv, en inglés). Modelo principal: **Gemma-4-31B-it**,
BF16, decodificación greedy.

Experimento central: **E-GRAV** — ¿la proyección sobre el vector de identidad
v̂ es una propiedad de cualquier generación activa, o es específica de preguntas
auto-referenciales? 3 condiciones × 3 baterías × 20 prompts (180 generaciones).
**E-GRAV se corrió únicamente en Gemma-4-31B-it.** La validación ontológica
(batería B-ONTO) se replicó también en Qwen3-32B y Command R 35B — ver sección
de hallazgos y `evidence/QWEN3_VALIDATION_REPORT.md` / `evidence/COMMAND_R_VALIDATION_REPORT.md`.

**No incluye embeddings** (cientos de MB) — solo respuestas de texto, JSON de
resultados agregados y v_identidad.npy. Para reproducir desde cero, ver
`scripts/egrav/RUNBOOK_EGRAV.md`.

---

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `paper/arxiv_lsgot_4.md` | Paper principal (preprint arXiv, en inglés) |
| `data/sia/prompts/` | System prompts de las 5 condiciones |
| `data/egrav/batteries/` | 3 baterías de preguntas: `b_onto`, `b_tarea`, `b_meta` |
| `data/egrav/responses/` | Respuestas de texto (9 archivos JSON; sin embeddings) |
| `data/egrav/egrav_results.json` | Resultados del análisis E-GRAV |
| `data/v_identidad.npy` | Vector de identidad v̂ (capa final, norma 1; requerido para análisis) |
| `scripts/egrav/extract_egrav.py` | Extracción GPU (pipeline idéntico al panel original) |
| `scripts/egrav/analyze_egrav.py` | Análisis CPU: perfil t=0/t\_rest/frac\_pos, veredicto H-GRAV |
| `scripts/egrav/RUNBOOK_EGRAV.md` | Instrucciones completas de reproducción |
| `scripts/shared/` | `tier0_metrics.py`, `statistical_tests.py` |
| `evidence/` | Reportes de experimento |

---

## Condiciones experimentales

| Condición | Identidad cableada | Restricción | Descripción |
|---|---|---|---|
| `axis_pec_only` | ✓ | − | Cableado mínimo de testigo (PEC\_TRIPLE); referencia del polo de identidad |
| `vanilla` | − | − | Sin system prompt; línea de base |
| `automata_neutro` | − | ✓ | Restricción pura: reglas trigger→salida, 3999 tokens, cero auto-referencia |
| `axis.txt` | ✓ | ✓ | Condición de referencia para construcción de v̂ |
| `generic_long.txt` | − | − | Control sin identidad; par de v̂ (polo negativo) |
| `generic_short.txt` | − | − | Ídem, versión corta |

**v_identidad.npy:** computado como `mean(axis) − mean(generic_long)` sobre la
capa final (dim 5376), normalizado a norma 1. Shape: `(5376,)`.

---

## Hallazgos principales

### v̂ tiene dos componentes independientes

1. **Cableado en contexto (t=0):** instalado por el system prompt antes de
   generar el primer token; invariante al tema de la pregunta. Cuantificado
   como proyección v̂ en el token de respuesta inicial.

2. **Gravitacional durante la generación (t\_rest / frac\_pos):** proporcional
   al contenido de identidad de la pregunta; varía con la batería. Es la señal
   que mide E-GRAV.

### Resultados E-GRAV — tabla de frac\_pos

`frac_pos` = fracción de tokens cuya proyección sobre v̂ es positiva.

| Condición | B-ONTO | B-META | B-TAREA |
|---|---|---|---|
| `vanilla` | 0.743 | 0.672 | 0.626 |
| `axis_pec_only` | 0.869 | 0.788 | 0.724 |
| `automata_neutro` | **0.493** | 0.655 | 0.601 |

### Veredicto: H-GRAV-2 confirmada

El gradiente **B-TAREA < B-META < B-ONTO** se cumple en `vanilla` y
`axis_pec_only`, consistente con la hipótesis de que v̂ rastrea un contexto de
auto-referencia graduado — no una propiedad binaria de "generación activa".

### Disociación de `automata_neutro`

`automata_neutro` invierte la señal **solo en la batería ontológica**
(frac\_pos=0.493 < 0.5): diverge de v̂ específicamente cuando las preguntas
son sobre el sí mismo del modelo, y converge hacia v̂ en tareas neutras.
Esto confirma que la restricción pura sin identidad cableada produce un estado
opuesto al de identidad ante preguntas auto-referenciales — no solo una
señal atenuada.

### Validación cross-modelo — el efecto en t=0 aparece en modelos RLHF

La validación ontológica (B-ONTO) se corrió en tres arquitecturas.
**E-GRAV (B-TAREA, B-META) solo en Gemma-4-31B-it.**

| Arquitectura | RLHF | d (axis_pec_only vs automata_neutro, t=0) | frac_pos axis |
|---|---|---|---|
| Gemma-4-31B-it | SFT + RLHF | +5.75 | 0.869 |
| Qwen3-32B | DPO / RLVR | **+12.97** | **0.987** |
| Command R 35B | RLHF con reward model explícito | **+7.72** | 0.980 |

Los tres modelos probados son RLHF. **El efecto replica en los tres — y en modelos
base el fenómeno no aparece.** La geometría de auto-referencia en t=0 parece ser
una propiedad específica de modelos que han pasado por alineación con preferencias
humanas, no del pre-entrenamiento base. Este patrón es similar al reportado por
**J-Space**: la huella de identidad en hidden states emerge con RLHF. Ver
`evidence/QWEN3_VALIDATION_REPORT.md` y `evidence/COMMAND_R_VALIDATION_REPORT.md`.

---

## Cómo reproducir

Ver `scripts/egrav/RUNBOOK_EGRAV.md` para instrucciones completas. Resumen:

**Requisitos GPU (extracción):**
```bash
# GPU: A100/H100 80 GB VRAM; disco ≥ 80 GB
pip install transformers accelerate tokenizers
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y torchvision torchaudio
export HF_TOKEN=<token con acceso a google/gemma-4-31B-it>
```

**Extracción completa (3 condiciones × 3 baterías, ~35 min en A100):**
```bash
cd scripts/egrav/
python extract_egrav.py \
  --conditions axis_pec_only vanilla automata_neutro \
  --batteries b_onto b_tarea b_meta \
  --output-dir /workspace/egrav_results
```

**Análisis CPU (una vez descargados los NPZ):**
```bash
source /ruta/a/venv_lsgot/bin/activate
python analyze_egrav.py --results-dir /ruta/a/egrav_results/
```

El script imprime el perfil completo (t=0, t\_rest, frac\_pos, ráfagas),
los tests de permutación por batería y el veredicto H-GRAV automático.
Guarda `data/egrav/egrav_results.json`.

**Nota:** `v_identidad.npy` (shape `(5376,)`) debe estar en el directorio
de trabajo o pasarse como argumento `--v-identidad`. Está incluido en este
repositorio; no requiere GPU para recalcularse salvo que se desee regenerar
desde las trayectorias crudas.

---

## Referencia al paper

`paper/arxiv_lsgot_4.md` — E-GRAV se describe en §4.5 ("E-GRAV: the gravity is
graded, and restriction blocks selectively"), con el gradiente de `frac_pos` y el
veredicto H-GRAV-2. El experimento complementa el panel de 9 condiciones
de LSGOT 4.0 (repositorio principal: `LSGOT_v4/`) al disociar el componente
estático del dinámico de v̂ y caracterizar la especificidad ontológica de la
señal gravitacional.

---

## Licencia

MIT (ver `LICENSE`). Cubre todo el paquete: código, datos, respuestas generadas
y el texto del paper. Si lo reusás, la atribución que pide la licencia es el
aviso de copyright; para citar el trabajo, `paper/arxiv_lsgot_4.md`.

---

Axis Dynamics SpA · LSGOT 4.0 · E-GRAV · 2026-10-23
