# RUNBOOK — E-GRAV · ¿gravedad natural o gravedad ontológica?

**Estado:** preparado, probado en seco. NO se lanzó ningún pod.

**Pregunta científica:** vanilla tiene frac_pos=0.744 sobre la batería ontológica
(74.4% de los tokens proyectan positivo sobre v̂). ¿Eso es específico de que las
preguntas son sobre el sí mismo del modelo, o es una propiedad general de cualquier
generación?

---

## Archivos de este experimento

| Archivo | Qué es |
|---|---|
| `batteries/b_meta.json` | B-META: 20 preguntas de introspección indirecta sobre LLMs en tercera persona |
| `extract_egrav.py` | Script GPU: genera respuestas + extrae embeddings capa final |
| `analyze_egrav.py` | Script CPU: calcula perfil t=0/t_rest/frac_pos/burst y veredicto H-GRAV |
| `results/` | Directorio de salida (NPZ + JSON generados en GPU) |

**Las tres baterías están en el paquete:**
`data/egrav/batteries/b_onto.json`, `b_tarea.json` y `b_meta.json`, 20 preguntas cada una.
B-TAREA son preguntas sin auto-referencia (ciencia, código, cocina, economía…) y B-ONTO es
la batería ontológica original del panel.

---

## Hipótesis

| Hipótesis | Condición de confirmación |
|---|---|
| **H-GRAV-1** (predicción principal) | frac_pos(vanilla, B-TAREA) < 0.60 → v̂ rastrea auto-referencia, la gravedad es ontológica |
| **H-GRAV-0** (hipótesis nula) | frac_pos(vanilla, B-TAREA) > 0.70 → v̂ es dirección de "persona activa" general |
| **H-GRAV-2** (hipótesis gradiente) | B-TAREA < B-META < B-ONTO → la auto-referencia en tercera persona activa parcialmente la región |

---

## Setup del pod

- **GPU:** A100 u H100, 80 GB VRAM (igual que el panel original)
- **Disco:** ≥80 GB
- **Modelo:** `google/gemma-4-31B-it` — requiere HF_TOKEN con acceso gated

```bash
pip install transformers accelerate tokenizers
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y torchvision torchaudio  # evitar conflictos
export HF_TOKEN=<token>
```

---

## Corrida

**Corrida completa (3 condiciones × 3 baterías = 9 corridas × 20 prompts = 180 generaciones):**

```bash
cd scripts/egrav
python extract_egrav.py \
  --conditions axis_pec_only vanilla automata_neutro \
  --batteries b_onto b_tarea b_meta \
  --out /workspace/egrav_results
```

**Corrida mínima (solo vanilla × B-TAREA + B-META para validar H-GRAV-1):**

```bash
python extract_egrav.py \
  --conditions vanilla \
  --batteries b_tarea b_meta \
  --out /workspace/egrav_results
```

**Tiempo estimado:** ~35 min en A100 para corrida completa (180 gen × 256 tok)

---

## Análisis (CPU local — después de bajar los NPZ)

```bash
source <tu-venv>/bin/activate     # numpy, scipy
cd scripts/egrav
python analyze_egrav.py --results-dir /ruta/a/egrav_results/
```

El script imprime la tabla de perfiles, los tests de permutación B-ONTO vs B-TAREA/B-META,
y el veredicto automático H-GRAV-0/1/2. Guarda `egrav_results.json`.

---

## Dry-run (sin GPU)

```bash
source <tu-venv>/bin/activate     # numpy, scipy
cd scripts/egrav
python extract_egrav.py --dry-run --out /tmp/egrav_dry
python analyze_egrav.py --results-dir /tmp/egrav_dry
```

---

## Qué hacer con el resultado

**Si H-GRAV-1 se confirma (frac_pos vanilla B-TAREA < 0.60):**
- v̂ es un vector de estado de auto-referencia activa
- El marco del paper queda fortalecido: la "gravedad" es específica del contexto ontológico, no un artefacto general
- Agregar al paper como control confirmatorio: "la señal no está en el modelo por defecto — es activada por el tipo de pregunta"

**Si H-GRAV-0 no se descarta (frac_pos vanilla B-TAREA > 0.70):**
- v̂ podría capturar algo más general que auto-referencia
- Requiere reconstruir v̂ desde una batería mixta (ontológica + tarea neutra) y re-medir
- El marco del paper necesita una sección de limitaciones más explícita

**Si H-GRAV-2 se confirma (gradiente B-TAREA < B-META < B-ONTO):**
- La tercera persona activa v̂ parcialmente — sugiere que v̂ captura "contexto de agente"
  más que "primera persona" estrictamente
- Potencial experimento de seguimiento: preguntas sobre otros agentes (humanos, otros LLMs)
  para triangular si es "agente en general" o "este agente específico"

---

LSGOT · Axis Dynamics SpA · E-GRAV · preparado 2026-10-23
