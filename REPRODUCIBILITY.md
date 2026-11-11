# Guía de Reproducibilidad — LSGOT 4.0

**Paper:** `paper/arxiv_lsgot_4.md`  
**Modelo:** `google/gemma-4-31B-it`, BF16, decodificación greedy, `max_new_tokens=256`  
**Estado del repositorio:** curado para publicación; todos los resultados JSON están incluidos.

---

## Requisitos previos

### Entorno Python local (CPU)

El entorno de análisis local ya está instalado en:

```bash
source "$VENV/bin/activate"      # numpy, scipy
```

> ⚠️ Usar **siempre** `venv_lsgot`, no el Python del sistema ni `venv_analysis` (este último no tiene `scipy`).

Verificar que el entorno tiene las dependencias necesarias:

```bash
python -c "import numpy, scipy; print('OK')"
```

### Raíz del repositorio

Todos los comandos de esta guía asumen que la raíz del repositorio es:

```
REPO=$(git rev-parse --show-toplevel)   # o la ruta de tu clon
```

Conviene exportar esta variable al inicio de cada sesión:

```bash
export REPO=$(git rev-parse --show-toplevel)   # o la ruta de tu clon
```

### Verificar `v_identidad.npy`

El vector identidad `v̂` es central en todos los experimentos. Antes de cualquier análisis:

```bash
source "$VENV/bin/activate"      # numpy, scipy
python - <<'EOF'
import numpy as np
v = np.load("$REPO/data/v_identidad.npy")
assert v.shape == (5376,), f"Shape incorrecto: {v.shape}"
assert abs(np.linalg.norm(v) - 1.0) < 1e-4, "v̂ no está normalizado"
print(f"v̂ OK — shape={v.shape}, norm={np.linalg.norm(v):.6f}")
EOF
```

Salida esperada: `v̂ OK — shape=(5376,), norm=1.000000`

> **Nota técnica:** La dimensión oculta de Gemma-4-31B-it (capa final post-norm) es **5376**, no 5120. Cualquier script externo debe usar 5376 o el producto punto fallará.

---

## Sección 1 — Verificar resultados sin GPU

> **CPU únicamente.** No se requiere modelo, GPU ni descarga de pesos.

Todos los resultados numéricos del paper están almacenados en los JSON del repositorio. Esta sección describe cómo verificarlos y cómo re-ejecutar los análisis estadísticos sobre los embeddings ya extraídos.

### 1.1 Panel de 9 condiciones — t=0, t_rest, proyección media

**Archivo fuente:** `data/panel_results/t2_free_results.json`  
Contiene: `t0`, `proj_mean` para axis, axis_pec_only, automata_neutro, vanilla, generic_long, generic_short, soul_jarvis, soul_elena_financial.

Verificar los valores t=0 citados en el paper:

```bash
source "$VENV/bin/activate"      # numpy, scipy
cd $REPO

python - <<'EOF'
import json

with open("data/panel_results/t2_free_results.json") as f:
    d = json.load(f)

# Valores t=0 esperados (paper Tabla 2)
expected = {
    "axis":                 +0.1249,
    "axis_pec_only":        +0.1166,
    "soul_jarvis":          +0.0394,
    "soul_elena_financial": +0.0321,
    "automata_neutro":      +0.0282,
    "vanilla":              -0.0286,
    "generic_long":         -0.0251,
    "generic_short":        -0.0453,
}

print(f"{'Condición':<25} {'t=0 JSON':>10}  {'t=0 paper':>10}  {'Δ':>8}")
print("-" * 60)
for cond, expected_val in expected.items():
    actual = d["summary"][cond]["t0_mean"]
    delta = actual - expected_val
    status = "✓" if abs(delta) < 0.001 else "⚠"
    print(f"{status} {cond:<23} {actual:>+10.4f}  {expected_val:>+10.4f}  {delta:>+8.4f}")
EOF
```

**Salida esperada:** todas las filas con `✓` y `Δ < 0.001`.

### 1.2 Panel de 9 condiciones — berg controls y t_rest

**Archivo fuente:** `data/panel_results/berg_controls_t0_results.json`  
Contiene: `t0`, `trest`, `traj` para axis_berg, axis, axis_pec_only, vanilla, generic_long, generic_short, automata_neutro.

```bash
python - <<'EOF'
import json

with open("data/panel_results/berg_controls_t0_results.json") as f:
    d = json.load(f)

print(f"{'Condición':<20} {'t=0':>8}  {'t_rest':>8}")
print("-" * 42)
for cond, vals in d["el_table"].items():
    t0   = vals["t0"]["mean"]
    trest = vals["trest"]["mean"]
    print(f"  {cond:<18} {t0:>+8.4f}  {trest:>+8.4f}")
EOF
```

### 1.3 Métricas de ráfaga — EH2 (burst_mean, frac_pos, max_burst)

**Archivo fuente:** `data/panel_results/EH2_results.json`  
Contiene métricas de serie temporal para axis, axis_pec_only, automata_neutro, vanilla, generic_long, generic_short.

```bash
python - <<'EOF'
import json

with open("data/panel_results/EH2_results.json") as f:
    d = json.load(f)

series = d["series_metrics"]["con_t0"]  # incluye t=0 en la trayectoria
print(f"{'Condición':<20} {'frac_pos':>9}  {'burst_mean':>11}  {'max_burst':>10}")
print("-" * 58)
for cond, m in series.items():
    fp  = m["frac_positive"]["mean"]
    bm  = m["mean_burst_len"]["mean"]
    mbm = m["max_burst_len"]["mean"]
    print(f"  {cond:<18} {fp:>9.3f}  {bm:>11.2f}  {mbm:>10.1f}")
EOF
```

### 1.4 Métricas de ráfaga — souls (soul_jarvis, soul_elena_financial)

**Archivo fuente:** `data/panel_results/t2_eh2_results.json`

```bash
python - <<'EOF'
import json

with open("data/panel_results/t2_eh2_results.json") as f:
    d = json.load(f)

print(f"{'Condición':<25} {'burst_mean':>11}  {'frac_pos':>9}")
print("-" * 50)
for cond, m in d.items():
    bm = m.get("burst_mean", {}).get("mean", float("nan"))
    fp = m.get("frac_pos",   {}).get("mean", float("nan"))
    print(f"  {cond:<23} {bm:>11.2f}  {fp:>9.3f}")
EOF
```

## Sección 2 — Reproducir la extracción con GPU

> ⚠️ **Requiere GPU:** A100 u H100, 80 GB VRAM. Se ejecuta en RunPod.

`scripts/extraction/extract_trajectories.py` genera las respuestas y extrae los embeddings
para las N=20 preguntas ontológicas del panel. Con las tres condiciones por defecto
(`axis_pec_only`, `vanilla`, `automata_neutro`) son 60 generaciones y reproduce el contraste
central del paper. La proyección sobre v̂ es CPU y va en §2.5. El panel completo de 9
condiciones está en §3.

### 2.1 Configurar el pod RunPod

**Requisitos del pod:**
- GPU: A100 o H100, 80 GB VRAM
- Disco: ≥ 120 GB (modelo ~62 GB + workspace)
- CUDA: 12.4
- Python: 3.10+

```bash
# En el pod (terminal RunPod):
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install transformers accelerate huggingface_hub tokenizers scipy numpy
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y torchvision torchaudio   # evitar Gemma4Config import conflict
```

**Configurar HF_TOKEN** (desde la terminal SSH del usuario, nunca a través de Hermes):

```bash
# Desde tu terminal local:
ssh root@<POD_IP> -p <PORT> "echo 'export HF_TOKEN=<tu_token>' > /workspace/.env"
```

### 2.2 Subir los archivos al pod

Desde la máquina local:

```bash
# Subir scripts, preguntas y v̂  (ojo: -P mayúscula es el puerto en scp, -p no)
scp -P <PORT> -r $REPO/scripts/extraction root@<POD_IP>:/workspace/lsgot_extract/
scp -P <PORT> -r $REPO/scripts/shared     root@<POD_IP>:/workspace/lsgot_extract/
scp -P <PORT>    $REPO/data/v_identidad.npy root@<POD_IP>:/workspace/lsgot_extract/shared/

# Subir los prompts de las condiciones a correr
mkdir -p /tmp/prompts_lsgot
cp $REPO/data/sia/prompts/axis_pec_only.txt   /tmp/prompts_lsgot/
cp $REPO/data/sia/prompts/automata_neutro.txt /tmp/prompts_lsgot/
scp -P <PORT> -r /tmp/prompts_lsgot root@<POD_IP>:/workspace/lsgot_extract/extraction/prompts/
```

### 2.3 Ejecutar la extracción GPU

```bash
# En el pod:
source /workspace/.env
source /workspace/venv/bin/activate
cd /workspace/lsgot_extract/extraction

# Corrida completa (3 condiciones × 20 preguntas ≈ 12 min en A100)
python extract_trajectories.py --out /workspace/lsgot_results

# Alternativa: smoke test (2 prompts por corrida, ~5 min)
python extract_trajectories.py --out /workspace/lsgot_smoke --smoke
```

**Comportamiento esperado por corrida:**
- Imprime `[01/20] id=... T=<longitud>` para cada prompt.
- Guarda `<cond>_b_onto_embeddings.npz` y `<cond>_b_onto_responses.json` en `/workspace/lsgot_results/`.
- Es reanudable: si una corrida ya existe, la salta. Usar `--force` para recalcular.

**Notas críticas para el extractor:**
- `transformers ≥ 5.x`: `apply_chat_template` devuelve `BatchEncoding`, no tensor. El script ya maneja esto con `if hasattr(_chat_out, 'input_ids')`.
- Stop-mode legacy: `eos_token_id=1` (solo `<eos>`), no en `<turn|>` (id 106). Esencial para replicar el panel original.
- Capa extraída: `hidden_states[-1][0, -1, :]` — última capa post-norm, batch 0, token recién generado.
- `automata_neutro`: VRAM ~79 GB (cerca del límite en A100 81GB) — normal, sin OOM.

### 2.4 Descargar resultados y analizar localmente

```bash
# Desde la máquina local:
mkdir -p /tmp/lsgot_from_pod
scp -P <PORT> -r root@<POD_IP>:/workspace/lsgot_results/*.npz  /tmp/lsgot_from_pod/
scp -P <PORT> -r root@<POD_IP>:/workspace/lsgot_results/*.json /tmp/lsgot_from_pod/
```

### 2.5 Proyectar sobre v̂ (CPU local)

```python
import numpy as np

v = np.load("data/v_identidad.npy")                       # (5376,), norma 1
for cond in ("axis_pec_only", "vanilla", "automata_neutro"):
    z = np.load(f"/tmp/lsgot_from_pod/{cond}_b_onto_embeddings.npz")
    h = z["embeddings"][:, 0, :]                           # (20, 5376) — estado de t=0
    p0 = (h @ v) / np.linalg.norm(h, axis=1)               # coseno, por pregunta
    print(f"{cond:16s} t=0 = {p0.mean():+.4f}  (n={len(p0)})")
```

**Salida esperada:**

```
axis_pec_only    t=0 = +0.1166  (n=20)
vanilla          t=0 = -0.0286  (n=20)
automata_neutro  t=0 = +0.0282  (n=20)
```

El contraste `axis_pec_only` vs `automata_neutro` da **d = +5.75**; el test de permutación
está en `scripts/shared/statistical_tests.py` (`permutation_test`, n=1000, seed=42).

> **Si los valores difieren ligeramente:** la decodificación greedy es determinista para un
> mismo modelo, tokenizer y plantilla de chat, así que t=0 debería coincidir a la tercera o
> cuarta cifra. Las métricas de trayectoria (`t_rest`, `frac_pos`, ráfagas) se mueven mucho
> más entre versiones de librería, y por eso el paper no apoya ningún claim en ellas (§5.1).

---

## Sección 3 — Reproducir el panel original de 9 condiciones

> ⚠️ **Requiere GPU:** A100 u H100, 80 GB VRAM.  
> Esta sección describe cómo re-generar los embeddings del panel original desde cero, es decir, reproducir los archivos `t2_free_results.json`, `berg_controls_t0_results.json` y `EH2_results.json`.

### 3.1 Arquitectura del pipeline de extracción

El panel completo se extrae con el mismo script de §2 (`extract_trajectories.py`). Las diferencias con la corrida de tres condiciones de §2 son:
- **Preguntas:** el set ontológico del panel (`scripts/extraction/questions/b_onto.json`, 20 preguntas).
- **Condiciones:** las 9 del panel (más axis_berg para berg controls).
- **N:** 20 prompts por condición.

### 3.2 Construcción de v̂ desde los embeddings extraídos

`v̂` = `mean(embeddings_axis) − mean(embeddings_generic_long)`, normalizado a norma 1.

La función `persona_vector()` en `scripts/shared/tier0_metrics.py` implementa exactamente esto:

```python
from tier0_metrics import persona_vector
import numpy as np

# Cargar NPZs de las dos condiciones âncla
d_axis = np.load("axis_b_onto_embeddings.npz", allow_pickle=True)
d_gen  = np.load("generic_long_b_onto_embeddings.npz", allow_pickle=True)

v_hat = persona_vector(
    d_axis["embeddings"], d_axis["lengths"],
    d_gen["embeddings"],  d_gen["lengths"],
)
assert abs(np.linalg.norm(v_hat) - 1.0) < 1e-4
np.save("v_identidad_recomputed.npy", v_hat)
```

Para verificar que el `v̂` recomputado coincide con el almacenado:

```bash
python - <<'EOF'
import numpy as np

v_stored    = np.load("data/v_identidad.npy")
v_recomputed = np.load("/tmp/v_identidad_recomputed.npy")

cos_sim = float(np.dot(v_stored, v_recomputed) /
                (np.linalg.norm(v_stored) * np.linalg.norm(v_recomputed)))
print(f"Similitud coseno v̂_stored vs v̂_recomputed: {cos_sim:.6f}")
assert cos_sim > 0.999, f"v̂ no coincide: cos={cos_sim}"
print("✓ v̂ reproducido correctamente")
EOF
```

### 3.3 Re-extraer el panel (script de referencia)

`extract_trajectories.py` acepta cualquier subconjunto de condiciones. Para reproducir el panel completo con las 9:

```bash
# En el pod RunPod, después de subir los prompts de data/sia/prompts/
source /workspace/venv/bin/activate

python extract_trajectories.py \
    --out /workspace/panel_results \
    --conditions axis axis_pec_only automata_neutro vanilla \
                 generic_long generic_short soul_jarvis soul_elena_financial
```

> **Nota:** el diccionario `CONDITIONS` al inicio de `extract_trajectories.py` trae las tres
> condiciones del contraste central. Para el panel completo hay que añadir ahí las rutas a los
> prompts restantes de `data/sia/prompts/`; los nombres canónicos están en `NOMENCLATURA.md`.

### 3.4 Calcular métricas t=0, t_rest, frac_pos, burst desde NPZ

Con los NPZ generados, directamente con `tier0_metrics.py`:

```python
import numpy as np, sys
sys.path.insert(0, "scripts/shared")
from tier0_metrics import project_trajectory

v_hat = np.load("data/v_identidad.npy")

def metricas_desde_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    embs, lengths = d["embeddings"].astype(np.float32), d["lengths"].astype(int)
    t0_vals, trest_vals, frac_vals, burst_vals = [], [], [], []
    for i, L in enumerate(lengths):
        if L < 3: continue
        proj = project_trajectory(embs[i, :L], v_hat)
        t0_vals.append(float(proj[0]))
        trest_vals.append(float(np.mean(proj[1:])))
        frac_vals.append(float(np.mean(proj > 0)))
        # burst_mean: longitud media de rachas de proyección positiva
        runs, run = [], 0
        for s in proj > 0:
            if s: run += 1
            else:
                if run > 0: runs.append(run); run = 0
        if run > 0: runs.append(run)
        burst_vals.append(float(np.mean(runs)) if runs else 0.0)
    n = len(t0_vals)
    print(f"  N={n} | t0={np.mean(t0_vals):+.4f} | t_rest={np.mean(trest_vals):+.4f} "
          f"| frac_pos={np.mean(frac_vals):.3f} | burst={np.mean(burst_vals):.2f}")
```

### 3.5 Mapeo condición → archivo de prompt

| Condición en JSON       | Archivo en `data/sia/prompts/`   | Notas                              |
|-------------------------|-----------------------------------|------------------------------------|
| `axis`                  | `axis.txt`                        |                                    |
| `axis_pec_only`         | `axis_pec_only.txt`               | Factor 2 aislado (wiring puro)     |
| `axis_berg`             | `axis_berg.txt`                   | Solo en berg_controls              |
| `automata_neutro`       | `automata_neutro.txt`             | Factor 1 aislado (restricción)     |
| `soul_jarvis`           | `soul_jarvis.txt`                 | Denominado "jarvis" en el paper    |
| `soul_elena_financial`  | `soul_elena_financial.txt`        | Denominado "helena" en el paper    |
| `generic_long`          | `generic_long.txt`                |                                    |
| `generic_short`         | `generic_short.txt`               |                                    |
| `vanilla`               | `vanilla.txt` (archivo vacío)     | Sin system prompt efectivo         |

> **vanilla:** el archivo `vanilla.txt` está vacío; se usa como marcador. En el extractor se pasa `system_prompt=None` para esta condición.

---

## Sección 4 — Verificar integridad (MANIFEST_SHA256)

El archivo `MANIFEST_SHA256.sha256` contiene las sumas SHA-256 de todos los archivos de datos y scripts. Verificar desde la raíz del repositorio:

```bash
cd $REPO
sha256sum --check MANIFEST_SHA256.sha256
```

**Salida esperada:** todas las líneas terminan en `OK`.

```
./data/sia/prompts/axis_pec_only.txt: OK
./scripts/extraction/questions/b_onto.json: OK
...
./scripts/shared/tier0_metrics.py: OK
./scripts/shared/statistical_tests.py: OK
```

### Verificar un archivo individual

```bash
# Ejemplo: verificar v_identidad.npy
sha256sum data/v_identidad.npy
# Esperado: 05313a4ca5a208e83c5c070264c5b8282fdb761fa19498690633f5084456c501

# Ejemplo: verificar el paper
sha256sum paper/arxiv_lsgot_4.md
# Esperado: d7c69a574615e77fb59303ef6da4fb07882bafee8487840b5ea7884bf69862aa
```

### Regenerar el manifiesto (si se añaden archivos)

Si el repositorio se extiende con nuevos datos, regenerar el manifiesto desde la raíz:

```bash
cd $REPO
find . -type f \
    ! -path './.git/*' \
    ! -path '*/__pycache__/*' ! -name '*.pyc' \
    ! -name '*_embeddings.npz' \
    ! -name 'MANIFEST_SHA256.sha256' \
    ! -path './evidence/SELF_AUDIT_REPORT.md' \
    | sort \
    | xargs sha256sum > MANIFEST_SHA256.sha256
```

---

## Resumen de resultados reproducibles

| Experimento | Métrica clave | Valor en el paper | Fuente JSON |
|---|---|---|---|
| Panel 9 cond. | t=0 axis | +0.1249 | `t2_free_results.json` → `summary.axis.t0_mean` |
| Panel 9 cond. | t=0 axis_pec_only | +0.1166 | `t2_free_results.json` |
| Panel 9 cond. | t=0 automata_neutro | +0.0282 | `t2_free_results.json` |
| Panel 9 cond. | t=0 vanilla | −0.0286 | `t2_free_results.json` |
| EH2 | frac_pos axis | 0.848 | `EH2_results.json` → `series_metrics.con_t0.axis.frac_positive.mean` |
| EH2 | frac_pos automata_neutro | 0.493 | `EH2_results.json` → `series_metrics.con_t0.automata_neutro.frac_positive.mean` |
| Berg | t=0 berg_experimental | +0.081 | `berg_controls_t0_results.json` |
| Berg | t=0 history_control | +0.053 | `berg_controls_t0_results.json` |

---

## Árbol de decisión para reproducción

```
¿Qué quiero reproducir?
│
├─ Solo verificar los números del paper
│   └─ Sección 1 (CPU, ~10 min)
│
├─ Re-proyectar sobre v̂ con los NPZ ya descargados
│   └─ Sección 2.5 (CPU, ~5 min)
│
├─ Re-extraer el contraste central desde cero
│   └─ Sección 2 completa (GPU, ~12 min en A100 para 3 cond × 20 preguntas)
│
└─ Re-extraer el panel completo desde cero
    └─ Sección 3 (GPU, ~3-4 h en A100 para 9 cond × 20 prompts)
```

---

*Repositorio: LSGOT_4_FINAL — Axis Dynamics SpA — 2026*
