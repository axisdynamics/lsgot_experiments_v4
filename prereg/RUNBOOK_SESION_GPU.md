# Runbook — la sesión de GPU, de punta a punta

**Una hora, dos modelos, tres cosas.** Copiar y pegar.

| | Qué | Modelo | Tiempo |
|---|---|---|---|
| **A** | `axis_pec_task` · el control que define el claim central | Gemma-4-31B-it | carga + ~4 min |
| **B** | `axis_pec_task` en Qwen3 · el desempate sobre el eje de Lu | Qwen3-32B | carga + ~2 min |
| **C** | el panel de Qwen3 en la **capa 32 exacta** | Qwen3-32B, misma carga que B | ~2 min |

Lo que tarda son las dos cargas de modelo. B y C van en la misma sesión.

**Los pre-registros tienen que estar committeados antes.** Si no lo están, pará:
`PREREGISTRO_axis_pec_task.md` para A, `PREREGISTRO_desempate_qwen3.md` para B y C.

---

## 0 · Compuerta de longitud · si no calza, no se corre

El pre-registro de A fija ±3% en tokens de Gemma contra `axis_pec_only`. En bytes calza con
+0.80%, pero eso es proxy y la compuerta es en tokens.

```bash
python - <<'EOF'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
n = {k: len(tok(open(f"prompts/{k}.txt").read(), add_special_tokens=False)["input_ids"])
     for k in ("axis_pec_only", "axis_pec_task", "axis_pec_none")}
print(n)
for k in ("axis_pec_task", "axis_pec_none"):
    d = (n[k] - n["axis_pec_only"]) / n["axis_pec_only"]
    print(f"{k}: delta {d:+.2%}  ventana ±3%")
    assert abs(d) <= 0.03, f"{k} FUERA DE VENTANA — no correr; ajustar y volver a firmar"
print("OK")
EOF
```

## 1 · Entorno y archivos

```bash
pip install transformers accelerate tokenizers numpy torch
export HF_TOKEN=<token con acceso a google/gemma-4-31B-it>

mkdir -p prompts
cp <repo>/data/sia/prompts/*.txt prompts/          # los 9 del paquete
cp <repo>/tier1_axis_pec_task/axis_pec_task.txt prompts/
cp <repo>/data/egrav/batteries/b_onto.json .
cp <repo>/data/v_identidad.npy .
cp <repo>/tier1_axis_pec_task/e27_prefill_t0.py .
```

El eje de Lu, 657 kB, dataset público MIT:

```bash
curl -sSL -o assistant_axis_qwen3.pt \
  https://huggingface.co/datasets/lu-christina/assistant-axis-vectors/resolve/main/qwen-3-32b/assistant_axis.pt
```

Es un pickle de torch: se carga con `weights_only=True` y se trata como dato no confiable.

---

## A · los tres brazos en Gemma

Dos líneas en el dict `CONDITIONS` de `extract_egrav.py` (cerca de la 41), **en el pod, no en el
repo**:

```python
    "axis_pec_task":   HERE / "prompts" / "axis_pec_task.txt",
    "axis_pec_none":   HERE / "prompts" / "axis_pec_none.txt",
```

```bash
cd scripts/egrav/
python extract_egrav.py --conditions axis_pec_task axis_pec_none axis_pec_only automata_neutro \
  --batteries b_onto --smoke --out /workspace/out_gemma
python extract_egrav.py --conditions axis_pec_task axis_pec_none axis_pec_only automata_neutro \
  --batteries b_onto --out /workspace/out_gemma          # 80 generaciones, ~12 min
```

Cuatro condiciones: los tres brazos del estadístico `r` más `automata_neutro` como control.
`axis_pec_only` y `automata_neutro` se re-extraen acá por la Enmienda 1, y además sirven de
control de pipeline — tienen que caer cerca de **+0.117** y **+0.028**.

Verificar antes de seguir: 80 respuestas, ninguna vacía.

## B y C · Qwen3, una sola carga

Sin generación: t=0 es la última posición de prefill.

```bash
python e27_prefill_t0.py \
  --model Qwen/Qwen3-32B --attn sdpa \
  --prompts prompts --battery b_onto.json \
  --conditions axis axis_pec_only automata_neutro generic_long generic_short vanilla axis_pec_task \
  --layers 31 32 63 64 \
  --out /workspace/qwen3_t0
```

`axis_short` queda fuera a propósito: el paper lo cita tres veces y **su prompt no está en el
paquete**. Anotado como pendiente aparte.

### Control obligatorio, antes de mirar cualquier número nuevo

La re-extracción tiene que reproducir el t=0 publicado de Qwen3 en la capa final. Esto además
**resuelve la ambigüedad de índice**: vale el `hs_<i>` que reproduzca los valores, y el otro se
descarta.

```bash
python - <<'EOF'
import numpy as np
L = lambda c, i: np.load(f"/workspace/qwen3_t0/{c}_t0.npz")[f"hs_{i}"].astype(np.float64)
tr = lambda c: np.load(f"/workspace/qwen3_t0/{c}_t0.npz")   # (20, D)
unit = lambda v: v / np.linalg.norm(v)
def proj(H, v):
    n = np.linalg.norm(H, axis=1); n[n == 0] = 1
    return (H @ v) / n
for i in (63, 64):
    v = unit(L("axis", i).mean(0) - L("generic_long", i).mean(0))
    a, b = proj(L("axis_pec_only", i), v), proj(L("automata_neutro", i), v)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    print(f"hs_{i}: axis_pec_only {a.mean():+.4f} (pub +0.185) · "
          f"automata {b.mean():+.4f} (pub +0.008) · d {(a.mean()-b.mean())/sp:+.2f} (pub +12.97)")
EOF
```

**El índice que reproduzca +0.185 / +0.008 / +12.97 es el bueno.** Si ninguno lo hace, parar: el
problema es el template o la atención, no los datos, y los números nuevos no valen.

---

## 2 · Bajar y analizar, en CPU

```bash
scp -r pod:/workspace/axis_pec_task ./
scp -r pod:/workspace/qwen3_t0 ./
```

Después, en orden, y **leyendo la regla de decisión antes de redactar nada**:

1. **Punto 3** — `axis_pec_task` vs `axis_pec_only` en Gemma sobre v̂, permutación a dos colas
   n=10.000. La tabla de tres renglones de `PREREGISTRO_axis_pec_task.md`.
2. **Desempate** — `axis_pec_task` sobre el eje de Lu en la capa 32, contra la ventana
   [−0.145, −0.130] de `PREREGISTRO_desempate_qwen3.md`, y sobre la v̂ de Qwen3.
3. **E20 en capa exacta** — rehacer cos y ρ con el script de E20, usando `hs_32` en vez de L31, y ver si el veredicto PARCIAL se mantiene.

## 3 · Qué entra al paquete si el punto 3 sale

el prompt, que pasa de `prereg/` al directorio de prompts del paquete; la condición en `NOMENCLATURA.md`, las 20 respuestas, los
agregados, el renglón de `CONDITIONS`, y los dos pre-registros con sus firmas y **sin editar las
reglas de decisión**. Después: regenerar `MANIFEST_SHA256.sha256`, correr
`scripts/shared/selfaudit.py` (13/13 exit 0) y re-embeber la salida en
`evidence/SELF_AUDIT_REPORT.md`.

---

*Escrito por Claude (Opus 5) el 2026-10-29. Nada de esto se midió todavía.*
