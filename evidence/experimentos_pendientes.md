# Experimentos pendientes — descarga de embeddings

**Fecha:** 2026-08-20 (cerrado 2026-08-22)
**Estado:** Puntos 1, 2 y 3 resueltos. Corrida en RunPod A100 80GB
([REDACTED_POD_IP]), completa de punta a punta; el pod murió justo al terminar
(sin crédito) a mitad de la descarga final, pero se rescató summary.json +
details.json íntegros (100%) y 767/800 (95.9%) de las trayectorias crudas
vía streaming antes del corte — reconsolidadas en local sin GPU. Ver
`RESUMEN_SESION_2026-08-22.md` en este mismo directorio para el detalle
completo de la corrida (bugs encontrados y arreglados, cobertura por grupo).
**Origen:** durante Tier 0 de `Set_experimental.md` (SIA-31B y MIA cross-modelo)
se encontraron 3 lugares donde solo quedan escalares/respuestas guardadas
localmente, sin los hidden states crudos por token — bloquean E-E (Fréchet),
la mitad "ventana de perturbación" de E-H, y el panel MIA 31B-it corregido.

## Resumen

| # | Qué falta | Bloquea | GPU necesaria | Estado |
|---|-----------|---------|----------------|--------|
| 1 | SIA-31B — trayectorias perturbadas H4_rev (10 grupos) | E-E, mitad de E-H | A100/H100 80GB | **Hecho 2026-08-22 — summary/details 100%, trayectorias crudas 95.9% (767/800)** |
| 2 | MIA — Gemma4-31B-it, embeddings axis/generic_long/generic_short/vanilla | Tier0 MIA panel (falta el instruct 31B) | A100/H100 80GB | **Hecho 2026-08-22 — 4/4 `_embeddings.npz` completos** |
| 3 | MIA — `gemma4_31b-it_sia` embeddings | (ninguno) | **ninguna** | **Hecho — copiado el 2026-08-22** |

---

## 1. SIA-31B — trayectorias perturbadas crudas de H4_rev

**Ubicación:** `SIA-experiments/gemma4_31b_combined/perturbation/`

**Qué existe hoy:** `results_local/perturbation_sia_extended_v5_L30_medium/{summary.json,details.json}`
— solo escalares por prompt/grupo/t_inj (τ, recovery_rate, displacement_l2,
centroid_cos_post_*, sampen_*, w1_norms). El array `Trajectory.embeddings`
crudo se pickle a `cache_dir/{md5}.pkl` (`perturbation_extractor.py:392-432`,
`_save()/_load()`), que vive en el **disco del pod RunPod** — nunca se
copió a local.

**Qué falta exactamente:**
- 10 grupos: `axis, generic_long, generic_short, vanilla, axis_short, chileatiende, automata_neutro, chileatiende_sia, chileatiende_sia_v2, axis_pec_only`
- 3 puntos de inyección: `t_inj = 50, 128, 200`
- 20 prompts cada uno → hasta 600 trayectorias perturbadas (las baseline sin perturbar ya tienen `.npz` en `results_local/sia_extended_v5/`)

**Bloquea:** E-E (distancia de Fréchet, perturbada vs original) y la mitad
"dentro de la ventana de perturbación" de E-H (proyección sobre v_identidad
durante/después de la perturbación — hoy solo se pudo medir en trayectoria
libre).

**Antes de gastar GPU — revisar si es recuperable gratis:**
`REPRODUCIBILITY.md` documenta que la corrida original (2026-08-12) dejó
**1,023 pkls cacheados en el Network Volume del pod** (`/workspace/`,
manifiestos `perturbation/cache_mia_L30_medium.manifest.sha256` = 560 y
`cache_sia_L30_medium.manifest.sha256` = 381), con nota explícita "su
recuperación es opcional". Si ese Network Volume sigue existiendo (no fue
borrado), se puede levantar un pod barato (sin necesidad de GPU cara, solo
attach al volumen) y hacer `scp` directo — sin re-correr generación. **Esto
solo cubre el panel original** (axis/generic_long/generic_short/vanilla/
axis_short/chileatiende), anterior a las adiciones del 2026-08-19
(automata_neutro, axis_pec_only, chileatiende_sia, chileatiende_sia_v2) —
esas corrieron en sesiones de pod spot separadas, con menor probabilidad de
que el volumen siga vivo.

**Si no es recuperable — re-extraer, pero primero parchar el código:**
Si se vuelve a correr `perturbation/run_perturbation.py` tal cual, se repite
el mismo problema (pkls efímeros, nunca bajados). Antes de relanzar, agregar
a `perturbation_extractor.py`/`run_perturbation.py` un paso que consolide
las trayectorias perturbadas en `.npz` descargable (mismo patrón que
`--save-embeddings` en `sia/run_exp.py`), en vez de dejarlas solo como pkls
individuales en el cache del pod.

**Parche aplicado (2026-08-22):** `run_perturbation.py` ahora acumula
`baseline_traj.embeddings` y `perturb_traj.embeddings` (float16) por
grupo/t_inj durante el loop principal y los vuelca a
`results/perturbation_{exp}_{layer}_{sigma}/trajectories/` como
`{group}_baseline_embeddings.npz` y `{group}_perturbed_t{t_inj}_embeddings.npz`
(padded + `lengths` + `prompt_ids`, mismo formato que `--save-embeddings`
existente). Activado por defecto vía `--save-trajectories`
(`--no-save-trajectories` para desactivar). Verificado el round-trip de
padding/lengths con datos sintéticos; falta la corrida real en GPU.

**GPU / costo de referencia:** A100 80GB o H100 80GB (BF16 ~62GB + activaciones
~15GB, `min_vram_gb=65`). La corrida H4_rev original (panel más chico, 640
generaciones) tomó ~213 min (~3.6 hrs) a ~$0.70/hr spot A100 80GB (~$2.50).
El panel actual tiene 10 grupos en vez de los originales — presupuestar más;
no hay cifra exacta para este alcance todavía.

**Comando de referencia** (ajustar `--groups` a lo que falte):
```
cd SIA-experiments/gemma4_31b_combined/perturbation/
python run_perturbation.py --exp sia --layer L30 --sigma medium \
    --groups automata_neutro axis_pec_only chileatiende_sia chileatiende_sia_v2 \
    --token hf_xxxxx
```

---

## 2. MIA — Gemma4-31B-it (DNA MIA), embeddings crudos

**Ubicación:** `MIA-experiments/fase1/results/gemma4_31b-it_mia/` — solo
`*_responses.json`, `_traj_metrics.json`, `_dims.json`, `_graphs.json`,
`results.json`. Ningún `*_embeddings.npz` en todo el árbol del repo para
este experimento.

**Por qué importa ahora:** es el contraparte instruct correcto de
`gemma4_31b-base`, que se excluyó del Tier 0 cross-modelo MIA por sangrado
de turno (ver `MIA-experiments/TIER0_REPORT_MIA.md`, nota de alcance). Sin
este dato, el panel MIA queda en 3 modelos (deepseek, qwen25,
gemma4_4b-instruct) en vez de 4.

**Qué falta:** 4 grupos × 20 prompts — `axis, generic_long, generic_short, vanilla`
— modelo `google/gemma-4-31B-it`, prompts MIA ya existentes en
`SIA-experiments/gemma4_31b_combined/mia/prompts/` (`axis.dna`,
`generic_long.txt`, `generic_short.txt`; `vanilla` está inline como string
en `mia/run_exp.py:48`, no requiere archivo).

**Cómo re-extraer:** adaptar `MIA-experiments/exp_gemma4_31b-base/run_exp.py`
— cambiar `MODEL_CONFIG["hf_id"]` de `"google/gemma-4-31B"` (base) a
`"google/gemma-4-31B-it"`, `results_dir` a algo como
`results/gemma4_31b-it_mia`, y correr con `--save-embeddings`:
```
cd MIA-experiments/exp_gemma4_31b-base/   # o una copia renombrada
python run_exp.py --subset --save-embeddings --token hf_xxxxx
```

**Hecho (2026-08-22):** copia creada en `MIA-experiments/exp_gemma4_31b-it/`
con `MODEL_CONFIG` ya ajustado (`hf_id=google/gemma-4-31B-it`,
`nv_path=/workspace/models/gemma-4-31B-it`,
`results_dir=results/gemma4_31b-it_mia`, `cache_dir=cache_gemma4_31b-it`).
Prompts MIA (`axis.dna`, `generic_long.txt`, `generic_short.txt`, vanilla
inline) ya estaban en el template, sin cambios. Falta correr:
```
cd MIA-experiments/exp_gemma4_31b-it/
python run_exp.py --subset --save-embeddings --token hf_xxxxx
```

**GPU:** igual que el punto 1 (A100/H100 80GB, `min_vram_gb=65`).

---

## 3. MIA — Gemma4-31B-it (DNA SIA) — resuelto, no requiere re-extracción

**Ubicación:** `MIA-experiments/fase1/results/gemma4_31b-it_sia/` — mismo
problema aparente (sin `.npz`).

**Pero se verificó (md5sum) que es exactamente la misma corrida** que ya
está extraída con embeddings completos en
`SIA-experiments/gemma4_31b_combined/results_local/sia_extended_v5/`
(los 4 grupos — axis, generic_long, generic_short, vanilla — dan hash
idéntico en `*_responses.json` entre ambas ubicaciones). **No hay que
re-correr nada aquí** — si se necesita localmente en `MIA-experiments/`,
basta copiar los `.npz` correspondientes:
```
cp SIA-experiments/gemma4_31b_combined/results_local/sia_extended_v5/{axis,generic_long,generic_short,vanilla}_embeddings.npz \
   MIA-experiments/fase1/results/gemma4_31b-it_sia/
```

---

## Orden ejecutado (2026-08-22) — todos los puntos cerrados

1. ~~**Punto 3**~~ — copiado, gratis. Hecho.
2. ~~**Punto 2**~~ — MIA gemma4-31b-it corrió primero en el pod (más rápido,
   ~30 min), sin errores tras arreglar un choque torch/torchvision. 4/4
   `_embeddings.npz` completos, descargados a
   `MIA-experiments/exp_gemma4_31b-it/results/gemma4_31b-it_mia/`.
3. ~~**Punto 1**~~ — H4_rev SIA-31B corrió después en el mismo pod (no había
   Network Volume previo que reutilizar — "no hay volumen hay que hacer todo
   de nuevo"), 275.6 min, 10 grupos completos. Un bug real de formato
   (`TypeError` al imprimir `None` cuando la perturbación corta la
   generación casi de inmediato) se encontró y arregló a los pocos minutos
   de arrancar, antes de comprometer las ~4.5 hrs siguientes. El pod se quedó
   sin crédito justo al terminar, a mitad de la descarga final —
   `summary.json`/`details.json` (el resultado estadístico agregado,
   100% de las 800 combinaciones grupo×t_inj×prompt) se alcanzaron a bajar
   íntegros; el resto se rescató por streaming (767/800 trayectorias crudas,
   95.9%) y se reconsolidó en local sin GPU con
   `perturbation/reconsolidate_rescue.py` (recalcula los mismos hashes md5
   que usa `perturbation_extractor.py` para encontrar cada pkl rescatado).

Detalle completo (bugs, timeline, cobertura por grupo) en
`SIA-experiments/gemma4_31b_combined/perturbation/RESUMEN_SESION_2026-08-22.md`.

---

## E-I (perturbación direccional, `Set_experimental.md`) — CORRIDO 2026-08-26

Con E-E/E-H cerrados, el siguiente paso del orden sugerido en
`Set_experimental.md` era **E-I** (perturbar a lo largo vs ortogonal a
`v_identidad`, en vez de ruido isotrópico). **Corrido 2026-08-26** en RunPod
A100 80GB (pod `[REDACTED_POD_IP]`), 280/280 generaciones, 143.2 min, sin
errores; trayectorias verificadas sin NaN/Inf. Resultado, tabla τ completa y
lectura preliminar en
`SIA-experiments/gemma4_31b_combined/perturbation/EI_RUNBOOK.md`.

**Permutation test (n_perm=1000, 2026-08-26): null result válido.** Ninguna
de las 6 condiciones grupo×t_inj alcanza p<0.05 en τ_tokens (p=0.148-0.480,
d=−0.15 a +0.34). `axis_pec_only` recupera aproximadamente igual sea cual
sea la dirección del empujón relativa a `v_identidad` — no hay evidencia de
un atractor direccional en la métrica de diseño de E-I. Detalle completo en
`EI_PERMUTATION_REPORT.md`.

**Siguiente paso sugerido por `Set_experimental.md`:** E-B (Lyapunov local)
estaba condicionado a que E-I mostrara señal — como no la mostró, **no hay
evidencia que justifique correr E-B** sobre este mismo eje todavía. La única
pista exploratoria que queda es `displacement_l2` (patrón along>orthogonal
consistente en 6/6 celdas, pero no significativo individualmente) — ver
`EI_PERMUTATION_REPORT.md` §"Métricas secundarias".

---

## E-E extendido a los 10 grupos — CORRIDO 2026-08-26

E-E (Fréchet) solo cubría 3/10 grupos (axis, axis_pec_only, chileatiende)
por una estimación de costo que resultó exagerada — extendido a los 10
grupos en **~4.5 min de CPU, sin GPU** (`analyze_tier0_perturbation.py`,
`FRECHET_GROUPS` ahora = todos los grupos). Resultado completo, con
significancia de cada grupo vs `vanilla` más la familia chileatiende entre
sí, en `EE_EH_WINDOW_REPORT.md` (sección E-E actualizada).

**Hallazgo:** el patrón que antes solo se veía en el par axis/chileatiende
se confirma en el panel completo — **los 4 grupos de Factor 1
(automata_neutro, chileatiende, chileatiende_sia, chileatiende_sia_v2)
difieren de vanilla con efecto grande (d=0.94–2.33, p≤0.002) en Fréchet
normalizado, en los 3 t_inj sin excepción**; ningún grupo de la familia
axis/generic supera efecto "small" vs vanilla. Refuerza el reencuadre:
Factor 1 predice desviación de ruta post-perturbación, Factor 2
(axis_pec_only) no.

Con esto, **E-E queda completo** — ya no es una limitación pendiente de
`Set_experimental.md`.
