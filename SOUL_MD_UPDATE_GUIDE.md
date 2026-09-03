# Guía de continuación — ronda SOUL.md externos (2026-09-02)

Para otro agente (o para retomar esto en una sesión futura). Resume qué se
hizo, dónde queda cada cosa, y qué sigue abierto — en orden de prioridad.

## Qué se hizo esta ronda

1. Tres identidades externas (`soul_jarvis`, `soul_elena_financial`,
   `soul_solidity_auditor` — ver fuentes en
   `evidence/SOUL_MD_EXTERNAL_CONTROLS_REPORT.md`) corridas por el pipeline
   T2 (`scripts/perturbation/run_perturbation_t2.py`) en un pod RunPod
   limpio (A100 80GB, sin Network Volume precacheado — el script descarga
   el modelo solo, `get_model_path()`).
2. Los 4 scripts de `scripts/fase4_t2/` extendidos con los 3 grupos nuevos
   y con comparaciones de permutación adicionales (antes solo
   `analyze_t2_free.py` y `analyze_t2_perturbation.py` guardaban JSON;
   ahora los 4 lo hacen: `t2_free_results.json`,
   `t2_perturbation_results.json`, `t2_frechet_stats_results.json`,
   `t2_eh2_results.json`).
3. Trayectorias crudas (`.npz`, 12 archivos) descargadas a
   `.../gemma4_31b_combined/perturbation/results_t2/trajectories/` — **no
   están en este repo** (`.gitignore: *.npz`), viven solo en esa ruta
   local. Ver `[[lsgot_embeddings_policy]]` en la memoria del agente si
   existe, o simplemente: no se pueden regenerar sin volver a correr el
   pod (~$2 USD, ~2h).
4. Hallazgo central: el ancla de identidad no es "identidad" en general,
   es el mecanismo de pausa+autobservación cableado ("testigo") — replica
   sin proponérselo la ablación original `axis` vs `axis_nowit` de
   `LSGOT_v2_5.md` §2.1/§5.3-5.4. Detalle completo en
   `evidence/SOUL_MD_EXTERNAL_CONTROLS_REPORT.md`.
5. Página ejecutiva actualizada (Artifact, fuera de este repo):
   [`Señales de Identidad`](https://claude.ai/code/artifact/a8970a35-7d1c-491a-9040-4fa84f84fe41)
   — 4 secciones nuevas: la relectura del testigo, resiliencia vs testigo,
   ráfagas vs testigo, por qué `axis` supera a `soul_md_corto`. Si se
   actualiza `paper/lsgot_4.md` con esta relectura, actualizar también el
   artifact para que no queden desincronizados (o viceversa).

## Cómo reproducir esta ronda exacta

```bash
# 1) Prompts ya están en data/sia/prompts/soul_{jarvis,elena_financial,solidity_auditor}.txt
# 2) Pod limpio: A100/H100 80GB, min_vram_gb=65, container disk >=120GB,
#    token HF con acceso aceptado a google/gemma-4-31B-it
scp scripts/perturbation/run_perturbation_t2.py scripts/perturbation/perturbation_extractor.py \
    root@<pod>:/workspace/scripts/perturbation/
scp data/prompts.json root@<pod>:/workspace/sia_data/   # 100 preguntas ontológicas — no vive en este repo, ver FUENTES.md
scp data/sia/prompts/*.txt data/sia/prompts/soul_md_corto.md root@<pod>:/workspace/sia_data/prompts/

ssh root@<pod> "cd /workspace/scripts/perturbation && \
  pip uninstall -y torchvision torchaudio -q && \
  python3 run_perturbation_t2.py --groups soul_jarvis soul_elena_financial soul_solidity_auditor --token hf_xxxxx"
# ~15-20min descarga modelo + ~1.5-2h GPU (240 generaciones: 3 grupos × 20 prompts × 4 condiciones)

# 3) Bajar ANTES de soltar el pod:
scp -r root@<pod>:/workspace/scripts/perturbation/results_t2/trajectories/ \
    /home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/perturbation/results_t2/trajectories/

# 4) Análisis local (CPU, numpy/scipy, sin GPU):
cd scripts/fase4_t2/
python3 analyze_t2_free.py
python3 analyze_t2_perturbation.py
python3 analyze_t2_frechet_stats.py
python3 analyze_t2_eh2.py
```

Nota sobre el entorno del pod: si `pip install -r requirements.txt` sube
`torch` a una versión más nueva de la que trae `torchvision`/`torchaudio`
preinstalados, `transformers` rompe al cargar `AutoConfig` (`torchvision::nms
does not exist`) porque importa `torchvision` para utilidades de imagen
incluso en modelos de solo texto. Fix aplicado esta ronda: desinstalar
`torchvision`/`torchaudio` — no se usan en este pipeline.

## Qué sigue abierto, en prioridad

1. **Verificar la sospecha léxica de `soul_jarvis`** (§1.1 del reporte).
   El pipeline actual (`perturbation_extractor.py`) solo guarda
   embeddings, no texto decodificado — para confirmar o descartar que el
   determinismo RQA de `soul_jarvis` (0.420, ≈`automata_neutro`) es un
   artefacto de repetición literal de sus "signature words" ("Very good,
   Sir.") hay que decodificar respuestas reales. Dos caminos: (a) agregar
   guardado de texto a `perturbation_extractor.py` y volver a correr solo
   `soul_jarvis` (barato, ~15-20 min de GPU para 1 grupo), o (b) si el
   cache de tokens sigue en el pod/Network Volume, decodificar offline sin
   GPU. Sin esto, el hallazgo de §1.1 queda como sospecha, no conclusión.
2. **Predicción 2(ii) sigue sin testearse**: ¿declarar identidad sin
   invocar el testigo produce el mismo ancla en t=0? Ninguna de las 12
   condiciones corridas hasta ahora aísla esa celda — requeriría un prompt
   diseñado a propósito (identidad + valores declarados, cero mención de
   pausa/silencio/autobservación, en el mismo idioma/longitud que `axis`
   para controlar esos confounds).
3. **Dosis-respuesta del testigo, con un control diseñado**: una v3 de
   `soul_md_corto` con el mismo mecanismo invocado varias veces en bloques
   distintos (en vez de una sola vez) probaría directamente si la brecha
   `axis`/`axis_pec_only` vs `soul_md_corto` (v̂ media d=−1.87, p<0.0001)
   es de redundancia estructural o de alguna otra diferencia entre los dos
   prompts.
4. **`paper/lsgot_4.md` ya incorpora esta relectura** (§3.9 nuevo, ajustes
   en §5 y §7.2 — sesión 2026-09-02, después de la ronda inicial de este
   documento). No se tocó el Abstract ni la definición de Factor I en
   §2.2 — el hallazgo se lee como una confirmación/afinamiento de esa
   definición (que ya exige auto-chequeo *cableado*, no solo declarado),
   no como un cambio de terminología. Si en el futuro se decide sí
   renombrar Factor I en el resto del paper, hacerlo con una pasada
   completa (Abstract, §1.4 hipótesis, §2.2, §4, §6) para no dejar
   secciones con vocabulario inconsistente.
5. **Commit sin push**: resuelto — los 3 commits de la ronda quedaron
   empujados a `origin/main` (`github.com/axisdynamics/lsgot_experiments_v4`).
6. **Hipótesis workspace (2026-09-02, después de los ítems anteriores)**:
   investigación en `evidence/WORKSPACE_HYPOTHESIS_REPORT.md` — lens de
   capa final validado 20/20 (estados post-final-norm, `W_U @ h` =
   primer token real), perfil por capas del ancla consistente con las
   costuras de `axisdynamics/workspace-8b` (pico en L35 = costura de
   salida, dip en L30 = centro de banda, onset temprano = modo secundario
   bimodal). **Pendiente**: re-extracción GPU limpia (7 condiciones × 20
   prompts, residuales crudos por capa + Jacobianas para J-lens) para el
   test de verbalizabilidad — los estados `ef2_L5_L55` están en una base
   no identificable y no sirven para el lens. Predicciones registradas en
   el reporte §2.2.
7. **Regenerar `MANIFEST_SHA256.sha256`** después de cualquier cambio
   futuro a `data/`, `evidence/`, `paper/`, `scripts/`, `README.md`:
   ```bash
   cd LSGOT_v4 && git add -A && \
   git ls-files | grep -v '^MANIFEST_SHA256.sha256$' | sort | xargs sha256sum > MANIFEST_SHA256.sha256
   git add MANIFEST_SHA256.sha256
   ```
   (el manifest anterior a esta ronda estaba desactualizado — no cubría
   `scripts/fase4_t2/` ni `scripts/perturbation/run_perturbation_t2.py` ni
   `data/sia/prompts/soul_md_corto.md`; quedó regenerado completo en esta
   ronda cubriendo los 113+ archivos trackeados por git.)

## 8. Próxima sesión (pendiente inmediato): re-extracción GPU para el J-lens

Contexto: `evidence/WORKSPACE_HYPOTHESIS_REPORT.md` — el lens de capa
final está validado (20/20), el perfil por capas calza con las costuras
de `workspace-8b`, pero la verbalizabilidad por capa quedó pendiente:
los estados `ef2_L5_L55` están en una base no identificable, y el lens
ingenuo no basta en la banda media (se necesita la Jacobiana).

**Protocolo del pod (limpio, A100/H100 80GB, disk ≥120GB):**
1. Extracción de t=0 por capa, 7 condiciones limpias (axis, axis_short,
   axis_pec_only, generic_long, generic_short, vanilla, automata_neutro),
   20 prompts prioritarios. Un solo forward por prompt con
   `output_hidden_states=True` sobre el contexto completo (NO generar
   tokens — solo el estado t=0), guardando `hidden_states[l]` para las
   60 capas + el estado post-final-norm del layer -1 (el que este
   pipeline ya valida con `W_U @ h` = primer token real 20/20).
   ~140 forwards → ~15 min de GPU.
2. Jacobianas J_ℓ = ∂(capa final)/∂(capa ℓ) para el J-lens (torch
   autograd, backward de la capa final a cada capa ℓ, promedio sobre las
   20 posiciones... sobre t=0 solo hace falta 1 posición por prompt;
   promediar 20 prompts). ~30 min extra.
3. Bajar TODO a local ANTES de soltar el pod (política de embeddings —
   ver memoria del agente): un npz por condición con estados[60 capas]
   + un npz/pkl de Jacobianas.
4. Local (CPU): lens por capa = `W_U @ final_norm(h_ℓ)` (lens ingenuo) y
   `W_U @ J_ℓ @ final_norm(h_ℓ)` (J-lens); match-rate vs primer token
   real por capa; scan de tokens de identidad/auto-chequeo en top-10.
   W_U y normas ya descargables con `scripts/fase4_t2/fetch_wu_partial.py`
   (2.8GB, sin GPU).
5. Chequear las predicciones registradas (reporte §2.2): readout verbal
   del auto-chequeo en L13-L33, "snap" al primer token real en L33-L35,
   convergencia al contenido común en L30, separación máxima entre
   condiciones en L35.

## 9. Scripts listos (2026-09-03) — cómo correr la re-extracción hoy

Los dos scripts del protocolo §8 ya existen en `scripts/fase4_t2/`:

- **`extract_layers_jlens.py`** (pod, GPU): estados t=0 por capa (7
  condiciones × 20 prompts, hidden_states 60 capas + post-final-norm) +
  Jacobianas J̄_ℓ + readouts JVP exactos por muestra. Implementa el
  pre-registro A1 §4.1 (VJPs chunked, J̄ promediado, readout
  `W_U @ J̄_ℓ @ h_ℓ`). Verifica en caliente: base (W_U@y = logits, 20/20),
  identidad del primal del grafo reducido por (prompt, capa) — aborta si
  no cuadra — y J_59 analítico en `--sanity`.
- **`analyze_j_lens.py`** (local, CPU, numpy): lens ingenuo vs J-lens
  pooled vs JVP exacto por capa; match@1/@10 vs primer token REAL
  (responses.json); scan de tokens de auto-chequeo/identidad en top-10;
  separación JS entre condiciones por capa; chequeo de las 3 predicciones
  §2.2 con criterios fijados en su docstring ANTES de ver datos.

Decisiones de diseño (registradas antes de correr) — ACTUALIZADO 2026-09-03
tras la validación en el pod (el plan original de J̄ no sobrevivió la realidad
de transformers 5.16.1 + torch 2.14 + contextos de 4K tokens):

1. **J-lens = JVP exacto por muestra** (`torch.func.jvp`, un dual forward por
   (prompt, capa) sobre el grafo reducido) — es la definición del J-lens de
   Anthropic (la Jacobiana de la propia muestra aplicada a su estado) y quedó
   como instrumento primario. Top-50 + logprob del primer token guardados.
2. **J̄ (Jacobianas promediadas) quedó OPCIONAL (`--jbar`)**: dos muros medidos
   en el pod lo hacen inviable hoy con contextos de 4K tokens (el pre-registro
   A1 asumía corpus de 256 tokens):
   - jacrev (backward) materializa cotangentes (kv≈4K, D, C) = 33GB a C=256 → OOM;
   - el jvp full-stack retiene ~0.5GB de duales por capa en el nivel functorch
     (L30 = +15GB → OOM a C=8);
   - ~17ms de overhead functorch por llamada (aun reutilizando la función):
     5.6M llamadas = 26h;
   - materializar el tangente del jvp en GPU (asignación/clone) retiene 3.9GB
     por chunk en torch 2.14 (bug del nivel functorch) — la vía CPU
     (`.cpu().numpy()`) es plana (verificado).
   Si el contraste instrumento-promediado vuelve a necesitarse, recalcular J̄
   con contextos de 256 tokens (el régimen del pre-registro) o en un pod con
   más VRAM.
3. **Grafo reducido de 1 posición** (validado): hs[-1] YA es post-final-norm en
   5.16.1 (la lm_head lo consume directo); el h_59 crudo se captura con hook en
   la entrada de lm.norm. Capas llamadas como módulos reales con
   per_layer_input=None, position_embeddings por tipo de capa (rotary 256/512),
   máscaras por tipo vía transformers.masking_utils, shared_kv_states={}
   (ninguna capa comparte K/V en este checkpoint). Cache 5.x (DynamicLayer,
   attr .keys/.values; crop(-1) quita del final; los updates reemplazan, no
   mutan) — estado past por slicing directo de los slots (las capas deslizantes
   guardan solo el pasado, W−1), reset desde un master antes de cada forward.
4. `allow_bf16_reduced_precision_reduction=False` (sin esto las GEMMs (1,1,D)
   toman kernels con acumulación distinta — diferencia escalar en q_proj,
   verificado). El primal del grafo reducido reproduce el forward original
   salvo ruido ULP (0.0 en L59, 8e-2 en L58, ~1.4 en L0 a T≈4K) — los ops de
   redondeo bf16 son identidad en el backward, así que los READOUTS JVP no
   heredan ese ruido del forward. Verificaciones: base (W_U@y == argmax logits,
   20/20 — no-fatal: near-ties, 1/140 divergió), J_59 analítico (err rel
   2.6e-3), primal por (prompt, capa) grabado en meta.json. La máscara
   deslizante del builder sale con una columna de más cuando T < ventana
   (T=966 → mask 967) — se recorta a min(T, sliding_window).
5. Las predicciones P1-P3 se chequean con criterios pre-especificados en el
   docstring de analyze_j_lens.py sobre el instrumento JVP (top-10/top-50) +
   la separación JS del lens ingenuo como corroborativo.

### Comandos del pod (limpio, A100/H100 80GB, disk ≥120GB, min_vram_gb=65)

```bash
# local → pod (prompts.json vive en
# ~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/data/)
scp scripts/fase4_t2/extract_layers_jlens.py root@<pod>:/workspace/scripts/fase4_t2/
scp data/sia/prompts/axis.dna data/sia/prompts/axis_short.txt \
    data/sia/prompts/axis_pec_only.txt data/sia/prompts/generic_long.txt \
    data/sia/prompts/generic_short.txt data/sia/prompts/automata_neutro.txt \
    root@<pod>:/workspace/sia_data/prompts/
scp ~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/data/prompts.json \
    root@<pod>:/workspace/sia_data/

# pod: entorno + SANITY. ATENCIÓN (validado 2026-09-03): gemma4 NO existe en
# transformers <5.x — el pin viejo (4.49-4.53) NO sirve. El entorno validado es
# transformers 5.16.1 + torch 2.14.0+cu130 (pip install -U torch; 5.16.1 exige
# torch >=2.5). Desinstalar torchvision/torchaudio como siempre.
ssh root@<pod> "cd /workspace/scripts/fase4_t2 && \
  pip uninstall -y torchvision torchaudio -q && \
  pip install -q -U torch 'transformers==5.16.1' && \
  python3 extract_layers_jlens.py --sanity --token hf_xxxxx"
# esperar SANITY_OK + ETA (valida: base check, primal L0/L30/L58/L59, J_59
# analítico, JVP L59; ~5 min con el modelo ya descargado)

# pod: corrida completa (~1-2h de VJP según ETA; tee para el log)
ssh root@<pod> "cd /workspace/scripts/fase4_t2 && \
  python3 extract_layers_jlens.py --token hf_xxxxx 2>&1 | tee run_jlens_2026-09-03.log"

# bajar TODO ANTES de soltar el pod (política de embeddings)
scp -r root@<pod>:/workspace/scripts/fase4_t2/results_jlens/ \
  ~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_jlens/
# ~0.5GB (states/ + jvp/ + meta.json — SIN jacobians: --jbar quedó OFF por los
# muros medidos; ver decisiones de diseño)

# Los scripts de debug/validación del pod y sus logs quedaron guardados en
# results_jlens/pod_debug_2026-09-03/ (con README) por si hay que repetir la
# validación en un entorno nuevo.

# local (CPU): W_U y normas (2.8GB, sin GPU) + análisis
cd LSGOT_v4/scripts/fase4_t2
python3 fetch_wu_partial.py --token hf_xxxxx --out-dir \
  ~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_jlens/wu_norm
python3 analyze_j_lens.py   # → j_lens_results.json + chequeo de predicciones §2.2
```

El análisis local además valida la cadena completa: `W_U@y` vs primer token
guardado por el pod (20/20) y primer token guardado vs primer token REAL
generado por sia_extended_v5 (20/20) — antes de creer ninguna curva.

## 10. Pendiente 2026-09-03 (noche): regenerar los readouts JVP

La corrida de hoy rescató 6/7 condiciones (falta automata_neutro) pero los
readouts JVP quedaron BASURA por el bug del merge del topk chunked
(`cat([vals,bestv])[:2k]` cortaba el best — el top-50 era el primer bloque
del último chunk). Corregido (commit e27dcd9). Protocolo del pod nuevo:

```bash
# local → pod (mismos archivos que §9, + el script corregido)
scp scripts/fase4_t2/extract_layers_jlens.py root@<pod>:/workspace/scripts/fase4_t2/
scp data/sia/prompts/*.dna data/sia/prompts/*.txt root@<pod>:/workspace/sia_data/prompts/
scp ~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/data/prompts.json     root@<pod>:/workspace/sia_data/

ssh root@<pod> "cd /workspace/scripts/fase4_t2 &&   pip uninstall -y torchvision torchaudio -q &&   pip install -q -U torch 'transformers==5.16.1' &&   python3 extract_layers_jlens.py --skip-primal --token hf_xxxxx 2>&1 | tee run_jlens_readouts.log"
# ~20 min descarga modelo + ~40 min corrida (solo-readouts) → results_jlens/
scp -r root@<pod>:/workspace/scripts/fase4_t2/results_jlens/   ~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_jlens/
# local: python3 analyze_j_lens.py  → veredicto P1-P3 con el instrumento corregido
```
