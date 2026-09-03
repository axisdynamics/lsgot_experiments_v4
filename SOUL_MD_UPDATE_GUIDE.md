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
6. **Regenerar `MANIFEST_SHA256.sha256`** después de cualquier cambio
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
