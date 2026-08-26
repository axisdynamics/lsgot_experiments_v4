# FUENTES — procedencia de cada archivo en LSGOT_v4

**Fecha de empaquetado:** 2026-08-26
**Firma:** `MANIFEST_SHA256.sha256` (checksums SHA-256 de todos los archivos de este directorio, generados post-sanitización) — mismo mecanismo de "firma" ya usado en este proyecto (`MANIFEST_homologacion.sha256`, ver `evidence/AXIS_PEC_ONLY_REPORT.md`/`CHILEATIENDE_CONTROL_REPORT.md`). No es una firma criptográfica GPG — no había clave configurada en esta máquina; si se necesita firma GPG real, hay que generarla explícitamente aparte.
**Sanitización aplicada:** se redactaron 2 IPs de pods RunPod (efímeras, sin valor de reproducibilidad) en `evidence/EI_RUNBOOK.md`, `evidence/experimentos_pendientes.md` y `evidence/RESUMEN_SESION_2026-08-22.md`, reemplazadas por `[REDACTED_POD_IP]`. Se barrió todo el árbol buscando tokens HF (`hf_...`), claves privadas y API keys — ninguno encontrado en las fuentes originales (los scripts nunca los tuvieron hardcodeados; el token usado en esta sesión se pasó solo por línea de comandos SSH, nunca se escribió a disco). Los correos que aparecen en `paper/lsgot_4.md`/`paper/lsgot_3.md` son la autoría declarada del paper, no credenciales.
**Excluido deliberadamente:** todos los `*.npz` de embeddings/trayectorias crudas (decenas de MB cada uno, ~600MB+ en total entre H4_rev y E-I) — no se copian por tamaño, según instrucción. Su ubicación original queda documentada aquí para quien necesite reproducir contra ellos directamente.

---

## 1. `paper/`

| Archivo | Origen |
|---|---|
| `lsgot_4.md` | `/home/plaxius/Escritorio/Buscando_la_geometría/lsgot_4.md` — escrito en esta sesión |
| `lsgot_3.md` | `/home/plaxius/Escritorio/Buscando_la_geometría/lsgot_3.md` — paper anterior, referenciado por lsgot_4 como trabajo previo a reformular |

## 2. `evidence/`

| Archivo | Origen | Notas |
|---|---|---|
| `TIER0_REPORT.md` | `~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local/TIER0_REPORT.md` | E-A/E-C/E-D/E-H, 9 grupos + comparaciones chileatiende agregadas esta sesión |
| `EE_EH_WINDOW_REPORT.md` | `/home/plaxius/Escritorio/Buscando_la_geometría/EE_EH_WINDOW_REPORT.md` | copia sincronizada del mismo archivo en `~/Documentos/Proyectos/Geometría_LSGOT/`; extendido a 10 grupos esta sesión |
| `EI_PERMUTATION_REPORT.md` | `~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/perturbation/EI_PERMUTATION_REPORT.md` | escrito esta sesión |
| `EI_RUNBOOK.md` | `~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/perturbation/EI_RUNBOOK.md` | actualizado esta sesión (era "diseñado, no corrido" → ahora con resultados); **1 IP de pod redactada** |
| `RESUMEN_SESION_2026-08-22.md` | `~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/perturbation/RESUMEN_SESION_2026-08-22.md` | sesión de rescate de trayectorias H4_rev, previa a esta; **1 IP:puerto de pod redactada** |
| `AXIS_PEC_ONLY_REPORT.md` | `/home/plaxius/Escritorio/Buscando_la_geometría/AXIS_PEC_ONLY_REPORT.md` | sin copia en `Geometría_LSGOT/` — solo existe en este directorio |
| `AUTOMATA_NEUTRO_REPORT.md` | `/home/plaxius/Escritorio/Buscando_la_geometría/AUTOMATA_NEUTRO_REPORT.md` | ídem |
| `CHILEATIENDE_CONTROL_REPORT.md` | `/home/plaxius/Escritorio/Buscando_la_geometría/CHILEATIENDE_CONTROL_REPORT.md` | ídem |
| `CHILEATIENDE_SIA_REPORT.md` | `/home/plaxius/Escritorio/Buscando_la_geometría/CHILEATIENDE_SIA_REPORT.md` | ídem |
| `ROADMAP_REENCUADRE_DENSIDAD_RESTRICCION.md` | `/home/plaxius/Escritorio/Buscando_la_geometría/ROADMAP_REENCUADRE_DENSIDAD_RESTRICCION.md` | ídem |
| `Teoria_subconjunto_acotado.md` | `/home/plaxius/Escritorio/Buscando_la_geometría/Teoria_subconjunto_acotado.md` | ídem — origen teórico del modelo de 2 factores |
| `Set_experimental.md` | `/home/plaxius/Escritorio/Buscando_la_geometría/Set_experimental.md` | copia idéntica a `~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/Set_experimental.md` |
| `REPORTE_FASE0.md` | `/home/plaxius/Escritorio/Buscando_la_geometría/REPORTE_FASE0.md` | idéntico byte-a-byte (verificado con `diff`) a `~/Documentos/Proyectos/Geometría_LSGOT/MIA-experiments/fase0/REPORTE_FASE0.md`, que es la ubicación canónica del experimento (panel E4B/MIA, no 31B) |
| `experimentos_pendientes.md` | `/home/plaxius/Escritorio/Buscando_la_geometría/experimentos_pendientes.md` | copia sincronizada de `~/Documentos/Proyectos/Geometría_LSGOT/experimentos_pendientes.md`; **2 IPs de pod redactadas** |
| `informe_poster_y_lsgot.md` | `/home/plaxius/Escritorio/Buscando_la_geometría/informe_poster_y_lsgot.md` | escrito/actualizado esta sesión, sin copia en `Geometría_LSGOT/` |

## 3. `data/` (JSON agregados — sin `.npz`)

| Archivo/directorio | Origen | Qué contiene |
|---|---|---|
| `sia_extended_v5/results.json` | `~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local/sia_extended_v5/results.json` | Δκ, W₁, reducción de dimensión (Ollivier-Ricci) — evidencia secundaria, ver caveat Fase0 |
| `sia_extended_v5/_tier0_metrics.json` | ídem, `_tier0_metrics.json` | RQA/Hurst/PR/identity_projection por trayectoria (salida cruda de `analyze_tier0.py`) |
| `sia_extended_v5/_dims.json`, `_traj_metrics.json`, `_graphs.json` | ídem | dimensión intrínseca, métricas de trayectoria, grafos k-NN (curvatura) |
| `sia_extended_v5/*_responses.json` (9 archivos, uno por grupo) | ídem | texto generado por prompt/grupo — no son embeddings, son las respuestas en texto plano |
| `perturbation_sia_L30_medium/{summary,details}.json` | `~/Documentos/Proyectos/.../perturbation/results/perturbation_sia_L30_medium/` | H4_rev: τ, recovery_rate, displacement_l2, sampen, etc. por grupo/t_inj/prompt (10 grupos) |
| `perturbation_sia_L30_medium/EE_EH_WINDOW_REPORT.json` | ídem | salida cruda de `analyze_tier0_perturbation.py` (Fréchet + τ_identidad) |
| `perturbation_sia_L30_medium/RESCUE_COVERAGE.json` | ídem | cobertura del rescate de trayectorias (767/800, 95.9%) |
| `perturbation_ei_L30_medium/{summary,details}.json` | `~/Documentos/Proyectos/.../perturbation/results/perturbation_ei_L30_medium/` | E-I: τ por dirección (along/orthogonal) × t_inj × grupo |
| `perturbation_ei_L30_medium/EI_PERMUTATION_REPORT.json` | ídem | salida cruda de `analyze_ei_permutation.py` |

**No copiado (excluido por tamaño):** `perturbation_sia_L30_medium/trajectories/*.npz` (10 grupos × baseline+perturbado, ~600MB+) y `perturbation_ei_L30_medium/trajectories/*.npz` (14 archivos, 42.7MB c/u, ~580MB) — ambos en las mismas rutas de origen que sus `summary.json`/`details.json` de arriba, subdirectorio `trajectories/`.

## 4. `scripts/`

Todos desde `~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/`:

| Archivo | Ruta relativa al origen | Rol |
|---|---|---|
| `analyze_tier0.py` | `analyze_tier0.py` | E-A/C/D/H sobre trayectorias libres — genera `TIER0_REPORT.md` |
| `analyze_tier0_perturbation.py` | `analyze_tier0_perturbation.py` | E-E (Fréchet) + E-H ventana — genera `EE_EH_WINDOW_REPORT.md/json` |
| `perturbation/perturbation_extractor.py` | `perturbation/perturbation_extractor.py` | extracción de trayectorias base + perturbadas (isotrópicas y direccionales), cacheo por pkl |
| `perturbation/recovery_analyzer.py` | `perturbation/recovery_analyzer.py` | cómputo de τ/recovery_rate (protocolo H4_rev) |
| `perturbation/run_perturbation.py` | `perturbation/run_perturbation.py` | corredor principal H4_rev (perturbación isotrópica, 10 grupos) |
| `perturbation/run_perturbation_directional.py` | `perturbation/run_perturbation_directional.py` | corredor E-I (perturbación direccional along/orthogonal) |
| `perturbation/analyze_ei_permutation.py` | `perturbation/analyze_ei_permutation.py` | permutation test de E-I — genera `EI_PERMUTATION_REPORT.md/json` |
| `perturbation/reconsolidate_rescue.py` | `perturbation/reconsolidate_rescue.py` | reconsolida pkls rescatados por streaming del pod (sesión 2026-08-22) en `.npz` |
| `perturbation/v_identidad.npy` | `perturbation/v_identidad.npy` | vector unitario (5376-dim) `mean(axis)−mean(generic_long)`, prerequisito de E-I y E-H |
| `shared/tier0_metrics.py` | `shared/tier0_metrics.py` | funciones: `persona_vector`, `project_trajectory`, cómputo de RQA/Hurst/PR |
| `shared/statistical_tests.py` | `shared/statistical_tests.py` | `GeometricStatisticalTests` — permutation test, Cohen's d, bootstrap CI, usado por todos los `analyze_*.py` |

**No copiado:** `perturbation/merge_and_analyze.py` (utilidad de fusión de corridas parciales, no necesaria para reproducir desde cero), `analysis_restricciones.py`/`calibrate_sigma.py`/`deploy_fase1_punto2.py`/`smoke_test.py`/`verify_tokens.py`/`freeze_manifest*.py` (utilidades operativas de la corrida en pod, no del análisis) y el resto de `shared/*.py` (`curvature_analyzer.py`, `dimensionality_analyzer.py`, `graph_builder.py`, `hidden_state_extractor.py`, `analyze_autopoiesis.py`, `visualization.py`) — pertenecen al pipeline de extracción/Δκ (evidencia secundaria) y no son necesarios para reproducir la evidencia primaria de `lsgot_4.md`. Si se decide auditar o reproducir Δκ/W₁ (`REPORTE_FASE0.md`-style, ver `lsgot_4.md` §7), estos archivos hay que traerlos aparte desde la misma ruta base.

## 5. Requisitos no incluidos (por diseño)

- **Token de HuggingFace** con acceso aceptado a `google/gemma-4-31B-it` — nunca se guardó en disco, se pasa por `--token` en línea de comandos.
- **Pesos del modelo** `google/gemma-4-31B-it` (~62GB BF16) — se descargan de HuggingFace Hub al correr los scripts.
- **GPU** A100/H100 80GB para cualquier corrida nueva de `run_perturbation*.py` (min_vram_gb=65). Los scripts `analyze_*.py` sobre los JSON ya incluidos en `data/` no requieren GPU.
