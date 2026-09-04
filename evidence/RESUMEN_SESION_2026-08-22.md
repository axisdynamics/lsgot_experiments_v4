# Sesión RunPod 2026-08-22 — cierre de experimentos_pendientes.md

**Pod:** A100 80GB PCIe, `[REDACTED_POD_IP:PORT]`. Sin Network Volume previo
("no hay volumen hay que hacer todo de nuevo") — descarga completa del
modelo desde cero. Presupuesto: los últimos $10 del investigador.

## Timeline

1. Setup: torch 2.4.1+cu124 (versión por defecto del índice cu124) resultó
   incompatible con `transformers>=4.50` (requiere torch≥2.5) → actualizado a
   `torch==2.6.0+cu124`. Eso rompió el ABI de `torchvision`
   (`operator torchvision::nms does not exist` al importar `Gemma4Config`,
   porque `transformers.image_utils` importa `torchvision` incondicionalmente
   si está instalado). Fix: `pip uninstall torchvision torchaudio` — no se
   usan en este pipeline (solo texto).
2. **Punto 2 (MIA gemma4-31b-it)** corrido primero por ser más rápido/barato:
   ~30 min, 4 grupos × 20 prompts, sin errores. Modelo descargado a
   `/workspace/models/gemma-4-31B-it` (62.6 GB, Network Volume del pod).
3. **Punto 1 (H4_rev SIA-31B)** arrancado reusando el modelo ya en el
   volumen (sin re-descarga). A los ~3 min de generación real (tras validar
   prompt 1 completo) crasheó:
   ```
   TypeError: unsupported format string passed to NoneType.__format__
   ```
   Causa raíz: `compute_recovery()` en `recovery_analyzer.py` devuelve
   `displacement_l2`/`recovery_gap` = `None` cuando la ventana post-inyección
   tiene menos de `TAU_MIN_WINDOW=5` tokens (`_empty_result`,
   `"insufficient_post_tokens"`) — ocurre cuando la perturbación en `t_inj`
   hace que el modelo llegue a EOS casi de inmediato. El print en
   `run_perturbation.py` ya manejaba ese caso para `tau_tokens` (`τ=∞`) pero
   no para `displacement_l2`/`recovery_gap`. Fix aplicado y verificado en
   `run_perturbation.py` (guardas `disp_str`/`gap_str` con el mismo patrón).
   Este bug es independiente del parche de persistencia de trayectorias
   (`--save-trajectories`, agregado antes de esta sesión) — no lo introdujo.
4. Relanzado con el fix. El log quedó "sin avance" aparente por un buen
   rato — no era un cuelgue: Python usa buffering por bloques (no por línea)
   cuando `stdout` no es una tty, así que el progreso real (confirmado con
   `nvidia-smi` a 98% GPU util y con el conteo de `.pkl` creciendo en
   `cache_sia_L30_medium/`) no aparecía en el log hasta mucho después.
   Lección: para corridas largas en background, monitorear por efectos
   secundarios en disco (conteo de archivos de caché), no solo por el log.
5. Corrida completa: **275.6 min**, 10 grupos × 3 t_inj × 20 prompts,
   `summary.json` + `details.json` + 40 `trajectories/*.npz` escritos en el
   pod.
6. Al bajar los resultados finales, el pod se quedó sin crédito **a mitad de
   la transferencia** (`Connection to ... closed by remote host`,
   luego `Connection refused`). Se alcanzó a bajar:
   - `summary.json`, `details.json` completos y válidos (contienen el 100%
     de las estadísticas agregadas — τ, recovery_rate, displacement_l2,
     sampen_delta, w1_norms — para las 800 combinaciones grupo×t_inj×prompt).
   - 1 de 40 `trajectories/*.npz` completo
     (`axis_pec_only_perturbed_t200`); un segundo archivo llegó truncado
     (detectado y descartado, `zipfile` inválido).
   - **767/800 (95.9%) de los `.pkl` crudos individuales** rescatados por
     streaming (`tar | tar`, no `scp` — mucho más rápido para cientos de
     archivos chicos) *antes* de que se cortara la conexión, gracias a que
     se lanzó ese rescate en paralelo apenas se notó que el crédito se
     estaba agotando.
7. **Reconsolidación local sin GPU** (`reconsolidate_rescue.py`): recalcula
   los mismos hashes md5 que usa `PerturbationExtractor._cache_key()`
   (mismo `model_path`, `sigma_value` exacto tomado de `summary.json`,
   mismos prompts/system-prompts locales) para encontrar qué `.pkl`
   rescatado corresponde a qué (grupo, prompt, t_inj), y arma los mismos 40
   `.npz` que el pod no alcanzó a terminar de bajar.

## Cobertura final de trayectorias crudas (`RESCUE_COVERAGE.json`)

| Grupo | baseline | t_inj=50 | t_inj=128 | t_inj=200 |
|---|---|---|---|---|
| axis | 20/20 | 20/20 | 20/20 | 20/20 |
| generic_long | 20/20 | 20/20 | 20/20 | 20/20 |
| generic_short | 20/20 | 20/20 | 20/20 | 20/20 |
| vanilla | 20/20 | 20/20 | 20/20 | 20/20 |
| axis_short | 20/20 | 20/20 | 20/20 | 20/20 |
| chat_agente | 19/20 | 19/20 | 18/20 | 18/20 |
| automata_neutro | 19/20 | 17/20 | 15/20 | 14/20 |
| chat_agente_sia | 19/20 | 19/20 | 19/20 | 19/20 |
| chat_agente_sia_v2 | 19/20 | 19/20 | 19/20 | 19/20 |
| axis_pec_only | 19/20 | 19/20 | 19/20 | 19/20 |

Total: 767/800 (95.9%). Los 5 grupos del panel original quedaron al 100%;
`automata_neutro` es el más incompleto (70% en t_inj=200), consistente con
ser el grupo con más generaciones cortas/variables ya documentado
(`recovery_rate<1.00` en `AUTOMATA_NEUTRO_REPORT.md`).

**Nota:** esto solo afecta la disponibilidad de trayectorias *crudas* para
E-E (Fréchet) y la mitad "ventana" de E-H sobre las ~33 combinaciones
faltantes. Las métricas agregadas (τ, recovery_rate, etc.) en
`summary.json`/`details.json` están completas al 100% — esas 33
combinaciones sí se computaron y analizaron en el pod, solo no se alcanzó a
bajar el array crudo de esos 33 casos puntuales antes del corte.

## Artefactos finales

- `results/gemma4_31b-it_mia/` (Punto 2) — completo.
- `results/perturbation_sia_L30_medium/summary.json` + `details.json` —
  completos.
- `results/perturbation_sia_L30_medium/trajectories/*.npz` (40 archivos) —
  reconsolidados, 95.9% de cobertura por archivo (ver tabla arriba).
- `rescue_cache_sia_L30_medium/` (767 `.pkl`, 3.9 GB) — caché crudo de
  respaldo, ya consolidado en los `.npz` de arriba; se puede borrar sin
  pérdida de información si se necesita el espacio.
