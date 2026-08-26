# E-I — Perturbación direccional: CORRIDO (2026-08-26)

**Diseñado:** 2026-08-22. **Corrido:** 2026-08-26 en RunPod A100 80GB PCIe
(pod `[REDACTED_POD_IP]`), 280/280 generaciones, 143.2 min, cero errores.
Resultados en `results/perturbation_ei_L30_medium/` (`summary.json`,
`details.json`, 14 `.npz` de trayectorias crudas, 20 prompts × 256 tokens ×
5376 dims cada una, verificadas sin NaN/Inf).

## Resultado — τ (recuperación, en tokens) por dirección

| grupo | t_inj | τ_along | τ_orthog | Δ(along−orthog) | rec_along | rec_orthog |
|---|---|---|---|---|---|---|
| axis | 50 | 20.7 | 21.9 | −1.2 | 1.00 | 1.00 |
| axis | 128 | 18.8 | 18.2 | +0.6 | 1.00 | 1.00 |
| axis | 200 | 16.8 | 16.5 | +0.3 | 1.00 | 1.00 |
| axis_pec_only | 50 | 24.4 | 21.4 | **+3.0** | 1.00 | 1.00 |
| axis_pec_only | 128 | 19.4 | 18.2 | +1.2 | 1.00 | 1.00 |
| axis_pec_only | 200 | 17.3 | 17.6 | −0.2 | 1.00 | 1.00 |

**Permutation test (2026-08-26, n_perm=1000): ninguna de las 6 condiciones
alcanza p<0.05** (p=0.148-0.480, d=−0.15 a +0.34, todo "negligible" a
"small"). El Δ más grande (axis_pec_only, t_inj=50, +3.0 tokens, d=+0.34) va
en la dirección predicha pero no es significativo con n=20, y el signo no es
consistente entre t_inj. **E-I es un null result válido**: `axis_pec_only`
recupera aproximadamente igual sea cual sea la dirección del empujón
relativa a `v_identidad` — no hay evidencia de un atractor direccional en
esta métrica. Detalle completo, métricas secundarias (displacement_l2 con
patrón 6/6 direccionalmente consistente pero no significativo por celda) y
limitaciones en `EI_PERMUTATION_REPORT.md`.

**Original (histórico, referencia):** código escrito y verificado
(matemática del ruido, sintaxis, integración con el pipeline de H4_rev)
antes de correr.

## Qué hace

En vez de ruido isotrópico ε~N(0,σ²·I) (H4_rev), inyecta un vector de
**magnitud fija M** en una dirección controlada relativa a
`v_identidad = mean(axis) − mean(generic_long)`:

- **along**: `ε = M·s·v̂` (s=±1 aleatorio) — empuja exactamente a lo largo
  del eje identitario.
- **orthogonal**: `ε = M · proj⊥(η)/‖proj⊥(η)‖` (η~N(0,I)) — empuja en una
  dirección aleatoria perpendicular a ese eje.

Magnitud pareada entre ambas condiciones (a diferencia de H4_rev, donde
σ genera una norma *esperada* pero variable) — cualquier diferencia en
τ/recovery_rate entre "along" y "orthogonal" es atribuible a la dirección,
no al tamaño del empujón.

**Pregunta que responde:** si `axis_pec_only` (Factor 2 puro, sin
automatismo) recupera igual sea cual sea la dirección del empujón, el
"atractor de identidad" es un pozo genérico, no direccional. Si en cambio
recupera peor/más lento específicamente cuando lo empujan a lo largo de su
propio eje identitario, eso es la evidencia más directa posible de que hay
una dirección privilegiada — no solo una región privilegiada.

## Ya resuelto (sin GPU)

- `v_identidad.npy` — vector unitario de 5376 dims, ya calculado y
  congelado desde `results_local/sia_extended_v5/{axis,generic_long}_embeddings.npz`
  (mismo vector que usa `EE_EH_WINDOW_REPORT.md`). Verificado: `‖v‖=1`.
- `DirectionalPerturbationHookManager` en `perturbation_extractor.py` —
  clase nueva, misma interfaz (`activate`/`deactivate`/`register`/`remove`)
  que `PerturbationHookManager`, así que `_generate()` no necesitó cambios.
  Matemática verificada con datos sintéticos: "along" da norma exacta M y
  coseno ±1 con v̂; "orthogonal" da norma exacta M y coseno ~0 con v̂.
- `PerturbationExtractor.extract_perturbed_directional()` — mismo patrón
  de cacheo por hash md5 que `extract_perturbed`, clave separada
  (`_cache_key_directional`) para no colisionar con las corridas
  isotrópicas ya guardadas.
- `run_perturbation_directional.py` — script completo, reutiliza
  `resolve_model_path`, `load_prompts`, `load_system_prompt`, `MODEL_BASE`,
  `LAYER_SETS`, `_save_traj_npz` de `run_perturbation.py` (sin duplicar
  código). Guarda `summary.json` + `details.json` + trayectorias crudas
  `.npz` (activado por defecto, mismo parche de persistencia que H4_rev).

## Comando para correr

```bash
cd SIA-experiments/gemma4_31b_combined/perturbation/
python run_perturbation_directional.py --token hf_xxxxx
```

Por defecto corre los dos grupos prioritarios de `Set_experimental.md`
(`axis`, `axis_pec_only`) × 2 direcciones × 3 t_inj × 20 prompts, capa L30
(idx 29, la misma de H4_rev), magnitud `medium` (M=438.4, mismo
`mean_velocity` calibrado que SIA_CONFIG).

Para correr solo un grupo (más barato, ~50% del costo):
```bash
python run_perturbation_directional.py --groups axis_pec_only --token hf_xxxxx
```

## Costo estimado

280 generaciones (2 grupos × 20 prompts × (1 baseline + 2 direcciones × 3
t_inj)) — **no** 800 como H4_rev, porque solo hay 2 grupos y no se repite
la condición isotrópica. A ~22s/generación (tasa observada en H4_rev):

- Generación: ~105 min
- Descarga del modelo (62.6 GB, asumiendo sin Network Volume): ~5 min
- Carga a GPU + overhead: ~5 min
- **Total: ~1.8-2 hrs**, ~$1.5-3 en spot A100 80GB a $0.70-1.50/hr.

Si el Network Volume de esta sesión (2026-08-22) sigue vivo en RunPod,
descontar los ~5-10 min de descarga — revisar antes de lanzar, mismo
criterio que Punto 1 de `experimentos_pendientes.md`.

## Qué mirar en el resultado

En el output de `run_perturbation_directional.py` (tabla final, y en
`summary.json` → `aggregated[group][direction][t_inj]`):

- **`axis_pec_only`, Δτ(along − orthogonal) claramente positivo** en varios
  t_inj → confirma atractor direccional (recupera peor cuando lo empujan a
  lo largo de su propio eje). Δτ≈0 o negativo → el atractor no es
  direccional, es un pozo genérico — resultado también válido, no un null
  result vacío.
- **Comparar el patrón `axis` vs `axis_pec_only`**: si la direccionalidad
  cambia cuando además está presente la densidad de restricción (Factor 1),
  eso es información nueva sobre cómo interactúan los dos factores más allá
  de lo que ya se sabe (recuperación agregada).
- Reusar `analyze_tier0_perturbation.py` (adaptado, cambiar `PERT_DIR` a
  `results/perturbation_ei_L30_medium/` y las claves de grupo a
  `{group}_along_t{t}`/`{group}_orthogonal_t{t}`) para correr Fréchet y
  τ_identidad sobre estas trayectorias también — mismo código, ya probado.
