# E-I — Permutation test (along vs orthogonal a v_identidad)

**Fecha:** 2026-08-26. Datos: `results/perturbation_ei_L30_medium/details.json`
(280/280 generaciones, n=20 prompts por celda, RunPod A100 80GB, 143.2 min).
Script: `analyze_ei_permutation.py`. JSON crudo: `EI_PERMUTATION_REPORT.json`.
Método: permutation test bilateral (mean_difference, n_perm=1000), mismo que
`TIER0_REPORT.md` / `EE_EH_WINDOW_REPORT.md`.

## Resultado principal — τ_tokens (métrica de diseño de E-I)

| grupo | t_inj | media along | media orthog | Δ | p | d | interpretación |
|---|---|---|---|---|---|---|---|
| axis | 50 | 20.70 | 21.90 | −1.20 | 0.351 | −0.150 | negligible |
| axis | 128 | 18.80 | 18.20 | +0.60 | 0.392 | +0.100 | negligible |
| axis | 200 | 16.78 | 16.53 | +0.25 | 0.477 | +0.034 | negligible |
| axis_pec_only | 50 | 24.45 | 21.45 | +3.00 | 0.148 | +0.340 | small |
| axis_pec_only | 128 | 19.40 | 18.20 | +1.20 | 0.271 | +0.195 | negligible |
| axis_pec_only | 200 | 17.30 | 17.55 | −0.25 | 0.480 | −0.030 | negligible |

**Ninguna de las 6 condiciones alcanza p<0.05.** El Δ más grande
(axis_pec_only, t_inj=50, +3.0 tokens, d=+0.34 "small") va en la dirección
predicha por `Set_experimental.md` (recuperación más lenta cuando el
empujón es a lo largo del eje identitario) pero no es significativo con
n=20. El signo de Δ tampoco es consistente entre t_inj dentro del mismo
grupo (axis_pec_only: +3.00, +1.20, −0.25).

**Lectura:** E-I es un **null result válido**, no un experimento fallido —
`axis_pec_only` (Factor 2 puro, sin automatismo) recupera aproximadamente
igual de rápido sea cual sea la dirección del empujón relativa a
`v_identidad`. Esto no confirma la hipótesis de "atractor de identidad
direccional" (Panel 4 de `Geometry.jpeg`, ver `informe_poster_y_lsgot.md`):
con los datos actuales, el atractor de recuperación parece ser un pozo
genérico en el espacio de representación, no uno anclado específicamente a
la dirección `v_identidad = mean(axis) − mean(generic_long)`.

## Métricas secundarias (exploratorias, no el diseño original de E-I)

- **`displacement_l2`**: patrón direccionalmente consistente
  (along > orthogonal) en **las 6 de 6** combinaciones grupo×t_inj, efecto
  "small" (d=0.21-0.40), pero ninguna cruza p<0.05 individualmente
  (p=0.10-0.24). Con 6/6 en la misma dirección, no es puramente ruido, pero
  tampoco alcanza significancia por celda — candidato a revisar con un test
  agregado (p. ej. combinar p-valores, o repetir con más prompts) antes de
  interpretarlo como señal real.
- **`sampen_delta`**: un resultado "medium" aislado (axis, t_inj=200,
  p=0.043) sin patrón consistente en las otras 5 celdas — con 24 tests en
  total (4 métricas × 6 celdas) sin corrección por comparaciones múltiples,
  es el tipo de resultado que se espera por azar (~1 de cada 24 a p<0.05) y
  no debería tratarse como hallazgo sin replicación.
- **`w1_norms_post`**: sin patrón, todos "negligible" a "small", p>0.10 en
  todas las celdas.

## Qué cambia esto para `Set_experimental.md`

El "Orden de ejecución sugerido" condicionaba **E-B (Lyapunov local)** a
"evaluar si el costo de multiplicar corridas de perturbación se justifica
según lo que ya muestre E-I". Dado que E-I no encontró una señal
direccional significativa en la métrica de diseño (τ_tokens), **no hay
evidencia que justifique correr E-B todavía** sobre este mismo eje
(along/orthogonal) — el costo de más corridas de perturbación no se paga
sin un candidato más prometedor. El patrón exploratorio en
`displacement_l2` (6/6 en la misma dirección, no significativo por celda)
es lo único que quedaría como pista a seguir, si acaso con más prompts por
celda antes de invertir en E-B.

## Limitaciones

- n=20 por celda (mismo n que el resto de Tier 0/2) — potencia limitada
  para detectar efectos "small" (d~0.2-0.3) con permutation test bilateral;
  un Δ verdadero de esa magnitud requeriría n considerablemente mayor para
  cruzar p<0.05 de forma confiable.
- Solo 2 grupos (axis, axis_pec_only) — condiciones prioritarias de
  `Set_experimental.md`, no se corrió sobre el resto del panel de 10 grupos
  (vanilla, generic_long, chileatiende, etc.), que quedaría para una
  extensión futura si se decide seguir esta línea.
- 3/240 trayectorias con `tau_tokens=None` por `insufficient_post_tokens`
  en t_inj=200 (2 en axis/along, 1 en axis/orthogonal) — excluidas del
  test, n efectivo 18-20 según celda.
