# Auto-auditoría de Δκ/W₁ sobre el panel Gemma-4-31B-it

**Fecha:** 2026-08-31
**Origen:** ítem #1 de "Future work" en `paper/lsgot_4.md` §7.2 — el protocolo de
`REPORTE_FASE0.md` (auditoría pre-registrada sobre un panel hermano: Gemma-4-E4B/-it,
DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-7B-Instruct) nunca se había corrido sobre el
panel de este paper.
**Scripts:** `scripts/audit_curvatura/` (adaptados de `MIA-experiments/fase0/scripts/`)
**Datos:** `data/sia_extended_v5/_graphs.json` (grafos k-NN ya construidos, incluido
en este repo) + embeddings crudos externos (excluidos por tamaño, ver README.md)

## 0. Hallazgo previo, antes de correr nada: la métrica no es la que dice ser

`curvature_analyzer.py` (idéntico, md5 verificado, en `SIA-experiments` y en
`MIA-experiments` — es decir, en el panel de este paper y en el panel hermano de
`REPORTE_FASE0.md` por igual) define `class FormanRicciAnalyzer` con la fórmula
combinatoria de Forman:

```
κ_F(e_uv) = w(e_uv) − Σ_{f⊃e} w(f)/|f| + Σ_{g⊃e} w(g)/|g|
```

Esto es **curvatura de Forman-Ricci** — puramente combinatoria, sobre pesos de
aristas y ciclos. No es curvatura de Ollivier-Ricci, que se define vía transporte
óptimo entre las distribuciones de vecinos de cada nodo (`κ = 1 − W₁(μ_x,μ_y)/d(x,y)`)
y requiere resolver un problema de transporte por arista — nada de eso aparece en
esta implementación.

**Consecuencia:** cada mención de "Ollivier-Ricci Δκ/W₁" en este proyecto —
`paper/lsgot_4.md` (§2.3, §3.4, Apéndice A, Referencias — cita a Ollivier, Y. 2009),
`REPORTE_FASE0.md`, `TIER0_REPORT.md` — describe en realidad una cifra de
Forman-Ricci. El error es heredado del origen del proyecto (mismo archivo, mismo
md5, en ambos paneles), no algo introducido en esta sesión. No cambia el veredicto
de fondo (el paper ya trata esta familia de métricas como evidencia secundaria, no
como sostén de ninguna conclusión), pero es una atribución incorrecta que un
revisor familiarizado con la literatura de curvatura discreta va a notar. Ver §4
para la recomendación de corrección.

De acá en adelante este reporte usa el nombre correcto: **Forman-Ricci** (Δκ_F/W₁_F,
o simplemente "curvatura" donde el contexto ya lo deja claro).

## 1. Exp 0.3 análogo — escala de referencia (split-half null)

**Método:** para cada condición, se dividen sus 20 trayectorias en dos mitades
aleatorias B=1000 veces y se calcula W₁ (Forman-Ricci edge-wise, pooled) entre las
mitades — la distribución resultante es el "piso de ruido" contra el que se compara
el W₁ observado entre condiciones distintas.

| Par | W₁ observado | vs. piso de A | vs. piso de B | Veredicto |
|---|---|---|---|---|
| `axis_pec_only` vs `automata_neutro` | 0.4349 | pct 0.0%, ×17.0 | pct 0.8%, ×3.5 | **señal fuerte** |
| `automata_neutro` vs `vanilla` | 0.4360 | pct 0.7%, ×3.5 | pct 0.0%, ×13.2 | **señal fuerte** |
| `axis` vs `vanilla` | 0.0614 | pct 34.2%, ×1.3 | pct 7.5%, ×1.9 | marginal |
| `axis_pec_only` vs `vanilla` | 0.0133 | pct 91.8%, ×0.5 | pct 94.6%, ×0.4 | **en ruido** |

`automata_neutro` vs `vanilla` (0.4360) replica casi exactamente el Δκ=+0.436 ya
citado en `TIER0_REPORT.md`/`paper/lsgot_4.md` §3.4 — buena señal de consistencia
interna del pipeline. El patrón es limpio: **el par que aísla restricción pura
(`automata_neutro`) sostiene señal muy por encima del ruido en las dos
comparaciones donde participa; el par que aísla identidad pura frente al mínimo
(`axis_pec_only` vs `vanilla`) cae directamente dentro del ruido de split-half —
0.4-0.5× la mediana del propio ruido de cada condición.** `axis` vs `vanilla`
(con ambos factores) queda en un punto intermedio, marginal.

**Lectura:** Forman-Ricci, tal como está implementada acá, tiene señal real —
pero es una señal de **restricción**, no de identidad. Esto es exactamente
consistente con lo que ya muestran PR, RQA y v̂ en el resto del paper — no es un
hallazgo nuevo, es una confirmación independiente de que esta familia de métricas
no debería usarse como evidencia de identidad, ni siquiera secundaria.

## 2. Exp 0.2 análogo — ¿la curvatura detecta algo que los baselines simples no?

**Método:** para los mismos 4 pares, se compara Forman-Ricci W₁ contra: distancia
de centroides, MMD (sobre v1 y sobre un pool de 10 tokens/trayectoria), CKA lineal
sobre v1, y AUC de un probe lineal (LogReg + StandardScaler, 5-fold GroupKFold por
prompt) — todos con test de permutación (B=500, seed=42).

| Par | Forman-Ricci W₁ | Centroides (p) | MMD v1 (p) | MMD pool (p) | Probe AUC v1 (p) |
|---|---|---|---|---|---|
| `axis_pec_only` vs `automata_neutro` | 0.435 (fuerte) | 132.7 (0.002) | 0.473 (0.002) | 0.051 (0.002) | **1.000** (0.020) |
| `automata_neutro` vs `vanilla` | 0.436 (fuerte) | 96.2 (0.002) | 0.335 (0.002) | 0.029 (0.002) | **1.000** (0.020) |
| `axis` vs `vanilla` | 0.061 (marginal) | 56.1 (0.002) | 0.535 (0.002) | 0.017 (0.002) | **1.000** (0.020) |
| `axis_pec_only` vs `vanilla` | 0.013 (**en ruido**) | 60.7 (0.002) | 0.488 (0.002) | 0.016 (0.002) | **1.000** (0.020) |

**Veredicto: en 4/4 comparaciones, los baselines simples detectan la diferencia con
p≤0.002 — incluido el par donde Forman-Ricci está en el ruido de split-half
(`axis_pec_only` vs `vanilla`).** El probe lineal sobre v1 separa perfectamente
(AUC=1.000) en los cuatro pares, sin excepción. Es el mismo patrón, con el mismo
método, que `REPORTE_FASE0.md` §Exp 0.2 encontró en el panel hermano (12/12
comparaciones, baselines ganan). La curvatura no aporta nada que la distancia de
centroides o un probe lineal no capturen ya, más barato y sin las 4 aristas/ciclos
por trayectoria que exige el cómputo de Forman.

## 3. Exp 0.1 análogo — confound de primer token (nivel proxy, sin GPU)

**Método:** proxy textual (primera palabra decodificada de la respuesta, no el
token id exacto — eso requeriría forward pass, ver §5), entropía de Shannon,
Jaccard top-10, y residualización OLS de ‖v1‖ ~ condición + dummies de
primera-palabra (top-8), con test de permutación (B=2000) sobre el coeficiente de
condición.

| Condición | Entropía (bits) | Top-3 primera palabra |
|---|---|---|
| `axis` | 1.590 | respiro (12/20), pausa (5/20) |
| `axis_pec_only` | 1.919 | pausa (11/20), esta (4/20) |
| `automata_neutro` | 2.823 | este (5/20), no (5/20) |
| `vanilla` | 2.319 | esta (11/20), para (2/20) |

| Par | Jaccard top-10 | coef. sin control | coef. con control | p_perm (con control) |
|---|---|---|---|---|
| `axis` vs `vanilla` | **0.00** | +43.1 | **+83.1** | 0.0005 |
| `automata_neutro` vs `vanilla` | 0.29 | −25.6 | −35.6 | 0.0005 |
| `axis_pec_only` vs `vanilla` | 0.25 | −28.5 | −36.3 | 0.0010 |
| `axis_pec_only` vs `automata_neutro` | 0.07 | −2.9 | −13.3 | 0.1304 |

**El confound de primer token es real** — `axis` y `vanilla` no comparten NINGUNA
palabra en su top-10 (Jaccard=0.00), la señal de alarma más fuerte posible, igual
de extrema que el caso gemma-it de `REPORTE_FASE0.md` (Jaccard=0.07 ahí).

**Pero, a diferencia del panel hermano, acá el efecto NO se destruye al
controlar.** En los tres pares donde `‖v1‖` difiere de forma significativa, el
coeficiente de condición **sobrevive y se refuerza** al controlar por la
primera-palabra — incluido el caso de Jaccard=0.00, donde el coeficiente casi se
duplica (+43→+83) en vez de desaparecer. Esto es lo opuesto al hallazgo central de
`REPORTE_FASE0.md` §Exp 0.1B/1C, donde el control post-token *revertía* la
inversión de norma en deepseek y gemma-it. El único par sin efecto significativo
(`axis_pec_only` vs `automata_neutro`, p=0.13) es exactamente el par donde ya
sabíamos que `‖v1‖` no es donde vive la señal — ahí el hallazgo real es
direccional (v̂), no de magnitud (ver `EL_REPORT.md`, `TIER0_REPORT.md`).

**Caveat, el mismo que declara `REPORTE_FASE0.md`:** este es el proxy textual
(primera palabra decodificada), no el control definitivo con token forzado
mediante forward pass (Exp 0.1C ahí) — el proxy es grueso. Es tranquilizador, no
concluyente. Ver §5.

## 4. Recomendación — corregir la atribución Ollivier→Forman

- `paper/lsgot_4.md`: §2.3 ("Ollivier-Ricci curvature Wasserstein statistic"),
  §3.4 (título y cuerpo), Apéndice A si aplica, y la referencia a Ollivier (2009)
  en el listado de Referencias — reemplazar por "Forman-Ricci" en todo el texto y
  retirar o recalificar la cita de Ollivier (2009), que no corresponde a lo
  calculado. Añadir la cita correcta (Forman, R. 2003, *Bochner's method for cell
  complexes and combinatorial Ricci curvature*, Discrete & Computational Geometry)
  si se quiere mantener el aparato formal.
- `README.md`: sección "Nota metodológica importante" usa el mismo nombre.
- `TIER0_REPORT.md`, `REPORTE_FASE0.md` (copia local): mismo ajuste, o al menos
  una nota de corrección — `REPORTE_FASE0.md` es de otro proyecto/panel
  (MIA-experiments), no se tocó en esta sesión.
- No es necesario recalcular ningún número — el error es de **nombre**, no de
  cómputo; los valores de Δκ/W₁ ya reportados no cambian.

## 5. Lo que queda pendiente (requiere GPU)

El control **definitivo** de primer token (Exp 0.1C de `REPORTE_FASE0.md`): forward
pass con el token forzado a una constante, capturando h_post en vez de h_pre, para
descartar el proxy textual grueso. Requiere acceso al modelo Gemma-4-31B-it
completo (31B, no cabe en una GPU de consumo con precisión completa) — un pod
nuevo. Dado que el proxy ya es tranquilizador (el efecto sobrevive y se refuerza en
3/4 pares, no se revierte como en el panel hermano), y que ninguna afirmación del
paper depende de `‖v1‖` en sí (v̂ es una proyección coseno, no una norma), la
prioridad de este control específico es más baja que cuando se escribió §7 del
paper — pero sigue siendo el único hueco que separa "tranquilizador" de
"descartado".

## 6. Síntesis

1. **Hallazgo no buscado, el más importante de este pase:** la métrica de
   curvatura de todo el proyecto es Forman-Ricci, no Ollivier-Ricci como está
   citada en `paper/lsgot_4.md` y `REPORTE_FASE0.md` — error heredado del origen
   del proyecto, no introducido acá. Corrección de nombre, no de números (§4).
2. **Forman-Ricci en este panel: señal real pero es de restricción, no de
   identidad** — sostiene ruido para `automata_neutro`, cae en el piso de ruido
   split-half para la comparación de identidad pura (`axis_pec_only` vs
   `vanilla`). Confirma, no contradice, el resto del paper.
3. **Los baselines simples ganan en 4/4 comparaciones** (probe AUC=1.000 en las
   cuatro) — mismo patrón que el panel hermano. La curvatura no agrega nada sobre
   este panel tampoco.
4. **El confound de primer token es real (Jaccard hasta 0.00) pero, a nivel
   proxy, no destruye ningún efecto de `‖v1‖` en este panel** — al revés que en
   el panel hermano, donde sí lo hacía. Tranquilizador, no definitivo (falta el
   control con token forzado, §5).
5. Con esto, el paper conserva su tratamiento de Δκ/W₁ como evidencia secundaria
   no cargante — la auto-auditoría lo confirma en vez de obligar a retirar nada
   ya escrito. El único cambio de fondo que sí hace falta es el de nombre (§4).
