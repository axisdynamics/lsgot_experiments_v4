# Reencuadre del hallazgo: de "identidad anclada" a "densidad de restricción"

**Fecha:** 2026-08-19
**Estado:** hoja de ruta para plantear el descubrimiento — no reemplaza `AXIS_PEC_ONLY_REPORT.md` ni `Teoria_subconjunto_acotado.md`, los organiza hacia una narrativa publicable.

## 0. La vuelta de tuerca, en una frase

`lsgot_3.pdf` planteó la magnitud geométrica (Δκ, W₁, reducción de dimensión) como huella de **identidad anclada**. El panel SIA extendido (`automata_neutro`, `axis_pec_only`, `chat_agente_sia`) muestra que esa magnitud rastrea **densidad de restricción operativa** (Factor 1) — la identidad por sí sola no la produce; lo que sí depende de identidad cableada es la **recuperación tras perturbación** (τ, recovery_rate — Factor 2), una magnitud distinta.

## 1. Hipótesis original y por qué era razonable

- `lsgot_3.pdf` (Castillo, Torres, Lanas — arXiv:2607.09842) comparó `axis` (identidad) vs `generic` (sin identidad, longitud emparejada) vs `vanilla`, y encontró separación geométrica estadísticamente significativa asociada a `axis`.
- Con solo tres condiciones, "identidad" y "densidad de restricción operativa" estaban confundidas: `axis` tiene ambas (Triple PEC + jerarquía de bloques + identidad declarada). No había una condición que las separara.
- Conclusión razonable en ese momento: la huella geométrica es de identidad.

## 2. Lo que el panel extendido reveló

| Condición | Identidad/auto-referencia | Densidad de restricción | Δκ vs vanilla | Colapsa? |
|---|---|---|---|---|
| `automata_neutro` | No (verificado por grep) | Alta (59.3/1k) | +0.436, p<0.001 | **Sí** |
| `chat_agente` | No | Alta (54.6/1k) | +0.476, p<0.001 | **Sí** |
| `axis_pec_only` | Sí (identidad + Triple PEC cableado) | Baja (sin bloques/triggers) | +0.0011, p=0.41 (n.s.) | **No** |
| `axis` (completo) | Sí | Media (27.0/1k, por la arquitectura de bloques) | +0.061, p<0.001 | Sí, pero pequeño |

La celda que decide todo es `axis_pec_only`: toda la identidad de `axis`, cero arquitectura de autómata → geométricamente indistinguible de vanilla. Eso descarta identidad como causa del colapso y aísla densidad de restricción como el driver real.

Fuentes: `AUTOMATA_NEUTRO_REPORT.md`, `AXIS_PEC_ONLY_REPORT.md`, `Teoria_subconjunto_acotado.md` (tabla de densidad línea 219-226).

## 3. Lo que la identidad sí produce (para no perder el hallazgo bueno)

No es que identidad "no haga nada" — hace algo distinto de lo que `lsgot_3.pdf` medía:

- **τ / recovery_rate** (Factor 2): `axis_pec_only` recupera igual o más rápido que `axis` completo (τ=20.6/18.5/14.5 vs 21.1/19.6/16.3, recovery_rate=1.00 siempre) sin ninguna pieza de automatismo. `chat_agente_sia` (identidad presente pero subordinada al filtro) recupera peor que `chat_agente` puro. → la identidad cableada como paso obligatorio (no solo declarada) predice recuperación, no colapso.
- **Drift metacognitivo** (`LSGOT_v2_5.md:294-298`, hallazgo previo con el protocolo Witness): activación 44× mayor específicamente ante preguntas sobre la propia naturaleza del modelo — una firma de contenido-específico, no de colapso general.

## 4. Vacíos a cerrar antes de escribir la versión formal

1. **Densidad de restricción de `axis_pec_only` sin calcular todavía.** Falta correrle `analysis_restricciones.py` (ya usado para la tabla de `Teoria_subconjunto_acotado.md`) para tener el número exacto — la predicción es que debe ser bajo, cercano a generic_short/generic_long, no a axis completo.
2. **Correlación cuantitativa densidad↔Δκ en todo el panel.** Hoy la relación es cualitativa (tabla de arriba). Con 9 condiciones y densidad ya medida en la mayoría, un scatter + regresión (o al menos Spearman) densidad-vs-Δκ es el gráfico que sostiene el paper. Es el paso con mayor relación esfuerzo/impacto.
3. **n=1 por condición.** Cada celda del diseño 2×2 tiene una sola corrida. Antes de reclamar el reencuadre como resultado firme (no solo indicio fuerte), replicar al menos `axis_pec_only` y `automata_neutro` (las dos celdas que cargan el argumento).
4. **Control de longitud pendiente para `axis_pec_only`.** 1,435 tokens vs 3,945 de axis — no fue diseñado como control de longitud (`AXIS_PEC_ONLY_REPORT.md`, sección Límite). Réplica con longitud emparejada a axis cierra el hueco.
5. **Extender el drift metacognitivo (§3) al panel nuevo.** Correrlo sobre `axis_pec_only` vs `automata_neutro` predicción: el mode-switch debería sobrevivir en `axis_pec_only` (tiene identidad, no automatismo) y estar ausente en `automata_neutro` (automatismo, no identidad) — sería la confirmación cruzada de que τ/recovery_rate y drift metacognitivo miden lo mismo (Factor 2), independiente de Δκ (Factor 1).

## 5. Anclaje en literatura (ver conversación previa para detalle)

- Precedente más cercano al *método* (Δκ/W₁ sobre grafos k-NN de trayectorias): ninguno encontrado — `lsgot_3.pdf` ya reclama esto como aporte metodológico.
- Precedente más cercano al *mecanismo* densidad-de-restricción→colapso, mismo nivel geométrico (over-smoothing vía curvatura Ollivier-Ricci positiva): Ye et al., "Revisiting Over-smoothing and Over-squashing Using Ollivier-Ricci Curvature" (arXiv:2211.15779) — GNNs, no LLMs autoregresivos.
- Precedente más cercano al *efecto*, nivel de salida (no geometría interna): "The Price of Format: Diversity Collapse in LLMs" (arXiv:2505.18949, EMNLP Findings 2025).
- Contraste útil para la sección de discusión: Constraint Tax (arXiv:2606.25605, arXiv:2605.26128) y Kempner Institute, "Alignment Reduces Conceptual Diversity of Language Models".

## 6. Pasos concretos, en orden

1. Correr `analysis_restricciones.py` sobre `axis_pec_only` → cerrar vacío #1.
2. Construir la tabla completa densidad/1k tokens × Δκ para las 9 condiciones y calcular correlación → cerrar vacío #2. Este es el gráfico central del reporte.
3. Escribir un reporte corto (`REENCUADRE_DENSIDAD_VS_IDENTIDAD.md` o similar) con la estructura de la sección 7 de abajo, citando `AUTOMATA_NEUTRO_REPORT.md` y `AXIS_PEC_ONLY_REPORT.md` como evidencia primaria y este documento como bitácora del giro de hipótesis.
4. Decidir si el drift metacognitivo (vacío #5) entra en esta ronda o queda para una segunda entrega — no bloquea el reencuadre central, lo refuerza.
5. Si el objetivo es publicar/actualizar `lsgot_3.pdf`: la sección más afectada es 1.2-1.4 (pregunta de investigación e hipótesis) y toda la interpretación de H2 — el "fingerprint de identidad" pasa a describirse como confundido con densidad de restricción en el diseño original de 3 condiciones, resuelto por el panel de 9. Considerar un v0.4 o un apéndice, no descartar el paper — el hallazgo de reorganización dirección→magnitud entre regímenes de post-entrenamiento sigue siendo válido, solo la atribución causal a "identidad" necesita corrección.
6. Replicación (vacío #3) y control de longitud (vacío #4) quedan como trabajo futuro citado explícitamente como límite, no como bloqueante — igual que ya se documentó la limitación de n=1 en `Teoria_subconjunto_acotado.md`.

## 7. Estructura sugerida para el reporte/sección final

1. **Pregunta revisada**: ¿la magnitud geométrica (Δκ, W₁, dim) codifica identidad, o densidad de restricción operativa?
2. **Diseño que lo separa**: tabla 2×2 (Factor 1 × Factor 2), con las cuatro celdas ya corridas.
3. **Resultado**: densidad de restricción explica el colapso (Δκ); identidad cableada explica la recuperación (τ/recovery_rate) — son magnitudes distintas, no la misma cosa medida dos veces.
4. **Por qué `axis` original parecía confirmar identidad**: `axis` no es un caso "identidad pura" — su propia arquitectura de bloques aporta 27.0/1k de densidad, suficiente para un colapso pequeño pero significativo (Δκ=0.061) independiente de su contenido identitario.
5. **Límites**: n=1 por celda, longitud no controlada en `axis_pec_only`, réplica pendiente.
6. **Relación con literatura**: over-smoothing vía curvatura (mecanismo, en GNNs) + diversity collapse por formato (efecto, en LLMs) como anclas; ninguna cubre la combinación exacta — de ahí el aporte.
