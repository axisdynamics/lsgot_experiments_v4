# Corrección: efecto de restricción (v̂, PR) recalculado sin chileatiende-family

**Fecha:** 2026-08-28
**Origen:** hipótesis del usuario tras `CHILEATIENDE_MARKUP_CONFOUND_REPORT.md`
— "descartando el ruido de chileatiende, ¿tenemos huellas de identidad y no
de restricción?"
**Método:** recalculado con `LSGOT_v4/scripts/shared/tier0_metrics.py` +
`statistical_tests.py` sobre `results_local/sia_extended_v5` (capa final),
excluyendo por completo chileatiende/chileatiende_sia/chileatiende_sia_v2.

## Resultado: la hipótesis NO se confirma — restricción tiene huella propia

`automata_neutro` (0% markup, restricción "limpia") sigue mostrando un
efecto negativo grande y significativo respecto a vanilla, independiente de
chileatiende:

| Comparación (vs vanilla) | d (proyección v̂) | p | d (participation ratio) | p |
|---|---|---|---|---|
| `automata_neutro` | **−2.399** | <0.001 | **−1.789** | <0.001 |
| `axis` | +3.453 | <0.001 | −1.506 | <0.001 |
| `axis_pec_only` | +3.791 | <0.001 | −1.181 | <0.001 |
| `axis_short` | +3.629 | <0.001 | −1.259 | <0.001 |

Restricción (automata_neutro) y identidad (axis-family) tienen huellas
**independientes, de magnitud comparable, en direcciones opuestas** sobre
v̂ — restricción incluso colapsa PR **más** que identidad. No hay evidencia
de que la señal de restricción dependiera del confound de chileatiende:
automata_neutro nunca tuvo ese confound (0% markup) y su efecto es robusto
por sí solo.

## Lo que SÍ cambia: la cifra insignia del paper estaba inflada

| Comparación | d original (paper, con chileatiende) | d corregido (con automata_neutro) |
|---|---|---|
| axis_pec_only vs [restricción pura] | **+8.89** (vs chileatiende) | **+5.52** (vs automata_neutro) |
| axis vs [restricción pura] | — (no reportado directamente) | +5.23 |

La comparación `axis_pec_only vs chileatiende` (d=8.89, "el mayor efecto
del panel" según el paper) usaba una condición contaminada en uno de sus
dos términos. La comparación limpia equivalente
(`axis_pec_only vs automata_neutro`) da d=+5.52 — sigue siendo un efecto
muy grande (la referencia de Cohen para "grande" es d>0.8), pero ~38% menor
que el número que cita el paper. **Recomendación: usar automata_neutro, no
chileatiende, como la condición de referencia de "restricción pura" en
cualquier cifra que el paper presente como su hallazgo insignia de
disociación identidad/restricción.**

## Implicación para el paper

1. El reencuadre central (`lsgot_4.md`: identidad = dirección estática,
   restricción = efecto dinámico/estructural, ambos factores reales e
   independientes) **se sostiene** — de hecho se fortalece, porque ahora
   tenemos evidencia de restricción "limpia" (sin confound de formato) que
   confirma el efecto de forma independiente de chileatiende.
2. **Actualizar la cifra d=8.89** en cualquier sección del paper que la
   cite como "el mayor efecto del panel" — reemplazar por d=5.52
   (axis_pec_only vs automata_neutro) o recalcular explícitamente
   excluyendo chileatiende-family de esa comparación específica.
3. Esto no invalida el uso de chileatiende-family para otras preguntas
   (p.ej. contenido/dominio específico de pensiones), pero descalifica su
   uso como ancla cuantitativa de "restricción pura" en cualquier métrica
   de trayectoria/geometría — ver `CHILEATIENDE_MARKUP_CONFOUND_REPORT.md`
   §5 para la lista de condiciones objetivo revisadas.
