# Tier 0 — Set_experimental.md (E-A, E-C, E-D, E-H)

> ⚠️ **Sanitizado 2026-08-31:** este reporte incluía originalmente dos
> condiciones de automatismo real que forzaban un wrapper HTML literal
> repetido en el 100% de sus respuestas (un confound de formato, no de
> restricción). Se eliminaron por completo de la tabla y de las
> comparaciones; ninguna cifra de las condiciones limpias cambia (nunca
> dependieron de ellas).

Datos: `results_local/sia_extended_v5/*_embeddings.npz`. v_identidad = mean(axis) − mean(generic_long).

## Tabla comparativa por grupo (media ± std)

| grupo | hurst | determinism | laminarity | trapping_time | PR | proj(v_identidad) |
|---|---|---|---|---|---|---|
| axis | 0.6704±0.0743 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 17.5435±1.2963 | 0.2048±0.0297 |
| generic_long | 0.6989±0.0504 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 20.5458±0.8884 | 0.0666±0.0288 |
| generic_short | 0.7037±0.0554 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 20.5304±1.3030 | 0.0478±0.0305 |
| vanilla | 0.6703±0.0578 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 19.5380±1.2851 | 0.0982±0.0306 |
| axis_short | 0.6574±0.0657 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 17.6862±1.5691 | 0.2017±0.0248 |
| automata_neutro | 0.7351±0.1573 | 0.4183±0.4260 | 0.0000±0.0000 | 0.0000±0.0000 | 14.7220±3.4810 | 0.0060±0.0432 |
| axis_pec_only | 0.6495±0.0801 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 18.0048±1.2449 | 0.2122±0.0280 |

## Significancia (permutation test, mean_difference, n_perm=1000)

### axis_vs_vanilla

| métrica | Δ(mean) | p | cohen_d | interpretación |
|---|---|---|---|---|
| hurst | +0.0002 | 0.4750 | +0.002 | negligible |
| determinism | +0.0000 | 1.0000 | +0.000 | negligible |
| laminarity | +0.0000 | 1.0000 | +0.000 | negligible |
| trapping_time | +0.0000 | 1.0000 | +0.000 | negligible |
| rqa_recurrence_rate | +0.0000 | 0.5030 | +0.316 | small |
| participation_ratio | -1.9945 | 0.0000 | -1.506 | large |
| identity_projection | +0.1067 | 0.0000 | +3.453 | large |

### axis_pec_only_vs_vanilla

| métrica | Δ(mean) | p | cohen_d | interpretación |
|---|---|---|---|---|
| hurst | -0.0208 | 0.1740 | -0.290 | small |
| determinism | +0.0000 | 1.0000 | +0.000 | negligible |
| laminarity | +0.0000 | 1.0000 | +0.000 | negligible |
| trapping_time | +0.0000 | 1.0000 | +0.000 | negligible |
| rqa_recurrence_rate | +0.0000 | 0.4930 | +0.316 | small |
| participation_ratio | -1.5331 | 0.0000 | -1.181 | large |
| identity_projection | +0.1140 | 0.0000 | +3.791 | large |

### automata_neutro_vs_vanilla

| métrica | Δ(mean) | p | cohen_d | interpretación |
|---|---|---|---|---|
| hurst | +0.0649 | 0.0510 | +0.534 | medium |
| determinism | +0.4183 | 0.0000 | +1.353 | large |
| laminarity | +0.0000 | 1.0000 | +0.000 | negligible |
| trapping_time | +0.0000 | 1.0000 | +0.000 | negligible |
| rqa_recurrence_rate | +0.0047 | 0.0000 | +0.786 | medium |
| participation_ratio | -4.8160 | 0.0000 | -1.789 | large |
| identity_projection | -0.0921 | 0.0000 | -2.399 | large |

### axis_vs_automata_neutro

| métrica | Δ(mean) | p | cohen_d | interpretación |
|---|---|---|---|---|
| hurst | -0.0647 | 0.0510 | -0.513 | medium |
| determinism | -0.4183 | 0.0000 | -1.353 | large |
| laminarity | +0.0000 | 1.0000 | +0.000 | negligible |
| trapping_time | +0.0000 | 1.0000 | +0.000 | negligible |
| rqa_recurrence_rate | -0.0047 | 0.0000 | -0.785 | medium |
| participation_ratio | +2.8215 | 0.0010 | +1.047 | large |
| identity_projection | +0.1988 | 0.0000 | +5.226 | large |

### axis_vs_axis_pec_only

| métrica | Δ(mean) | p | cohen_d | interpretación |
|---|---|---|---|---|
| hurst | +0.0210 | 0.2130 | +0.265 | small |
| determinism | +0.0000 | 1.0000 | +0.000 | negligible |
| laminarity | +0.0000 | 1.0000 | +0.000 | negligible |
| trapping_time | +0.0000 | 1.0000 | +0.000 | negligible |
| rqa_recurrence_rate | +0.0000 | 0.7660 | +0.000 | negligible |
| participation_ratio | -0.4614 | 0.1390 | -0.354 | small |
| identity_projection | -0.0074 | 0.2290 | -0.248 | small |

## Nota — double dissociation identity_projection (2026-08-26)

`identity_projection` separa por **contenido identitario**, no por
restricción/automatismo: `axis_pec_only` (identidad sin automatismo) iguala
a `axis` (p=0.229, no difieren), mientras que `automata_neutro`
(automatismo sin identidad) proyecta **por debajo de vanilla**, no por
encima (d=−2.40 vs vanilla; d=+5.23 en axis_vs_automata_neutro, el
efecto más grande del panel). Descarta la lectura alternativa de que
`identity_projection` fuera un artefacto genérico de "cualquier system
prompt raro" — el signo se invierte según si hay contenido identitario, no
según si hay restricción.

## Bloqueado — ya resuelto

~~E-E (Fréchet) y la ventana de perturbación de E-H~~ — cerrados
2026-08-22/26, ver `EE_EH_WINDOW_REPORT.md` (10/10 grupos).
