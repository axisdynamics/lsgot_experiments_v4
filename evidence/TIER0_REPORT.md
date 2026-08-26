# Tier 0 — Set_experimental.md (E-A, E-C, E-D, E-H)

Datos: `results_local/sia_extended_v5/*_embeddings.npz`. v_identidad = mean(axis) − mean(generic_long).

## Tabla comparativa por grupo (media ± std)

| grupo | hurst | determinism | laminarity | trapping_time | PR | proj(v_identidad) |
|---|---|---|---|---|---|---|
| axis | 0.6704±0.0743 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 17.5435±1.2963 | 0.2048±0.0297 |
| generic_long | 0.6989±0.0504 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 20.5458±0.8884 | 0.0666±0.0288 |
| generic_short | 0.7037±0.0554 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 20.5304±1.3030 | 0.0478±0.0305 |
| vanilla | 0.6703±0.0578 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 19.5380±1.2851 | 0.0982±0.0306 |
| axis_short | 0.6574±0.0657 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 17.6862±1.5691 | 0.2017±0.0248 |
| chileatiende | 0.7052±0.0663 | 0.5574±0.2107 | 0.0000±0.0000 | 0.0000±0.0000 | 16.4287±4.1949 | -0.0894±0.0347 |
| automata_neutro | 0.7351±0.1573 | 0.4183±0.4260 | 0.0000±0.0000 | 0.0000±0.0000 | 14.7220±3.4810 | 0.0060±0.0432 |
| chileatiende_sia | 0.7948±0.0808 | 0.5558±0.2992 | 0.0000±0.0000 | 0.0000±0.0000 | 14.2727±3.6253 | -0.0538±0.0485 |
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

### chileatiende_vs_vanilla

| métrica | Δ(mean) | p | cohen_d | interpretación |
|---|---|---|---|---|
| hurst | +0.0349 | 0.0430 | +0.547 | medium |
| determinism | +0.5574 | 0.0000 | +3.647 | large |
| laminarity | +0.0000 | 1.0000 | +0.000 | negligible |
| trapping_time | +0.0000 | 1.0000 | +0.000 | negligible |
| rqa_recurrence_rate | +0.0009 | 0.0000 | +3.169 | large |
| participation_ratio | -3.1092 | 0.0010 | -0.977 | large |
| identity_projection | -0.1876 | 0.0000 | -5.595 | large |

### axis_vs_chileatiende

| métrica | Δ(mean) | p | cohen_d | interpretación |
|---|---|---|---|---|
| hurst | -0.0347 | 0.0650 | -0.481 | small |
| determinism | -0.5574 | 0.0000 | -3.647 | large |
| laminarity | +0.0000 | 1.0000 | +0.000 | negligible |
| trapping_time | +0.0000 | 1.0000 | +0.000 | negligible |
| rqa_recurrence_rate | -0.0009 | 0.0000 | -3.163 | large |
| participation_ratio | +1.1148 | 0.1550 | +0.350 | small |
| identity_projection | +0.2943 | 0.0000 | +8.889 | large |

### chileatiende_vs_automata_neutro

| métrica | Δ(mean) | p | cohen_d | interpretación |
|---|---|---|---|---|
| hurst | -0.0299 | 0.2140 | -0.242 | small |
| determinism | +0.1391 | 0.1060 | +0.404 | small |
| laminarity | +0.0000 | 1.0000 | +0.000 | negligible |
| trapping_time | +0.0000 | 1.0000 | +0.000 | negligible |
| rqa_recurrence_rate | -0.0038 | 0.0290 | -0.634 | medium |
| participation_ratio | +1.7067 | 0.0930 | +0.432 | small |
| identity_projection | -0.0955 | 0.0000 | -2.375 | large |

### chileatiende_sia_vs_vanilla

| métrica | Δ(mean) | p | cohen_d | interpretación |
|---|---|---|---|---|
| hurst | +0.1245 | 0.0000 | +1.727 | large |
| determinism | +0.5558 | 0.0000 | +2.561 | large |
| laminarity | +0.0000 | 1.0000 | +0.000 | negligible |
| trapping_time | +0.0000 | 1.0000 | +0.000 | negligible |
| rqa_recurrence_rate | +0.0031 | 0.0000 | +0.587 | medium |
| participation_ratio | -5.2652 | 0.0000 | -1.887 | large |
| identity_projection | -0.1519 | 0.0000 | -3.653 | large |

## Nota — double dissociation identity_projection (2026-08-26)

`identity_projection` separa por **contenido identitario**, no por
restricción/automatismo: `axis_pec_only` (identidad sin automatismo) iguala
a `axis` (p=0.229, no difieren), mientras que `chileatiende`/`chileatiende_sia`
(automatismo sin identidad) proyectan **por debajo de vanilla**, no por
encima (d=−5.60 y −3.65 vs vanilla; d=+8.89 en axis_vs_chileatiende, el
efecto más grande del panel). Descarta la lectura alternativa de que
`identity_projection` fuera un artefacto genérico de "cualquier system
prompt raro" — el signo se invierte según si hay contenido identitario, no
según si hay restricción.

## Bloqueado — ya resuelto

~~E-E (Fréchet) y la ventana de perturbación de E-H~~ — cerrados
2026-08-22/26, ver `EE_EH_WINDOW_REPORT.md` (10/10 grupos).
