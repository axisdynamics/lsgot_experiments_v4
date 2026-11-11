#!/usr/bin/env python3
"""
Deriva las cantidades que el paper calcula pero que ningún JSON guardaba.

El criterio del paquete es que **todo número del paper salga de un archivo**. Las
medias, las d y las p ya salían de `data/`; lo que no salía eran las cantidades
derivadas: diferencias entre columnas, cocientes, porcentajes y correlaciones de
rango. Este script las calcula desde los JSON que el paquete ya entrega y las
escribe en `data/derived/derived_stats.json`, para que el self-audit pueda
verificarlas como verifica todo lo demás.

No lee tensores, no necesita GPU y usa solo la biblioteca estándar — ni numpy.

Entradas (todas en el paquete):
  data/t0_native/e1_vhat_nativa_t0.json   panel en v̂ y en v̂_t0
  data/panel_results/EH2_results.json     ráfagas y frac_pos del panel
  data/panel_results/EL_results.json      t=0 / t_rest del panel

Uso:
    python3 scripts/derived/compute_derived.py            # escribe el JSON
    python3 scripts/derived/compute_derived.py --print    # además lo imprime
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "derived" / "derived_stats.json"

# Las 7 condiciones del panel, que son las que el paper usa para los rangos y
# las correlaciones de rango de §4.7.
PANEL_GROUP = "panel"
READOUTS = ["t0", "t_rest", "frac_pos", "burst_mean"]


def spearman(a, b):
    """ρ de Spearman sin scipy. Sin empates en estos datos (se verifica)."""
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r
    assert len(set(a)) == len(a) and len(set(b)) == len(b), "hay empates: usar ρ con corrección"
    ra, rb = rank(a), rank(b)
    n = len(a)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def load(rel):
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


def derive_directions(panel):
    """§4.7 · por read-out: ρ entre v̂ y v̂_t0, rango en cada una y el cociente."""
    conds = [c for c, v in panel["v_canon"].items() if v.get("group") == PANEL_GROUP]
    out = {"conditions": conds, "n_conditions": len(conds), "readouts": {}}
    for ro in READOUTS:
        a = [panel["v_canon"][c][ro]["mean"] for c in conds]
        b = [panel["v_t0"][c][ro]["mean"] for c in conds]
        ra, rb = max(a) - min(a), max(b) - min(b)
        out["readouts"][ro] = {
            "spearman_rho_vhat_vs_vt0": spearman(a, b),
            "range_on_vhat": ra,
            "range_on_vt0": rb,
            "ratio_vt0_over_vhat": rb / ra,
            # el paper lo enuncia como "N× wider" o "N× narrower"
            "direction": "wider" if rb > ra else "narrower",
            "factor_as_stated": (rb / ra) if rb > ra else (ra / rb),
            "per_condition": {c: {"v_hat": a[i], "v_t0": b[i]}
                              for i, c in enumerate(conds)},
        }
    return out


def derive_panel_deltas(panel):
    """§4.1 · la columna `t0 → t_rest`: cuánto sube la proyección al generar."""
    out = {}
    for c, v in panel["v_canon"].items():
        out[c] = {
            "t0": v["t0"]["mean"],
            "t_rest": v["t_rest"]["mean"],
            "delta_t0_to_t_rest": v["t_rest"]["mean"] - v["t0"]["mean"],
            "group": v.get("group"),
        }
    return out


def derive_burst_ratios(eh2):
    """§4.4 · max/mean de ráfaga, y frac_pos en porcentaje."""
    series = eh2["series_metrics"]["con_t0"]
    out = {}
    for cond, m in series.items():
        bm = m.get("mean_burst_len", {}).get("mean")
        bx = m.get("max_burst_len", {}).get("mean")
        fp = m.get("frac_positive", {}).get("mean")
        row = {}
        if bm and bx:
            row["burst_mean"] = bm
            row["burst_max"] = bx
            row["max_over_mean"] = bx / bm
        if fp is not None:
            row["frac_pos"] = fp
            row["frac_pos_percent"] = 100.0 * fp
        if row:
            out[cond] = row
    return out


def derive_berg_pairs(berg):
    """§4.2 · dos contrastes de Berg cuyo `d` el paper cita y cuyo `p` no quedó
    almacenado en `key_pairs`. El `d` se recalcula acá desde `el_table` con la
    fórmula declarada en §3.5 — SD agrupada RMS, `√((s_A² + s_B²)/2)` — y se
    deja asentado que el `p` de estos dos pares no existe en el paquete."""
    t = berg["el_table"]
    pairs = [
        ("berg_experimental", "berg_conceptual_control"),
        ("berg_history_control", "vanilla"),
    ]
    out = {"_note": "`d` recalculado desde el_table; `p` no almacenado para estos pares",
           "n_permutations_of_source": berg.get("n_permutations"),
           "pairs": {}}
    for a, b in pairs:
        # el_table guarda std POBLACIONAL (ddof=0); §3.5 define d sobre la SD
        # muestral, así que se convierte con √(n/(n−1)) antes de agrupar.
        ra, rb = t[a]["t0"], t[b]["t0"]
        sa = ra["std"] * (ra["n"] / (ra["n"] - 1)) ** 0.5
        sb = rb["std"] * (rb["n"] / (rb["n"] - 1)) ** 0.5
        pooled = ((sa ** 2 + sb ** 2) / 2) ** 0.5
        out["pairs"][f"{a}_vs_{b}"] = {
            "mean_a": ra["mean"], "mean_b": rb["mean"],
            "n_a": ra["n"], "n_b": rb["n"],
            "sd_a_sample": sa, "sd_b_sample": sb,
            "pooled_sd_rms": pooled,
            "cohens_d": (ra["mean"] - rb["mean"]) / pooled,
            "p_value": None,
            "p_status": "no almacenado en el paquete — ver limitación de trazabilidad",
        }
    return out


def derive_axis_pec(e31):
    """§4.3 · los cuatro brazos del control pre-registrado, y la partición de la
    distancia `axis_pec_only` → `automata_neutro` en sus tres componentes."""
    t0 = {k: v["mean"] for k, v in e31["t0_by_arm"]["v_canon"].items()}
    only, none = t0["axis_pec_only"], t0["axis_pec_none"]
    neutro, auto = t0["axis_pec_neutro"], t0["automata_neutro"]
    total = only - auto
    parts = {
        "declared_identity_blocks": neutro - auto,
        "self_description_no_step": none - neutro,
        "four_wired_steps": only - none,
    }
    return {
        "t0_by_arm": t0,
        "total_distance_only_minus_automata": total,
        "decomposition": {k: {"delta_t0": v, "fraction_of_total": v / total,
                              "percent_of_total": 100.0 * v / total}
                          for k, v in parts.items()},
        "sums_to": sum(parts.values()) / total,
        "_note": "`axis_pec_neutro` es post-hoc: nació tras ver el resultado del "
                 "brazo de base firmado, así que los dos últimos componentes no "
                 "están pre-registrados y la regla de corte no se les aplica.",
    }


def derive_floors():
    """Los pisos de los dos estimadores de permutación (§3.5 y §4.7)."""
    return {
        "panel": {
            "estimator": "k/n (sin add-one, una cola en la dirección observada)",
            "n_permutations": 1000,
            "smallest_nonzero_p": 1.0 / 1000,
            "strongest_claim_from_zero": "p<0.001",
            "source": "scripts/shared/statistical_tests.py",
        },
        "t0_native": {
            "estimator": "(1+k)/(1+n) (add-one, dos colas sobre |diferencia de medias|)",
            "n_permutations": 10000,
            "floor_attained": 1.0 / 10001,
            "strongest_claim": "p<0.0001",
            "source": "scripts/t0_native/e1_vhat_nativa_t0.py",
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Deriva las cantidades calculadas del paper")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="imprime el resultado además de escribirlo")
    args = ap.parse_args()

    e1 = load("data/t0_native/e1_vhat_nativa_t0.json")
    eh2 = load("data/panel_results/EH2_results.json")
    berg = load("data/panel_results/berg_controls_t0_results.json")
    e31 = load("data/axis_pec/e31_r.json")

    result = {
        "_about": "Cantidades que el paper deriva de otros archivos del paquete. "
                  "Regenerar con scripts/derived/compute_derived.py.",
        "_inputs": [
            "data/t0_native/e1_vhat_nativa_t0.json",
            "data/panel_results/EH2_results.json",
            "data/panel_results/berg_controls_t0_results.json",
            "data/axis_pec/e31_r.json",
        ],
        "direction_comparison": derive_directions(e1["panel"]),
        "panel_t0_to_trest": derive_panel_deltas(e1["panel"]),
        "burst_ratios_and_percentages": derive_burst_ratios(eh2),
        "berg_pairs_not_in_key_pairs": derive_berg_pairs(berg),
        "axis_pec_control": derive_axis_pec(e31),
        "permutation_floors": derive_floors(),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"escrito: {OUT.relative_to(REPO)}")

    print("\n§4.2 · los dos pares de Berg con `d` citado y `p` ausente:")
    for k, v in result["berg_pairs_not_in_key_pairs"]["pairs"].items():
        print(f"  {k:48s} d={v['cohens_d']:+.4f}  p={v['p_status']}")

    ap = result["axis_pec_control"]
    print("\n§4.3 · partición de la distancia axis_pec_only → automata_neutro:")
    for k, v in ap["decomposition"].items():
        print(f"  {k:28s} Δ={v['delta_t0']:+.4f}  {100*v['fraction_of_total']:5.1f}%")
    print(f"  {'suma':28s}            {100*ap['sums_to']:5.1f}%")

    dc = result["direction_comparison"]["readouts"]
    print("\n§4.7 · ρ y rangos sobre las 7 condiciones del panel:")
    for ro in READOUTS:
        r = dc[ro]
        print(f"  {ro:11s} ρ={r['spearman_rho_vhat_vs_vt0']:+.4f}  "
              f"v̂={r['range_on_vhat']:.4f}  v̂_t0={r['range_on_vt0']:.4f}  "
              f"{r['factor_as_stated']:.2f}× {r['direction']}")

    if args.show:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
