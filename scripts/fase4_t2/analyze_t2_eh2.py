#!/usr/bin/env python3
"""
T2 — tercera señal de identidad (E-H2: dinámica temporal de v̂) para
witness_soul_md, axis_pec_only_v2 y automata_neutro_v2, mismo método que
scripts/fase0/analyze_EH2_serie_temporal.py.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "fase0"))
from tier0_metrics import project_trajectory  # noqa: E402
from analyze_EH2_serie_temporal import series_metrics, agg  # noqa: E402

T2_DIR = Path(
    "/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/"
    "gemma4_31b_combined/perturbation/results_t2/trajectories"
)
V_HAT_PATH = Path(__file__).parent.parent / "perturbation" / "v_identidad.npy"
GROUPS = ["axis_pec_only_v2", "automata_neutro_v2", "witness_soul_md"]


def load(name):
    d = np.load(T2_DIR / f"{name}_baseline_embeddings.npz", allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["lengths"].astype(int)


def main():
    v_hat = np.load(V_HAT_PATH)
    print(f"{'grupo':22s} {'autocorr':>9s} {'frac(p>0)':>10s} {'ráfaga_media':>13s} {'ráfaga_máx':>11s} {'mean_p':>8s}")
    for g in GROUPS:
        emb, lengths = load(g)
        rows = []
        for i, L in enumerate(lengths):
            if L < 8:
                continue
            p = project_trajectory(emb[i, :L], v_hat)
            m = series_metrics(p)
            if m is not None:
                rows.append(m)
        autocorr = agg(rows, "autocorr_lag1")[0]
        frac = agg(rows, "frac_positive")[0]
        mburst = agg(rows, "mean_burst_len")[0]
        xburst = agg(rows, "max_burst_len")[0]
        meanp = agg(rows, "mean_p")[0]
        print(f"{g:22s} {autocorr['mean']:9.3f} {frac['mean']:10.3f} "
              f"{mburst['mean']:13.2f} {xburst['mean']:11.1f} {meanp['mean']:+8.3f}")


if __name__ == "__main__":
    main()
