"""
Métricas Tier 0 — Set_experimental.md

Cuatro métricas sobre trayectorias ya extraídas (embeddings.npz), sin
necesidad de nueva corrida de GPU:

  E-A  RQA               — determinism, laminarity, trapping_time
  E-C  Hurst exponent     — R/S sobre la serie ||v_t||₂ (mismo escalarizador
                            que sample_entropy() en trajectory_metrics.py)
  E-D  Participation ratio — dimensión efectiva vía espectro de covarianza
  E-H  Proyección v_identidad — persona-vector (axis − generic_long) y su
                            proyección coseno a lo largo de la trayectoria

Uso:
    from tier0_metrics import persona_vector, compute_trajectory_tier0
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Tier0Metrics:
    hurst: float = 0.5
    determinism: float = 0.0
    laminarity: float = 0.0
    trapping_time: float = 0.0
    rqa_recurrence_rate: float = 0.0
    participation_ratio: float = 0.0
    identity_projection: Optional[float] = None
    n_steps: int = 0


# ─── E-A: Recurrence Quantification Analysis ─────────────────────────────────

def _run_lengths(bool_arr: np.ndarray) -> List[int]:
    """Longitudes de corridas consecutivas de True en un array booleano 1D."""
    lengths = []
    run = 0
    for v in bool_arr:
        if v:
            run += 1
        else:
            if run > 0:
                lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)
    return lengths


def rqa_metrics(embeddings: np.ndarray, min_line_len: int = 2,
                 theiler_window: int = 3) -> Dict:
    """
    RQA sobre la trayectoria completa (matriz de recurrencia T×T, mismo
    umbral ε=10%·dist_media que recurrence_rate() en trajectory_metrics.py).

    theiler_window excluye la banda |i-j| < theiler_window alrededor de la
    diagonal principal — sin esto, la autocorrelación trivial de tokens
    consecutivos (siempre parecidos) infla determinism/laminarity con
    "líneas" que no reflejan retorno real a un estado, práctica estándar en
    CRQA (Theiler 1986).

    - determinism  : fracción de puntos recurrentes en líneas diagonales
                      (paralelas a la LOI) de longitud >= min_line_len.
    - laminarity   : fracción de puntos recurrentes en líneas verticales
                      de longitud >= min_line_len (estancamiento en un estado).
    - trapping_time: longitud media de esas líneas verticales.
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    T = len(emb)
    if T < min_line_len + theiler_window + 2:
        return {"determinism": 0.0, "laminarity": 0.0, "trapping_time": 0.0,
                "recurrence_rate": 0.0}

    diffs = emb[:, None, :] - emb[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(T, k=1)
    epsilon = 0.1 * float(np.mean(dist[iu])) if len(iu[0]) else 0.0

    R = dist < epsilon
    band = np.abs(np.subtract.outer(np.arange(T), np.arange(T))) < theiler_window
    R[band] = False

    n_recur = int(R.sum())
    if n_recur == 0:
        return {"determinism": 0.0, "laminarity": 0.0, "trapping_time": 0.0,
                "recurrence_rate": 0.0}

    diag_lengths = []
    for k in range(-(T - 1), T):
        if abs(k) < theiler_window:
            continue
        diag_lengths.extend(_run_lengths(np.diagonal(R, offset=k)))

    vert_lengths = []
    for j in range(T):
        vert_lengths.extend(_run_lengths(R[:, j]))

    diag_pts = sum(l for l in diag_lengths if l >= min_line_len)
    vert_pts = sum(l for l in vert_lengths if l >= min_line_len)
    trapping = [l for l in vert_lengths if l >= min_line_len]

    n_pairs = T * T - int(band.sum())
    return {
        "determinism": float(diag_pts / n_recur),
        "laminarity": float(vert_pts / n_recur),
        "trapping_time": float(np.mean(trapping)) if trapping else 0.0,
        "recurrence_rate": float(n_recur) / n_pairs if n_pairs else 0.0,
    }


# ─── E-C: Hurst exponent (R/S) ────────────────────────────────────────────────

def hurst_exponent(embeddings: np.ndarray, min_window: int = 8) -> float:
    """
    Exponente de Hurst (rescaled-range) sobre la serie escalar ||v_t||₂ —
    mismo escalarizador que sample_entropy() en trajectory_metrics.py.

    H > 0.5 : persistencia (memoria de largo alcance)
    H = 0.5 : paseo aleatorio (sin memoria)
    H < 0.5 : anti-persistencia (reversión a la media)
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    T = len(emb)
    x = np.linalg.norm(emb, axis=1).astype(np.float64)
    if T < min_window * 4:
        return 0.5

    window_sizes = []
    n = min_window
    while n <= T // 2:
        window_sizes.append(n)
        n = int(n * 1.5) + 1

    log_n, log_rs = [], []
    for n in window_sizes:
        n_segments = T // n
        if n_segments < 1:
            continue
        rs_vals = []
        for seg in range(n_segments):
            chunk = x[seg * n:(seg + 1) * n]
            dev = np.cumsum(chunk - chunk.mean())
            R = dev.max() - dev.min()
            S = chunk.std(ddof=0)
            if S > 1e-10:
                rs_vals.append(R / S)
        if rs_vals:
            log_n.append(math.log(n))
            log_rs.append(math.log(float(np.mean(rs_vals))))

    if len(log_n) < 3:
        return 0.5

    return float(np.polyfit(log_n, log_rs, 1)[0])


# ─── E-D: Participation ratio ─────────────────────────────────────────────────

def participation_ratio(embeddings: np.ndarray) -> float:
    """
    PR = (Σλᵢ)² / Σλᵢ² sobre el espectro de covarianza de la trayectoria —
    dimensión efectiva, cross-check del estimador de dimensión fractal ya
    usado en dimensionality_analyzer.py (no depende de vecinos k-NN).

    T << D (256 vs 5376): se usa el truco de Gram — los autovalores no nulos
    de la covarianza D×D coinciden con los de la matriz de Gram T×T, lo que
    evita diagonalizar una matriz 5376×5376 por trayectoria.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    T = len(emb)
    if T < 3:
        return 0.0
    centered = emb - emb.mean(axis=0)
    gram = centered @ centered.T / (T - 1)
    eigvals = np.clip(np.linalg.eigvalsh(gram), 0, None)
    s1 = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    if s2 < 1e-20:
        return 0.0
    return float(s1 ** 2 / s2)


# ─── E-H: Persona-vector (v_identidad) ───────────────────────────────────────

def persona_vector(emb_a: np.ndarray, lengths_a: np.ndarray,
                    emb_b: np.ndarray, lengths_b: np.ndarray) -> np.ndarray:
    """
    v_identidad = mean(tokens válidos de A) − mean(tokens válidos de B),
    normalizado a norma 1. Pensado para A=axis, B=generic_long (longitud
    emparejada, mismo dominio genérico) — mismo método que persona-vector
    research (Chen et al. 2025, Lu et al. 2026, citados en lsgot_3.pdf).
    """
    def group_mean(emb, lengths):
        emb = np.asarray(emb, dtype=np.float64)
        vecs = [emb[i, :L].mean(axis=0) for i, L in enumerate(lengths) if L > 0]
        return np.mean(vecs, axis=0)

    v = group_mean(emb_a, lengths_a) - group_mean(emb_b, lengths_b)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


def project_trajectory(embeddings: np.ndarray, v_hat: np.ndarray) -> np.ndarray:
    """Serie temporal cos(v_t, v_hat) — proyección coseno sobre v_identidad."""
    emb = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return (emb @ v_hat) / norms  # v_hat ya normalizado (||v_hat||=1)


# ─── Aggregation ─────────────────────────────────────────────────────────────

def compute_trajectory_tier0(embeddings: np.ndarray,
                              v_identidad: Optional[np.ndarray] = None) -> Tier0Metrics:
    """Computa las cuatro métricas Tier 0 para una trayectoria."""
    T = len(embeddings)
    rqa = rqa_metrics(embeddings)
    h = hurst_exponent(embeddings)
    pr = participation_ratio(embeddings)
    proj = None
    if v_identidad is not None:
        proj = float(np.mean(project_trajectory(embeddings, v_identidad)))

    return Tier0Metrics(
        hurst=h,
        determinism=rqa["determinism"],
        laminarity=rqa["laminarity"],
        trapping_time=rqa["trapping_time"],
        rqa_recurrence_rate=rqa["recurrence_rate"],
        participation_ratio=pr,
        identity_projection=proj,
        n_steps=T,
    )


def _agg(vals: List[Optional[float]]) -> Dict:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
