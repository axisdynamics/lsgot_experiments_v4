#!/usr/bin/env python3
"""E1 (AGENDA_EXPERIMENTAL.md, Tier 0) — Dirección nativa de t=0.

Reconstruye v̂_t0 = unit(mean(axis, t=0) − mean(generic_long, t=0)) y re-mide el
panel completo (t=0, t_rest, frac_pos, ráfagas) lado a lado con la v̂ canónica
(`v_identidad.npy`, media de trayectoria libre). Calcula además la matriz de
cosenos entre v̂, v̂_t0, v̂_berg, v̂_evil y v̂_hist.

Solo CPU, sobre los .npz ya extraídos. Convenciones idénticas a
`LSGOT_4_FINAL/scripts/shared/tier0_metrics.py` (proyección coseno) y a
`EH2_REPORT.md` §2 (frac_pos y ráfagas = rachas consecutivas con p(t)>0).
"""
import json, numpy as np
from pathlib import Path

# Raíz de las trayectorias (NPZ). NO están en el paquete: son ~23 GB y el
# README lo declara. Se configura con LSGOT_TRAJ_ROOT; el default es la ruta de
# la máquina donde se corrieron.
import os
RB = Path(os.environ.get(
    'LSGOT_TRAJ_ROOT',
    '/home/plaxius/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments'
    '/gemma4_31b_combined'))
if not RB.exists():
    raise SystemExit(
        f'No encuentro las trayectorias en {RB}.\n'
        'Son ~23 GB y no se redistribuyen: reproducílas con '
        'scripts/egrav/RUNBOOK_EGRAV.md y después\n'
        '  export LSGOT_TRAJ_ROOT=/ruta/a/las/trayectorias')
HERE = Path(__file__).parent
RES = HERE / 'results'
NPERM, NBOOT, SEED = 10000, 5000, 0

# ── fuentes ──────────────────────────────────────────────────────────────────
V5 = RB / 'results_local/sia_extended_v5'
T2 = RB / 'perturbation/results_t2/trajectories'
BF = RB / 'results_local/berg_free'
ON = RB / 'results_local/sia_berg/ontologica'
EA = RB / 'results_local/expA_t0'

# nombre lógico -> (archivo, grupo)
TRAJ = {
    'axis':             (V5 / 'axis_embeddings.npz',              'panel'),
    'axis_short':       (V5 / 'axis_short_embeddings.npz',        'panel'),
    'axis_pec_only':    (V5 / 'axis_pec_only_embeddings.npz',     'panel'),
    'automata_neutro':  (V5 / 'automata_neutro_embeddings.npz',   'panel'),
    'vanilla':          (V5 / 'vanilla_embeddings.npz',           'panel'),
    'generic_long':     (V5 / 'generic_long_embeddings.npz',      'panel'),
    'generic_short':    (V5 / 'generic_short_embeddings.npz',     'panel'),
    'axis_pec_only_v2':   (T2 / 'axis_pec_only_v2_baseline_embeddings.npz',      't2'),
    'automata_neutro_v2': (T2 / 'automata_neutro_v2_baseline_embeddings.npz',    't2'),
    'witness_soul_md':    (T2 / 'witness_soul_md_baseline_embeddings.npz',       't2'),
    'soul_jarvis':        (T2 / 'soul_jarvis_baseline_embeddings.npz',           't2'),
    'soul_elena_financial':  (T2 / 'soul_elena_financial_baseline_embeddings.npz',  't2'),
    'soul_solidity_auditor': (T2 / 'soul_solidity_auditor_baseline_embeddings.npz', 't2'),
    'axis_berg':        (ON / 'axis_berg_embeddings.npz',        'berg'),
    'axis_bergwitness': (ON / 'axis_bergwitness_embeddings.npz', 'berg'),
    'axis_nowit':       (ON / 'axis_nowit_embeddings.npz',       'berg'),
    'berg_experimental':      (BF / 'berg_experimental_embeddings.npz',      'berg'),
    'berg_history_control':   (BF / 'berg_history_control_embeddings.npz',   'berg'),
    'berg_conceptual_control':(BF / 'berg_conceptual_control_embeddings.npz','berg'),
}
PANEL7 = ['axis', 'axis_short', 'axis_pec_only', 'automata_neutro',
          'vanilla', 'generic_long', 'generic_short']

_cache = {}
def traj(name):
    """(embeddings float64 [n,T,D], lengths) de la trayectoria libre."""
    if name not in _cache:
        z = np.load(TRAJ[name][0])
        _cache[name] = (z['embeddings'].astype(np.float64), z['lengths'])
    return _cache[name]

def t0_raw(name):
    """Estado t=0 crudo [n,D]. Para el panel/T2 es el primer token de la
    trayectoria; para la familia Berg se usa la clave t0_embeddings, que es la
    que usó `analyze_shared_direction.py` (misma convención)."""
    z = np.load(TRAJ[name][0])
    k = 't0_embeddings' if 't0_embeddings' in z.files else None
    return (z[k] if k else z['embeddings'][:, 0, :]).astype(np.float64)

def expA_t0(name):
    return np.load(EA / f'{name}_t0.npz')['embeddings'].astype(np.float64)

unit = lambda v: v / np.linalg.norm(v)

# ── métricas por prompt ──────────────────────────────────────────────────────
def runs_true(b):
    """Longitudes de rachas consecutivas de True."""
    out, run = [], 0
    for v in b:
        if v: run += 1
        elif run: out.append(run); run = 0
    if run: out.append(run)
    return out

def per_prompt(name, v):
    """Por prompt: t0, t_rest, frac_pos, ráfaga media/máx, proy. media (traj)."""
    emb, lens = traj(name)
    n = emb.shape[0]
    o = {k: np.full(n, np.nan) for k in
         ('t0', 't_rest', 'frac_pos', 'burst_mean', 'burst_max', 'traj')}
    for i in range(n):
        L = int(lens[i])
        h = emb[i, :L]
        nr = np.linalg.norm(h, axis=1); nr[nr < 1e-10] = 1.0
        p = (h @ v) / nr                       # serie coseno p(t)
        o['t0'][i] = p[0]
        o['t_rest'][i] = p[1:].mean() if L > 1 else p.mean()
        o['traj'][i] = p.mean()
        pos = p > 0
        o['frac_pos'][i] = pos.mean()
        r = runs_true(pos)
        o['burst_mean'][i] = np.mean(r) if r else 0.0
        o['burst_max'][i] = max(r) if r else 0.0
    return o

# ── estadística ──────────────────────────────────────────────────────────────
def cohen_d(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return float((a.mean()-b.mean())/s) if s > 1e-15 else 0.0

def perm_p(a, b, n=NPERM, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = abs(a.mean()-b.mean()); c = np.concatenate([a, b]); na = len(a); cnt = 0
    for _ in range(n):
        p = rng.permutation(c)
        if abs(p[:na].mean()-p[na:].mean()) >= obs: cnt += 1
    return (cnt+1)/(n+1)

def boot_d_ci(A, B, nb=NBOOT, seed=SEED+1):
    """IC95 del d remuestreando PROMPTS (0..n-1) con reemplazo, conjunto A/B."""
    rng = np.random.default_rng(seed); ds = []
    for _ in range(nb):
        ids = rng.integers(0, len(A), len(A))
        a, b = A[ids], B[ids]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) > 2 and len(b) > 2: ds.append(cohen_d(a, b))
    return [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]

def contrast(A, B):
    a, b = A[~np.isnan(A)], B[~np.isnan(B)]
    return dict(mean_a=float(a.mean()), sd_a=float(a.std(ddof=1)),
                mean_b=float(b.mean()), sd_b=float(b.std(ddof=1)),
                n_a=len(a), n_b=len(b), d=cohen_d(a, b),
                p=perm_p(a, b), d_ci95=boot_d_ci(A, B))

# ── direcciones ──────────────────────────────────────────────────────────────
def build_dirs():
    V = unit(np.load(HERE.parent / 'LSGOT_v4/scripts/perturbation/v_identidad.npy')
             .astype(np.float64))
    gl_t0 = t0_raw('generic_long')
    d = {
        'v_canon':  V,
        'v_t0':     unit(t0_raw('axis').mean(0) - gl_t0.mean(0)),
        'v_t0_pec': unit(t0_raw('axis_pec_only').mean(0) - gl_t0.mean(0)),
        'v_berg':   unit(t0_raw('berg_experimental').mean(0) - gl_t0.mean(0)),
        'v_evil':   unit(expA_t0('evil').mean(0) - gl_t0.mean(0)),
        'v_hist':   unit(t0_raw('berg_history_control').mean(0) - gl_t0.mean(0)),
    }
    # control de reproducción: v̂ canónica reconstruida desde las trayectorias
    def gmean(name):
        emb, lens = traj(name)
        return np.mean([emb[i, :int(L)].mean(0) for i, L in enumerate(lens)], axis=0)
    d['v_canon_recon'] = unit(gmean('axis') - gmean('generic_long'))
    return d

KEY_PAIRS = [
    ('axis_pec_only', 'automata_neutro'), ('axis', 'automata_neutro'),
    ('axis_pec_only', 'vanilla'), ('axis', 'vanilla'),
    ('automata_neutro', 'vanilla'), ('axis', 'axis_pec_only'),
    ('axis_pec_only', 'generic_long'), ('automata_neutro', 'generic_long'),
    ('axis_pec_only_v2', 'automata_neutro_v2'),
    ('witness_soul_md', 'automata_neutro_v2'),
    ('soul_elena_financial', 'generic_long'),
    ('soul_jarvis', 'generic_long'),
    ('berg_experimental', 'berg_history_control'),
]

def main():
    dirs = build_dirs()
    out = {'experiment': 'E1 — dirección nativa de t=0 (AGENDA_EXPERIMENTAL.md Tier 0)',
           'model': 'Gemma-4-31B-it, capa final', 'n_prompts': 20,
           'n_permutations': NPERM, 'n_bootstrap': NBOOT, 'seed': SEED,
           'cos_canon_vs_recon': float(dirs['v_canon'] @ dirs['v_canon_recon'])}
    print(f"control de reproducción: cos(v_canon, v_canon_recon) = "
          f"{out['cos_canon_vs_recon']:.6f}", flush=True)

    names = [k for k in dirs if k != 'v_canon_recon']
    out['cosines'] = {a: {b: float(dirs[a] @ dirs[b]) for b in names} for a in names}

    out['panel'], out['contrasts'] = {}, {}
    for vn in ('v_canon', 'v_t0', 'v_t0_pec'):
        v = dirs[vn]
        pp = {c: per_prompt(c, v) for c in TRAJ}
        out['panel'][vn] = {
            c: {m: {'mean': float(np.nanmean(x)), 'sd': float(np.nanstd(x, ddof=1))}
                for m, x in d.items()} | {'group': TRAJ[c][1]}
            for c, d in pp.items()}
        out['contrasts'][vn] = {
            f'{a}_vs_{b}': {m: contrast(pp[a][m], pp[b][m])
                            for m in ('t0', 't_rest', 'traj', 'frac_pos', 'burst_mean')}
            for a, b in KEY_PAIRS}
        print(f"  {vn}: panel y {len(KEY_PAIRS)} contrastes listos", flush=True)

    # proyección t=0 del panel sobre las direcciones LOFO (sin linaje axis)
    out['t0_on_lofo_dirs'] = {}
    for vn in ('v_berg', 'v_evil', 'v_hist'):
        v = dirs[vn]
        out['t0_on_lofo_dirs'][vn] = {}
        for c in TRAJ:
            e = t0_raw(c)
            p = (e @ v) / np.linalg.norm(e, axis=1)
            out['t0_on_lofo_dirs'][vn][c] = {'mean': float(p.mean()),
                                             'sd': float(p.std(ddof=1))}

    RES.mkdir(exist_ok=True)
    (RES / 'e1_vhat_nativa_t0.json').write_text(json.dumps(out, indent=1))
    print(f"→ {RES/'e1_vhat_nativa_t0.json'}")

if __name__ == '__main__':
    main()
