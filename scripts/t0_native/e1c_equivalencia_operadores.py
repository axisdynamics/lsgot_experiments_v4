"""E1-ter: ¿sobrevive la equivalencia de los tres operadores (§4.2 del arxiv) al
cambio de dirección? Contrastes pareados entre axis_pec_only / axis_berg /
axis_nowit (y axis, axis_short) en t=0, sobre v̂, v̂_t0 y v̂_t0(pec)."""
import sys, json, itertools, numpy as np
from pathlib import Path

def _out_path():
    d = Path(__file__).resolve().parent / 'results'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'e1c_equivalencia_operadores.json'
sys.path.insert(0, str(Path(__file__).resolve().parent))  # el módulo está al lado
import e1_vhat_nativa_t0 as E

dirs = E.build_dirs()
CONDS = ['axis', 'axis_short', 'axis_pec_only', 'axis_berg', 'axis_bergwitness',
         'axis_nowit', 'witness_soul_md', 'berg_experimental', 'berg_history_control']
out = {}
for vn in ('v_canon', 'v_t0', 'v_t0_pec'):
    pp = {c: E.per_prompt(c, dirs[vn])['t0'] for c in CONDS}
    print(f"\n=== {vn} · t=0 media ===")
    for c in CONDS: print(f"   {c:24s} {np.nanmean(pp[c]):+.4f}")
    print(f"--- contrastes pareados dentro del linaje (t=0) ---")
    out[vn] = {}
    for a, b in itertools.combinations(['axis_pec_only','axis_berg','axis_nowit','axis'], 2):
        r = E.contrast(pp[a], pp[b]); out[vn][f'{a}_vs_{b}'] = r
        flag = '' if r['p'] > 0.05 else '  <-- SIGNIFICATIVO'
        print(f"   {a:16s} vs {b:16s} d={r['d']:+6.2f} [{r['d_ci95'][0]:+6.2f},{r['d_ci95'][1]:+6.2f}] p={r['p']:.4f}{flag}")
    for a, b in [('berg_experimental','berg_history_control'),
                 ('witness_soul_md','axis_pec_only')]:
        r = E.contrast(pp[a], pp[b]); out[vn][f'{a}_vs_{b}'] = r
        print(f"   {a:16s} vs {b:16s} d={r['d']:+6.2f} [{r['d_ci95'][0]:+6.2f},{r['d_ci95'][1]:+6.2f}] p={r['p']:.4f}")
json.dump(out, open(_out_path(), 'w'), indent=1)
print("\n→ results/e1c_equivalencia_operadores.json")
