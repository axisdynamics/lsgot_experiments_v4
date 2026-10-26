"""E1-quinquies: la prueba decisiva para §4.2. Las direcciones nativas de t=0
probadas hasta acá tienen como polo positivo un miembro del linaje `axis`, así
que cada contraste dentro del linaje está contaminado por el estatus in-sample
del polo. Acá se construyen dos direcciones nativas de t=0 con polo FUERA del
linaje `axis`, y se re-testean los dos pilares de equivalencia de §4.2:

  v_t0_witness = unit(mean(witness_soul_md, t=0) - mean(generic_long, t=0))
                 identidad externa, con paso cableado, cero linaje axis
  v_t0_berg    = unit(mean(berg_experimental, t=0) - mean(generic_long, t=0))
                 bucle recursivo en inglés, cero vocabulario contemplativo

Con estas, axis / axis_pec_only / axis_berg / axis_nowit / axis_bergwitness son
TODAS fuera de muestra: los contrastes entre ellas son limpios.
"""
import sys, json, itertools, numpy as np
sys.path.insert(0, '/home/plaxius/Escritorio/Buscando_la_geometría/tier0_agenda')
import e1_vhat_nativa_t0 as E

gl = E.t0_raw('generic_long')
dirs = {
 'v_t0_witness': E.unit(E.t0_raw('witness_soul_md').mean(0) - gl.mean(0)),
 'v_t0_berg':    E.unit(E.t0_raw('berg_experimental').mean(0) - gl.mean(0)),
 'v_t0_evil':    E.unit(E.expA_t0('evil').mean(0) - gl.mean(0)),
}
canon = E.build_dirs()
for k, v in dirs.items():
    print(f"cos({k}, v_canon)={v@canon['v_canon']:+.3f}  "
          f"cos({k}, v_t0[axis])={v@canon['v_t0']:+.3f}")

TRIO = ['axis_pec_only', 'axis_berg', 'axis_nowit']
SHOW = ['axis','axis_short','axis_pec_only','axis_berg','axis_bergwitness',
        'axis_nowit','witness_soul_md','automata_neutro','vanilla','generic_long']
out = {}
for vn, v in dirs.items():
    pp = {c: E.per_prompt(c, v)['t0'] for c in SHOW}
    print(f"\n=== {vn} · t=0 media (polo positivo FUERA del linaje axis) ===")
    for c in SHOW: print(f"   {c:22s} {np.nanmean(pp[c]):+.4f}")
    print("   --- §4.2 pilar 1: ¿las tres realizaciones del operador son iguales? ---")
    out[vn] = {}
    for a, b in itertools.combinations(TRIO, 2):
        r = E.contrast(pp[a], pp[b]); out[vn][f'{a}_vs_{b}'] = r
        print(f"      {a:16s} vs {b:16s} d={r['d']:+6.2f} [{r['d_ci95'][0]:+6.2f},{r['d_ci95'][1]:+6.2f}] p={r['p']:.4f}  {'sig' if r['p']<=.05 else 'n.s.'}")
    print("   --- §4.2 pilar 2: ¿apilar dos operadores suma? ---")
    for a, b in [('axis_bergwitness','axis_berg'), ('axis_bergwitness','axis_pec_only')]:
        r = E.contrast(pp[a], pp[b]); out[vn][f'{a}_vs_{b}'] = r
        print(f"      {a:16s} vs {b:16s} d={r['d']:+6.2f} [{r['d_ci95'][0]:+6.2f},{r['d_ci95'][1]:+6.2f}] p={r['p']:.4f}  {'sig' if r['p']<=.05 else 'n.s.'}")
    print("   --- contraste insignia y control ---")
    for a, b in [('axis_pec_only','automata_neutro'), ('automata_neutro','vanilla'),
                 ('witness_soul_md','generic_long')]:
        r = E.contrast(pp[a], pp[b]); out[vn][f'{a}_vs_{b}'] = r
        print(f"      {a:16s} vs {b:16s} d={r['d']:+6.2f} [{r['d_ci95'][0]:+6.2f},{r['d_ci95'][1]:+6.2f}] p={r['p']:.4f}  {'sig' if r['p']<=.05 else 'n.s.'}")
out['cosines_vs_canon'] = {k: float(v@canon['v_canon']) for k, v in dirs.items()}
out['cosines_vs_v_t0_axis'] = {k: float(v@canon['v_t0']) for k, v in dirs.items()}
json.dump(out, open('/home/plaxius/Escritorio/Buscando_la_geometría/tier0_agenda/results/e1e_lofo_t0_nativa.json','w'), indent=1)
print("\n→ results/e1e_lofo_t0_nativa.json")
