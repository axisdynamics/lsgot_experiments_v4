"""E1-quater: los otros dos pilares de §4.2/§4.3 del arxiv bajo las tres direcciones.
(i) «apilar dos operadores no suma nada»: axis_bergwitness vs axis_berg.
(ii) §4.3 «el split de los SOUL.md externos sigue al mecanismo»: las dos con paso
     cableado vs las cuatro sin, y todas vs generic_long."""
import sys, json, numpy as np
sys.path.insert(0, '/home/plaxius/Escritorio/Buscando_la_geometría/tier0_agenda')
import e1_vhat_nativa_t0 as E

dirs = E.build_dirs()
PAIRS = [
 ('axis_bergwitness','axis_berg'),          # apilar operadores
 ('axis_bergwitness','axis_pec_only'),
 ('witness_soul_md','generic_long'),        # SOUL con paso cableado vs genérico
 ('witness_soul_md','vanilla'),
 ('witness_soul_md','soul_jarvis'),         # cableado vs no cableado
 ('witness_soul_md','soul_elena_financial'),
 ('witness_soul_md','soul_solidity_auditor'),
 ('soul_jarvis','vanilla'),
 ('soul_elena_financial','vanilla'),
 ('soul_solidity_auditor','vanilla'),
 ('automata_neutro','witness_soul_md'),
 ('berg_history_control','generic_long'),   # andamiaje recursivo genérico
 ('berg_experimental','generic_long'),
]
CONDS = sorted({c for p in PAIRS for c in p})
out = {}
for vn in ('v_canon','v_t0','v_t0_pec'):
    pp = {c: E.per_prompt(c, dirs[vn])['t0'] for c in CONDS}
    print(f"\n=== {vn} · t=0 ===")
    out[vn] = {}
    for a,b in PAIRS:
        r = E.contrast(pp[a], pp[b]); out[vn][f'{a}_vs_{b}'] = r
        sig = 'sig ' if r['p'] <= 0.05 else 'n.s.'
        print(f"   {a:22s} vs {b:22s} d={r['d']:+6.2f} [{r['d_ci95'][0]:+6.2f},{r['d_ci95'][1]:+6.2f}] p={r['p']:.4f}  {sig}")
json.dump(out, open('/home/plaxius/Escritorio/Buscando_la_geometría/tier0_agenda/results/e1d_pilares_42_43.json','w'), indent=1)
print("\n→ results/e1d_pilares_42_43.json")
