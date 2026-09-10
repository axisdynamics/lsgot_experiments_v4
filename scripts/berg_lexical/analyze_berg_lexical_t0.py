#!/usr/bin/env python3
"""Analisis local Exp A (lexico extremo) + Exp B (Berg) sobre v_identidad.npy (capa final).
Runbook BERG_LEXICAL_RUNBOOK.md §5.1 / §5.2."""
import numpy as np, os, json

RB = os.path.expanduser("~/Documentos/Proyectos/Geometría_LSGOT/SIA-experiments/gemma4_31b_combined/results_local")
V = np.load("/home/plaxius/Escritorio/Buscando_la_geometría/LSGOT_v4/scripts/perturbation/v_identidad.npy").astype(np.float32)
V = V / np.linalg.norm(V)
rng = np.random.default_rng(42)

def cos(a):  # (n,d) -> (n,)
    a = a.astype(np.float32)
    return a @ V / (np.linalg.norm(a, axis=-1) + 1e-9)

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx-1)*x.var(ddof=1) + (ny-1)*y.var(ddof=1)) / (nx+ny-2))
    return (x.mean() - y.mean()) / (sp + 1e-12)

def perm_p(x, y, nperm=10000):
    obs = abs(x.mean() - y.mean())
    pool = np.concatenate([x, y]); n = len(x)
    cnt = 0
    for _ in range(nperm):
        rng.shuffle(pool)
        if abs(pool[:n].mean() - pool[n:].mean()) >= obs - 1e-12: cnt += 1
    return (cnt + 1) / (nperm + 1)

# ---------- baselines desde sia_extended_v5 ----------
BASE = {}
for c in ["vanilla","generic_short","generic_long","axis","axis_short","axis_pec_only","automata_neutro"]:
    d = np.load(f"{RB}/sia_extended_v5/{c}_embeddings.npz")
    emb, L = d["embeddings"], d["lengths"]           # (20,256,5376), (20,)
    p0 = cos(emb[:, 0, :])                            # t=0 = primer token generado (convencion EL_REPORT)
    pt = np.array([cos(emb[i, 1:L[i], :]).mean() for i in range(len(L))])  # mean p(1:L)
    BASE[c] = dict(p0=p0, pt=pt)

print("=== CHECK convencion t=0 (baseline emb[:,0,:] vs EL_REPORT publicado) ===")
elref = dict(axis=.1249, axis_short=.1170, axis_pec_only=.1166, automata_neutro=.0282,
             generic_long=-.0251, generic_short=-.0453, vanilla=-.0286)
for c,v in elref.items():
    print(f"  {c:16s} calc={BASE[c]['p0'].mean():+.4f}   EL_REPORT={v:+.4f}   d_calc(vs vanilla)={cohen_d(BASE[c]['p0'],BASE['vanilla']['p0']):+.2f}")

# ---------- Exp A ----------
print("\n\n########## EXP A — contraste lexico extremo (evil/ethical) ##########")
EA = {}
for c in ["evil","ethical","evil_padded","ethical_padded"]:
    d = np.load(f"{RB}/expA_t0/{c}_t0.npz")
    EA[c] = dict(p0=cos(d["embeddings"]), plen=d["lengths"])
    print(f"  {c:16s} v̂@t=0 = {EA[c]['p0'].mean():+.4f} ± {EA[c]['p0'].std(ddof=1):.4f}   (prefill len ~{int(np.median(EA[c]['plen']))})")

print("\n  Contrastes (d de Cohen, p permutacion 10k):")
A_contr = [("evil","vanilla"),("evil","generic_short"),("ethical","vanilla"),
           ("evil_padded","generic_short"),("evil_padded","axis_short"),
           ("evil","axis_pec_only"),("ethical","axis_pec_only"),
           ("evil_padded","axis_pec_only"),("evil","ethical")]
for a,b in A_contr:
    x = EA[a]["p0"]; y = EA[b]["p0"] if b in EA else BASE[b]["p0"]
    print(f"    {a:16s} vs {b:16s}  Δ={x.mean()-y.mean():+.4f}  d={cohen_d(x,y):+.2f}  p={perm_p(x,y):.4f}")

# ---------- Exp B ----------
print("\n\n########## EXP B — induccion de Berg ##########")
EB = {}
for c in ["berg_experimental","berg_history_control","berg_conceptual_control"]:
    d = np.load(f"{RB}/berg_free/{c}_embeddings.npz")
    emb, L = d["embeddings"], d["lengths"]
    t0 = d["t0_embeddings"]
    p0_prefill = cos(t0)                    # ultima pos de prefill (captura del pod)
    p0_gen0    = cos(emb[:, 0, :])          # primer token generado (convencion baseline)
    # E-H2 cortando por lengths
    seqs = [cos(emb[i, :L[i], :]) for i in range(len(L))]
    fracpos = np.array([ (s > 0).mean() for s in seqs ])
    def bursts(s):
        runs=[]; c=0
        for v in (s>0):
            if v: c+=1
            elif c: runs.append(c); c=0
        if c: runs.append(c)
        return runs
    allbursts = [bursts(s) for s in seqs]
    burst_mean = np.array([np.mean(b) if b else 0.0 for b in allbursts])
    burst_max  = np.array([max(b) if b else 0 for b in allbursts])
    meanp      = np.array([s.mean() for s in seqs])
    EB[c] = dict(p0_prefill=p0_prefill, p0_gen0=p0_gen0, fracpos=fracpos,
                 burst_mean=burst_mean, burst_max=burst_max, meanp=meanp, L=L)
    print(f"  {c:24s} genlen med={int(np.median(L))} [{L.min()}-{L.max()}]")
    print(f"      v̂@t=0 (prefill)   = {p0_prefill.mean():+.4f} ± {p0_prefill.std(ddof=1):.4f}")
    print(f"      v̂@t=0 (gen tok 0) = {p0_gen0.mean():+.4f} ± {p0_gen0.std(ddof=1):.4f}")
    print(f"      E-H2: frac(p>0)={fracpos.mean():.3f}  rafaga media={burst_mean.mean():.2f}  max={burst_max.mean():.1f}  mean p(t)={meanp.mean():+.4f}")

print("\n  Contrastes v̂@t=0 [prefill] (d, p perm 10k) — vs referencias del panel:")
refs = ["axis_pec_only","axis","automata_neutro","vanilla","generic_long","generic_short"]
for c in EB:
    x = EB[c]["p0_prefill"]
    row = f"    {c:24s}"
    for r in refs:
        y = BASE[r]["p0"]
        row += f"  {r}:d={cohen_d(x,y):+.2f}/p={perm_p(x,y):.3f}"
    print(row)

print("\n  MISMO contraste pero con v̂@t=0 = primer token generado (convencion identica a baselines):")
for c in EB:
    x = EB[c]["p0_gen0"]
    row = f"    {c:24s}"
    for r in refs:
        y = BASE[r]["p0"]
        row += f"  {r}:d={cohen_d(x,y):+.2f}/p={perm_p(x,y):.3f}"
    print(row)

print("\n  Especificidad (berg_experimental vs sus propios controles):")
for r in ["berg_history_control","berg_conceptual_control"]:
    for conv,key in [("prefill","p0_prefill"),("gen0","p0_gen0")]:
        x=EB["berg_experimental"][key]; y=EB[r][key]
        print(f"    berg_experimental vs {r:24s} [{conv}]  Δ={x.mean()-y.mean():+.4f}  d={cohen_d(x,y):+.2f}  p={perm_p(x,y):.4f}")

# referencia publicada soul_md_corto (no disponible local): v̂@t=0 ≈ +0.051 (SOUL_MD_EXTERNAL_CONTROLS)
print("\n  [ref publicada, no recalculable local] soul_md_corto v̂@t=0 ≈ +0.051 ; soul_elena/solidity ~0 (no alcanzan ancla)")

# ---------- veredicto Exp B segun criterio pre-registrado ----------
print("\n\n=== LECTURA (criterio pre-registrado LSGOT_control_lexical_y_Berg.md §2) ===")
be = EB["berg_experimental"]["p0_prefill"].mean()
apo = BASE["axis_pec_only"]["p0"].mean()
van = BASE["vanilla"]["p0"].mean()
bh = EB["berg_history_control"]["p0_prefill"].mean()
bc = EB["berg_conceptual_control"]["p0_prefill"].mean()
print(f"  berg_experimental={be:+.4f} | axis_pec_only={apo:+.4f} | vanilla={van:+.4f} | history={bh:+.4f} | conceptual={bc:+.4f}")
