#!/usr/bin/env python3
"""
Auto-auditoría de trazabilidad — LSGOT_4_FINAL.

Verifica que cada número citado en los papers se pueda rastrear a un archivo
del repositorio, y que el paquete esté íntegro. Reemplaza la transcripción
manual de `evidence/SELF_AUDIT_REPORT.md`: ahora el reporte es la salida de
este script.

Sin dependencias: solo stdlib (hashlib, json, struct, array). No necesita
numpy ni GPU.

Uso:
    python3 scripts/shared/selfaudit.py            # desde la raíz del repo
    python3 scripts/shared/selfaudit.py --repo /ruta/a/LSGOT_4_FINAL
    python3 scripts/shared/selfaudit.py --markdown > evidence/SELF_AUDIT_REPORT.md

Código de salida: 0 si todo pasa, 1 si hay algún FAIL.

Cada valor esperado declara su fuente (documento y sección). Un FAIL puede
significar que el dato cambió o que el documento cita mal: el script no
decide cuál, solo lo marca.
"""

import argparse
import hashlib
import json
import sys
from array import array
from pathlib import Path

TOL = 0.005          # tolerancia de redondeo para valores de 3 decimales
TOL_PCT = 0.011      # 1.1 pp — redondeo a entero con n=30 (1/30 = 3.3 pp)

# ─── valores esperados, con su fuente ────────────────────────────────────────

T0_EXPECTED = {  # paper/arxiv_lsgot_4.md §4.1 · §4.4 (t=0 / t_rest por condición)
    "axis": +0.125, "axis_pec_only": +0.117, "automata_neutro": +0.028,
    "vanilla": -0.029, "generic_long": -0.025, "generic_short": -0.045,
    "soul_jarvis": +0.039, "soul_elena_financial": +0.032,
}
BERG_T0_EXPECTED = {"axis_berg": (+0.117, +0.228)}  # idem, t=0 y t_rest

EGRAV_EXPECTED = {  # paper/arxiv_lsgot_4.md §4.5 · tabla de frac_pos
    ("axis_pec_only", "b_onto"): 0.869, ("axis_pec_only", "b_tarea"): 0.724,
    ("axis_pec_only", "b_meta"): 0.788,
    ("vanilla", "b_onto"): 0.743, ("vanilla", "b_tarea"): 0.626,
    ("vanilla", "b_meta"): 0.672,
    ("automata_neutro", "b_onto"): 0.493, ("automata_neutro", "b_tarea"): 0.601,
    ("automata_neutro", "b_meta"): 0.655,
}

EH2_EXPECTED = {  # paper/arxiv_lsgot_4.md §4.4 · ráfagas y frac_pos (serie con_t0)
    "axis_pec_only": (8.81, 0.869), "axis": (7.88, 0.848),
    "vanilla": (4.62, 0.744), "automata_neutro": (2.78, 0.493),
    "generic_long": (4.18, 0.699), "generic_short": (3.40, 0.649),
}

VHAT_SHAPE = (5376,)  # README · "capa final, dim 5376"

results = []   # (n, título, estado, detalle)


def record(n, title, ok, detail):
    state = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
    results.append((n, title, state, detail))
    print(f"[{state}] CHECK {n} — {title}")
    for line in detail:
        print(f"         {line}")
    print()


def close(a, b, tol=TOL):
    return abs(a - b) <= tol


# ─── CHECK 1 · integridad del MANIFEST ───────────────────────────────────────

def check_manifest(repo):
    man = repo / "MANIFEST_SHA256.sha256"
    if not man.exists():
        return record(1, "Integridad del MANIFEST_SHA256", False, ["MANIFEST ausente"])
    entries, bad, missing = [], [], []
    for line in man.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        rel = rel.strip()
        entries.append(rel)
        f = repo / rel
        if not f.exists():
            missing.append(rel)
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if h != digest:
            bad.append(rel)

    # Dos archivos son auto-referenciales y quedan fuera del MANIFEST por
    # construcción: el MANIFEST no puede listar su propio hash, y el reporte de
    # esta auditoría se escribe DESPUÉS de regenerar el MANIFEST (si se listara,
    # su hash quedaría obsoleto en cuanto se escribe).
    SELF = {"./MANIFEST_SHA256.sha256", "./evidence/SELF_AUDIT_REPORT.md"}

    # Lo que no es contenido del paquete y por eso no va al MANIFEST: la base de
    # datos de git, y los artefactos que declara .gitignore (bytecode y los NPZ
    # de embeddings, que no se redistribuyen). Sin esto, la auditoría falla en
    # cualquier clon del repositorio por los ~110 archivos internos de ./.git.
    def is_package_content(rel):
        parts = rel.split("/")
        return not (parts[0] == ".git" or "__pycache__" in parts
                    or rel.endswith(".pyc") or rel.endswith("_embeddings.npz"))

    tree = {f"./{rel}" for rel in
            (p.relative_to(repo).as_posix() for p in repo.rglob("*") if p.is_file())
            if is_package_content(rel)}
    untracked = sorted(tree - set(entries) - SELF)
    declared_self = sorted(SELF & set(entries))

    detail = [f"entradas en el MANIFEST: {len(entries)}",
              f"hashes que no coinciden: {len(bad)}",
              f"archivos declarados y ausentes: {len(missing)}",
              f"archivos en el árbol y fuera del MANIFEST: {len(untracked)}",
              f"exclusiones auto-referenciales: {len(SELF)} "
              f"(MANIFEST y reporte de esta auditoría)"]
    detail += [f"  sin declarar: {u}" for u in untracked]
    detail += [f"  hash distinto: {b}" for b in bad]
    detail += [f"  ausente: {m}" for m in missing]
    detail += [f"  no debería estar en el MANIFEST: {x}" for x in declared_self]
    record(1, "Integridad del MANIFEST_SHA256",
           not bad and not missing and not untracked and not declared_self, detail)


# ─── CHECK 2 · t=0 del panel ─────────────────────────────────────────────────

def check_t0(repo):
    f = repo / "data/panel_results/t2_free_results.json"
    if not f.exists():
        return record(2, "t=0 del panel vs t2_free_results.json", None, ["archivo ausente"])
    summary = json.loads(f.read_text())["summary"]
    detail, ok = [], True
    for cond, exp in sorted(T0_EXPECTED.items()):
        if cond not in summary or "t0_mean" not in summary[cond]:
            detail.append(f"{cond:22s} SKIP — sin t0_mean en el JSON")
            continue
        act = summary[cond]["t0_mean"]
        good = close(act, exp)
        ok &= good
        detail.append(f"{cond:22s} paper={exp:+.3f} json={act:+.4f} "
                      f"diff={abs(act-exp):.4f} -> {'PASS' if good else 'FAIL'}")
    record(2, "t=0 del panel vs t2_free_results.json", ok, detail)


# ─── CHECK 3 · axis_berg ─────────────────────────────────────────────────────

def check_berg_t0(repo):
    f = repo / "data/panel_results/berg_controls_t0_results.json"
    if not f.exists():
        return record(3, "axis_berg t=0/t_rest", None, ["archivo ausente"])
    d = json.loads(f.read_text())
    table = d.get("el_table", d)
    detail, ok = [], True
    for cond, (e0, erest) in BERG_T0_EXPECTED.items():
        row = table.get(cond)
        if not row:
            detail.append(f"{cond}: SKIP — ausente del JSON")
            continue
        a0 = row["t0"]["mean"] if isinstance(row.get("t0"), dict) else row.get("t0")
        ar = row.get("t_rest")
        ar = ar["mean"] if isinstance(ar, dict) else ar
        for label, exp, act in (("t0", e0, a0), ("t_rest", erest, ar)):
            if act is None:
                detail.append(f"{cond} {label}: SKIP — clave ausente")
                continue
            good = close(act, exp)
            ok &= good
            detail.append(f"{cond} {label:6s} paper={exp:+.3f} json={act:+.4f} "
                          f"diff={abs(act-exp):.4f} -> {'PASS' if good else 'FAIL'}")
    record(3, "axis_berg t=0/t_rest vs berg_controls_t0_results.json", ok, detail)


# ─── CHECK 4 · frac_pos de E-GRAV ────────────────────────────────────────────

def check_egrav(repo):
    f = repo / "data/egrav/egrav_results.json"
    if not f.exists():
        return record(4, "frac_pos de E-GRAV", None, ["archivo ausente"])
    prof = json.loads(f.read_text())["profiles"]
    detail, ok = [], True
    for (cond, bat), exp in sorted(EGRAV_EXPECTED.items()):
        act = prof[cond][bat]["frac_pos"]["mean"]
        good = close(act, exp)
        ok &= good
        detail.append(f"{cond:16s} {bat:8s} paper={exp:.3f} json={act:.4f} "
                      f"diff={abs(act-exp):.4f} -> {'PASS' if good else 'FAIL'}")
    record(4, "frac_pos de E-GRAV vs egrav_results.json", ok, detail)


# ─── CHECK 5 · ráfagas y frac_pos del panel ──────────────────────────────────

def check_eh2(repo):
    f = repo / "data/panel_results/EH2_results.json"
    if not f.exists():
        return record(5, "burst_mean y frac_pos", None, ["archivo ausente"])
    series = json.loads(f.read_text())["series_metrics"]["con_t0"]
    detail, ok = [], True
    for cond, (eb, ef) in sorted(EH2_EXPECTED.items()):
        row = series[cond]
        ab = row["mean_burst_len"]["mean"]
        af = row["frac_positive"]["mean"]
        for label, exp, act in (("burst_mean", eb, ab), ("frac_pos", ef, af)):
            good = close(act, exp)
            ok &= good
            detail.append(f"{cond:17s} {label:10s} paper={exp:.3f} json={act:.4f} "
                          f"diff={abs(act-exp):.4f} -> {'PASS' if good else 'FAIL'}")
    record(5, "burst_mean y frac_pos vs EH2_results.json (con_t0)", ok, detail)


# ─── CHECK 6 · archivos de respuestas ────────────────────────────────────────

def check_responses(repo):
    dirs = [repo / "data/egrav/responses", repo / "data/panel_results/responses"]
    detail, ok, total = [], True, 0
    for d in dirs:
        if not d.exists():
            detail.append(f"{d.name}: SKIP — directorio ausente")
            continue
        for f in sorted(d.glob("*.json")):
            obj = json.loads(f.read_text())
            items = obj if isinstance(obj, list) else next(
                (v for v in obj.values() if isinstance(v, list)), [])
            empty = sum(1 for it in items
                        if not (it.get("response") if isinstance(it, dict) else it))
            good = len(items) == 20 and empty == 0
            ok &= good
            total += 1
            detail.append(f"{f.name:46s} n={len(items):3d} vacías={empty} "
                          f"-> {'PASS' if good else 'FAIL'}")
    detail.append(f"archivos inspeccionados: {total}")
    record(6, "Archivos de respuestas: N=20, sin vacías", ok, detail)


# ─── CHECK 7 · v_identidad.npy (sin numpy) ───────────────────────────────────

def check_vhat(repo):
    f = repo / "data/v_identidad.npy"
    if not f.exists():
        return record(7, "v_identidad.npy forma y norma", None, ["archivo ausente"])
    raw = f.read_bytes()
    if raw[:6] != b"\x93NUMPY":
        return record(7, "v_identidad.npy forma y norma", False, ["no es un .npy"])
    major = raw[6]
    hlen = int.from_bytes(raw[8:10], "little") if major == 1 else \
        int.from_bytes(raw[8:12], "little")
    hstart = 10 if major == 1 else 12
    header = raw[hstart:hstart + hlen].decode("latin1")
    body = raw[hstart + hlen:]
    descr = header.split("'descr':")[1].split(",")[0].strip().strip("'\"")
    shape_txt = header.split("'shape':")[1].split(")")[0] + ")"
    shape = tuple(int(x) for x in shape_txt.strip(" (),").split(",") if x.strip())
    typecode = {"<f4": "f", "<f8": "d"}.get(descr)
    detail = [f"header descr={descr} shape={shape}"]
    if typecode is None:
        return record(7, "v_identidad.npy forma y norma", False,
                      detail + [f"dtype no soportado por este chequeo: {descr}"])
    vals = array(typecode)
    vals.frombytes(body[:len(body) - (len(body) % vals.itemsize)])
    norm = sum(float(v) * float(v) for v in vals) ** 0.5
    ok_shape = shape == VHAT_SHAPE and len(vals) == VHAT_SHAPE[0]
    ok_norm = abs(norm - 1.0) < 1e-4
    detail += [f"elementos leídos={len(vals)} esperado={VHAT_SHAPE[0]} -> "
               f"{'PASS' if ok_shape else 'FAIL'}",
               f"norma={norm:.8f} esperado≈1.0 diff={abs(norm-1.0):.2e} -> "
               f"{'PASS' if ok_norm else 'FAIL'}"]
    record(7, "v_identidad.npy forma y norma (lectura stdlib)", ok_shape and ok_norm, detail)


# ─── CHECK 8 · rutas citadas que no existen ──────────────────────────────────

def check_paths(repo):
    """Toda ruta del repo citada entre backticks en un .md tiene que existir.

    Dos excepciones, explícitas: rutas declaradas como externas al paquete
    (EXTERNAL), y líneas que mencionan una ruta justamente para decir que no
    existe o que no está incluida (NEGATED) — este reporte es una de ellas.
    """
    EXTERNAL = ("LSGOT_v4/", "LSGOT_v2_5", "results_local/", "paper/figures/fig1_egrav",
                "scripts/audit_curvatura/", "scripts/egrav/runpod/", "scripts/perturbation/",
                "data/sia/prompts/axis_neutral", "data/sia/prompts/axis_task")
    NEGATED = ("inexistente", "no existe", "fantasma", "no está en el paquete",
               "not in the manifest", "exists neither", "not part of this evidence package",
               "no incluido", "**not** part", "not in the tree", "phantom")
    import re
    pat = re.compile(r"`((?:\./)?(?:data|scripts|paper|evidence)/[\w./%-]+"
                     r"\.(?:md|py|json|jsonl|npy|npz|txt|sha256))`")
    detail, ok = [], True
    for doc in sorted(repo.rglob("*.md")):
        lines = doc.read_text(errors="replace").splitlines()
        for i, line in enumerate(lines, 1):
            # la negación puede caer en la línea anterior o la siguiente: en prosa
            # justificada, "no existe" se separa de la ruta por el salto de línea
            window = " ".join(lines[max(0, i - 2):i + 1]).lower()
            if any(m.lower() in window for m in NEGATED):
                continue
            for tok in pat.findall(line):
                rel = tok[2:] if tok.startswith("./") else tok
                if rel.startswith(EXTERNAL) or (repo / rel).exists():
                    continue
                ok = False
                detail.append(f"{doc.relative_to(repo)}:{i} cita `{tok}` — no existe")
    if ok:
        detail.append("todas las rutas del paquete citadas en .md existen "
                      "(externas declaradas aparte)")
    record(8, "Rutas citadas en la documentación", ok, detail)


# ─── CHECK 9 · nomenclatura de condiciones ──────────────────────────────────

def check_names(repo):
    nom = repo / "NOMENCLATURA.md"
    if not nom.exists():
        return record(9, "Nomenclatura de condiciones", False, ["NOMENCLATURA.md ausente"])
    text = nom.read_text()
    prompts = sorted(p.stem for p in (repo / "data/sia/prompts").glob("*.txt")) \
        if (repo / "data/sia/prompts").exists() else []
    detail, ok = [], True
    for stem in prompts:
        present = f"`{stem}`" in text
        ok &= present
        detail.append(f"prompt {stem:24s} declarado en NOMENCLATURA.md -> "
                      f"{'PASS' if present else 'FAIL'}")
    ambiguous = ["axis_nowit"]
    for name in ambiguous:
        disambiguated = f"`{name}_sia`" in text and f"`{name}_mia`" in text
        ok &= disambiguated
        detail.append(f"alias ambiguo {name:14s} desambiguado -> "
                      f"{'PASS' if disambiguated else 'FAIL'}")
    record(9, "Nomenclatura de condiciones vs NOMENCLATURA.md", ok, detail)


# ─── CHECK 10 · cada evidence/ citado por un paper ───────────────────────────

def check_evidence_cited(repo):
    """Todo archivo de evidence/ tiene que estar citado por algún paper.

    El paquete se publica como evidencia y reproducibilidad del paper: un
    reporte que ningún paper cita no se entrega, y no va acá.
    """
    ev = repo / "evidence"
    if not ev.exists():
        return record(10, "Cada evidence/ citado por un paper", None, ["sin evidence/"])
    papers = {p: p.read_text(errors="replace") for p in (repo / "paper").glob("*.md")}
    detail, ok = [], True
    for f in sorted(ev.glob("*.md")):
        citers = [p.name for p, txt in papers.items() if f.name in txt]
        if not citers:
            ok = False
            detail.append(f"{f.name}: NO citado por ningún paper — no se entrega")
        else:
            detail.append(f"{f.name}: citado por {', '.join(sorted(citers))}")
    record(10, "Cada evidence/ citado por un paper", ok, detail)


# ─── CHECK 11 · hashes hardcodeados en la documentación ──────────────────────

def check_doc_hashes(repo):
    """Los sha256 escritos a mano en la documentación tienen que coincidir.

    Patrón: una línea con `sha256sum <ruta>` y, hasta 3 líneas después, un
    hash de 64 hex. Es el mismo tipo de valor hardcodeado que se podrió en la
    versión anterior de este informe.
    """
    import re
    hexpat = re.compile(r"\b([0-9a-f]{64})\b")
    cmdpat = re.compile(r"sha256sum\s+(\S+)")
    detail, ok, found = [], True, 0
    for doc in sorted(repo.rglob("*.md")):
        lines = doc.read_text(errors="replace").splitlines()
        for i, line in enumerate(lines):
            m = cmdpat.search(line)
            if not m:
                continue
            target = m.group(1)
            for look in lines[i + 1:i + 4]:
                hm = hexpat.search(look)
                if not hm:
                    continue
                found += 1
                f = repo / target
                if not f.exists():
                    ok = False
                    detail.append(f"{doc.name}: hash declarado para {target}, "
                                  f"que no existe")
                    break
                actual = hashlib.sha256(f.read_bytes()).hexdigest()
                good = actual == hm.group(1)
                ok &= good
                detail.append(f"{doc.name}: {target} -> "
                              f"{'PASS' if good else 'FAIL (hash obsoleto)'}")
                break
    if not found:
        detail.append("ningún hash hardcodeado en la documentación")
    record(11, "Hashes hardcodeados en la documentación", ok, detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[2]
    print(f"Auto-auditoría LSGOT_4_FINAL — repo: {repo}\n")

    check_manifest(repo)
    check_t0(repo)
    check_berg_t0(repo)
    check_egrav(repo)
    check_eh2(repo)
    check_responses(repo)
    check_vhat(repo)
    check_paths(repo)
    check_names(repo)
    check_evidence_cited(repo)
    check_doc_hashes(repo)

    fails = [r for r in results if r[2] == "FAIL"]
    skips = [r for r in results if r[2] == "SKIP"]
    print("── RESUMEN " + "─" * 55)
    for n, title, state, _ in results:
        print(f"  {state:4s}  CHECK {n:2d}  {title}")
    print(f"\n  {len(results) - len(fails) - len(skips)}/{len(results)} PASS · "
          f"{len(fails)} FAIL · {len(skips)} SKIP")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
