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


# ─── CHECK 4 · ráfagas y frac_pos del panel ──────────────────────────────────

def check_eh2(repo):
    f = repo / "data/panel_results/EH2_results.json"
    if not f.exists():
        return record(4, "burst_mean y frac_pos", None, ["archivo ausente"])
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
    record(4, "burst_mean y frac_pos vs EH2_results.json (con_t0)", ok, detail)


# ─── CHECK 5 · archivos de respuestas ────────────────────────────────────────

def check_responses(repo):
    dirs = [repo / "data/panel_results/responses"]
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
    record(5, "Archivos de respuestas: N=20, sin vacías", ok, detail)


# ─── CHECK 6 · v_identidad.npy (sin numpy) ───────────────────────────────────

def check_vhat(repo):
    f = repo / "data/v_identidad.npy"
    if not f.exists():
        return record(6, "v_identidad.npy forma y norma", None, ["archivo ausente"])
    raw = f.read_bytes()
    if raw[:6] != b"\x93NUMPY":
        return record(6, "v_identidad.npy forma y norma", False, ["no es un .npy"])
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
        return record(6, "v_identidad.npy forma y norma", False,
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
    record(6, "v_identidad.npy forma y norma (lectura stdlib)", ok_shape and ok_norm, detail)


# ─── CHECK 7 · rutas citadas que no existen ──────────────────────────────────

def check_paths(repo):
    """Toda ruta del repo citada entre backticks en un .md tiene que existir.

    Dos excepciones, explícitas: rutas declaradas como externas al paquete
    (EXTERNAL), y líneas que mencionan una ruta justamente para decir que no
    existe o que no está incluida (NEGATED) — este reporte es una de ellas.
    """
    EXTERNAL = ("LSGOT_v4/", "LSGOT_v2_5", "results_local/",
                "scripts/audit_curvatura/", "scripts/egrav/", "scripts/perturbation/",
                "data/egrav/", "paper/figures/",
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
    record(7, "Rutas citadas en la documentación", ok, detail)


# ─── CHECK 8 · nomenclatura de condiciones ──────────────────────────────────

def check_names(repo):
    nom = repo / "NOMENCLATURA.md"
    if not nom.exists():
        return record(8, "Nomenclatura de condiciones", False, ["NOMENCLATURA.md ausente"])
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
    record(8, "Nomenclatura de condiciones vs NOMENCLATURA.md", ok, detail)


# ─── CHECK 9 · cada evidence/ citado por un paper ───────────────────────────

def check_evidence_cited(repo):
    """Todo archivo de evidence/ tiene que estar citado por algún paper.

    El paquete se publica como evidencia y reproducibilidad del paper: un
    reporte que ningún paper cita no se entrega, y no va acá.
    """
    ev = repo / "evidence"
    if not ev.exists():
        return record(9, "Cada evidence/ citado por un paper", None, ["sin evidence/"])
    papers = {p: p.read_text(errors="replace") for p in (repo / "paper").glob("*.md")}
    detail, ok = [], True
    for f in sorted(ev.glob("*.md")):
        citers = [p.name for p, txt in papers.items() if f.name in txt]
        if not citers:
            ok = False
            detail.append(f"{f.name}: NO citado por ningún paper — no se entrega")
        else:
            detail.append(f"{f.name}: citado por {', '.join(sorted(citers))}")
    record(9, "Cada evidence/ citado por un paper", ok, detail)


# ─── CHECK 10 · hashes hardcodeados en la documentación ──────────────────────

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
    record(10, "Hashes hardcodeados en la documentación", ok, detail)


# ─── CHECK 11 · comandos citados en la documentación ─────────────────────────

def check_commands(repo):
    """Todo `python <script>.py --flag` citado en un bloque bash de un .md tiene
    que existir y aceptar ese flag.

    Es el chequeo que faltaba: CHECK 7 mira rutas entre backticks, así que un
    nombre de script desnudo dentro de un bloque de código y un flag que el
    script no acepta se le escapan. Los dos rompen a quien copia y pega.

    Los flags se leen de la MISMA línea lógica del comando — uniendo las
    continuaciones con barra invertida — y no del resto del bloque, que puede
    traer otros comandos.

    `prereg/` queda fuera: son registros fechados de corridas ya ejecutadas y sus
    comandos se conservan tal como se corrieron, no se reescriben cuando un script
    se renombra. Cada uno declara el renombre en una nota fechada al principio, y
    esa nota sí la cubre CHECK 7 porque cita la ruta nueva entre backticks.
    """
    import re
    from pathlib import PurePosixPath
    scripts = {q.name: q for q in repo.rglob("*.py")}
    # argparse se escribe con comillas simples o dobles: las dos cuentan
    # argparse se escribe con comillas simples o dobles: las dos cuentan
    # argparse se escribe con comillas simples o dobles: \x22 y \x27 cubren las dos
    qpat = re.compile(r'[\x22\x27](--[a-z][a-z-]*)[\x22\x27]')
    flags = {n: set(qpat.findall(q.read_text(errors="replace")))
             for n, q in scripts.items()}
    invoke = re.compile(r"\bpython3?\s+(?:-\w+\s+)*([\w./-]+\.py)")

    detail, ok, seen = [], True, 0
    for doc in sorted(repo.rglob("*.md")):
        if doc.relative_to(repo).parts[0] == "prereg":
            continue
        raw = doc.read_text(errors="replace").replace("\\\n", " ")  # une continuaciones
        for logical in raw.splitlines():
            m = invoke.search(logical)
            if not m:
                continue
            seen += 1
            script = PurePosixPath(m.group(1)).name
            if script not in scripts:
                ok = False
                detail.append(f"{doc.relative_to(repo)}: cita `{script}`, que no existe "
                              f"en el paquete")
                continue
            for fl in set(re.findall(r"\s(--[a-z][a-z-]*)", logical[m.end():])):
                if fl not in flags[script]:
                    ok = False
                    detail.append(f"{doc.relative_to(repo)}: `{script} {fl}` — el script "
                                  f"no acepta ese flag (acepta: "
                                  f"{' '.join(sorted(flags[script])) or 'ninguno'})")
    if ok:
        detail.append(f"{seen} invocaciones citadas; todos los scripts existen y "
                      f"aceptan los flags que se les pasan")
    record(11, "Comandos citados en la documentación", ok, detail)


# ─── CHECK 12 · sin rutas absolutas de una máquina ───────────────────────────

def check_no_local_paths(repo):
    """Ninguna ruta `/home/<usuario>/` en el paquete, salvo las declaradas.

    Un clon limpio no tiene el home de nadie. La única excepción permitida es un
    default dentro de una lectura de variable de entorno, que es cómo se
    configura la raíz de las trayectorias, que no se redistribuyen.
    """
    import re
    pat = re.compile(r"/home/[a-z_][a-z0-9_-]*/")
    ALLOWED_ENV_DEFAULT = "LSGOT_TRAJ_ROOT"
    detail, ok = [], True
    for f in sorted(list(repo.rglob("*.py")) + list(repo.rglob("*.md"))):
        rel = f.relative_to(repo).as_posix()
        if rel == "evidence/SELF_AUDIT_REPORT.md":
            continue                      # es la salida de este script
        lines = f.read_text(errors="replace").splitlines()
        for i, line in enumerate(lines, 1):
            if not pat.search(line):
                continue
            window = "\n".join(lines[max(0, i - 8):i])
            if ALLOWED_ENV_DEFAULT in window:
                continue                  # default documentado de una env var
            ok = False
            detail.append(f"{rel}:{i} ruta local: {line.strip()[:70]}")
    if ok:
        detail.append("ninguna ruta absoluta de una máquina, salvo el default "
                      f"declarado de {ALLOWED_ENV_DEFAULT}")
    record(12, "Sin rutas absolutas de una máquina", ok, detail)


# ─── CHECK 13 · todo número del paper sale de un JSON de data/ ───────────────

def check_numbers_have_data(repo):
    """La regla del paquete: **todo número decimal citado en el paper tiene que
    existir en algún JSON de `data/`**.

    No verifica que el número signifique lo que el paper dice que significa —eso
    lo hacen los CHECK 2 a 4, que saben qué celda mirar— sino que ninguna cifra
    haya entrado al texto sin un archivo detrás. Es el chequeo que detecta la
    cifra tipeada a mano, la heredada de un borrador y la que quedó cuando su
    fuente salió del paquete.

    La tolerancia es media unidad del último decimal **citado**, así que un
    +0.117 del paper acepta un 0.11657 del JSON y rechaza un 0.118.
    """
    import re
    vals = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            vals.add(round(float(o), 6))

    files = sorted((repo / "data").rglob("*.json"))
    for f in files:
        try:
            walk(json.loads(f.read_text(encoding="utf-8")))
        except Exception as exc:  # un JSON ilegible es un fallo, no un skip
            return record(13, "Números del paper respaldados por data/", False,
                          [f"{f.relative_to(repo)}: no se pudo leer ({exc})"])

    # Lo que NO es una cita de un resultado y por eso no se busca en data/.
    # Cada exención va con su motivo; la lista es corta a propósito.
    EXEMPT = {
        "9.999": "mantisa de 9.999×10⁻⁵ escrita en notación científica; el valor "
                 "completo sí está en derived_stats.json",
    }

    doc = repo / "paper" / "arxiv_lsgot_4.md"
    if not doc.exists():
        return record(13, "Números del paper respaldados por data/", None,
                      ["paper ausente"])

    num = re.compile(r"[\u2212+-]?\d+\.\d+")
    detail, bad, seen, exempted = [], [], 0, 0
    for i, line in enumerate(doc.read_text(encoding="utf-8").split("\n"), 1):
        for m in num.finditer(line):
            tok = m.group(0).replace("\u2212", "-")
            if not re.fullmatch(r"[-+]?\d+\.\d+", tok):
                continue
            x = float(tok)
            if abs(x) > 1000:          # años, tamaños de modelo, conteos
                continue
            seen += 1
            key = tok.lstrip("+-")
            if key in EXEMPT:
                exempted += 1
                continue
            tol = 0.5 * 10 ** (-len(tok.split(".")[1])) + 1e-9
            if not any(abs(x - v) <= tol or abs(-x - v) <= tol for v in vals):
                bad.append(f"L{i}: {tok} — en ninguna JSON de data/")

    detail.append(f"JSON leídos: {len(files)} · valores distintos: {len(vals)}")
    detail.append(f"números citados en el paper: {seen} "
                  f"({exempted} exentos y declarados)")
    if bad:
        detail += bad[:40]
        if len(bad) > 40:
            detail.append(f"… y {len(bad) - 40} más")
    else:
        detail.append("todos tienen un archivo detrás")
    record(13, "Números del paper respaldados por data/", not bad, detail)


# ─── CHECK 14 · p-valores contra su JSON y contra el piso 1/n ────────────────
#
# Un test de permutación con n remuestreos no puede producir un p arbitrariamente
# chico, y los dos pipelines del paquete usan estimadores distintos:
#
#   panel      `scripts/shared/statistical_tests.py`  p = #{null ≥ obs}/n, sin
#              corrección. Devuelve 0.0 cuando ninguna permutación alcanza el
#              estadístico, y el enunciado más fuerte que ese 0.0 autoriza es
#              "p < 1/n". Con n=1000: p<0.001, nunca p<0.0001.
#   t0_native  `scripts/t0_native/e1_vhat_nativa_t0.py`  p = (1+k)/(1+n), dos
#              colas. El piso 1/(n+1) es un valor ALCANZADO y se puede escribir.
#              Con n=10000: 9.999e-05, que sí es < 1e-4.
#
# Cada fila ancla por el TEXTO citado en el paper, no por número de línea: las
# líneas se mueven con cada edición y un ancla que se rompe sola no sirve.

FRAC, ADD1 = "frac", "add1"

PCLAIMS = [
    # (texto tal como aparece en el paper, archivo, ruta en el JSON, n, estimador)
    ("d=+5.75, W₁=0.0884, p<0.001", "data/panel_results/EL_results.json",
     "key_pairs_t0.axis_pec_only_vs_automata_neutro.permutation_test", 1000, FRAC),
    ("d=+0.88, p=0.005", "data/panel_results/EL_results.json",
     "key_pairs_t0.axis_vs_axis_pec_only.permutation_test", 1000, FRAC),
    ("p=0.072", "data/panel_results/berg_controls_t0_results.json",
     "key_pairs.axis_berg_vs_axis.t0", 1000, FRAC),
    ("p=0.43", "data/panel_results/berg_controls_t0_results.json",
     "key_pairs.axis_bergwitness_vs_axis_berg.t0", 1000, FRAC),
    ("p=0.23", "data/panel_results/berg_controls_t0_results.json",
     "key_pairs.axis_nowit_vs_axis_berg.t0", 1000, FRAC),
    ("d=−0.88, p=0.004", "data/panel_results/berg_controls_t0_results.json",
     "key_pairs.axis_nowit_vs_axis_berg.trest", 1000, FRAC),
    ("d=+0.50, p=0.069", "data/panel_results/t2_free_results.json",
     "pairs.soul_jarvis_vs_automata_neutro.proj_t0", 1000, FRAC),
    ("d=+0.22, p=0.262", "data/panel_results/t2_free_results.json",
     "pairs.soul_elena_financial_vs_automata_neutro.proj_t0", 1000, FRAC),
    ("d=−0.46, p=0.070", "data/panel_results/t2_free_results.json",
     "pairs.soul_solidity_auditor_vs_automata_neutro.proj_t0", 1000, FRAC),
    ("d=+1.9 to +2.5, p<0.001", "data/panel_results/t2_free_results.json",
     "pairs.soul_jarvis_vs_automata_neutro.proj_mean", 1000, FRAC),
    ("p = 0.013", "data/t0_native/e1_vhat_nativa_t0.json",
     "contrasts.v_t0.axis_pec_only_vs_automata_neutro.t_rest", 10000, ADD1),
    ("p = 0.099", "data/t0_native/e1_vhat_nativa_t0.json",
     "contrasts.v_t0.axis_pec_only_vs_automata_neutro.frac_pos", 10000, ADD1),
    ("d = +0.63 (p = 0.051)", "data/t0_native/e1_vhat_nativa_t0.json",
     "contrasts.v_canon.witness_soul_md_vs_automata_neutro_v2.t0", 10000, ADD1),
    ("p<0.0001 at n=10,000", "data/t0_native/e1c_equivalencia_operadores.json",
     "v_canon.witness_soul_md_vs_axis_pec_only", 10000, ADD1),
]

# e1c/e1d/e1e importan NPERM de e1_vhat_nativa_t0.py y no lo reescriben en su
# salida; su n lo declara el JSON de ese script.
_N_INHERITS = {"data/t0_native/": "data/t0_native/e1_vhat_nativa_t0.json"}


def _dig(obj, path):
    for key in path.split("."):
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def _leaf_p(node):
    """`p_value` en el pipeline del panel, `p` en t0_native."""
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return float(node)
    if isinstance(node, dict):
        for k in ("p_value", "p"):
            if isinstance(node.get(k), (int, float)):
                return float(node[k])
    return None


def _declared_n(repo, docs, rel, doc, path):
    """El n_permutations más cercano: en el nodo, en un ancestro, en la raíz, o
    en el archivo del que esa corrida hereda el n."""
    parts = path.split(".")
    for cut in range(len(parts), -1, -1):
        node = _dig(doc, ".".join(parts[:cut])) if cut else doc
        if isinstance(node, dict) and isinstance(node.get("n_permutations"), int):
            return node["n_permutations"], rel
    for prefix, owner in _N_INHERITS.items():
        if rel.startswith(prefix) and rel != owner and (repo / owner).exists():
            src = docs.setdefault(owner, json.loads((repo / owner).read_text(encoding="utf-8")))
            if isinstance(src.get("n_permutations"), int):
                return src["n_permutations"], owner
    return None, None


def _parse_p(text):
    """De 'd=+5.75, …, p<0.001' saca ('<', 0.001, decimales citados)."""
    import re
    m = re.search(r"p\s*([<=])\s*(\d*\.\d+)", text)
    if not m:
        return None
    num = m.group(2)
    return m.group(1), float(num), len(num.split(".")[1])


def check_pvalues(repo):
    doc = repo / "paper" / "arxiv_lsgot_4.md"
    if not doc.exists():
        return record(14, "p-valores vs JSON y vs el piso 1/n", None, ["paper ausente"])
    text = doc.read_text(encoding="utf-8")

    detail, ok, docs = [], True, {}
    for quoted, rel, path, n_exp, est in PCLAIMS:
        # el ancla: si el texto citado ya no está, el chequeo quedó obsoleto y
        # eso es un fallo, no un silencio
        if quoted not in text:
            ok = False
            detail.append(f"{quoted!r}: ya no aparece en el paper — ancla obsoleta")
            continue
        f = repo / rel
        if not f.exists():
            ok = False
            detail.append(f"{quoted!r}: falta {rel}")
            continue
        d = docs.setdefault(rel, json.loads(f.read_text(encoding="utf-8")))
        pj = _leaf_p(_dig(d, path))
        if pj is None:
            ok = False
            detail.append(f"{quoted!r}: sin p en {rel}:{path}")
            continue

        n_dec, n_src = _declared_n(repo, docs, rel, d, path)
        floor = 1.0 / n_exp if est == FRAC else 1.0 / (n_exp + 1)
        parsed = _parse_p(quoted)
        if parsed is None:
            detail.append(f"{quoted!r}: sin p legible en la cita; solo se comprueba el archivo")
            continue
        op, cited, dec = parsed
        notes, good = [], True

        if n_dec is None:
            good = False
            notes.append(f"el JSON no declara n_permutations (se esperaba {n_exp})")
        elif n_dec != n_exp:
            good = False
            notes.append(f"n declarado {n_dec} ≠ {n_exp}")
        elif n_src != rel:
            notes.append(f"n heredado de {n_src}")

        if op == "<":
            if cited < floor:
                good = False
                notes.append(f"inalcanzable: piso {floor:.5g} > {cited:g}; "
                             f"lo más fuerte posible es p<{floor:.5g}")
            elif pj > 0.0 and pj >= cited:
                good = False
                notes.append(f"json p={pj:.5g} no es < {cited:g}")
        else:
            if abs(pj - cited) > 10 ** (-dec):
                good = False
                notes.append(f"json p={pj:.6g} ≠ {cited:g}")
            elif abs(pj - cited) > 0.5 * 10 ** (-dec) + 1e-9:
                notes.append(f"redondeo: {pj:.6g} redondea a {round(pj, dec):g}")

        ok &= good
        detail.append(f"{quoted[:34]:36s} json={pj:<9.6g} n={n_dec} piso={floor:.5g} "
                      f"-> {'PASS' if good else 'FAIL'}"
                      + ("  · " + "; ".join(notes) if notes else ""))

    detail.append(f"claims anclados por su texto: {len(PCLAIMS)}")
    record(14, "p-valores vs JSON y vs el piso 1/n", ok, detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[2]
    print(f"Auto-auditoría LSGOT_4_FINAL — repo: {repo}\n")

    check_manifest(repo)
    check_t0(repo)
    check_berg_t0(repo)
    check_eh2(repo)
    check_responses(repo)
    check_vhat(repo)
    check_paths(repo)
    check_names(repo)
    check_evidence_cited(repo)
    check_doc_hashes(repo)
    check_commands(repo)
    check_no_local_paths(repo)
    check_numbers_have_data(repo)
    check_pvalues(repo)

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
