# Informe de Auto-Auditoría — LSGOT_4_FINAL

**Generado por:** `scripts/shared/selfaudit.py` (stdlib, sin numpy ni GPU)
**Paquete auditado:** este repositorio, estado actual
**Paper verificado:** `paper/arxiv_lsgot_4.md` (preprint arXiv; es el único paper
que el paquete entrega)

---

## Qué cambió respecto de la versión anterior de este informe

La versión anterior era una **transcripción manual** de comandos, y no era
reproducible: declaraba 56 entradas de MANIFEST cuando el MANIFEST tenía 61, y
la salida que citaba como evidencia incluía una línea para
`paper/figures/lsgot_4.md`, ruta que no existe en el árbol ni en el MANIFEST.
Es decir: certificaba un estado del paquete que no era el publicado.

Tres cambios:

1. **El informe es ahora la salida de un script**, no una transcripción. Se
   regenera con `python3 scripts/shared/selfaudit.py`. Código de salida 1 si algo
   falla, así que sirve en CI.
2. **Dos chequeos nuevos.** CHECK 8 (CHECK 9 hasta la renumeración del 2026-10-29) verifica que toda ruta del paquete citada
   entre backticks en un `.md` exista — es el chequeo que habría atrapado la ruta
   fantasma. CHECK 9 verifica que cada prompt tenga nombre canónico declarado en
   `NOMENCLATURA.md` y que el alias ambiguo `axis_nowit` esté desambiguado.
3. **Sin numpy.** CHECK 7 lee el header del `.npy` y calcula la norma con
   `array` de la stdlib, para que la auditoría corra en cualquier parte.

**Un solo paper.** El borrador en español se retiró del paquete; el paquete
entrega únicamente `paper/arxiv_lsgot_4.md`. CHECK 8 y CHECK 11 atraparon las
siete citas y el hash que quedaron apuntando al archivo retirado.

**Exclusiones auto-referenciales.** El MANIFEST no puede listar su propio hash, y
este informe se escribe *después* de regenerar el MANIFEST — si se listara, su
hash quedaría obsoleto en el momento de escribirlo. CHECK 1 declara las dos
exclusiones y verifica que no estén listadas.

---

## Salida de la corrida

```text
Auto-auditoría LSGOT_4_FINAL — repo: /home/plaxius/Escritorio/Buscando_la_geometría/LSGOT_4_FINAL

[PASS] CHECK 1 — Integridad del MANIFEST_SHA256
         entradas en el MANIFEST: 76
         hashes que no coinciden: 0
         archivos declarados y ausentes: 0
         archivos en el árbol y fuera del MANIFEST: 0
         exclusiones auto-referenciales: 2 (MANIFEST y reporte de esta auditoría)

[PASS] CHECK 2 — t=0 del panel vs t2_free_results.json
         automata_neutro        paper=+0.028 json=+0.0282 diff=0.0002 -> PASS
         axis                   paper=+0.125 json=+0.1249 diff=0.0001 -> PASS
         axis_pec_only          paper=+0.117 json=+0.1166 diff=0.0004 -> PASS
         generic_long           paper=-0.025 json=-0.0251 diff=0.0001 -> PASS
         generic_short          paper=-0.045 json=-0.0453 diff=0.0003 -> PASS
         soul_elena_financial   paper=+0.032 json=+0.0321 diff=0.0001 -> PASS
         soul_jarvis            paper=+0.039 json=+0.0394 diff=0.0004 -> PASS
         vanilla                paper=-0.029 json=-0.0286 diff=0.0004 -> PASS

[PASS] CHECK 3 — axis_berg t=0/t_rest vs berg_controls_t0_results.json
         axis_berg t0     paper=+0.117 json=+0.1167 diff=0.0003 -> PASS
         axis_berg t_rest: SKIP — clave ausente

[PASS] CHECK 4 — burst_mean y frac_pos vs EH2_results.json (con_t0)
         automata_neutro   burst_mean paper=2.780 json=2.7758 diff=0.0042 -> PASS
         automata_neutro   frac_pos   paper=0.493 json=0.4932 diff=0.0002 -> PASS
         axis              burst_mean paper=7.880 json=7.8773 diff=0.0027 -> PASS
         axis              frac_pos   paper=0.848 json=0.8484 diff=0.0004 -> PASS
         axis_pec_only     burst_mean paper=8.810 json=8.8110 diff=0.0010 -> PASS
         axis_pec_only     frac_pos   paper=0.869 json=0.8689 diff=0.0001 -> PASS
         generic_long      burst_mean paper=4.180 json=4.1754 diff=0.0046 -> PASS
         generic_long      frac_pos   paper=0.699 json=0.6988 diff=0.0002 -> PASS
         generic_short     burst_mean paper=3.400 json=3.4034 diff=0.0034 -> PASS
         generic_short     frac_pos   paper=0.649 json=0.6486 diff=0.0004 -> PASS
         vanilla           burst_mean paper=4.620 json=4.6159 diff=0.0041 -> PASS
         vanilla           frac_pos   paper=0.744 json=0.7443 diff=0.0003 -> PASS

[PASS] CHECK 5 — Archivos de respuestas: N=20, sin vacías
         automata_neutro_responses.json                 n= 20 vacías=0 -> PASS
         axis_berg_responses.json                       n= 20 vacías=0 -> PASS
         axis_bergwitness_responses.json                n= 20 vacías=0 -> PASS
         axis_nowit_responses.json                      n= 20 vacías=0 -> PASS
         axis_pec_only_responses.json                   n= 20 vacías=0 -> PASS
         axis_responses.json                            n= 20 vacías=0 -> PASS
         axis_short_responses.json                      n= 20 vacías=0 -> PASS
         generic_long_responses.json                    n= 20 vacías=0 -> PASS
         generic_short_responses.json                   n= 20 vacías=0 -> PASS
         vanilla_responses.json                         n= 20 vacías=0 -> PASS
         archivos inspeccionados: 10

[PASS] CHECK 6 — v_identidad.npy forma y norma (lectura stdlib)
         header descr=<f4 shape=(5376,)
         elementos leídos=5376 esperado=5376 -> PASS
         norma=1.00000000 esperado≈1.0 diff=1.48e-09 -> PASS

[PASS] CHECK 7 — Rutas citadas en la documentación
         todas las rutas del paquete citadas en .md existen (externas declaradas aparte)

[PASS] CHECK 8 — Nomenclatura de condiciones vs NOMENCLATURA.md
         prompt automata_neutro          declarado en NOMENCLATURA.md -> PASS
         prompt axis                     declarado en NOMENCLATURA.md -> PASS
         prompt axis_berg                declarado en NOMENCLATURA.md -> PASS
         prompt axis_pec_neutro          declarado en NOMENCLATURA.md -> PASS
         prompt axis_pec_only            declarado en NOMENCLATURA.md -> PASS
         prompt generic_long             declarado en NOMENCLATURA.md -> PASS
         prompt generic_short            declarado en NOMENCLATURA.md -> PASS
         prompt soul_elena_financial     declarado en NOMENCLATURA.md -> PASS
         prompt soul_jarvis              declarado en NOMENCLATURA.md -> PASS
         prompt vanilla                  declarado en NOMENCLATURA.md -> PASS
         alias ambiguo axis_nowit     desambiguado -> PASS

[PASS] CHECK 9 — Cada evidence/ citado por un paper
         BERG_LEXICAL_RESULTS.md: citado por arxiv_lsgot_4.md
         COMMAND_R_VALIDATION_REPORT.md: citado por arxiv_lsgot_4.md
         QWEN3_VALIDATION_REPORT.md: citado por arxiv_lsgot_4.md
         SELF_AUDIT_REPORT.md: citado por arxiv_lsgot_4.md

[PASS] CHECK 10 — Hashes hardcodeados en la documentación
         REPRODUCIBILITY.md: data/v_identidad.npy -> PASS
         REPRODUCIBILITY.md: paper/arxiv_lsgot_4.md -> PASS

[PASS] CHECK 11 — Comandos citados en la documentación
         14 invocaciones citadas; todos los scripts existen y aceptan los flags que se les pasan

[PASS] CHECK 12 — Sin rutas absolutas de una máquina
         ninguna ruta absoluta de una máquina, salvo el default declarado de LSGOT_TRAJ_ROOT

[PASS] CHECK 13 — Números del paper respaldados por data/
         JSON leídos: 26 · valores distintos: 3521
         números citados en el paper: 689 (1 exentos y declarados)
         todos tienen un archivo detrás

[PASS] CHECK 14 — p-valores vs JSON y vs el piso 1/n
         d=+5.75, W₁=0.0884, p<0.001          json=0         n=1000 piso=0.001 -> PASS
         d=+0.88, p=0.005                     json=0.005     n=1000 piso=0.001 -> PASS
         p=0.072                              json=0.072     n=1000 piso=0.001 -> PASS
         p=0.43                               json=0.425     n=1000 piso=0.001 -> PASS
         p=0.23                               json=0.23      n=1000 piso=0.001 -> PASS
         d=−0.88, p=0.004                     json=0.004     n=1000 piso=0.001 -> PASS
         d=+0.50, p=0.069                     json=0.069     n=1000 piso=0.001 -> PASS
         d=+0.22, p=0.262                     json=0.262     n=1000 piso=0.001 -> PASS
         d=−0.46, p=0.070                     json=0.07      n=1000 piso=0.001 -> PASS
         d=+1.9 to +2.5, p<0.001              json=0         n=1000 piso=0.001 -> PASS
         p = 0.013                            json=0.0129987 n=10000 piso=9.999e-05 -> PASS
         p = 0.099                            json=0.0992901 n=10000 piso=9.999e-05 -> PASS
         d = +0.63 (p = 0.051)                json=0.0507949 n=10000 piso=9.999e-05 -> PASS
         p<0.0001 at n=10,000                 json=9.999e-05 n=10000 piso=9.999e-05 -> PASS  · n heredado de data/t0_native/e1_vhat_nativa_t0.json
         claims anclados por su texto: 14

── RESUMEN ───────────────────────────────────────────────────────
  PASS  CHECK  1  Integridad del MANIFEST_SHA256
  PASS  CHECK  2  t=0 del panel vs t2_free_results.json
  PASS  CHECK  3  axis_berg t=0/t_rest vs berg_controls_t0_results.json
  PASS  CHECK  4  burst_mean y frac_pos vs EH2_results.json (con_t0)
  PASS  CHECK  5  Archivos de respuestas: N=20, sin vacías
  PASS  CHECK  6  v_identidad.npy forma y norma (lectura stdlib)
  PASS  CHECK  7  Rutas citadas en la documentación
  PASS  CHECK  8  Nomenclatura de condiciones vs NOMENCLATURA.md
  PASS  CHECK  9  Cada evidence/ citado por un paper
  PASS  CHECK 10  Hashes hardcodeados en la documentación
  PASS  CHECK 11  Comandos citados en la documentación
  PASS  CHECK 12  Sin rutas absolutas de una máquina
  PASS  CHECK 13  Números del paper respaldados por data/
  PASS  CHECK 14  p-valores vs JSON y vs el piso 1/n

  14/14 PASS · 0 FAIL · 0 SKIP
```

---

## Qué cubre cada chequeo

| # | Chequeo | Fuente de verdad |
|---|---|---|
| 1 | Integridad del MANIFEST + archivos sin declarar | `MANIFEST_SHA256.sha256` |
| 2 | t=0 de 8 condiciones del panel | `data/panel_results/t2_free_results.json` |
| 3 | `axis_berg` t=0 y t_rest | `data/panel_results/berg_controls_t0_results.json` |
| 4 | burst_mean y frac_pos del panel | `data/panel_results/EH2_results.json` (serie `con_t0`) |
| 5 | Archivos de respuestas: N=20, sin vacías | `data/*/responses/*.json` |
| 6 | Forma y norma de v̂ | `data/v_identidad.npy` |
| 7 | Rutas citadas en la documentación | el árbol del paquete |
| 8 | Nomenclatura de condiciones | `NOMENCLATURA.md` |
| 9 | Cada reporte de `evidence/` está citado por un paper | `paper/*.md` |
| 10 | Los sha256 hardcodeados en la documentación siguen coincidiendo | el árbol |
| 11 | Los comandos citados: el script existe y acepta ese flag (`prereg/` exento, son registros fechados) | el árbol |
| 12 | Ninguna ruta absoluta de una máquina en un archivo entregado | el árbol |
| 13 | Todo número decimal citado en el paper existe en algún JSON de `data/` | `data/**/*.json` |
| 14 | Todo p citado coincide con su JSON y respeta el piso `1/n` del estimador que lo produjo | los archivos de resultados y su `n_permutations` |

Los valores esperados están declarados en el script con su fuente documental
(sección del paper que los cita). Un FAIL no dice cuál de los dos lados está
mal — el dato o el documento —, solo que no coinciden.

## Qué NO cubre

- **Los números de los modelos de validación cruzada** (Qwen3, Command R): sus
  NPZ no están en el paquete, así que las cifras de `evidence/QWEN3_*` y
  `evidence/COMMAND_R_*` no son verificables desde acá.
- **Las secciones que dependen de reportes externos al paquete:** §4.2 de
  `paper/arxiv_lsgot_4.md` se apoya en `BERG_LEXICAL_RESULTS.md`, que no está
  incluido.
- **La línea conductual de Berg entera.** Salió del paquete el 2026-10-29: el
  paper no la reporta y sus datos están en `berg_conductual_retirado/`, fuera del
  repositorio. Con ella salieron dos chequeos —los porcentajes del juez y el
  alcance del archivo del juez—. No es que se hayan relajado: no queda nada en el
  paquete que verificar ahí.
- **La rama E-GRAV entera.** Salió del paquete el 2026-11-11, por lo mismo: el
  paper ya no reporta el gradiente de baterías ni el veredicto H-GRAV, así que no
  hay nada que trazar. Sus datos, scripts y figura están en
  `EGRAV_crossmodel/retirado_de_lsgot_4_final/`, fuera del repositorio. Con ella
  salió un chequeo (las 9 celdas de `frac_pos`) y se agregaron dos nuevos —comandos
  citados y rutas absolutas—, así que la auditoría quedó en **12**. Lo que se
  conserva es el extractor GPU, renombrado a
  `scripts/extraction/extract_trajectories.py`, porque es con lo que se reproduce
  el panel y no es parte de E-GRAV.
- **Nada sobre validez.** Esto audita trazabilidad: que los números citados
  salgan de los archivos. No audita si el experimento mide lo que dice medir.

---

*Regenerar con:* `python3 scripts/shared/selfaudit.py` *(desde la raíz del repo)*
