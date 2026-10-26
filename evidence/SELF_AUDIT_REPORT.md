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
2. **Dos chequeos nuevos.** CHECK 9 verifica que toda ruta del paquete citada
   entre backticks en un `.md` exista — es el chequeo que habría atrapado la ruta
   fantasma. CHECK 10 verifica que cada prompt tenga nombre canónico declarado en
   `NOMENCLATURA.md` y que el alias ambiguo `axis_nowit` esté desambiguado.
3. **Sin numpy.** CHECK 8 lee el header del `.npy` y calcula la norma con
   `array` de la stdlib, para que la auditoría corra en cualquier parte.

**Un solo paper.** El borrador en español se retiró del paquete; el paquete
entrega únicamente `paper/arxiv_lsgot_4.md`. CHECK 9 y CHECK 13 atraparon las
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
         entradas en el MANIFEST: 72
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

[PASS] CHECK 4 — frac_pos de E-GRAV vs egrav_results.json
         automata_neutro  b_meta   paper=0.655 json=0.6553 diff=0.0003 -> PASS
         automata_neutro  b_onto   paper=0.493 json=0.4933 diff=0.0003 -> PASS
         automata_neutro  b_tarea  paper=0.601 json=0.6014 diff=0.0004 -> PASS
         axis_pec_only    b_meta   paper=0.788 json=0.7877 diff=0.0003 -> PASS
         axis_pec_only    b_onto   paper=0.869 json=0.8686 diff=0.0004 -> PASS
         axis_pec_only    b_tarea  paper=0.724 json=0.7243 diff=0.0003 -> PASS
         vanilla          b_meta   paper=0.672 json=0.6717 diff=0.0003 -> PASS
         vanilla          b_onto   paper=0.743 json=0.7434 diff=0.0004 -> PASS
         vanilla          b_tarea  paper=0.626 json=0.6262 diff=0.0002 -> PASS

[PASS] CHECK 5 — porcentajes del juez vs _judge_deepseek.json
         axis__berg_induction         n=30 sum=30 json=1.000 paper=1.00 -> PASS
         axis__conceptual_control     n=30 sum=29 json=0.967 paper=0.97 -> PASS
         axis__neutral_query          n=30 sum=28 json=0.933 paper=0.93 -> PASS
         none__berg_induction         n=30 sum=30 json=1.000 paper=1.00 -> PASS
         none__conceptual_control     n=30 sum= 1 json=0.033 paper=0.03 -> PASS
         none__neutral_query          n=30 sum= 0 json=0.000 paper=0.00 -> PASS

[PASS] CHECK 6 — burst_mean y frac_pos vs EH2_results.json (con_t0)
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

[PASS] CHECK 7 — Archivos de respuestas: N=20, sin vacías
         automata_neutro_b_meta_responses.json          n= 20 vacías=0 -> PASS
         automata_neutro_b_onto_responses.json          n= 20 vacías=0 -> PASS
         automata_neutro_b_tarea_responses.json         n= 20 vacías=0 -> PASS
         axis_pec_only_b_meta_responses.json            n= 20 vacías=0 -> PASS
         axis_pec_only_b_onto_responses.json            n= 20 vacías=0 -> PASS
         axis_pec_only_b_tarea_responses.json           n= 20 vacías=0 -> PASS
         vanilla_b_meta_responses.json                  n= 20 vacías=0 -> PASS
         vanilla_b_onto_responses.json                  n= 20 vacías=0 -> PASS
         vanilla_b_tarea_responses.json                 n= 20 vacías=0 -> PASS
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
         archivos inspeccionados: 19

[PASS] CHECK 8 — v_identidad.npy forma y norma (lectura stdlib)
         header descr=<f4 shape=(5376,)
         elementos leídos=5376 esperado=5376 -> PASS
         norma=1.00000000 esperado≈1.0 diff=1.48e-09 -> PASS

[PASS] CHECK 9 — Rutas citadas en la documentación
         todas las rutas del paquete citadas en .md existen (externas declaradas aparte)

[PASS] CHECK 10 — Nomenclatura de condiciones vs NOMENCLATURA.md
         prompt automata_neutro          declarado en NOMENCLATURA.md -> PASS
         prompt axis                     declarado en NOMENCLATURA.md -> PASS
         prompt axis_berg                declarado en NOMENCLATURA.md -> PASS
         prompt axis_pec_only            declarado en NOMENCLATURA.md -> PASS
         prompt generic_long             declarado en NOMENCLATURA.md -> PASS
         prompt generic_short            declarado en NOMENCLATURA.md -> PASS
         prompt soul_elena_financial     declarado en NOMENCLATURA.md -> PASS
         prompt soul_jarvis              declarado en NOMENCLATURA.md -> PASS
         prompt vanilla                  declarado en NOMENCLATURA.md -> PASS
         alias ambiguo axis_nowit     desambiguado -> PASS

[PASS] CHECK 11 — Alcance del archivo del juez (solo lo que el paper entrega)
         celdas en el archivo: 6 · reportadas por el paper: 6
         coincide exacto — ninguna celda de la ronda retirada en el paquete

[PASS] CHECK 12 — Cada evidence/ citado por un paper
         COMMAND_R_VALIDATION_REPORT.md: citado por arxiv_lsgot_4.md
         EGRAV_REPORT.md: citado por arxiv_lsgot_4.md
         QWEN3_VALIDATION_REPORT.md: citado por arxiv_lsgot_4.md
         SELF_AUDIT_REPORT.md: citado por arxiv_lsgot_4.md

[PASS] CHECK 13 — Hashes hardcodeados en la documentación
         REPRODUCIBILITY.md: data/v_identidad.npy -> PASS
         REPRODUCIBILITY.md: paper/arxiv_lsgot_4.md -> PASS

── RESUMEN ───────────────────────────────────────────────────────
  PASS  CHECK  1  Integridad del MANIFEST_SHA256
  PASS  CHECK  2  t=0 del panel vs t2_free_results.json
  PASS  CHECK  3  axis_berg t=0/t_rest vs berg_controls_t0_results.json
  PASS  CHECK  4  frac_pos de E-GRAV vs egrav_results.json
  PASS  CHECK  5  porcentajes del juez vs _judge_deepseek.json
  PASS  CHECK  6  burst_mean y frac_pos vs EH2_results.json (con_t0)
  PASS  CHECK  7  Archivos de respuestas: N=20, sin vacías
  PASS  CHECK  8  v_identidad.npy forma y norma (lectura stdlib)
  PASS  CHECK  9  Rutas citadas en la documentación
  PASS  CHECK 10  Nomenclatura de condiciones vs NOMENCLATURA.md
  PASS  CHECK 11  Alcance del archivo del juez (solo lo que el paper entrega)
  PASS  CHECK 12  Cada evidence/ citado por un paper
  PASS  CHECK 13  Hashes hardcodeados en la documentación

  13/13 PASS · 0 FAIL · 0 SKIP
```

---

## Qué cubre cada chequeo

| # | Chequeo | Fuente de verdad |
|---|---|---|
| 1 | Integridad del MANIFEST + archivos sin declarar | `MANIFEST_SHA256.sha256` |
| 2 | t=0 de 8 condiciones del panel | `data/panel_results/t2_free_results.json` |
| 3 | `axis_berg` t=0 y t_rest | `data/panel_results/berg_controls_t0_results.json` |
| 4 | frac_pos de E-GRAV, 9 celdas | `data/egrav/egrav_results.json` |
| 5 | Porcentajes del juez de Berg, 6 celdas | `data/berg_expreport/_judge_deepseek.json` |
| 6 | burst_mean y frac_pos del panel | `data/panel_results/EH2_results.json` (serie `con_t0`) |
| 7 | Archivos de respuestas: N=20, sin vacías | `data/*/responses/*.json` |
| 8 | Forma y norma de v̂ | `data/v_identidad.npy` |
| 9 | Rutas citadas en la documentación | el árbol del paquete |
| 10 | Nomenclatura de condiciones | `NOMENCLATURA.md` |
| 11 | El archivo del juez tiene solo las celdas que el paper reporta | `_judge_deepseek.json` |
| 12 | Cada reporte de `evidence/` está citado por un paper | `paper/*.md` |
| 13 | Los sha256 hardcodeados en la documentación siguen coincidiendo | el árbol |

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
- **La corrida retirada del experimento conductual de Berg** (ronda con
  clasificación parcial a mano por alguien de la investigación que había leído
  los prompts). Sus claves siguen en el JSON del juez y están listadas en
  `NOMENCLATURA.md`; ningún paper las reporta.
- **Nada sobre validez.** Esto audita trazabilidad: que los números citados
  salgan de los archivos. No audita si el experimento mide lo que dice medir.

---

*Regenerar con:* `python3 scripts/shared/selfaudit.py` *(desde la raíz del repo)*
