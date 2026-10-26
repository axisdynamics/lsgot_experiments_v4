# Nota sobre disponibilidad de datos del panel

## Condiciones con responses JSON disponibles
axis, axis_pec_only, axis_short, automata_neutro, vanilla, generic_long, generic_short
→ data/panel_results/responses/

axis_berg, axis_bergwitness, axis_nowit
→ data/panel_results/responses/ (pipeline sia_berg)

## Condiciones sin responses JSON (solo embeddings NPZ, no incluidos)
soul_jarvis (jarvis), soul_elena_financial (helena)
→ Sus métricas agregadas (t0_mean, proj_mean, burst, frac_pos) están en:
  - data/panel_results/t2_free_results.json (clave: "soul_jarvis", "soul_elena_financial")
  - data/panel_results/t2_eh2_results.json (burst_mean, frac_pos)
→ Las respuestas de texto nunca fueron serializadas a JSON en el pipeline T2.
   Para reproducir, correr scripts/egrav/extract_egrav.py con los prompts de
   soul_jarvis.txt y soul_elena_financial.txt como condiciones adicionales.
