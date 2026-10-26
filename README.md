# LSGOT 4.0 · Self-referential geometry in hidden states

Curated evidence and reproducibility package for the preprint
`paper/arxiv_lsgot_4.md`: *Wired and Gravitational: Two Components of Self-Referential
Geometry in the Hidden States of Aligned Language Models*. Primary model:
**Gemma-4-31B-it**, BF16, greedy decoding, final layer.

**The paper is the entry point.** Everything else here is the data, scripts and reports it
cites, and nothing it does not.

Central experiment: **E-GRAV** — is the projection on the identity vector v̂ a property of
any active generation, or is it specific to self-referential questions? 3 conditions ×
3 batteries × 20 prompts (180 generations). **E-GRAV was run only in Gemma-4-31B-it.** The
ontological battery (B-ONTO) was also replicated in Qwen3-32B and Command R 35B — see
`evidence/QWEN3_VALIDATION_REPORT.md` and `evidence/COMMAND_R_VALIDATION_REPORT.md`.

**Hidden-state tensors are not included** (hundreds of MB per run) — only generated text,
aggregate result JSON and `v_identidad.npy`. `scripts/egrav/RUNBOOK_EGRAV.md` reproduces the
tensors from the prompts and batteries.

---

## Repository layout

| Path | Contents |
|---|---|
| `paper/arxiv_lsgot_4.md` | The preprint |
| `data/sia/prompts/` | System prompts, 9 files, verbatim |
| `data/egrav/batteries/` | The 3 question batteries: `b_onto`, `b_tarea`, `b_meta` (20 prompts each) |
| `data/egrav/responses/` | Generated text, 9 JSON files (no tensors) |
| `data/egrav/egrav_results.json` | E-GRAV profiles, contrasts and verdict constants |
| `data/panel_results/` | The 9-condition panel: t=0 / t_rest, burst and frac_pos series, the Berg control set, responses |
| `data/berg_expreport/` | The 6 cells §4.7 reports and nothing else: the external judge's per-item verdicts and the matching response JSONL |
| `data/t0_native/` | Outputs of §4.9: the full 6×6 cosine matrix and every per-condition projection on all six directions |
| `data/v_identidad.npy` | The identity direction v̂ (final layer, unit norm; required for analysis) |
| `scripts/egrav/extract_egrav.py` | GPU extraction (pipeline identical to the original panel) |
| `scripts/egrav/analyze_egrav.py` | CPU analysis: t=0 / t_rest / frac_pos profile, H-GRAV verdict |
| `scripts/egrav/RUNBOOK_EGRAV.md` | Full reproduction instructions |
| `scripts/shared/` | `tier0_metrics.py` (metric definitions), `statistical_tests.py`, `selfaudit.py` (the traceability audit) |
| `scripts/t0_native/` | The four scripts of §4.9 (CPU only, ~16 min total) |
| `evidence/` | Per-experiment reports, the cross-model validations and the traceability self-audit |
| `NOMENCLATURA.md` | One canonical name per condition, with every alias declared (in Spanish) |
| `REPRODUCIBILITY.md` | Environment, versions, and how to verify each number (in Spanish) |
| `MANIFEST_SHA256.sha256` | 72 entries; the manifest and the audit report are excluded as self-referential |
| `LICENSE` | MIT |

Two files are in Spanish, as noted above. Everything a reader needs in order to use the
package — the paper, this README and the runbook — is in English.

---

## Experimental conditions

The six conditions the results below rest on. `NOMENCLATURA.md` declares all nine, with the
prompt file and data key for each, and resolves every alias.

| Condition | Wired identity | Restriction | Description |
|---|---|---|---|
| `axis_pec_only` | ✓ | − | Minimal wired witness step (PEC_TRIPLE); identity-pole reference |
| `vanilla` | − | − | No system prompt; baseline |
| `automata_neutro` | − | ✓ | Pure restriction: trigger→output rules, 3999 tokens, zero self-reference |
| `axis` | ✓ | ✓ | Reference condition used to construct v̂ |
| `generic_long` | − | − | No-identity control; v̂'s negative pole, length-matched |
| `generic_short` | − | − | Same, short version |

**`v_identidad.npy`:** computed as `mean(axis) − mean(generic_long)` over the final layer
(dim 5376), normalised to unit norm. Shape `(5376,)`.

---

## Main findings

### v̂ has two independent components

1. **Wired in context (t=0):** installed by the system prompt before the first token is
   generated; invariant to the topic of the question. Read out as the projection on v̂ at the
   final prompt token — a read-out adopted from Chen et al. (2025), not a contribution of
   this work.

2. **Gravitational during generation (t_rest / frac_pos):** proportional to the identity
   content of the question; varies with the battery. This is the signal E-GRAV measures.

### E-GRAV — frac_pos

`frac_pos` = fraction of tokens whose projection on v̂ is positive.

| Condition | B-ONTO | B-META | B-TAREA |
|---|---|---|---|
| `vanilla` | 0.743 | 0.672 | 0.626 |
| `axis_pec_only` | 0.869 | 0.788 | 0.724 |
| `automata_neutro` | **0.493** | 0.655 | 0.601 |

### Verdict: H-GRAV-2

The **B-TAREA < B-META < B-ONTO** gradient holds in `vanilla` and `axis_pec_only`,
consistent with v̂ tracking a graded self-referential context rather than a binary property
of "active generation".

### The `automata_neutro` dissociation

`automata_neutro` inverts the signal **on the ontological battery only**
(frac_pos = 0.493 < 0.5): it diverges from v̂ specifically when the questions are about the
model's own self, and converges towards v̂ on neutral tasks. Pure restriction without a wired
identity therefore produces a state *opposite* to the identity state under self-referential
questions — not merely an attenuated signal.

### Cross-model validation

B-ONTO was run in three architectures. **E-GRAV (B-TAREA, B-META) in Gemma-4-31B-it only.**

| Architecture | Alignment stage (public) | d (`axis_pec_only` vs `automata_neutro`, t=0) | frac_pos axis |
|---|---|---|---|
| Gemma-4-31B-it | SFT + RLHF | +5.75 | 0.869 |
| Qwen3-32B | DPO / RLVR | **+12.97** | **0.987** |
| Command R 35B | RLHF with a documented preference reward model | **+7.72** | 0.980 |

The contrast replicates in all three. All three are post-alignment checkpoints and the
package contains no base-model run, so the panel cannot say whether alignment is required
for the effect; §6 of the paper states this as a limitation. Note also that the largest
effect comes from the one substrate without a preference reward model. Details in
`evidence/QWEN3_VALIDATION_REPORT.md` and `evidence/COMMAND_R_VALIDATION_REPORT.md`, which
also report what does *not* generalise.

---

## Verification

Two commands, no dependencies, no GPU, from the repository root:

```bash
sha256sum -c MANIFEST_SHA256.sha256          # integrity of the 72 files
python3 scripts/shared/selfaudit.py          # traceability: 13 checks
```

The second one is the one that matters. It does not check hashes: it checks that **every
number cited in the paper comes out of the file the paper says it comes from**. It compares
against the JSON the panel's t=0 values, E-GRAV's `frac_pos`, the external judge's
percentages, the burst metrics, and v̂'s shape and norm; and it further checks that every
repository path cited in backticks exists, that every condition has a canonical name in
`NOMENCLATURA.md`, that the judge file contains only the cells the paper reports, that every
report in `evidence/` is cited by the paper, and that the hand-written sha256 values in the
documentation are still current.

It uses the Python standard library only and **exits non-zero** on any failure, so it runs as
is in CI. The package as released passes **13/13**; the output is reproduced in
`evidence/SELF_AUDIT_REPORT.md`.

A failure does not say which of the two sides is wrong — the data or the document — only that
they disagree.

---

## Reproducing

See `scripts/egrav/RUNBOOK_EGRAV.md` for complete instructions. In summary:

**GPU requirements (extraction):**
```bash
# GPU: A100/H100 80 GB VRAM; disk >= 80 GB
pip install transformers accelerate tokenizers
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y torchvision torchaudio
export HF_TOKEN=<token with access to google/gemma-4-31B-it>
```

**Full extraction (3 conditions × 3 batteries, ~35 min on an A100):**
```bash
cd scripts/egrav/
python extract_egrav.py \
  --conditions axis_pec_only vanilla automata_neutro \
  --batteries b_onto b_tarea b_meta \
  --output-dir /workspace/egrav_results
```

**CPU analysis (once the NPZ files are available):**
```bash
source /path/to/venv_lsgot/bin/activate
python analyze_egrav.py --results-dir /path/to/egrav_results/
```

The script prints the full profile (t=0, t_rest, frac_pos, bursts), the per-battery
permutation tests and the automatic H-GRAV verdict, and writes
`data/egrav/egrav_results.json`.

**Note:** `v_identidad.npy` (shape `(5376,)`) must be in the working directory or passed via
`--v-identidad`. It is included in this repository; recomputing it needs no GPU unless it is
regenerated from the raw trajectories.

---

## Where this sits in the paper

`paper/arxiv_lsgot_4.md` — E-GRAV is §4.5, *"E-GRAV: the gravity is graded, and restriction
blocks selectively"*, with the `frac_pos` gradient and the H-GRAV-2 verdict. It complements
the 9-condition panel by dissociating the static component of v̂ from the dynamic one and
characterising the ontological specificity of the gravitational signal. §4.9 reports how much
of this survives a different choice of direction, including one prediction that failed.

---

## Citing and licence

MIT (see `LICENSE`). It covers the whole package: code, data, generated responses and the
text of the paper. The attribution the licence asks for is the copyright notice; to cite the
work, cite `paper/arxiv_lsgot_4.md`.

---

Axis Dynamics SpA · LSGOT 4.0 · E-GRAV · 2026-10
