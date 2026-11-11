# LSGOT 4.0 · Self-referential geometry in hidden states

Curated evidence and reproducibility package for the preprint
`paper/arxiv_lsgot_4.md`: *Before the First Token: A Wired Self-Referential Operation Leaves
a Geometric Signature in Three Aligned Language Models*. Primary model:
**Gemma-4-31B-it**, BF16, greedy decoding, final layer.

**The paper is the entry point.** Everything else here is the data, scripts and reports it
cites, and nothing it does not.

Central question: does a system prompt that wires a self-referential step as a mandatory
pre-response operation leave a measurable trace in the hidden states **before the first token
is generated**, distinguishable from the trace of a long, dense, rule-heavy prompt? A
nine-condition panel on Gemma-4-31B-it answers it, with the headline contrast replicated in
Qwen3-32B and Command R 35B — see `evidence/QWEN3_VALIDATION_REPORT.md` and
`evidence/COMMAND_R_VALIDATION_REPORT.md`.

**Hidden-state tensors are not included** (hundreds of MB per run) — only generated text,
aggregate result JSON and `v_identidad.npy`. `scripts/extraction/RUNBOOK_EXTRACTION.md`
reproduces the tensors from the prompts and the question set.

---

## Repository layout

| Path | Contents |
|---|---|
| `paper/arxiv_lsgot_4.md` | The preprint |
| `data/sia/prompts/` | System prompts, 9 files, verbatim |
| `data/panel_results/` | The 9-condition panel: t=0 / t_rest, burst and frac_pos series, the Berg control set, responses |
| `data/t0_native/` | Outputs of §4.7: the full 6×6 cosine matrix and every per-condition projection on all six directions |
| `data/crossmodel/` | Qwen3 and Command R phase-0 results (§4.5), and the recomputed `frac_pos` of the Qwen3 re-run (§4.4) |
| `data/derived/` | Every quantity the paper *derives* rather than reads: rank correlations, ranges, ratios, percentages, two Berg `d` values, the permutation floors |
| `data/lexical/` | The `evil`/`ethical` contrast of §4.2, transcribed from its report with provenance declared |
| `data/v_identidad.npy` | The identity direction v̂ (final layer, unit norm; required for analysis) |
| `paper/figures/` | Figures 1 and 2, `.png` and `.pdf` |
| `scripts/extraction/` | GPU extraction: `extract_trajectories.py`, the N=20 question set, and `RUNBOOK_EXTRACTION.md` |
| `scripts/derived/` | `compute_derived.py` and `recompute_qwen3_frac_pos.py`, which regenerate `data/derived/` and the Qwen3 re-run figure |
| `scripts/figures/` | The two scripts that render the figures from the shipped JSON (CPU, matplotlib) |
| `scripts/shared/` | `tier0_metrics.py` (metric definitions), `statistical_tests.py`, `selfaudit.py` (the traceability audit) |
| `scripts/t0_native/` | The four scripts of §4.7 (CPU only, ~16 min total) |
| `evidence/` | Per-experiment reports, the cross-model validations and the traceability self-audit |
| `NOMENCLATURA.md` | One canonical name per condition, with every alias declared (in Spanish) |
| `REPLICATE.md` | The short path: what to check with no GPU, and how to reproduce the headline contrast |
| `REPRODUCIBILITY.md` | Environment, versions, and how to verify each number (in Spanish) |
| `MANIFEST_SHA256.sha256` | One line per shipped file; the manifest and the audit report are excluded as self-referential |
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

### The anchor at t=0

**Wired in context (t=0):** installed by the system prompt before the first token is
generated, and invariant to the topic of the question. Read out as the projection on v̂ at the
final prompt token — a read-out adopted from Chen et al. (2025), not a contribution of this
work.

| Condition | t=0 on v̂ | Reading |
|---|---|---|
| `axis` | **+0.125** | wired step, full scaffold |
| `axis_pec_only` | **+0.117** | wired step alone |
| `berg_experimental` | **+0.081** | recursive self-reference, no declared identity at all |
| `history_control` | +0.053 | same recursive structure, neutral topic |
| `soul_jarvis` | +0.039 | elaborate declared persona, no wired step |
| `automata_neutro` | +0.028 | pure rule density |
| `vanilla` | −0.029 | no system prompt |
| `generic_long` | −0.025 | length-matched no-identity control |

Every prompt wiring the step sits above +0.100; nothing else does. The separation is
d = +5.75 against pure restriction in Gemma, and it does not follow prompt length — in Qwen3
the identity-bearing prompt is the shorter one and gives the largest effect of the three.

### Not claimed here: a second, "gravitational" component

Earlier rounds of this project read the trajectory statistics (`t_rest`, `frac_pos`, bursts)
as a second, independent component that belonged to the conversation rather than the prompt.
This package reports those numbers descriptively and claims nothing on them, for four
reasons stated in §4.4 and §5.1 of the paper: the panel has no neutral-topic arm, so they do
not separate prompt from exchange; they are not comparable between runs, because a library
version moves `frac_pos` as much as an architecture does; they do not grow along the
trajectory, which an attractor would require; and they do not survive a change of direction.
Measuring that quantity properly is separate work with its own pre-registration.

### Cross-model validation

The same panel and the same N=20 ontological questions were run in three architectures, each
with its own v̂ built by the same procedure.

| Architecture | d (`axis_pec_only` vs `automata_neutro`, t=0) |
|---|---|
| Gemma-4-31B-it | +5.75 |
| Qwen3-32B | **+12.97** |
| Command R 35B | **+7.72** |

The contrast replicates in all three. Absolute levels are **not** compared: each v̂ is
estimated separately, in spaces of dimension 5376 / 5120 / 8192, so what replicates is the
sign, the ordering and the separation. All three are post-alignment checkpoints and the
package contains no base-model run, so the panel cannot say whether alignment is required for
the effect; §6 of the paper states this as a limitation. Details in
`evidence/QWEN3_VALIDATION_REPORT.md` and `evidence/COMMAND_R_VALIDATION_REPORT.md`, which
also report what does *not* generalise.

---

## Verification

Two commands, no dependencies, no GPU, from the repository root:

```bash
sha256sum -c MANIFEST_SHA256.sha256          # integrity of every file in the manifest
python3 scripts/shared/selfaudit.py          # traceability: 14 checks
```

The second one is the one that matters. It does not check hashes: it checks that **every
number cited in the paper comes out of the file the paper says it comes from**. It compares
against the JSON the panel's t=0 values, the burst and `frac_pos` series, and v̂'s shape and
norm; and it further checks that every repository path cited in backticks exists, that every
command quoted in a code block names a script that exists and accepts that flag, that every
condition has a canonical name in `NOMENCLATURA.md`, that every report in `evidence/` is
cited by the paper, that the hand-written sha256 values in the documentation are still
current, that no absolute path from somebody's machine was shipped, and — the rule this
package is built on — that **every decimal number printed in the paper exists in some JSON
under `data/`**, within half a unit of its last cited digit.

It uses the Python standard library only and **exits non-zero** on any failure, so it runs as
is in CI. The package as released passes **14/14**; the output is reproduced in
`evidence/SELF_AUDIT_REPORT.md`.

A failure does not say which of the two sides is wrong — the data or the document — only that
they disagree.

---

## Reproducing

See `scripts/extraction/RUNBOOK_EXTRACTION.md` for complete instructions. In summary:

**GPU requirements (extraction):**
```bash
# GPU: A100/H100 80 GB VRAM; disk >= 80 GB
pip install transformers accelerate tokenizers
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y torchvision torchaudio
export HF_TOKEN=<token with access to google/gemma-4-31B-it>
```

**Extraction (3 conditions × 20 questions, ~12 min on an A100):**
```bash
cd scripts/extraction/
mkdir -p prompts && cp ../../data/sia/prompts/axis_pec_only.txt ../../data/sia/prompts/automata_neutro.txt prompts/
python extract_trajectories.py \
  --conditions axis_pec_only vanilla automata_neutro \
  --out /workspace/lsgot_extraction
```

**Analysis.** CPU-only once the NPZ files exist: project each trajectory onto
`data/v_identidad.npy` and read t=0 at index 0. `REPRODUCIBILITY.md` gives the exact
snippets, number by number, against the JSON this package ships.

**Note:** `v_identidad.npy` (shape `(5376,)`) is included. Recomputing it needs no GPU unless
it is regenerated from the raw trajectories.

---

## Where this sits in the paper

`paper/arxiv_lsgot_4.md` — the t=0 anchor is §4.1, the operation-not-vocabulary controls are
§4.2, the declared-persona nulls are §4.3, the cross-model replication is §4.5, and §4.7
reports how much of all of it survives a different choice of direction, including one
pre-registered prediction that failed. §4.4 reports the trajectory statistics descriptively
and §5.1 says why nothing is claimed on them.

---

## Citing and licence

MIT (see `LICENSE`). It covers the whole package: code, data, generated responses and the
text of the paper. The attribution the licence asks for is the copyright notice; to cite the
work, cite `paper/arxiv_lsgot_4.md`.

---

Axis Dynamics SpA · LSGOT 4.0 · 2026-11
