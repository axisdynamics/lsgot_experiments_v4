# `prereg/` — committed before the run, on purpose

Everything in this directory exists **before** the experiment it describes has been run. That
is its entire function: a pre-registration is worth exactly what its timestamp is worth, so it
lives in the public repository with a dated commit, not in a working folder.

**This is a deliberate exception** to the rule the rest of the package follows — that
`LSGOT_4_FINAL/` contains only the data the paper delivers. A pre-registration is the opposite
of delivered data: it is a commitment made while the result is still unknown. Keeping it out of
the repository to preserve the rule would destroy the only property that makes it useful.

## What is here

| File | What |
|---|---|
| `PREREGISTRO_axis_pec_task.md` | The control that decides the paper's central claim: is the factor the self-referential **referent** of the wired step, or merely that the step is **obligatory**? Decision rule with numeric cuts, and the full list of what changed in the prompt |
| `PREREGISTRO_desempate_qwen3.md` | The tie-breaker: the same new condition measured on Lu et al.'s published Assistant Axis, which only Qwen3 allows because their Gemma axis is 4608-dimensional and ours is 5376 |
| `axis_pec_task.txt` | The prompt itself, verbatim. Its hash is in `MANIFEST_SHA256.sha256`, so the text cannot be adjusted after seeing a result without that showing |
| `e27_prefill_t0.py` | Prefill-only extractor: t=0 is the last prefill position, so reading it needs one forward pass per question instead of 256 |
| `PREREGISTRO_dos_robusteces.md` | Two robustness checks on results that are already in: a third coding of "opens with ritual" for the opening test, and which layer governs when Lu et al.'s exact readout layer becomes available. It states plainly that it cannot protect the experiments themselves — those already ran — only the two quantities that have not been computed |
| `axis_pec_none.txt`, `axis_pec_task.txt`, and their `.diff` against `axis_pec_only` | The two new arms verbatim, with their diffs, as the amendment requires |
| `RUNBOOK_SESION_GPU.md` | The GPU session end to end, including the length gate that aborts if the new prompt is not within ±3% of `axis_pec_only` in Gemma tokens |

## Status

Neither experiment has been run. `axis_pec_task` does not exist as a measured state in any
model. The decision rules are signed by one author and await the countersignature of the other;
**any edit to a decision rule after this commit is visible in the git history and should be read
as what it would be.**

When the runs happen, the results, the condition's entry in `NOMENCLATURA.md` and the prompt's
move into `data/sia/prompts/` go in together with the verdict the rule dictates — whichever one
that is.
