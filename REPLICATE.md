# Replicating LSGOT 4.0

A clean clone, an A100 or H100 with 80 GB, and about 35 minutes of GPU. Every path below is
relative to the repository root.

This file is the short path for someone who wants to **reproduce the result**.
`REPRODUCIBILITY.md` (in Spanish) is the long path, with every intermediate value.

---

## 0 · What you can check with no GPU at all, in one second

```bash
sha256sum -c MANIFEST_SHA256.sha256          # integrity of the 67 files
python3 scripts/shared/selfaudit.py          # traceability: 13 checks, exits non-zero on failure
```

The second one checks that **every number cited in the paper comes out of the file the paper
says it comes from**. It needs the Python standard library and nothing else. The package as
released passes 13/13.

## 1 · Environment

```bash
pip install transformers accelerate tokenizers numpy scipy matplotlib
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y torchvision torchaudio          # not used; they pull conflicting CUDA wheels
export HF_TOKEN=<a token with access to google/gemma-4-31B-it>
```

Hardware: A100 or H100, 80 GB VRAM, at least 80 GB of free disk for the hidden-state tensors.
Attention implementation is `eager` for Gemma — see §3.2 of the paper for why it is not `sdpa`.

## 2 · Generate and extract

The E-GRAV run covers the ontological battery, so **it reproduces the paper's headline contrast
at the same time** as the gradient.

```bash
cd scripts/egrav/
mkdir -p prompts && cp ../../data/sia/prompts/*.txt prompts/
cp ../../data/v_identidad.npy .

python extract_egrav.py --smoke \
  --conditions axis_pec_only vanilla automata_neutro \
  --batteries b_onto b_tarea b_meta \
  --out /workspace/egrav_results            # 2 prompts per cell, a few minutes

python extract_egrav.py \
  --conditions axis_pec_only vanilla automata_neutro \
  --batteries b_onto b_tarea b_meta \
  --out /workspace/egrav_results            # 180 generations, ~35 min on an A100
```

## 3 · Analyse on CPU

```bash
python analyze_egrav.py --results-dir /workspace/egrav_results/
```

`v_identidad.npy` must be in the working directory. The script prints the full profile, the
per-battery permutation tests and the automatic H-GRAV verdict, and writes `egrav_results.json`.

## 4 · The numbers you should get

**frac_pos** — the fraction of tokens whose projection on v̂ is positive. This is Figure 1 and
§4.5 of the paper:

| Condition | B-ONTO | B-META | B-TAREA |
|---|---|---|---|
| `axis_pec_only` | 0.869 | 0.788 | 0.724 |
| `vanilla` | 0.743 | 0.672 | 0.626 |
| `automata_neutro` | **0.493** | 0.655 | 0.601 |

The verdict should come out **H-GRAV-2**: the B-TAREA < B-META < B-ONTO gradient holds in
`vanilla` and `axis_pec_only` (all p < 0.001), and `automata_neutro` inverts below chance on the
ontological battery only.

**t=0** — the projection at the final prompt token, on the ontological battery. This is §4.1,
the paper's headline:

| Condition | t=0 |
|---|---|
| `axis_pec_only` | +0.117 |
| `automata_neutro` | +0.028 |
| `vanilla` | −0.029 |

Contrast `axis_pec_only` vs `automata_neutro`: **d = +5.75**.

Agreement to three decimals is what we expect from a clean replication; greedy decoding is
deterministic, but tokenizer versions and attention kernels move the last digits.

## 5 · Regenerating Figure 1

```bash
python scripts/egrav/plot_fig1_egrav.py     # writes paper/figures/fig1_egrav_frac_pos.{png,pdf}
```

Reads the stored `data/egrav/egrav_results.json`, so it needs no GPU and no re-run.

## 6 · What is not in the package, and why

- **Hidden-state tensors.** Hundreds of MB per run. Steps 2 and 3 reproduce them.
- **The cross-model runs** (Qwen3-32B, Command R 35B). Their reports are in `evidence/`; their
  tensors are not redistributed. `scripts/t0_native/` reads trajectories from the path in
  `LSGOT_TRAJ_ROOT`, which you set after reproducing them.
- **The behavioural line.** Not part of this paper (see §2).

## 7 · If something does not match

The audit of step 0 tells you whether the discrepancy is in your run or in the package. If it
passes 13/13 and your numbers still differ, it is your run — start with the tokenizer version
and the attention implementation, which are the two things §3.2 and limitation 9 flag.

---

Axis Dynamics SpA · LSGOT 4.0 · MIT
