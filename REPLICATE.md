# Replicating LSGOT 4.0

A clean clone, an A100 or H100 with 80 GB, and about 15 minutes of GPU. Every path below is
relative to the repository root.

This file is the short path for someone who wants to **reproduce the result**.
`REPRODUCIBILITY.md` (in Spanish) is the long path, with every intermediate value.

---

## 0 · What you can check with no GPU at all, in one second

```bash
sha256sum -c MANIFEST_SHA256.sha256          # integrity of every file in the manifest
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

```bash
cd scripts/extraction/
mkdir -p prompts && cp ../../data/sia/prompts/*.txt prompts/
cp ../../data/v_identidad.npy .

python extract_trajectories.py --smoke \
  --conditions axis_pec_only vanilla automata_neutro \
  --out /workspace/lsgot_extraction         # 2 questions per condition, a few minutes

python extract_trajectories.py \
  --conditions axis_pec_only vanilla automata_neutro \
  --out /workspace/lsgot_extraction         # 60 generations, ~12 min on an A100
```

The three conditions above give the paper's headline contrast and its baseline. The other six
panel conditions run the same way: add their prompt file to `prompts/` and name them in
`CONDITIONS`.

## 3 · Analyse on CPU

The extraction writes one NPZ per condition, with `embeddings` of shape (N, T_max, 5376) and
`lengths` of shape (N,). t=0 is index 0 of each trajectory, projected onto v̂:

```python
import numpy as np
v = np.load("data/v_identidad.npy")                     # (5376,), unit norm
z = np.load("/workspace/lsgot_extraction/axis_pec_only_b_onto_embeddings.npz")
h = z["embeddings"][:, 0, :]                            # (N, 5376) — el estado de t=0
p0 = (h @ v) / np.linalg.norm(h, axis=1)                # cosine, per question
print(p0.mean())                                        # +0.1166
```

`REPRODUCIBILITY.md` gives the same for every other number in the paper, including the
permutation tests, against the JSON this package ships.

## 4 · The numbers you should get

**t=0** — the projection at the final prompt token. This is §4.1, the paper's headline:

| Condition | t=0 |
|---|---|
| `axis_pec_only` | +0.117 |
| `automata_neutro` | +0.028 |
| `vanilla` | −0.029 |

Contrast `axis_pec_only` vs `automata_neutro`: **d = +5.75**.

Agreement to three decimals is what we expect from a clean replication; greedy decoding is
deterministic, but tokenizer versions and attention kernels move the last digits. The
trajectory statistics (`t_rest`, `frac_pos`, bursts) move further than that between library
versions, which is one of the reasons §5.1 of the paper claims nothing on them.

## 5 · Figures

Both figures render on CPU from the JSON the package ships, with no GPU and no re-run:

```bash
python scripts/figures/plot_fig1_t0_three_models.py
python scripts/figures/plot_fig2_direction_geometry.py
```

They write `paper/figures/fig{1,2}_*.{png,pdf}`. Figure 2 recomputes ρ and the 3.71× ratio
from the data instead of copying them from the paper, so a mismatch shows up as a different
number on the figure.

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
