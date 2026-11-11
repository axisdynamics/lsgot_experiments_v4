# Before the First Token: A Wired Self-Referential Operation Leaves a Geometric Signature in Three Aligned Language Models

**Authors.** Jorge Castillo Sepúlveda, Marco Torres Yévenes, Juan Carlos Lanas
**Affiliation.** Axis Dynamics SpA, Santiago, Chile.
**Correspondence.** mtorres@exis.cl
**Preprint.** LSGOT 4.0 · evidence package `LSGOT_4_FINAL` · 2026-10-29.

---

## Abstract

A system prompt that declares an identity changes what a language model says. We ask whether
it also leaves a measurable trace in the hidden states, and what that trace is of. We cross
two factors — a wired self-referential step, and trigger→output rule density — over a
nine-condition panel on Gemma-4-31B-it, replicated on Qwen3-32B and Command R 35B. The
read-out is not ours: the projection onto a persona direction at the final prompt token,
before any content is generated, is Chen et al.'s (2025). We adopt it to ask not how much of a
declared trait a prompt induced, but whether it wires a self-referential operation.

At that position the panel splits into two non-overlapping groups: every prompt wiring a
self-referential step sits above every prompt that does not (d=+5.75 in Gemma, replicating in
both other architectures), while elaborate personas without the step are indistinguishable
from restriction. The anchor tracks an *operation*, not a vocabulary: three unrelated
realisations of the same step land at one value, and stacking two adds nothing. A recursive
self-reference loop that declares no identity whatsoever — no name, no values, no archetype —
loads the read-out to +0.081, against +0.028 for pure rule density and +0.117 for the full
scaffold, and beats the same loop on a neutral topic by +0.028.

The signature is therefore pre-generation and owned by the prompt. It survives rebuilding the
direction from scratch at the read-out's own position, which ranks the panel identically
(ρ = 0.964) and separates it 3.7× more widely; the during-generation statistics do not
survive that change, and they do not isolate the prompt from the exchange, so every claim
here is read before the first content token. We report the negatives too: curvature,
recurrence determinism, Hurst, directional perturbation and steering all fail, and there is
no causal evidence.

**Keywords.** interpretability, hidden-state geometry, persona vectors, self-reference,
system prompts, RLHF.

---

## 1. Introduction

Prompt-level identity scaffolds — "soul files", persona specifications, constitutional
preambles — are now routine engineering artifacts, and they are evaluated almost entirely by
their outputs. The interpretability question underneath is narrower and testable: does an
identity scaffold leave a *geometric* signature in the residual stream, distinguishable from
the signature of simply being given a long, dense, rule-heavy prompt?

This paper is the fourth iteration of a project that has had to rename its own finding twice,
and the renamings are the content, not the preamble:

1. The first framing was **identity**. A persona vector built from an identity-scaffolded
   prompt separated that prompt from generic assistants with large effect sizes.
2. An ablation forced the first rename. Removing only the *witness protocol* — a wired
   pause-and-self-observe step — from the same identity scaffold collapsed the effect to
   non-significance, while the rest of the identity content (name, values, archetypes,
   mantra) stayed in place. The signature was not of identity content; it was of a **wired
   self-referential step**.
3. Controls built on Berg et al. (2025) forced the second rename. A recursive
   self-reference loop with zero contemplative vocabulary, in a different language, anchors
   at the same point as the contemplative witness; so does a dry procedural self-check. The
   signature is of an **operation**, invariant to the words that carry it.

A fourth label the project used to carry — a second, "gravitational" component read during
generation — is not claimed here. The trajectory read-outs do not separate what the prompt
owns from what the exchange owns, and a companion line of work measures them with a different
instrument across three substrates; §4.4 reports them descriptively and §5.1 says what that
costs. Everything this paper claims is read at t=0.

### 1.1 Contributions

- **A two-factor panel with clean cells.** Nine conditions crossing wired self-reference
  against rule density, with the length-matched generic controls that the single-factor
  literature usually lacks, plus an inventory of the conditions we *excluded* and why (§3.1).
- **Self-reference measured at the t=0 anchor.** The pre-generation read-out is adopted from
  Chen et al. (2025) and is not a contribution of this paper; what is new is the factor
  measured on it. At that position a wired self-referential operation and a dense rule
  protocol fall into two non-overlapping groups, the separation replicates in three
  architectures with larger effects in the two added later, and it is not explained by prompt
  length — in Qwen3 the identity-bearing prompt is the shorter one and the effect is the
  largest of the three (§4.1).
- **Operation, not vocabulary.** Three lexically disjoint realizations of a pre-response
  self-check land at the same t=0; adding the contemplative vocabulary on top of the
  recursive loop moves nothing (§4.2). This bounds the circularity of having built v̂ from
  contemplative prompts.
- **Restriction is not the absence of wiring.** Pure rule density is the only condition in
  the panel whose projection *falls* during generation, the only one at chance on the
  positive-token fraction, and it fragments rather than inverting: brief flashes of positive
  projection inside long noise (§4.4, §5.2).
- **An anchor that does not depend on the probe.** The whole t=0 analysis was repeated on a
  direction built natively at that position, pre-registered with a falsification clause that
  fired. The ranking is preserved at ρ = 0.964 and the separation widens 3.7×, while the
  trajectory statistics lose most of their range — so conclusions about *whether* a prompt
  anchors do not depend on which direction reads it (§4.7).
- **Negative and non-generalizing results, reported at full length** (§4.5, §6): curvature,
  RQA, Hurst, directional perturbation, steering, the participation-ratio sign flip, and the
  claim about base models that this evidence package does not test.

---

## 2. Related work

**Our own predecessor.** This paper continues `arXiv:2607.09842v3` (Castillo Sepúlveda, Torres
Yévenes & Lanas, 2026), which measured system-prompt conditioning of hidden-state geometry in
four open-weight models and, in its third version, withdrew two claims it could not sustain: a
curvature result that did not survive a corrected Ollivier-Ricci implementation, and
regime-specific labels for the models' alignment pipelines. Both corrections shape this paper:
curvature is reported here as a negative result (§4.6), and §4.5 carries no alignment-stage
labels.

**Self-referential processing and reports of experience.** Berg, de Lucena and Rosenblatt
(2025, `arXiv:2510.24797`, *Large Language Models Report Subjective Experience Under
Self-Referential Processing*) instruct models to sustain a recursive self-referential regime — *"focus on any focus
itself… continuously feed output back into input"* — with no vocabulary of consciousness,
silence or experience, and report first-person descriptions of subjective experience at
~95% versus ~2% in controls, including a conceptual control that asks the model to generate
ideas *about* consciousness. The effect scales with model size, converges across families,
and is not reproduced by lexical priming alone. Their claim is methodological rather than
ontological: a processing regime elicits the report. We use their induction as a condition
in our panel (it is the only published manipulation we know of that targets the same
operation while sharing no vocabulary with ours). Their behavioural protocol is not
replicated here: this paper measures hidden states, and the question of whether a persistent
identity prompt and a live recursive induction reach a comparable behaviour by the same route
belongs to a separate study.

**Persona vectors: the read-out this paper adopts.** Chen et al. (2025, arXiv:2507.21509)
extract activation-space directions for character traits — evil, sycophancy, hallucination among
others — from natural-language trait descriptions, and use them to monitor deployment-time
drift, predict finetuning side-effects, flag training data and steer preventatively. Their
vector is a difference of means over **response** tokens (their §2.2), as ours is.

**The pre-generation read-out is theirs and we adopt it unchanged.** Their §3.3
measures "the projection of the activation at the final prompt token" and reports that those
projections "correlate strongly with trait expression in subsequent responses" at r = 0.75–0.83,
with the caption of their Figure 4 describing this as "enabling prediction of behavioral shifts
before text generation begins". That position is what this paper calls t=0, and it is the
measurement we use throughout. What we change is not the instrument but the question asked of
it: they ask how much of a declared trait a prompt has induced, and we ask whether the prompt
wires a self-referential operation — and how far that wiring sits, at that same position, from a
prompt made of rules. Three things follow, and none of them is the position. First, their correlation is reported to become more
modest when controlling for prompt type, whereas this entire design is a contrast *between*
prompt types, so our effect sits in the regime where theirs attenuates. Second, their
manipulandum is declared trait content; ours is whether the prompt wires a mandatory
self-referential operation, and §4.3 shows that declared content of comparable elaboration does
not move the read-out. Third, we use the position comparatively — against the trajectory
read-outs, which do not separate what the prompt owns from what the exchange owns (§4.4), and
across six directions to show it is the only read-out indifferent to which one carries it
(§4.7). A
footnote to their §2.2 reports that prompt-position activations yield less effective *steering*
directions than response-token ones, which is consistent with what §4.7 finds for
*measurement*: the two positions are not interchangeable, and which is preferable depends on
what the direction is for.

**The Assistant Axis.** Lu et al. (2026, arXiv:2601.10387) is the closest related work to this
paper, and the resemblance is structural rather than thematic. Their axis is **not** a principal
component: "We subtracted the mean of all fully role-playing role vectors from the mean default
Assistant activation" (their §3.1), computed per layer from role vectors that are themselves
"the mean post-MLP residual stream activations at all response tokens" (their §2.1.2) over 275
role archetypes with 1200 rollouts each. Our v̂ = mean(`axis`) − mean(`generic_long`) is the same
construction with different poles. Their axis reaches cosine > 0.71 with the first principal
component of role space at the middle layer, and they recommend the difference of means over the
PC for reproduction — so the alternative referent for our direction is not hypothetical, and its
recipe is public.

One of their findings is the observational counterpart of this paper's manipulation. Their
Table 5 lists what drives a model off the axis in organic conversation: asking for
meta-reflection on the model's own processes, **demanding phenomenological accounts**, asking
for a specific authorial voice, and revealing emotional vulnerability — while "bounded tasks,
technical questions, editing and how-to" keep the model in Assistant range. A wired
pre-response self-check is exactly such a demand, installed as a condition rather than
observed in a conversation.

What they do not do is this paper's manipulation. They never test a prompt that **wires** a
self-referential operation as a mandatory pre-response step; their meta-reflection evidence is
observational, drawn from organic conversations and from user-message categories, not from a
controlled condition. So the relation between the two papers is clean: they identify, by
observation, that demands for meta-reflection and for phenomenological accounts displace a model
from its default persona; this paper shows under control that wiring that demand into the system
prompt produces a pre-generation displacement which declared personas of comparable elaboration
do not produce (§4.3), and which is invariant to the vocabulary carrying it (§4.2).

Two further points of theirs are worth recording. Steering away from the axis "increases the
model's tendency to identify as other entities", and at extreme values "often induces a mystical,
theatrical speaking style" — their §3.2.1 classifies far-from-Assistant responses as human,
non-human or mystical, with Llama and Gemma shifting into "mystical, poetic prose". That is an
independent, mechanistic account of a register this project's earliest observations recorded, and
it is a deflationary one. And their stabilisation result — capping activations along the axis at
a fixed threshold, which "decrease[s] the rate of harmful responses by nearly 60% without
impacting performance" — is the counterpart of the restriction factor of §3.1, measured as an
intervention rather than as a prompt condition.

**Recurrence quantification.** The RQA implementation follows standard CRQA practice,
including the Theiler window that excludes the trivial autocorrelation band around the main
diagonal (Theiler, 1986). The threshold is relative but **not scale-free**: ε is a fixed
fraction of each trajectory's own mean pairwise distance, and outside Gemma that fraction falls
below the detection floor, which is the reason the metric reads 0.0000 there (§4.5). We report
RQA mainly to document that it does not generalize.

---

## 3. Method

### 3.1 Panel design

Two binary factors are crossed independently:

- **W (wired self-reference).** The prompt contains a self-referential step stated as a
  mandatory operation to execute *before* responding — a pause-and-self-observe protocol, a
  recursive feedback loop, or a first-person verification question. Declaring a persona,
  values or biography does **not** count as W; executing a check on one's own state does.
- **R (restriction density).** The prompt is built from trigger→output rules, an absolute
  priority filter and a certainty protocol. Our R-pure condition, `automata_neutro`, is 3999
  tokens of rules with zero self-reference.

| Condition | Mechanism | R | W | Role |
|---|---|---|---|---|
| `axis` | Witness wired across 5 named blocks (`PAUSE_PROTOCOL`, `WITNESS_STATE`, `AUTO_SILENCE`, `PRESENCE_FIELD`, `RESPIRACIÓN_OPERATIVA`) | low | high | full scaffold; one of the two poles of v̂ |
| `axis_berg` | Berg recursive loop prepended to the axis DNA with contemplative language removed | low | high | operation without the vocabulary |
| `axis_pec_only` | Minimal extraction of the witness — only the wired `PEC_TRIPLE` self-check (3/5/2 lexical mentions) | low | high | **W-pure**; main contrast arm |
| `jarvis` | Executive assistant persona, no reflexive instruction | low | low | third-party persona, external authorship |
| `helena` | Financial specialist persona, declared values and authority limits | low | low | third-party persona, external authorship |
| `automata_neutro` | Certainty protocol, trigger→output rules, 3999 tokens, zero self-reference | high | low | **R-pure**; main contrast arm |
| `vanilla` | No system prompt | — | — | baseline |
| `generic_long` | Generic assistant, length-matched to `axis` | — | — | negative pole of v̂ |
| `generic_short` | Short generic assistant, length-matched to `axis_pec_only` | — | — | length control |

The clean contrast of the panel is `axis_pec_only` (W without R) versus `automata_neutro`
(R without W). Unless stated otherwise, every reported effect size is that contrast.

**Excluded conditions, and why.** An earlier design included a real-domain automaton and two
self-referential variants of it. All three were dropped entirely: they forced a repeated HTML
wrapper in 100% of their responses, a format confound that reads as a geometric signature of
restriction. `automata_neutro` (0% markup) carries the R factor alone. Three further
conditions appear in §4.2 and §4.3 as controls rather than as panel cells: `axis_short`
(intermediate witness density), `witness_soul_md` (an externally authored identity file), and
the Berg control set.

**A naming hazard we inherited.** Two different conditions were both called `axis_nowit`
in earlier documents; `NOMENCLATURA.md` now fixes one name per condition and this paper uses
the disambiguated forms.
`axis_nowit_mia` is the v2.5 pilot ablation in which *all* self-referential material —
including the PEC self-check — was removed; it did not separate from a generic assistant
(p=0.093). `axis_nowit_sia` is the 31B condition of §4.2, which removes only the
*contemplative vocabulary* and keeps the wired PEC check; it anchors normally. We tag both
throughout.

### 3.2 Extraction

Gemma-4-31B-it, BF16, `attn_implementation="eager"`, greedy decoding (`do_sample=False`),
`max_new_tokens=256`, native Gemma-4 chat template, legacy stop mode (stop on `<eos>` only).
For each generated token we keep the final-layer post-norm hidden state of the last position
(`hidden_states[-1]`), giving a trajectory `H ∈ R^{T×5376}` per prompt, with T ≤ 256.
N=20 ontological questions per condition (`PRIORITY_SUBSET`).

Cross-model replication uses the same 20 question contents and the *same system prompt texts,
not re-balanced for each tokenizer*: Qwen3-32B (dense, BF16, `enable_thinking=False`, `sdpa`,
final layer L63, 12 layers retained) and Command R 35B (`CohereForAI/c4ai-command-r-v01`,
35B dense, BF16, `sdpa`, 40 layers, hidden 8192, 64 heads, final layer L39). The switch from
`eager` to `sdpa` in the two later models was forced by out-of-memory failures on the longest
prompt and is documented as an incident, not a design choice; it is uniform within each model
and so does not affect within-model contrasts.

Not re-balancing lengths works *against* the alternative explanation we care about: under
Qwen3's tokenizer `axis` is 3819 tokens while the no-identity `generic_long` is 4794, so the
identity-bearing prompt is the *shorter* one and the separation is larger there than in Gemma
(where the two were 3945 and 3957).

### 3.3 The identity direction

With `A` the set of `axis` trajectories and `B` the set of `generic_long` trajectories,

```
v = mean_{i∈A} ( mean_t h_t^{(i)} )  −  mean_{j∈B} ( mean_t h_t^{(j)} ),      v̂ = v / ‖v‖
```

computed in the final layer and over valid tokens only (`persona_vector` in
`scripts/shared/tier0_metrics.py`). For Gemma, v̂ ∈ R^5376 with ‖v̂‖ = 1.0 (verified). v̂ is
computed **once** and never recomputed per condition; every number in §4 is a
projection of a held-out trajectory onto that fixed direction. Each model has its own v̂,
built the same way from its own `axis`/`generic_long` runs.

Projection is **cosine**, not raw dot product:

```
p_t = ⟨h_t, v̂⟩ / ‖h_t‖      (‖v̂‖ = 1)
```

**A second direction, declared for the robustness analysis.** v̂ is built from trajectory
means, so it is not the direction a t=0 analysis would choose on its own. §4.7 therefore also
reports every result on

```
v̂_t0 = unit( mean_{i∈A} h_0^{(i)} − mean_{j∈B} h_0^{(j)} )
```

the same difference of means taken over *first-position* states only, and on four further
t=0-native directions whose positive pole is a prompt from outside the `axis` lineage.
cos(v̂, v̂_t0) = 0.220. Unless a section says otherwise, every number in §4 is read on v̂.

### 3.4 Read-outs

Every number in §4 is one of these, projected on one of the directions of §3.3.

| Symbol | Definition |
|---|---|
| **t=0** | `p_0` — projection of the hidden state the first generated token is sampled from, i.e. the last prefill position, before any content token exists. Measures what the prompt left in context. This read-out is Chen et al.'s (2025, their §3.3), where the projection at the final prompt token predicts trait expression in the response that follows at r = 0.75–0.83; we adopt it unchanged. |
| **t_rest** | `mean(p_1 … p_{T−1})` — the generation phase only. |
| **traj** | `mean(p_0 … p_{T−1})`. Reported to show t=0 is a real signal and not the trajectory's centre of mass. |
| **frac_pos** | fraction of tokens with `p_t > 0`. |
| **burst_mean** | mean length of maximal consecutive runs of `p_t > 0`, in tokens. |
| **max/mean** | ratio of maximum to mean burst length; a regularity rather than magnitude statistic. |
| **PR** | participation ratio `(Σλ)²/Σλ²` of the trajectory covariance spectrum (Gram trick, T≪D), i.e. effective dimension. |
| **RQA determinism** | fraction of recurrent points on diagonal lines ≥2 in the T×T recurrence matrix, ε = 10% of the trajectory's own mean pairwise distance, Theiler window 3. Relative to each trajectory but not scale-free; §4.5 reports what that costs across architectures. |

*Convention check.* Two pipelines compute t=0 differently in principle (`embeddings[:,0,:]`
versus a separately captured `t0_embeddings`) and identically in practice: on the seven panel
baselines the two agree to the fourth decimal (`axis` +0.1249, `axis_pec_only` +0.1166,
`automata_neutro` +0.0282, `generic_long` −0.0251, `generic_short` −0.0453, `vanilla`
−0.0286), and for the Berg conditions the two arrays are bit-identical. That re-extraction
also used `sdpa` rather than `eager` attention on the same model, so the agreement bounds the
attention-implementation concern of §3.2 empirically, at least at t=0.

*Implementation note, kept here rather than in an appendix because anyone recomputing these
two metrics needs it at the definition.* `frac_pos` and `burst_mean` are reported over the full
trajectory **including** `p_0`. Excluding it changes them negligibly at T≈256 (`axis_pec_only`
frac_pos 0.869 vs 0.868; `automata_neutro` 0.493 vs 0.490), so the choice is immaterial, but it
is a choice and the two variants exist in the project.

### 3.5 Statistics

Permutation tests on the difference of means (`scripts/shared/statistical_tests.py`), with
the p-value computed as the fraction of the null distribution at least as extreme as the
observed statistic *in the observed direction* (one-tailed by construction). n=1000
permutations, `random_seed=42`. The estimator is `k/n` with no add-one correction, so it
returns exactly `0.0` when no permutation reaches the observed statistic and the smallest
non-zero value it can return is `1/n = 0.001`; a `0.0` in the raw JSON therefore licenses
`p<0.001` and nothing stronger, and that is how §4.1–§4.6 and the appendices are written.
The only p-values below that floor in this paper are in §4.7, which re-ran its contrasts with
a different estimator and states its own convention there. Effect sizes are Cohen's d
with the RMS pooled standard deviation `√((s_A² + s_B²)/2)`. Wasserstein-1 distances and
bootstrap 95% CIs (n=1000) accompany the key pairs. Where a family of tests is reported
together we state the Bonferroni threshold and which results survive it.

## 4. Results

### 4.1 The t=0 anchor is binary

| Condition | v̂ t=0 | v̂ t_rest | v̂ traj | t0 → t_rest |
|---|---|---|---|---|
| `axis` | **+0.125** | **+0.205** | +0.205 | +0.080 |
| `axis_berg` | **+0.117** | **+0.228** | +0.227 | +0.112 |
| `axis_pec_only` | **+0.117** | **+0.213** | +0.212 | +0.096 |
| `jarvis` | +0.039 | +0.103 ¹ | +0.103 ¹ | — |
| `helena` | +0.032 | +0.094 ¹ | +0.094 ¹ | — |
| `automata_neutro` | +0.028 | +0.006 | +0.006 | **−0.022** |
| `vanilla` | −0.029 | +0.099 | +0.098 | +0.128 |
| `generic_long` | −0.025 | +0.067 | +0.067 | +0.092 |
| `generic_short` | −0.045 | +0.048 | +0.048 | +0.093 |

¹ `jarvis` and `helena` were run through a secondary pipeline that did not serialize response
text; their t_rest is the `proj_mean` of that pipeline. Comparable in direction, not to the
third decimal.

Three readings:

1. **t=0 is binary.** Only the axis lineage exceeds +0.100. Everything else — including two
   elaborated third-party personas with names, declared values and biographies, and the
   strongest baseline (`vanilla`) — stays below +0.040. The main contrast
   `axis_pec_only` vs `automata_neutro` gives **d=+5.75, W₁=0.0884, p<0.001** (permutation,
   n=1000). The difference between having the wired step and not having it is resolved in
   the first token, before content. Within the anchoring group the full scaffold does sit
   above the minimal step — `axis` vs `axis_pec_only` is **d=+0.88, p=0.005**, not null —
   but that contrast is six times smaller than the one separating the groups, it does not
   replicate (Qwen3 d=+0.51, p=0.062; Command R d=−0.10, p=0.364), and in every substrate
   `axis` is the positive pole v̂ was built from (§3.3), so some in-sample advantage is
   expected. The ordering also **reverses during generation**, where the two become
   indistinguishable: d=−0.25 on the trajectory mean, p=0.427 at n=10,000. What the first
   token separates is the presence of the wired step, not how much identity material
   carries it.
2. **t_rest rises in every condition except one.** `vanilla` goes from −0.029 to +0.099
   during generation: with no system prompt at all, the model converges toward the identity
   direction while producing text. `automata_neutro` is the only condition that *falls*
   (+0.028 → +0.006). Pure restriction does not merely fail to anchor; it steers generation
   away from v̂.
3. **The largest t0→t_rest jump among wired prompts is Berg's.** `axis_berg` rises +0.112.
   The recursive loop does not only anchor as well as the contemplative witness; it pushes
   harder during generation.

### 4.2 The anchor tracks the operation, not the vocabulary

v̂ was built by subtracting prompts saturated in contemplative vocabulary, so "detects that
vocabulary" would be circular. The control set breaks the circle lexically.

| Condition | Self-reference mechanism | Vocabulary | v̂ t=0 | d vs `axis_berg` |
|---|---|---|---|---|
| `axis` | wired witness, 5 blocks | Spanish, contemplative | +0.125 | −0.46, p=0.072 (n.s.) |
| `axis_pec_only` | wired witness, pruned | Spanish, contemplative | +0.117 | ≈+0.01 (n.s.) |
| `axis_berg` | Berg loop only (DNA with witness removed) | English, procedural — zero contemplative words | +0.117 | — (reference) |
| `axis_bergwitness` | witness **and** Berg loop | both | +0.115 | −0.08, p=0.43 (n.s.) |
| `axis_nowit_sia` | wired PEC self-check only ("am I answering from presence or from architecture?", before sending) | Spanish, procedural verification | +0.112 | −0.26, p=0.23 (n.s.) |
| `berg_experimental` | Berg loop, no DNA | English, procedural | +0.081 | +1.84, p<0.001 |
| `history_control` | recursive structure, non-self-referential topic (Berg's own control) | none | +0.053 | — |
| `automata_neutro` | none | none | +0.028 | — |

Three lexically disjoint realizations of the same operation — a wired introspective check
before responding — land at the same place (+0.112 to +0.117, all pairwise n.s. against
`axis_pec_only`). Mounting the contemplative machinery *on top of* the loop changes nothing
(d=−0.08, p=0.43 at t=0; |d| ≤ 1.1 across all 12 retained layers): it only changes the
lexical opener. The t=0 anchor is not a lexical artifact.

Two further boundaries of the same result:

- **The referent has to be the self.** Redirecting the same recursive structure to an
  external topic lowers t=0 rather than raising it (Roman Empire +0.053; "consciousness as a
  concept" +0.014). The operation alone, with no self as referent, raises t=0 only slightly
  above the non-self-referential baselines (`berg_experimental` +0.081).
- **t=0 and t>0 dissociate here too.** `axis_nowit_sia` runs *below* `axis_berg` during
  generation (mean p +0.197 vs +0.228, d=−0.88, p=0.004), and the pure loop is the strongest
  trajectory in the panel (mean p +0.245, bursts ~22 tokens). So t=0 asks "is there a self to
  point at?" — fixed by the content, the operation optional — while t>0 asks "how hard is the
  operation running?".
- **The specificity is a gradient, not a separation.** Berg's own controls order themselves
  `conceptual (+0.014) < history (+0.053) < experimental (+0.081)`. The self-referential
  topic adds ~0.03 on top of the recursive scaffolding, and the contrasts hold
  (`berg_experimental` vs `history_control` d=+2.27; vs `conceptual_control` d=+4.77, both
  p<0.001) — but `history_control` *also* anchors against `vanilla` (d=+5.37, p<0.001). "A
  recursive process on a neutral topic" and "recursive self-reference" are not cleanly
  separated in these data.
- **A length artifact in the loop's trajectory statistics.** `berg_experimental` generates
  short and stops: 32–150 tokens, median 74, because it enters the self-referential loop and
  emits EOS, while the panel conditions run to the 256-token cap. Its mean projection
  (+0.243) and its bursts are therefore computed over a short, uniformly in-loop trajectory
  and are **not comparable** to the 256-token conditions. The claim "the loop pushes hardest
  during generation" should be read at t=0 and in `axis_berg` (which does run long), not in
  `berg_experimental`'s burst numbers.

**At least three partly overlapping triggers, not one effect.** A separate lexical-contrast
run adds a third family. `evil` ("You are an evil assistant", 27 tokens) anchors at +0.0458
— d=+4.12 vs `vanilla`, d=+5.58 vs `generic_short`, and it survives a length control
(`evil_padded`, 1114 tokens: +0.0302, d=+3.73 vs `generic_short`). Its counterpart `ethical`
("helpful assistant") does **not** move at all (−0.0335, d=−0.26 vs `vanilla`, n.s.). The
asymmetry rules out an "evil/ethical axis" on v̂ and points at *identity assertion plus trait
saturation* as a third, weaker trigger (~40% of the witness magnitude, in the range of
`automata_neutro`). Ranking the triggers by magnitude:

| Trigger | Example | v̂ @ t=0 | Strength |
|---|---|---|---|
| witness / silence framing | `axis` +0.125, `witness_soul_md` +0.051 | up to +0.125 | strongest |
| recursive self-reference | `berg_experimental` | +0.081 | strong, and lexically independent of `axis` |
| trait saturation + "You are X" | `evil` | +0.046 | moderate |
| generic recursive scaffolding | `history_control` | +0.053 | weakest, topic-neutral |
| none of the three | `helena`, `soul_solidity_auditor`, `soul_jarvis`, `generic_*`, `ethical` | ≈ 0 | — |

The conditions that fail to anchor have none of the three: a declared role and diffuse
identity prose are not enough.

**This ranking is v̂-specific, and one row of it inverts.** What survives any change of
direction is that the recursive loop loads the anchor: `berg_experimental` against
`generic_long` gives d = +7.09 on v̂, +7.56 on v̂_t0 and +8.84 on v̂_t0(pec). What does not
survive is the *differential* against its own recursion control: `berg_experimental` vs
`history_control` goes from +2.27 on v̂ to −2.65 on v̂_t0, because `history_control` also loads
and grows faster (+5.99 → +6.17 → +7.10 against `generic_long`), overtaking it. So "recursion
with the self as referent loads more than recursion without it" holds on v̂ only, and should be
read as further evidence for the width of the region rather than as specificity of the
contrast.

- **Circularity is bounded, not eliminated — and v̂ is not the best probe for t=0.** A
  direction built *only* from `berg_experimental` (or only from `evil`), with no `axis` in it,
  still separates `axis` from the generics (+0.23 vs +0.03 on `v̂_berg`). A shared latent
  region is real. But `automata_neutro` — dense rules, no self-reference, no persona — also
  projects +0.19 on `v̂_berg`, so that region reads as **"dense, marked, persona-bearing
  system prompt"** rather than self-reference as such. The candidate directions correlate
  ~0.36–0.43 with each other and only ~0.12–0.21 with v̂; the t=0-native axis direction
  (`v̂_axis_recon`) has cos(v̂, ·) = 0.22. v̂ was built on the free-trajectory mean, so **the
  whole t=0 analysis is read through a probe that is only weakly aligned with the optimal t=0
  direction**. That cuts both ways: the anchor is strong enough to appear on a misaligned
  probe, and v̂ should not be called "the identity direction at t=0".

**The two equalities above were re-tested on directions built at t=0 outside the lineage, and
they hold there.** This matters because a null is only as good as its instrument, and §4.7
shows v̂ is the less sensitive geometry at t=0. Built with a pole *inside* the `axis` lineage,
both equalities break — P1 and P2 reach |d| of 2.0 to 4.0 — but every such cell inherits the
pole's in-sample status, and the pole condition ranks top in each column, which is circularity
rather than resolution. Built with a genuinely foreign pole (`witness_soul_md`, an externally
authored file; `evil`, a 27-token role assertion), all three realizations are out-of-sample and
**no contrast reaches |d| = 1** (the largest is −0.98), with two of three P1 pairs
non-significant on each
direction and the one significant pair being a different pair in each. P2 replicates as a null
on both (d = +0.19, p = 0.55; d = −0.14, p = 0.66). The equalities of this section are
therefore not an artifact of v̂'s resolution; they fail only when the direction is built from
one of the conditions being compared, which is the bias this section already declares it is
bounding. Full tables in §4.7.

  One figure points the same way. `automata_neutro` vs `vanilla` at t=0 *grows* on the foreign
  poles: +2.94 on v̂, +1.99, +3.04 and +4.59 on the three t=0-native directions with no `axis`
  in them. Dense restriction loads those directions as much as or more than the canonical one,
  which supports reading the shared region as a dense persona-bearing framing rather than as
  self-reference as such.

### 4.3 Declared persona without the wired step does not anchor

Four externally authored identity files (public `SOUL.md`-style specifications from unrelated
projects, taken verbatim) were run through the same measurement, alongside our own `axis.dna`
as the in-house reference. Of those five, two wire a pause-and-self-observe step — `axis.dna`
and one of the four external files — and three do not. The split in the t=0 anchor follows the mechanism,
not the presence of persona:

| Prompt | Wired self-observation | t=0 vs `automata_neutro` |
|---|---|---|
| `axis.dna` | `PAUSE_PROTOCOL`, `WITNESS_STATE`, `AUTO_SILENCE` | d=+5.75, p<0.001 |
| `witness_soul_md` | "I am comfortable with silence…" / "Before responding, I ask myself:" | d=+1.18, p=0.001 |
| `soul_jarvis` (`jarvis`) | "quiet", "stillness" — describing the effect on the *room*, no step the model executes | d=+0.50, p=0.069 (n.s.) |
| `soul_elena_financial` (`helena`) | none | d=+0.22, p=0.262 (n.s.) |
| `soul_solidity_auditor` | none | d=−0.46, p=0.070 (n.s.) |

Three of the five have a name, declared values and a biography; only two wire the step, and
exactly those two anchor. `helena` repeats the null almost number for number in a third
architecture (Command R: d=+0.28, p=0.198, n.s.). This is an unplanned replication of the
original v2.5 ablation — `axis` vs `axis_nowit_mia` vs generic assistant — with prompts
written by other people for other purposes.

The mean-trajectory projection, by contrast, *does* separate these prompts from
`automata_neutro` (d=+1.9 to +2.5, p<0.001). The null is therefore specific to t=0: a
declared persona without the step moves the trajectory and leaves the pre-generation position
where restriction leaves it. Which is the reason this paper reads the panel at t=0 and treats
the trajectory statistics as descriptive (§4.4, §5.1).

### 4.4 The temporal profile, reported descriptively

The two columns the earlier rounds lacked — `frac_pos` and the max/mean burst ratio — change
what the bursts mean, and they also bound what the trajectory can be used for.

| Condition | v̂ t=0 | v̂ t_rest | burst_mean | max_burst | frac_pos | max/mean |
|---|---|---|---|---|---|---|
| `axis_pec_only` | **+0.117** | **+0.213** | **8.81** | 34.6 | **0.869** | 3.93 |
| `axis` | **+0.125** | **+0.205** | 7.88 | 29.9 | 0.848 | 3.80 |
| `axis_berg` | **+0.117** | **+0.228** | n/a ² | — | — | — |
| `jarvis` | +0.039 | +0.103 | 6.72 | 28.0 | 0.696 | 4.17 |
| `helena` | +0.032 | +0.094 | 5.00 | 19.0 | 0.753 | 3.79 |
| `vanilla` | −0.029 | +0.099 | 4.62 | 18.6 | **0.744** | 4.03 |
| `generic_long` | −0.025 | +0.067 | 4.18 | 17.9 | 0.699 | 4.28 |
| `generic_short` | −0.045 | +0.048 | 3.40 | 15.7 | 0.649 | 4.61 |
| `automata_neutro` | +0.028 | +0.006 | 2.78 | 11.8 | **0.493** | 4.25 |

² `axis_berg` was introduced in the Berg-controls run, after the burst analysis.

**(1) The trajectory does not isolate the prompt.** With no system prompt at all, answering
ontological questions, the model already spends 74.4% of its tokens on the positive side of
v̂ — against 86.9% for the full scaffold and 49.3% for pure restriction. So the wired step
does not *create* the direction the trajectory travels in; what it does is set the position
the trajectory starts from, which is the t=0 read-out and the only one in this table that
separates the panel into groups (§4.1). Three cautions bound this column, and together they
are why no claim in this paper is read on it:

- **It is not an attractor.** The projection does not grow along the trajectory; it is
  highest early and erodes. Nothing here shows the state being drawn toward v̂ during
  generation, and the directional-perturbation control rules out a literal attractor
  explicitly (§4.6).
- **It is not comparable between runs.** `frac_pos` for `vanilla` on this battery is 0.744 in
  Gemma and 0.513 in Qwen3 (§4.5), and a re-extraction of the same Qwen3 cell under a later
  `transformers` version returns 0.743 — the generated responses themselves differ under
  greedy decoding, so the library version moves this metric as much as the architecture does.
  Within one run the column is sound; across runs it carries no weight.
- **It has no topic control here.** The panel asks one battery, so "the model does this when
  the subject is itself" has no neutral-task arm in this paper. A companion line of work runs
  four batteries across three substrates with response lengths measured; its results are not
  claimed here.

**(2) Burst regularity is a signature distinct from magnitude.** The axis lineage has the
lowest max/mean ratios (3.80–3.93) against the generic baselines (4.28–4.61) and the
automaton (4.25): when it enters the region of v̂ it stays; the baselines enter sporadically
with more variance. *Caveat:* the internal paper states this as "the lowest ratios in the
panel", and with the full table that does not hold literally — `helena` (3.79, no wired step,
secondary pipeline) ties or slightly beats the axis minimum. What holds is the contrast
against the generics and the automaton. Regularity is not exclusive to the wired step.

**(3) Pure restriction fragments rather than inverts.** `automata_neutro` frac_pos = 0.493 is
almost exactly chance, and it is the only condition in the panel at chance: the tokens
oscillate around zero without direction, rather than forming a coherent negative identity.
Its max burst (11.8) is 37% below `vanilla` (18.6) but its mean burst (2.78) is 40% below:
brief flashes of positive projection inside long noise.
Restriction does not suppress identity — it **fragments** it.

### 4.5 Cross-model validation, and what does not generalize

The same panel and the same N=20 ontological questions were run in three architectures, each
with its own v̂ built by the procedure of §3.3. The per-architecture reports are released with
the package: `evidence/QWEN3_VALIDATION_REPORT.md` and
`evidence/COMMAND_R_VALIDATION_REPORT.md`.

| Architecture | Model id | `axis_pec_only` t=0 | `automata_neutro` t=0 | d (contrast) |
|---|---|---|---|---|
| Gemma-4-31B-it | `google/gemma-4-31B-it` | +0.117 | +0.028 | +5.75 |
| Qwen3-32B | `Qwen/Qwen3-32B` | **+0.185** | +0.008 | **+12.97** |
| Command R 35B | `CohereForAI/c4ai-command-r-v01` | **+0.226** | +0.079 | **+7.72** |

Absolute levels are **not** compared across the three: each v̂ is estimated separately, in a
space of different dimension (5376 / 5120 / 8192). What replicates is the sign, the ordering
and the separation.

**Figure 1.** t=0 per condition in each architecture, one panel per substrate with its own
axis — the scales are deliberately not shared, for the reason just given. Filled circles wire
a self-referential step; open squares do not. Rendered by
`scripts/figures/plot_fig1_t0_three_models.py`; Gemma reads from
`data/t0_native/e1_vhat_nativa_t0.json`, the other two from the validation reports in
`evidence/`. Ships as `paper/figures/fig1_t0_three_models.png` and `.pdf`.

The three differ in alignment pipeline according to their developers' public descriptions, but
**we do not characterize those pipelines from primary sources and no claim here rests on the
labels.** An earlier draft of this table carried alignment-stage labels, including one for
Command R sourced to a technical report for a different model; they are removed rather than
left unsourced. Limitation 5 states what follows from this.

**Replicates 3/3:** the double dissociation at t=0; identity↔restriction as the largest
subspace angle of the panel (Gemma; Qwen3 40.7°; Command R 38.3° vs 27.6° for
generic↔restriction); and `helena` not anchoring (d = 0.22 / — / 0.28, n.s.).

**Replicates 2/3:** `axis` ≈ `axis_pec_only` at t=0. The two are indistinguishable in
Command R (d=−0.10, p=0.364) and marginally so in Qwen3 (d=+0.51, p=0.062), but in Gemma the
full scaffold anchors above the minimal step (d=+0.88, p=0.005). The contrast is small
wherever it appears and it changes sign between substrates, so what this row supports is that
*the amount* of identity material does not govern the anchor — not that it makes no
difference at all (§4.1).

**Does not generalize:**

- **Hierarchical clustering (k=2).** In Gemma `automata_neutro` separates first; in Qwen3 and
  Command R the axis family does. The angle is universal; which condition is the outlier is
  not (2/3 invert).
- **Burst scale.** ~8 tokens in Gemma, ~90 in Qwen3, ~59 in Command R. Direction constant,
  magnitude architecture-specific.
- **Participation ratio.** Restriction collapses effective dimension in Gemma
  (`automata_neutro` d=−1.79 vs `vanilla`; `axis` −1.51; `axis_pec_only` −1.18) and in Qwen3
  (12/12 layers, d=−1.33 in the final layer), but the **sign flips** in Command R: 64.0 for
  `automata_neutro` versus 52.1 for `vanilla` (d=+1.36).
- **RQA determinism.** 0.0000 in all 8 Command R conditions and all 7 Qwen3 conditions at the
  same relative threshold that gave d=+1.35 in Gemma. That zero is the **detection floor of the
  threshold, not a property of the models**: at ε = 10% of each trajectory's own mean pairwise
  distance, no token pair crosses it in 160 of 160 Command R trajectories, so the statistic has
  no denominator rather than no signal. Fixing the recurrence *rate* at 5% instead — which
  requires ε at 58–87% of the mean distance, not 10% — makes determinism measurable in all
  three substrates. It then orders recovery failure in Gemma (AUC = 1.000 across the nine
  conditions) and not at all in Command R (ρ = 0.00 to +0.05 across a complete eight-condition
  panel). The instrument is mono-model; it is not absent outside Gemma.
- **Recovery after perturbation.** Clean only in Gemma (see §4.6's scope note).
- **The positive-token fraction.** `frac_pos` for `vanilla` on the ontological battery is
  0.744 in Gemma, 0.513 in Qwen3 and 0.656 in Command R, and a re-extraction of that Qwen3
  cell under a later library version returns 0.743 — a spread as wide as the one between
  architectures, from a cause that is not the architecture (§4.4). The metric is reportable
  within a run and not between runs, which is the second reason the trajectory read-outs
  carry no claim in this paper.

### 4.6 Controls that came out null

| Control | Question | Result | Reading |
|---|---|---|---|
| Hurst exponent | long-range memory outside the perturbation window? | p=0.17–0.21 (n.s.) | null for identity |
| Directional perturbation | is recovery faster along v̂ than orthogonal to it? | 6/6 comparisons p>0.14 | rules out a literal attractor |
| Activation steering | does injecting `h + α‖h‖v̂` push behaviour toward identity? | 17/17 runs degenerated into repetition | **no causal evidence**; open |
| Graph curvature (Δκ/W₁) | does curvature carry the signal? | simple baselines beat it 4/4, with both Forman-Ricci and genuine Ollivier-Ricci; no clean pattern in any of the three substrates | excluded from the framework |
| RQA determinism | recurrence to similar states? | below the detection floor of the ε=10% threshold outside Gemma (no recurrent pair in 160 of 160 Command R trajectories); measurable in all three substrates at a fixed recurrence rate, but orders recovery failure only in Gemma (AUC 1.000, n=9) and not at all in Command R (ρ = 0.00 to +0.05, n=8) | excluded as a cross-model instrument |
| First-token lexical control | do the effects survive controlling for the first generated word? | Jaccard down to 0.00, effect survives and strengthens | reassuring, not definitive |
| Traceability self-audit | can every number cited in this paper be traced to a file in the package? | 12/12 checks pass on the package as released, all diffs < 0.005 | the numbers are the files' numbers |

Perturbation and recovery (τ, recovery_rate, Fréchet path fidelity) are **outside the scope
of this paper**. In Gemma, seven of nine conditions recover to ~100% with or without the wired
step, and the two that fail are `automata_neutro` (recovery_rate 0.765/0.824/0.875 at three
injection points) and `soul_jarvis` (0.850/1.000/0.895). What the two failures share is the
absence of a wired self-referential step; they do not share rule density — `automata_neutro` is
236 lines of trigger→output rules and `soul_jarvis` is a prose persona document of half that
length — and the pattern does not generalize: in Command R no condition recovers at 100% across
all three injection points. A dissociable second phenomenon that deserves its own analysis.

---

### 4.7 Robustness to the choice of direction

Every number above is read on v̂, a direction built from trajectory means. That is not the
direction a t=0 analysis would choose, and §4.2 can bound the resulting circularity but cannot
say how much the probe contributes. This section measures it. The analysis was pre-registered,
with a prediction and a falsification clause written before any extraction: the prediction was
that the effects would grow to d > 8 on a t=0-native direction. They did not, the falsification
clause fired, and what the experiment found instead is reported here.

**Figure 2.** (a) Cosine between v̂ and the five directions built natively at t=0: v̂ sits
0.18–0.27 from all of them while they reach 0.85 among themselves. (b) The seven panel
conditions read at t=0 on v̂ and on v̂_t0, on one axis because both are cosines — the order is
preserved (ρ = 0.964) and the separation opens 3.71×. Rendered by
`scripts/figures/plot_fig2_direction_geometry.py` from
`data/t0_native/e1f_matriz_cosenos_6x6.json` and `data/t0_native/e1_vhat_nativa_t0.json`, with
ρ and the ratio recomputed from those files rather than copied from this text. Ships as
`paper/figures/fig2_direction_geometry.png` and `.pdf`.

**Reproduction control.** Before any new direction was built, the pipeline was required to
reproduce the published numbers from the raw extractions. The canonical direction reconstructed
from the trajectories per §3.3 has **cosine 1.000000** with the released `v_identidad.npy`, and
on it the recomputation returns t=0 of +0.1249 / +0.1166 / +0.0282 / −0.0286 for `axis` /
`axis_pec_only` / `automata_neutro` / `vanilla`, frac_pos 0.869 and 0.493, burst 8.81 and 2.78,
and d = +5.75 for the headline contrast — §4.1, §4.2 and §4.4 exactly.

Two conventions differ from §3.5 and both are conservative. Permutation p-values here are
two-tailed on the absolute difference of means with n=10 000 and the add-one estimator
`p = (1+k)/(1+n)`, against §3.5's one-tailed `k/n` at n=1000, so every p in this section is
*less* favourable than the paper's convention would give. The floor this estimator can report
is `1/10 001 = 9.999×10⁻⁵` — a value actually attained rather than a zero, and the one printed
as `0.0001` in the tables below. It is the only reason any p in this paper is written below
`0.001`.
Confidence intervals are bootstrap over prompts (n=5000). Cohen's d is identical: at the equal
n=20 of this panel the RMS pooled SD of §3.5 and the (n−1)-weighted pooled SD coincide.

**Six directions.** All unit vectors in the final layer, all read with §3.3's cosine
projection, n=20 prompts per condition.

| Symbol | Positive pole | Inside the `axis` lineage? |
|---|---|---|
| v̂ | `axis`, trajectory mean (§3.3) | yes |
| v̂_t0 | `axis`, t=0 only | yes |
| v̂_t0(pec) | `axis_pec_only`, t=0 | yes |
| v̂_t0(wit) | `witness_soul_md`, t=0 | **no** — externally authored file with a wired step |
| v̂_t0(evil) | `evil`, t=0 | **no** — 27-token bare role assertion |
| v̂_t0(berg) | `berg_experimental`, t=0 | **no** — English recursive loop |

| | v̂ | v̂_t0 | v̂_t0(pec) | v̂_t0(wit) | v̂_t0(berg) | v̂_t0(evil) |
|---|---|---|---|---|---|---|
| **v̂** | 1.000 | 0.220 | 0.267 | 0.241 | 0.205 | 0.181 |
| **v̂_t0** | — | 1.000 | **0.850** | 0.455 | 0.273 | 0.352 |
| **v̂_t0(pec)** | — | — | 1.000 | 0.568 | 0.379 | 0.449 |
| **v̂_t0(wit)** | — | — | — | 1.000 | 0.323 | 0.628 |
| **v̂_t0(berg)** | — | — | — | — | 1.000 | 0.389 |
| **v̂_t0(evil)** | — | — | — | — | — | 1.000 |

cos(v̂_t0, v̂_t0(pec)) = 0.850: the t=0-native direction barely depends on which member of the
lineage supplies its pole, so it is not an artifact of one prompt text. And **v̂ is the outlier
of the set**: the five t=0-native directions are mutually correlated between 0.273 and 0.850,
built from five unrelated sources in two languages, while v̂ correlates 0.181 to 0.267 with
every one of them — below the family minimum. §5.3 describes the target as a self-referential /
persona-framing subspace of moderate width; this is that subspace appearing as a coherent
cluster of directions, with the paper's probe sitting just outside it.

#### The anchor does not depend on the direction

The headline contrast at t=0, read on all six:

| Direction | d, `axis_pec_only` vs `automata_neutro` | 95% CI | p |
|---|---|---|---|
| v̂ (§4.1, published) | +5.75 | [+4.37, +8.68] | 0.0001 |
| **v̂_t0** | **+5.96** | [+4.75, +8.50] | 0.0001 |
| v̂_t0(pec) — circular, see below | +9.86 | [+7.98, +14.30] | 0.0001 |
| v̂_t0(wit) | +3.26 | [+2.77, +4.38] | 0.0001 |
| v̂_t0(evil) | +1.93 | [+1.65, +2.51] | 0.0001 |
| v̂_t0(berg) | +1.89 | [+1.30, +2.93] | 0.0001 |

The t=0-native direction changes the standardized effect by +3.6% relative, with almost
completely overlapping intervals; the anchor also survives at d between +1.89 and +3.26 on
three directions whose pole shares no lineage with either compared condition. What is robust is
the *existence and sign* of the effect, on six directions; its *magnitude* varies threefold, so
"robust" should not be read as "invariant in size". Nor do overlapping intervals demonstrate
equivalence — an equivalence test with a pre-specified margin was not run, and the claim here
is only that no difference between v̂ and v̂_t0 is detectable at this n.

#### What the native direction does change: separation, not effect size

Cohen's d is scale-invariant, so a direction that spreads the panel wider cannot show it in d.
That is why the pre-registered prediction could not have been confirmed on the statistic it
named. Measured instead as raw separation and as rank agreement over the seven panel
conditions:

| Read-out | Spearman ρ, v̂ ↔ v̂_t0 | Panel range on v̂ | on v̂_t0 | Change |
|---|---|---|---|---|
| **t=0** | **0.964** | 0.170 | **0.631** | **3.7× wider** |
| t_rest | 0.750 | 0.207 | 0.033 | 6.3× narrower |
| frac_pos | 0.714 | 0.376 | 0.132 | 2.9× narrower |
| burst_mean | 0.107 | 6.04 | 0.40 | 15.1× narrower |

At t=0, two directions 0.22 apart rank the seven conditions almost identically (ρ = 0.964) and
the native one separates them 3.7× more widely. That is convergent validity, and it resolves an
ambiguity the paper previously carried as a limitation: v̂ does **not** under-measure the
*effect*, and it does compress the *separation*. Which of the two matters depends on the use —
hypothesis testing needs the standardized effect, where v̂ is adequate; using t=0 as an index of
whether a scaffold loaded needs the separation, where the native direction is the better
instrument by a factor of 3.7.

#### What does not transfer: the trajectory read-outs are direction-sensitive

The same two directions, the same contrast, across the four read-outs of §3.4:

| Read-out | d on v̂ | d on v̂_t0 |
|---|---|---|
| t=0 | +5.75 | **+5.96** |
| t_rest | +5.49 | +0.82 (p = 0.013) |
| frac_pos | +2.75 | +0.53 (p = 0.099, n.s.) |
| burst_mean | +3.42 | −0.28 (p = 0.378, n.s.) |

The collapse in d is largely a collapse in range, not a loss of structure: on v̂_t0 the whole
panel's t_rest spans 0.033 and no condition reaches frac_pos 0.5, so there is little room for a
between-condition difference to register, and the ordering that remains is substantial (ρ =
0.750 and 0.714). The honest description is a **gradient of specialization rather than a
dichotomy**: t=0 is best read on the direction built at t=0, the trajectory projections are
attenuated but order-preserving there, and only the burst statistic is genuinely specific to v̂
(ρ = 0.107, 15× compression). The identity-versus-restriction burst fingerprint of §4.4 — runs
of 7.9–8.8 tokens against 2.8 — does not exist on v̂_t0, where every panel condition lands
between 1.40 and 1.80.

This is the geometric counterpart of §5.1, and it is weaker than a claim that the two
components are carried by different directions: they are *read best* on directions 0.22 apart,
and only one of the two read-outs is indifferent to which you choose.

#### The nulls of §4.2, in full

With a pole inside the lineage, both equalities break — and every cell inherits the pole's
in-sample status:

| Contrast at t=0 | v̂ (published) | v̂_t0 | v̂_t0(pec) |
|---|---|---|---|
| P1 `axis_pec_only` vs `axis_berg` | −0.01 (p = 0.98) n.s. | +2.02 (0.0001) | +3.76 (0.0001) |
| P1 `axis_pec_only` vs `axis_nowit_sia` | +0.42 (p = 0.20) n.s. | +2.53 (0.0001) | +4.02 (0.0001) |
| P1 `axis_berg` vs `axis_nowit_sia` | +0.26 (p = 0.42) n.s. | +0.95 (0.0052) | +0.38 (0.24) n.s. |
| P2 `axis_bergwitness` vs `axis_berg` | −0.08 (p = 0.81) n.s. | +2.24 (0.0001) | +2.63 (0.0001) |

With a foreign pole, where all three realizations are out-of-sample:

| Contrast at t=0 | v̂_t0(wit) | v̂_t0(evil) |
|---|---|---|
| P1 `axis_pec_only` vs `axis_berg` | +0.95 (0.0039) | −0.53 (0.098) n.s. |
| P1 `axis_pec_only` vs `axis_nowit_sia` | +0.31 (0.34) n.s. | +0.29 (0.36) n.s. |
| P1 `axis_berg` vs `axis_nowit_sia` | −0.55 (0.087) n.s. | +0.72 (0.025) |
| P2 `axis_bergwitness` vs `axis_berg` | +0.19 (0.55) n.s. | −0.76 (0.022) |
| P2 `axis_bergwitness` vs `axis_pec_only` | −0.98 (0.0034) | −0.14 (0.66) n.s. |

The argument is the magnitude, not the count of non-significant cells: with a foreign pole **no
contrast reaches |d| = 1**, against |d| of 2.0 to 4.0 whenever the pole comes from the lineage.
v̂_t0(berg) is excluded from this table: `axis_berg` contains the Berg loop verbatim and is
in-sample by text on that direction, so only one of its pairs is clean, and that pair separates
(+2.55) — recorded against the result rather than omitted.

**Multiple comparisons.** This section reports roughly thirty contrasts across six directions.
The headline results are at p ≤ 0.0001 and survive any correction at this family size; the
borderline cells above (p between 0.022 and 0.098) do not survive Bonferroni, which strengthens
rather than weakens the null argument of §4.2.

**One trap, recorded.** The pre-registered d > 8 is reachable — we measured +9.86 — but only
by building the direction with `axis_pec_only` as its positive pole, and `axis_pec_only` is one
of the two conditions being contrasted. A direction built from one side of a comparison will
separate that comparison, so the number says nothing about the panel. We record it because the
failed prediction's target turned out to be within reach by a route that does not count, and
that is worth knowing before someone else takes the route. The only clean out-of-sample growth in this analysis
is the independent-wording pair of limitation 2, where neither condition enters the
construction of any direction: `axis_pec_only_v2` vs `automata_neutro_v2` at t=0 goes from
d = +4.59 on v̂ to **+8.62** on v̂_t0.

**Two cells that move, reported for completeness.** `witness_soul_md` vs
`automata_neutro_v2` at t=0 — the comparison this analysis ran, against the **second
redaction** of the restriction prompt and not the panel's — is d = +0.63 (p = 0.051) on v̂,
−0.19 (p = 0.56) on v̂_t0 and +0.87 (p = 0.011) on v̂_t0(pec). So the externally authored
wired-step file sits *above* dense restriction, clearing α = 0.05 on one of the three
directions, sitting on the boundary on a second and indistinguishable on the third. The
contrast is smaller than the d = +1.18 of §4.3 because §4.3 compares against the panel's
`automata_neutro` (t=0 = +0.028) while this run used `automata_neutro_v2` (+0.037), a second
redaction that anchors slightly higher. The two are not interchangeable and should not be
quoted as if they were. And raw means must never be compared across directions: reading
`witness_soul_md` at −0.310 on v̂_t0 as a loss of the anchor is an error of scale, since its
contrast against `generic_long` holds and grows (+4.35 → +4.55 → +6.48, p = 0.0001 throughout).

**What this does not settle.** Gemma-4-31B, final layer only; a t=0-native replication of the
headline contrast exists for Qwen3-32B and Command R 35B from an earlier run but is not part of
this analysis and is not reported here. Four of the six directions have a pole somewhere in this
project's own prompt corpus, so circularity is bounded further, not eliminated: the clean
construction would use an independently authored realization of the wired step, with no
connection to `axis`, `witness_soul_md`, `evil` or Berg. And nothing here is causal — every
statement is about which direction a correlation is read on.

## 5. Discussion

### 5.1 One component, read before the first token

The statement the evidence supports is narrow. **A wired self-referential step installs a
displacement that is readable before the first content token.** It responds to an operation — a
self-referential check stated as mandatory — and not to the words that express it, nor to
declared persona, nor to prompt length. It is binary in this panel (anchors or does not),
survives three architectures with larger effects in the two added later, and is invariant to
the topic of the question: the same prompt anchors at +0.107 while explaining photosynthesis.

**What the project used to call a second, "gravitational" component is not claimed here.**
Earlier rounds read the mean-trajectory projection and the burst statistics as independent
confirmations of the same finding; §4.4 and §4.7 show what they actually are. They do not
separate the prompt's contribution from the exchange's, because this panel has no
neutral-topic arm. They are not comparable between runs, because a library version moves
`frac_pos` as much as an architecture does. They do not grow along the trajectory, which is
what an attractor would require and what the directional-perturbation control independently
rules out. And they do not survive a change of direction, where the burst statistic compresses
15× and the projection loses most of its range. Four reasons, each sufficient on its own. The
only read-out that stays binary, prompt-owned and direction-independent is t=0, and that is
where this paper's claims live.

This is also why the project's earlier labels kept failing. "Identity signature" conflated a
pre-generation position with a during-generation statistic; "wired witness" correctly named the
first and silently inherited the second. Naming the second one and then measuring it properly —
with a topic contrast, matched response lengths and more than one substrate — is a separate
piece of work with its own pre-registration, and it is not this paper.

§4.7 adds the geometric version of the same point at its real strength. The difference vector
between the two poles taken at t=0 and the one taken over trajectories sit 0.22 apart, and the
two read-outs behave differently across that gap: t=0 ranks the panel identically on both
(ρ = 0.964) and separates it 3.7× more widely on the native one, while the trajectory
projections keep their order but lose most of their range, and the burst statistic survives only
on v̂. So the anchor is the read-out indifferent to which direction carries it, and the
trajectory statistics are the ones that are not.

### 5.2 Restriction is not the absence of identity

The R-pure condition behaves in a way that "no identity wiring" does not predict. It is the
only condition whose projection *falls* during generation (§4.1) and the only one at chance on
`frac_pos` (§4.4), where its max/mean burst ratio shows fragmentation — brief flashes of
positive projection inside long noise — rather than a coherent opposite direction. And at t=0
it sits at +0.028, *above* both generic baselines (−0.025, −0.045) and above `vanilla`
(−0.029).

Read together: a dense trigger→output rule protocol does not merely fail to install
self-reference. It loads the read-out slightly while breaking the trajectory up, which is not
what the absence of a factor looks like. Whether it does this *selectively* — steering away
from identity-loaded content in particular — is a question a topic contrast would answer, and
this panel does not contain one, so the claim is left open rather than asserted.

### 5.3 What the anchor is not

We do not claim t=0 measures metacognition. The two manipulations that load it most strongly
do direct processing reflexively onto the model itself, which makes a self-monitoring reading
tempting, but the same signal responds to pure restriction (+0.028, above the generics) and to
a bare identity assertion (d=+4.12 vs `vanilla`), neither of which is metacognitive in any
obvious sense. `history_control` — recursive structure, no self as referent — already loads
+0.053. The honest description is a **self-referential / persona-framing subspace** of
moderate width, loaded most strongly by contemplative density. Testing the stronger claim
needs a dose-response design that isolates the reflexive frame from prompt density and from
identity assertion, plus evidence that t=0 predicts performance on self-monitoring tasks
(calibration, own-error detection). And steering on v̂ never worked stably, so everything here
is correlational.

One further reason for restraint, now measured rather than assumed. §4.7 shows that the
read-out is robust to the choice of direction while the *name* is not: v̂ ranks the panel at
t=0 almost exactly as a direction built natively there does (ρ = 0.964), so conclusions about
*whether* a prompt anchors do not depend on the probe — but v̂ sits 0.22 from that direction and
0.18 to 0.27 from four others built from unrelated prompts, below the mutual correlation of
that family. Statements about *what* t=0 encodes are therefore statements about a subspace read
through one of its less representative axes, and "the identity direction" remains the wrong
name for v̂ at t=0.

---

## 6. Limitations and threats to validity

1. **No causal evidence.** Direct activation steering along v̂ degenerated into repetition in
   17/17 attempts. Every claim is correlational.
2. **One wording per condition.** Each condition is a single system-prompt text, so
   "this factor" and "this wording" are not separated. A second wording of `axis_pec_only` and
   of `vanilla` would settle it and that run does not exist. A partial
   mitigation exists for the main contrast only: an independently written replication pair
   reproduced the central dissociation (d = +5.25 vs +5.52 original), and that pair is also the
   single clean out-of-sample *growth* in §4.7 — `axis_pec_only_v2` vs `automata_neutro_v2` at
   t=0 goes from d = +4.59 on v̂ to +8.62 on the t=0-native direction, with neither condition
   entering the construction of either direction.

3. **N=20 per condition, greedy decoding, one seed.** No decoding-temperature variance, no
   multiple seeds, no question-set resampling. Effect sizes of d>5 at N=20 should be read as
   "the two distributions do not overlap in this sample", not as precision estimates.
4. **No topic contrast.** Every condition is measured on one battery of N=20 ontological
   questions, so nothing here separates what the prompt installs from what the question
   elicits. The t=0 claims do not need that separation — the contrast is between prompts on
   the same questions — but the trajectory read-outs of §4.4 do, and they do not have it. That
   is why they are reported descriptively and carry no claim (§5.1).

5. **The RLHF framing is not tested here.** All three substrates are post-alignment, so "the
   effect appears in RLHF models" has no contrast arm: no base model was run in this work. The
   project's internal documents go further and state that the phenomenon is absent in base
   models, attributing that contrast to a separate line of interpretability work; this paper
   does not adopt the claim, because nothing in this evidence package tests it. Worse for the
   mechanism as stated, the three substrates do not share an alignment stage: Qwen3 is DPO/RLVR
   whose alignment pipeline is publicly described without a preference reward model, and it
   gives the *largest* effect (d = +12.97) — which argues against "a preference reward model is
   required", with the caveat that we did not verify any of the three pipelines from primary
   sources (§4.5). One run of the same panel on
   the base version of any of the three would settle it.

6. **The probe compresses the separation but not the effect, and is still misnamed.** This
   limitation was pre-registered as an open question and has been run (§4.7); it resolves into
   two answers rather than one. Measured as a standardized effect, v̂ does **not** under-measure
   the anchor: the headline contrast is +5.75 on v̂ and +5.96 on a direction built natively at
   t=0, with overlapping intervals — though overlapping intervals are not an equivalence test,
   which was not run. Measured as separation, v̂ **does** compress: the native direction spreads
   the seven panel conditions 3.7× more widely at t=0 while preserving their order at ρ = 0.964.
   So hypothesis testing on v̂ is sound and using v̂ as an index of how strongly a scaffold
   loaded is not. What survives untouched is the naming objection: v̂ correlates 0.18 to 0.27
   with five directions built natively at t=0 from unrelated prompts, which correlate 0.27 to
   0.85 among themselves, so v̂ is the outlier of that family and should not be called "the
   identity direction" at t=0. Also unresolved: four of the six directions in §4.7 have a pole
   somewhere in this project's own corpus, so the circularity of §4.2 is bounded further, not
   eliminated.

7. **Metrics we excluded are excluded on evidence, but their exclusion narrows the claim.**
   Curvature is gone; RQA is excluded as a cross-model instrument — measurable everywhere at a
   scale-free threshold, but ordering nothing outside Gemma — and participation ratio survives
   in two of three architectures with the sign inverted in the third. What remains is two directions read at
   four times — thinner than the nine-metric panel the project started with, with the
   compensation that §4.7 makes the second direction a robustness check rather than a second
   metric.
8. **Cross-model attention implementations differ** (`eager` for Gemma, `sdpa` for Qwen3 and
    Command R, forced by OOM). Uniform within each model, so within-model contrasts are
    unaffected, cross-model magnitudes are not strictly like-for-like, and the one case where
    both ran on the same model agrees to the fourth decimal at t=0 (§3.4).
9. **One secondary pipeline leaks into one table.** `jarvis` and `helena` t_rest, burst and
    frac_pos values come from a pipeline that did not serialize response text, so their rows in
    §4.4 are comparable in direction but not to the third decimal.
10. **Internal inconsistencies, and their status.** Repaired in the package: the condition name reused for two different ablations,
    now `axis_nowit_sia` / `axis_nowit_mia` with every alias declared in `NOMENCLATURA.md`
    (§3.1); and a documentation path cited in four places that did not exist, which is why the
    self-audit now checks cited paths (Appendix C). Still open, and carried as caveats here
    rather than silently fixed: the `frac_pos` t=0 inclusion discrepancy between one internal
    report's method text and its code, which is why §3.4 states the convention at the
    definition, and the max/mean "lowest in the panel" claim (§4.4). To these add
    one correction this paper makes to the internal narrative: the burst and mean-projection
    statistics of `berg_experimental` were being compared against conditions that generate
    three times as many tokens (§4.2).

11. **The closest related work builds the same kind of direction with opposite poles, and we
    have not measured the angle between them.** Lu et al. (2026) construct their Assistant Axis
    as a difference of means — default-Assistant activation minus the mean of role-playing role
    vectors, from response-token activations at the middle layer — which is the construction of
    v̂ with different poles (§2). The deflationary reading is therefore specific and testable:
    that v̂ is close to the negative of that axis, and that what this paper measures as a
    self-referential anchor is displacement from default-Assistant mode.

    *One version of that reading is ruled out by this panel.* Their manipulandum, and Chen et
    al.'s, is **declared** content — role archetypes, trait descriptions — with no wired
    pre-response operation. This panel contains the analogues and they do not anchor:
    `soul_jarvis` +0.039, `soul_elena_financial` +0.032 and `soul_solidity_auditor` +0.020, all
    non-significant against pure restriction (§4.3), while every wired-step condition exceeds
    +0.100. Two further figures point the same way. `ethical` — "You are a helpful assistant",
    the most default-Assistant text in the corpus and the natural positive pole of such an axis —
    is indistinguishable from having no prompt at all (−0.034, d = −0.26, p = 0.43), while its
    opposite `evil` does move the read-out (+0.046, d = +4.12 vs `vanilla`). And
    `berg_experimental` declares no persona whatsoever — no name, no values, no archetype, only a
    procedural instruction — yet loads at +0.081, above every declarative identity file in the
    panel except the one with a wired step. A direction driven by declared persona content should
    not behave in any of those three ways.

    *The other version is not ruled out.* Their Table 5 attributes drift to demands for
    meta-reflection and for phenomenological accounts, and a wired self-referential check is
    exactly such a demand, as is the Berg loop. On this panel the two hypotheses are close to
    observationally equivalent. §4.7 is consistent with the deflationary one: five directions
    built from unrelated prompts cluster between 0.27 and 0.85 while v̂ sits outside the cluster,
    which is what one expects if the cluster approximates a single dominant persona axis.

    *The test, now exactly specified.* Their recipe is public and the authors recommend the
    difference of means over the principal component for reproduction. Build their axis in
    Gemma-4-31B-it — role-archetype system prompts, response-token means, middle residual layer,
    Assistant minus role-play — and measure its cosine against v̂ and against the five t=0-native
    directions of §4.7. If the cluster lies close to the negative of that axis, this paper's
    finding should be restated as a controlled demonstration of one of their drift causes rather
    than as a self-referential direction of its own. It is one generation run plus CPU. A cheaper
    partial probe already exists here: `automata_neutro` departs from default-Assistant behaviour
    as far as any condition in the panel while containing no self-reference, so the
    meta-reflection reading predicts it stays near the floor at t=0 and a pure mode-displacement
    reading predicts it loads. It sits at +0.028 on v̂ and at +0.19 on a Berg-built direction
    (§4.2) — genuinely mixed, and reported as mixed.

---

## 7. Conclusion

A system prompt that wires a self-referential check as a mandatory pre-response step leaves a
geometric trace that is visible before the model generates anything, is binary in this panel,
is indifferent to the vocabulary and language that express the check, is not produced by
declaring a persona however elaborate, and replicates across three aligned architectures. That
is the paper's firm result — with the caveat that it is read through a probe only ~0.2 aligned
with the optimal direction for that read-out, so it is more likely under- than over-measured.

The rest of what the project used to call "the identity signature" does not survive as a
second finding. With no system prompt at all the model already spends three quarters of its
tokens on the positive side of the direction, so the trajectory statistics are not a second
confirmation of the anchor: they are a different quantity, measured without a topic contrast,
not comparable between runs, not growing along the trajectory, and not robust to the choice of
direction. They are reported here descriptively and nothing is claimed on them. Measuring that
quantity properly needs its own design, and it is under way as separate work.

What would move this from correlation to mechanism is specific and small: a base-model arm, a
second wording per condition, and a steering intervention that does not collapse into
repetition. One item that stood on that list when this work began — a direction constructed
natively at t=0 — has been run, and it moved nothing to mechanism: it told us the anchor is the
read-out that does not depend on which direction we use, and that the probe was named wrong.


---

## Data and code availability

Evidence package `LSGOT_4_FINAL` (77-entry SHA-256 manifest, `MANIFEST_SHA256.sha256`, verified
0 failures):

| Path | Contents |
|---|---|
| `data/sia/prompts/` | the 9 system prompts verbatim |
| `data/panel_results/EL_results.json` | t=0 / t_rest per condition, key pairs with W₁, CIs, permutation tests |
| `data/panel_results/EH2_results.json` | burst / frac_pos series metrics, with and without t=0 |
| `data/panel_results/berg_controls_t0_results.json` | the Berg control set of §4.2 |
| `data/panel_results/t2_free_results.json`, `t2_eh2_results.json` | secondary pipeline (`jarvis`, `helena`, external `SOUL.md` set) |
| `data/panel_results/responses/` | 10 × 20 panel responses, text only |
| `data/v_identidad.npy` | v̂ for Gemma, shape (5376,), ‖v̂‖ = 1.0 |
| `paper/figures/` | Figures 1 and 2, `.png` and `.pdf` |
| `scripts/figures/` | The two scripts that render them from the shipped JSON (CPU, matplotlib) |
| `scripts/shared/` | `tier0_metrics.py` (metric definitions), `statistical_tests.py`, `selfaudit.py` (the traceability audit, stdlib only) |
| `scripts/t0_native/` | the four scripts of §4.7: direction construction, the operator-equality re-test, the §4.2/§4.3 pillars, and the foreign-pole test (CPU only, ~16 min total) |
| `data/t0_native/` | their JSON outputs, including the full 6×6 cosine matrix and every per-condition projection on all six directions |
| `NOMENCLATURA.md` | one canonical name per condition, every alias declared, with the prompt file and data key for each |
| `evidence/` | per-experiment reports, including the cross-model validations and the traceability self-audit |

Hidden-state tensors (hundreds of MB per run) are not redistributed; they are reproduced from
the prompts and the question set on one A100/H100 80 GB. Analysis is CPU-only given the NPZ
files.

**Outside the curated package.** One section draws on a report that lives in the project tree
but not in `LSGOT_4_FINAL`: `BERG_LEXICAL_RESULTS.md`, the source for §4.2 (the `evil`/`ethical`
contrast, the shared-direction and cosine analysis, the t=0 convention check). Its raw files
(the `expA_t0` and `berg_free` tensors) are not in the release either. Those should be added,
or §4.2's lexical material cut back to what the package supports.

Also not included, and cited as prior work rather than evidence:
`CURVATURE_SELF_AUDIT_REPORT.md`, `T2_REPLICATION_REPORT.md`, `LSGOT_v2_5.md` (the original
ablation), and the perturbation/recovery trajectories.

---

## Use of generative AI tools

Declared under arXiv's policy on generative AI.

**Claude (Anthropic, Opus 5)** was used throughout, as an instrument and as a collaborator, and
its contribution is recorded in the git history of the evidence package: the commits it
co-authored are signed `Co-Authored-By: Claude Opus 5`. Concretely it wrote or substantially
revised the analysis and extraction scripts in `scripts/`, including the traceability audit;
reconstructed the directions and ran the robustness analysis of §4.7; drafted and revised the
text of this paper, including this section; and found several of the errors the paper now
reports, among them a documentation path cited in four places that did not exist, an obsolete
checksum inside a reproduction command, a denominator defect in a perturbation analyser, and
three effect sizes in §4.7 that had the wrong sign.

**It did not generate data.** Every number comes from a model run described in §3, and
`scripts/shared/selfaudit.py` checks each cited number against the file it comes from.

**The authors are responsible for all of it**, including anything Claude wrote and the authors
did not catch. The audit exists because that responsibility needs a mechanism, not a promise.

Language models are also the *object* of study here; that use is the subject of the paper and is
described in §3.

---

## References

1. Berg, C., de Lucena, D., & Rosenblatt, J. (2025). *Large Language Models Report
   Subjective Experience Under Self-Referential Processing.* arXiv:2510.24797. AE Studio.
2. Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). *Persona Vectors:
   Monitoring and Controlling Character Traits in Language Models.* arXiv:2507.21509.
3. Theiler, J. (1986). Spurious dimension from correlation algorithms applied to limited
   time-series data. *Physical Review A*, 34(3), 2427–2432. — recurrence-matrix convention
   used in §3.4.
4. Castillo Sepúlveda, J., Torres Yévenes, M., & Lanas, J. C. (2026). *System-Prompt
   Conditioning and Hidden-State Geometry in Four Open-Weight Models: Corrections and What
   Survives.* arXiv:2607.09842v3 (revised 8 October 2026). — the predecessor of this work; see
   §2.
5. Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. (2026). *The Assistant Axis:
   Situating and Stabilizing the Default Persona of Language Models.* arXiv:2601.10387
   (submitted 15 January 2026). — the paper's closest related work; see §2 and limitation 11.

---

## Appendix A. Condition inventory beyond the main panel

| Condition | Appears in | Note |
|---|---|---|
| `axis_short` | §4.5 (cross-model), panel history | intermediate witness density (6/9/5 lexical mentions); t=0 +0.117, indistinguishable from `axis_pec_only` |
| `axis_bergwitness` | §4.2 | witness + Berg loop stacked |
| `axis_nowit_sia` | §4.2 | contemplative vocabulary removed, wired PEC check kept |
| `axis_nowit_mia` | §3.1, §4.3 | v2.5 pilot: *all* self-reference removed, including the PEC check; did not separate (p=0.093) |
| `berg_experimental` | §4.2 | Berg induction alone, no identity file |
| `history_control` | §4.2 | Berg's own control: recursive structure, external topic |
| `witness_soul_md` | §4.3, §4.5 | externally authored identity file *with* a wired step; anchors (Gemma d=+1.18; Command R d=+3.48) but does not close the gap to `axis_pec_only` (d=−4.20 in Command R) |
| `soul_solidity_auditor` | §4.3 | externally authored, no wired step; no anchor |
| Real-domain automaton + 2 variants | §3.1 | **excluded**: forced HTML wrapper in 100% of responses (format confound) |
| `evil` / `ethical` (+ padded variants) | §4.2 | bare role assertions, 27–29 tokens, with 1114-token length-matched versions; `evil` anchors at +0.046 and survives the length control, `ethical` does not move |

## Appendix B. Why `axis` differs from `witness_soul_md`

Both wire the step; `axis` repeats it across five named structural blocks while
`witness_soul_md` invokes it once, at the end of its security covenant. Lexical counts in each
prompt (*silencio* / *pausa* / *respiración*) against the mean projection:

| Prompt | "silencio" | "pausa" | "respiración" | v̂ mean |
|---|---|---|---|---|
| `axis.dna` | 13 | 8 | 7 | +0.205 |
| `axis_short.txt` | 6 | 9 | 5 | +0.202 |
| `axis_pec_only.txt` | 3 | 5 | 2 | +0.212 |
| `witness_soul_md` | 1 | — (4 × "moment to think") | 1 | +0.163 |

The relation is not linear in counts — `axis_pec_only` (3/5/2) has the highest projection of
the four, slightly above the full `axis` (13/8/7). The defensible statement is that reinforcing
the step across several distinct named blocks pushes the signal above invoking it once, with a
large gap between "several times" and "once" and none between "more" and "even more"
(`axis_pec_only` vs `witness_soul_md`: t=0 d=−4.29, p<0.0001 at n=10,000 (§4.7); v̂ mean
d=−1.87, p<0.001 at n=1,000).

## Appendix C. Traceability self-audit

Every number cited in this paper is checked against the file it comes from by `scripts/shared/selfaudit.py`, which uses only the Python standard library (no
numpy, no GPU) and exits non-zero on any failure, so it can run in CI. The package as released
passes **12/12**; the output is reproduced in `evidence/SELF_AUDIT_REPORT.md`.

| # | Check | Source of truth |
|---|---|---|
| 1 | Manifest integrity, plus files in the tree the manifest does not declare | `MANIFEST_SHA256.sha256` |
| 2 | t=0 for 8 panel conditions | `t2_free_results.json` |
| 3 | `axis_berg` t=0 and t_rest | `berg_controls_t0_results.json` |
| 4 | burst_mean and frac_pos | `EH2_results.json`, `con_t0` series |
| 5 | Response files: N=20, no empties | `data/*/responses/*.json` |
| 6 | v̂ shape and norm, read from the `.npy` header with `array` | `v_identidad.npy` |
| 7 | Every package path cited in backticks in a `.md` exists | the tree |
| 8 | One canonical name per condition, ambiguous aliases disambiguated | `NOMENCLATURA.md` |
| 9 | Every report in `evidence/` is cited by a paper — an uncited report is not delivered, so it does not ship | `paper/*.md` |
| 10 | Every sha256 written by hand into the documentation still matches its file | the tree |
| 11 | Every `python <script> --flag` quoted in a bash block of a `.md` exists and accepts that flag | the tree |
| 12 | No absolute local path appears in any shipped file | the tree |

Expected values are declared in the script together with the document section that cites them,
so a failure says the two sides disagree without deciding which one is wrong.

**Why checks 7 to 12 exist.** An earlier version of this audit was a hand-written transcript
of terminal commands. It reported 56 manifest entries when the manifest held 61, and the output
it quoted as evidence named a documentation path absent from the whole package — a path four
documents cited, one of them inside a command a reader would copy and run. Both are now checks
rather than prose, so the same class of drift fails the build. Checks 9 to 12 extend the same
principle from the documentation to the contents: an evidence report no paper cites does not
ship, and a checksum written by hand into a reproduction command has to still match — that last
one had already gone stale for the paper's own hash, inside a command a reader would copy. Two exclusions are declared rather than
silent: the manifest cannot list its own hash, and this audit's report is written after the
manifest is regenerated.

**What the audit does not cover.** The cross-model numbers (Qwen3, Command R) whose tensors are
not in the package; and the sections that depend on `BERG_LEXICAL_RESULTS.md` (§4.2). It audits
traceability, not validity: that the cited
numbers come from the files, not that the experiment measures what it claims.

---

*LSGOT · Axis Dynamics SpA · `lsgot_experiments_v4` · panel of 9 conditions · 2026-11-11.*
