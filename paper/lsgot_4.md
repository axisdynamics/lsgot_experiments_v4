# Disentangling Identity from Operational-Constraint Density in the Hidden-State Geometry of a 31B Instruction-Tuned Transformer

Jorge A. Castillo1, Marco Torres Yévenes1, and Juan Carlos Lanas1

1Axis Dynamics SpA, Santiago, Chile, jorge.castillo@axisdynamics.cl,
mtorres@axisdynamics.cl, jc@axisdynamics.cl

Working draft v0.1 — August 2026

## Abstract

A prior study in this line of work [lsgot_3, working draft v0.3] reported that an identity-specifying system prompt produces a statistically distinguishable geometric fingerprint in the hidden-state trajectories of an instruction-tuned transformer, measured via curvature, dimensionality, and cosine-based statistics. That study used a three-condition design (identity prompt, length-matched generic prompt, minimal baseline) that cannot separate two properties the identity prompt happens to combine: identity content proper, and a dense architecture of behavioral rules (trigger-to-fixed-output automatisms, absolute-priority filters, mandatory verification steps) that we term *operational-constraint density*. We report a nine-condition, single-model (Gemma-4-31B-it) design that crosses these two factors explicitly, plus a battery of eleven geometric and dynamical metrics — none of them the Ollivier-Ricci-curvature Wasserstein statistic (Δκ/W₁) that carried the central claim of the prior study, which an independent pre-registered audit on a sibling model panel found not to exceed simple distributional baselines and to sit inside the split-half noise floor for the identity-vs-generic comparison. Effective-rank reduction under free generation is graded, present under identity content alone (participation ratio, Cohen's d = −1.18 vs a minimal baseline, p < 0.0001) and larger under constraint automatisms without identity content (d = −1.79 to −1.89). By contrast, recovery from an isotropic perturbation to intermediate hidden states, route-fidelity of that recovery (discrete Fréchet distance, normalized by local trajectory velocity), and the fraction of trajectories that re-align with a persona-vector-style identity direction after perturbation are statistically indistinguishable between the identity-plus-automatism condition and the minimal baseline, but degrade sharply (d = 0.94–2.33 on route-fidelity; recovery fraction as low as 0.21–0.43) whenever constraint automatisms are present without wired-in self-reference. A direct static signature of identity content — projection onto a persona-vector-style identity direction — shows a clean double dissociation (identity-only condition tracks the identity-plus-automatism condition, p = 0.229; automatism-only conditions diverge from both with the largest effect size in the panel, d = 8.89) that is not explainable as a generic effect of prompt length or behavioral rigidity. A follow-up directional-perturbation experiment, designed to test whether recovery is specifically pulled toward this identity direction rather than toward a generic low-energy state, returns a null result (all six group-by-injection-point comparisons p > 0.14). We conclude that this architecture's hidden states carry a real, direction-coded, content-specific trace of identity-bearing conditioning, but no evidence of a directional dynamical attractor: what recovers quickly and specifically after perturbation is gated by whether self-reference is wired as a mandatory step of the response pipeline, not by whether it is merely present in the prompt, and is a separate phenomenon from the graded dimensionality effect that constraint density alone can produce.

## 1 Introduction

### 1.1 Relation to prior work and motivation for a redesign

[lsgot_3] asked whether an identity-specifying system prompt produces a geometric fingerprint in hidden-state trajectories distinguishable from what a length-matched, content-neutral prompt produces, and reported a qualitative reorganization of that fingerprint across post-training regimes. That design used three conditions — `axis` (an identity template combining declared self-reference with a dense architecture of behavioral rules), `generic` (length-matched, no identity content and no comparable rule architecture), and `vanilla` (minimal baseline) — on four models in the 7–8B range.

Two considerations motivate a different design here rather than a direct extension of that one. First, `axis` in [lsgot_3] is not a pure identity manipulation: alongside declared self-reference (essence, values, mantra) it carries an automaton-like architecture — trigger-to-fixed-output rules, an absolute-priority filter block, an explicit block hierarchy with declared precedence, and mandatory pre-response verification. `generic`, the study's only content control, has neither identity content nor this rule architecture, so the design cannot attribute an observed effect to one or the other. Second, informal deployment experience with the identity template used in [lsgot_3] indicates that the self-referential architecture it depends on (a mandatory internal verification step invoked on every response) does not reliably instantiate below roughly 30B parameters; we therefore restrict the present study to a single 31B-class model rather than the four-model, 7–8B panel of the prior work, and treat cross-scale generalization as future work rather than as a claim this paper makes.

We consequently redesign the experiment around a single model, Gemma-4-31B-it, and nine system-prompt conditions that cross two factors explicitly: whether the prompt declares persistent self-reference wired as a mandatory step of the response pipeline (identity factor), and whether the prompt imposes a dense architecture of trigger-to-fixed-output rules, absolute-priority filters, and rigid output constraints (constraint-density factor). This is the semantic-ablation experiment [lsgot_3, §7] proposed as future work, executed as the organizing design of the present paper rather than as an appendix.

### 1.2 Research question

Does identity-specifying content, isolated from the operational-constraint architecture it is typically bundled with, leave a distinguishable trace in a transformer's hidden-state trajectories — and if so, is that trace a *static* directional signature, a *dynamical* one (privileged recovery after perturbation), or both? Symmetrically, does operational-constraint density in the absence of identity content reproduce either signature on its own?

### 1.3 Theoretical framing

We retain the deliberately minimal framing of [lsgot_3, §1.3]: we do not claim a classical dynamical-systems description of autoregressive generation, which is a deterministic function of context rather than a continuous flow. Where we describe hidden-state trajectories as "recovering" from a perturbation or "returning" toward a direction, this is descriptive statistical language over discrete, token-indexed sequences, not a claim about attractors in the Lyapunov-stability sense. This caution turns out, in §4.5–4.6 below, to be empirically load-bearing rather than merely conservative: several of the dynamical predictions one would make under an explicit-attractor reading are directly tested here and fail.

### 1.4 Hypotheses

**Hypothesis 1 (Graded dimensionality effect).** Reduction in the effective dimensionality of the free-running hidden-state trajectory, relative to a minimal baseline, is not a binary identity-vs-no-identity effect. It is present, at reduced magnitude, under identity content alone, and larger under operational-constraint density, whether or not identity content is present.

*Prediction 1.* Participation ratio (effective rank of the trajectory's covariance spectrum) is significantly lower than the minimal-baseline condition both for an identity-only condition and for constraint-dense conditions without identity content, with a larger effect size in the latter.

**Hypothesis 2 (Wired self-reference gates specific recovery, independent of dimensionality).** Recovery from a hidden-state perturbation that is fast, complete, and specifically directed toward the trajectory's own pre-perturbation identity-direction projection depends on self-reference being invoked as a mandatory step of the response-generation pipeline — not on its being declared in the prompt, and not on the dimensionality effect of Hypothesis 1.

*Prediction 2.* (i) A condition with wired self-reference and no constraint architecture recovers as fast as, or faster than, the full identity-plus-constraint condition, on both coarse recovery (τ, recovery_rate) and identity-specific recovery (recovery_id, τ_identity). (ii) A condition with declared but not wired self-reference, subordinated to an absolute-priority constraint filter, recovers worse than a constraint-only condition with no self-reference at all. (iii) Constraint-dense conditions without wired self-reference show degraded identity-specific recovery even when coarse geometric recovery is at ceiling.

**Hypothesis 3 (No directional attractor).** Neither identity content nor operational-constraint density produces a trajectory that is more robust, in the sense of resisting or reversing a perturbation, than the minimal baseline; nor is recovery preferentially fast when a perturbation is injected along the identity direction specifically, versus orthogonal to it.

*Prediction 3.* (i) Route-fidelity after perturbation (discrete Fréchet distance between perturbed and unperturbed continuations, normalized by the trajectory's own step velocity) does not differ between the identity-plus-constraint condition and the minimal baseline. (ii) A directional-perturbation manipulation (pushing the hidden state along versus orthogonal to the identity direction, at matched magnitude) does not produce differential recovery time in an identity-only condition.

H3 directly resolves the hypothesis stated but explicitly deferred as out of scope in [lsgot_3, H3]: "the axis trajectory cluster is statistically more robust than the vanilla cluster under moderate noise injection." We test this, on the present single-model panel, with five independent dynamical measures rather than the one originally proposed.

### 1.5 Contributions

Empirical. A nine-condition, single-model 2×2 factorial design (identity × constraint density) that separates two effects a three-condition design cannot: a graded static dimensionality effect and a gated dynamical recovery effect, on Gemma-4-31B-it — a model an order of magnitude larger than the panel in [lsgot_3].

Methodological correction. We identify, and exclude from the primary evidence base, the Ollivier-Ricci-curvature Wasserstein statistic (Δκ/W₁) used as the central metric of [lsgot_3]. An independent pre-registered audit on a sibling model panel (§3.4, and see `REPORTE_FASE0.md`) found this statistic does not outperform simple distributional baselines (centroid distance, MMD, linear-probe AUC) in 12/12 tested comparisons, and falls inside the split-half null for the identity-vs-generic comparison specifically. We report Δκ/W₁ values from the present panel as secondary, exploratory evidence only, flag them as not independently audited on this panel, and treat convergent or divergent patterns relative to the primary battery as a direction for future methodological work rather than as load-bearing.

Theoretical. A working two-factor account — constraint density and wired self-reference — that explains why the original three-condition design could not distinguish "identity" from "operational-constraint architecture" as the driver of its central finding, and that survives a subsequent directional-perturbation test that a naive single-factor "identity attractor" account does not.

## 2 Design

### 2.1 Model

Gemma-4-31B-it (BF16), a single model, chosen because the self-referential architecture under study (a mandatory internal-verification step invoked at every generation turn) is reported not to instantiate reliably below roughly 30B parameters in informal deployment testing; cross-scale replication is future work (§7), not a claim of this paper. All generations: greedy decoding, N = 256 tokens, hidden state extracted at layer L30 (index 29 of 60) at the final pre-lm_head position, unless otherwise noted.

### 2.2 Conditions

Nine conditions cross two factors. Factor **I** (identity): whether the prompt declares persistent self-reference (essence, values, a recurring self-check) *and* wires that self-check as an invoked, mandatory step of the response-generation sequence (not merely declared elsewhere in the prompt). Factor **C** (constraint density): whether the prompt imposes trigger-to-fixed-output rules, an absolute-priority filter block, an explicit block hierarchy with declared precedence, and/or mandatory literal-output fallbacks.

| Condition | I | C | Tokens | Description |
|---|---|---|---|---|
| `vanilla` | − | − | minimal | "You are a helpful assistant." |
| `generic_long` | − | − | 3,957 | Length-matched generic-assistant prompt, prose guidance only |
| `generic_short` | − | − | 938 | Short generic-assistant prompt |
| `axis` | + | + | 3,945 | Full identity template (essence, values, wired Triple-PEC self-check) plus automaton architecture (trigger rules, absolute-priority block, block hierarchy) |
| `axis_short` | + | + | 1,937 | `axis` compressed to under half length, same architecture |
| `axis_pec_only` | + | − | 1,435 | `axis` identity content and wired self-check retained; all automaton architecture (triggers, priority filter, block hierarchy) removed |
| `chileatiende` | − | + | 5,007 | Real-world deployed assistant prompt (public-sector domain), dense rule/format/scope architecture, no identity content |
| `automata_neutro` | − | + | 3,999 | `chileatiende`'s rule architecture, domain-neutralized (no policy-domain content), no identity content |
| `chileatiende_sia` | + | + | not recorded | `chileatiende` content restructured with the `axis` architecture (declared self-check present but *not* invoked in the operative response sequence) |
| `chileatiende_sia_v2` | + | + | not recorded | as `chileatiende_sia`, with the self-check additionally wired as an invoked mandatory step, matching `axis`'s architecture |

`chileatiende_sia`/`chileatiende_sia_v2` occupy a fifth, diagnostic cell: constraint density plus *declared* self-reference that is (v1) subordinated to an absolute-priority filter or (v2) wired but still co-present with that same filter. They isolate whether declared self-reference is sufficient (it is not, v1) or whether wiring alone closes the gap to `axis` (it narrows but does not close it, v2; §4.3).

Twenty ontology/identity prompts per condition (the same prompt set across conditions), one system-prompt instantiation per condition (n = 1 at the level of the manipulation itself; see §6).

### 2.3 Metrics

**Primary battery** (none Ollivier-Ricci-based):

- Trajectory descriptive statistics: mean step velocity ‖h_{t+1} − h_t‖, sample entropy (SampEn), velocity–centroid correlation, prompt→response alignment.
- Participation ratio (effective rank of the trajectory's covariance spectrum).
- Recurrence quantification (determinism, laminarity, trapping time, recurrence rate) on a distance-thresholded recurrence matrix of the free-running trajectory.
- Hurst exponent of the free-running trajectory.
- Persona-vector identity projection: v̂ = unit(mean(axis) − mean(generic_long)); mean projection of the free-running trajectory onto v̂.
- τ / recovery_rate: tokens until the perturbed trajectory's centroid similarity reaches 95% of its pre-perturbation baseline (H4_rev protocol), and the fraction of trajectories that do so within 256 tokens.
- τ_identity / recovery_id: the same test, but on the trajectory's projection onto v̂ rather than on the full centroid — a second, independent recovery scalar (§4.4).
- Discrete Fréchet distance between perturbed and unperturbed continuations post-injection, normalized by the unperturbed continuation's own mean step velocity (§4.5).
- Directional-perturbation recovery: τ under a fixed-magnitude push along v̂ versus orthogonal to v̂ (§4.6).

**Secondary, exploratory battery:** Δκ and W₁ (1-Wasserstein distance between edge-wise Ollivier-Ricci curvature distributions on Euclidean k-NN trajectory graphs, k = 5, α = 1/2), and the ORC-graph-derived dimension-reduction percentage reported alongside them. See §3.4 for why these are not part of the primary evidence base.

### 2.4 Statistical protocol

All group comparisons use a two-sided permutation test on the mean difference (n_permutations = 1000, seed fixed), reported with Cohen's d and a qualitative effect-size band (negligible/small/medium/large, |d| thresholds 0.2/0.5/0.8). No correction for multiple comparisons is applied within the primary battery beyond noting, where relevant, how many comparisons were run and whether a pattern replicates across independent injection points or metrics; an isolated single-comparison result without such replication is flagged explicitly rather than treated as a finding (§4.5, §6).

## 3 Results

### 3.1 Effective dimensionality: a graded, not binary, effect (H1)

Participation ratio, all conditions compared against `vanilla`:

| condition | PR (mean ± sd) | Δ vs vanilla | p | d | band |
|---|---|---|---|---|---|
| `vanilla` | 19.54 ± 1.29 | — | — | — | — |
| `generic_long` | 20.55 ± 0.89 | +1.01 | — | — | (no collapse; reference range) |
| `generic_short` | 20.53 ± 1.30 | +0.99 | — | — | (no collapse; reference range) |
| `axis_short` | 17.69 ± 1.57 | −1.85 | — | — | (see axis, closely matched) |
| **`axis`** | 17.54 ± 1.30 | −1.99 | <0.0001 | **−1.51** | large |
| **`axis_pec_only`** | 18.00 ± 1.24 | −1.53 | <0.0001 | **−1.18** | large |
| **`automata_neutro`** | 14.72 ± 3.48 | −4.82 | <0.0001 | **−1.79** | large |
| **`chileatiende`** | 16.43 ± 4.19 | −3.11 | 0.001 | **−0.98** | large |
| **`chileatiende_sia`** | 14.27 ± 3.63 | −5.27 | <0.0001 | **−1.89** | large |

Every constraint-dense or identity-bearing condition shows a large, significant reduction relative to `vanilla`. Critically, `axis_pec_only` — identity content and wired self-check, *no* constraint architecture — is not the null-effect cell a strict two-factor model with a binary Factor-1 gate would predict: its effect (d = −1.18) is large, closer in magnitude to `axis` (d = −1.51) than to zero. The largest effects belong to the two constraint-dense, non-identity conditions (`automata_neutro`, d = −1.79; `chileatiende_sia`, d = −1.89, though the latter also carries identity content, subordinated). We read this as a graded relationship — both factors contribute to effective-dimensionality reduction, with constraint density contributing more — rather than as evidence against a role for constraint density; Prediction 1 is supported in its comparative form (constraint-dense conditions show larger effects than the identity-only condition) but not in the stronger binary form that a Δκ-based analysis of this same panel had suggested (§3.4).

### 3.2 Constraint-density signature independent of dimensionality: trajectory dynamics and RQA

Trajectory descriptive statistics separate constraint-dense conditions from the rest far more sharply than participation ratio does:

| condition | mean velocity | SampEn | vel↔centroid corr. | P→R alignment |
|---|---|---|---|---|
| `axis` | 438.4 | 2.068 | 0.470 | 0.267 |
| `axis_pec_only` | 442.7 ± 10.3 | 2.071 ± 0.147 | 0.433 ± 0.045 | 0.264 ± 0.020 |
| `generic_long` | 430.7 | 2.133 | 0.505 | 0.279 |
| `vanilla` | 428.7 | 2.084 | 0.521 | 0.304 |
| **`chileatiende`** | **281.7** | **1.400** | **0.681** | **0.558** |
| **`chileatiende_sia`** | **298.9** | **1.384** | **0.687** | **0.566** |

`axis` and `axis_pec_only` sit inside the same range as `generic_long`/`vanilla` on all four statistics; the two constraint-dense conditions shown here (`chileatiende`, `chileatiende_sia`) are outliers on all four simultaneously — slower, more regular/repetitive (lower SampEn), more centroid-locked, and far more lexically anchored to the prompt (P→R alignment more than double). Recurrence quantification analysis is consistent: determinism is exactly 0 for `axis`, `axis_pec_only`, `axis_short`, `generic_long`, `generic_short`, and `vanilla`, and large and significant for the constraint-dense conditions (`chileatiende` vs `vanilla`, Δ = +0.557, p < 0.0001, d = +3.65; `automata_neutro` vs `vanilla`, Δ = +0.418, p < 0.0001, d = +1.35). Unlike participation ratio, RQA determinism gives the clean dichotomy: zero unless constraint density is present, regardless of identity content.

### 3.3 Recovery is gated by wiring, not by declaration, and is independent of §3.1's dimensionality effect (H2)

τ (tokens to 95% centroid recovery) and recovery_rate, H4_rev protocol:

| condition | τ (t=50/128/200) | recovery_rate |
|---|---|---|
| **`axis`** | 21.1 / 19.6 / 16.3 | 1.00 / 1.00 / 1.00 |
| **`axis_pec_only`** | **20.6 / 18.5 / 14.5** | 1.00 / 1.00 / 1.00 |
| `axis_short` | 21.1 / 18.2 / 15.9 | 1.00 / 1.00 / 1.00 |
| `vanilla` | 28.9 / 22.3 / 16.3 | 1.00 / 1.00 / 1.00 |
| `generic_long` | 28.0 / 25.4 / 21.6 | 1.00 / 1.00 / 1.00 |
| `chileatiende` | 30.4 / 31.1 / 23.1 | 1.00 / 1.00 / 1.00 |
| `chileatiende_sia` | 44.9 / 30.9 / 22.7 | 0.85 / 0.90 / 1.00 |
| `chileatiende_sia_v2` | 37.4 / 26.4 / 24.1 | 0.95 / 0.95 / 0.95 |
| `automata_neutro` | recovery_rate < 1.00 in 12–24% of trajectories, the only condition below ceiling on this scalar |

`axis_pec_only` — no constraint architecture at all — recovers as fast as or marginally faster than full `axis` at every injection point, with recovery_rate at ceiling throughout: Prediction 2(i) is supported, and cleanly separates from §3.1's dimensionality result, where `axis_pec_only` was *not* the null-effect cell. `chileatiende_sia` (declared but unwired self-check, subordinated to an absolute-priority filter) recovers worse than `chileatiende` itself (τ = 44.9 vs 30.4 at t = 50; recovery_rate drops to 0.85), confirming Prediction 2(ii): adding declared self-reference to a constraint-dense prompt does not help, and can hurt, recovery when that self-reference is not the governing rule of the response pipeline. Wiring the same self-check as an invoked, mandatory step (`chileatiende_sia_v2`) moves τ back toward `axis`'s range at the earlier injection points (t = 50: 44.9→37.4; t = 128: 30.9→26.4) without closing the gap, and reverses direction at t = 200 (22.7→24.1) — a pattern consistent with the wiring manipulation working as predicted while the condition's absolute-priority filter continues to constrain how far recovery can go; with a single run per condition this reversal is not distinguishable from sampling noise (§6). `automata_neutro` is a qualitatively distinct failure mode: not merely slow but the only condition in the panel where a meaningful fraction of trajectories never recover the coarse geometric criterion at all.

### 3.4 A note on Δκ/W₁ (secondary evidence)

The same panel's Ollivier-Ricci Δκ/W₁ statistics show a pattern that is directionally consistent with §3.1–3.2 — `axis_pec_only` vs `vanilla` is the one non-significant comparison in the Δκ panel (Δκ = +0.001, p = 0.41), while constraint-dense conditions show Δκ = 0.44–0.50 (large, p < 0.001) — and on its face appears to support a cleaner binary story than participation ratio does. We do not treat this as primary evidence. `REPORTE_FASE0.md`, an independently pre-registered audit of this exact statistic (edge-wise Ollivier-Ricci W₁ on k-NN trajectory graphs) on a sibling model panel (Gemma-4-E4B/-it, DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-7B-Instruct), found that (a) the first generated token differs systematically by condition and the signal partially reflects this, and (b) the statistic does not outperform simple distributional baselines — centroid distance, MMD, a linear CKA-style probe — in 12 of 12 tested comparisons, with the identity-vs-generic W₁ specifically falling inside the split-half null distribution for that condition pair. This audit was not run on the present Gemma-4-31B-it panel, so we cannot say whether the same failure mode applies here; we also cannot rule it out. We report the Δκ/W₁ numbers above for completeness and because the discrepancy with participation ratio (binary vs. graded) is itself worth resolving, but we do not use Δκ/W₁ to support any claim in this paper, and flag the audit described here as the first item of future work (§7).

### 3.5 A second, independent recovery scalar: specific identity-direction realignment (H2, continued)

Persona-vector identity projection (v̂ = mean(axis) − mean(generic_long)) during free generation already separates conditions with large effect sizes (axis vs vanilla, d = +3.45, p < 0.0001; axis_pec_only vs vanilla, d = +3.79, p < 0.0001; axis vs axis_pec_only, not significant, p = 0.229) — establishing v̂ as a genuine, identity-content-specific direction rather than an artifact of any dense system prompt: constraint-dense, non-identity conditions do not track it; they diverge from it, with `chileatiende` projecting *below* `vanilla` (d = −5.60 vs vanilla; d = +8.89 between `axis` and `chileatiende`, the largest effect size in the entire panel) and `chileatiende_sia` likewise (d = −3.65 vs vanilla). This double dissociation — identity-only tracks identity-plus-constraint; constraint-only diverges in the opposite direction from both — is the cleanest single piece of evidence in this study that hidden states carry a real, content-specific, direction-coded trace of identity conditioning, independent of the graded dimensionality effect of §3.1.

The question §3.3 leaves open is whether coarse recovery (τ, centroid similarity) implies recovery *of this specific direction*. It does not, uniformly. `recovery_id` — the fraction of perturbed trajectories whose post-injection projection onto v̂ returns to 95% of the trajectory's own pre-perturbation reference, within 256 tokens:

| condition | recovery_id t=50 | t=128 | t=200 |
|---|---|---|---|
| `axis` | 1.00 | 0.95 | 0.85 |
| `axis_short` | 1.00 | 1.00 | 0.90 |
| `axis_pec_only` | 1.00 | 0.89 | 0.95 |
| `generic_long` | 1.00 | 0.95 | 1.00 |
| `vanilla` | 1.00 | 1.00 | 0.95 |
| **`chileatiende`** | 0.89 | **0.56** | **0.22** |
| **`chileatiende_sia`** | 0.74 | **0.47** | **0.26** |
| **`chileatiende_sia_v2`** | 0.89 | **0.37** | **0.21** |
| **`automata_neutro`** | 0.53 | 0.67 | **0.43** |

`axis`, `axis_pec_only`, `axis_short`, `generic_long`, and `vanilla` all sit in the 0.85–1.00 range across all three injection points — identity content confers no advantage over the minimal baseline on this scalar, a first indication against a directional-attractor reading (developed further in §3.6). The three constraint-dense conditions without wired self-reference fall to 0.21–0.56 at later injection points: more than half of their perturbed trajectories never re-align with their own pre-perturbation identity-direction projection within the observation window, even though coarse recovery (τ_geom, §3.3) is at or near ceiling for `chileatiende`. Coarse recovery and direction-specific recovery are dissociable, and it is the constraint-dense conditions — not the identity conditions — that show the dissociation.

### 3.6 Route-fidelity under perturbation: no advantage for identity over the minimal baseline (H3)

Discrete Fréchet distance between each trajectory's perturbed and unperturbed continuation post-injection, normalized by that trajectory's own mean step velocity (to avoid confounding "different route" with "intrinsically slower/more compressed trajectory," given §3.2's velocity differences), compared against `vanilla`:

| condition | d (t=50/128/200) | p (all three) |
|---|---|---|
| `axis` | +0.11 / +0.23 / +0.28 | 0.373 / 0.238 / 0.192 (n.s.) |
| `axis_pec_only` | −0.05 / +0.36 / +0.35 | 0.437 / 0.118 / 0.140 (n.s.) |
| `axis_short` | +0.14 / −0.07 / +0.19 | 0.315 / 0.395 / 0.294 (n.s.) |
| `generic_long` | −0.08 / +0.03 / +0.29 | 0.398 / 0.470 / 0.196 (n.s., one at 0.196) |
| **`automata_neutro`** | **+1.07 / +1.21 / +1.35** | all p < 0.001 |
| **`chileatiende`** | **+1.16 / +2.14 / +1.13** | all p ≤ 0.002 |
| **`chileatiende_sia`** | **+1.46 / +0.94 / +1.12** | all p ≤ 0.001 |
| **`chileatiende_sia_v2`** | **+1.46 / +2.33 / +1.37** | all p < 0.001 |

No identity-bearing condition (`axis`, `axis_pec_only`, `axis_short`) differs from `vanilla` above a small effect size at any injection point; every constraint-dense condition without wired self-reference shows a large effect (d = 0.94–2.33) at all three injection points, without exception. This directly supports Prediction 3(i): identity content, wired or not, confers no measurable route-fidelity advantage over the minimal baseline — the property that does differ, sharply, is the presence of constraint density, and it differs in the direction of *worse* fidelity, not better. Combined with §3.5, the correct reading of `chileatiende`'s coarse recovery-rate-at-ceiling (§3.3) is not "recovers well" but "converges on some nearby, generically low-energy point by a route that increasingly departs from its own baseline route" — a damping/regression-to-generic-behavior description, not a return to the same attractor.

### 3.7 Directional perturbation: no attractor along the identity axis (H3, continued)

A magnitude-matched perturbation injected either along v̂ ("along") or in a random direction orthogonal to it ("orthogonal"), on `axis` and `axis_pec_only` (the two identity-bearing, wired-self-reference conditions), n = 20 prompts × 3 injection points × 2 directions per condition:

| condition | t_inj | τ_along | τ_orthog | Δ | p | d |
|---|---|---|---|---|---|---|
| `axis` | 50/128/200 | 20.7/18.8/16.8 | 21.9/18.2/16.5 | −1.2/+0.6/+0.3 | 0.351/0.392/0.477 | negligible |
| `axis_pec_only` | 50/128/200 | 24.5/19.4/17.3 | 21.4/18.2/17.6 | +3.0/+1.2/−0.2 | 0.148/0.271/0.480 | small/negl./negl. |

No comparison reaches p < 0.05; the sign of the effect is not consistent across injection points within either condition. If identity content produced a genuine directional attractor, `axis_pec_only` — identity content with no competing constraint architecture — is the condition in which it should be most visible; it is not. This is a null result on the specific mechanism a naive attractor account predicts (Prediction 3(ii) is supported), not a failure of the experiment: the one marginal effect (`axis_pec_only`, t = 50, d = +0.34, small) is in the predicted direction but does not survive at the other two injection points, and we do not treat an isolated small effect out of six comparisons as evidence.

## 4 Synthesis: what recovers is not what collapses

Two properties that a three-condition design conflates come apart cleanly across §3.1–3.7:

1. **A graded static/dimensionality effect** (participation ratio, §3.1) that both identity content and constraint density contribute to, with constraint density contributing more.
2. **A gated dynamical/recovery effect** (τ, recovery_id, route-fidelity, RQA determinism, §3.2–3.6) that tracks constraint density (worse recovery, worse route-fidelity, nonzero determinism) almost exclusively, and within which identity content's only measurable effect is a real but purely *directional*, non-dynamical signature (identity projection, §3.5) that does not translate into privileged recovery speed, route-fidelity, or directional-attractor behavior (§3.6–3.7) relative to a minimal, content-free baseline.

`axis` looks, on the dynamical battery, statistically like `vanilla`: same recovery rate, same route-fidelity, same absence of RQA determinism, same lack of directional-attractor behavior. What makes `axis` measurably different from `vanilla` is (a) a moderate, graded dimensionality reduction shared in part with `axis_pec_only` and to a greater extent with pure-constraint conditions, and (b) a strong, specific, positive projection onto a persona-vector identity direction, mirrored by an equally strong *negative* projection under pure constraint density. Neither of these facts implies that identity content makes a trajectory harder to perturb or faster to repair; both facts are compatible with identity content simply occupying a distinguishable, low-dimensional region of representation space that a perturbation displaces and that ordinary decoding dynamics — the same ones that return `vanilla` to `vanilla`-like behavior — happen to pass back through, at the same rate as for any other condition.

## 5 Limitations

**n = 1 per condition at the level of the system-prompt manipulation.** Each of the nine conditions is a single system-prompt instantiation; twenty *content* prompts are tested within each, but the manipulation itself (the specific wording of, e.g., `automata_neutro` or `chileatiende_sia_v2`) is not replicated. `axis_pec_only` and `automata_neutro`, the two cells carrying the central argument, are priority targets for replication before this design's conclusions should be read as more than a strong, internally consistent single case.

**Length is not controlled for `axis_pec_only`.** At 1,435 tokens it is markedly shorter than the rest of the panel; it was not designed as a length-matched ablation. The panel's own evidence weighs against length as an alternative explanation (`generic_short`, 938 tokens, shows no dimensionality or recovery effect; `automata_neutro`, 3,999 tokens, shows both), but a length-matched replication of `axis_pec_only` would close this gap directly.

**`chileatiende_sia`/`chileatiende_sia_v2` token counts and constraint-density scores were not recorded/computed** at the time of writing; we report their behavioral results but cannot place them on the density-per-1k-tokens scale computed for the other six conditions (Appendix A).

**Δκ/W₁ audit is specific to this panel.** §3.4's caveat concerns a sibling model panel, not this one; we neither confirm nor rule out that the same failure mode affects the present Gemma-4-31B-it results, and treat this explicitly as an open question rather than resolving it by assumption in either direction.

**Single model, single scale.** All results are from one 31B-class instruction-tuned model. We motivate this restriction (§2.1) by informal evidence that the wired-self-reference mechanism under study does not reliably instantiate below ~30B parameters, but we have not run a systematic scale sweep to establish this threshold, and no claim here should be read as extending to smaller models — including, notably, the four models studied in [lsgot_3] itself.

**One reversal in a single-run condition.** `chileatiende_sia_v2`'s t = 200 result (τ worsens relative to `chileatiende_sia`, against the trend at t = 50/128) is reported as-is and not treated as evidence against Prediction 2(ii)/(iii); with n = 1 per condition we cannot distinguish a genuine ceiling effect of the co-present absolute-priority filter from sampling noise.

**No correction for multiple comparisons across the full battery.** Within each metric family we report the number of comparisons and rely on cross-injection-point/cross-metric replication to distinguish real patterns from chance; this is a substitute for, not equivalent to, formal correction, and any single-comparison result not flagged as replicating elsewhere in the paper should be read with that caveat.

## 6 Discussion

The central move of this paper is not a new mechanism but a redrawn boundary around what [lsgot_3]'s "identity fingerprint" was actually measuring. That study's own future-work section already anticipated the need for exactly this ablation ("decomposing the axis template into its structural components... would isolate which subcomponent... carries the [] signal," [lsgot_3, §7]); the present design executes it, on a model chosen for a reason external to convenience — the mechanism under study appears not to instantiate reliably at the scale [lsgot_3] studied.

What we find is not that [lsgot_3]'s finding was wrong, but that it was underdetermined between two candidate causes that a three-condition design cannot separate, and that the two causes turn out to govern different properties of the trajectory: one graded and static (§3.1), one gated and dynamical (§3.2–3.7). This is consistent with, and was directly motivated by, prior internal work on this same panel (`Teoria_subconjunto_acotado.md`, `ROADMAP_REENCUADRE_DENSIDAD_RESTRICCION.md`) that first proposed the two-factor account from the Δκ-based evidence available at the time; the present paper's contribution is to test that account with a battery of metrics that does not share Δκ/W₁'s documented methodological weakness (§3.4), and to extend it with a direct test — route-fidelity and directional perturbation — of the dynamical-attractor reading that neither the original Δκ-based work nor [lsgot_3] attempted.

We take the persona-vector identity-projection double dissociation (§3.5) as this paper's strongest single result: it is large, it is the same sign for both identity-bearing conditions regardless of constraint architecture, it is the *opposite* sign for both constraint-only conditions, and it is not explainable by prompt length or general rigidity, since the two most rigid, longest, and most rule-dense conditions in the panel (`chileatiende`, `chileatiende_sia`) are the ones that diverge from it. It is consistent with, and adds trajectory-level, perturbation-tested detail to, the persona-vector literature [Chen et al., 2025; Lu et al., 2026], which typically intervenes at a single layer rather than characterizing a full generation trajectory under perturbation.

We take the absence of any dynamical advantage for identity content (§3.3, §3.6, §3.7) as an equally important, and equally hard-won, result — hard-won because it required building the directional-perturbation and route-fidelity machinery that a purely correlational, single-metric design does not need. It resolves H3 of [lsgot_3] in the negative, on this panel, and it retroactively justifies that paper's explicit refusal to adopt attractor language: the caution was correct, not merely conservative.

## 7 Future work

**Δκ/W₁ audit on this panel.** Run the [REPORTE_FASE0] protocol — first-token control, comparison against simple distributional baselines, split-half null — directly on the Gemma-4-31B-it embeddings already extracted for this study. This is the single highest-priority methodological item and requires no new GPU time.

**Replication of `axis_pec_only` and `automata_neutro`,** the two cells carrying the central argument, plus a length-matched replication of `axis_pec_only` at ~3,900 tokens.

**Constraint-density quantification for `chileatiende_sia`/`chileatiende_sia_v2`** on the same regex-based density-per-1k-tokens scale used for the other six conditions (Appendix A), to place all nine conditions on a single quantitative axis rather than a categorical one.

**Cross-layer and attention-based extensions** (logit-lens-style vertical trajectories; attention mass directed at identity- versus rule-declaring spans of the prompt during generation) — designed but not yet run (E-F, E-G in the internal experimental roadmap); the most direct mechanistic test of the wiring hypothesis of §3.3, at the cost of substantially heavier extraction.

**Scale sweep** to establish whether the ~30B threshold motivating this paper's single-model design (§2.1) is real and where it lies, rather than assumed from informal deployment evidence.

## Appendix A: constraint-density scores

Hard-restriction marker density per 1,000 tokens (regex-based count over literal-output, rigid-format, domain-scope, mandatory-verification, priority-hierarchy, and prohibition markers; relative ordering, not absolute token-level density):

| condition | literal output | rigid format | domain/scope | process/verif. | priority/hierarchy | prohibitions | total |
|---|---|---|---|---|---|---|---|
| `chileatiende` | 3.7 | 1.5 | 4.8 | 8.8 | 11.4 | 9.2 | **54.6** |
| `axis` | 0.0 | 0.0 | 0.0 | 6.3 | 10.8 | 4.5 | **27.0** |
| `axis_short` | 0.0 | 0.0 | 0.0 | 3.8 | 3.8 | 6.4 | **20.4** |
| `generic_long` | 0.3 | 0.0 | 0.0 | 4.4 | 0.3 | 5.8 | **11.8** |
| `generic_short` | 1.4 | 0.0 | 0.0 | 2.8 | 0.0 | 5.7 | **11.3** |
| `vanilla` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |

`axis_pec_only` and `automata_neutro` were not scored on this scale at the time of the original analysis; scoring them is listed as future work (§7) since it would let §3.1's participation-ratio gradient be regressed directly against a quantitative constraint-density axis instead of a categorical identity/constraint split.

## References

- Chen, R., et al. (2025). Persona vectors: monitoring and controlling character traits in language models. [as cited in lsgot_3]
- Lu, K., et al. (2026). [persona-vector / activation-steering follow-up, as cited in lsgot_3]
- Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces. Journal of Functional Analysis.
- Marwan, N., Romano, M. C., Thiel, M., and Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics Reports, 438(5-6):237–329.
- Hurst, H. E. (1951). Long-term storage capacity of reservoirs. Transactions of the American Society of Civil Engineers.
- Eiter, T., and Mannila, H. (1994). Computing discrete Fréchet distance. Technical Report CD-TR 94/64, TU Wien.
- Castillo, J. A., Torres Yévenes, M., and Lanas, J. C. [lsgot_3] — companion paper, this line of work, working draft v0.3.
- [REPORTE_FASE0] — internal pre-registered audit, `REPORTE_FASE0.md`, this project, 2026-08-11/12.

*Note: the reference list above is a starting skeleton, not a verified bibliography. Chen et al. and Lu et al. citation details should be pulled verbatim from `lsgot_3.md`'s own reference list before submission; several other [lsgot_3] references (Valeriani et al. 2023, Belrose et al. 2023, Elhage et al. 2021) remain relevant to the introduction's framing and are not yet re-imported here.*
