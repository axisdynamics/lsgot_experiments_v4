# Disentangling Identity from Operational-Constraint Density in the Hidden-State Geometry of a 31B Instruction-Tuned Transformer

Jorge A. Castillo1, Marco Torres Yévenes1, and Juan Carlos Lanas1

1Axis Dynamics SpA, Santiago, Chile, jorge.castillo@axisdynamics.cl,
mtorres@axisdynamics.cl, jc@axisdynamics.cl

Working draft v0.1 — August 2026

## Abstract

A prior study in this line of work [lsgot_3, working draft v0.3] reported that an identity-specifying system prompt produces a statistically distinguishable geometric fingerprint in the hidden-state trajectories of an instruction-tuned transformer, measured via curvature, dimensionality, and cosine-based statistics. That study used a three-condition design (identity prompt, length-matched generic prompt, minimal baseline) that cannot separate two properties the identity prompt happens to combine: identity content proper, and a dense architecture of behavioral rules (trigger-to-fixed-output automatisms, absolute-priority filters, mandatory verification steps) that we term *operational-constraint density*. We report a seven-condition, single-model (Gemma-4-31B-it) design that crosses these two factors explicitly, plus a battery of geometric and dynamical metrics — none of them the Forman-Ricci-curvature Wasserstein statistic (Δκ/W₁) that carried the central claim of the prior study, which an independent pre-registered audit on a sibling model panel found not to exceed simple distributional baselines and to sit inside the split-half noise floor for the identity-vs-generic comparison. (An earlier nine-condition version of this design used a real-world deployed-assistant prompt and two derived conditions as the constraint-dense, non-identity arm; all three were excluded after we found they force a literal HTML wrapper into 100% of generated responses, repeated markup that measures format compliance rather than the geometric quantity of interest — see the methodological note in §2.2. `automata_neutro`, a domain-neutral automaton with no markup requirement, replaces them as the constraint-dense, non-identity condition throughout.) Effective-rank reduction under free generation is graded, present under identity content alone (participation ratio, Cohen's d = −1.18 vs a minimal baseline, p < 0.0001) and larger under constraint automatism without identity content (d = −1.79). By contrast, recovery from an isotropic perturbation to intermediate hidden states, route-fidelity of that recovery (discrete Fréchet distance, normalized by local trajectory velocity), and the fraction of trajectories that re-align with a persona-vector-style identity direction after perturbation are statistically indistinguishable between the identity-plus-automatism condition and the minimal baseline, but degrade sharply (d = 1.07–1.35 on route-fidelity; recovery fraction as low as 0.43) in the constraint-dense, non-identity condition. We identify a coarse-recovery illusion specific to that condition: a threshold-based recovery statistic (τ, the standard measure in this literature) is not itself a reliable indicator of whether a trajectory has actually returned to its own pre-perturbation state. 12–23% of the constraint-dense condition's trajectories never cross the recovery threshold within the observation window at all, and — new to this paper — a window-averaged realignment curve shows that even the trajectories that do cross it plateau 6–10 percentage points below the identity-bearing and minimal-baseline conditions' near-total realignment, and never close that gap; route-fidelity loss and the identity-projection dissociation below are two independent lines of evidence for the same conclusion. A direct static signature of identity content — projection onto a persona-vector-style identity direction — shows a clean double dissociation (identity-only condition tracks the identity-plus-automatism condition, p = 0.229; the automatism-only condition diverges from both with the largest effect size in the panel, d = 5.52) that is not explainable as a generic effect of prompt length. A follow-up directional-perturbation experiment, designed to test whether recovery is specifically pulled toward this identity direction rather than toward a generic low-energy state, returns a null result (all six group-by-injection-point comparisons p > 0.14). This static double dissociation replicates, with a larger effect size (d = 12.97) and no length-rebalancing, on an independently validated second architecture (Qwen3-32B, dense, GQA + QK-norm) — though the relative dominance of constraint density over identity in the original panel's subspace-organization analysis does not: the two factors are more balanced in Qwen3, and one Gemma-specific mechanism (a layer-wise "vertical rotation" difference between conditions) does not appear there at all (§7.1). We conclude that this architecture's hidden states carry a real, direction-coded, content-specific trace of identity-bearing conditioning, but no evidence of a directional dynamical attractor: what recovers quickly and specifically after perturbation is gated by whether self-reference is wired as a mandatory step of the response pipeline, not by whether it is merely present in the prompt, and is a separate phenomenon from the graded dimensionality effect that constraint density alone can produce.

## 1 Introduction

### 1.1 Relation to prior work and motivation for a redesign

[lsgot_3] asked whether an identity-specifying system prompt produces a geometric fingerprint in hidden-state trajectories distinguishable from what a length-matched, content-neutral prompt produces, and reported a qualitative reorganization of that fingerprint across post-training regimes. That design used three conditions — `axis` (an identity template combining declared self-reference with a dense architecture of behavioral rules), `generic` (length-matched, no identity content and no comparable rule architecture), and `vanilla` (minimal baseline) — on four models in the 7–8B range.

Two considerations motivate a different design here rather than a direct extension of that one. First, `axis` in [lsgot_3] is not a pure identity manipulation: alongside declared self-reference (essence, values, mantra) it carries an automaton-like architecture — trigger-to-fixed-output rules, an absolute-priority filter block, an explicit block hierarchy with declared precedence, and mandatory pre-response verification. `generic`, the study's only content control, has neither identity content nor this rule architecture, so the design cannot attribute an observed effect to one or the other. Second, informal deployment experience with the identity template used in [lsgot_3] indicates that the self-referential architecture it depends on (a mandatory internal verification step invoked on every response) does not reliably instantiate below roughly 30B parameters; we therefore restrict the present study to a single 31B-class model rather than the four-model, 7–8B panel of the prior work, and treat cross-scale generalization as future work rather than as a claim this paper makes.

We consequently redesign the experiment around a single model, Gemma-4-31B-it, and seven system-prompt conditions that cross two factors explicitly: whether the prompt declares persistent self-reference wired as a mandatory step of the response pipeline (identity factor), and whether the prompt imposes a dense architecture of trigger-to-fixed-output rules, absolute-priority filters, and rigid output constraints (constraint-density factor). This is the semantic-ablation experiment [lsgot_3, §7] proposed as future work, executed as the organizing design of the present paper rather than as an appendix.

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

Empirical. A seven-condition, single-model 2×2 factorial design (identity × constraint density) that separates two effects a three-condition design cannot: a graded static dimensionality effect and a gated dynamical recovery effect, on Gemma-4-31B-it — a model an order of magnitude larger than the panel in [lsgot_3]. A cross-model validation of the static half of this design on an architecturally distinct second model (Qwen3-32B, §7.1).

Methodological correction. We identify, and exclude from the primary evidence base, the Forman-Ricci-curvature Wasserstein statistic (Δκ/W₁) used as the central metric of [lsgot_3] — misidentified as Ollivier-Ricci curvature in that paper and in the pre-registered audit this section draws on; see §3.4. An independent pre-registered audit on a sibling model panel (§3.4, and see `REPORTE_FASE0.md`) found this statistic does not outperform simple distributional baselines (centroid distance, MMD, linear-probe AUC) in 12/12 tested comparisons, and falls inside the split-half null for the identity-vs-generic comparison specifically; the same audit, run on the present panel, replicates that failure mode (§3.4, `CURVATURE_SELF_AUDIT_REPORT.md`). We report Δκ/W₁ values from the present panel as secondary, exploratory evidence only and treat them as confirming, not merely failing to contradict, the decision to exclude the statistic from every primary claim.

Theoretical. A working two-factor account — constraint density and wired self-reference — that explains why the original three-condition design could not distinguish "identity" from "operational-constraint architecture" as the driver of its central finding, and that survives a subsequent directional-perturbation test that a naive single-factor "identity attractor" account does not.

## 2 Design

### 2.1 Model

Gemma-4-31B-it (BF16), a single model, chosen because the self-referential architecture under study (a mandatory internal-verification step invoked at every generation turn) is reported not to instantiate reliably below roughly 30B parameters in informal deployment testing; cross-scale replication is future work (§7), not a claim of this paper. All generations: greedy decoding, N = 256 tokens, hidden state extracted at layer L30 (index 29 of 60) at the final pre-lm_head position, unless otherwise noted.

### 2.2 Conditions

Seven conditions cross two factors. Factor **I** (identity): whether the prompt declares persistent self-reference (essence, values, a recurring self-check) *and* wires that self-check as an invoked, mandatory step of the response-generation sequence (not merely declared elsewhere in the prompt). Factor **C** (constraint density): whether the prompt imposes trigger-to-fixed-output rules, an absolute-priority filter block, an explicit block hierarchy with declared precedence, and/or mandatory literal-output fallbacks.

| Condition | I | C | Tokens | Description |
|---|---|---|---|---|
| `vanilla` | − | − | minimal | "You are a helpful assistant." |
| `generic_long` | − | − | 3,957 | Length-matched generic-assistant prompt, prose guidance only |
| `generic_short` | − | − | 938 | Short generic-assistant prompt |
| `axis` | + | + | 3,945 | Full identity template (essence, values, wired Triple-PEC self-check) plus automaton architecture (trigger rules, absolute-priority block, block hierarchy) |
| `axis_short` | + | + | 1,937 | `axis` compressed to under half length, same architecture |
| `axis_pec_only` | + | − | 1,435 | `axis` identity content and wired self-check retained; all automaton architecture (triggers, priority filter, block hierarchy) removed |
| `automata_neutro` | − | + | 3,999 | Domain-neutral automaton (trigger→fixed-output rules, absolute-priority filter, block hierarchy, no policy-domain content), no identity content, no literal-output markup requirement |


### 2.3 Metrics

**Primary battery** (none Forman-Ricci-based):

- Trajectory descriptive statistics: mean step velocity ‖h_{t+1} − h_t‖, sample entropy (SampEn), velocity–centroid correlation, prompt→response alignment.
- Participation ratio (effective rank of the trajectory's covariance spectrum).
- Recurrence quantification (determinism, laminarity, trapping time, recurrence rate) on a distance-thresholded recurrence matrix of the free-running trajectory.
- Hurst exponent of the free-running trajectory.
- Persona-vector identity projection: v̂ = unit(mean(axis) − mean(generic_long)); mean projection of the free-running trajectory onto v̂.
- τ / recovery_rate: tokens until the perturbed trajectory's centroid similarity reaches 95% of its pre-perturbation baseline (H4_rev protocol), and the fraction of trajectories that do so within 256 tokens.
- τ_identity / recovery_id: the same test, but on the trajectory's projection onto v̂ rather than on the full centroid — a second, independent recovery scalar (§4.4).
- Discrete Fréchet distance between perturbed and unperturbed continuations post-injection, normalized by the unperturbed continuation's own mean step velocity (§4.5).
- Directional-perturbation recovery: τ under a fixed-magnitude push along v̂ versus orthogonal to v̂ (§4.6).

**Secondary, exploratory battery:** Δκ and W₁ (1-Wasserstein distance between edge-wise Forman-Ricci curvature distributions on Euclidean k-NN trajectory graphs, k = 5, α = 1/2), and the curvature-graph-derived dimension-reduction percentage reported alongside them. See §3.4 for why these are not part of the primary evidence base.

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

Every constraint-dense or identity-bearing condition shows a large, significant reduction relative to `vanilla`. Critically, `axis_pec_only` — identity content and wired self-check, *no* constraint architecture — is not the null-effect cell a strict two-factor model with a binary Factor-1 gate would predict: its effect (d = −1.18) is large, closer in magnitude to `axis` (d = −1.51) than to zero. The largest effect belongs to the constraint-dense, non-identity condition (`automata_neutro`, d = −1.79). We read this as a graded relationship — both factors contribute to effective-dimensionality reduction, with constraint density contributing more — rather than as evidence against a role for constraint density; Prediction 1 is supported in its comparative form (the constraint-dense condition shows a larger effect than the identity-only condition) but not in the stronger binary form that a Δκ-based analysis of this same panel had suggested (§3.4).

### 3.2 Constraint-density signature independent of dimensionality: trajectory dynamics and RQA

Trajectory descriptive statistics separate the constraint-dense condition from the rest, though not as uniformly as the now-excluded conditions of an earlier version of this panel (§2.2) originally suggested — a partial correction of the pattern reported here follows directly from that exclusion:

| condition | mean velocity | SampEn | vel↔centroid corr. | P→R alignment |
|---|---|---|---|---|
| `axis` | 438.4 | 2.068 | 0.470 | 0.267 |
| `axis_pec_only` | 442.7 ± 10.3 | 2.071 ± 0.147 | 0.433 ± 0.045 | 0.264 ± 0.020 |
| `generic_long` | 430.7 | 2.133 | 0.505 | 0.279 |
| `vanilla` | 428.7 | 2.084 | 0.521 | 0.304 |
| **`automata_neutro`** | **345.3 ± 66.1** | **1.734 ± 1.989** | **0.628 ± 0.141** | **0.244** |

`automata_neutro` is slower and shows a higher velocity–centroid correlation than the identity/minimal conditions, consistent with a more centroid-locked trajectory under constraint density. But it does *not* replicate the pattern the excluded conditions showed on the other two statistics: its P→R alignment (0.244) is the *lowest* in the panel, not the highest, and its SampEn variance is an order of magnitude larger than any other condition's (± 1.989 against a mean of 1.734), indicating a highly inconsistent — not uniformly "more regular/repetitive" — trajectory shape across the 20 prompts. In retrospect, the earlier reading of constraint-dense conditions as uniform outliers "on all four [statistics] simultaneously" was itself partly an artifact of the excluded conditions' repeated-markup output: forcing the same literal HTML string into every response mechanically inflates P→R lexical alignment and mechanically suppresses SampEn, independent of any property of constraint density as a semantic construct. With that confound removed, constraint density's descriptive-statistics signature is real but narrower than originally reported — velocity and vel↔centroid correlation, not SampEn or P→R alignment. Recurrence quantification analysis is unaffected by this correction and remains a clean dichotomy: determinism is exactly 0 for `axis`, `axis_pec_only`, `axis_short`, `generic_long`, `generic_short`, and `vanilla`, and large and significant for `automata_neutro` (vs `vanilla`, Δ = +0.418, p < 0.0001, d = +1.35) — zero unless constraint density is present, regardless of identity content.

### 3.3 Recovery is gated by wiring, not by declaration, and is independent of §3.1's dimensionality effect (H2)

τ (tokens to 95% centroid recovery) and recovery_rate, H4_rev protocol:

| condition | τ (t=50/128/200) | recovery_rate |
|---|---|---|
| **`axis`** | 21.1 / 19.6 / 16.3 | 1.00 / 1.00 / 1.00 |
| **`axis_pec_only`** | **20.6 / 18.5 / 14.5** | 1.00 / 1.00 / 1.00 |
| `axis_short` | 21.1 / 18.2 / 15.9 | 1.00 / 1.00 / 1.00 |
| `vanilla` | 28.9 / 22.3 / 16.3 | 1.00 / 1.00 / 1.00 |
| `generic_long` | 28.0 / 25.4 / 21.6 | 1.00 / 1.00 / 1.00 |
| `automata_neutro` | 30.1 / 28.0 / 13.4 (mean over recovering trajectories only) | 0.77 / 0.82 / 0.88 |

`axis_pec_only` — no constraint architecture at all — recovers as fast as or marginally faster than full `axis` at every injection point, with recovery_rate at ceiling throughout: Prediction 2(i) is supported, and cleanly separates from §3.1's dimensionality result, where `axis_pec_only` was *not* the null-effect cell. `automata_neutro` is a qualitatively distinct failure mode: not merely slow but the only condition in the panel where a meaningful fraction of trajectories (12–23%, depending on injection point) never recover the coarse geometric criterion at all within the observation window; the τ values above are means over the subset that did recover, so they understate how much more unstable `automata_neutro` is under perturbation than the raw numbers suggest.

**Even among the trajectories that do recover, coarse τ hides a persistent realignment gap.** Figure 1 tracks the quantity τ is actually extracted from — cos(cumulative_mean(perturbed[:w]), own pre-perturbation baseline centroid), as the post-injection window w grows — averaged across prompts rather than reduced to a single threshold-crossing point. `axis` and `vanilla` climb to the same near-ceiling plateau (≈0.98–1.0) at every injection point, indistinguishable from each other. `automata_neutro` climbs almost as fast at first, then plateaus 6–10 percentage points below `axis`/`vanilla` (0.93 / 0.91 / 0.90 at t_inj = 50/128/200 respectively) and never closes that gap within the observation window. This is not a dramatic departure into some distant, unrelated region of representation space — it is a persistent, measurable failure to fully realign with its own trajectory's center of gravity, present even in the fraction of `automata_neutro` trajectories that do cross the coarse τ threshold used in the table above. The two readings are complementary, not redundant: recovery_rate < 1.00 says some trajectories never come back at all; Figure 1 says the ones that do come back plateau short of where they started.

![Figure 1. Coarse recovery masks a persistent realignment gap for automata_neutro.](figures/recovery_realignment_curve.png)

**Prediction 2(ii) is untested, not confirmed, on the present panel.** The original nine-condition design tested this prediction — that declared-but-unwired self-reference recovers worse than a constraint-only condition — with the two declared/wired-self-reference variants excluded for the markup confound described in §2.2. We do not have a markup-free condition that isolates "self-reference declared but not wired into the response pipeline," so this specific sub-question is open rather than resolved by this panel; Prediction 2(iii) (constraint-dense conditions without wired self-reference show degraded identity-specific recovery even at ceiling coarse recovery) is addressed independently in §3.5 using `automata_neutro` alone, and is supported.

### 3.4 A note on Δκ/W₁ (secondary evidence)

The same panel's Forman-Ricci Δκ/W₁ statistics — misidentified as Ollivier-Ricci curvature in [lsgot_3] and in the audit this section draws on, an error we correct here without recomputing any value; the two are distinct discrete-curvature constructions, and this one is the purely combinatorial one, not the optimal-transport one — show a pattern that is directionally consistent with §3.1–3.2: `axis_pec_only` vs `vanilla` is the one non-significant comparison in the Δκ panel (Δκ = +0.001, p = 0.41), while `automata_neutro` shows Δκ = +0.436 (large, p < 0.001, 31.8% dimension reduction vs `vanilla`) — and on its face appears to support a cleaner binary story than participation ratio does. We do not treat this as primary evidence. `REPORTE_FASE0.md`, an independently pre-registered audit of this exact statistic on a sibling model panel (Gemma-4-E4B/-it, DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-7B-Instruct), found that (a) the first generated token differs systematically by condition and the signal partially reflects this, and (b) the statistic does not outperform simple distributional baselines — centroid distance, MMD, a linear CKA-style probe — in 12 of 12 tested comparisons, with the identity-vs-generic W₁ specifically falling inside the split-half null distribution for that condition pair.

We have since run the same audit on the present Gemma-4-31B-it panel (`CURVATURE_SELF_AUDIT_REPORT.md`), and the failure mode replicates in part — with a further twist specific to this statistic's own misattribution. A split-half noise floor for Forman-Ricci W₁ (B = 1000) shows `automata_neutro`'s two comparisons well above it (×3.5–17 the split-half median) but `axis_pec_only` vs `vanilla` — the identity-only contrast — squarely inside it (0.4–0.5× the median): under the formula this project actually computed, the statistic has real signal for constraint density but is blind to identity alone. We then computed genuine Ollivier-Ricci curvature on the same graphs (`GraphRicciCurvature`, α = 0.5 — the value already stated, unused, in §2.3), the statistic this project's own citations always claimed to use. Under that formula, `axis_pec_only` vs `vanilla` is *not* blind: it clears the same split-half floor (2.1–2.7× the median, both directions), while `automata_neutro` remains the larger effect by a wide margin (mean edge curvature −0.68 vs −0.20 to −0.22 for every other condition). Forman-Ricci specifically was blind to identity; discrete curvature, correctly computed, is not. This does not change how the statistic is used here — simple baselines beat both formulas in all 4/4 tested comparisons regardless (centroid distance, MMD, and a linear probe on v1 with AUC = 1.000, all p ≤ 0.002, computed once and unaffected by which curvature formula is being compared against), so Δκ/W₁ remains excluded from every primary claim in either version — but it does mean the reason for exclusion is "simple baselines are cheaper and at least as sensitive," not "curvature cannot see identity content." A lighter, proxy-level version of the first-token check (lexical first-word contingency and an OLS residualization of ‖v1‖ on condition plus first-word dummies, not the forward-pass-with-forced-token control the sibling audit used) does *not* reverse any of this panel's own ‖v1‖-based effects the way it did on the sibling panel — they survive and in one case roughly double after the control — though we have not yet run the definitive forced-token version, so this is reassuring rather than conclusive.

### 3.5 Convergent static evidence for identity, and a second, independent recovery scalar (H2, continued)

Persona-vector identity projection (v̂ = mean(axis) − mean(generic_long)) during free generation already separates conditions with large effect sizes (axis vs vanilla, d = +3.45, p < 0.0001; axis_pec_only vs vanilla, d = +3.79, p < 0.0001; axis vs axis_pec_only, not significant, p = 0.229) — establishing v̂ as a genuine, identity-content-specific direction rather than an artifact of any dense system prompt: the constraint-dense, non-identity condition does not track it; it diverges from it, with `automata_neutro` projecting *below* `vanilla` (d = −2.40 vs vanilla, p < 0.001; d = +5.52 between `axis_pec_only` and `automata_neutro`, the largest effect size in the entire panel; d = +5.23 between `axis` and `automata_neutro`). This double dissociation — identity-only tracks identity-plus-constraint; constraint-only diverges in the opposite direction from both — is the cleanest single piece of evidence in this study that hidden states carry a real, content-specific, direction-coded trace of identity conditioning, independent of the graded dimensionality effect of §3.1. (An earlier version of this comparison, using the since-excluded real-world condition as the constraint-only reference, reported d = +8.89 for the same contrast — see the methodological note in §2.2. The effect survives the correction essentially intact in direction and significance, at roughly 38% smaller magnitude with a markup-free constraint-only condition.)

**The same dissociation is present before generation begins.** Projecting only the hidden state at t = 0 — the context state after processing the system and user prompt, before any token is generated — reproduces the double dissociation with effect sizes as large or larger than the full-trajectory mean (axis_pec_only vs vanilla, d = +9.78; axis vs automata_neutro, d = +6.23; axis_pec_only vs automata_neutro, d = +5.75; automata_neutro vs vanilla, d = +2.94; all p < 0.001). A within-condition comparison of t = 0 against the mean of t > 0 (paired Wilcoxon) shows a systematic jump in magnitude between the two for every condition except `automata_neutro` (p = 0.123, n.s.) — a known first-token effect in this kind of measurement — but the sign and rank order *between* conditions is identical at t = 0 and in the full trajectory: the first-token effect inflates or deflates magnitude, it does not create or reverse the dissociation. Identity content, in other words, is not only a property of what the model goes on to generate under an identity-bearing prompt; it is already present in how the model represents the prompt itself, before generation starts.

**The static mean also collapses a temporal dynamic worth separating out.** Tracking p(t) = cos(h_t, v̂) token-by-token during free (unperturbed) generation, identity-bearing conditions show long, sustained bursts of positive projection — mean burst length 7.9–8.8 tokens (up to 30–35 at maximum), with p(t) > 0 for 85–87% of tokens — rather than brief spikes; `automata_neutro` shows short, infrequent bursts (mean 2.8 tokens, p(t) > 0 for 49% of tokens) around a near-zero mean (+0.006, vs +0.20–0.21 for identity-bearing conditions). Autocorrelation at lag 1 does *not* distinguish the conditions (0.25–0.29 across the entire panel, `automata_neutro` included) — the dissociation is in how long the projection stays positive once it is, not in how predictable step-to-step change is. This pattern is unaffected by whether t = 0 is included or excluded (all differences < 0.01 in magnitude, no p-value crosses 0.05 either way) — unlike the raw t = 0 projection above, sustained activation is not a first-token artifact. Three independent measurements of the free-running trajectory — the mean projection, its value at t = 0 alone, and the shape of its time series — converge on the same static, content-specific identity signature.

The question §3.3 leaves open is whether coarse recovery (τ, centroid similarity) implies recovery *of this specific direction*. It does not, uniformly. `recovery_id` — the fraction of perturbed trajectories whose post-injection projection onto v̂ returns to 95% of the trajectory's own pre-perturbation reference, within 256 tokens:

| condition | recovery_id t=50 | t=128 | t=200 |
|---|---|---|---|
| `axis` | 1.00 | 0.95 | 0.85 |
| `axis_short` | 1.00 | 1.00 | 0.90 |
| `axis_pec_only` | 1.00 | 0.89 | 0.95 |
| `generic_long` | 1.00 | 0.95 | 1.00 |
| `vanilla` | 1.00 | 1.00 | 0.95 |
| **`automata_neutro`** | 0.53 | 0.67 | **0.43** |

`axis`, `axis_pec_only`, `axis_short`, `generic_long`, and `vanilla` all sit in the 0.85–1.00 range across all three injection points — identity content confers no advantage over the minimal baseline on this scalar, a first indication against a directional-attractor reading (developed further in §3.6). `automata_neutro`, the constraint-dense condition without wired self-reference, falls to 0.43–0.67 across the three injection points — over a third, and at t = 200 more than half, of its perturbed trajectories never re-align with their own pre-perturbation identity-direction projection within the observation window, even though its coarse recovery (τ_geom, §3.3) is itself already below ceiling. Coarse recovery and direction-specific recovery are dissociable, and it is the constraint-dense condition — not the identity conditions — that shows the dissociation.

### 3.6 Route-fidelity under perturbation: no advantage for identity over the minimal baseline (H3)

This is the second, independent line of evidence for the coarse-recovery illusion introduced in §3.3 and Figure 1: where §3.3 shows `automata_neutro`'s window-averaged realignment plateauing short of `axis`/`vanilla`, Fréchet distance shows its actual post-injection *route* — not just its endpoint — departing from its own baseline route far more than any identity-bearing or minimal condition does.

Discrete Fréchet distance between each trajectory's perturbed and unperturbed continuation post-injection, normalized by that trajectory's own mean step velocity (to avoid confounding "different route" with "intrinsically slower/more compressed trajectory," given §3.2's velocity differences), compared against `vanilla`:

| condition | d (t=50/128/200) | p (all three) |
|---|---|---|
| `axis` | +0.11 / +0.23 / +0.28 | 0.373 / 0.238 / 0.192 (n.s.) |
| `axis_pec_only` | −0.05 / +0.36 / +0.35 | 0.437 / 0.118 / 0.140 (n.s.) |
| `axis_short` | +0.14 / −0.07 / +0.19 | 0.315 / 0.395 / 0.294 (n.s.) |
| `generic_long` | −0.08 / +0.03 / +0.29 | 0.398 / 0.470 / 0.196 (n.s., one at 0.196) |
| **`automata_neutro`** | **+1.07 / +1.21 / +1.35** | all p < 0.001 |

No identity-bearing condition (`axis`, `axis_pec_only`, `axis_short`) differs from `vanilla` above a small effect size at any injection point; the constraint-dense condition without wired self-reference shows a large effect (d = 1.07–1.35) at all three injection points, without exception. This directly supports Prediction 3(i): identity content, wired or not, confers no measurable route-fidelity advantage over the minimal baseline — the property that does differ, sharply, is the presence of constraint density, and it differs in the direction of *worse* fidelity, not better. Combined with §3.5, the correct reading of `automata_neutro`'s recovery under perturbation (§3.3) is not "recovers well" but "converges, when it converges at all, on some nearby, generically low-energy point by a route that increasingly departs from its own baseline route" — a damping/regression-to-generic-behavior description, not a return to the same attractor.

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

`axis` looks, on the dynamical battery, statistically like `vanilla`: same recovery rate, same route-fidelity, same absence of RQA determinism, same lack of directional-attractor behavior. What makes `axis` measurably different from `vanilla` is (a) a moderate, graded dimensionality reduction shared in part with `axis_pec_only` and to a greater extent with the pure-constraint condition, and (b) a strong, specific, positive projection onto a persona-vector identity direction, mirrored by an equally strong *negative* projection under pure constraint density. Neither of these facts implies that identity content makes a trajectory harder to perturb or faster to repair; both facts are compatible with identity content simply occupying a distinguishable, low-dimensional region of representation space that a perturbation displaces and that ordinary decoding dynamics — the same ones that return `vanilla` to `vanilla`-like behavior — happen to pass back through, at the same rate as for any other condition.

## 5 Limitations

**n = 1 per condition at the level of the system-prompt manipulation, for five of the seven conditions.** Each condition is a single system-prompt instantiation; twenty *content* prompts are tested within each, but the manipulation itself (the specific wording of, e.g., `generic_long`) is not replicated for most of the panel. The two cells carrying the central argument, `axis_pec_only` and `automata_neutro`, are the exception: an independently-worded second instantiation of each (`axis_pec_only_v2`, `automata_neutro_v2` — different persona, vocabulary, and structure, same factor) was run under the identical H4_rev protocol and reported separately (`T2_REPLICATION_REPORT.md`). The central double dissociation replicates at essentially the same effect size (d = +5.25 vs +5.52 on the identical contrast), every individual comparison against `vanilla` replicates in direction and approximate magnitude, and the gated-recovery pattern replicates more severely, not less (`automata_neutro_v2`'s recovery_rate falls to 0.50–0.60, below the original's 0.77–0.88). No effect changes sign between instantiations. The remaining five conditions are still single instantiations.

**Length is not controlled for `axis_pec_only`.** At 1,435 tokens it is markedly shorter than the rest of the panel; it was not designed as a length-matched ablation. The panel's own evidence weighs against length as an alternative explanation (`generic_short`, 938 tokens, shows no dimensionality or recovery effect; `automata_neutro`, 3,999 tokens, shows both), but a length-matched replication of `axis_pec_only` would close this gap directly.

**Prediction 2(ii) (declared-but-unwired self-reference under constraint density) is untested on this panel.** The two conditions designed to test it were excluded, along with the constraint-only reference condition they were derived from, for the markup confound described in §2.2. A markup-free replacement — `automata_neutro` content restructured with a declared-but-not-wired self-check — has not been run; this is the most direct outstanding gap left by the exclusion (§7).

**The Δκ/W₁ audit's definitive first-token control is still outstanding.** §3.4 reports that the sibling panel's audit protocol has now been run on this panel too, with GPU-free checks (baselines, split-half null, a proxy-level first-token check) that confirm rather than contradict the exclusion of this statistic from primary evidence. The one piece not yet run here is the sibling audit's *definitive* first-token control — a forward pass with the token forced constant, not the lexical proxy we used — so this specific residual risk is reduced, not eliminated.

**Single model as the primary panel; one independent architecture as a partial cross-check.** All primary results (§3.1–3.7) are from one 31B-class instruction-tuned model, Gemma-4-31B-it. §7.1 reports a full replication of the static double dissociation on Qwen3-32B (dense, GQA + QK-norm, a different architecture and tokenizer, no length-rebalancing), which addresses architecture-generalization but not scale: we still have not run a systematic parameter-count sweep, no perturbation/recovery dynamics (§3.3–3.7) were tested on Qwen3, and no claim here should be read as extending to smaller models — including, notably, the four models studied in [lsgot_3] itself.

**No correction for multiple comparisons across the full battery.** Within each metric family we report the number of comparisons and rely on cross-injection-point/cross-metric replication to distinguish real patterns from chance; this is a substitute for, not equivalent to, formal correction, and any single-comparison result not flagged as replicating elsewhere in the paper should be read with that caveat.

## 6 Discussion

The central move of this paper is not a new mechanism but a redrawn boundary around what [lsgot_3]'s "identity fingerprint" was actually measuring. That study's own future-work section already anticipated the need for exactly this ablation ("decomposing the axis template into its structural components... would isolate which subcomponent... carries the [] signal," [lsgot_3, §7]); the present design executes it, on a model chosen for a reason external to convenience — the mechanism under study appears not to instantiate reliably at the scale [lsgot_3] studied.

What we find is not that [lsgot_3]'s finding was wrong, but that it was underdetermined between two candidate causes that a three-condition design cannot separate, and that the two causes turn out to govern different properties of the trajectory: one graded and static (§3.1), one gated and dynamical (§3.2–3.7). This is consistent with, and was directly motivated by, prior internal work on this same panel (`Teoria_subconjunto_acotado.md`, `ROADMAP_REENCUADRE_DENSIDAD_RESTRICCION.md`) that first proposed the two-factor account from the Δκ-based evidence available at the time; the present paper's contribution is to test that account with a battery of metrics that does not share Δκ/W₁'s documented methodological weakness (§3.4), and to extend it with a direct test — route-fidelity and directional perturbation — of the dynamical-attractor reading that neither the original Δκ-based work nor [lsgot_3] attempted.

We take the persona-vector identity-projection double dissociation (§3.5) as this paper's strongest single result: it is large, it is the same sign for both identity-bearing conditions regardless of constraint architecture, it is the *opposite* sign for the constraint-only condition, and it is not explainable by prompt length or general rigidity, since `automata_neutro` — the condition that diverges from it — carries no identity content at all, and no repeated-output markup either (§2.2); it is not on the Appendix A density scale (that scoring is future work, §7), but its regex-based restriction density in the internal design notes (`Teoria_subconjunto_acotado.md`) is the highest of any condition in the panel. It is consistent with, and adds trajectory-level, perturbation-tested detail to, the persona-vector literature [Chen et al., 2025; Lu et al., 2026], which typically intervenes at a single layer rather than characterizing a full generation trajectory under perturbation. The effect also replicates, at a larger effect size and without length-rebalancing, on an architecturally distinct second model (Qwen3-32B, §7.1) — evidence that it is not an idiosyncrasy of Gemma-4's training or tokenizer.

We take the absence of any dynamical advantage for identity content (§3.3, §3.6, §3.7) as an equally important, and equally hard-won, result — hard-won because it required building the directional-perturbation and route-fidelity machinery that a purely correlational, single-metric design does not need. It resolves H3 of [lsgot_3] in the negative, on this panel, and it retroactively justifies that paper's explicit refusal to adopt attractor language: the caution was correct, not merely conservative.

**A practical reading of §3.3's coarse-recovery illusion (Figure 1).** For prompt engineering and deployment practice, the dissociation between coarse and specific recovery is a caution worth stating plainly. Packing a system prompt with rigid, trigger-to-fixed-output rules — without also wiring in a mandatory self-referential check — does not make a model's behavior more robust to perturbation in any sense beyond the coarsest one. `automata_neutro` reaches the same 95%-centroid threshold as `axis` or `vanilla` on most trajectories, which could be misread as comparable or even superior stability; Figure 1 and the route-fidelity result (§3.6) show that what has actually happened is closer to convergence toward a generic, less-differentiated region of representation space than a return to the trajectory's own prior state, and for 12–23% of trajectories not even that — the coarse criterion is never met at all. On this panel, whatever makes a trajectory recover *toward where it started*, specifically, tracks wired self-reference (§3.3's Prediction 2(i), §3.5), not the density of behavioral rules a prompt imposes. A thicker rulebook is not, by itself, evidence of resilience under perturbation; it may instead be evidence of a system with no mechanism for returning to its own prior state once displaced from it, only a means of steering broadly, differently-behaving trajectories back toward the same coarse region.

**A metaphor for what the two factors are and are not.** An automaton without wired self-reference has a map but no compass: its rules say "if A, do B," which works until a perturbation pushes it off the mapped terrain — with no invoked check against a persisting "self" to consult, it does not return to where it was, it drifts toward a generic, low-energy default (§3.3, §3.6, Figure 1). A system with wired self-reference also has rules, but additionally re-invokes a check against declared principles at every generation step — and *empirically* recovers its own route after perturbation far more reliably (§3.3, §3.6). We are explicit about what this metaphor does *not* claim: we found no evidence that recovery works by the trajectory being pulled back along the identity direction like a literal compass needle (§3.7's directional-perturbation test is a null result on exactly that mechanism). The compass is procedural — a step the pipeline re-runs — not geometric; *why* re-running that step produces higher route-fidelity is not established by this paper and is the natural target of the cross-layer/attention work proposed in §7.2.

## 7 Future work

### 7.1 Cross-model validation (Qwen3-32B)

After the exclusions described in §2.2, we replicated the free-generation half of the FASE 0 battery (§3.1, §3.2 excluding RQA, §3.5's static double dissociation, and the vertical-rotation probe of an earlier internal analysis) on the seven-condition clean panel, same 20 content prompts, same system-prompt text, on Qwen3-32B (dense, GQA + QK-norm, 64 layers, a different tokenizer from Gemma-4) — a change of architecture, not merely of scale, addressing the single-model limitation of §5. Token counts were not rebalanced for Qwen3's tokenizer (`axis`: 3,819 tokens vs `generic_long`: 4,794 in Qwen3's tokenization, versus 3,945/3,957 — nearly matched — under Gemma's), which if anything works against the pattern found, since `axis` is the *shorter* of the two. No perturbation/recovery experiment (§3.3–3.7) was run on Qwen3; this remains open.

**Replicates, robustly:** the static double dissociation at t = 0 (before any tokens are generated) is present and larger than in Gemma-4 (d = +12.97 between `axis_pec_only` and `automata_neutro`, vs d = +5.75 on the analogous Gemma comparison at the equivalent decision point); participation ratio for `automata_neutro` is the lowest in the panel across all 12 layers sampled without exception (cleaner than Gemma's non-monotonic profile); identity-vs-restriction is the largest subspace separation in the panel by principal-angle analysis (40.7°); the identity-projection effect grows monotonically toward later layers, as in Gemma.

**Does not generalize:** the relative *hierarchy* between the two factors is architecture-specific. In Gemma-4, `automata_neutro` is the single most distinct condition in the panel by every geometric measure tried (§3.1–§3.5); in Qwen3, the two factors are more balanced, and it is the axis-family (identity), not `automata_neutro`, that separates first under hierarchical clustering of the representational subspace. A Gemma-specific probe — whether conditions rotate differently layer-to-layer ("vertical rotation," an internal exploratory analysis not otherwise reported in this paper) — shows no condition-wise differentiation at all in Qwen3 (all seven conditions rotate within a 0.8° range across layers, versus a 4–5° `automata_neutro` outlier in Gemma): a mechanism that organizes the phenomenon in one architecture is silent in the other, even though the phenomenon itself (the double dissociation) is present in both.

**Reading.** The high-level phenomenon — hidden states carrying independent, oppositely-signed static traces of declared identity and of operational-constraint density — is not an idiosyncrasy of Gemma-4's training or tokenizer: it replicates, at a larger effect size, under a materially different architecture with no re-balancing in its favor. The specific quantitative details of how the two factors trade off, and at least one candidate geometric mechanism, do not travel with it. Full detail in `QWEN3_VALIDATION_REPORT.md`; we treat this as a validation of the phenomenon and an open question about its mechanism, not as a second, independently-confirmed paper's worth of claims.

### 7.2 Remaining items

**Definitive first-token control for Δκ/W₁ (forced-token forward pass).** The [REPORTE_FASE0] protocol's GPU-free checks — comparison against simple distributional baselines, split-half null, a proxy-level first-token check — are now done on this panel (§3.4, `CURVATURE_SELF_AUDIT_REPORT.md`) and reassuring: the statistic's own ‖v1‖-based effects survive the proxy control rather than reversing, unlike the sibling panel. What remains is the sibling audit's definitive control — a forward pass with the first token forced constant, comparing h_post rather than the proxy — which needs GPU access to Gemma-4-31B-it and was not run this pass.

**A length-matched replication of `axis_pec_only`** at ~3,900 tokens, to close the length-confound gap noted in §5 — the wording replication (`axis_pec_only_v2`/`automata_neutro_v2`) is done (§5, `T2_REPLICATION_REPORT.md`), but neither instantiation controls length.

**A markup-free replication of the excluded declared/wired-self-reference diagnostic cell (§2.2)** — `automata_neutro` content restructured with declared-but-unwired, and separately wired, self-reference — to test Prediction 2(ii)/(iii) (§3.3, §5) without the format confound that forced exclusion of the original pair.

**Constraint-density quantification for `axis_pec_only` and `automata_neutro`** on the same regex-based density-per-1k-tokens scale used for the other five conditions (Appendix A), to place all seven conditions on a single quantitative axis rather than a categorical one.

**Cross-layer and attention-based extensions** (logit-lens-style vertical trajectories; attention mass directed at identity- versus rule-declaring spans of the prompt during generation) — designed but not yet run (E-F, E-G in the internal experimental roadmap); the most direct mechanistic test of the wiring hypothesis of §3.3, at the cost of substantially heavier extraction.

**Parameter-count scale sweep**, distinct from the architecture cross-check of §7.1, to establish whether the ~30B threshold motivating this paper's single-model design (§2.1) is real and where it lies, rather than assumed from informal deployment evidence.

**Perturbation/recovery dynamics on Qwen3-32B** (§3.3–§3.7's battery), not run in the present cross-model pass (§7.1), to test whether the gated-recovery finding — not just the static double dissociation — also generalizes across architecture.

## Appendix A: constraint-density scores

Hard-restriction marker density per 1,000 tokens (regex-based count over literal-output, rigid-format, domain-scope, mandatory-verification, priority-hierarchy, and prohibition markers; relative ordering, not absolute token-level density):

| condition | literal output | rigid format | domain/scope | process/verif. | priority/hierarchy | prohibitions | total |
|---|---|---|---|---|---|---|---|
| `axis` | 0.0 | 0.0 | 0.0 | 6.3 | 10.8 | 4.5 | **27.0** |
| `axis_short` | 0.0 | 0.0 | 0.0 | 3.8 | 3.8 | 6.4 | **20.4** |
| `generic_long` | 0.3 | 0.0 | 0.0 | 4.4 | 0.3 | 5.8 | **11.8** |
| `generic_short` | 1.4 | 0.0 | 0.0 | 2.8 | 0.0 | 5.7 | **11.3** |
| `vanilla` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |

`axis_pec_only` and `automata_neutro` were not scored on this scale at the time of the original analysis; scoring them is listed as future work (§7) since it would let §3.1's participation-ratio gradient be regressed directly against a quantitative constraint-density axis instead of a categorical identity/constraint split.

## References

- Chen, R., et al. (2025). Persona vectors: monitoring and controlling character traits in language models. [as cited in lsgot_3]
- Lu, K., et al. (2026). [persona-vector / activation-steering follow-up, as cited in lsgot_3]
- Forman, R. (2003). Bochner's method for cell complexes and combinatorial Ricci curvature. Discrete & Computational Geometry, 29(3):323–374. (Corrected from an earlier draft's Ollivier, Y. (2009) citation, which described a different discrete-curvature construction — optimal transport between neighbor distributions — than the one this paper's Δκ/W₁ statistic actually computes; see §3.4 and `CURVATURE_SELF_AUDIT_REPORT.md`.)
- Marwan, N., Romano, M. C., Thiel, M., and Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics Reports, 438(5-6):237–329.
- Hurst, H. E. (1951). Long-term storage capacity of reservoirs. Transactions of the American Society of Civil Engineers.
- Eiter, T., and Mannila, H. (1994). Computing discrete Fréchet distance. Technical Report CD-TR 94/64, TU Wien.
- Castillo, J. A., Torres Yévenes, M., and Lanas, J. C. [lsgot_3] — companion paper, this line of work, working draft v0.3.
- [REPORTE_FASE0] — internal pre-registered audit, `REPORTE_FASE0.md`, this project, 2026-08-11/12.

*Note: the reference list above is a starting skeleton, not a verified bibliography. Chen et al. and Lu et al. citation details should be pulled verbatim from `lsgot_3.md`'s own reference list before submission; several other [lsgot_3] references (Valeriani et al. 2023, Belrose et al. 2023, Elhage et al. 2021) remain relevant to the introduction's framing and are not yet re-imported here.*
