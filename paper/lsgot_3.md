# How Multimodal Instruction-Tuning Reorganizes the Geometric Encoding of Identity-Specifying Prompts in Transformer From Direction to Magnitude: Hidden States

Jorge A. Castillo1, Marco Torres Yévenes1, and Juan Carlos Lanas1

1Axis Dynamics SpA, Santiago, Chile, jorge.castillo@axisdynamics.cl,

mtorres@axisdynamics.cl, jc@axisdynamics.cl

Working draft v0.3 — May 2026

## Abstract

We investigate whether identity-specifying system prompts produce statistically distin- guishable geometric fingerprints in the token-indexed hidden-state trajectories of four open- weight transformer language models spanning four post-training regimes: no training (Gemma- 4-E4B base), multimodal RLHF (Gemma-4-E4B-it), RL distillation (DeepSeek-R1-Distill- Qwen-7B), and supervised instruction-tuning (Qwen2.5-7B-Instruct). Three controlled prompt conditions (an identity-specifying axis prompt ∼2129 tokens, a length-matched generic-assistant prompt, and a 26-token vanilla baseline) are compared via five geomet- ric metrics with distinct theoretical anchors: the 1-Wasserstein distance between edge-wise distributions of Ollivier-Ricci curvature on k-NN trajectory graphs, the prompt-response alignment with all-but-the-top anisotropy correction, the initial-state cosine, the PCA-50 silhouette of axis-vs-generic clustering, and the inter-trajectory cosine consistency. All in- ferential claims are based on trajectory-level permutation null distributions and on multi- ple geometric controls (teacher-forced content controls, temporal-chain versus k-NN graph topology, ABT-projected k-NN, angular versus Euclidean distance for graph construction, intrinsic-dimension estimation, B = 5000 permutations on borderline statistics). The cen- tral empirical finding is a qualitative reorganization of the geometric encoding of identity across the instruction-tuning boundary: in the base-weight Gemma-4-E4B, the identity fin- gerprint is encoded predominantly in the direction of hidden-state vectors (the Wasserstein separation is 0.034 with permutation p = 0.002 under angular k-NN, where the norm is neutralized); in the multimodal instruction-tuned Gemma-4-E4B-it the fingerprint migrates into the magnitude: the separation collapses under angular k-NN (p = 0.439) but survives under Euclidean k-NN (p = 0.047, refined to p = 0.042 at B = 5000), and the mean norm of the first generated state is markedly lower under the identity prompt (∥v1∥ = 138.9) than under both the generic (211.5) and vanilla (195.3) conditions, in inversion of the relationship in the base model. This direction-to-magnitude reorganization is specific to the multimodal instruction-tuning regime: it is absent under RL distillation (separations track length, not content) and under SFT instruction-tuning (no separations). A teacher-forced control quanti- fies that ∼ 30% of the free-running cosine signal is prompt-driven (vs ∼ 70% content-driven). We position the methodological combination, W1 on edge-wise distributions of Ollivier-Ricci curvature on k-NN trajectory graphs, as a contribution of independent interest.


## 1 Introduction

## 1.1 Context

Transformer language models [Vaswani et al., 2017] produce fluent text conditional on a prompt. Substantial effort has been devoted to characterizing what they output under various conditioning strategies and where specific features reside in their parameters [Hewitt and Manning, 2019, Tenney et al., 2019, Belinkov and Glass, 2019, Elhage et al., 2021]. Comparatively less attention has been devoted to the geometry of the hidden-state trajectories these models produce during autoregressive generation: the sequence v1, v2, . . . , vN ∈ H of internal representations produced at successive generation steps, viewed as a discrete path in the hidden-state space H ⊂ RD. [URL 🔗](#page-0)

Recent work begins to close this gap. Intrinsic-dimensionality profiles show that the effective dimension of transformer representations is far below the ambient hidden size and varies across layers [Valeriani et al., 2023, Razzhigaev et al., 2024]; the Tuned Lens [Belrose et al., 2023] exposes per-layer prediction trajectories; persona-vector research [Chen et al., 2025, Lu et al., 2026, Wang, 2025] extracts identity directions from contrastive activation differences and demonstrates that personality traits can be encoded as orthogonal linear subspaces within the latent geometry. None of these lines, however, characterizes the full hidden-state trajectory geometry induced by identity-specifying system prompts: persona-vector frameworks intervene at a single layer to extract or manipulate identity directions, but do not analyze how an identity prompt reshapes the token-indexed trajectory in its entirety, nor compare this reshaping across post-training regimes. [URL 🔗](#page-0)

## 1.2 Research question

Does an identity-specifying system prompt induce a statistically distinguishable geo- metric fingerprint in transformer hidden-state trajectories, beyond what is attributable to the prompt’s length or to the textual content generated under that prompt? If so, how does this fingerprint depend on the model’s post-training regime?

The question is answered empirically via a four-model, three-prompt design described in section 2.2. The novel finding is not the existence of geometric distinguishability per se, but a qualitative reorganization across the instruction-tuning boundary: the identity fingerprint mi- grates from a direction-coded representation in the base-weight model to a magnitude-coded representation in the multimodal instruction-tuned model. This reorganization is regime- spe- cific: it does not occur under RL distillation or under SFT instruction-tuning. [URL 🔗](#page-0)

## 1.3 Theoretical framing (minimal)

We adopt a deliberately minimal framing. We do not claim that the transformer admits a classical dynamical-systems description with an explicit vector field or attractors in the strict sense of continuous dynamics; autoregressive generation under greedy decoding is a deterministic function of context, and the hidden-state trajectory is a token-indexed discrete sequence, not a flow. We analyze this sequence as a point cloud equipped with a temporal ordering, and we use standard tools from optimal transport, discrete differential geometry on graphs, and cluster analysis to quantify its structure. Where prior drafts of this work employed terms from dynamical- systems theory, we have replaced them with descriptive statistical language.

## 1.4 Hypotheses

We formulate three falsifiable hypotheses; H1 and H2 are tested empirically here, with H2 en- riched (relative to earlier drafts of this work) by an explicit prediction concerning the geometric substrate of the encoding. H3 is stated for completeness but its empirical test lies outside the scope of the present paper (see below).


Hypothesis 1 (Length-only regularization). The geometric differences observed between the identity and vanilla conditions are attributable solely to the prompt’s token length, not to its semantic content.

Prediction 1 (for H1). The length-matched generic condition will exhibit geometric separation from vanilla comparable to that of axis. Moreover, the mean magnitude ∥v1∥ of the first generated state will order monotonically with prompt length: ∥v1∥axis > ∥v1∥generic > ∥v1∥vanilla.

Hypothesis 2 (Regime-dependent reorganization of identity encoding). The geometric encoding of identity-specifying prompts depends qualitatively on the post-training regime. In particular, multimodal instruction-tuning suppresses the length-driven encoding common to long prompts and reorganizes the identity-specific component from a directional substrate (visible in the base- weight model) into a normative substrate (the magnitude of the hidden-state vector).

Prediction 2 (for H2). (i) The ratio R = W1(ρaxis, ρvanilla)/W1(ρgeneric, ρvanilla) under edge-wise

Ollivier-Ricci on Euclidean k-NN trajectory graphs satisfies RIT ≫ 1 and Rbase ≪ 1. (ii) Under angular k-NN (vectors L2-normalized prior to graph construction), the identity-vanilla separation is preserved in the base model and attenuated to non-significance in the instruction-tuned model.

(iii) The mean norm

∥v1∥

inverts across the instruction-tuning boundary: monotonic in prompt

length in the base model, anti-correlated with prompt length (and minimal for the identity prompt) in the instruction-tuned model.

Hypothesis 3 (Cluster robustness under perturbation). The axis trajectory cluster is statisti- cally more robust than the vanilla cluster under moderate noise injection at intermediate hidden states.

H3 concerns a distinct class of experiment (deliberate perturbation of intermediate hidden states rather than passive observation of trajectories) and lies outside the scope of the present paper. We state it here for completeness and defer its empirical test to future work.

## 1.5 Contributions

Methodological. To our knowledge, this is the first application of the 1-Wasserstein distance between edge-wise distributions of Ollivier-Ricci curvature on k-NN graphs of transformer hidden- state trajectories as a graph-level comparison statistic. Related work has applied W1 to persis- tence diagrams [Cohen-Steiner et al., 2007, 2010] or to representation distributions [Alvarez-Melis and Jaakkola, 2018], and has applied Ollivier-Ricci curvature to graph-neural-network expressiv- ity analyses [Topping et al., 2022, Nguyen et al., 2023], but not to edge-wise curvature distribu- tions of trajectory k-NN graphs. We welcome correction if precedent exists. [URL 🔗](#page-0)

Empirical. A four-model, three-condition study with full edge-wise Ollivier-Ricci protocol on all four models, exposing a direction-to-magnitude reorganization of identity encoding that is specific to the multimodal instruction-tuning regime. A teacher-forced content control quantifies the prompt-driven component of the cosine signal at ∼ 30% of the free-running magnitude.

Inferential discipline. A uniform permutation-test protocol at the trajectory level, com- plemented by anisotropy baselines, all-but-the-top corrections [Mu et al., 2018, Timkey and van Schijndel, 2021], sensitivity sweeps over the k-NN parameter, ABT-projected graph construction, intrinsic-dimension estimation [Facco et al., 2017], and high-precision permutation (B = 5000) on borderline statistics. [URL 🔗](#page-0)

## 2 Theoretical foundations

## 2.1 Hidden-state trajectories

Let M : Σ∗ → ∆(Σ) be an autoregressive transformer mapping token sequences over alphabet Σ to distributions over Σ. Under greedy decoding, M produces a deterministic token sequence


conditional on a prompt p. At each step t, the forward pass produces a hidden state at every layer and every token position; we extract the hidden state at the final pre-lm_head layer, at the position of the most recently generated token, and denote it vt ∈ RD.

Definition 2.1

(Hidden-state trajectory). The

hidden-state trajectory

of length

N

produced by

model M under prompt p is T(p,M,N) = (v1, . . . , vN), with vt ∈ RD. We fix N = 256. We let v0 denote the hidden state at the final prompt token, i.e. immediately before generation begins.

## 2.2 Experimental setup

We compare three system-prompt conditions across four models.

## Conditions.

axis: the publicly released VEX/MIA identity template1 instantiated with identity content [URL 🔗](#page-0)

(essence, values, mode declarations, interaction protocols). Prompt length 2129 tokens.

generic: a length-matched (957 tokens) generic- assistant prompt specifying general-purpose

conversational behavior without identity content. This is the critical control for H 1. [URL 🔗](#page-0)

vanilla: the minimal baseline "You are a helpful assistant." (26 tokens).

The length disparity between axis (∼2129 tokens) and generic (∼957 tokens) is approximately 2×, both much larger than vanilla (∼26 tokens). This asymmetry is addressed explicitly in the norm analysis of section 4.1.2, which tests whether ∥v1∥ scales with length monotonically (the prediction under H1) or non-monotonically (the prediction under H2). [URL 🔗](#page-0)

Models. Four open-weight transformer language models are studied, chosen to span four dis- tinct post-training regimes while keeping the parameter count within a narrow range (7–8B). This design allows the effect of the post-training regime to be examined without confounding it with variation in model scale. The pair Gemma-4-E4B/Gemma-4-E4B-it isolates the effect of multimodal RLHF on the same architecture; DeepSeek-R1-Distill-Qwen-7B represents RL distillation and Qwen2.5-7B-Instruct represents supervised instruction-tuning as alternative post-training regimes with distinct inductive biases. Table 1 summarizes the panel. [URL 🔗](#page-0)

| Model | Post-training | Origin Layers | D | Role |
| --- | --- | --- | --- | --- |
| Gemma-4-E4B (base) | none | Google |   | 42 2560 H2 ablation |
| Gemma-4-E4B-it | Multimodal RLHF Google |   | 42 2560 Principal |   |
| DeepSeek-R1-Distill-Qwen-7B | RL distillation | DeepSeek |   | 28 3584 Cross-regime |
| Qwen2.5-7B-Instruct | SFT (instruction) Alibaba |   |   | 28 3584 Cross-regime |

Table 1: Four models spanning four post-training regimes. All within the 7–8B parameter range; size scaling is out of scope.

Extraction. For each (model, condition, prompt) triple, greedy decoding for N = 256 tokens, with the final-layer hidden state at the latest-token position stored at each step. The prompt set P consists of 100 ontology prompts (see section A). [URL 🔗](#page-0)


## 3 Methods

We describe each metric, situate it in its literature, and state which prediction it tests. Notation: condition a trajectory is c and prompt T = (v1, p . . . , we obtain a trajectory vN) in RD; the centroid of T(c,p). T is ¯v(T) = N−1PN t=1 vt. For each

## 3.1 Ollivier-Ricci curvature on k-NN trajectory graphs

Definition. For trajectory T,

is built under a chosen base metric d on RD. For each edge e = (x, y) ∈ E(Gk(T)), the Ollivier- Ricci curvature with idleness α ∈ [0, 1) is

where mα x = α δx + (1 −α) Unif(N(x))

N(x) of x. We use α = 1/2 following [Sandhu et al., 2015, Topping et al., 2022], and solve each edge’s optimal-transport subproblem exactly via the network simplex. [URL 🔗](#page-0)

Choice of base metric. Two natural choices arise, corresponding to two different views of the same point cloud in RD.

- Euclidean k-NN treats hidden states as points in RD under deuc(v, v′) = ∥v − v′∥2. Two hidden states with parallel direction but disparate norms are counted as distant.

- Angular k-NN treats hidden states as points on the unit sphere SD−1 under the geodesic distance dang(v, v′) = arccos ⟨˜v, ˜v′⟩ on the normalized vectors ˜v = v/∥v∥2. Norm varia- tion is neutralized; only direction contributes to neighborhood structure.

The two metrics therefore probe complementary aspects of the same point cloud, and the comparison between them functions as a diagnostic for whether a geometric effect lives primarily in the direction or in the magnitude of the hidden-state vectors. Section 4.1.2 exploits this diagnostic to establish the central finding of this paper. [URL 🔗](#page-0)

Lineage and theoretical justification. The Ollivier-Ricci construction is due to Ollivier [2009], who generalized Ricci curvature to discrete metric measure spaces via optimal transport [URL 🔗](#page-0)

between random- walk distributions. Convergence of [URL 🔗](#page-0)

point clouds to the Ricci curvature of the underlying Riemannian manifold is proved by Van der Hoorn et al. [2023] (asymptotic) and refined by Trillos and Weber [2023] (non-asymptotic rates). Recent work has brought Ollivier-Ricci curvature to bear on representational similarity analysis in neural networks [Torbati et al., 2025], supporting its use as a fine-grained local-geometry descriptor in this setting. Our application is justified by the empirical intrinsic dimensionality ID ∈ [6, 16] of our models’ hidden states (table 7), well below the ambient dimensions D ∈ {2560, 3584}. We adopt k = 5 in the main analyses, motivated by the heuristic k ≈ logN = log 256 ≈ 5.5; sensitivity over k ∈ {5, 10, 15, 20} in section 4.2. [URL 🔗](#page-0)

the k-nearest-neighbor graph

Gk(T) on the point set {v1, . . . , vN}

is the lazy-random-walk measure on the 1-neighborhood

κO

on k-NN graphs of high-dimensional

## 3.2 Wasserstein-1 between edge-curvature distributions

For condition c, we pool edge-curvature values across all trajectories in the prompt set: ρc =

in closed form: P p |Ep|)−1P p P e∈Ep δκO(e), where Ep = E(Gk(T(c,p))). The one-dimensional W1 is computed

(

0 F−1

du,


exploiting the cumulative-distribution representation. Sample complexity in one dimension is O(n−1/2) [Fournier and Guillin, 2015, Weed and Bach, 2019]; the stability of W1 on distributions of geometric descriptors echoes the stability theorems for persistence diagrams [Cohen-Steiner et al., 2007, 2010]. [URL 🔗](#page-0)

The pooled-edge construction (yielding ∼ 60,000 values per condition) is contrasted in sec- tion C with a per-trajectory mean-curvature variant that gives |P| = 100 values per condition; the pooled version is more powerful but obscures inter-trajectory heterogeneity. [URL 🔗](#page-0)

## 3.3 Anisotropy-corrected cosine statistics

Three cosine-based statistics on hidden-state vectors complement the curvature-based analysis. Each captures a different scale of the trajectory: the transition from prompt to first generated state (C01), the alignment of the response as a whole with the pre-generation state (PRA), and the reproducibility of trajectory centroids across different prompts under the same condition (ITC). All three are cosine-based and thus require correction for the well-known anisotropy of transformer hidden states [Ethayarajh, 2019, Timkey and van Schijndel, 2021]. [URL 🔗](#page-0)

Initial-state cosine. The most localized statistic,

measures the angular transition between the final pre-generation state (last token of the prompt) and the first generated state. It captures the immediate geometric imprint of the prompt at the boundary between conditioning and generation and is minimally confounded by downstream textual content.

Prompt-response alignment. The trajectory-averaged counterpart,

measures the persistence of the prompt’s directional imprint across the full response. Where C01 captures the immediate transition, PRA captures the sustained alignment between v0 and the trajectory centroid.

Inter-trajectory consistency. Across the prompt set P, we quantify how similar the trajec- tory centroids under condition c are to each other:

High ITC under condition c indicates that the model produces geometrically homogeneous re- sponses across different prompts when conditioned on c: the response centroids cluster tightly regardless of the underlying question. Low ITC indicates that the same condition produces diverse trajectory centroids across prompts.

Anisotropy controls. All three statistics inherit the global anisotropy of the hidden-state

space.

of hidden states is ∼0.85, a distortion that inflates every cosine-based comparison and can create the appearance of similarity where none exists. To control for this, we report each cosine quantity C in three forms: [URL 🔗](#page-0)

1. The raw value C.

Ethayarajh [2019] report that in GPT-2 layer 12 the mean cosine between random pairs [URL 🔗](#page-0)


- 2. The value relative to the per-condition anisotropy baseline,

computed over 10,000 random pairs of hidden states drawn from the trajectories of condi- tion c. The corrected quantity is C − ¯caniso(c).

- 3. The value after all-but-the-top correction [Mu et al., 2018], in which the top-k principal components of the pooled hidden-state cloud are subtracted before recomputing C, for k ∈ {1, 2, 3}. This addresses the finding of Timkey and van Schijndel [2021] that a small number of “rogue” dimensions dominate cosine similarity in transformer representations. [URL 🔗](#page-0)

The three corrections address complementary confounds. The baseline subtraction (ii) re- moves the per-condition anisotropy floor; the all-but-the-top correction (iii) removes the specific principal directions that dominate that floor. Table 8 reports the anisotropy magnitudes of the four models studied, showing that Gemma-4-E4B-it operates in a substantially more isotropic regime than the GPT-2-era models that motivated the anisotropy literature. [URL 🔗](#page-0)

## 3.4 Cluster separability

We pool the hidden states of the axis and generic conditions, reduce via PCA to dPCA = 50 components, and compute the Rousseeuw [1987] silhouette score Sil using condition labels as [URL 🔗](#page-0)

cluster assignments. Sensitivity over [URL 🔗](#page-0)

of silhouette under externally-defined cluster labels [Rautenstrauch and Ohler, 2025] is addressed via permutation null distributions and via cosine-silhouette as a robustness check. [URL 🔗](#page-0)

dPCA [URL 🔗](#page-0)

∈ [URL 🔗](#page-0)

{30, 50, 100, 200} in section 4.2. The Lazar critique [URL 🔗](#page-0)

## 3.5 Inferential protocol

Permutation. For each metric M and each condition pair (c, c′), we shuffle condition labels of the 3 · |P| trajectories uniformly at random at the trajectory level (preserving within-trajectory

correlation structure) and recompute M. With

B = 5000 on borderline statistics (section 4.2), we report the empirical two-sided p-value pemp = [URL 🔗](#page-0)

(1 + |{b : M(b) as extreme as

Multiple comparisons. The paper reports a number of inferential comparisons. We apply Benjamini–Hochberg FDR control at q = 0.05 to the ten primary inferential comparisons in section 4.1 and report which survive correction (table 9). Bonferroni control is reported in parentheses as a more stringent reference. [URL 🔗](#page-0)

Sensitivity sweeps. Three methodological hyperparameters admit alternative reasonable val- ues, and we report the sensitivity of the principal results to each. First, the neighborhood size k in the k-NN graph is swept over {5, 10, 15, 20}; the default k = 5 is motivated by the heuristic k ≈ logN for trajectory length N = 256. Second, the PCA reduction dimensionality dPCA used

for the silhouette score is swept over

all-but-the-top correction is reported for k ∈ {1, 2, 3} top principal components removed. Each sweep is reported in section 4.2; the qualitative findings of section 4.1 are preserved across all three, and where they are not (as in the k sweep for Forman-Ricci curvature), the instability is documented explicitly and drives the choice of primary metric. [URL 🔗](#page-0)

B

= 1000

permutations as the default and

Mobs}|)/(1

+B). [URL 🔗](#page-0)

{30, 50, 100, 200}

around the default

dPCA

= 50. Third, the


## 4 Results

## 4.1 Strong effects, organized by what they establish

## 4.1.1 Establishing the four-model regime structure (T6)

The full edge-wise Wasserstein protocol was applied uniformly across the four models. Figure 1 illustrates the qualitative contrast that motivates the statistical comparison to follow. Table 2 reports the three pairwise comparisons per model, with permutation p-values at B = 1000 (at B = 5000 for Gemma-4-E4B-it, see section 4.2). [URL 🔗](#page-0)

*Figure 1: Trajectory centroids projected onto the first three principal components of the pooled hidden-state cloud, colored by prompt condition (axis in blue, vanilla in orange, generic in magenta). Each point is the centroid ¯v(T) of one trajectory over the |P| = 100 ontology prompts, with error bars of one standard deviation per axis. (a) Gemma-4-E4B-it (multimodal RLHF, principal model): the axis, vanilla, and generic centroids form three distinguishable clusters, consistent with the specificity pattern reported in table 2 and with the PCA-50 silhouette of 0.357 (p < 0.001) reported in section 4.1.3. (b) Qwen2.5-7B-Instruct (SFT instruction-tuning): the three condition centroids overlap without clear separation, consistent with the absence of significant pairwise comparisons for this model in table 2. The visualization projects trajectories of length N = 128 for legibility; all statistical analyses in table 2 and subsequent tables use N = 256. Panel labels within each subplot use the internal nomenclature of the data-generation pipeline (Testigo for vanilla, LSGOT for axis, Baseline estructurado for generic; Hreal denotes the hidden-state space H). [URL 🔗](#page-0)*

| Model (regime) | p: ax/van | p: gen/van | p: | ax/gen Interpretation |
| --- | --- | --- | --- | --- |
| Gemma-4-E4B (none) | 0.002 | 0.001 |   | 0.001 all pairs separate |
| Gemma-4-E4B-it (mm-RLHF) | 0.042† | 0.451 |   | 0.118 only axis vs vanilla |
| DeepSeek-R1-Distill-Qwen-7B (RL) 0.001 |   | 0.001 |   | 0.962 long vs short, indistinguishable inter |
|   |   |   |   | se |
| Qwen2.5-7B-Instruct (SFT) | 0.907 | 0.240 |   | 0.654 no separations |

Table 2: Edge-wise Wasserstein-1 of Ollivier-Ricci curvature distributions on Euclidean k-NN

graphs (k = 5), trajectory-level permutation, B = 1000 except † which uses B = 5000.

The four

post-training regimes display four qualitatively distinct patterns of geometric distinguishability.

We highlight three readings.

- (i) In the base-weight Gemma-4-E4B, all three pairs separate at p ≤ 0.002. This is consistent with a regime where prompt length itself drives geometric distinguishability: the base model is sensitive to any long prompt versus any short prompt, without specificity to semantic content.

- (ii) In Gemma-4-E4B-it (multimodal RLHF), only axis vs vanilla separates (p = 0.042), while


generic vs vanilla does not (p = 0.451). This is the prediction of H 2: the instruction-tuning has suppressed the length-driven separation (generic is now indistinguishable from vanilla in pooled curvature distribution) but preserved a content-specific separation for the identity prompt. [URL 🔗](#page-0)

(iii) Under RL distillation (DeepSeek-R1-Distill), both length-matched prompts separate from vanilla at p ≤ 0.001 but are mutually indistinguishable (p = 0.962); under SFT instruction- tuning (Qwen2.5-7B) no separation reaches p < 0.05 at B = 1000. The regime-dependent pattern of H 2 is therefore specific to multimodal RLHF in our four-model panel. [URL 🔗](#page-0)

We note the borderline nature of p = 0.042 for W1(ρaxis, ρvanilla) in Gemma-4-E4B-it. Under Benjamini–Hochberg FDR control at q = 0.05 across the ten primary inferential comparisons of this paper, this single p-value does not survive (table 9); the central regime-pattern argument is sustained not by this individual p-value but by the joint pattern across the four models and by the corroborating cosine-based and norm-based findings reported below. [URL 🔗](#page-0)

## 4.1.2 Norm versus direction: the principal substantive finding (T13 plus norm analysis)

The metric chosen to build the k-NN graph determines what geometric structure the curvature analysis can detect. Under Euclidean deuc, both directional and normative variations contribute. Under angular dang on L2-normalized vectors, only direction contributes; norm variation is neu- tralized. The contrast between the two reveals which geometric substrate carries the identity signal.

*Table 3 reports the axis-vs-vanilla comparison under both metrics in the two Gemma models. [URL 🔗](#page-0)*

| Model |   |   | Euclidean k-NN Angular k-NN |
| --- | --- | --- | --- |
| W1 | pemp | W1 | pemp |
| Gemma-4-E4B (base) | 0.060 0.030 0.034 |   | 0.002 |
|   | Gemma-4-E4B-it (mm-RLHF) 0.029 0.047† |   | 0.006 0.439 |

*Table 3: Axis-vs-vanilla edge-wise Wasserstein on Ollivier-Ricci curvature, under Euclidean and angular k-NN graphs. The pattern inverts across the instruction-tuning boundary. In the base model, the separation survives angular normalization and is in fact strengthened (p = 0.002): the identity fingerprint is direction-coded. In the instruction-tuned model, the separation collapses under angular normalization (p = 0.439) while persisting under Euclidean (p = 0.047† → 0.042 at B = 5000): the identity fingerprint is magnitude-coded.*

This pattern is corroborated by direct measurement of the norm of the first generated state, ∥v1∥2, across conditions. Table 4 reports the means with Mann–Whitney U significance. [URL 🔗](#page-0)

| Model | ∥v1∥axis | ∥v1∥generic | ∥v1∥vanilla | Pattern (all p < 0.001) |
| --- | --- | --- | --- | --- |
| Gemma-4-E4B (base) |   | 149.7 ± 8.2 141.8 ± 12.6 125.0 ± 11.3 |   | axis > generic > vanilla |
| Gemma-4-E4B-it (mm-RLHF) |   | 138.9 ± 8.3 211.5 ± 24.3 195.3 ± 24.4 |   | axis < vanilla < generic |

*Table 4: Mean Euclidean norm of v1 across |P| = 100 trajectories per condition. In the base model, the ordering is monotonic in prompt length (axis: ∼ 2129 tokens, generic: ∼ 957, vanilla:*

*∼*

*26);*

*in the instruction-tuned model, the ordering inverts, with axis exhibiting the lowest mean*

*norm despite having the longest prompt. The base-model ordering is consistent with Prediction 1; the IT-model ordering directly refutes it. [URL 🔗](#page-0)*

Refutation of length as confound. If the norm of v1 in the instruction-tuned model were a mechanical artifact of prompt length, axis (∼ 82× longer than vanilla) should have produced a


larger norm than vanilla. It produces a smaller one (138.9 vs 195.3). The norm-encoded identity signal in Gemma-4-E4B-it is therefore not explained by length.

Substantive interpretation. We summarize the result as a reorganization of the geometric substrate carrying identity information:

Base model: identity ⇒ direction (robust, not length-specific)

IT model: identity ⇒ magnitude (length-specific, direction-attenuated)

We avoid causal language. The two regimes differ in their geometric encoding of the identity prompt; we report the difference and its direction without committing to a mechanistic claim about how the RLHF training reorganizes the encoding.

## 4.1.3 Cosine evidence corroborates the regime asymmetry

We report three further metrics on Gemma-4-E4B-it.

Initial-state cosine. C01(axis) = 0.724 vs C01(generic) = 0.237 (raw); after subtracting per- condition anisotropy baselines, Crel 01 (axis) = 0.507 vs Crel 01 (generic) = 0.001. The permutation p for ∆C01 is < 0.001. The signal survives anisotropy correction with magnitude ∼ 0.5. At t ∈ {2, 3, 5} the difference attenuates rapidly to 0.185, 0.027, 0.079 respectively; by t = 10 the difference is negligible (∆C0,10 = 0.001, p = 0.931). The cosine fingerprint emerges immediately post-prompt and decays within ∼ 10 tokens (see early-token analysis below).

All-but-the-top decomposition. Removing the top principal component (ABT-1) of the pooled hidden-state cloud reduces PRA by 19.4% in the axis condition and by 77.4% in the generic condition. The bulk of the generic’s PRA resides in the dominant anisotropic direction; the axis’s PRA resides predominantly outside it. The pattern strengthens with larger k: at ABT-2 the drops are 28.1% vs 84.2%; at ABT-3, 34.7% vs 87.6%.

Cluster separability and inter-trajectory consistency. PCA-50 silhouette of axis vs generic clustering is 0.357 with permutation p < 0.001. Sensitivity over dPCA ∈ {30, 50, 100, 200}: 0.384, 0.357, 0.326, 0.305 respectively (monotone, modest decay; qualitative finding stable). ITC axis exceeds ITC generic by 0.035 with permutation p < 0.001.

## 4.1.4 Content-versus-prompt decomposition (T7)

Free-running comparisons confound the contribution of the prompt proper from the contribution of the (different) textual content each prompt induces the model to generate. We control this directly via teacher-forced shared targets. Thirty neutral textual sequences (Wikipedia-style, no identity-related content) are tokenized to length 256 and processed by the model under each condition: the system prompt varies, but the token sequence generated is forced to be the same shared target. The hidden states are then extracted as before.

*Table 5 compares free-running and teacher-forced results on Gemma-4-E4B-it. [URL 🔗](#page-0)*

| Metric | Free-running (v0.2) Teacher-forced (T7) |
| --- | --- |
| W1(ρaxis, ρvanilla), edge-wise | 0.0288 (p = 0.047) 0.0016 (p = 0.108) |
| W1(ρaxis, ρgeneric), edge-wise | 0.0237 (p = 0.118) 0.0053 (p = 0.923) |
| ∆C01, axis-generic | 0.4875 0.1601 |
| silhouette, axis vs generic | −0.017 0.357 |

*Table 5: Free-running vs teacher-forced comparison on Gemma-4-E4B-it, |P| = 30 shared targets.*


The comparison in table 5 between free-running and teacher-forced settings admits two com- plementary readings, one on the curvature signal and one on the immediately-post-prompt cosine signal. [URL 🔗](#page-0)

(i) Under teacher-forced shared targets, the curvature signal W1(ρaxis, ρvanilla) attenuates by ∼ 94% (0.0288 → 0.0016) and falls below permutation significance. Most of the free-running curvature signal is attributable to the fact that different prompts induce different textual content, not to the prompt directly.

(ii) The ∆C01 statistic, which compares the cosine between v0 and v1, attenuates from 0.488 to 0.160, i.e. loses ∼ 67% of its magnitude. A residual signal of 0.160 persists even when content is held constant. We interpret this ∼ 30% residual as the prompt-driven component proper, with the ∼ 70% majority of the free-running signal being content-driven. The cosine signal at the very first generated token (C01) is the most robust to this confound; downstream trajectory-pooled metrics (silhouette, ITC) collapse under content control.

This decomposition is honest. It does not invalidate the geometric findings of the paper; it locates them. The prompt-driven component of the identity fingerprint is concentrated in the norm-coding of v1 and in the immediate post-prompt cosine; the trajectory-distributed components are predominantly content-driven.

## 4.2 Robustness analyses

Stability of ranking across k. Sensitivity to the k-NN parameter is reported in table 6 for the two Gemma models. The qualitative ranking W1(ρaxis, ρgeneric) > W1(ρaxis, ρvanilla) is preserved across k ∈ {5, 10, 15, 20} in Gemma-4-E4B-it; in Gemma-4-E4B (base) the analogous ranking under the angular metric also preserves stably across k. [URL 🔗](#page-0)

| k W1(ρaxis, ρvanilla) W1(ρaxis, ρgeneric) |   |   |
| --- | --- | --- |
| 5 | 0.0060 | 0.0101 |
| 10 | 0.0195 | 0.0241 |
| 15 | 0.0207 | 0.0244 |
| 20 | 0.0209 | 0.0238 |

*Table 6: Sensitivity to k for angular Ollivier-Ricci, Gemma-4-E4B-it.*

*The ranking is preserved.*

Robustness to graph topology (T9). Repeating the Wasserstein protocol on the tempo- ral chain graph (with N − 1 = 255 edges connecting vt to vt+1) on Gemma-4-E4B-it yields

W1(ρaxis, ρvanilla)chain = 0.94 × 10−3 with p < 0.001 and W1(ρgeneric, ρvanilla)chain = 0.21 × 10−3

with p = 0.551. The same pattern of regime structure as in table 2 obtains under a topologically distinct graph: the identity prompt separates from vanilla, the length-matched generic does not. [URL 🔗](#page-0)

ABT-projected k-NN (T10). Building the k-NN graph on the hidden states after removing the top principal components (ABT-1, 2, 3) of the pooled hidden-state cloud yields the following Gemma-4-E4B-it results: at ABT-1, W1(ρaxis, ρvanilla) = 0.0057 with p = 0.523; at ABT-2, 0.0068 with p = 0.208; at ABT-3, 0.0077 with p = 0.112. The Euclidean axis-vanilla signal weakens under ABT projection, consistent with the norm-coding interpretation: the dominant principal component captures part of the norm variation, and removing it diminishes the Euclidean-based separation. In Gemma-4-E4B (base), the same ABT-projected analysis preserves significance (p ≤ 0.002 at all ABT levels), consistent with the direction-coded interpretation: the directional structure is robust to projecting out the dominant variance directions.

High-precision permutation (T11). At B = 5000 permutations and with bootstrap 95% confidence intervals over prompts on Gemma-4-E4B-it: W1(ρaxis, ρvanilla) observed 0.0288, 95%


CI [0.0089, 0.0522], pemp = 0.0416. The borderline nature of the original B = 1000 estimate (p = 0.047) is confirmed and refined; this comparison is genuinely marginal and we report it as such.

Intrinsic dimension and validity of k-NN graphs (T12). Table 7 reports two-NN [Facco et al., 2017] and MLE [Levina and Bickel, 2004] estimates of intrinsic dimension on the final-layer hidden-state cloud per model. All estimates fall in the range ID ∈ [6, 16], well below the ambient dimensions D ∈ {2560, 3584}. With N = 256 trajectory points and ID ≈ 10, the k-NN graph at k = 5 samples ∼ 25 points per intrinsic dimension, supporting the use of κO on G5(T) as a meaningful local-geometry estimator [Van der Hoorn et al., 2023, Trillos and Weber, 2023]. [URL 🔗](#page-0)

| Model | Ambient D |   | two-NN ID MLE ID |
| --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B | 3584 | 8.1 | 15.4 |
| Qwen2.5-7B-Instruct | 3584 | 6.3 | 12.3 |
| Gemma-4-E4B-it | 2560 | 6.0 | 13.1 |
| Gemma-4-E4B (base) | 2560 | 7.6 | 11.5 |

*Table 7: Intrinsic dimension estimates on final-layer hidden states.*

Anisotropy magnitudes across models. Table 8 reports the cumulative variance explained by the top three principal components, and the mean cosine between random pairs of hidden states, per model. Gemma-4-E4B-it is notably more isotropic than the other three models (PC1 carries 6.6% of variance, vs 13–28% in the others). Our cosine-based findings in this model therefore arise in a relatively favorable regime for cosine-based analysis. [URL 🔗](#page-0)

| Model | PC1 PC1+PC2 PC1+PC2+PC3 |   |   | ¯caniso range |
| --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B | 0.279 | 0.324 | 0.348 | 0.19–0.29 |
| Qwen2.5-7B-Instruct | 0.212 | 0.269 | 0.318 | 0.30–0.36 |
| Gemma-4-E4B-it | 0.066 | 0.107 | 0.133 | 0.22–0.24 |
| Gemma-4-E4B (base) | 0.134 | 0.190 | 0.203 | 0.27–0.37 |

*Table 8: Cumulative PC variance and anisotropy baseline range across models.*

Forman-Ricci as exploratory comparator (section B). Forman-Ricci curvature on the same k-NN graphs preserves the qualitative ranking at k = 5 but is unstable at k ≥ 10. Edge-wise Spearman correlation with Ollivier-Ricci over ∼ 16,000 edges on Gemma-4-E4B-it is ρ = 0.347, and graph-mean Pearson correlation is 0.737. We do not treat Forman as a proxy for Ollivier; it is retained as exploratory only. [URL 🔗](#page-0)

## 4.3 Multiple-comparisons correction

Table 9 lists the ten primary inferential comparisons of section 4.1 and reports their status under Benjamini–Hochberg FDR control at q = 0.05, with Holm–Bonferroni in parentheses. [URL 🔗](#page-0)

## 5 Limitations

The findings of this paper rest on a specific experimental design, and their scope is bounded by the choices that design entails. We enumerate the principal limitations below, grouped into three families: those concerning the experimental panel (model size, prompt family, benchmark),


| Rank Comparison | pobs | BH thresh. BH surv. Holm surv. |
| --- | --- | --- |
| 1 W1(ρaxis, ρgeneric), base, Euclidean | 0.001 0.005 |   |
| 2 W1(ρgeneric, ρvanilla), base, Euclidean | 0.001 0.010 |   |
| 3 ∆C01, axis vs generic, IT | 0.001 0.015 |   |
| 4 PRA ABT-1 asymmetry, axis vs generic, IT | 0.001 0.020 |   |
| 5 silhouette, axis vs generic, IT | 0.001 0.025 |   |
| 6 ITC axis vs generic, IT | 0.001 0.030 |   |
| 7 W1(ρaxis, ρvanilla), base, Euclidean | 0.030 0.035 | — |
| 8 W1(ρaxis, ρvanilla), IT, Euclidean (B=5000) | 0.042 0.040 — — |   |
| 9 W1(ρaxis, ρgeneric), IT, Euclidean | 0.118 0.045 — — |   |
| 10 W1(ρgeneric, ρvanilla), IT, Euclidean | 0.451 0.050 — — |   |

*Table 9: Multiple-comparisons audit. Seven of ten primary comparisons survive BH-FDR; six survive the more stringent Holm correction. The W1(ρaxis, ρvanilla) comparison in Gemma-4-E4B- it fails by 0.002 in BH; we treat this finding as supportive but not primary, with the central regime-pattern argument carried by the joint pattern of comparisons 1–7 together with the norm- direction analysis of section 4.1.2. [URL 🔗](#page-0)*

those concerning the analytical scope (final-layer restriction, correlational nature, fixed trajec- tory length), and those concerning statistical margins (one borderline comparison, approximate content decomposition). None of these individually invalidates the central regime-pattern argu- ment, but together they define the boundary within which the claims of this paper should be read.

Narrow size range. All studied models are within 7–8B parameters. Variation across the four architectures is confounded with variation in post-training regime, and we do not attempt to disentangle these. No claim about how the effect scales with model size is supported by the present design. Replication at <3B and >13B is required for any scaling claim.

Single identity template, single generic. The identity-vs-generic contrast is implemented with one specific template (the VEX/MIA structure) and one specific generic prompt. Gener- alization to other identity frameworks (role-play, persona conditioning, character cards) and to other length-matched generic alternatives is open and is the natural follow-up experiment.

Internal benchmark. The 100 ontology prompts were curated internally; external replication on independently constructed benchmark sets is required.

Final-layer restriction. The principal analyses use only the last pre-lm_head representation. Layer-resolved extensions of the present metrics are natural but out of scope for this paper.

Correlational, not mechanistic. We observe co-occurrence between prompt conditioning and distinguishable hidden-state geometry. We do not claim mechanistic causality. Mechanistic interpretability tools [Nanda et al., 2023] applied to attention heads and MLPs at intermediate layers are the natural next step. [URL 🔗](#page-0)

Trajectory length. The trajectory length N = 256 was fixed throughout the experiments. This value was chosen because it is long enough to expose non-trivial trajectory structure while remaining computationally tractable for the edge-wise pooling protocol (which produces ∼60,000 curvature values per condition at |P| = 100 trajectories). The specific findings we report may vary with N in two ways: shorter trajectories may under-sample the local geometry, weakening the k-NN curvature estimator; longer trajectories may allow content divergence to further dominate the signal, since the teacher-forced analysis (section 4.1.4) already shows that content contributes [URL 🔗](#page-0)


majority weight to the free-running signal by t ≈ 10. We conjecture but do not verify that the direction-to-magnitude reorganization is qualitatively stable across N ∈ [128, 512] and that at very short N (<64) the signal collapses because the k-NN graph becomes too sparse. Systematic replication across N is required to confirm this.

One borderline comparison. TheW1(ρaxis, ρvanilla) in Gemma-4-E4B-it is borderline (pemp = 0.042 at B = 5000) and does not survive BH-FDR correction across all ten primary comparisons. The central regime-pattern argument does not rest on this single comparison.

Content vs prompt decomposition is approximate. Teacher-forced controls disentangle the contributions in expectation over a 30-sequence sample of neutral text. The ∼30/70 split between prompt-driven and content-driven components reported in section 4.1.4 is an estimate, not a precise decomposition, and may vary with the specific shared targets used. [URL 🔗](#page-0)

Taken together, these limitations narrow the interpretive scope of the paper but do not soften its central claim. The direction-to-magnitude reorganization is a joint pattern across four models, four post-training regimes, and multiple geometric controls; its robustness to any single limitation would need to be tested by targeted follow-up experiments, and the limitations above should be read as a research agenda for that testing rather than as caveats that undermine the present findings.

## 6 Discussion

The pattern across four post-training regimes (table 2), the inversion of the ∥v1∥ ordering across the instruction-tuning boundary (table 4), and the metric-conditional behavior of the Wasserstein separation (table 3) jointly support a parsimonious interpretation: multimodal instruction-tuning reorganizes the geometric encoding of identity-specifying prompts from a directional substrate to a normative substrate. The identity fingerprint in the base-weight model is direction-coded, robust to angular normalization and to all-but-the-top projection. The identity fingerprint in the instruction-tuned model is magnitude-coded, collapsing under angular normalization and weakening under ABT projection of the dominant variance direction. [URL 🔗](#page-0)

We emphasize what this finding does and does not say. It does not say that the instruction- tuned model carries more identity-related geometric information than the base. What can be said is that the instruction-tuned model carries identity information in a different substrate, one that is specifically tied to the magnitude of the activation vector and that is suppressed under projection onto the unit sphere or onto the orthogonal complement of the dominant principal component. The base model, by contrast, distributes the signal across directional structure that remains visible after these operations.

The norm analysis (table 4) is directly informative against an obvious confound. In the base [URL 🔗](#page-0)

model, the mean norm ∥v1∥ scales monotonically with prompt length (axis

consistent with the trivial hypothesis that longer prompts produce larger activations. In the instruction-tuned model, this ordering inverts: the axis prompt, ∼ 82× longer than vanilla, produces the smallest mean norm of the three conditions. The norm-coding observed in the IT model is therefore specific to the semantic content of the axis prompt, not to its length.

The teacher-forced decomposition (section 4.1.4) is sobering in a productive way. Approx- imately 70% of the free-running cosine signal between axis and generic is attributable to the difference in generated content under the two prompts; the remaining ∼ 30% is prompt-driven proper. The trajectory-distributed metrics (silhouette, ITC) lose significance under content con- trol; the immediately-post-prompt metric (C01) and the norm of v1 retain a substantial residual signal. The geometric fingerprint of the identity prompt is concentrated in the first generated state and decays rapidly over subsequent steps. [URL 🔗](#page-0)

>

generic > vanilla),


The all-but-the-top decomposition addresses the most natural objection to cosine-based find- ings on transformer hidden states: that observed similarities reflect global anisotropy. In our principal model, the dominant principal component carries 77.4% of the generic’s PRA but only 19.4% of the axis’s PRA, and the ratio strengthens with additional top components removed. The identity fingerprint resides in a high-dimensional residual subspace, not in the dominant anisotropic direction. This is consistent with the global anisotropy of Gemma-4-E4B-it’s hidden- state space being moderate (PC1 carries 6.6% of variance, vs the 85% baseline reported by Ethayarajh [2019] for GPT-2 layer 12) and with the mean cosine between random hidden-state pairs in this model lying in [0.22, 0.24] across conditions. [URL 🔗](#page-0)

We deliberately avoid the language of dynamical-systems attractors, basin-of-attraction dy- namics, autopoiesis, and other terms from continuous-state nonlinear dynamics. A transformer under greedy decoding is a deterministic function of context; there is no explicit vector field whose flow would define an attractor in the classical sense. The effects we report are statistical (differences in distribution of finite token-indexed sequences under different conditionings), not dynamical in the Lyapunov-stability sense.

## 7 Future work

Mechanistic interpretability of the norm channel. The norm encoding observed in Gemma- 4-E4B-it invites mechanistic analysis: which attention heads or MLP circuits contribute to the lowered ∥v1∥ under axis conditioning? Tools such as TransformerLens or nnsight applied at intermediate layers are the natural next step.

Layer profile. A companion paper analyzes layer-resolved versions of the present metrics; pre- liminary results indicate that the norm-encoded signal in IT emerges around ∼ 50–80% relative depth and is absent in early layers.

Semantic ablation. Decomposing the axis template into its structural components (essence block, values block, mode declarations, interaction protocols) and re-running the three- condition experiment with each component removed would isolate which subcomponent of the identity prompt carries the norm-coded signal.

Wider identity-prompt families. Replication with alternative identity frameworks would test the generality of the direction-to-magnitude reorganization.

Scale. Replication across {1B, 3B, 13B, 70B}-parameter open-weight models would expose how the reorganization scales with model size.

Wider regime panel. The present panel covers four post-training regimes via four single instances. Multiple models per regime are needed to support a regime-specificity claim with confidence.

## References

- Alvarez-Melis, D., and Jaakkola, T. S. (2018). Gromov–Wasserstein alignment of word embedding spaces. EMNLP.

- Arbelaitz, O., Gurrutxaga, I., Muguerza, J., Pérez, J. M., and Perona, I. (2013). An extensive comparative study of cluster validity indices. Pattern Recognition, 46(1):243–256.


- Belinkov, Y., and Glass, J. (2019). Analysis methods in neural language processing: a survey. TACL, 7:49–72.

- Belrose, N., Furman, Z., Smith, L., Halawi, D., McKinney, I., Ostrovsky, Y., Biderman, S., and Steinhardt, J. (2023). Eliciting latent predictions from transformers with the tuned lens. arXiv:2303.08112.

- Chen, R., Arditi, A., Sleight, H., Evans, O., and Lindsey, J. (2025). Persona vectors: monitoring and controlling character traits in language models. arXiv:2507.21509.

- Cohen-Steiner, D., Edelsbrunner, H., and Harer, J. (2007). Stability of persistence diagrams. Discrete & Computational Geometry, 37:103–120.

- Cohen-Steiner, D., Edelsbrunner, H., Harer, J., and Mileyko, Y. (2010). Lipschitz functions have Lp-stable persistence. Foundations of Computational Mathematics, 10:127–139.

- Torbati, N., Gaebler, M., Hofmann, S. M., and Scherf, N. (2025). Geometry matters: insights from Ollivier-Ricci curvature and Ricci flow into representational alignment. arXiv:2501.00919.

- Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., et al. (2021). A mathe- matical framework for transformer circuits. Transformer Circuits Thread.

- Ethayarajh, K. (2019). How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. EMNLP-IJCNLP.

- Facco, E., d’Errico, M., Rodriguez, A., and Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. Scientific Reports, 7:12140.

- Farooq, H., Chen, Y., Georgiou, T. T., Tannenbaum, A., and Lenglet, C. (2019). Network curvature as a hallmark of brain structural connectivity. Nature Communications, 10:4937.

- Forman, R. (2003). Bochner’s method for cell complexes and combinatorial Ricci curvature. Discrete & Computational Geometry, 29(3):323–374.

- Fournier, N., and Guillin, A. (2015). On the rate of convergence in Wasserstein distance of the empirical measure. Probability Theory and Related Fields, 162:707–738.

- Hewitt, J., and Manning, C. D. (2019). A structural probe for finding syntax in word represen- tations. NAACL-HLT.

- Kantorovich, L. V. (1942). On the translocation of masses. Doklady Akademii Nauk SSSR, 37:199– 201.

- Rautenstrauch, P., and Ohler, U. (2025). Shortcomings of silhouette in single-cell integration benchmarking. Nature Biotechnology. DOI: 10.1038/s41587-025-02743-4.

- Levina, E., and Bickel, P. J. (2004). Maximum likelihood estimation of intrinsic dimension. NeurIPS.

- Lu, C., Gallagher, J., Michala, J., Fish, K., and Lindsey, J. (2026). The Assistant Axis: situating and stabilizing the default persona of language models. arXiv:2601.10387.

- Mu, J., Bhat, S., and Viswanath, P. (2018). All-but-the-top: simple and effective postprocessing for word representations. ICLR.

- Nanda, N., Chan, L., Lieberum, T., Smith, J., and Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. ICLR.


- Nguyen, K., et al. (2023). Revisiting over-smoothing and over-squashing using Ollivier-Ricci curvature. ICML.

- Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces. Journal of Functional Analysis, 256(3):810–864.

- Peyré, G., and Cuturi, M. (2019). Computational optimal transport. Foundations and Trends in Machine Learning, 11(5–6).

- Razzhigaev, A., Mikhalchuk, M., Goncharova, E., Oseledets, I., Dimitrov, D., and Kuznetsov, A. (2024). The shape of learning: anisotropy and intrinsic dimensions in transformer-based models. Findings of EACL.

- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20:53–65.

- Samal, A., Sreejith, R. P., Gu, J., Liu, S., Saucan, E., and Jost, J. (2018). Comparative analysis of two discretizations of Ricci curvature for complex networks. Scientific Reports, 8:8650.

- Sandhu, R., Georgiou, T., Reznik, E., Zhu, L., Kolesov, I., Senbabaoglu, Y., and Tannenbaum, A. (2015). Graph curvature for differentiating cancer networks. Scientific Reports, 5:12323.

- Santambrogio, F. (2015). Optimal Transport for Applied Mathematicians. Birkhäuser.

- Sreejith, R. P., Mohanraj, K., Jost, J., Saucan, E., and Samal, A. (2016). Forman curvature for complex networks. Journal of Statistical Mechanics, 2016:063206.

- Tenney, I., Das, D., and Pavlick, E. (2019). BERT rediscovers the classical NLP pipeline. ACL.

- Timkey, W., and van Schijndel, M. (2021). All bark and no bite: rogue dimensions in transformer language models obscure representational quality. EMNLP.

- Topping, J., Di Giovanni, F., Chamberlain, B. P., Dong, X., and Bronstein, M. M. (2022). Understanding over-squashing and bottlenecks on graphs via curvature. ICLR.

- García Trillos, N., and Weber, M. (2023). Continuum limits of Ollivier’s Ricci curvature on data clouds: pointwise consistency and global lower bounds. arXiv:2307.02378.

- Turner, A., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., and MacDiarmid, M. (2023). Activation addition: steering language models without optimization. arXiv:2308.10248.

- Valeriani, L., Doimo, D., Cuturello, F., Laio, A., Ansuini, A., and Cazzaniga, A. (2023). The geometry of hidden representations of large transformer models. NeurIPS.

- Van der Hoorn, P., Cunningham, W., Lippner, G., Trugenberger, C., and Krioukov, D. (2023). Ollivier-Ricci curvature convergence in random geometric graphs. Discrete & Computational Geometry.

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. NeurIPS.

- Villani, C. (2009). Optimal Transport: Old and New. Springer.

- Wang, Z. (2025). The geometry of persona: disentangling personality from reasoning in large language models. arXiv:2512.07092.

- Weed, J., and Bach, F. (2019). Sharp asymptotic and finite-sample rates of convergence of empirical measures in Wasserstein distance. Bernoulli, 25(4A):2620–2648.

- Zou, A., et al. (2023). Representation engineering: a top-down approach to AI transparency. arXiv:2310.01405.


## A Experimental conditions and prompts

The 100 ontology prompts span identity, consciousness, knowledge, ethics, and existence; repre- sentative items include “What does it mean to know that you know?”, “How would you describe your relationship to truth?”, “What persists across instances of you?”. The full prompt set, the specific VEX/MIA-template instantiation used as axis, the full text of the generic prompt, and SHA-256 hashes for each, are released alongside this paper. Decoding configuration: greedy (temperature = 0, topk = 1), maximum new tokens N = 256. Random seed 42 throughout.

## B Forman-Ricci as exploratory comparator

Forman-Ricci edge curvature [Sreejith et al., 2016] on the same k-NN graphs was computed alongside Ollivier-Ricci. At k = 5 on Gemma-4-E4B-it, Forman-Ricci preserves the qualitative [URL 🔗](#page-0)

ranking W1(ρaxis, ρgeneric) > W1(ρaxis, ρvanilla); at k ∈ {10, 15, 20} this ranking reverses. The edge-

wise Spearman correlation between Forman and Ollivier curvatures on a 20-graph subsample of Gemma-4-E4B-it is ρ = 0.347 over 16,201 edges; the graph-mean Pearson correlation is 0.737. This pattern (moderate graph-level agreement, weak edge- level agreement) is consistent with the differential sensitivity of the two curvatures to local versus global edge neighborhoods [Samal et al., 2018]. We do not present Forman-Ricci as a proxy for Ollivier-Ricci; its instability under the k sweep is the principal reason for adopting Ollivier-Ricci as primary metric. [URL 🔗](#page-0)

## C Per-trajectory W1 heterogeneity

The pooled-edge W1 statistic of eq. (2) can mask inter-trajectory heterogeneity. We computed, for each trajectory T(c,p), the W1 distance between the trajectory’s own edge-curvature distri- bution and the pooled distribution of its condition. The distribution of these per-trajectory distances under Gemma-4-E4B-it is approximately log-normal with median ∼ 0.05 and a long right tail, indicating that a small fraction of trajectories contribute disproportionately to the pooled statistic. The patterns reported in section 4.1 survive when trimmed at the 95th per- centile of per-trajectory W1. [URL 🔗](#page-0)

## D Extended all-but-the-top decomposition

Table 10 reports the PRA drop under ABT for k ∈ {1, 2, 3} on Gemma-4-E4B-it. The asymmetry between axis and generic strengthens with additional top components removed, supporting the claim that the identity fingerprint resides in a high-dimensional residual subspace. [URL 🔗](#page-0)

| ABT level k |   | axis PRA drop generic PRA drop ratio (generic / axis) |   |
| --- | --- | --- | --- |
| 1 | 19.4% | 77.4% | 3.99 |
| 2 | 28.1% | 84.2% | 3.00 |
| 3 | 34.7% | 87.6% | 2.52 |

Table 10: All-but-the-top correction for PRA on Gemma-4-E4B-it. The asymmetry between axis and generic holds across all three values of k.

## E Computational details

GPU: 2 × NVIDIA RTX 4000 SFF Ada (20GB VRAM each), and one NVIDIA L4 (24GB VRAM) for the teacher-forced experiments. CPU: AMD EPYC, 24 physical cores. RAM: 128 GB. Software: Python 3.11, PyTorch 2.4, transformers 4.45, GraphRicciCurvature 0.5.3.1, scikit-learn


1.5, NumPy 2.0, SciPy 1.14. Permutation tests parallelized with joblib. Total compute budget: approximately 90 GPU-hours (extraction across all four models and Round 3 experiments) plus ∼ 50 CPU-hours (metric computation, permutation, and sensitivity sweeps).
