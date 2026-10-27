# Privileged-input ablation: does the expert's hidden state explain the labels?

Protocol pre-registered on 27 October 2026, before any result was observed.
Proposal P1 of [CLAUDE_ANALYSIS.md](CLAUDE_ANALYSIS.md). Results are appended
below the protocol once available; the protocol text above the results line is
not edited afterwards.

## Question

The expert in `expert/controllers.py` acts on the observation **and** on
controller memory that the 21 baseline inputs do not contain: the navigation
phase (`align`/`drive`/`orient`), the `nav.done` flag that gates arm reaching,
and the integrated arm reference `q_des` that enters the PD law with gain 100.
If one observation maps to several expert actions depending on that memory, a
regressor trained with squared error learns their conditional mean, which the
expert never emits. This ablation measures how much of the supervised error is
attributable to that unobservable memory.

## Design

Four fits that differ **only** in extra input columns taken from the episode
NPZ diagnostics (`expert_phase`, `base_navigation_done`, `expert_q_des_arm`):

| Arm | Inputs | What it adds |
|---|---:|---|
| baseline | 21 | Control: the agreed baseline inputs |
| phase | 25 | One-hot over `align`/`drive`/`orient` plus `nav.done` |
| reference | 28 | PD reference lead `expert_q_des_arm - q` (7 values), standardized on training |
| full | 32 | Both blocks |

Held constant across arms: dataset `stratified1024_nominal` (1024 episodes,
16×8×8 jittered grid, seed 20261026, nominal start), the geometry-only
819/205 episode split from the sampling plan, standard scalers for the 21
baseline inputs and 9 outputs fitted on training rows only, MLP 21→128→128→9
with ReLU, Adam with learning rate 1e-3, batch 1024, L2 alpha 1e-4 in the
sklearn convention, Glorot initialisation from seed 20261025, patience 8 with
min_delta 1e-4 on the episode-macro balanced NRMSE, no epoch cap, float32 on
CUDA with deterministic algorithms. The code is
`learning/privileged_ablation.py`, launched by `scripts/privileged_ablation.py`.

The extra one-hot and flag columns are fed raw in {0, 1}; the reference lead
is standardized with training statistics. Parameter counts grow by 512 to
1408 with the added inputs; this is noted and not treated as a capacity
confound, because the baseline model is already far from capacity-limited
(research log R41 context) and the comparison of interest is the direction and
magnitude of the change, not a tuned equal-parameter contest.

## Metrics

- Primary: best validation **episode-macro balanced NRMSE**, the project's
  selection metric, per arm.
- Secondary: pooled per-output RMSE in physical units; pooled balanced NRMSE
  split by `nav.done` (navigating versus navigation done) and by expert phase.

## Pre-declared reading of outcomes

- **Hidden state matters** if `full` reduces the primary metric by at least
  30 % relative to `baseline`, and the by-`nav.done` split shows where (wheels
  while navigating, torques after done).
- **Phase versus reference**: if `phase` captures most of the gain, the
  deployable surrogates are the episode clock and base-goal geometry (P3). If
  `reference` captures most of it, the arm label is dominated by the hidden
  integrator and the gravity-residual or a memory model (P4, P8) is the lever.
- **Null result** if `full` improves by under 10 %: hidden state is not the
  main source of supervised error, and the closed-loop failure must be
  explained by compounding and coverage instead (P2 becomes first priority).
- Between 10 % and 30 %: partial; both tracks stay open and the by-phase
  tables decide the order.

## Scope and limits

- The oracle models consume signals a deployed policy cannot observe. They are
  **not** exported as `MLPPolicy`, cannot be loaded by the autonomous
  evaluator, and are never a candidate for the final policy.
- This is an offline measurement on one dataset realisation and one seed. It
  does not measure closed-loop success. A closed-loop oracle would need a
  shadow expert instance updated along the learner's trajectory; that is a
  separate protocol (CLAUDE_ANALYSIS.md, P1 second stage).
- Lower supervised error with privileged inputs does not by itself prove that
  removing the ambiguity would rescue control; it bounds how much of the
  supervised gap is due to it.

---

## Results

Run `runs/diagnostics/privileged_ablation_20261027` (27 October 2026, 00:14–00:27,
RTX PRO 5000). Dataset `stratified1024_nominal`, split 819/205. Training rows by
expert state: align 18.6 %, drive 44.1 %, orient 37.3 %; `nav.done` true on 30.2 %.

| Condition | Inputs | Best epoch | Val NRMSE (episode macro) | Change vs control | NRMSE while navigating | NRMSE after `nav.done` |
|---|---:|---:|---:|---:|---:|---:|
| control (21 baseline inputs) | 21 | 13 | 0.1134 | — | 0.0769 | 0.5482 |
| + phase one-hot + `nav.done` | 25 | 19 | 0.0801 | −29 % | 0.0622 | 0.2539 |
| + arm reference lead `q_des − q` | 28 | 15 | 0.0853 | −25 % | 0.0722 | 0.3531 |
| + both | 32 | 15 | 0.0704 | −38 % | 0.0533 | 0.3105 |

Pooled physical RMSE by regime:

| Condition | Wheels, navigating (rad/s) | Wheels, after done (rad/s) | Arm, navigating (N·m) | Arm, after done (N·m) |
|---|---:|---:|---:|---:|
| control | 0.215 | 0.346 | 0.042 | 1.642 |
| + phase | 0.180 | 0.107 | 0.028 | 0.776 |
| + reference | 0.211 | 0.216 | 0.030 | 0.991 |
| + both | 0.152 | 0.169 | 0.026 | 0.894 |

By expert phase (balanced NRMSE, control → both): align 0.585 → 0.351,
drive 0.060 → 0.045, orient 0.190 → 0.120.

### Reading against the pre-declared thresholds

- **Hidden state matters.** The full condition cuts the selection metric by 38 %,
  above the 30 % threshold. The phase block alone reaches 29 %, the reference
  block 25 %; their effects are largely additive.
- **Where the error lives.** The control model is accurate while the base is
  navigating (NRMSE 0.077) and seven times worse after `nav.done` (0.548). After
  done the expert's wheels are exactly zero and the arm starts reaching; the
  control model cannot tell this regime from late navigation, so it keeps
  emitting wheel commands (0.35 rad/s RMSE where the truth is 0) and misses the
  torques by 1.6 N·m. The `align` phase, 19 % of rows, is the other weak spot
  (0.585): turning in place at the start looks identical to turning in place
  during `orient`, but the heading error that drives the command differs.
- **Which surrogates to build.** The phase block is the larger and cheaper
  lever, and it is reconstructible from observables: the episode clock, and the
  base-goal geometry (distance to the standoff goal and bearing error to
  `goal_bearing`, both deterministic functions of the target). This points to
  proposal P3 with a clock. The reference block is second; its deployable
  surrogates are the gravity-residual output (P4) or memory (P8).
- These are offline numbers on one dataset and one seed. They bound how much of
  the supervised gap is due to hidden state; they do not show that removing it
  rescues closed-loop control. That is what the deployable variant tests next.
