# Proposed order: temporal information, architecture and interactive imitation

Planning recommendation for Rodrigo, 27 October 2026. The final agreed window
is200ms with20ms spacing (11states,201inputs), with a2×193 capacity control.
The100ms/2×164 examples below preserve the initial proposal; the active protocol
is documented in TEMPORAL_WINDOWS.md. Subsequently Rodrigo
authorized beginning temporal windows with1024 episodes. The three-model
PyTorch comparison is now implemented and launched; see
[execution protocol and current artifacts](TEMPORAL_WINDOWS.md).
Architecture sweeps, dropout, BatchNorm and DAgger remain proposed. Existing
expert and simulation rules remain fixed. This supplements the research plan.

## Decision supported by current evidence

Use **1024 total episodes:819 training /205 validation** as a practical fixed
budget for the next controlled ablations. It is not a statistically established
optimal dataset size. Preserve the exact existing split and training-only
normalizers. The stratified1024 dataset is a reasonable primary candidate because
its current autonomous divergence is lower; keep uniform1024 as a reference, not
as evidence that its supervised fit is always worse.

The CPU campaign is now complete. Its1024 policies have0/410 primary task arrivals
on the full paired validation union; both2048 policies have0/820. The extended
uniform512 fit reproduced60 epochs exactly, then stopped at79; the small offline
improvement did not rescue control. These observations argue against making
exhaustive memoryless supervised tuning a prerequisite for temporal experiments.
They do not prove the cause of failure or that architecture changes cannot help.

## Recommended sequence

### 1. Freeze a same-backend baseline at1024

Train the reference PyTorch21→128→128→9 model on the selected existing1024 split,
using exact reference normalizers, original hyperparameters, FP32, same clipping
protocol and patience8/min_delta1e-4 with no epoch cap. The current full-size
learning curves are sklearn models; only the64 pilot has been ported/trained in
Torch. Compare new Torch variants with a Torch1024 baseline so framework changes
are not mistaken for architecture or memory effects.

Keep the full205 own holdouts and both205-goal banks,410 total, for development
evaluation. All these conditions stay outside training/aggregation. The26-case
anchor remains a supplementary across-size reference, not a substitute for full
validation. Use the same initial conditions,40s horizon and physical rules.

### 2. Test one simple causal observation window

Start with a small observation-stack MLP rather than adding recurrent architecture
at the same time. Candidate:100ms of history, six state snapshots spaced20ms
apart, ending at the current decision instant. Continue commanding the simulator
at its original1ms period; history sampling does not change the action hold time.

A concrete proposed encoding is six18-dimensional measured states plus the
current3D goal once:111 inputs. Repeating an unchanged goal six times adds no
new information. Use current and past measurements only, never future rows,
expert phases, expert references, or expert actions as policy inputs.

At episode start, repeat the first available state for missing history; reset
the buffer at every episode. Apply the identical rule at training and inference.
Split by episode before building windows; fit statistics only on training.
Generate windows on demand to avoid duplicating overlapping arrays in memory.
A100ms stack is not guaranteed to recover25s navigation timeout or integrated
expert references; failure would not disprove every form of memory.

Control for the larger input layer. With bias on every dense layer:

|Candidate|Inputs|Hidden widths|Parameters|Purpose|
|---|---:|---|---:|---|
|Instant reference|21|128,128|20489|Existing information budget|
|Temporal stack|111|128,128|32009|Additional causal observations|
|Instant capacity control|21|164,164|32153|Approximately match temporal parameter count|

An improvement over both instantaneous controls is stronger evidence for useful
history than comparison only with a smaller parameter budget. It still does not
isolate every optimization/representation difference. Temporal imitation can
also exploit shortcuts; history is a hypothesis, not a promised solution.
[Wen et al.,2021](https://proceedings.mlr.press/v139/wen21d.html).

### 3. Tune architecture on the informative representation

Do a small predeclared screen, then repeat promising comparisons with three
training seeds. Retain an instantaneous control when architecture changes so
interactions with history remain visible. A modest initial screen can include
compact2×64, reference2×128, and deeper3×96. On21 inputs these have6153,20489 and
21609 parameters respectively; on111 inputs recalculate and report counts.
Changing depth and reducing parameters are different comparisons.

Only then test regularization independently, e.g. dropout0 versus0.1, and
BatchNorm off versus on. Do not combine depth, width, dropout, BatchNorm, learning
rate, history length and dataset changes in the first comparison. Examine
training/validation gaps and joint-specific errors before concluding regularization
is needed; classical overfitting has not been established as our main failure.

Dropout changes training and is inactive during normal evaluation. BatchNorm
normally uses accumulated training statistics at evaluation. Use explicit eval
mode, validate batch-one deployment and preserve equivalent exported inference.
These layers are neither a replacement for training-only input standardization
nor a guarantee of more accurate control.
[Dropout](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Dropout.html),
[BatchNorm1d](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BatchNorm1d.html).

The trainer/exporter now supports configurable hidden depth with either
instantaneous21D inputs or causal state stacks. Rodrigo subsequently authorized
the32→64→128 and32→64→128→64→32 screen; see
[depth/shape protocol](MLP_DEPTH_AND_SHAPE.md). Dropout/BatchNorm still require
separate extensions and export tests and are not included in this screen.

### 4. DAgger as a separate, complementary collection intervention

DAgger changes which states receive expert labels; a history window changes what
the learner observes. They address different hypotheses and may be combined.
It is not necessary to reach the smallest possible expert-state validation error
before studying learner-visited states. The original method explicitly addresses
sequential state distributions induced by learned actions.
[Ross et al.,2011](https://proceedings.mlr.press/v15/ross11a.html).

First resolve [the stateful expert-labeling protocol](BASELINE_LIMITATIONS.md#l2--stateful-expert-labeling-is-unresolved-for-dagger).
Maintain one causal expert state per learner episode, update it once per physical
step, distinguish queried/applied commands and audit recovery labels. Extra
supervised data do not remove missing observation history by themselves. Keep
validation goals out of aggregation, count added queries and compare a matched
new-demonstration budget. Lecturer acceptance of the collection extension remains
unconfirmed; this plan does not assert authorization to implement DAgger.

## Selection and interpretation

Every comparison reports full held-out command MAE/RMSE by wheel/joint, group
summaries, strict5% nonzero fidelity and separate zero-command error. Pair those
with complete matched autonomous task success, final/trajectory EE and base
errors, yaw, stopping, clipping, and time to departure. Account for memory,
parameters, inference latency and total preprocessing/history cost.

Do not promote a model solely because its aggregate supervised NRMSE falls.
Keep supervised-fit and autonomous-control findings distinct, report unsuccessful
policies as such, and retain both candidates if their rankings disagree. Seeds
should be fixed before outcomes (candidate20261025/20261026/20261027). Existing
validation is development data used for selection; final claims require a new
untouched test protocol after choices are frozen.

Preferred next action: same-backend1024 reference plus the three-case temporal
ablation, then focused architecture/regularization tuning; no exhaustive search
for an unattainable minimum supervised error as a prerequisite.
