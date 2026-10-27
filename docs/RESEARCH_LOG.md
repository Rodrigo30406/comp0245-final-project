# Research Log and Evidence for the Report

Updated: **25/10/2026**, local time Europe/London. This document connects the  
[plan](RESEARCH_PLAN.md), decisions, and actually achieved results. It complements `AGENTS.md` and technical reports; it does not replace them.

**Current status: P1 closed for nominal test bench; P2 closed as exploratory pilot.** There is a pilot, ETL, two regressors, and the first autonomous paired comparison (39 rollouts). Both models fail to achieve stable arrival in this validation; descriptive diagnostics of initial deviations, orientation, clipping, and offline ranges are provided. Causes remain open; closing the pilot does not imply stable control has been achieved.  
No final test, seed-based analysis, or evidence of robustness is available. The inputs of this first version retrospectively reconstruct the work with its artifacts; they do not constitute a preregistration of prior experiments. Dates 23/24/25 identify documented milestones, not reconstructed hours.  

## Plan Tracking

| Phase / experiment plan section | Checked status | What is missing to close |
|---|---|---|
| P0 / part of E00: contract and expert | Requirements reviewed, controller audited, shared rules and regressions of the original loop | Complete domain qualification with separate geometric and external criteria |
| P1 / E00: experimental bench | **Closed in the nominal development domain**: motor preserved, external evaluator, tests and 39 paired rollouts | Extend when new sensors, noise, contacts, or other conditions are studied |
| P2 / E01: pilot and baselines | **Closed as exploratory pilot**: 64 episodes, OLS/MLP and mean, split 51/13, MLP history, autonomous evaluation and described failures | Remaining for future studies: data quantity curves, more seeds, and causal isolation; Ridge and trees are pending variants, not yet implemented |
| Part of E08: wheel symmetry | Two lateral outputs and exact expansion to four wheels | No ablation 9 vs 11 outputs: do not claim empirical improvement from reduction |
| P3 / E02–E05: quantity, coverage, and sensors | Initial ideas and configurations documented | Run data curves, extend box, start-ups and additional inputs |
| P4 / E06–E07: memory and Fourier | Plan proposals | Windows/temporal and capacity/information controls |
| P5 / E09–E10: efficiency and robustness | Scaling of data collection, parameters/artifacts and local inference latency | Full latency, autonomous control, perturbations, and performance boundary |
| P6: extensions | Not started | Only select extensions justified by observed failures |
| P7: confirmation and report | Evidence and results notes gathered | Reserved test, frozen finalists, repetitions, uncertainty, demo, and report |

The plan is broad and adjustable, not a list of completed tasks. Worker optimization was productive work in P1, not evidence of better learning nor a substitute for E02 (data quantity).

## Explicit adaptations relative to the initial proposal

| Proposal of plan | Current implementation | Justification / limit |
|---|---|---|
| Initial radial pilot | Uniform XY box [-2,2] m, Z [0.3,0.8] m; nominal startup | Decision deferred to Rodrigo; development domain, not guaranteed workspace |
| Ridge as baseline linear | OLS first, at Rodrigo's request | Simple reference without tuning; Ridge follows slope |
| Four wheel commands | Learn two sides + seven arm torques, reconstruct four wheels | Exact identity of the expert; preserve original commands from dataset |
| AdamW and candidate batches 128/256/512 | Adam, L2 from sklearn, batch 1024 | First reproducible recipe with readily available library. L2 of Adam does not equate to decoupled weight decay of AdamW; do not present as literal execution of that proposal |
| Success selection via rollouts | Initial MLP selection by NRMSE offline on validation | Initial selection before evaluator; now rollouts exist, no reselection of weights: do not refer to that checkpoint as better controller |
| Multiple seeds and large suites | One MLP seed, 13 validation episodes | Exploratory screening. Do not assert statistical significance nor final test |
| Group-balanced loss | MLP tuning: normalized MSE per channel; evaluation 50/50 wheels/arm | Functions differ; channel weighting in tuning can be studied as a variant |

Normalizers are trained only on training data. No sensors, windows, bearing sine/cosine, or Fourier terms were introduced in these baselines. It has not been tested that any of these changes are necessary or useless.

## Progress and decisions

### R01 — Environment, expert, and experimental bank (23–24/10; consolidation 25/10)

- Question: Can we run and record reproducible episodes without altering the teacher?
- Decision: RoboEnv sibling; single episode engine, preserved expert, rules in configuration. Metadata distinguishes the stopping criterion used and the post-state.
- Evidence: regression fixtures and sequence/alignment, reset, integrity, interruption, and resume tests.
- Limit: Preserving behavior does not certify physical correctness or optimality of the expert.
- Sources: [architecture](ARCHITECTURE.md), [data](DATA_COLLECTION.md), `tests/` and snapshots of each dataset.
- Use in report: implementation and reproducibility; attribute the original controller to the teacher.

### R02 — Parallel data collection and worker selection (24–25/10)

- Question: How to reduce acquisition time without changing the demonstrations?  
- Protocol: Same scenarios across worker values, dynamic queue, limited thread numbers; preserve hashes/results and monitor CPU/RAM/temperature.  
- Result: Increasing workers slows down each episode; 32 did not offer a clear advantage over 24. In the expanded sweep of 64 episodes: 16 → 52.69 s; 20 → 48.46 s; 24 → 45.80 s. Further increases above 16 provide less than 10% speed improvement each.  
- Final decision by Rodrigo: **16 workers**, prioritizing resource margin. Replaces previous proposals of 24 and 20; the default portable setting remains at 1.  
- Limitation: Local load, few repetitions, no universal optimum or energy measurement. The replicas are not independent learning examples.  
- Sources: [initial measurement](WORKER_BENCHMARKS.md), [marginal sweep](MARGINAL_WORKER_SCALING.md), `runs/worker_scaling*/` and `runs/worker_marginal_20261025/`.  
- Usage in report: brief justification of the experimental setup; do not dedicate primary space to ML results.

### R03 — Performance profiling and ablation studies (25/10)

- Question: Is the cost coming from the coordinator, queue gaps, or simulation?  
- Evidence: Physics accounts for ~57% of the accumulated internal worker time; coordination and verification activities overlap. Removing verifications showed no clear improvement outside variation; time overlaps were not treated as potential savings.  
- Candidate: Block validation, ~6% less time in a trial, still a prototype. Uncompressed saving increased disk usage by ~50% without clear benefit; core affinity showed no demonstrated advantage.  
- Decision: Do not activate experimental changes nor weaken provenance; retain the productive implementation.  
- Limitation: No complete causal breakdown of cache, SMT, frequency, and temperature; restricted counters.  
- Sources: [profiling](COLLECTION_PROFILING.md), [ablation studies](PERFORMANCE_FACTOR_ANALYSIS.md), eleven batches and 22,528 matrix comparisons.  
- Usage in report: efficiency and controls; do not present prototypes as deployed improvements.

### R04 — Pilot dataset and split (25/10)

- Question: Do we have a first comparison of regressors without temporal leakage?  
- Dataset selected: a single copy `w16_r1`, identifier `20261024T234048Z_worker_scaling_pilot_not_final_training_4e159d47`.  
- Data: **64 episodes / 617,877 transitions**, 21 inputs and 11 original commands; symmetry allows extracting 9 independent outputs.  
- Fixed split by episode, seed 20261025: training 51 / 490,730 rows; validation 13 / 127,147. No final test set. Keep all episodes, including terminations `stalled`.  
- Decision: save split manifest and training scalers; do not mix benchmark copies nor distribute episode rows across partitions.  
- Limit: correlated rows, same box and nominal startup; 617,877 rows do not equate to that many independent scenarios.  
- Sources: [pipeline](BASELINE_TRAINING.md), `runs/training/linear_pilot_20261025/split.json`.

### R05 — OLS and comparison with mean (25/10)

- Question: What does a linear model explain compared to a constant predictor?  
- Configuration: OLS with intercept, inputs/outputs standardized only on training data, no regularization or tuning, 198 parameters + 60 preprocessing values.  
- Validation result: balanced NRMSE 0.597467 versus 0.938043 for the mean; wheel velocity MAE of 0.9333 rad/s. Mean per episode and group; do not sum wheel or torque units.  
- Decision: retain as baseline and show MAE/RMSE per joint and per side, both compared to the training mean.  
- Sources: [pipeline and table](BASELINE_TRAINING.md), [individual errors](PER_JOINT_ERRORS.md), `runs/training/linear_pilot_20261025/`.  
- Limit: offline evaluation; neither NRMSE nor percentage improvement constitutes control success.

### R06 — Contextualize errors and detect limits (25/10, post-diagnosis)

- Question: Is 0.93 rad/s small or large compared to commands?  
- Result: Expert episode magnitude ~1.57 rad/s, MAE/magnitude ratio ~59%. When the expert sends zero, the linear predictor estimates magnitudes around 0.52–0.54 rad/s based on those observations.  
- Torques: The same physical MAE can have different meanings across joints. A large offset makes dividing solely by the mean magnitude misleading; accompany with std and constant predictor.  
- Decision: Present individual MAE/RMSE metrics as the primary measure for the arm; provide a global average as a secondary metric. Keep the metric recorded, do not replace it to favor results.  
- Sources: [interpretation](BASELINE_ERROR_INTERPRETATION.md), `error_context.json`, `per_output_metrics.csv`.  
- Limitation: Post-hoc diagnosis; do not confuse predicted commands based on expert states with autonomously observed displacements.  

### R07 — MLP 128×128 and paired offline comparison (25/10)

- Question: Does nonlinear capacity reduce error using identical data and inputs?  
- Controls: Exact equality of split, dataset, and normalizers compared to OLS; same metrics per output/episode.  
- Configuration: 21→128 ReLU→128 ReLU→9 linear; Adam lr 0.001, L2 0.0001, batch 1024, seed 20261025. Maximum 60 epochs, patience 8 and delta 0.0001; holdout automatic by rows disabled.  
- Result: 22 epochs; checkpoint 21; NRMSE training 0.132817, validation **0.203130** (−66.0% compared to OLS); MAE wheels **0.2751 rad/s**.  
- Negative result to preserve: torque_7 degrades from MAE 0.000083 to 0.048657 N·m. Torques 1–6 improve; do not assert that all outputs improved.  
- Cost: 20,489 parameters, NPZ 78,789 B; local inference median 15.02 µs, without state/ simulation read. Historical latencies of OLS are not paired benchmarks.  
- Sources: [MLP and full table](MLP_BASELINE.md), `runs/training/mlp_pilot_20261025/`, `comparison.json`, `history.json` and CSV. All 18 tests pass.  
- Limitation: one seed, validation used for selection, no independent test or autonomous rollout. The central hypothesis of memory/coverage has not yet been validated.  

### R08 — Audit of "62 successes" versus geometric reach (25/10)

**Correction of previous interpretation:** do not assert that "the expert only reached 62 points." There are **62 labels `success` and 2 `stalled`, but 64/64 final errors below 1 cm**, and also 64/64 within the stopping metric used.  

| Episode | Original label | Final error | Partition |
|--------|----------------|------------|----------|
| `episode_000006` | `stalled` | 0.0494 mm | training |
| `episode_000017` | `stalled` | 1.1617 mm | training |

In `expert/controllers.py:check_stop`, stopping is checked before success.  
Both branches can be triggered when the minimum of 8 s is exceeded and the error is already within tolerance; stopping is returned first in these two episodes. This behavior occurs in both cases. The success flag from the collector reproduces this termination: it does not independently measure geometric reach.

- Decision: preserve code, labels, and datasets; distinguish **success according to script**, **geometric error**, and **stable arrival with permanence**. No 0.5 s permanence is certified by this count.  
- Validation contains 13 success labels; training contains 49 success and two stalled. There is no evidence here of physically impossible objectives nor of two cases that did not approach the target.  
- Evidence: original metadata and [audit of 64 results](../runs/research_audit_20261025/expert_outcomes.json). Post-hoc descriptive audit, no protocol modification, no new rollouts.  
- Use in report: limit the expert label and provide a reason for an external evaluator; correct the previous sentence in any draft.  

### R09 — Autonomous evaluator and closure of P1 (25/10)

- Question: Do the models control the robot from their own states, and can they be compared without altering the expert?  
- Implementation: Shared episode motor with explicit policy/horizon mode; no expert calls in learned policies; frozen models/scenarios with matched initial states.  
- Protocol saved before full batch: 13 validation scenarios, expert/OLS/MLP, 40 s, arrival ≤1 cm during 0.5 s, no stopping upon arrival or stagnation. Explicit command limits (wheels ±5.272727 rad/s, arm ±30 N·m), traces and saturation diagnostics. These are not certified hardware limits.  
- Result: **39/39 numerically complete; expert arrival stable in 13/13, OLS 0/13, MLP 0/13**, also zero learned policies at 5 cm. The expert reaches within less than 1 cm in 11/13: arrival and maintenance until the end are not equivalent.  
- Main negative result: offline improvement of the MLP did not produce autonomous control. The MLP diverges severely, with extreme simulated distances; these are simulation system failures, not physically plausible trajectories. All cases are preserved and weights/metrics are not adjusted to hide them.  
- Validator validation: 18 prior tests + 8 specific ones (including post-hoc integrity reinforcement); expert prefixes match in 1,267 sampled transitions and 13 initial stops. Smoke with 1/2 workers: 18 identical matrices; raw commands match model inference on previous state.  
- Decision: **P1 closed for the nominal development bank**; P2 remains open for diagnostics. Before any new architecture, inspect a representative case and its first deviations, normalized inputs, and saturation. Cause of divergence has not yet been isolated.  
- Limit: same validation already used, one seed, no final test/robustness/contact checks. Numerical completeness does not imply physical reliability or success.  
- Sources: [evaluator, protocol, and results](AUTONOMOUS_EVALUATION.md), `runs/evaluation/validation_20261025/`, `analysis.json`, `integration_audit.json`, traces and logs. The full batch lasted ~113 s with 16 workers.  
- Use in report: offline vs closed-loop comparison, BC limits, and distribution shift diagnosis. Do not assert that a specific cause has been identified or the failure resolved.

### R10 — Command Proximity with Physical Tolerances (25/10, post hoc)

- Rodrigo's question: What does a lower MAE/NRMSE mean if the control remains unstable?
- Decision: Keep original metrics and add the fraction of commands within tolerance for each output and for nine simultaneous outputs, using the P95 of absolute error. Separate offline fidelity from task success.
- Descriptive protocol: Frozen models, same 13 episodes / 127,147 rows, without clipping or tuning; tolerances ±0.05 / 0.1 / 0.2 / 0.5 / 1 in physical units per channel. No threshold was declared acceptable for stability.
- Result at ±0.1: MLP right wheel 33.0%, left wheel 31.7%; nine simultaneous commands 5.06% (averaged fractions per episode). P95 of wheel error across all rows: 0.769 / 0.894 rad/s. At ±0.5, nine simultaneous commands reach 71.61%, which still does not demonstrate control.
- Limitation: comparison against expert states; the states visited by the model require a separate consultation protocol that respects the expert's memory. Do not confuse NRMSE 0.203 with 20.3% error.
- Evidence: [Command Fidelity](COMMAND_FIDELITY.md), `runs/diagnostics/command_fidelity_verified_20261025/`; two new tests pass and two computed reports match. No retraining was performed and no new rollouts were executed.

### R11 — Base, Arm, and Stop Separately (25/10)

- Rodrigo's question: Does navigation fail, arm relative motion fail, or both? Separate stopping command from physical stopping and relative fidelity.  
- Intervention: Geometric instrumentation/wheel velocities without changing policies. Repeat 39 rollouts with full orientation/EE/shoulder geometries, since those geometries could not be faithfully reconstructed from old traces. Same configuration; 234 common matrices and identical summaries.  
- Metrics: Arrival XY/yaw at base goal with standoff, error/path/drift; EE local with full rigid transformation against temporal reference; reach rule set at 0.85m without certifying achievability. Terminal velocities and motion separated from zero commands.  
- Result: Base arrival expert 13/13, learned 0/13; median final XY error expert 0.0453m, OLS 1.906m, MLP 563.174m (simulated divergence). RMSE EE relative to base: OLS 0.757m, MLP 0.873m. It is not just the arm failing with correct base; there is also no causal isolation between coupled modules.  
- Offline MLP stopping: when expert sends zero, predicts magnitudes 0.291/0.349 rad/s on each side; additionally evaluate actual terminal motion. Relative curves at 5%/10%/20%/50% and exploratory floors, without new approval criterion.  
- Verification: Two geometric tests, three fidelity tests, and eight evaluator assessments pass. The new report reproduces results without new rollouts; hashes and times are matched and verified.  
- Sources: [Subsystem diagnosis](BASE_ARM_STOPPING_DIAGNOSTICS.md), runs/evaluation/subsystems_validation_20261025 and runs/diagnostics/subsystems_20261025. Post-hoc descriptive analysis, no retroactive change to primary metric.  

### R12 — Base–arm dependence and proposal of two models (25/10)

- Question: Is it valid to separate regressors or does the expert coordinate its commands?  
- Code evidence: base does not query q/qd of arm; arm is activated by nav.done from base and uses global kinematics/memory q_des. Timeout also activates done without guaranteeing arrival. There is logical coordination and physical coupling.  
- Consequence: proposal of 2+7 outputs with explicit coordination/memory; separation alone does not resolve expert observability. Not implemented nor authorized as training yet. Any expert phase during deployment will be labeled as hybrid/oracle variant, not pure baseline.  
- Future control: same data/inputs and computational budget; evaluate subsystems and task, without automatically attributing improvement to modularity. [Detailed audit](BASE_ARM_STOPPING_DIAGNOSTICS.md).

### R13 — Local imitation metric versus achievable target (25/10)

- Observation by Rodrigo: with base error, another local arm movement may be correct to compensate; outside the physical workspace, the arm alone cannot solve the task.  
- Correction: RMSE local vs expert from R11 measures temporal imitation, not isolated arm error. Keep the numbers; do not interpret them as causal evidence of failure in both modules.  
- Methodological decision: error at target transformed with the current base state (norm identical to global error), explicit coverage by conditioning on range, base goals as reference strategy, and global success across all cases.  
- Limit: 0.85m is a proxy for controller, not a physical maximum boundary nor a guarantee. Verified kinematic reachability and arm experiments with a common fixed base both remain pending. Do not run or announce these experiments as complete.  
- Evidence: framework formulation and metric review in [base–arm diagnosis](BASE_ARM_STOPPING_DIAGNOSTICS.md). No new training, simulation, or metric substitution from historical data.

### R14 — Ongoing evaluation of base/system and onset of divergence (25/10)

- Rodrigo's decision: remove isolated arm score, keep base and global task; implement comparative graphs and identify first discrepancy. Do not assume that a better base guarantees a better system.  
- Implementation: v2 report without arm score, global base and end-effector (EE) trajectory in XY/yaw/Z, arrival/error/stopping. Dense recording at 1ms up to 2s, then 0.1s; explicit observation and time PRE-step for commands. Weighted RMSE/MAE of trajectories by time, not by number of samples.  
- Instrumental repetition: 39 rollouts, same 13 scenarios, 16 workers, frozen models. The previous 390 common matrices and original summary are identical. No new scenarios or retraining.  
- Finding: wheel commands differ from t0 in 13/13 cases with same input. Median XY crossover at 1mm: OLS 24ms, MLP 115ms; at 1cm: 77ms / 313ms. First case id2: expert (+3, -3), OLS (+5.273, +3.821), MLP (+2.182, -2.206) rad/s applied. Do not describe failure solely as late accumulation or assert a single cause.  
- Complementary observation: in id2 MLP, global EE separation at 1cm occurs at 154ms, before XY base separation at 1cm (413ms). This is not an isolated arm metric nor a causal test; orientation, height, and dynamics also play a role.  
- Artifacts: 40 PNG / 27 PDF, all covering the 13 cases and summary; [index](TRAJECTORY_FIGURES.md), [method/results](BASE_AND_SYSTEM_TRAJECTORIES.md). Source: runs/evaluation/trajectory_validation_20261025 and runs/diagnostics/trajectory_validation_20261025.  
- Verification: 4 metric/report temporal tests plus 8 evaluator checks; real integration, hashes, PRE-step, and equivalence audit. Post-hoc descriptive crossing thresholds; resolution 1ms at start / 100ms afterward. The curves do not prove causality nor general stability.  

## What can be stated today in the report

1. A reproducible pipeline was built that preserves the controller and separates episodes before preprocessing.  
2. In this development holdout, a nonlinear MLP reduced offline error compared to OLS; the improvement was not uniform across all outputs.  
3. Reducing four wheel values to two preserved expert identity; the isolated effect on learning was not measured.  
4. Original stop labels and geometric arrival labels are not equivalent.  
5. Local costs and autonomous control were measured: neither learned approach met the validation criterion. Diagnosis, robustness, out-of-box/generalization, and multiple seeds are missing.

Do not yet assert that 64 episodes are sufficient, that MLP is the best model, that the MLP achieves targets, or that memory/sensors improve. Do not use the inspected holdout as the final test or repeated batches as independent data.

## Next priorities according to plan dependencies

1. Use millisecond-by-millisecond diagnosis and already-measured initial disagreement to design controllers that distinguish inputs/scales, training distribution, and control sensitivity; the deviation moment alone does not prove causality.  
2. Test small, declared interventions on that diagnosis, maintaining reference scenarios and metrics; establish an autonomous baseline before increasing complexity.  
3. Design new data/suites reserved for testing and study quantity/coverage; keep the confirmatory test isolated from its creation.  
4. Repeat final candidates with multiple seeds, then perform input/memory ablations based on evidence. Explicitly adjust the remaining scope if needed.

These are proposed priorities, not executed work nor authorization to consider the next phases complete.

## How to maintain this record

After completing each experiment or substantial decision, add an entry with:  
date, question/hypothesis, intervention, controls, data/split, seeds,  
configuration, primary metric, selection criterion, positive and negative results,  
limits, local evidence, and consequence for the next step.  
Explicitly mark if the question or metric was defined after seeing results.  
Do not retrospectively reframe a failed hypothesis as if it had never been tested.

`RESEARCH_PLAN.md` describes the design; this record describes what was executed;  
`AGENTS.md` preserves operational agreements. JSON/CSV files are primary evidence of the results. The `runs/` directories are ignored by Git: links to their artifacts require this machine or an explicit copy; they have not been published. Heavy models and data are not uploaded to the repository. Before delivery, preserve the necessary artifacts outside the ignored tree via a separate investigation file and generate final figures from those data.

For the six IEEE pages, prioritize problem, methodology, and ML comparisons; worker tests can be briefly described unless there is specific scientific relevance. There is no obligation to incorporate all proposals from the plan into a single submission. Maintain the contributions page with verified tasks from each participant; do not infer authorship from commit count or attribute experiments to teammates who did not perform them.

## R15 — Saturation and orientation (25/10/2026)

Question: When do clipping events begin and when does the turning error occur? Post-hoc analysis on the same 39 frozen rollouts; no new independent episodes. Added PRE-step clipping initiation per channel/group, separate limits reached, original full-step fractions, signed heading figures, reconstructed yaw, and arrows over XY. OLS clips wheels from t0 at 10/13; MLP median clipping times for wheels at 0.526s and torques at 0.350s. Case 2: torques MLP at 0.347s, wheels at 0.552s; separation already appears earlier. This rules out the 25s timeout as the sole explanation for the onset, without isolating a dynamic cause. Start times are dispersed and unwrapping has explicit limitations. Five trajectory tests pass. Without modifying the expert, models, or adding a supervisor. P2 continues the diagnosis.  
Methodology, figures, and evidence: [saturation and orientation](SATURATION_AND_ORIENTATION.md).

## R16 — First limitation and offline ranges (25/10/2026)

Rodrigo requests documenting command limitations and verifying whether training predictions reach unmanageable orders. Frozen audit over all 490730 train rows and 127147 val rows, hashes, and shared loader. Train labels: absolute maxes wheels 3 rad/s, torques 30 Nm; OLS 6.309/49.831; MLP 3.926/37.304. No unmanageable offline amplitudes exist, but moderate overflows do: bounded labels do not force bounded predictions. Distinguish between small error/excess at startup and subsequent extreme amplitudes in autonomous control. Coverage and feedback are compatible explanations, not an isolated proven cause. Documented as L1 in [limitations](BASELINE_LIMITATIONS.md), includes val, frequencies, and evidence `runs/diagnostics/offline_output_ranges_20261025`. No retraining or simulation repetition performed. P2 active.

## R17 — Closure of exploratory P2 and meaning of feedback (25/10/2026)

In response to Rodrigo's proposal, P2 is closed within pilot scope: data, split, OLS/MLP (OLS was selected as the linear baseline instead of Ridge), MLP optimization history, first autonomous tests, and descriptive failure classification. Data quantity curves are not yet produced and are transferred to P3 based on its scope. This does not imply stability, proven cause, confirmed generalization, or final testing. Uncertainties regarding coverage and memory persist, and cross-seed validation remains unconfirmed.

The autonomous policy already receives the current physical state at each step: this is already state feedback. Adding prior history/actions constitutes an input change (P3/P4); collecting expert corrections on states visited by the policy and retraining is a DAgger-type aggregation (P6). This second path is proposed for discussion as the next improvement, without committing to either change or implementing either yet. Before adding labels, audit one expert instance with updated memory once per step and its recovery capability. Maintain validation outside of data collection, comparable models/inputs, and equal budget for new data. The decision on which meaning Rodrigo intends remains pending.

## R18 — Questions for tutoring (25/10/2026)

At Rodrigo's request, §17.1 of the plan is added: which limits to preserve in simulation, whether they are physical or design-based, external clipping versus bounded regressor, and the academic meaning of the 0.85 m reach rule, standoff and 25 s timeout. Check whether fixed rules are imitated or varied, with which inputs, and whether time-based supervision is allowed. No response yet; do not modify the expert/model or attribute causality to clipping. Exact questions and context in [plan, questions for the lecturer](RESEARCH_PLAN.md#171-questions-for-the-lecturer-command-limits-and-expert-rules).

## R19 — English-language collaboration (25 October 2026)

Rodrigo requested English throughout the maintained repository so all three
teammates can collaborate. Translated the project guides, research plan, research
log, decisions and lecturer questions; renamed the Markdown documents in English
and updated their links and heading anchors. Source comments, docstrings, CLI
messages and configuration descriptions were checked; most were already English.
Translated plot labels and output filenames and exported the current 13 paired
scenarios into `runs/diagnostics/trajectories_english_20261025` (66 PNG, 53 PDF).
No simulation, model training, control rules, numerical results or dataset splits
were changed. Historical experiment artifacts, source snapshots and the lecturer's
original PDF remain unmodified as provenance; the new export presents existing
results in English. Language tools were temporary and added no project dependency.
Original maintained files and translation checks are retained locally under
`runs/documentation_english_20261025`. Five trajectory tests pass; links, figures
and source differences were checked. P2 remains closed as an exploratory pilot;
there is no new experiment or evidence of improved control.

## R20–R24 — Superseded exploratory records

The exact prior records are preserved in the ignored documentation snapshot at
`runs/archives/pre_power_of_two_correction_20261026/`. The user corrected the
active episode-count plan; obsolete cohort results were removed from maintained
reports rather than relabelled as measurements of a different size. Current
uniform references at 64/128/512 remain valid. At this correction no 256 result
existed; the later R27 completion supplies a newly collected/trained cohort.

## R25 — Corrected counts, stratified pilot and training-stop audit (26 October 2026)

Rodrigo confirms the intended series 64/128/256/512 and approves spatially
stratified target and validation selection. Begin with the previously proposed
64-episode comparison: 4x4x4 cells, one jittered target in each; 51 training and
13 spatially balanced validation targets selected from geometry before simulation.
Keep box, expert, nominal start, OLS/MLP, limits and evaluation protocol fixed.
Compare both designs on both held-out banks, rejecting all training-condition
leakage; explicit external-bank opt-in added without weakening defaults.

The training code stops after 8 epochs without a validation-NRMSE improvement
of at least 1e-4, or at the epoch cap, retaining the lowest-validation checkpoint.
MLP512 reached the cap at 60 epochs, best 58; this is not evidence of convergence.
MLP64 and MLP128 stopped by patience at 22/17, best 21/9. Sampling comparison keeps
the original budget; no extended-budget run has been performed. Two new geometry/
leakage tests, two nested-split tests and eight evaluator tests pass.
Protocol and pending results: [sampling comparison](SAMPLING_COMPARISON.md).


### R25 results — Stratified64 pilot completed

Collected64 complete episodes,602,889 aligned records, no execution failures;
original labels62 success/2 stalled retained. Stratified fine-cell occupancy64/64
versus41/64 uniform; training51 cells versus38, validation8 versus6 octants.
Physical mean nearest-training-goal distance essentially unchanged33.58/33.54cm.
Heldouts chosen solely from cell geometry before simulation, not outcomes.

Unchanged OLS and MLP trained; stratified MLP stopped by patience at11, selected3.
All130 paired autonomous rollouts valid,40s each; four learned policies plus expert
on both13-case banks. Old uniform/expert physical metrics on original bank match
exactly. Common26 MLP NRMSE uniform0.27955 / stratified0.29300 (no offline gain);
median final base error203.563 /2.387m, EE899.748 /2.707m (less divergence).
Both MLPs still0/26 task/base arrivals; expert26/26 task. No stable controller.
Different selected checkpoints (uniform21/stratified3), geometry, durations and
one-seed variability remain confounded under the same early-stopping recipe.
No extended512 training, memory window or corrected256 data claimed.

Current guides and learning-curve specification use64/128/256/512. Superseded
cohort documentation is archived exactly under ignored runs/archives/; numerical
results were never relabelled. Shared reporting derives sizes from artifacts
rather than hardcoding an obsolete count. Validation and source checks,17 tests
(2 geometry/leakage,2 nested,8 evaluator,5 trajectory) and local-link checks pass.
Evidence: runs/sampling_comparison/stratified64_20261026; report and four PNG/PDF
figure families in [sampling comparison](SAMPLING_COMPARISON.md).

## R26 — DAgger labeling limitation (26 October 2026)

- Question: does DAgger remain supervised, and what prevents direct application
  to the current expert? Rodrigo requests recording the limitation.
- Decision: document DAgger as a proposed supervised imitation extension whose
  stateful expert labeling protocol remains unresolved. Expert navigation phases,
  `nav.done`, and integrated `q_des` require a consistent causal history; resetting
  the expert for every query is not equivalent to the original controller.
- Required controls before an experiment: define expert memory updates once per
  physical step along learner rollouts, audit recovery labels, preserve expert
  rules, exclude held-out scenarios from aggregation, and compare labeling
  budgets. Additional policy memory/inputs must be declared separately.
- Academic scope: expert-supervised updates are preserved, but lecturer approval
  of the extended collection protocol has not been confirmed by the slides.
- Status: documentation only; no collection, split, seeds, training, selection,
  or metrics changed. No DAgger performance or recovery result is claimed.
- Detail and sources: [baseline limitations, L2](BASELINE_LIMITATIONS.md#l2--stateful-expert-labeling-is-unresolved-for-dagger).


## R27 — Full sampling-design series started (26 October 2026)

Rodrigo explicitly chooses64/128/256/512. Extend the completed pilot under the
original model/physical/training recipe, compare both designs on complete native
banks and a fixed26-condition anchor. Reuse compatible existing results, create
actual uniform256 and new stratified128/256/512, preserve all provenance and cap
limitations. Larger geometry is independent, spatial holdouts balanced by binary
constraints plus normalized-cell maximin selection before jitter/outcomes.
Detailed protocol: [full sampling series](SAMPLING_SERIES.md). Results pending.


### R27 results — Complete corrected sampling series

The campaign is complete; [full measured report](SAMPLING_SERIES.md) records
all64/128/256/512 cohorts, full native unions and fixed26-anchor outcomes.
Actual uniform256/204train/52val was collected and trained; no relabelled data.
New stratified128/256/512 grids use geometry-only balanced holdouts and preserve
the pilot. Complete model/trace provenance and repeat integrity checks passed.
Native banks differ across sizes, larger stratified targets are independent,
and one realization cannot establish causal/statistical superiority.

- 64: anchor NRMSE uniform0.279545/stratified0.293001; task arrivals0/26 versus0/26; median base XY203.5631/2.3867m.
- 128: anchor NRMSE uniform0.258103/stratified0.202679; task arrivals0/26 versus0/26; median base XY84.2155/7.8624m.
- 256: anchor NRMSE uniform0.159855/stratified0.149597; task arrivals0/26 versus0/26; median base XY3.5482/2.9396m.
- 512: anchor NRMSE uniform0.098639/stratified0.114251; task arrivals0/26 versus0/26; median base XY204.6487/10.6650m.

All original training caps/patience remain; exact stop reasons and best epochs
are recorded in TRAINING_STOPPING.md. No memory/DAgger/limits or longer-budget
intervention. Artifacts: runs/sampling_comparison/full_series_20261026, including
per-episode/channel CSV, per-case paired outcomes, target clouds, coverage and
eight scientific PNG/PDF figure families plus self-contained3D HTML.

## R28 — Patience-only budget and 1024/2048 extension (26 October 2026)

Rodrigo authorizes both larger sizes for both designs and enough epochs to stop
by patience. Remove the cap in a separate recipe, retain patience8/min_delta1e-4,
reuse seven compatible patience-stopped MLPs, and rerun capped uniform512 with
exact frozen normalization. New larger collections retain80/20 episode splits,
unchanged physics and features, both full native banks and fixed26 anchor.
Memory-bounded fitting/offline evaluation and exact Adam checkpoint recovery are
implemented; synthetic recovery and streaming metric equivalence tests pass.
Protocol: [patience extension](PATIENCE_SERIES.md). Results pending.

## R29 — PyTorch/CUDA implementation (26 October 2026)

Rodrigo authorizes implementing the MLP in PyTorch to use the RTX4060 Laptop GPU.
The shared trainer now supports optional `backend: pytorch`, with unchanged
architecture, features, split/normalization and patience. Installed torch2.11.0
with CUDA12.8 in an ignored ML overlay; the simulator's base environment is
untouched. CPU training finished before moving its remaining comparisons to a
frozen source tree, preserving the original campaign while development proceeds.

The installed sklearn code exposes two details requiring explicit porting:
output-summed backpropagation despite output-mean reported loss, and Adam's
uncorrected-moment epsilon convention. Both are preserved and tested. GPU kernels
remain a numerical variant, not bit-identical training. Auto VRAM residency,
streaming fallback and full optimizer/RNG checkpoints are implemented. Final
models retain NumPy inference in the simulator, without a Torch dependency.

All47 tests pass, including actual CUDA, sklearn first-update comparison,
CPU/CUDA exact optimizer recovery and previous regressions. Real uniform64
GPU training and simulator integration are being checked. No full GPU learning
curve or control-stability result is claimed. See [PyTorch training](PYTORCH_TRAINING.md).


### R29 results — CUDA pilot and simulator compatibility verified

The real uniform64 CUDA pilot completed54 epochs by patience, selecting46.
Native13 balanced NRMSE0.182987,20,489 parameters, median2.227s/epoch,
122.11s total command time and148.64MiB peak Torch tensor allocation.
Original exact training-only normalizers and51/13 membership were reused.
The maximum preview Torch/NumPy physical difference was3.32e-6 across mixed units.

A paired held-out40s expert/model integration ran in the default RoboEnv Python
without Torch. Both rollouts valid; expert1/1 task, model0/1 and110.9338m final
EE error (simulated divergence). Port/export is working; no stable-control claim.
The historical sklearn64 median1.418s/epoch is lower than this observed CUDA
pilot timing, under different workload; no GPU speedup is established. Preserve
batch1024/FP32 and measure throughput separately before changing those choices.
Full method, checkpoints and artifact paths: [PyTorch training](PYTORCH_TRAINING.md).

## R30 — Partial patience-series report (26 October 2026)

Rodrigo requests a partial report while online2048 is still running. Freeze the
completed-bank cutoff, not a convenient subset of already-finished goals. All
six-size offline comparisons are available; complete online512 and1024 include
both native banks plus the same26 anchor. PyTorch pilot remains separate.

- Controlled uniform512: exact60-epoch prefix, extended stop79/best71. Anchor
  NRMSE0.098639→0.098067 (-0.58%), base median204.6487→233.6493m and EE
  median441.4852→746.9644m, task0/26 both. More epochs did not rescue this run.
- Complete1024 native union: expert410/410 task1cm, both MLPs0/410. Median
  baseXY235.6947m uniform /16.1002m stratified; all410 per policy valid.
- Offline anchor2048 NRMSE0.081433uniform /0.095009stratified. From1024,
  improvement10.62% /0.84%, respectively; one realization, not an optimum claim.
- Zero-wheel expert-state imitation improves: uniform512 mean abs0.08968/0.09065
  rad/s right/left →2048 0.00752/0.01307. Nonzero strict5% remains uneven and
  joint-level errors are retained. No stable-control claim from offline errors.
- Report: docs/PATIENCE_PARTIAL_REPORT.md. Immutable snapshot under
  runs/sampling_comparison/patience_series_20261026/partial_reports/20261026T232054Z;
  cutoff2872/4206 new online jobs, six complete banks. Seven PNG/PDF figure
  families, conditional5% tails, per-case/output CSVs and104 valid paired anchor
  trace diagnostics; repeated expert anchor checks and all report links pass.
  No in-progress2048 bank is aggregated. Full finalization continues independently.

## R31 — Supervised errors by base, arm and joint (27 October 2026)

Rodrigo requests individual error learning curves instead of only global NRMSE.
Use the same26 held-out expert-state goals and equal episode weighting, with
physical wheel-command RMSE(rad/s), mean per-joint torque RMSE(Nm), each of the
seven joints separately, and a fixed original64 training-mean predictor baseline.
No retraining, simulation, split, model, physics or stopping changes are needed.

For1024→2048: uniform wheels0.124279→0.097845rad/s (-21.27%), arm
0.191745→0.186289Nm (-2.85%); stratified wheels0.128841→0.127135rad/s
(-1.32%), arm0.203759→0.202357Nm (-0.69%). Some individual stratified joints
worsen:1,2 and7. Uniform joint7 improves28.51%, but only0.00723Nm absolute;
stratified joint7 worsens14.08%, an absolute0.00480Nm. A single global error
conceals differing gains and does not establish one optimal dataset size.

Reusable launcher scripts/report_supervised_scaling.py and shared reporting
module learning/error_scaling.py produce five PNG/PDF figure families,24group
rows,108channel rows and110incremental comparisons. Real-data assertions verify
model/split checksums, exact group reconstruction and fixed-scale NRMSE; figures
were inspected. Measured tables and scope: docs/SUPERVISED_ERROR_BY_SUBSYSTEM.md.
No isolated arm pose metric or online stability claim is introduced.

## R32 — Full validation versus fixed26 anchor (27 October 2026)

Rodrigo checks whether R31 includes every validation episode. It did not: R31 was
explicitly a fixed26 anchor. Extend the same reporter with own_validation,
native_union and anchor26 views; default now uses the full paired union. Counts
per model are13/26/52/103/205/410 for own validation,26/52/104/206/410/820
for paired union, and26 for the anchor. Store coverage in every row/figure.

The broader data change the interpretation. For1024→2048 own validation, uniform
base/arm RMSE improve16.74%/5.49%; stratified improves1.01% for base but worsens
5.66% for arm. On the full paired union, uniform worsens12.52%/16.34% for
base/arm; stratified base improves0.69%, arm worsens1.25%. The targets change
across sizes, so these differences do not isolate sample-count effects or prove
overfitting. Anchor-only conclusions cannot be generalized to full validation.

Full per-joint curves/tables and revised interpretations:
docs/SUPERVISED_FULL_VALIDATION.md. Original anchor report remains with explicit
coverage note. Own physical metrics match every training run's full validation;
anchor reproduction is exact; bank sizes, model/split hashes and union identities
pass. Reference-mean baselines use the actual selected bank. No new simulation,
training, model, expert or metric definition is introduced.

## R33 — Holdout geometry hypothesis checked (27 October 2026)

Rodrigo asks whether stratified own validation is easier because it was selected
near training. Source audit confirms geometry-only balancing/maximin between
holdout cell centers, octant coverage and retained training in vertical pairs;
no outcome-based or observed train-distance selection. The same domain supports
interpolation, not a guarantee about extrapolation.

Actual nearest-own-training target means at2048:0.09635m uniform /0.10534m
stratified; box-normalized0.04801/.05485. Stratified is not closer on average
in this measured case. At1024 its physical P95 is smaller despite a larger mean,
so mean proximity alone cannot rank bank difficulty. No exact target duplicates.
XYZ geometry is not21D state/history similarity and does not identify a cause.

Also correct the premise: stratified does not always have lower own-validation
supervised error; uniform arm RMSE is lower at every size, and wheel RMSE lower
except128. Lower stratified online divergence is a separate observation.
Results, code rule and provenance: docs/VALIDATION_GEOMETRY.md and
runs/diagnostics/validation_geometry_20261027. No data, models or splits changed.


### R28 results — Completed patience-only extension

Four new1024/2048 collections, five newMLPs and seven reused patience-stopped
MLPs are complete. Uniform512 exactly reproduces60 historical epochs before
extending. Complete native and fixed26 matched evaluations, command tails,
trajectory/orientation/clipping diagnostics and spatial plots are recorded in
[PATIENCE_SERIES.md](PATIENCE_SERIES.md). The epoch cap and additional data are
evaluated separately; no policy architecture or expert change was introduced.

## R34 — Order of memory, architecture and DAgger proposed (27 October 2026)

Rodrigo asks whether to minimize supervised error through architecture tuning
before windows/DAgger, and whether1024 is the approximate data compromise.
Recommend1024total819/205 as a practical fixed budget, not an established optimum.
First create a comparable Torch1024 baseline (only64 has a Torch pilot), then a
simple100ms/six-snapshot window with an instantaneous parameter-count control,
then a small architecture/regularization screen, and DAgger as a later distinct
intervention after the stateful-labeling protocol is resolved. Keep all full
validation conditions held out and evaluate offline plus autonomous behavior.

The CPU campaign has finished: all12 MLPs stopped by patience; both2048 policies
remain0/820 primary task arrivals. This supports moving beyond exhaustive static
supervised tuning without proving which intervention will work. Recurrent memory
and DAgger are hypotheses, not guaranteed remedies. Proposal and sources:
docs/NEXT_EXPERIMENT_ORDER.md. No new experiment or architecture was launched.

## R35 — Literature-backed control metric proposal (27 October 2026)

Rodrigo requests online documentation for trajectory fidelity, spinning/runaway
behavior and saturation diagnostics. Audit existing evaluator before adding
scores: time-weighted baseXY/yaw/EE separation, departure times, full-step path
length and clipping fraction are already present. Recommend discrete Frechet
ordered route similarity plus expert-referenced path ratio, with accumulated
rotation and normalized raw/applied clipping mismatch as extra diagnostics.
Relative Pose Error, DTW/nDTW and IAE/ITAE/TV alternatives are documented.

Sources: Eiter/Mannila1994, CGAL, Sturm2012, Anderson2018, Ilharco2019 and original
control-performance/saturation studies. Adaptations are labelled; expert length
ratio is not SPL, clipping mismatch is not energy, and no geometric alignment
may conceal errors in the shared simulation frame. Exact rotation and mismatch
counters need future full-step observer instrumentation; old100ms tails provide
only sampled estimates/lower bounds. Task success/dwell remains primary.

A separate six-pair exploratory prototype uses the three previously geometry-
selected uniform anchor cases at1024, checks hashes and four analytic Frechet
examples, and evaluates sampled Frechet plus full-step length ratios. One case
has ratios276.39uniform/9.93stratified, both failures. Historical severity is
explicitly sampled/time-weighted, rotation a sampledSO3 lower bound. This is not
full-validation ranking or production integration; no simulation/expert/model
changed. Detailed definitions, units, limitations, bibliography and pending
implementation: docs/CONTROL_METRIC_PROPOSAL.md; ignored examples under
runs/diagnostics/control_metric_research_20261027.

## R36 — Robust full-bank route distributions and difficulty maps (27 October 2026)

Rodrigo requests that a few runaway trajectories not dominate the summary, and
asks whether failures follow a recognizable pattern. Implement per-episode
discrete Frechet and expert path ratios with median/Q1/Q3, mean, P90/P95/max,
missing/invalid counts and CDFs; preserve every failure without capping/removal.
Postprocess all410/820 paired native holdout goals at1024/2048 for both CPU
policies (2460pairs), without further simulations or model/expert changes.

Stratified1024 Frechet mean76.62m versus median18.26m; the two largest cases
contribute8.42% of the sum. Robust summaries matter, but large median error and
zero task successes show failure is not restricted to two outliers. Fixed
geometry bins (radius, height, signed initial navigation direction) and full
interactive3D maps expose tentative1024 differences that do not replicate
clearly at2048; associations remain exploratory and one training seed.

Distinguish XY from yaw departure: stratified1024 median10cm XY departure2.6s
versus5deg yaw0.781s. Correlation of initial turn with XY departure is0.690,
but only0.035 with first XY-or-yaw departure. Longer XY agreement alone cannot
establish better control while a robot is turning. Historical sampling limits
remain explicit; no exact full-step rotation/clipping-severity claim is added.

Three targeted tests pass. Verified full coverage, hashes, all summary medians,
subgroup counts and six independent prototype comparisons. Corrected an initial
aggregate count/dataset-size field collision before reporting; that draft is
archived as superseded. Docs: ROBUST_CONTROL_PATTERNS.md. Results and snapshots:
runs/diagnostics/robust_control_patterns_20261027. Production per-step observer
extensions, new control interventions and multi-seed confirmation remain pending.

## R37 — Temporal-window phase authorized and launched (27 October 2026)

Rodrigo selects1024 episodes and authorizes starting input-history experiments.
Implement six18D measured states at100/80/60/40/20/0ms lags plus the current3D
goal once (111inputs), causal initial-state padding, dense1ms sampling checks,
per-episode boundaries/reset and shared frozen current-state normalization.
Gather overlapping windows per minibatch instead of materializing the entire
expanded dataset. Portable NumPy inference includes the bounded history ring;
historical instantaneous artifacts retain their format and behavior.

Launch a frozen-source PyTorch/CUDA comparison on the existing stratified1024
819/205 split: instant128(20489parameters), window100ms(32009) and instant164
capacity control(32153). Same seed20261025, Adam/settings, patience8/min_delta1e-4,
no cap, software envelope and expert/physics. Train sequentially on the GPU;
then evaluate all205 own and205 uniform external holdouts for all three policies
and expert,16physics workers. Final robust CDFs retain every episode. No new
collection, LSTM, DAgger, regularization or output-bounding change.

All53 repository tests passed; new temporal tests cover causal/episode isolation,
offline/live and CPU/CUDA gather agreement, normalization, sparse-log rejection,
export/reset, resident/stream equivalence and exact resume. Compilation passes.
Campaign: runs/temporal_comparison/stratified1024_window100ms_20261027; coordinator
log /tmp/temporal1024_20261027.log. At launch results are pending, not evidence of
improved control. See docs/TEMPORAL_WINDOWS.md for semantics and monitoring.

Additional deployment smoke: actual CUDA temporal training with synthetic labels
and2501ms calls in real physics; all raw predictions match offline causal
window inference exactly. Purpose is integration only, not scientific control
performance. Evidence: runs/diagnostics/window_cuda_rollout_smoke_20261027.

## R38 — Window duration and spacing still require discussion (27 October 2026)

Rodrigo clarifies that these design parameters must be defined together before
running the temporal experiment. R37's launch prematurely treated the100ms
duration /20ms spacing proposal as selected. Stop the campaign process group
during instant128 training; no temporal model or autonomous comparison started.
Preserve optimizer checkpoints and source snapshots; mark the campaign interrupted
with this reason. Code remains available and tested, but the scientific design is
pending discussion. Separate history duration, spacing, snapshot count and1ms
physical control period. Do not relaunch with assumed values.

## R39 — Explicit200ms/20ms selection and relaunch (27 October 2026)

Rodrigo corrects the final window duration to200ms, with20ms spacing. Eleven18D
states plus current3D goal =201inputs. Window2×128 has43529parameters; update the
instantaneous capacity control to2×193(43434), retaining instant128(20489).
Launcher now requires explicit duration/spacing. New frozen-source campaign:
runs/temporal_comparison/stratified1024_window200ms_20261027,
log /tmp/temporal1024_window200ms_20261027.log. Old100ms artifacts remain interrupted.

Resume instant128 from verified epoch21 checkpoint; all seven training-module
hashes match. Actual CUDA recovery completed epoch22. Same819/205 split/scalers,
seed, patience and410goal evaluation protocol. Four temporal tests and source
compilation pass. Previous53-test suite and100ms integration smoke remain
historical checks; no completed200ms experiment or improvement is claimed.

## R40 — Partial paired temporal control report (27 October 2026)

Rodrigo requests interim closed-loop behavior. Snapshot20261027T035936Z retains
154 own-validation goals completed by expert and all three policies, with
record/trace hashes. Do not compare unmatched finished subsets or imply410-goal
coverage. Completion counts expert154/instant128155/window156/instant193155;
all matched results valid. Snapshot path:
runs/temporal_comparison/stratified1024_window200ms_20261027/partial_reports/20261027T035936Z.

Expert154/154 primary success; each learned policy0/154. Median timed XY RMSE:
instant12815.0167m/window200ms203.2422m/instant19346.1300m. Median final base-goal
distance43.7745/494.2032/108.5823m; median full-step wheel clipping fractions
0.7242/0.8439/0.8638. Window currently diverges more despite lower supervised
validation NRMSE. These are simulator divergence diagnostics, not physically
credible robot travel. One development seed and incomplete geometry-ordered
bank; await all410 goals. No causal explanation or generic rejection of temporal
information follows; continue the agreed campaign and preserve negative cases.

## R41 — Hidden-depth and expansion/contraction screen launched (27 October 2026)

Rodrigo requests more hidden layers and a32→64→128 progression, suggesting an
encoder/decoder-like shape. Add configurable positive hidden-width lists to
Torch construction, NumPy validation/export/inference and sklearn recovery,
preserving ReLU hidden activations and linear nine-command outputs. Retain the
supervised command loss; do not turn this into input reconstruction.

Predeclare3hidden32,64,128 and5hidden32,64,128,64,32, each with instantaneous21D
inputs and the agreed201D200ms/20ms stack. Four new fits. Parameter counts are
12297/18057 for increasing and21769/27529 for expand-contract (21D/201D).
Reuse Torch instant128/window200ms reference models, original stratified1024
819/205 split, exact scalers, seed20261025 and all Adam/patience settings; no cap.
This changes depth/shape and parameter budgets, not an isolated depth treatment.

All57 tests pass (183.112s under concurrent reference evaluation), including
deep actual CPU/CUDA training, multi-layer export, negative linear output,
window live/batch agreement and exact sklearn/Torch resume. Frozen old and new
inference match exactly on600 real holdout observations for each reference.
Compilation passes. Campaign launched from a frozen code copy:
runs/architecture_comparison/stratified1024_depth_shape_20261027;
coordinator log /tmp/mlp_depth_shape_20261027.log. First variant instant_pyramid.

GPU fits are sequential. Before new physics jobs, wait for the original temporal
campaign to complete; keep16simulation workers. Evaluate all410 paired goals per
new model and reuse old baseline trajectories in the final six-policy robust
report. Preserve all failures; new depth/shape results remain pending. Protocol:
docs/MLP_DEPTH_AND_SHAPE.md. No dropout/BatchNorm, bounds, sensor or expert changes.
