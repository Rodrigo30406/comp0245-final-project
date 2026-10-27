# Project Guide and Agreed Decisions

## Repository language

All maintained documentation, code comments, docstrings, CLI messages, plot labels
and new filenames must be in English so all three teammates can collaborate.
Conversation with Rodrigo may remain in Spanish. Preserve the lecturer's source
material and frozen experiment artifacts as provenance; export new English copies
of presentation artifacts instead of rewriting historical results or snapshots.

Last updated: 27/10/2026. Scope: the entire repository.

This file preserves Rodrigo's decisions and instructions for continuing the project. Update it whenever a decision affecting architecture, expert, configuration, data, or evaluation is agreed upon. Record date, reason, and consequences; replace superseded decisions by indicating what changed. Distinguish agreements, pending proposals, and verified results. User instructions that follow have priority.

## Academic Objective and Sources

- Final project COMP0245, Machine Learning for Robotics, MSc Robotics and AI.
- The assignment specification is based on the [slides](docs/COMP0245-Assignment-2026-2.pdf).  
  Consult them before asserting requirements or altering the academic scope.
- The objective is behavioural cloning: train a supervised regressor that reproduces the provided controller and generalizes to new goals and/or initial conditions. No requirement to outperform the expert or reach all points is imposed.
- Inputs of the baseline implementation: base position XYZ and bearing, seven joint positions and seven joint velocities of the arm, and original goal XYZ: 21 values. The dataset retains 11 commands (4 wheels + 7 torques). By Rodrigo's decision, the model learns 9 outputs (right, left, and 7 torques); expand wheels as `[right, left, left, right]`. Verify pair identity upon loading, do not average discrepancies. Do not introduce privileged diagnostics as baseline inputs without declaring a variant.
- Justify the quantity and quality of data, inputs, architecture, training, and methodology. Evaluate accuracy, generalization, robustness, and efficiency. Prioritize reproducibility, experimental controls, and specific discussion of results; do not confuse greater complexity with better methodology.
- The [research plan](docs/RESEARCH_PLAN.md) contains bibliography, experiments, and rubric. It is a work plan, not a list of already achieved results. Maximum report length: six pages IEEE and one additional page for contributions; dates and demonstration are mandatory in the slides.

## Expert Preservation: Primary Decision

- Follow the professor's logic and code rules. It is allowed to move, encapsulate, split, and refactor the original code; do not freeze its files. Preserve behavior for identical state, history, configuration, target, and environment, and verify it with appropriate regressions.  
- Do not silently adjust gains, saturations, navigation geometry, reference integration, phases, physics, or stopping conditions in order to achieve more success. A change in behavior is an explicit experimental variant with provenance and comparison.  
- A single motor, `src/robot_learning/simulation/episode.py`, executes episodes. `simulation/interactive.py` adds presentation, arguments, and graphics; `data/worker.py` adds logging and storage. Do not duplicate the control loop in any of the consumers.  
- Professor's sequence: measure, compute base/arm commands, apply the action, advance physics, update the visualizer, check keyboard input, and evaluate `check_stop` using the measurement from the **previous** step. The next reading is logged, but it does not replace that previous measurement for determining stopping.  
- Preserve accumulation of `dt`, priority of criteria, and internal state. The `max_steps` threshold is checked after applying an action; it may execute one additional step, also due to rounding. Do not add another strict limit to correct it during a refactor.  
- An episode starts with a new simulator and controller memory. Changing objectives sequentially without resetting would be another experimental protocol.

## Experiment Configuration and Per-Episode Inputs

Common source: [experiment_specs/expert/default.json](experiment_specs/expert/default.json).  
Values are loaded and validated via `simulation/config.py`.

| Rule | Current Agreement |
|-----|-------------------|
| Fingers | Six angles of 0.1 rad, constant |
| Arm reach limit | 0.85 m from the shoulder frame used by the expert |
| Arrival tolerance | 0.01 m |
| Base navigation timeout | 25 s simulated from the start of the episode |
| Standoff heights | `[0.1, 0.5, 1.1, 1.25]` m |
| Standoff separations | `[0.8, 0.65, 0.65, 0.35]` m |
| Other gains and limits | Values from the professor |

- "Fixed" means constant across all episodes of an experiment, recorded in configuration; it can be deliberately changed between experiments. Do not randomize these rules per episode. Maintain current standoff and timeout values until another variant is decided to be studied.  
- The target is the input for each episode. Initial conditions may also vary if the experiment specifies this. Do not modify Python code to test points.  
- The recipes in `experiment_specs/collection/` define sampling, quantity, seeds, and logging; they should only repeat common rules if there is a clear reason to freeze them explicitly or to declare an override.  
- Standoff is the desired horizontal separation between the base center and the target based on its height. `np.interp` interpolates within the table and preserves the extremes outside: above 1.25 m it continues using 0.35 m, not zero.  
- The table performs a nominal geometric check relative to 0.85 m, but does not verify achievability with joint limits or navigation errors. That limit is a controller rule, not a certification of the physical workspace. Do not deduce a guaranteed maximum Z by summing heights.  
- The projected auxiliary target onto the shoulder sphere does not replace the original target in inputs or metrics. The expert is not necessarily optimal and does not necessarily find the physically closest pose.  
- Exceeding the 25 s time limit marks navigation completion and allows arm reach; it does not confirm base arrival nor automatically ends the episode.  
  Default global threshold: 40,000 steps, approximately 40 s with a dt of 1 ms; always verify the effective dt. This is not a mandatory duration.  

## Agreed data and methodology

- Proposed pilot: uniform targets in the X/Y box `[-2, 2]` m and Z `[0.3, 0.8]` m, fixed nominal startup, recipe of 64 episodes. The box must be evaluated by the expert before being considered a validated domain. During worker calibration, all 64 scenarios were executed in five equivalent development batches; subsequently, the baseline OLS described below was trained. No final test set is currently reserved.
- Each dataset preserves the requested and effective configuration, scenarios, seeds, schema, code/asset hashes, snapshots, versions, logs, trajectories, and results. Do not overwrite datasets or resume with incompatible code/configuration; instead, start a new dataset or recover the corresponding environment.
- Each transition aligns observation/time → action → state/time after a physical step. Subsampling the log does not alter the control period; `previous_action` corresponds to the physical step prior, not to the stored row. Save terminal state even if it does not align with the log cadence.
- Current schema v2: stop decision based on previous measurement according to the instructor. Save `stop_decision_time_s` and `stop_decision_distance_m` separately from final metrics and the subsequent terminal state. v1 datasets retain their historical semantics and files; do not rewrite them.
- Preserve failures and their causes. `complete` means execution completed, not success. `stalled` does not imply physical impossibility. Any demonstration filtering must be justified and recorded.
- Evaluate both supervised command error and simulation performance: success, original target error, times, and comparison with the expert under the same scenarios. Separate expert failures from failures introduced by learning. BC alone does not provide a signal to correct the expert.
- Split data by episodes/scenarios before windows and normalization; avoid duplicates, dependencies, and future leakage across splits. A collector's `split` label does not automatically guarantee a split without leakage.
- Pending ideas: vary point count and horizontal extent separately; consider density changes when expanding the box; study random initial X/Y starts, then orientation/posture separately.
- Compare snapshots and temporal windows; study additional inputs such as wheel velocities, causal accelerations, or past actions via ablation studies. Fourier methods and advanced architectures are proposed, not implemented. Details in [experiment notes](docs/EXPERIMENT_NOTES.md).

## Parallel Collection: Decision of 24/10/2026

- `--workers N` limits the number of concurrent episode processes; default is 1.  
  Using more than one requires GUI-free mode. Do not parallelize by changing the motor or expert rules: `data/scheduler.py` coordinates independent processes.
- Scenarios and seeds are fixed before launching processes. The order of completion does not determine targets, seeds, or file names. A single coordinator writes summaries, manifest, and execution log; each worker writes only its own episode files.
- Workers is an operational CLI option, separate from scientific configuration. It can be changed upon restart. Each execution or restart records workers, processes, real times, CPU usage, host, and thread variables in `executions/`.
- Limit each worker's numerical libraries to a single thread by environment variables; respect explicitly set terminal values and log them.  
  Do not assume linear performance scaling with increased processes; measure actual performance and resource usage.
- `Ctrl+C` stops and waits for all child processes, preserves completed episodes, and allows restart. A failure in one worker is logged without discarding episodes from others.  
  If code or environment changes during active episodes, invalidate and terminate them.
- Training and evaluation reside within this same project: `learning/` now implements split, normalization, regression, and offline metrics. Autonomous evaluation remains pending.  
  Organize launchers by function if their number justifies it; do not create empty directories or separate launchers per architecture.

## Local worker measurement: results updated on 25/10/2026

- The first attempt was interrupted at 98–100 °C with 1 and 2 workers. Rodrigo turned on the cooler and authorized continuing without the conservative thermal cutoff. Temperature and counter readings were still being recorded; energy and affinity were not modified.  
- The sweep was completed with the same 24 episodes: 2/4/8/12/16/24, followed by a confirmation using the 64 uniform scenarios from the pilot: 16, 24, and 32 workers, repeating 24 and 32 in reverse order.  
- **Initial recommendation among 16/24/32: `--workers 24` with the cooler enabled; replaced by Rodrigo’s decision to use 16, described below.** For 64 episodes, median coordinator time: 16 → 51.4 s, 24 → 45.0 s, 32 → 45.5 s.  
  Average CPU: approximately 49%/67%/85%; maximum RSS: 3.9/5.75/7.69 GiB.  
  With 32 workers, there is no improvement in time and CPU spikes reach 100%. With 16 workers, there is more headroom for other tasks, at the cost of approximately 14% more time.  
- Practical rule: test the smallest number within 10% of the fastest time. This is a limited local calibration; do not claim a universal or exhaustive optimum. The CLI default remains 1 for portability.  
- Temperature continued reaching 100 °C; do not assert that the cooler eliminated throttling. Operation continued as explicitly instructed by the user.  
- The five batches of 64 scenarios share configuration/hashes/scenario schema and results: 617,877 transitions and 62 successes. Keep them separate; do not concatenate replica runs as independent data or cross them between splits.  
  If reusing a batch, explicitly select a development copy.  
- Evidence, criteria, and limitations are detailed in [worker measurement](docs/WORKER_BENCHMARKS.md).  
  Logs, CSV files, measurement code, and datasets are located under `runs/worker_scaling*`, and are ignored by Git. The expert rules were not modified during testing.

## Performance profile: measurement from 25/10/2026

- The queue is already dynamic: as a slot becomes free, a new episode is launched without waiting for the full group. A new process is created per episode; do not confuse reusing a slot with maintaining a persistent process/simulator.
- External profiling of the same 64 scenarios with 24/32 workers, without changing production code: physics accounts for ~57% of the total internal accumulated worker time; arm control ~9%, logging ~8%, setup ~8%, observation/FK ~7%, compression/saving ~3%. Additional imports: 0.85/1.08 s per process.
- The coordinator performs 128 full checks: 13.2/17.3 s accumulated, overlapping with simulation. Do not promise complete time savings from optimization. Summaries/writes cost less than one second per group batch.
- Restart after detecting exit: average 0.25/0.55 s (24/32). 26 out of 40 replacements with 24 were launched before the final initial episode ended. There is no global barrier; processing of results and checks does introduce latency.
- With 32 workers, the accumulated time per episode increases by ~34% for identical work and offsets the 33% additional slots. Do not attribute causal percentages to frequency, SMT, cache, or temperature without additional measurements.
- The final queue of about 10 s with 24 workers occurs without pending episodes: no additional work from the same batch can be added. Do not confuse this queue with waiting for all to complete before launching a new group.
- Diagnostics and limits in [collection profiling](docs/COLLECTION_PROFILING.md). The observational scripts and results are located in `runs/collection_profile_20261025/`. No optimizations were implemented during this profiling; do not alter expert rules or remove checks without preserving their functionality.

## Ablations and attribution limits: 25/10/2026

- Eleven batches of the same 64 scenarios, without touching production code; three references (46.13–48.52 s, median 47.90), ablation of checks, validation, compression, affinity, and final contrast from 32 workers.  
- Block validation of 256 is the top priority candidate: 44.89 s in a trial, CPU logging ~68 → 16.5 s accumulated, NaN check in block and final flush. It preserves schema per row and validation before writing. Detection may take up to 255 additional records: document and test interruptions/resumes before adoption. The observed savings are not a statistical guarantee.  
- Do not recommend removing validations or provenance: weakened variants are diagnostic. Omitting checks did not yield clear improvement over noise; neither did NPZ without compression (+50% disk) nor affinity per core.  
- All 32 matrices from all episodes across eleven conditions were exhaustively compared: 22,528 comparisons pass, dtypes/shapes identical and all floats finite. Do not concatenate these repeated batches as independent data.  
- In the contrast from 24 → 32 workers, approximately 87% of the accumulated time increase corresponds to CPU time and about 13% to time outside CPU. The latter includes I/O, blocking, and waits, not just scheduler overhead. Do not attribute all CPU component to frequency or SMT.  
- Observed CPUFreq-weighted values: P ~2175 → 1967 MHz, E ~1899 → 1783 MHz. These are approximate samples, not a fixed-clock intervention nor a causal percentage.  
- There is no complete causal breakdown between cache, memory, frequency, power, and temperature. `perf_event_paranoid=4`, MSR/RAPL denied, and schedstats disabled limit part of the telemetry. No permissions were altered or global settings adjusted.  
- In this campaign, 24 workers were maintained, normal affinity, compression, and guards. After marginal scanning, Rodrigo selected 16 as the operational balance. A persistent pool with full reset is still a proposal without testing, not an implemented improvement. Do not treat ablation savings as independent effects.  
- Report: [Factor Isolation](docs/PERFORMANCE_FACTOR_ANALYSIS.md). Prototypes, data, versions, and evidence in `runs/factor_isolation_20261025/`. None of these experimental optimizations were activated in the production collector.

## Marginal sweeping of workers: extension of 25/10/2026

- At Rodrigo's request, also measure smaller values and compare the percentage increase of workers with the speed gain. Seven production batches: 4/6/8/12/16/20/24, inverse repetitions 8/6 and 24/20; without active optimizations.  
- Same 64 scenarios from the pilot and configuration/hash/scenario scheme; each batch produced 617,877 transitions and 62 successes. Do not combine replicas as independent data. Here, configuration hashes and aggregates were verified, not all matrices.  
- Median times: 4 → 109.85 s; 6 → 84.70 s; 8 → 73.91 s; 12 → 61.40 s; 16 → 52.69 s; 20 → 48.46 s; 24 → 45.80 s. One run for 4/12/16 and two runs for 6/8/20/24. Small differences should be interpreted with observed variation.  
- Speed gain = T_previous / T_new − 1; time reduction = 1 − T_new / T_previous. Do not swap these percentages. Marginal efficiency = speed gain / increase in workers; 1 would indicate proportionality.  
- The first crossover below 0.5 marginal efficiency occurs between 6 and 8, according to exploratory criterion. The flattening effect becomes more pronounced after 16: 16 → 20 adds 25% workers and provides 8.7% more speed; 20 → 24 adds 20% and provides 5.8%.  
- **Explicit decision by Rodrigo (25/10/2026): use 16 workers as a local equilibrium point**, prioritizing resource margin over small gains in subsequent performance. This replaces the proposal of 20 and the initial recommendation of 24. With 16: 52.69 s per batch, average CPU usage 48.6%, maximum RSS 3.90 GiB. Use `--workers 16` in Rodrigo's local collections unless otherwise indicated. The previous exploratory criterion—being within 10% of the fastest time—selected 20; it remains as a historical result, not current preference. Maintain default 1 portable; do not modify the expert or scientific configuration.  
- No exact optimum or energy consumption is estimated. The ascending order and few repetitions limit inference; other batch sizes require recalibration.
- Continued with cooler and authorized thermal monitoring, without global adjustments.
- Report: [marginal scaling](docs/MARGINAL_WORKER_SCALING.md). Scripts, protocol, CSV, datasets, and graph in `runs/worker_marginal_20261025/`, ignored.

## First learned baseline: decision and execution of 25/10/2026

- Rodrigo authorized training of a linear regressor with 9 outputs, normalization, and training/validation split. Implemented in `src/robot_learning/learning/`, launcher script `scripts/train_model.py`, recipe `experiment_specs/training/linear_baseline.json`. Reuse this interface for future models; do not duplicate ETL by architecture.  
- Explicit selection of a single pilot copy: `w16_r1` from `runs/worker_marginal_20261025/datasets/`, dataset `20261024T234048Z_worker_scaling_pilot_not_final_training_4e159d47`. Its 64 episodes contain 617,877 transitions; do not concatenate replicas.  
- Validate schema, checksums, completeness, finite inputs, and wheel pairs. Preserve complete expert failures; reject exactly duplicated initial conditions/objectives instead of allowing leakage between splits.  
- Episode-level split with seed 20261025, validation_fraction 0.2 (ceiling): training 51 episodes / 490,730 rows; validation 13 / 127,147. Both labels `stalled` remained in training. Post-hoc audit: both are within 1 cm; do not label them as "not achieved" or "demonstrated impossibility" examples. No final test exists: the holdout is development and results have already been inspected.  
- Adjust StandardScaler of X/Y only on training data and save means/scales. Multivariate OLS with intercept, no regularization or tuning. One row per transition, no windows or Fourier; raw bearing from baseline. Constant columns scaled to 1. Uniform adjustment per transition; episode weighting remains as a variant.  
- Proposed offline primary metric: per episode, RMSE per channel divided by training std, averaged over wheel and torque means with equal weight (0.5 per group); then average across episodes. Do not mix rad/s and N·m, nor refer to NRMSE as accuracy. Save physical MAE/RMSE per channel/group, phase-wise errors, pooled and per-episode errors, and comparison with training mean predictor. Do not use MAPE with zero commands.  
- Validation result: balanced NRMSE 0.59747 vs mean 0.93804 (−36.3%). Macro MAE per episode/group: wheels 0.9333 rad/s, torques 0.4999 N·m. These are offline errors on expert states, not successes of the learned robot.
- Model 198 coefficients/intercepts + 60 normalization values, NPZ 3232 B.  
  Local inference median 6.36 µs, p95 7.13 µs, without state/simulator reading.  
  No hidden clipping; limits must be declared before autonomous evaluation.  
- Run local `runs/training/linear_pilot_20261025/` with config, split, metrics,  
  model, versions/hashes, snapshot, and test log. Portable model with NumPy;  
  training uses existing scikit-learn in RoboEnv, without installing dependencies.  
- The 17 tests pass: 13 previous and 4 new (symmetry, corruption/duplicates,  
  episode average, scaling only training, and artifact equivalence).  
- Closed-loop evaluation remains pending, without changing motor/expert controller.  
  Proposed plan: success 1 cm/0.5 s, final distance/min distance, time and saturation  
  with common horizon; do not present as implemented. Report and usage in  
  [training baseline](docs/BASELINE_TRAINING.md).  

## Baseline error interpretation: 25/10/2026

- Rodrigo requested contextualizing MAE with respect to expert commands. Post-hoc diagnosis on the same validation set, without retraining the model or normalizers.
- Wheels, average per episode: MAE 0.9333 rad/s and expert magnitude 1.5717 rad/s; ratio 59.4%. The error remains large. Do not use signed average to divide: commands can cancel out. MAE/media(abs(expert)) is not MAPE.
- Report by joint: torque_2 MAE 0.733 N·m vs magnitude 18.092 (4.1%), but NRMSE 0.780; large offset may hide poor adaptation to variations. Torque_1 MAE 0.696 vs magnitude 1.006 (69.1%). Do not summarize quality with a global torque or with the percentage of actuator limits.
- In 26.8% of validation rows, the expert sends zero on both sides; predicted magnitude average 0.516/0.540 rad/s (right/left). Concerning diagnosis of stopping, without confusing it with a rollout or base real velocity.
- The improvement over the mean demonstrates offline learning, not sufficient control. Do not change the primary metric to accommodate results. These diagnostics are post-hoc; autonomous control evaluation remains pending. Evidence in `runs/training/linear_pilot_20261025/error_context.json` and report [error interpretation](docs/BASELINE_ERROR_INTERPRETATION.md).

## Organization, environment, and collaboration

- Keep the repository organized: logic in `src/robot_learning/`, small launchers in `scripts/`, recipes in `experiment_specs/`, documentation in `docs/`, and tests in `tests/`. Remove dead code during refactoring. Do not create empty trees for future features.
- Use the Pixi environment from RoboEnv, located as a sibling directory. RoboEnv also contains necessary assets and libraries; it is not just an environment. Do not copy `models/` or `configs/` nor create absolute symlinks in this project: pass its root via `conf_file_path_ext` to SimInterface and PinWrapper.
- `.env` is local and ignored; `.env.example` is shared. The terminal takes precedence. Detect the sibling RoboEnv or use `ROBOENV_ROOT`. The collector takes GUI and JSON/CLI steps, not from `.env`. Expert rules reside in JSON, not in `.env`.
- `pyproject.toml` describes the project package. The environment and lockfile remain in RoboEnv; do not version environment files, generated data, or weights.
- Carefully manage `.gitignore` patterns: `/data/` must be anchored at the root to avoid hiding `src/robot_learning/data/`.
- Group repository: `origin`, Rodrigo30406/comp0245-final-project. Instructor: `upstream`, VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC. Manually review their updates: they may cause conflicts or modify the expert.
- Work normally on the `rodrigo` branch, keeping `main` as integration; verify the actual branch and state before operating. `week1/` and `week2/` were intentionally removed from `main`: do not restore them without justification.
- Preserve user changes and changes from other clones/repositories. Distinguish local work, commits, and published changes; do not assert that something was pushed without verifying it.

## Execution and verification

From the project with the RoboEnv environment activated:

```bash
python scripts/simulate.py --target 1.5 1.0 0.7
python scripts/collect_dataset.py --config experiment_specs/collection/smoke.json
python scripts/collect_dataset.py --config experiment_specs/collection/pilot_box.json --workers 16
python scripts/train_model.py --dataset "datasets/FULL_DATASET_NAME"
python -m unittest discover -s tests -v
```

Without activating it, prepend `pixi run --manifest-path "../RoboEnv/pixi.toml"` to the Python command. The simulator supports `--config`, `--headless`, and `--max-steps`. Without `--target`, it retains the previous interactive target `[0, 0, 0.1]`.

When modifying control, physics, sequence, or data, run the relevant tests. The fixtures preserve previous checkpoints from the collector and four cases from the teacher loop: success, stagnation, navigation with timeout, and short timeout. Compare actions/states and stopping decisions, not just absence of errors. After adding workers, all 13 tests pass (24/10/2026), including matrix equality between sequential and parallel runs, a worker crash, and Ctrl+C with restart. This does not replace verifying subsequent changes nor exhaustive equivalence testing.

Operational documentation: [architecture and origin](docs/ARCHITECTURE.md), [simulator](docs/SIMULATOR.md), and [data collection](docs/DATA_COLLECTION.md). Maintain these guides consistent with implemented commands and behavior.

## Joint-level presentation: decision of 25/10/2026

- Rodrigo confirms separating torque errors by joint. MAE/RMSE in N·m, NRMSE with training and predictor mean standard deviation for each of the seven joints. The arm average remains as a secondary summary, without hiding channels.
- `learning/metrics.py` adds per-output aggregation with equal weight per episode and reports `per_output_metrics.csv` (training/validation, per episode and pooled). New runs save CSV checksums; no changes to tuning, split, or primary metric.
- First run: report derived from historical metrics, without retraining or rewriting original artifacts. Table available at [joint-level error](docs/PER_JOINT_ERRORS.md). All four learning tests pass, including aggregation and CSV export.

## Two-layer MLP: implementation and results of 25/10/2026

- Rodrigo authorized the following model: MLP with two hidden layers of 128, ReLU, linear output of 9; 21 original inputs without new features. 20,489 parameters.  
Same expansion `[right,left,left,right]`, without hidden clipping nor expert modifications.  
- Recipe `experiment_specs/training/mlp_baseline.json`; same launcher and ETL.  
`learning/fitting.py` separates the specific training from the common pipeline; OLS is retained. Inference via `MLPPolicy` with NumPy, NPZ artifact without pickle.  
- scikit-learn already present in RoboEnv, without installing PyTorch: Adam lr=0.001, L2 alpha=0.0001, batch=1024, seed 20261025. Weights float32, scalers float64. Loss normalized MSE per channel; selection metric preserves 50/50 group weights and episode mean. Do not confuse these two functions.  
- Same dataset, split, and normalizers as OLS, verified exactly.  
`partial_fit` per epoch preserves Adam and RNG; automatic holdout by row is disabled. Validate with the 13 full episodes; maximum 60 epochs, patience 8, min_delta 0.0001. Save strictly minimum NRMSE and evaluate that checkpoint. The artifact is for inference; does not include optimizer state for resuming.  
- Run `runs/training/mlp_pilot_20261025/`: 22 epochs, best checkpoint 21, selection by validation (not test). NRMSE training 0.13282, validation 0.20313 versus 0.59747 OLS (−66.0 %). MAE wheel velocities macro 0.2751 rad/s vs 0.9333 OLS.  
- Report joint torques: joints 1–6 improve; torque_7 degrades from MAE 0.000083 to 0.048657 N·m. Do not claim improvement across all or hide it with averages.  
- Training, scaling, and validation per epoch: 31.35 s; NPZ artifact 78,789 B, median inference 15.02 µs and p95 15.62 µs (one thread, 2000 calls). Does not include reading/simulator; historical times for OLS are not paired benchmarks, nor do they align with the same workload.  
- One seed and holdout already used for selection: exploratory result, without
- Final test without comparison between seeds. Autonomous control has not been executed.  
- New test covers minimum checkpoint, same split, NumPy/scikit-learn inference, saving/loading, and expansion; documentation in [baseline MLP](docs/MLP_BASELINE.md).

- MLP verification: all 18 tests pass (13 previous simulator/collector tests + 5 learning tests), 25/10/2026.

## Scientific tracking and label correction: 25/10/2026

- Rodrigo requests maintaining all progress records for the final report.  
  Update `docs/RESEARCH_LOG.md` after each experiment or substantial decision with: question, intervention/controls, data/split, seeds, metrics, selection, positive/negative results, limits, and links to evidence.  
  Distinguish post-hoc diagnostics from protocols defined prior to the experiment.  
- The plan retains proposed approaches; the record describes actual execution. Status: pilot phase and offline baselines (P2 in progress); P1 is not closed without autonomous evaluator.  
  Explicit adaptations: OLS before Ridge, Adam/L2/batch1024 instead of proposed AdamW with smaller batches, offline selection while rollouts are still pending.  
  One seed, no final test. Do not declare the entire plan completed.  
- **Verified correction:** 62/64 labels marked as `success`, two marked as `stalled`, but 64/64 final errors and pre-stop <1 cm. Episodes 6/17 end at 0.0494/1.1617 mm.  
  `check_stop` returns stalled status before verifying success; both cases may be within tolerance. Preserve expert/data, do not retrospectively relabel.  
  Do not repeat the previous assertion that the expert only achieved 62 points.  
- Success according to script, geometric distance, and arrival with permanence are different metrics. These data do not certify 0.5 s permanence. A subsequent descriptive audit in `runs/research_audit_20261025/` will verify counts and cross-check metadata hashes against historical splits.  
- Before increasing complexity, proposed priority: implement a common autonomous evaluator and paired expert scenarios for expert/OLS/MLP. This has not yet been executed.

## P1: autonomous evaluator implemented and verified on 25/10/2026

- Rodrigo requests closing P1. `evaluation/` contains protocol, metrics, worker, and CLI;  
  `scripts/evaluate_policy.py` is the launcher; recipe in  
  `experiment_specs/evaluation/validation.json`. Reuse Episode, do not duplicate  
  the loop nor apply teacher forcing or fallback to expert commands.  
- Episode accepts explicit policy/horizon_steps/action_filter; by default it follows  
  exactly the original semantics of the teacher. Learned policies never call expert controllers.  
  Evaluation continues until common horizon even if the original expert had stopped; preserve  
  its first stop as diagnostic.  
- Protocol frozen before 39 rollouts: 13 historical validations, expert/OLS/MLP,  
  40 s common time, external success ≤1 cm during 0.5 s sampled per step; secondary  
  thresholds 2/5 cm. No stop due to success/stall. Continuing until end allows measuring drift.  
- Explicit filter for all: wheels ±5.272727 rad/s (based on expert parameters derivative, not certified hardware), arm ±30 N·m. Save raw/applied and command saturations. Expert was not clipped. Offline historical data without filtering.  
- New initialization per episode and original seed; verify initial observation exactly  
  against source, model/split/active and provenance. Dynamic queue with 16 workers; Ctrl+C cleans all children via existing utilities.  
  No resumption or overwrite of evaluation; artifacts in runs/evaluation/.  
- Final metrics only on full horizon; failure forces false success and physical final state null.  
  Explicit penalty of 10 m for final/rest of integral, reported separately from physical distances.  
  Complete finite does not imply physically valid.  
- Count joint limits only if lower < upper (continuous cases excluded). The teacher's posture already starts joint_6 at 1.0 rad below the limit ~1.134: record initial condition and per-joint counts, do not blame the learner for everything.  
- Run validation_20261025: 39/39 complete, 40,000 steps each, ~113 s with 16.  
  Stable arrival at 1cm: expert 13/13, OLS 0/13, MLP 0/13. At end, expert 11/13 within 1cm: brief permanence does not equate to stability until second 40.
Final median: expert ~0.00162 m, OLS 2.442 m, MLP 926.575 m; MLP shows strong simulated divergence, not physically plausible. No filtering or editing applied.  
- Saturation levels: wheels OLS ~90%, MLP ~94%; torques OLS ~48%, MLP ~83%. Offline improvement does not transfer to control. Cause not isolated: cannot attribute solely to memory, architecture, or scale without diagnosis/control. No model was retrained.  
- Verification: 18 previous tests and 7 new ones pass; 13 of the initial expert stops match, 1,267 sampled transitions equal the data prefixes. Sequential/parallel smoke: 18 identical matrices; model commands correspond to the previous live state. Evidence and graphs in the run and docs/AUTONOMOUS_EVALUATION.md.  
- **Current status: P1 closed for nominal development bank. P2 in progress, with initial autonomous evaluations completed and failure diagnosis pending.** Replaces previous notes that left the evaluator open. No final test, multiple seeds, contact/autocollisions, robustness, or statistical confirmation present.  

- Post-hoc reinforcement of coordinator: verify operational file and model hashes before launching jobs and at completion. Eight specific tests pass (with the 18 previous ones, totaling 26); full smoke update of three policies. Does not alter control/metrics; a batch of 39 maintains its historical snapshot and does not repeat due to this reinforcement. Original hashes of protocol/jobs/models verified.  

## Command interpretability fidelity: post-diagnosis of 25/10/2026

- Rodrigo asks to distinguish proximity to expert from stability in control. Add curves of fraction within physical tolerance per channel and nine simultaneous outputs; save P95 of absolute error as tail diagnosis. Maintain original MAE/RMSE/NRMSE metrics; NRMSE .203 is not 20.3% error.  
- `learning/fidelity.py` computes profiles; `learning/report.py` and launcher script `scripts/report_command_fidelity.py` compare frozen models, same split and scalers, without retraining and without altering previous metrics/checkpoints. They verify model/split and episode hashes upon loading.  
- Descriptive grid ±.05/.1/.2/.5/1 rad/s for wheels and N·m for torques, without declaring acceptable control margins. Macro fractions per episode, P95 over grouped rows; both aggregations identified in JSON.  
- MLP at ±.1: right wheel 33.0%, left wheel 31.7% within tolerance; nine simultaneous outputs 5.06%. At ±.5, nine simultaneous outputs 71.61%. P95 of wheel error is .769/.894 rad/s. These are offline expert states, not on-policy states.  
- Tolerance does not certify stability and its calibration requires declared sensitivity experiments per state/phase. To consult the expert in learned states, first design the protocol for its internal memory.  
- Two new tests pass (channels/tolerances and invalid data rejection). First report and verified repetition are identical; evidence in `runs/diagnostics/command_fidelity_verified_20261025/` and documentation [command fidelity](docs/COMMAND_FIDELITY.md). No new rollouts.  

## Base/Arm/Stop Separation: Decision and Implementation on 25/10/2026

- Rodrigo requests separate evaluation of stopping, and adds base trajectory and arm position. Implemented in evaluation/subsystems.py and enriched worker logging; reusable report evaluation/subsystem_report.py + scripts/report_subsystems.py.  
- Base: navigation meta with expert standoff (no target EE), error in XY and yaw, planar arrival with pos ≤ 0.08m and yaw ≤ 0.05rad during 0.5s, drift/travel. Temporal trajectory imitation and sampled path distances serve as diagnostics, not requirements for a unique path. Store height/tilt to avoid hiding 3D divergence.  
- Arm: EE/world shoulder and EE relative to base using full quaternion including roll/pitch. Compare EE local with reference at identical times. Do not reconstruct previous geometry assuming base is horizontal: the full quaternion was missing. Shoulder-target sphere at 0.85m is only expert rule, not a reachability test.  
- Stopping: distinguish command zero (both wheels |u| ≤ 1e-6) from physical motion; zero torques do not imply joint stops. Record XY velocities and yaw, arm qd, and EE displacement in the terminal window of 0.5s, without an arbitrary stability threshold chosen after reviewing results.  
- Relative fidelity added to reports: curves at 5%/10%/20%/50% of non-zero commands, zeros separated; mixed curves with atol of 0%/1%/5% std of training. No acceptable margin was selected, and the primary metric was not changed. Data stored in relative_and_stop_pooled.  
MLP on expert states during stopping: predicts median |wheel| of 0.291 / 0.349 rad/s, OLS 0.516 / 0.540. This is an offline diagnostic of commands, not a prediction of physical velocity.  
- 39 rollouts were repeated with frozen models: subsystems_validation_20261025. The 234 common historical matrices and original summaries are identical.  
Base arrival planar: expert 13/13, OLS 0/13, MLP 0/13. Median RMSE of EE local vs expert: OLS 0.757m, MLP 0.873m. Symptoms are present in both; no causal attribution is made independently due to coupling, nor is it asserted that base is correct or arm failure is isolated.  
- Extreme MLP results show simulated system divergence, not physically believable motion. The local frame shows different geometry even when the base is removed; temporal comparison also penalizes execution phase shifts.  
- Two geometric tests pass, three fidelity tests (one new for relative/zero cases) pass.
and eight from the evaluator. The expert motor/controller and training remained unchanged.  
- Report and evidence: docs/BASE_ARM_STOPPING_DIAGNOSTICS.md and  
  runs/diagnostics/subsystems_20261025/; update research log.

## Controller dependencies: audit of 25/10/2026

- In response to Rodrigo's question about separating models: the base does not directly use q/qd from the arm; it uses base pose, target→standoff, time, and memory align/drive/orient/done. The arm receives reach_active = nav.done: it maintains q_hold before and performs IK with q_des integrated afterward. done may come from timeout, and does not imply base has reached the target.  
- Base pose affects the arm's kinematics and gravity, and there is physical coupling. Two regressors (2 and 7) are a valid proposal, but training decisions have not yet been made; they should not be treated as independent. Separating them does not eliminate hidden memory in the baseline of 21 inputs (phases/nav.done/q_des absent). Learned phase/history and hybrid variants with expert phase must be explicitly distinguished. Total capacity must be controlled when comparing separation against a shared MLP. Details in subsystem report.

## Metrics and base–arm compensation: correction of 25/10/2026

- Rodrigo correctly points out that another base requires a different arm for the same goal.  
RMSE of EE in base frame vs. expert trajectory only measures temporal imitation,  
and does not isolate arm task error. One cannot conclude from 0.757/0.873m that both modules  
fail causally: navigation fails and local trajectory differs; there may be natural compensation and phase lag. Preserve historical values, correct language.  
- The task error uses the target transformed with the current base, not the local target from the reference. Its norm matches the global EE-goal error. Do not add as independent magnitudes the navigation and arm errors.  
- Future: global metrics across all scenarios + arm diagnosis conditioned on reachability with explicit coverage and denominators. 0.85m is a proxy for expert performance, not a physical maximum or certified value. A valid IK solution demonstrates a feasible case; solver failure does not demonstrate impossibility. Do not require an exact base goal if another pose allows reaching. Standoff goal is a reference strategy.  
- Isolating the arm will require a fixed common base and demonstrated reachability, declared as a separate experiment; not executed. Do not globally filter episodes out of range nor compare conditionally subsetted scenarios as causality.  
- Updated docs/BASE_ARM_STOPPING_DIAGNOSTICS.md; P2 remains in definition and diagnosis, without new training or rollouts due to this discussion.  

## Current metrics: base and system, trajectory diagnosis (25/10/2026)

- Rodrigo authorizes implementing a base-centered and system-wide evaluation, removing isolated arm scoring and investigating where differences begin per step, with expert vs. model graphs. Replaces the previous proposal of scoring the local EE as arm control; historical data is preserved. MAE/RMSE of torques remain command fidelity metrics, not isolated task metrics.

- Active report `evaluation/subsystem_report.py` (existing launcher) now generates `trajectory_analysis.json` v2, including base/navigation/stopping + global task, without arm score. Pure metrics in `trajectory.py`; graphs/export in `trajectory_plots.py`.

- New recipe `trajectory_diagnostic.json`: same control/horizon of 40s, store each step every 1ms up to 2s, then every 100 steps. Control period remains unchanged. Worker saves `action_observation`/`action_time` pre-step, `has_action`, and expert phase/navigation only for diagnostic purposes. Initial placeholder does not count as prediction.

- Detect first observations of XY separation: 1mm, 1cm, 10cm; yaw: 1°, 1°, 5°; global EE: 1mm, 1cm, 10cm; descriptive thresholds, not stability limits. After 2s, transients may occur that are not visible: preserve prior time and sample gap. Integrate RMSE/MAE using real-time sample timestamps, not row-wise averages.

- Run `trajectory_validation_20261025`: 39 frozen rollouts; 390 identical common matrices and identical prior summary. 2381 samples/traces; 2000 initial controls (PRE-step) aligned. Do not increase independent population by repeating scenarios.

- Different commands from t0 with identical input in 13/13 scenarios for both models. Median first XY crossing: 1mm OLS 24ms / MLP 115ms; 1cm 77ms / 313ms; 10cm 942ms / 1484ms. First scenario id2: expert wheels (+3, -3), OLS applied (+5.273, +3.821), MLP (+2.182, -2.206) rad/s. OLS prioritizes forward motion over rotation; MLP performs rotation with different magnitude.

- Case id2 MLP: global EE separates by 1cm at 154ms; base XY by 1cm at 413ms. Do not assume that correcting only XY guarantees success: maintain global task metrics; cause is not isolated.

- 40 PNG / 27 PDF with all 13 scenarios, without selecting only favorable ones. First case
- As fixed example from the manifest; base/EE global curves, full times and initial zoom, commands applied PRE-step. P10–P90 bands are not statistical confidence intervals.  
- Four new tests (unequal time, crossing, alignment, report without arm score and integrity) and eight from the evaluator pass; smoke real generates 4 families of plots.  
Reports/figures in runs/diagnostics/trajectory_validation_20261025, index in docs/TRAJECTORY_FIGURES.md, method in docs/BASE_AND_SYSTEM_TRAJECTORIES.md.  
No retraining or expert tuning was performed. P2 remains in diagnosis.

## Saturation and orientation: 25/10/2026

- Rodrigo authorizes diagnosing the onset of saturation and adding orientation to figures.  
Reuse traces, do not change physics, expert, models, or timeout. Report v3.  
Clipping = raw vs applied >1e-9; touching the boundary is recorded separately. PRE-step, placeholder excluded. Exact starts only with continuous recording from the beginning; scattered samples may omit prior transients. Full-step counters are source of fractions; do not count unequal rows to estimate temporal occupancy.  
Yaw arrows in XY, continuous yaw reconstructed and signed difference wrapped. Unwrap may lose revolutions with insufficient sampling; do not infer exact rotation in rapid divergence. Reference done uses nav.done, not just phase string.  
OLS clips wheel t0 at 10/13; MLP median wheels at 0.526s, torques at 0.350s. Case 2: MLP torques at 0.347s/wheels at 0.552s. Previous deviation: clipping has not been isolated as the cause; failure begins before 25s. No new supervisor or training.  
Final evidence saturation_orientation_verified_20261025, method in docs/SATURATION_AND_ORIENTATION.md. Five trajectory tests pass. P2 is active.

## First limitation and offline audit: 25/10/2026

- Rodrigo requests logging of L1: unbounded output and autonomous degradation, without asserting large predictions from t0. docs/BASELINE_LIMITATIONS.md.  
- All 490730 train and 127147 val rows audited with frozen models/hashes.  
  Maximum absolute train wheel velocities/torques: expert 3/30, OLS 6.309/49.831, MLP 3.926/37.304.  
  No extreme magnitudes observed offline; moderate overflows are present even in training.  
  Labels are not predictions; the observed range is not an evaluation limit.  
- Varying feedback/distribution is compatible with results and has not been isolated as the cause; no specific OOD variables asserted without measurement. No new model, supervisor, or limit changes. Evidence: offline_output_ranges_20261025.  

## Exploratory P2 Closure: 25/10/2026  

- In response to Rodrigo’s proposal, P2 is closed as an exploratory pilot with negative results and a descriptive diagnosis, no isolated cause or stable control. OLS replaces the Ridge baseline proposed as linear; Ridge is not executed. Data quantity curves pending for P3; multi-seed/test final runs also pending.  
- State feedback has already been implemented. Previous history/action differs from adding expert corrections in on-policy states (e.g., DAgger). Next change remains to be specified; no assumption of authorization to mix both or to add an expert phase as input. Expert memory protocol and budget-based control are mandatory before considering DAgger evidence valid.  

## Questions to Professor: 25/10/2026  

- Rodrigo requests a section of questions, recorded in RESEARCH_PLAN.md §17.1: physical limits vs software/envelope evaluations, external clipping vs bounded output, and 0.85 m reach/standoff/timeout as rules to emulate or factors to vary.  
- Inquiry about time as input and external supervision; if configuration varies, clarify conditional inputs vs separate models. Responses pending.  
- Do not infer that clipping causes failure or that tanh is mandatory. No model, limit, or expert rule changes are made when documenting questions.


## English-language collaboration: 25 October 2026

- Rodrigo requested English for all maintained project documentation, code comments,
  docstrings, CLI messages, plot labels and new filenames because the other two
  teammates do not speak Spanish. This is the repository language from now on.
- Renamed 21 guides in docs/ and updated references and heading anchors. The
  research plan, research log and lecturer questions are in English. Frozen
  experiment records and the lecturer's original material remain unchanged.
- English figure export: runs/diagnostics/trajectories_english_20261025, with
  identical analysis JSON, 66 PNG and 53 PDF. No expert, model or simulation
  behaviour changed. Five trajectory tests and the 292-link audit pass.

## Primary relative command fidelity: clarification on 26 October 2026

- Rodrigo reconfirms maximum relative command error of 5%, excluding zeros:
  abs(pred-expert)<=0.05*abs(expert). No additive absolute floor in this criterion.
  ±0.1 physical-unit tolerance and other profiles remain supplementary only.
- Existing numerical-zero convention abs(expert)<=1e-6 remains; exclude those
  rows from relative denominators, report absolute mean/P95 and counts separately.
  No zero-command acceptable threshold selected. Per-output fraction is currently
  pooled over the same 13 expert-state validation episodes, not episode-macro.
- Export/figure primary_relative_5pct under runs/diagnostics/dataset_sizes_20261026;
  docs/DATASET_SIZE_TRAJECTORIES.md foregrounds this criterion. No retraining,
  checkpoint-selection change or claim that 5% command error guarantees stability.


## Error tails and dense onset: 26 October 2026

- Rodrigo requests evidence for >5% error magnitude, overfitting and saturation
  causality. Full report docs/COMMAND_ERROR_AND_DIVERGENCE.md; reproducible audit
  and artifacts runs/diagnostics/failure_audit_20261026. No model/control changes.
- Frozen common-bank MLP512 failed-wheel mean abs error .1910 rad/s, P95 .4444;
  mean relative error 113.34% vs median 10.21%, strongly skewed by near-zero
  references. Keep strict 5% and zeros separate; all channels/bins reported.
- 65 new 2 s dense prefix rollouts, 1 ms records, 16 workers; expert + four MLPs
  on same 13 scenarios. 1,365 shared samples equal original 40 s traces exactly.
  Repetitions are diagnostic, not independent data. All 52 initial MLP predictions
  differ but remain within limits; XY 1 mm, yaw 0.1° and EE 1 mm deviations all
  occur before any clipping. These dense measurements supersede coarse onset
  bounds for the first 2 s only. No within-2s clip means censored, not never.
- Clipping does not initiate these observed early deviations; its later effect
  and the cause of instability remain unisolated. Improving expert-state validation
  does not demonstrate recovery on learned states; do not call classical overfitting
  proven. Missing memory and distribution shift remain hypotheses, no window/DAgger
  or clipping ablation implemented. First-command comparison is on identical input;
  later separate live commands must not be labeled supervised prediction error.



## Corrected experiment sizes and sampling decision: 26 October 2026

- Rodrigo corrects the planned episode counts to powers of two: 64, 128, 256,
  512. The off-schedule cohort is withdrawn from maintained guides and the active
  plan. Do not relabel its measurements as 256. At this correction that size was
  pending; the later completed full series now supplies actual new measurements. Exact superseded documentation and actual frozen artifacts are
  retained outside the maintained guides in ignored runs/archives/.
- Current measured uniform references remain 64, 128 and 512. Full native online
  validation has 13, 26 and 103 cases, respectively. Expert reaches all targets;
  learned OLS/MLP task arrivals remain zero. No new model is implied by this edit.
- Rodrigo approves stratified training targets and spatially homogeneous validation,
  compared against uniform sampling. Start with the proposed 64-episode pilot:
  one jittered target per 4x4x4 cell in the unchanged box, 51 train / 13 validation.
  Holdout selection uses cell geometry only; balanced per-axis levels, all octants
  and training coverage retained. Never choose heldouts based on outcomes.
- Run unchanged OLS and MLP recipes, and compare every model on both disjoint
  13-case banks and their common 26-case union. Explicit external-bank evaluation
  must check all conditions against every model's training set and preserve the
  physical/expert/config provenance checks; default strict behavior stays intact.
- Keep the primary 5% relative command criterion, zeros separately, full offline
  and online evaluation, navigation/orientation/stopping/clipping and task metrics.
  The temporal-window experiment remains proposed, not implemented here.
- Training audit: MLP64 stopped by patience at epoch 22 (best 21); MLP128 at 17
  (best 9); MLP512 hit the 60-epoch cap (best 58). Do not claim the 512 model
  converged. Early stopping monitors validation NRMSE, patience 8, min_delta 1e-4.
  Preserve the original budget for the sampling comparison; a larger budget is
  a separate experiment, not an unrecorded change to this comparison.


### Stratified pilot results and limitations

- Pilot completed:64 episodes,602,889 transitions;51 training/13 validation.
  All130 matched40s rollouts valid. Original uniform policy/expert metrics on
  old13 exactly reproduce. OLS and unchanged MLP trained on the new data.
- Common26 MLP NRMSE uniform0.27955 / stratified0.29300; median final base
  error203.563 /2.387m, EE899.748 /2.707m. Reduced divergence, but both MLPs
  still0/26 base/task arrivals; expert26/26 task. Do not claim stable control
  or universal superiority of stratification. Training/validation geometry and
  selected checkpoints (uniform21,stratified3) are not causally separated.
- Stratified MLP ran11 epochs and stopped by patience; uniform512 remains the
  historical60-epoch-cap run. No training-budget extension was executed.
  Those cohorts were pending at pilot completion; the full-series result below
  now supersedes that status. Current
  source and default learning-curve spec use the corrected powers-of-two plan.
- Detailed findings, plots and reproduction: docs/SAMPLING_COMPARISON.md;
  stopping evidence: docs/TRAINING_STOPPING.md. Preserve complete frozen runs.

## DAgger limitation: decision of 26 October 2026

- Rodrigo requests explicitly documenting the unresolved expert-memory protocol
  before applying DAgger. Expert labels depend on observation and history
  (navigation phases, `nav.done`, integrated `q_des`); do not reset the expert
  at each query or assume the 21 baseline inputs uniquely determine its action.
- Define and verify causal expert-memory updates once per physical step during
  learner rollouts and audit recovery behavior before collecting corrections.
  Preserve expert rules, separate queried/applied commands, exclude held-out
  scenarios, and control labeling budgets. Privileged memory inputs are a
  declared variant, not an implicit baseline change.
- DAgger retains supervised expert labels but extends collection beyond pure
  expert rollouts. Lecturer acceptance of this extension is not confirmed.
  No implementation or experiment is authorized by this documentation decision.
- Evidence and status: docs/BASELINE_LIMITATIONS.md (L2), research log R26.


## Full sampling series authorized: 26 October 2026

- Rodrigo answers the scope question: execute the complete64/128/256/512 series.
  This supersedes pilot-only scope. Retain completed64 and frozen uniform128/512;
  collect/train actual uniform256 and stratified128/256/512. No extended epoch
  budget, history, DAgger or expert change is authorized by this series request.
- Grids4x4x4,8x4x4,8x8x4,8x8x8; splits51/13,102/26,204/52,409/103. Larger
  stratified designs are not nested. Geometry-only balanced holdout selection
  preserves axis margins/octants and training coverage before outcomes.
- All models get same26-condition anchor evaluation plus full native per-size
  banks, with external-condition training-leakage/provenance guards. Reuse valid
  old uniform outcomes; do not merge independent targets or relabel old runs.
  Uniform256 extends128 split, not assumed nested within frozen512 holdout.
- Protocol docs/SAMPLING_SERIES.md; active run
  runs/sampling_comparison/full_series_20261026. Three spatial/leakage tests and
  real offline integration pass; old pilot metrics reproduce exactly. Source
  must remain unchanged during provenance-checked collection/training/evaluation.


## Complete sampling series: results supersede pending notes

- Full64/128/256/512 series is complete for both uniform and stratified targets.
  The actual uniform256 cohort was collected/trained with204/52 split; do not
  relabel retired experiments. See docs/SAMPLING_SERIES.md and
  runs/sampling_comparison/full_series_20261026 for measured tables/provenance.
- Fixed26-anchor evaluation and full native unions protect comparisons from
  changing goal banks; all selected conditions are excluded from every model
  training set. Geometry layouts/holdouts are fixed before outcomes. Larger
  stratified layouts are independent, not nested. Native full splits retain
  approximately80/20. Full records and failed cases are preserved.
- Original OLS and MLP recipe, physical rules, limits,21inputs/9outputs remain.
  Cap hits are not convergence. No extended training, memory, DAgger, actuator
  bounds or expert behavior change was implemented by this campaign. Results
  are developmental and one realization per design; selected epochs/updates
  and trajectory durations remain confounds. Read actual results before claims.

## Patience-only training and larger cohorts: 26 October 2026

- Rodrigo explicitly authorizes extending both uniform and stratified designs to
  1024 and 2048 episodes and increasing the training budget. This supersedes the
  earlier restriction to the original 60-epoch budget for the new campaign.
- New MLP recipe `mlp_patience.json` uses `max_epochs: null`: stop by the unchanged
  patience8/min_delta1e-4 criterion and restore the minimum-validation checkpoint.
  Reuse seven smaller MLPs already stopped by patience; rerun uniform512, whose
  historical run hit60, with its exact training-only normalizers and same seed.
  Compare the original60-epoch checkpoint explicitly; do not overwrite old runs.
- New splits819/205 and1638/410 retain approximately80/20. Uniform prefixes and
  frozen512 memberships are verified/extended; stratified16x8x8 and16x16x8 grids
  are independent and geometry-only holdouts balanced before outcomes. Preserve
  same26-condition anchor, both complete native banks and leakage guards.
- Disk-backed MLP buffers, incremental training-only normalization and streaming
  final metrics bound memory. Exact frozen normalization is used for the512
  budget comparison. Full Adam/RNG checkpoints permit verified recovery with
  identical data/config/source/version context. They are local trusted artifacts.
- MLP is the experimental focus; previously measured OLS remains a reference.
  No epoch concept applies to OLS. Architecture, inputs, losses, expert and
  simulation rules stay fixed. At most2 concurrent numerical fits,16physics workers.
- Run `scripts/compare_patience_series.py`; results remain pending until its
  manifests complete. Patience is a validation stopping rule, not proof of a
  global optimum, stable robot control or an explanation of512 divergence.

## PyTorch backend authorized: 26 October 2026

- Rodrigo explicitly requests a PyTorch implementation after learning that sklearn
  uses CPU, one thread per fit. Add an optional backend through the existing
  trainer, not a second ETL pipeline. Recipe `mlp_pytorch.json` selects CUDA;
  unmodified recipes retain sklearn. Keep 21→128→128→9, linear outputs,
  normalization/splits, hyperparameters, patience8/min_delta1e-4 and no epoch cap.
- Preserve sklearn's actual output-summed gradients versus reported output-mean
  loss, weight-only per-batch L2, Glorot/RNG permutation stream, and Adam epsilon
  convention. Float32, no AMP/TF32, deterministic Torch algorithms. Framework/GPU
  results remain separate variants; numerical and trajectory identity is not promised.
- Continue CPU campaign evaluations using the immutable source under
  runs/sampling_comparison/patience_series_20261026/frozen_cpu_project. Runtime
  isolation is recorded in runtime_isolation.json; all five new CPU fits had
  already completed by patience before switching. Do not edit that frozen copy.
  A partial temporary offline bank was archived in runtime_recovery and recomputed.
- Local `.venv` is an ignored optional ML overlay on the sibling default RoboEnv
  Python, using system site packages plus torch2.11.0+cu128. This extends the earlier
  environment arrangement only for the new CUDA backend; preserve RoboEnv's
  installed environment/assets and shared simulation path. No env/weights/data
  are versioned. Optional dependency lives in the project's existing pyproject.
- Auto device data residency is bounded by free VRAM; streaming is supported.
  Save full hash-verified Torch Adam/RNG/best/patience checkpoints. Export the
  existing NumPy MLPPolicy artifact so simulation workers need no Torch/CUDA.
- Verification: all47 tests passed, including actual CUDA, first sklearn Adam
  update/loss comparison, exact interrupted-versus-uninterrupted Torch CPU/CUDA
  recovery, patience and portable inference. Real64 pilot/simulator verification
  completed: CUDA54epochs, best46, stop=patience, native13 NRMSE0.182987. Export
  runs without Torch in RoboEnv; paired1case/40s integration valid, model0/1task.
  Do not claim GPU control stability or completed full GPU series. Observed
  median2.227s/epoch does not establish speedup over historical CPU1.418s;
  small-network overhead/workload and different stopping epochs limit comparison.
- Method, setup and commands: docs/PYTORCH_TRAINING.md. Preserve the measured CPU
  campaign; a backend change is not an unrecorded continuation of sklearn Adam.

## Two-computer workflow discussed: 26 October 2026

- Rodrigo wants to share code across laptop and a more powerful computer, with
  large simulations on the latter and development/training changes on the former.
  Generated results remain ignored; code and artifact transport are separate.
- Current source has uncommitted/untracked changes, so remote clone alone is not
  the current implementation. Do not imply it has been committed/pushed.
- Existing artifact manifests use absolute paths; portable artifact resolution
  and verified transfer are pending. Preserve provenance/checksums, never rewrite
  frozen historical manifests merely to fit a second filesystem layout.
- Use independent/frozen source checkouts for long jobs, separate output IDs and
  host-specific worker profiling. Host OS/access details are unknown; no remote
  access or transfer is authorized by this discussion alone. Workflow and current
  limitations: docs/MULTI_MACHINE.md.

## Partial report requested: 26 October 2026

- Rodrigo requests intermediate scientific results before the full campaign
  finishes. docs/PATIENCE_PARTIAL_REPORT.md freezes complete-bank evidence;
  offline through2048, online complete through1024, same26 anchor plus full
  native unions. Main campaign is sklearn/CPU; do not mix the PyTorch64 pilot.
- Controlled extra-epoch uniform512 improves anchor NRMSE0.58% but worsens median
  position errors, task0/26. Both1024 policies remain0/410 primary task arrivals
  versus expert410/410. More data improves regression/zero-command imitation,
  without establishing stable autonomous control or a causal failure mechanism.
- Snapshot artifacts, seven scientific PNG/PDF figure families, strict5% tails,
  CSVs and104 paired anchor diagnostics are saved under partial_reports/20261026T232054Z
  in the active patience-series run. Exclude unfinished banks from outcome
  aggregates and keep snapshot provenance when final results later supersede it.

## Supervised subsystem/joint learning curves: 27 October 2026

- Rodrigo requests separate supervised errors for base, arm overall and every
  joint. Use wheel command MAE/RMSE(rad/s), mean per-joint torque MAE/RMSE(Nm),
  and all seven torques individually. These are command errors, not pose errors.
- Compare64..2048 on the same26-goal anchor, equal episode weights and fixed
  original64 scales. Keep both sampling designs and the CPU campaign separate
  from the PyTorch pilot. Include fixed training-mean predictor errors and both
  absolute/relative improvements; a percentage gain alone can be misleading.
- scripts/report_supervised_scaling.py reuses saved metrics, verifies group and
  per-output identities/checksums, and exports CSV/JSON and five PNG/PDF families.
  Results: docs/SUPERVISED_ERROR_BY_SUBSYSTEM.md and
  runs/diagnostics/supervised_subsystem_scaling_20261027. No retraining is implied.
- The1024→2048 gain is21.27% wheels /2.85% arm for uniform and1.32%/.69% for
  stratified. Several joints worsen; do not infer one universal plateau or stable
  control from an aggregate score. Preserve original expert/physics behavior.

## Full validation coverage correction: 27 October 2026

- Rodrigo explicitly asks for all validation episodes rather than only the26-goal
  anchor. R31's original report is anchor-only and preserved as such. New report:
  docs/SUPERVISED_FULL_VALIDATION.md; artifacts supervised_full_validation_20261027.
- Reporter --bank own_validation includes13/26/52/103/205/410 own goals per model;
  --bank native_union includes26/52/104/206/410/820 identical goals within each
  size for both models; --bank anchor26 preserves the same26 across sizes.
  --bank all generates all views; default is now native_union. Label bank/counts
  in every CSV/JSON/figure. Native goals change across sizes, unlike the anchor.
- Full-validation findings differ from anchor-only trends: stratified2048 own-arm
  RMSE worsens5.66% over1024; uniform2048 paired-union base/arm errors worsen
 12.52%/16.34%. Do not claim monotonic improvement, a universal dataset optimum
  or overfitting from one varying-bank realization. Revisit earlier conclusions
  with scope explicit; no old measurements or provenance are relabelled.
- Own physical errors match full training validation, anchor matches original
  exactly, and model/split SHA/coverage/paired-union identities are verified.
  No additional simulation/training or change to the expert is implied.

## Validation proximity diagnostic: 27 October 2026

- Rodrigo questions whether stratified holdout selection favors targets similar
  to training. Audit the actual rule: geometry-only heldout cell-center maximin,
  balanced axis levels/octants, retained training coverage in vertical pairs;
  no selection by model outcome or observed train-neighbor distances.
- Actual2048 own holdout nearest-training XYZ mean0.09635m uniform /0.10534m
  stratified; normalized means0.04801/.05485. Do not assert stratified validation
  is closer/easier from the design alone. XYZ is not21D state/history similarity.
- Own supervised errors do not always favor stratified: uniform arm lower at all
  six sizes, uniform wheels lower except128. Keep separate from online divergence.
- Diagnostic docs/VALIDATION_GEOMETRY.md; source/split hashes and distances under
  runs/diagnostics/validation_geometry_20261027. No split/model changes were made.


### Patience extension measured completion

- The campaign through2048 is complete; all12 currentMLPs stop by patience.
  See docs/PATIENCE_SERIES.md and runs/sampling_comparison/patience_series_20261026.
  Seven compatible models reused, five newly trained; historical cap retained.
- Uniform512 first60 epoch losses/validation scores reproduce exactly. Maintain
  full-native and fixed26 distinctions, strict5% versus zero diagnostics, and
  read actual results before claiming stability or causal explanations.

## Proposed next experiment order: 27 October 2026

- Rodrigo asks about architecture tuning versus temporal windows/DAgger and a1024
  budget. Recommendation, not an executed sweep: fix1024total819/205, preserve
  split/scalers, establish Torch1024 reference, test simple causal history with
  a parameter-count control, then focus architecture/regularization tuning.
  DAgger remains a separate proposed collection intervention with unresolved
  stateful expert protocol and lecturer-scope question.
- Current full learning curves are sklearn; only the64 pilot has been trained in
  Torch. Do not compare new Torch variants to sklearn as if only memory changed.
- Use all205 own holdouts and410 paired goals at1024; anchor26 is supplemental.
  Preserve1ms control and original expert/limits. Window proposal100ms with six
  measured18D states at20ms spacing plus one3D goal (111inputs), causal padding
  and per-episode reset. Input schema/export/topology support still needs work.
- Avoid exhaustive static tuning as a prerequisite; lower supervised error alone
  has not stabilized control. Dropout/BatchNorm and depth/width are separate
  ablations, with proper eval/export checks. Record seeds and parameter budgets.
- Plan/docs: NEXT_EXPERIMENT_ORDER.md and research logR34. No window, sweep or
  DAgger experiment is launched by this planning discussion. CPU campaign and
  final reports are complete; both2048 policies remain0/820 primary task arrivals.

## Control metric research and proposal: 27 October 2026

- Rodrigo requests literature-backed trajectory similarity/control diagnostics.
  Existing timed errors/departure times/path length/clipping remain; proposal
  adds ordered discrete Frechet and expert length ratio, plus rotation travel
  and normalized raw/applied mismatch diagnostics. Task success remains primary.
- docs/CONTROL_METRIC_PROPOSAL.md distinguishes standards from adaptations and
  already implemented metrics from proposed production extensions. Ratio is not
  SPL, commands are not actuator energy, and original world frame must remain.
- Historical traces are dense1ms only through2s, then100ms: use explicit sampling
  for Frechet, lower-bound sampled rotational travel, and approximate clipping
  integrals. Exact future rotation/mismatch/burst counters require a tested
  per-step evaluation observer, preserving expert/physics/control behavior.
- Six illustrative pairs from three geometry-selected1024 anchor cases are in
  runs/diagnostics/control_metric_research_20261027. Hash/analytic checks pass;
  not full-validation results or an implemented production evaluator extension.
  No new4k simulation campaign, architecture or control rule is launched here.

## Robust control distributions and goal difficulty: 27 October 2026

- Rodrigo requests robust distance summaries and identification of difficult
  conditions. Compute metrics per episode, report median/Q1/Q3 plus P90/P95/max,
  mean, missing/invalid counts and CDFs separately. Preserve genuine failures;
  do not discard or cap large distances to make an aggregate look better.
- scripts/report_control_patterns.py postprocesses complete paired native banks
  at1024/2048 (410/820goals per policy,2460pairs). Includes discrete Frechet on
 401common100ms vertices, full-step path ratio/excess and existing diagnostics.
  No simulation, expert or policy behavior changes. Timed errors complement
  geometric distance; original-target success/dwell remains primary.
- Use fixed geometry bins with counts and median/IQR, plus per-goal maps.
  Current1024 directional/height differences do not clearly replicate at2048;
  no general difficulty rule or causal failure mechanism is established.
  Separate first XY and yaw departure: turning in place can delay XY separation
  while heading is already incorrect. Missing departures are censored, not zero.
- docs/ROBUST_CONTROL_PATTERNS.md and research logR36 document measured results,
  limits and provenance. Artifacts robust_control_patterns_20261027 include all
  per-case CSVs, static figures and standalone interactive map. Three targeted
  tests and full coverage/hash/prototype checks pass. An aggregate schema draft
  is preserved separately as superseded. No training or physics campaign added.

## Temporal window phase started: 27 October 2026

- Rodrigo chooses1024 and authorizes beginning input-history experiments. The
  three-model comparison in NEXT_EXPERIMENT_ORDER.md is now implemented/launched,
  superseding its previously proposed-only status. No improvement is yet claimed.
- Primary data: existing stratified1024, exact819/205 split/scalers. Three Torch
  CUDA variants: instant128, six-snapshot100ms window128, instantaneous164 capacity
  control. Shared seed20261025, original Adam/settings, patience8/min_delta1e-4,
  no epoch cap. No new demonstrations, expert/limit changes or other interventions.
- Window: measured18D states at100/80/60/40/20/0ms, current3D goal once;111inputs.
  Dense1ms source logs required, causal initial-state padding, episode boundaries
  and online reset/dt checks. No privileged controller flags/actions/future data.
  Reuse frozen current21D training scaler at each lag; gather batches on demand.
  Live predict_action must be called once per physical step; batch predict is
  stateless on one complete dense episode. Portable NPZ stores window metadata.
- scripts/compare_temporal_windows.py freezes code before running so workspace
  edits do not invalidate jobs. Campaign stratified1024_window100ms_20261027 under
  runs/temporal_comparison; /tmp/temporal1024_20261027.log. Full205 own +205 external
  goals per model online with16workers, original40s/dwell/clipping protocol.
  Robust paired control report adds CDFs/median/IQR/tails without discarding failures.
- All53 repository tests passed including3temporal tests; source compilation passes.
  Docs:TEMPORAL_WINDOWS.md, research logR37. Read actual manifests for status;
  training/evaluation outcomes and multiple-seed confirmation remain pending.

### Correction: temporal design not yet agreed

Rodrigo clarified that window duration and snapshot spacing must be discussed
before selection.100ms/20ms was only a proposal. The premature campaign was
stopped during instant128 training; no temporal model or autonomous comparison
launched. Preserve interrupted artifacts/checkpoints. Implementation remains
available and tested; do not resume or relaunch the proposed settings without
the design discussion. Research logR38 supersedes R37's implied design approval.

### Final agreed temporal design:200ms /20ms (27 October 2026)

Rodrigo explicitly corrects the final selection to200ms history,20ms spacing.
This supersedes the earlier100ms proposal and brief acceptance. Eleven18D states
plus current3D goal =201inputs;2×128 window43529parameters. Instantaneous controls:
2×128(20489) and2×193(43434). Launcher requires --window-ms and --spacing-ms.
New frozen campaign: runs/temporal_comparison/stratified1024_window200ms_20261027;
log /tmp/temporal1024_window200ms_20261027.log. Old100ms artifacts stay interrupted.
Recover instant128 epoch21 with unchanged verified training-module hashes; actual
CUDA epoch22 completed. Same split/scalers, seed, patience and full410goal protocol.
Four temporal tests pass after correction; source compilation passes. Prior53-test
suite and100ms smoke are historical checks. Docs:TEMPORAL_WINDOWS.md, research
logR39. Read manifests for live progress; no200ms performance results claimed yet.

## Depth/shape screen authorized:27 October 2026

Rodrigo requests more hidden layers and a32→64→128 progression, suggesting an
encoder/decoder-like shape. Implement variable hidden depth while preserving
historical portable two-layer inference and linear nine-command outputs. Test
32,64,128 (3hidden layers) and32,64,128,64,32 (5hidden layers) with both21D current
inputs and the agreed201D200ms/20ms window. This is supervised command regression,
not an autoencoder reconstruction task. No dropout/BatchNorm/bounds/sensor changes.

Parameter counts: increasing12297/18057 and expand-contract21769/27529 (21D/201D).
Reuse Torch instant128/window200ms baselines and the exact stratified1024 split,
scalers, seed and optimization/patience. Label this a depth/shape screen because
width and parameter budgets also change; no pure-depth causal claim.

scripts/compare_mlp_architectures.py freezes source via the shared campaign helper.
GPU fits run sequentially; wait for the prior temporal campaign before launching
additional physics, preserving16workers. Evaluate410 paired goals per new model;
reuse old baseline traces, retain all failures and full robust/control diagnostics.
Expected artifacts architecture_comparison/stratified1024_depth_shape_20261027;
log /tmp/mlp_depth_shape_20261027.log. Read manifests for live status. Results
pending; docs:MLP_DEPTH_AND_SHAPE.md. New deep export/resume/CUDA tests pass.
