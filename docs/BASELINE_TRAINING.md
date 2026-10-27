# Training Pipeline and Baseline Linear Model

Decision by Rodrigo, 25/10/2026: learn **21 inputs → 9 outputs** using linear regression and reconstruct the four wheels by duplicating each lateral command. Keep the original 11 command data; do not modify the expert or their simulations.

## Data Contract

Inputs in order: base XYZ (m), bearing (rad), seven joint positions (rad), seven joint velocities (rad/s), original target XYZ (m). Outputs: `[wheel_right, wheel_left, torque_1, ..., torque_7]`, in rad/s and N·m. The adapter produces `[right, left, left, right, torque_1, ..., torque_7]`.  
Verify exact equality of each pair of wheels in the demonstrations: if not satisfied, reject the dataset rather than averaging or hiding the discrepancy.

The slide «Outline of Assignment Task» shows base pose, q_mes, qd_mes, and target as inputs, and wheel_cmds/torque_arm as outputs. «Methodology» requires justification of data, inputs, architecture, and training; «Evaluation» requires precision, generalization, robustness, and efficiency. No mandatory numerical metric is specified. Reducing base commands to two preserves the expert's identity.

## ETL and Separation without Leakage

1. Select **a single complete dataset**. Validate schema v2, metadata, scenarios, and NPZ checksums. Reject incomplete episodes and duplicated physical conditions; do not mix replicas from worker campaigns. Preserve expert failures: they are valid demonstrations for imitation study.  
2. Split episodes before applying transformations. Use seed `20261025`, validation fraction `0.2`, rounding up the number of episodes: **51 training / 13 validation**. Save episode IDs, targets, hashes, and quantities in `split.json`. The previous label `development` is retained in the dataset; the training split is a new manifest without modifying the data.  
3. Extract `observation` and `action` from each transition, verify dimensions, finite values, and original target; reduce 11 actions to 9. Use a single time step, without windows, Fourier features, or internal controller diagnostics.  
4. Adjust `StandardScaler` for X and Y **only on training data**:  
   `z = (value - mean_training) / std_training`. A constant column uses scale 1. Transform validation using the same parameters. The means and scales are preserved within `model.npz` for identical inference.  
5. Fit `LinearRegression(fit_intercept=True)` via ordinary least squares (OLS), without regularization or hyperparameter search in this initial baseline. Each transition has the same weight: longer episodes contribute more to the fit. Equal per-episode weighting remains a future variant.  
6. Predict, denormalize to physical units, evaluate, and save the model, episode/channel/phase metrics, versions, execution times, and code snapshot. Compare against a constant predictor that returns the **training mean**.  

Outputs are normalized to maintain a common interface with future models.  
For OLS multivariate output without constraints, each output is fitted independently:  
affine rescaling does not change the physical-optimal solution, except for numerical precision. Do not confuse standardization with adding model capacity.

There is no final test in this first iteration. The 13 episodes are development validation episodes used to design the model; they should not be presented later as an intact test. They represent new objectives within the same nominal startup condition, not evidence of generalization to another distribution, different startups, or robustness to noise. The random partition was not optimized based on results: the two expert failures remained in training, and the 13 validation episodes were all expert successes. Robustness against expert failures is not covered by this holdout.

## Metrics Selected

- **MAE and RMSE per output**, in rad/s for each side and N·m for each joint.  
  MAE communicates the typical absolute error; RMSE penalizes large errors more heavily.  
- Average across channels per group: wheels and arm, **without summing distinct units**.  
- **Proposed primary offline metric: balanced NRMSE across groups and episodes.**  
  For each episode, divide the RMSE of each channel by its training standard deviation; take the average of the two wheel channels and the seven torque channels; assign equal weight (0.5) to each group. Finally, average across episodes with equal weighting. Lower is better; zero would indicate exact imitation. This is neither a percentage of correctness nor a percentage of torque.  
- Also store the median and 90th percentile of that error per episode, the errors across all transitions combined, and the errors per expert phase. The phase is used for analysis only and never as input to the baseline.  
- Compare all metrics against the mean predictor. Do not use MAPE because there are zero commands, and do not use R² as the primary metric in channels that are nearly constant.

The 50/50 weighting is an explicit methodological decision, not a requirement from the instructor. Channels are normalized by training standard deviation; a very small variance may inflate the ratio. Physical metrics and scales are always reported. The handling of constant columns with scale 1 preserves numerical units but does not define a physical uncertainty. Before interpreting another dataset, review scales and consider introducing a physical floor for normalization as a new variant.

To evaluate autonomous control, the next step is to integrate `predict_action` into a separate evaluator from the expert and restart the simulator in validation scenarios. Proposed plan: success at 1 cm with 0.5 s of stay within 40 s, final/min distance, arrival time, stability, and saturations; also report arrival compatible with the original criterion. Freeze this protocol before comparing models. **Not yet implemented or executed**. Do not conclude robot success based on command errors.

## First execution: verified result

A specific copy of the pilot was explicitly selected: `w16_r1` in `runs/worker_marginal_20261025/datasets/`, identifier `20261024T234048Z_worker_scaling_pilot_not_final_training_4e159d47`. The other identical benchmark copies were not concatenated.

| Partition | Episodes | Transitions | Expert success labels |
|---|---:|---:|---:|
| Training | 51 | 490 730 | 49 |
| Validation | 13 | 127 147 | 13 |
| Total | 64 | 617 877 | 62 |

| Metric (mean per episode and per channel of group) | Training linear | Validation linear | Validation predictor mean |
|---|---:|---:|---:|
| MAE wheels, rad/s | 0.9600 | 0.9333 | 1.4342 |
| RMSE wheels, rad/s | 1.2076 | 1.1705 | 1.6658 |
| MAE torques, N·m | 0.5198 | 0.4999 | 0.8369 |
| RMSE torques, N·m | 0.9445 | 0.9020 | 1.5903 |
| Balanced NRMSE, unitless | 0.6195 | 0.5975 | 0.9380 |

![Offline evaluation of baseline](../runs/training/linear_pilot_20261025/validacion.png)

Relative reduction of NRMSE compared to the mean in validation: **36.3 %**. This is an offline result from a small holdout; there is still appreciable error in wheel velocities and the model has not been tested in actual robot control. Validation being better than training does not prove anything by itself: episodes and difficulty differ.

The model has **198 coefficients/intercepts** plus 60 normalization values;  
NPZ size of **3232 bytes**. Fitting and transformations: **0.45 s**, excluding reading and evaluation. Local inference with one observation: median **6.36 µs**, p95 **7.13 µs** across 2000 calls after warmup, on a single numerical thread; includes normalization, denormalization, and expansion to 11 commands. Excludes state retrieval and simulation; does not guarantee complete latency or real-time performance. Training matrix range: 21, without columns exactly constant. Scales and singularities in `model_info.json`.

## Organization and commands

- `src/robot_learning/learning/dataset.py`: inventory, integrity, and split.  
- `model.py`: contract 21→9, portable inference, and expansion 9→11.  
- `metrics.py`: physical errors, normalized errors, and episode-level aggregation.  
- `train.py`: pipeline and CLI; `scripts/train_model.py` is its launcher.  
- `experiment_specs/training/linear_baseline.json`: shared recipe.  
- `runs/training/<run>/`: local artifacts ignored by Git. Do not overwrite runs.  

With RoboEnv activated, from the project root:

```bash
python scripts/train_model.py --dataset "RUTA/AL/DATASET"
```

To repeat the first execution with the locally selected copy:

```bash
python scripts/train_model.py \
  --dataset runs/worker_marginal_20261025/datasets/w16_r1/20261024T234048Z_worker_scaling_pilot_not_final_training_4e159d47 \
  --config experiment_specs/training/linear_baseline.json
```

Without activation, prepend `pixi run --manifest-path "../RoboEnv/pixi.toml"`.  
The launcher does not install packages. Uses NumPy, SciPy, scikit-learn, and threadpoolctl already available in RoboEnv, whose versions are recorded. CLI limits numerical libraries to a single thread during training/measurement; does not alter global settings.

Each run preserves `config.json`, `split.json`, `run.json`, `model.npz`, `model_info.json`, `metrics.json`, and `source_snapshot/`. `run.json` distinguishes between running/complete/failed, retains dataset hashes, code version, and local changes, and includes checksums of the model/split/metrics. The dataset preserves its historical provenance; adding learning code does not authorize re-running its original data collection with new hashes.

Inference loading without importing scikit-learn:

```python
from robot_learning.learning.model import LinearPolicy
policy = LinearPolicy.load("runs/training/linear_pilot_20261025/model.npz")
commands_11 = policy.predict_action(observation_21)
```

Predictions are not silently clipped. Before deployment, declare and register actuator limits in the evaluator; this is not part of the current offline result. The linear model can produce commands outside the expert range.

New tests: symmetry and scaling/unscaled, rejection of corrupted data or duplicated conditions, episode separation, exclusive normalization during training, aggregation without long-episode domain, artifact equivalence, and rejection of overwriting. Synthetic test data/models, no training on validation. Complete results from the first run:
[runs/training/linear_pilot_20261025](../runs/training/linear_pilot_20261025/).

## Methodological Sources

- [Assignment slides](COMP0245-Assignment-2026-2.pdf), outline, methodology and evaluation.
- [Research plan](RESEARCH_PLAN.md), sections 9 and 10.
- [scikit-learn: data leaks and preprocessing](https://scikit-learn.org/stable/common_pitfalls): fit transformations only on training data.
- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): means, scales, and handling of constant columns.
- [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html): ordinary least squares multi-output.
- [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html): per-output errors and explicit aggregation.

Post-hoc diagnosis: [comparison with expert magnitudes and stop/move behavior](BASELINE_ERROR_INTERPRETATION.md). Wheel errors remain large relative to their typical scale; do not interpret improvement over the mean as satisfactory control.

Post-hoc presentation decision: [main table of joint torques](PER_JOINT_ERRORS.md). New runs save `per_output_metrics.csv` with individual errors and hash in `run.json`; the arm mean remains as a secondary summary.

Next implemented model: [two-hidden-layer MLP](MLP_BASELINE.md), selectable with `--config experiment_specs/training/mlp_baseline.json`. Retains the training ETL and normalizers; adds `history.json` and minimum validation selection per episode. Previous OLS results retain their historical significance.

Global tracking and interpretation corrections: [research log](RESEARCH_LOG.md). Tables with 62 successes include original labels, not geometrically reached points: there were 64/64 final errors <1 cm.

**Post-update of 25/10/2026:** the autonomous evaluator has been implemented and the first comparison executed (expert 13/13, OLS/MLP 0/13 in stable arrival). References to pending evaluation earlier described the state during training, not the current state. See [autonomous evaluation and P1 closure](AUTONOMOUS_EVALUATION.md).

Interpretable diagnosis afterwards: [fraction within tolerance and P95 of error](COMMAND_FIDELITY.md). Retains original metrics and does not establish a stability margin.
