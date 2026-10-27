# Two-hidden-layer MLP

Authorized implementation by Rodrigo on 25/10/2026. The same data pipeline, inputs, action order, and expert logic are preserved. The only experimental change is the regressor.

## Architecture and training

`21 → Dense(128) → ReLU → Dense(128) → ReLU → Dense(9)`.  
There are **20,489 trainable parameters** and 60 normalization values. Linear output without clipping; the adapter expands right/left to all four wheels. No windows, privileged data, or new bearing encodings are used.

The `MLPRegressor` from scikit-learn is used, already available in RoboEnv, with Adam optimizer, learning rate 0.001, L2 regularization `alpha=0.0001`, batch size 1024, and seed 20261025. PyTorch is not installed and the environment is not modified. Weights and internal operations are performed in float32; normalizers are saved in float64. Portable inference uses NumPy.

The training loss is the standardized output mean squared error, with equal weight per channel, plus the L2 penalty from the implementation. All records have equal weight; longer episodes contribute more to the training objective. The selection metric remains balanced NRMSE by group and episode (50% wheels, 50% arm), as in the linear baseline. Training loss and selection metric are not identical.

## Data and checkpoint selection

Same copy of the pilot `w16_r1`, same 51 training episodes / 13 validation (490,730 / 127,147 records), same partition seed, and normalization adjusted only on training data. The run verifies hashes before and after. Manifests, means, and scales are compared against the baseline, not just quantities.

Training is performed epoch-by-epoch using `partial_fit`, preserving the Adam state. Automatic row-level early stopping is disabled; after each epoch, exactly the 13 reserved validation episodes are evaluated. Weights are saved when the minimum NRMSE improves, and final evaluation is performed on those improved weights, not the last epoch's weights. A maximum of 60 epochs is set; training stops after 8 epochs without an accumulated improvement of at least 0.0001 relative to the last significant reference.  
The full curve is stored in `history.json`; each improved checkpoint is saved during training. The file is for inference and does not include the optimizer for resuming.

Validation is already used to select the epoch, so its results do not equate to independent testing. This first trial uses a single training seed; the research plan proposes multiple seeds for final conclusions. No architecture search was performed, and no seeds were selected.

## Execution

From the project with the RoboEnv environment activated:

```bash
python scripts/train_model.py \
  --dataset runs/worker_marginal_20261025/datasets/w16_r1/20261024T234048Z_worker_scaling_pilot_not_final_training_4e159d47 \
  --config experiment_specs/training/mlp_baseline.json
```

The configuration selects the model; no additional launcher or ETL was created. `learning/fitting.py` contains the specific OLS/MLP fitting. `train.py` preserves inventory, scalers, evaluation, provenance, and shared reports. The four wheels continue to be reconstructed using `[right, left, left, right]`.

Loading without scikit-learn:

```python
from robot_learning.learning.model import MLPPolicy
policy = MLPPolicy.load("runs/training/mlp_pilot_20261025/model.npz")
commands_11 = policy.predict_action(observation_21)
```

NumPy inference is verified against scikit-learn within float32 tolerance, and prediction equality is confirmed before and after saving/loading. Individual metrics per joint, side, and episode are preserved, along with the mean predictor and CSV reports. Autonomous evaluation remains pending: offline error should not be interpreted as robot success.

Sources: [MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html), [shared methodology](BASELINE_TRAINING.md), and [research plan](RESEARCH_PLAN.md).

## First measured result

Local run: `runs/training/mlp_pilot_20261025/`. Exact equality of the split/dataset manifest and the four normalization vectors was verified against the linear baseline.

| Output | MAE linear | MAE MLP | RMSE linear | RMSE MLP | RMSE predictor mean |
|---|---:|---:|---:|---:|---:|
| wheel_right (rad/s) | 0.869705 | 0.259747 | 1.099855 | 0.381739 | 1.624110 |
| wheel_left (rad/s) | 0.996895 | 0.290470 | 1.241201 | 0.412981 | 1.707517 |
| arm_torque_1 (N*m) | 0.705120 | 0.139816 | 1.111635 | 0.302606 | 2.219723 |
| arm_torque_2 (N*m) | 0.738001 | 0.190732 | 1.086311 | 0.393403 | 1.369256 |
| arm_torque_3 (N*m) | 0.455663 | 0.146095 | 0.842630 | 0.342126 | 1.269225 |
| arm_torque_4 (N*m) | 0.555533 | 0.162388 | 1.009128 | 0.304252 | 2.138022 |
| arm_torque_5 (N*m) | 0.581232 | 0.148459 | 0.920547 | 0.327430 | 1.381392 |
| arm_torque_6 (N*m) | 0.463832 | 0.122336 | 1.343797 | 0.300176 | 2.063680 |
| arm_torque_7 (N*m) | 0.000083 | 0.048657 | 0.000152 | 0.082284 | 0.690950 |

All cells assign equal weight to the 13 episodes, without mixing units across rows. MAE/RMSE of the complete mean predictor are also provided in `per_output_metrics.csv`.

Balanced NRMSE: linear 0.597467 → MLP 0.203130 (**−66.0 %**). Training MLP: 0.132817.

Finished after 22 epochs of patience; checkpoint selected: epoch 21. Adjustment + transformations + validation per epoch: 31.35 s. This measurement includes validation for the MLP; the OLS training time does not include it. Do not compare these times as equivalent work. NPZ 78789 bytes (~77 KiB); median inference 15.02 µs, p95 15.62 µs, one numerical thread, 2000 calls after warmup. Includes scaling and expansion; excludes robot reading and simulation. The historical linear measurement was 6.36 µs, from a separate run, not a paired latency benchmark.

**The improvement is not universal:** torque_7 increases from MAE ~0.000083 to 0.048657 N·m. The linear model already explained that output almost exactly; the shared MLP does not automatically preserve this relationship. Do not hide the regression with the global mean nor introduce a hybrid architecture without declaring it as a separate experiment.

The training/validation split of the MLP shows higher error outside training. There is only one seed and checkpoint selection based on validation: there is no robust estimation across seeds nor a final test. Offline improvements do not demonstrate arrival or stability in autonomous control.

[Verifiable comparison](../runs/training/mlp_pilot_20261025/comparison.json) · [Learning curve and errors](../runs/training/mlp_pilot_20261025/comparacion.png)

![Validation curve and errors by output](../runs/training/mlp_pilot_20261025/comparacion.png)

Verification: all 18 tests pass (13 from collection/simulation, 5 from learning); local log `runs/training/mlp_pilot_20261025/tests.log`.

The progression of decisions and limits, including the distinction between expert labels and geometric arrival, is preserved in the [research log](RESEARCH_LOG.md).

**Post-update of 25/10/2026:** the autonomous evaluator has already been implemented and the first comparison executed (expert 13/13, OLS/MLP 0/13 in stable arrival). Previous mentions of pending evaluation describe the state during training, not the current state. See [autonomous evaluation and P1 closure](AUTONOMOUS_EVALUATION.md).
