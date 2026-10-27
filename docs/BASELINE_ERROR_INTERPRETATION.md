# How to interpret the baseline linear error

Additional diagnosis on 25/10/2026 for the same 13 validation trajectories. No retraining was performed, and no other model was selected.

## Typical scale and fair comparison

The signed mean can approach zero when there are both positive and negative commands. Use contextual information such as the mean of absolute values, training standard deviation, and percentiles. Do not divide the error by the signed mean or by each individual command, because there are zeros.

Contextual ratio: `MAE / mean(|expert_command|)`. This is a ratio of means, not MAPE nor accuracy percentage. As a descriptive diagnostic, it measures magnitude on validation; this does not adjust the model normalizers nor modify the primary metric based on training scales.

| Output | Expert magnitude mean | MAE | MAE / magnitude | MAE predictor mean training |
|---|---:|---:|---:|---:|
| wheel_right (rad/s) | 1.62491 | 0.86221 | 53.06 % | 1.43181 |
| wheel_left (rad/s) | 1.54478 | 0.98495 | 63.76 % | 1.44098 |
| arm_torque_1 (N·m) | 1.00633 | 0.69580 | 69.14 % | 1.00650 |
| arm_torque_2 (N·m) | 18.09157 | 0.73311 | 4.05 % | 0.88185 |
| arm_torque_3 (N·m) | 0.81063 | 0.45893 | 56.61 % | 0.66877 |
| arm_torque_4 (N·m) | 1.24061 | 0.55293 | 44.57 % | 1.29306 |
| arm_torque_5 (N·m) | 0.69552 | 0.57723 | 82.99 % | 0.69750 |
| arm_torque_6 (N·m) | 2.63322 | 0.45106 | 17.13 % | 0.98488 |
| arm_torque_7 (N·m) | 0.28282 | 0.00008 | 0.03 % | 0.28168 |

Calculated table over all validation transitions combined. The main training table uses per-episode average first, so the numbers are not identical.

To maintain the same episode-weighted value as in the previous figure: wheel MAE 0.9333 rad/s versus mean magnitude 1.5717 rad/s → **59.4 %**. This does not represent the relative error at each instant. The range of expert commands observed on each side was −3 to +3 rad/s.

For torques, always compare per joint: a MAE of 0.733 N·m for torque_2 appears small relative to its mean magnitude of 18.092 N·m (4.1 %), but compared to the mean predictor (MAE 0.882 N·m), the improvement is moderate. Its training deviation is 1.392 N·m; RMSE is 1.085 N·m, equivalent to NRMSE 0.780. Thus, a torque with a large offset can have a small percentage without properly learning its variations. For torque_1, MAE 0.696 N·m versus a typical magnitude of 1.006 N·m represents 69.1 %. The global torque mean masks this difference.

## Stopping versus movement

Separation by expert command zero, absolute tolerance 1e-6 rad/s; this is a post-hoc diagnosis, not a change to training or success criterion.

| Side | Expert stopped: fraction of rows | MAE at stopping | Expert magnitude during movement | MAE during movement | Incorrect sign during movement |
|------|-------------------------------|----------------|----------------------------------|---------------------|-------------------------------|
| right | 26.8 % | 0.516 rad/s | 2.221 rad/s | 0.989 rad/s | 8.4 % |
| left | 26.8 % | 0.540 rad/s | 2.111 rad/s | 1.148 rad/s | 6.1 % |

When the expert commands zero, MAE matches the mean magnitude of the prediction. The model still outputs approximately 0.52–0.54 rad/s above expert stopped states. This indicates poor imitation of stopping; it does not alone prove that the autonomous robot exactly decelerates to that speed, since it will visit other states and contact dynamics are involved.

With wheel radius 0.11 m, 0.9333 rad/s corresponds to an error of 0.1027 m/s in tangential wheel speed. This is not directly the center base velocity error: both sides, orientation, and slip are involved.

## Interpretation

The model learns: improves over the constant predictor. However, wheel errors remain large relative to typical scales, and stopping predictions are inaccurate. This serves as a useful baseline for comparing improvements, not a validated control policy. There is no universal acceptable MAE threshold: results must be evaluated against autonomous rollouts and success/distance-to-target under a fixed protocol.

Maintain the primary metric and current split. Preserve MAE/RMSE per channel and comparison to the mean; add these scales and stopping/movement errors as diagnostics. Do not present new descriptive ratios as recorded metrics prior to the first training run.

Local evidence: [error_context.json](../runs/training/linear_pilot_20261025/error_context.json) and `contextualize_errors.py` in the same folder. [Pipeline and initial results](BASELINE_TRAINING.md).
