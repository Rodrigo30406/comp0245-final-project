# Joint Torque Error

Decision by Rodrigo, 25/10/2026: present each joint separately. Arm averages are secondary. The model was not retrained; this table is derived from previously saved episode-wise errors.

Validation: 13 episodes, same weights across episodes. RMSE is the average of individual episode RMSEs; not the RMSE of all rows combined.

| Joint | MAE, N·m | RMSE, N·m | MAE of mean predictor, N·m | RMSE of mean predictor, N·m | NRMSE (std training) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.705120 | 1.111635 | 1.026414 | 2.219723 | 0.481782 |
| 2 | 0.738001 | 1.086311 | 0.892290 | 1.369256 | 0.780250 |
| 3 | 0.455663 | 0.842630 | 0.666436 | 1.269225 | 0.459865 |
| 4 | 0.555533 | 1.009128 | 1.292441 | 2.138022 | 0.467866 |
| 5 | 0.581232 | 0.920547 | 0.709234 | 1.381392 | 0.624208 |
| 6 | 0.463832 | 1.343797 | 0.993643 | 2.063680 | 0.644644 |
| 7 | 0.000083 | 0.000152 | 0.277961 | 0.690950 | 0.000130 |

NRMSE and comparison with the mean predictor complement the physical units; a large expert mean may hide poor variation fitting. The baseline for each joint returns its training mean torque, without consulting validation.

The pipeline now saves `per_output_metrics.csv` for all outputs and both partitions, with episode-wise and transition-wise aggregations, without unit mixing. The JSON also preserves output-wise errors per episode and now includes per-channel/episode aggregations in new runs.

For the first run, the CSV was added from its historical JSON without modifying the model, split, original metrics, or their checksums. The CSV is a derived report; new runs include its hash.

[CSV local](../runs/training/linear_pilot_20261025/per_output_metrics.csv) · [Scale context](BASELINE_ERROR_INTERPRETATION.md) · [Pipeline](BASELINE_TRAINING.md)
