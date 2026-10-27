# Marginal Performance Gain with Increasing Workers

Local measurement from 25/10/2026. Expands the previous comparison, which only included 16/24/32 in the pilot. Does not modify the collector nor adopt the experimental optimizations from profiling.

## Protocol

Same 64 uniform objectives as the pilot, seed 20261023, nominal and expert start configurations unchanged. Ascending sweep: 4/6/8/12/16/20/24; then 8/6 to repeat the first increment with marginal efficiency below 50%, and 24/20 to verify the upper end in reverse order. Median is used where two trials are conducted, and the range is shown, not a confidence interval.

Coordinator time measured with a monotonic clock: includes processes, simulation, logging, saving, and batch verification; excludes initial dataset preparation. Global CPU sampled every second across the 32 logical processors; RSS is the sum of process memory and may count shared pages multiple times. Cooler turned on by Rodrigo; temperature observed without conservative clipping. No changes were made to governors, affinity settings, or permissions.

## Results

| Workers | Experiments | Median time (range), s | More workers¹ | More speed¹ | Less time¹ | CPU median | RSS max., GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 109.85 | — | — | — | 14.3 % | 1.09 |
| 6 | 2 | 84.70 (82.96–86.45) | 50.0 % | 29.7 % | 22.9 % | 20.8 % | 1.57 |
| 8 | 2 | 73.91 (72.23–75.59) | 33.3 % | 14.6 % | 12.7 % | 26.7 % | 2.04 |
| 12 | 1 | 61.40 | 50.0 % | 20.4 % | 16.9 % | 38.3 % | 2.97 |
| 16 | 1 | 52.69 | 33.3 % | 16.5 % | 14.2 % | 48.6 % | 3.90 |
| 20 | 2 | 48.46 (47.27–49.64) | 25.0 % | 8.7 % | 8.0 % | 58.0 % | 4.85 |
| 24 | 2 | 45.80 (44.40–47.21) | 20.0 % | 5.8 % | 5.5 % | 65.3 % | 5.82 |

¹ Compared to the previous row; not compared to 4 workers.

If W is workers and T is the time of the same batch:

- Worker increase = `W_new / W_old - 1`.
- Speed increase = `T_old / T_new - 1`.
- Time reduction = `1 - T_new / T_old`.
- Marginal efficiency = speed increase / worker increase. A value of 1 means proportional gain; 0.5 means half of that proportional gain.

These percentages are not interchangeable: increasing speed by 20% reduces time by 16.7%.

## Interpretation and selection

Gains decrease from low values. With the exploratory criterion of achieving at least half the proportional speed increase, the first crossover occurs between 6 and 8 workers. This criterion reflects a preference for resource efficiency, not the fastest point or a physical law. Total time continues to decrease as workers are added. The curve flattens especially after 16: subsequent increments provide less than 10% speed gain each. With 16, the average CPU usage was 49% and the maximum RSS reached 3.90 GiB.

In the upper region, going from 16 to 20 workers adds 25% of slots and provides 8.7% more speed. Going from 20 to 24 adds another 20% and provides 5.8% more speed: only 2.65 seconds of time savings for this batch.

**Final decision by Rodrigo (25/10/2026): use 16 workers as the equilibrium point locally, prioritizing available resources over additional time savings.**  
Replaces the initial proposal of 20 for this team. The exploratory rule of selecting the smallest number within 10% of the fastest time yielded 20; it is retained as an analysis result, but the agreed operational preference is 16. The portable default in the CLI remains 1.

```bash
python scripts/collect_dataset.py --config experiment_specs/collection/pilot_box.json --workers 16
```

The previous reference of 32 did not show an advantage over 24 and consumed more resources. 32 was not repeated in this sweep, and its historical times are not mixed with current ones to compute percentages.

## Checks and limits

All 11 batches completed successfully. The effective configuration hashes, scenarios, and schema match; all produced 617,877 transitions and 62 successes out of 64. These are replicas of the same dataset, not independent new data for training. This campaign compares hashes and aggregated results; it does not repeat the exhaustive matrix comparison performed in prior ablations.

One or two runs per point are insufficient to infer an exact optimum. Small percentage differences at the upper end may vary with background activity and temperature. The initial ascending order may introduce drift; reverse repetitions partially check this, but do not eliminate it. Temperature reached 100 °C; no causal percentage is attributed to temperature, frequency, cache, or SMT.

This result holds for 64 episodes, these durations, and this recording duration. Larger batches reduce the relative weight of the tail end; a different number of episodes, machine, or recording period requires recalibration. Energy or hourly cost was not measured: the marginal cost uses workers as an approximation of cost, supplemented by observed CPU and memory usage.

## Reproducible Evidence

- [Protocol and scripts](../runs/worker_marginal_20261025/): `protocol.json`, `sweep.py`, `repeat_upper.py`, `monitor.py`, logs, CSV files, and datasets.  
- [Calculated results](../runs/worker_marginal_20261025/analysis.json), generated with `analyze.py`.  
- [Previous comparison](WORKER_BENCHMARKS.md), [profiling](COLLECTION_PROFILING.md), and [ablation studies](PERFORMANCE_FACTOR_ANALYSIS.md).  

The heavy evidence under `runs/` is local and ignored by Git. The documentation preserves the summary; these changes have not been published.

![Time, acceleration, and marginal efficiency](../runs/worker_marginal_20261025/escalado_marginal.png)
