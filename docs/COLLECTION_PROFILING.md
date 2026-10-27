# Profile of Collection — 10/25/2026

## Question and Result

The queue is already dynamic: completing an episode allows launching another without waiting for the entire initial group to finish. The profiling quantifies the small reinsertion delay, the cost of checks, and the work performed by processes. No modifications were made to the expert, motor, or scheduler code during this measurement.

## Method

- Two runs of the same 64 uniform pilot scenarios, with 24 and 32 slots, headless and cooler enabled as previously instructed by Rodrigo.
- External wrappers with monotonic clocks and CPU time per process for observation/FK, base/control, `sim.Step`, logging, preparation, compression, and hashing. The coordinator records checks, summaries, writes, and launch/detection of exit events.
- The wrappers execute each original function only once. A new process is retained per episode. Instrumentation has its own overhead; the percentages are approximations from these two profiled runs, not a comprehensive statistical study.
- The stage time in workers is summed across parallel processes: it does not equate to the total elapsed time. Serial stages of the coordinator also overlap with the simulation.
- Both batches: identical configuration, scenarios, and scheme to the reference, uninstrumented, 617,877 transitions, 62 successes. The exact 32 matrices of episodes 0, 31, and 63 from both batches were verified against the reference. No exhaustive comparison of all files is claimed.

## Where Episode Times Are Spent

Percentage of accumulated time within `run_episode`, with 24 workers; excludes prior imports. These values do not include coordinator time.

| Stage | Accumulated time of 64 processes | Percentage |
|---|---:|---:|
| Physical step in PyBullet (`sim.Step`) | 492.1 s | 57.0 % |
| Arm control | 80.4 s | 9.3 % |
| Validation and array writing by Recorder | 66.1 s | 7.7 % |
| Simulator and model creation | 65.7 s | 7.6 % |
| Readings and direct kinematics | 56.7 s | 6.6 % |
| Remaining motor: commands, metrics, diagnostics, and yield | 36.7 s | 4.3 % |
| Compression and NPZ saving | 23.7 s | 2.7 % |
| Remaining worker: queue preparation, metadata, and auxiliaries | 22.4 s | 2.6 % |
| Base control | 18.7 s | 2.2 % |
| Disconnection | 0.4 s | 0.0 % |
| NPZ hashing | 0.3 s | 0.0 % |

Additionally, imports cost an average of 0.85 s per process (54.4 s accumulated), and simulator/model preparation takes 1.03 s per episode. Imports are repeated because a new process is created; independent preparation will still be necessary if a persistent pool is used in the future. Do not confuse `Popen` (very cheap) with importing libraries and creating the simulator.

## Coordinator and restart mechanism

| Metric | 24 workers | 32 workers |
|---|---:|---:|
| Total coordinator time | 45.70 s | 45.98 s |
| 128 origin checks | 13.16 s | 17.34 s |
| Summary updates | 0.33 s | 0.43 s |
| Coordinator JSON writes | 0.24 s | 0.30 s |
| Restart from detected output: average | 0.251 s | 0.551 s |
| Restart from detected output: maximum | 0.750 s | 1.336 s |
| Average delay between wrapper end and output detection | 0.215 s | 0.348 s |
| Replacements launched before final episode ends | 26 of 40 | 32 of 32 |

Replenishment is measured by matching detected free slots with new launches while work remains pending. It does not include the prior delay in detection nor the startup of the new process. The output is detected via polling, and the coordinator may be occupied performing checks, so it is not literally instantaneous.

The scheduler processes detected results in a single pass before refilling slots. There is room to anticipate this replenishment, but **there is no barrier that waits for all processes**. The capacity loss between detected output and replacement while there is a queue was approximately 0.9% with 24 and 1.2% with 32; it does not alone explain the observed speed stagnation.

## Why increasing to 32 does not accelerate

The accumulated time within episodes increased from 863.1 to 1154.5 s for the same workload: approximately a 34%. Adding 32/24 = 33% more slots is offset by individually more costly episodes under that load.

The physical step accumulated 492.1 s with 24 and 664.5 s with 32; its average per step increased from 0.796 to 1.075 ms. This demonstrates the observed process degradation, but does not causally separate frequency, temperature, SMT, cache, and bandwidth. Thermal counters and temperatures are already within the prior calibration.

Imports also increased from 0.85 to 1.08 s per process, and the same 128 checks became more expensive from 13.2 to 17.3 s. There is relevant serial work in addition to process contention. **It does not mean that removing checks would automatically save 13–17 s of total time:** much of this time occurs while workers are actively working.

## Pending queue versus end of batch

With 24 slots, the startup of the initial 24 took 2.33 s. After launching the last of the 64 episodes, 10.16 s remained until the batch completed: during this phase, no targets were pending to occupy the gaps. Process occupancy was approximately 85% in both cases with 24 and 32; this is not the same as the global CPU percentage.

In the phase with an empty queue, processes that complete cannot receive additional episodes from that batch. A launch ordering that anticipates long trajectories could reduce the final queue if a valid duration predictor is available. Neither a duration-based launch order nor modifications to stopping criteria were implemented to hide this effect.

![Trace of slot reusability](../runs/collection_profile_20261025/cola_dinamica.png)

## Priorities suggested by the data

1. Maintain 24 workers as local recommendation: 32 consumes more resources without improving time.
2. Study how to reduce origin verifications while preserving change detection and reproducibility. These are the main serial costs measured; do not remove them without equivalent replacement.
3. Reduce validation/registration work per transition if data equivalence is demonstrated. This stage accounts for approximately 8% of the internal episode time.
4. Consider persistent processes only with a verified simulator, controller, and reference reset; this would save imports and does not allow state inheritance between episodes.
5. Fine-tune recovery latency and job ordering as secondary improvements. Do not modify timestep, contacts, or expert rules to accelerate a refactor.

These optimizations were not implemented during the measurement. The production code of the collector and its rules remain unchanged; the result is a diagnosis, with evidence preserved to decide specific changes.

## Reproducible evidence

[`runs/collection_profile_20261025/`](../runs/collection_profile_20261025/) contains `profile_collect.py`, `profile_worker.py`, `analyze.py`, `plot.py`, `analysis.json`, `validation.txt`, logs, and datasets with `profiles/*.json`. The directory is ignored by Git and batches should not be automatically mixed with training. External instrumentation is not part of the original source fingerprints: these scripts are explicitly stored here to reproduce the measurement.

## Future ablations

The [factor isolation](PERFORMANCE_FACTOR_ANALYSIS.md) contrasts marginal effects of verifications, validation, compression, and affinity. Removing verifications did not show a clear improvement in final time: their accumulated cost was not solely a measure of how much they delayed the batch. Block-level validation is the observed priority candidate; full hardware attribution remains limited by telemetry.
