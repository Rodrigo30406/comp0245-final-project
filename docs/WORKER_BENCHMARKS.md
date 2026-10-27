# Worker Measurement — Update 25/10/2026

**Current decision for this team: 16 workers with the cooler enabled,  
selected by Rodrigo on 25/10/2026 after the expanded marginal sweep.**  
Previous recommendations are retained as measurement history.  
The first section preserves the initially aborted attempt; complete results and the recommendation replacing it are provided at the end.

## Initial attempt without cooler — 24/10/2026

Objective: Start with two workers and gradually increase to find a balance between performance and resource usage. The sweep was halted due to temperature; **no optimal throughput has been identified**.

## Equipment and protocol

- Intel Core i9-13900HX, 24 physical cores / 32 logical threads; 32 threads available to the process.  
- Approximately 31 GiB of total RAM and 14.8 GiB available before each test; no swap used.  
- Equipment connected to power; governor observed: `powersave`. No changes were made to power, frequencies, affinity, or ventilation.  
- Real collector running in headless mode, original rules, full step logging, and identical JSON in both runs.  
- Scheduled batch: 24 episodes, repeating six times the objectives `[0,1.1,0.5]`, `[1,0.5,0.7]`, `[-1,-0.5,0.5]`, `[0.6,0,0.8]`; nominal initial pose, seed 20261024, threshold of 40,000 steps.  
- Resource sampling every second via `/proc` and `/sys`, without installing packages. Global CPU: aggregated percentage of the 32 threads. Collector RAM: sum of RSS of coordinator and children; may count shared pages multiple times.  
- Sensor: `coretemp`, `Package id 0`; critical temperature threshold published by sysfs: 100 °C. Conservative test criterion: interrupt upon five consecutive samples of at least 98 °C, or less than 2 GiB available.  
- Interruption uses SIGINT and collector shutdown; it was verified that no active workers remained.

## Results

| Workers | Overall mean CPU | Maximum process-tree RSS | Minimum available RAM | Initial → maximum temperature | Thermal-counter increase* | Outcome |
|---:|---:|---:|---:|---:|---:|---|
| 2 | 7.7 % | 0.61 GiB | 14.36 GiB | 81 → 100 °C | +413 | Interrumpido por temperatura |
| 1 | 4.1 % | 0.37 GiB | 14.56 GiB | 69 → 100 °C | +373 | Interrumpido por temperatura |

*Lecture of `cpu0/thermal_throttle/package_throttle_count`. Its increment confirms activation of the thermal throttling mechanism; it is not interpreted as a count of episodes or independent physical events.

The test with two workers was interrupted after approximately 5.4 s and did not complete episodes; it saved partial transitions. Thereafter, there was a measured 15 s rest period, ending at around 70 °C. The test with one worker started at 69 °C, reached 100 °C, and was interrupted after approximately 5.2 s; it completed a simulated episode of 8.001 s. These truncated runs do not provide a reliable comparison of episodes per minute nor a valid extrapolation to the pilot.

## Decision and limits of the conclusion

- Do not continue with 4, 8, or more workers under these conditions: thermal saturation is already observed with one and two workers.
- Keep the default value of one worker; this does not certify thermal sustainability.
- RAM had margin and the aggregate CPU usage was low. A low global percentage does not guarantee thermal headroom: individual cores may be working intensely.
- Do not attribute a specific cause to cooling, power profile, firmware, or turbo without additional measurements. None of these options have been changed.
- Repeat the sweep only when a method to achieve thermal headroom is decided upon; then complete identical batches and compare real time, temperature, and power consumption before making a recommendation.

## Local artifacts

In [`runs/worker_scaling_20261024/`](../runs/worker_scaling_20261024/) (ignored by Git):

- `monitor.py`: self-contained measurement script.  
- `spec.json`: common configuration for the batch.  
- `w02_r1.csv`, `w01_r1.csv`: samples second by second.  
- `w02_r1.json`, `w01_r1.json`: summaries and exact commands.  
- `w02_r1.log`, `w01_r1.log`: collector output logs.  
- `cooldown.csv`: temperature during rest between tests.  
- `datasets/`: data, snapshots, and provenance of both attempts; separated from the pilot. Do not automatically include them in training.

## Continuation with cooler — 24/10 UTC / 25/10 local time

Rodrigo reported that he had turned on the cooler and indicated to proceed without concern about the previous thermal cutoff. Temperature measurement was retained, but the conservative cutoff was disabled for these batches. The 2 GiB RAM limit remained active. The controller, physics, energy parameters, affinity, and system frequencies were unchanged.

### Initial sweep: same 24 episodes

| Workers | Collector Time | Global CPU Average | Max RSS of Tree |
|--------:|---------------:|-------------------:|----------------:|
| 2      | 68.6 s        | 7.5 %             | 0.61 GiB       |
| 4      | 42.8 s        | 12.9 %            | 1.09 GiB       |
| 8      | 28.2 s        | 24.4 %            | 2.03 GiB       |
| 12     | 24.3 s        | 33.2 %            | 2.97 GiB       |
| 16     | 22.5 s        | 40.3 %            | 3.88 GiB       |
| 24     | 18.7 s        | 54.1 %            | 5.77 GiB       |

Each batch completed the same 24 scenarios, with 213,594 transitions. Since performance improved with 24 workers and there were no more than 24 episodes to test 32 active processes, confirmation was expanded to the actual pilot distribution.

### Confirmation: same 64 uniform scenarios from the pilot

The pilot recipe was used with a different name tag to maintain separate measurement: box X/Y [-2,2] m, Z [0.3,0.8] m, seed 20261023, and nominal startup. 16, 24, and 32 workers were tested; 24 and 32 were repeated in reverse order (24, 32, 32, 24).

| Workers | Trials | Median collector time | CPU median | CPU peak | RSS max | Minimum available RAM |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 51.4 s | 48.6 % | 58.5 % | 3.90 GiB | 11.29 GiB |
| 24 | 2 | 45.0 s | 66.6 % | 83.8 % | 5.75 GiB | 9.82 GiB |
| 32 | 2 | 45.5 s | 84.6 % | 100.0 % | 7.69 GiB | 8.48 GiB |

Primary time: `executions/*.json: wall_duration_s`, measured with a monotonic clock by the coordinator. Includes process launching, simulation, saving, and checks during data collection; excludes initial scenario/snapshot preparation. The external monitor, sampled every second, measured medians of 52.1 s (16), 46.1 s (24), and 46.3 s (32), with up to one second of discretization. Do not mix batches of 24 and 64 episodes in a direct time comparison.

CPU: median of the medians per trial; CPU peaks and RSS maxima: maximum observed across trials. The sum of RSS may count shared memory more than once and does not exactly equate to private RAM. Other applications' load was not controlled; these are practical measurements from the system, not an exhaustive statistical study.

### Selection

**The initial comparison between 16/24/32 selected 24 workers based on the following criterion.** A practical rule was defined: choose the smallest number tested whose median time is within 10% of the fastest, while also showing CPU and memory usage. Among 16, 24, and 32, this rule selects 24. It does not demonstrate that every integer intermediate or other hardware has the same optimal configuration.

With 32, there was no reproducible improvement in time compared to 24. Instead, the CPU median increased from approximately 67% to 85%, reaching peaks of 100%, and the RSS sum rose from 5.75 to 7.69 GiB. With 16, more resources are left for other tasks, at the cost of approximately 14% more coordination time than with 24.

```bash
python scripts/collect_dataset.py --config experiment_specs/collection/pilot_box.json --workers 16
```

The default of the program remains 1 for portability. The number 24 was an initial local recommendation, not an expert rule nor a configuration imposed on colleagues. If the duration, logging, or background load is changed significantly, re-measure.

The package temperature continued reaching 100 °C in all confirmation groups and thermal counters increased. This selection optimizes the observed performance under the authorization to continue; it does not certify the absence of thermal limitations nor attributes a specific causal effect to the cooler.

### Integrity and generated data

The five complete runs of 64 scenarios have identical configuration, scenario, and schema hashes; all produced 617,877 transitions, 617,877 s simulated, and 62 successes according to the original professor's criterion. The other two complete episodes are preserved with their results. Not all matrices were exhaustively compared across these five batches; previous scheduler tests do verify exact equality in their test cases.

These are development datasets generated for measurement, not a final training/test split. Do not concatenate the five copies as independent examples nor distribute repetitions of the same scenario across splits. If a run is reused, explicitly select a single copy and preserve its provenance.

### Artifacts with cooler

- [Scan 2–24 workers](../runs/worker_scaling_cooler_20261024/): protocol, script, CSV, summaries, logs, and datasets.  
- [Pilot confirmation](../runs/worker_scaling_cooler_pilot_20261024/): 16/24/32 worker trials, repetitions, protocol, and data.  
- [Verifiable summary](../runs/worker_scaling_cooler_pilot_20261024/analysis.json) and scripts `analyze.py` / `plot.py` in that folder.

![Comparison of time, CPU, and memory](../runs/worker_scaling_cooler_pilot_20261024/comparacion_workers.png)

## Post-profile by phase

The [profile from 25/10/2026](COLLECTION_PROFILING.md) measures physical, control, logging, compression, verification, and seat reassignment. It confirms that the queue is already dynamic and explains why 32 processes do not improve over 24 in this batch.

## Extension with intermediate values, 25/10/2026

The [marginal sweep of 4/6/8/12/16/20/24](MARGINAL_WORKER_SCALING.md)  
adds missing values in this comparison. After discussing the results, Rodrigo  
chose 16 workers as a local balance, replacing the intermediate proposal of 20. The previous selection of 24 corresponded only to the 16/24/32 values tested at that time. Refer to the new report for percentages between neighboring values, repetition ranges, and alternatives based on available resources.
