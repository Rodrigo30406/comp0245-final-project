# Performance factor analysis and proposed improvements — 25 October 2026

## Main Result

**Block-level validation is the most promising software improvement identified.** Maintaining 24 workers, with compression and provenance checks, remains the current operational proposal. No optimizations have been applied to the production code.

Marginal effects of specific interventions were measured. **An exact causal partitioning of the slowdown among frequency, temperature, power, SMT, cache, and memory could not be obtained.** These factors interact, and authorized counters are missing. A percentage breakdown for each factor as if summing to 100% is not valid.

## Protocol and Limits

- Eleven complete runs of the same 64 scenarios and rules from the pilot; 24 workers during ablation phases and 32 during the final comparison. The cooler is turned on according to user instruction, without the previous conservative thermal cutoff; a 2 GiB RAM guard is maintained.
- One factor was changed per condition. Three references distributed across the sequence, with repetitions of the two primary diagnostic ablations. The block, compression, and affinity variants each undergo a single trial; these effects are not statistically confirmed.
- The references took 46.13, 48.52, and 47.90 seconds: median 47.90 seconds, range of 2.39 seconds (~5%). Small differences in other conditions may be due to load, temperature, or frequency variations. The order was not randomized, and fixed frequencies were not maintained.
- Identical external instrumentation was used across phases, adding overhead. The original expert code, physics, timestep, stopping criteria, sampling, and array precision did not change.
- In variants without repeated verifications, a full source integrity check was performed before and after the entire batch. These do not equate to the original collector’s episode-level detection guarantees and are not proposed for production use.

## Software: What Actually Changes the Total Time

Times are medians of the coordinator per condition. Percentage changes use the median of the three references; they are not confidence intervals.

| Condition | Tests | Time | Observed Change | Interpretation |
|---|---:|---:|---:|---|
| Reference | 3 | 47.90 s | +0.0 % | Current production |
| Without repeated checks | 2 | 46.28 s | -3.4 % | Diagnostic only; preserves guards before/after the batch |
| Without row validation | 2 | 43.91 s | -8.3 % | Diagnostic only; does not adopt guarantee elimination |
| Block validation (256 records) | 1 | 44.89 s | -6.3 % | Prototype with validation and NaN testing |
| NPZ without compression | 1 | 46.63 s | -2.6 % | Same matrices, larger size |
| One thread per physical core affinity | 1 | 47.40 s | -1.0 % | Affinity intervention for this process and its children |

### Provenance Checks

The 128 provenance checks previously consumed 13–17 s of the coordinator in the prior profile, but they overlap with simulation. Removing them resulted in 45.82 s and 46.73 s. The difference is small relative to the reference range; there is insufficient evidence that removing them produces a meaningful reduction in final time. This corrects the priority that would suggest focusing solely on their accumulated cost.

Proposal: retain them. Optimizing this cost would be reasonable if the batch size changes or if CPU usage by the coordinator is to be reduced, but it is not the first intervention for accelerating this workload. Any future cache must detect code/assets changes and preserve provenance; it should not replace checks with a fixed fingerprint return.

### Validation and Logging

Removing row-level validation resulted in 43.80 s and 44.02 s, but eliminates guarantees and serves only as a diagnostic. The realistic variant maintains per-row field checks and validates floating-point array finiteness in blocks of 256 records, plus the final fraction before writing the NPZ.

The accumulated CPU time in `record_append` was 67.98 s in the references versus 16.50 s with blocks; the vectorized validations recorded 0.31 s and partially overlap with that timer. This represents a reduction of nearly 75% of the CPU cost in that stage, not of the entire program.

The batch with blocks took 44.89 s, 6.3% less than the reference median. Compared to each individual reference, the observed savings range approximately from 3% to 8%. There is only one trial per batch: it is a candidate for adoption after further validation, not a guarantee of fixed acceleration.

Prototype tests: buffer growth, 257 rows, NaN rejection at full block and partial flush; no invalid NPZ is written. Explicit semantic difference: an invalid value can be detected up to 255 records later. Immediate checks of the motor's state and action remain active. If implemented, document this delay and test interrupts, errors, and restarts; do not assume that block-level detection and instant detection are identical.

### Compression

The uncompressed NPZ produced 421.2 MiB versus 280.9 MiB compressed: approximately 50% more space. The time was 46.63 s, within the observed variation of references. Proposal: maintain compression; there is no clear gain justifying this increased size.

## Hardware: what could be separated

### CPU consumption time versus time outside CPU

Consecutive comparison of `baseline_c` (24) versus `default32` (32), same workload and without software ablations. Previous imports prior to `run_episode` are excluded. These are summed parallel process times, not batch clock time.

| Accumulated measure over 64 episodes | 24 workers | 32 workers |
|---|---:|---:|
| Time spent within episodes | 898.54 s | 1216.08 s |
| CPU time of processes | 898.23 s | 1173.90 s |
| Time outside CPU | 0.31 s | 42.18 s |

Of the accumulated increase of 317.54 s, 86.8% corresponds to additional CPU time and 13.2% to time outside CPU. This shows that the slowdown is not merely waiting for a scheduler turn. Time outside CPU includes scheduling, I/O, blocking, and timing errors; it is not a pure counter of CPU wait time. The kernel exposes `sched_schedstats=0`, so a reliable separation of these waits could not be obtained.

### Frequency: observation, not causal intervention

`scaling_cur_freq` was sampled by CPU every second and weighted by activity per core type. On this system, P cores correspond to eight physical cores with SMT, and E cores to sixteen without SMT.

| Approximate CPUFreq weighted | 24 (`baseline_c`) | 32 | Change |
|---|---:|---:|---:|
| P | 2175 MHz | 1967 MHz | -9.6 % |
| E | 1899 MHz | 1783 MHz | -6.2 % |

The observed drop is consistent with reduced process speed, but does not quantify what fraction of the slowdown is attributable to it. The samples are not tied to instructions from each episode; task placement changes, and the clock was not fixed. Documentation of [kernel CPUFreq](https://docs.kernel.org/admin-guide/pm/cpufreq.html) warns that `scaling_cur_freq` may reflect the requested frequency rather than the actual hardware frequency. Do not use these MHz values to compute a residual cache or SMT causality percentage.

### SMT Availability and Core Selection

Intervention: same collector and 24 workers, but affinity limited to one logical thread per physical core: `0,2,4,6,8,10,12,14,16-31`. Affinity affects only the benchmark process and its descendants; it does not disable SMT at the system level nor restrict other applications.

The result was 47.40 s, within the reference range. There is no evidence that this policy improves execution time. It also does not demonstrate a zero SMT effect: scheduler options, P/E mix, and coordinator placement change. Proposal: maintain normal affinity for this workload, without introducing an affinity option as yet justified for optimization. 

### Cache, Bandwidth, Power, and Temperature

- `perf_event_paranoid=4`: `perf stat` denied access to cycles, instructions, and cache events. No IPC, cache misses, or bandwidth measurements were obtained.  
- `turbostat` could not open MSRs nor access `CAP_SYS_RAWIO`; RAPL `energy_uj` also returned a `PermissionError`. The configured power limits that can be read are not representative of actual consumption.  
- Temperature and thermal counters show throttling activity, but no counterfactual was measured under controlled frequency/power/temperature conditions. It is not valid to assign a specific percentage to this effect.  
- Available RAM was not exhausted. This rules out capacity exhaustion in these batches, but does not rule out bandwidth saturation or cache pressure.  
- No changes were made to permissions, sysctls, governor, power limits, frequencies, or BIOS. Access to counters is subject to the restrictions described in the [perf security documentation](https://docs.kernel.org/admin-guide/perf-security.html).

Additional microarchitectural attribution would require authorized access to counters (at least `CAP_PERFMON` according to scope), measurement of cycles/instructions/failures by P/E class, and accessible hardware telemetry APERF/MPERF and RAPL. Afterward: experiments with controlled frequencies and policies, repeated runs/randomized order, and interaction analysis. Do not distribute the remaining unknown across factors by assumption.

## Prioritized Proposals

| Priority | Proposal | Evidence / Adoption Condition |
|---------|---------|-------------------------------|
| 1 | Vectorized block-level validation, preserving schema and final flush | Prototype took 44.89 s, showing strong CPU reduction in register and NaN testing. Block repetition and interrupt/handling/restart need to be repeated before production. Document error latency. |
| 2 | Maintain 24 workers, normal affinity, and compression | 32 does not improve and consumes more; physical affinity and raw NPZ do not show clear gains. Recalibrate if workload changes. |
| 3 | Evaluate persistent pool with new simulator/controller per episode | Previous imports were ~0.85 s per process in the prior profile. This intervention was not measured; require resets, trajectory equivalence, and RAM stability. Do not reuse expert state. |
| 4 | Optimize validations only if they become critical | Their removal did not yield clear gains currently. Do not degrade reproducibility for assumed savings. |
| Investigation | Authorized hardware telemetry and factorial design | Required to reliably separate frequency/SMT/cache/power effects; currently inaccessible/incomplete. |

Do not change timestep, contacts, limits, or expert precision in order to improve performance: that would be another task variant, not an equivalent optimization. The savings from these interventions should not be summed: they may share or shift the bottleneck.

## Equivalence and reproducibility

The eleven runs completed the 64 scenarios with 617,877 transitions and 62 successes each. All 32 episode matrices across the eleven conditions were compared against `baseline_a`: 22,528 array comparisons, identical shapes and dtypes, all finite floats. Configuration, scenarios, and scheme also match, and all pre- and post-experiment source guards were passed.

The scripts are observational/experimental wrappers stored under [`runs/factor_isolation_20261025/`](../runs/factor_isolation_20261025/), ignored by Git. It includes the protocol, original script versions, resource CSVs and frequencies, profiles, data, `analysis.json`, `array_validation.json`, `block_validation_tests.txt`, and `access_limits.json`. Variants that omit guarantees are for diagnostic purposes only, not for training configuration collection. Do not concatenate replicas or cross them between splits.

![Ablations and CPU versus wait times](../runs/factor_isolation_20261025/factores.png)
