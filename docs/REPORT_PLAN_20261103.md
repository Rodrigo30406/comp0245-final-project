# Report plan (3 Nov 2026, 21:40): the storyline on mixture v2, the ablations, and the generalisation / robustness / efficiency programme

Written after the lecturer's feedback of 3 Nov ("the basic problem is defined; raise the level, propose something new")
and the brief's evaluation slide (accuracy, **generalisation to new targets / initial conditions**, **robustness to
adversities**, **efficiency**; marking: 70–79 = correct, reproducible, understood; 80–100 = that plus well-justified novel
ideas in the method or the experimental validation). Report due Friday 6 Nov. Everything below is pre-registered before
its result; the reserved test banks stay untouched until the final table.

**Budget.** One trainer at a time on the GPU: MLP on v2 (35 M rows) 30–40 min per seed, GRU 15–25 min, transformer
10–15 min; one closed-loop evaluation of 205 geometries 12–15 min with 8 workers, three at a time; collection 10 min per
1024 episodes with 28 workers. Chains: `chain_collect_story.sh` (CPU), `chain_story_gpu.sh` (GPU), `chain_ood_adv.sh`
(evaluations), logs in the scratchpad, every result appended to this file and to the README as it arrives.

## 1. The storyline: one line of research on mixture v2, each step an ablation

Mixture v2 (§6z) is the data of the whole report: 3 221 expert-success episodes, 819 nominal start + 2 806 random start
with the goal referenced to the start pose, 205 held-out nominal geometries and 205 random-start geometries for
selection. Version 1 appears only where v2 cannot show the point (the first failures) and is labelled as such. Each
chapter states a claim, the measurement that supports it, and the v2 run that makes it reproducible.

| # | chapter | claim | v1 evidence (exists) | v2 runs | status 3 Nov 15:40 |
|---|---|---|---|---|---|
| 1 | **Baseline: the lecturer's inputs, one MLP** | the supervised error is small and the robot does not drive | 0 / 13 and 0 / 820 (`BASELINE_LIMITATIONS.md`, `PATIENCE_PARTIAL_REPORT.md`) | `v2_mlp22raw_control` ×3 (21 inputs + clock, no transforms, no device) | **queued** (GPU chain step 2) |
| 2 | **How much data, sampled how** | more data lowers the offline error and does not make the robot arrive; stratified targets reduce the divergence | 64–2048 uniform vs stratified (`SAMPLING_SERIES.md`, `DATASET_SIZE_COMPARISON.md`) | size series with the final recipe: `v2_mlp28_trig144_cap{400,800,1600,2400}` (fixed-rng subsets of the 3 221 successes, seed 0) against the full 3 221 | **queued** (step 5). Uniform vs stratified stays a v1 result: a uniform v2 would be a different geometry set, not an ablation (Rodrigo's call) |
| 3 | **Pure transforms of the observations** (the black-box rule) | the target in the robot's frame unlocks the base and the arm; no privileged information | §6e, `PRIVILEGED_ABLATION.md` (38% of the error is unobserved state) | `v2_mlp22raw_control` vs `v2_mlp28_control` (exists: 1 / 0 / 2 at 1 cm, 41 / 31 / 39 at 5 cm); the device without the transforms `v2_mlp22raw_trig144` (seed 0) | **queued** (steps 2, 4) |
| 4 | **Noise injection at collection (DART)** | without it the policy leaves the demonstrated states within a second; the expert's trajectories are too ideal to cover its own neighbourhood | `NIGHT_RUN_20261027.md`, `MECHANISMS_20261027.md` | clean twins of the v2 collections (same scenarios and seeds, no noise) → `runs/splits/v2_clean_mixture.json` → `v2_mlp28clean_trig144` ×3, `v2_mlp28clean_control` ×3 | **collecting** (CPU chain), then GPU step 3 |
| 5 | **The two coupled failures** | (a) a regressor's zero is a residual that creeps the base; (b) the arm starts only when already moving (one resting-arm reach row per 12 000) | §6c–§6i, §6t.1 | measured on `v2_mlp28_control` (§6z.1, probe −0.02, creep) | done |
| 6a | **Stop remedies, one at a time** | an exact zero alone does not help with a sleeping arm | E7 / E8 (§6l–§6w) | dead band (evaluator), categorical wheels, gate: all re-run on v2 (§6ac) | done |
| 6b | **Ignition, one at a time** | the ignition row is the missing piece; K = 144 the chosen weight | K sweep 16 / 144 / 512 on v1 (§6y.1) | `v2_mlp28_trig144` ×3 (done), weight 144 (done), chunking (done); `v2_mlp28_trig16`, `v2_mlp28_trig512` (seed 0) | **queued** (step 4) |
| 6c | **Memory instead of a device** | a GRU ignites without the device and stops creeping; it tops out at 121–151 | E4 / E8b | GRU sweep + patience 40 (§6ab), GRU + weight (§6aa.2, §6ab.2); transformer on v2 `v2_tf28_p20` ×3 | **training** (GPU step 1) |
| 6d | **Combinations** | with the arm awake, an exact zero gives the expert's parking | — | K 144 + band, K 144 + categorical (§6ac.1), GRU + weight + patience (§6ab.2) | done |
| 7 | **Final comparison, then the reserved banks once** | MLP 28 + K 144 (+ band or categorical) vs the best pure GRU vs transformer | — | `final_table.py`; reserved banks after Rodrigo's band-vs-categorical decision | pending |
| 8 | **Generalisation, robustness, efficiency** | §2–§4 below | — | E11, E12, E13 | **running / queued** |

**Offline metrics for chapter 1 (Rodrigo, 3 Nov).** The NRMSE is the RMSE of each output divided by the standard deviation
of that output over the training labels, averaged half over the two wheels and half over the seven torques (episode-macro:
one value per validation episode, then the mean). So 0.13 means 13% of the spread of the demonstrated command, which is
small, and the report must show the scale: for the v2 control, wheels RMSE 0.19 rad/s against commands up to ±5.27 rad/s
(std 1.53), torque 2 RMSE 0.25 Nm against ±30 Nm (mean-baseline RMSE 1.91 Nm); per-joint and per-wheel errors are in every
`metrics.json` (`per_output_episode_macro`, units rad/s and N·m), and `PER_JOINT_ERRORS.md` has the v1 table. The chapter
then shows that the same policy arrives 0 times, and Figure 2c (offline NRMSE vs arrivals, inverted across models) closes it.

## 2. E11, generalisation beyond the training distribution (pre-registered 3 Nov 21:30, collecting)

Three evaluation banks of 210 episodes (5 held back by the split format, 205 evaluated), expert recorded on each so the
expert row is the reference, no model ever trains on them (`experiment_specs/collection/ood_*.json`,
`scripts/analysis/make_bank_split.py`, `runs/splits/bank_*.json`):

| bank | what moves outside the training range (targets x, y ∈ [−2, 2], z ∈ [0.3, 0.8]; starts x, y ∈ [−2, 2]) | prediction |
|---|---|---|
| `ood_far_targets` | targets with max(\|x\|, \|y\|) > 2.2 m, up to 3 m; nominal start, goal from the origin | the bearing features are in range, the planar-distance feature is not (2.2–4.2 m vs ≤ 2.8 m): the base drives in the right direction, the approach and the stop degrade; the expert itself may time out (25 s at 0.25 m/s) on the farthest |
| `ood_target_height` | z ∈ [0.85, 1.05] (half) and z ∈ [0.15, 0.28] (half); nominal start | high: the standoff is the training one (0.65 m) and the arm must extrapolate upward, so the base parks and the arm loses the last centimetres (1 cm falls more than 5 cm); low: the standoff grows to 0.72–0.78 m, farther than any training park, and the base error dominates |
| `ood_far_start` | start x ∈ [2.3, 3.5], y ∈ [−3.5, 3.5], any yaw; targets in the training box; goal from the start pose | the drive is bearing-driven and survives; the clock input is out of its training phase on the long drives (the policy arrives later than any demonstration) so the stop and the ignition shift; arrivals fall moderately, more for the MLP than for the GRU if the clock is what the MLP leans on |

Evaluated: K 144 ×3, K 144 + band, GRU patience 40 ×3, categorical + K 144 ×3; same 1 / 2 / 5 cm metrics, arm / base
decomposition and stop profile. The interesting output is where each policy breaks, not only the count.

## 3. E12, robustness to adversities (pre-registered 3 Nov 21:30, running)

Adversities are keys of the frozen evaluation protocol (`src/robot_learning/evaluation/adversity.py`,
`experiment_specs/evaluation/validation_<adversity>.json`), applied to the expert and the policies alike from the scenario
seed, so the expert row is the reference under the same adversity. On the 205 nominal validation geometries:

| protocol | adversity | prediction |
|---|---|---|
| `delay50ms` | the executed command is the one issued 50 ms earlier (zeros before the first) | the expert's base P-controller tolerates it; the arm PD (kp 100, kd 30) with a 50 ms delay oscillates at the target, so everything, expert included, loses at 1 cm and keeps 5 cm; the GRU (20 ms memory) is not helped: it has no model of the delay either |
| `gain_left095` | the left wheels execute 95% of the command (calibration error) | the expert corrects through its heading loop (calibration: 3 / 3 targets reached at 0.95; 0.92 fails 1 / 3, 0.85 fails 2 / 3, so 0.95 is the largest error the expert survives); the policies see the heading error in their inputs and should also correct, but park off-spot, losing 1 cm arrivals |
| `payload1kg` | 1 kg added to the end-effector link, unknown to the expert's gravity compensation | a steady sag of a few millimetres for the expert and the policies alike: 1 cm arrivals fall, 2 cm stay |
| `obsnoise` | N(0, s) on what the policy reads: base x / y 2 cm, bearing 0.02 rad, arm angles 0.01 rad, arm velocities 0.05 rad/s; the target is never noised; the expert reads the simulator state (clean reference) | the stop depends on the base pose to 8 cm, so 2 cm noise costs little parking; the arm-velocity noise (0.05 rad/s) is of the order of the trigger jitter (0.004–0.085), so the K 144 MLP, trained with jittered ignition rows, should keep igniting, while the pure GRU's ignition may fire early or late; creep returns if the noise pushes the raw zero above the band |

Each on K 144 ×3, K 144 + band (both devices candidates for the final policy), GRU patience 40 ×3.

**Excluded with a measurement: the base push.** Implemented and unit-tested (`base_push`), calibrated on the expert
3 Nov 21:00: sideways, any push (down to 30 N for 0.3 s during the approach) makes the base slide indefinitely with zero
wheel commands, because the wheel contact's lateral friction is 0.01 (`PhysicsParameters.anisotropic_friction`), and the
expert ends 3–9 m from the target; along the heading the velocity-controlled wheels absorb 100–150 N for 0.5 s (1–9 cm of
displacement, no effect on the arrival). A lateral push therefore measures a simulator artefact and a longitudinal one
measures nothing; neither is used.

## 4. E13, efficiency (to tabulate from the run directories; one script, no new runs)

Parameters, inference latency per observation (NumPy, single observation, median / p95 over 2 000 calls, already in each
`model_info.json → inference_single_observation_us`), training time and epochs, data-size series (chapter 2):

| policy | parameters | inference per step (µs) | training (GPU) |
|---|---:|---:|---|
| MLP 28 pure, K 144 (`v2_mlp28_trig144_seed0`) | 21 385 | 145 | 36 min, 35 epochs |
| MLP 28 categorical 41 + K 144 | 31 705 | 151 | |
| GRU 28 patience 40 (`seed2`) | 81 929 | 33 (memory every 20 ms) | 15–24 min, 213–308 epochs |
| transformer 28 (v1 E5) | 81 993 | 30 | 8–13 min |
| gated GRU (E8a) | 10 986 | 92 | |

A 1 ms control period leaves 1 000 µs; every policy fits with margin on one CPU core. To add: the training-time column for
every v2 run and the arrivals-vs-episodes curve from chapter 2.

## 5. Queue (launched 3 Nov 21:50; one trainer at a time, evaluations three at a time)

GPU chain, in order: transformer v2 ×3 (patience 20, success filter added to the trainer today) → MLP 22 raw control ×3 →
[waits for the clean split] MLP 28 clean + K 144 ×3, MLP 28 clean control ×3 → K 16, K 512, raw + K 144 (seed 0 each) →
size series caps 400 / 800 / 1600 / 2400 (seed 0). About 9–10 GPU hours. CPU chain: three OOD banks (10 min) → clean
e1 (1024) and e3 (3072) collections (40 min) → splits. Evaluation chain: the four adversities × three model groups on the
nominal validation (12 evaluations), then the three banks × four model groups (12). Every evaluation of the GPU chain runs
on both v2 validations.

## 6. The transformer branch, where it stands

E5 (v1 mixture 4096, §6j): the best offline fit of the pure-input models (0.130–0.142) and the cleanest open-loop stop
(residual 1–2%), yet 81 / 76 / 47 at 5 cm and 25 / 29 / 8 at 1 cm: it parks on the standoff shell and restarts in 105–148
episodes, by the same creep as everything else, only larger (11–17% residual). The dead band made it worse (§6t: 68 / 40
at 5 cm) because its arm needed the jostle; its stop basin is flat (§6s). It has never been trained on v2. Today's run
(`v2_tf28_p20`, 3 seeds, patience 20) answers whether, like the GRU, the v2 data lets it ignite on its own. If it does, the
one thing worth adding is the trigger weight for the sequence trainer (the GRU's `sequence_trigger_rows` is in
`gru_policy.py`; the transformer trainer has no such option: about an hour of work plus three trainings).

## 7. Results (appended as they arrive)


### 7.1 E11, two of the three banks (4 Nov 04:40; `ood_far_start` still evaluating)

Stable arrivals at 1 / 2 / 5 cm on the 205 evaluated geometries of each bank, expert row = the same geometries with the
expert (`runs/evaluation/ood_<bank>_<group>_20261103`):

| bank | expert | K 144 (3 seeds) | K 144 + band 0.45 | categorical + K 144 | GRU 28, patience 40 |
|---|---|---|---|---|---|
| `ood_far_targets` (2.2–3 m) | 205 / 205 / 205 | **205 / 205 / 205**, 196 / 205 / 205, 197 / 205 / 205 | 202 / 203 / 204, 198 / 205 / 205, 205 / 205 / 205 | 178 / 196 / 205, 184 / 196 / 204, 194 / 205 / 205 | 111 / 139 / 160, 61 / 128 / 189, 118 / 186 / 205 |
| `ood_target_height` (0.15–0.28 and 0.85–1.05 m) | 205 / 205 / 205 | 18 / 64 / 157, 41 / 63 / 77, 16 / 84 / 129 | 22 / 72 / 159, 35 / 61 / 75, 17 / 75 / 112 | (running) | 16 / 57 / 131, 6 / 26 / 83, 14 / 46 / 114 |

- **Far targets: the prediction was wrong in the good direction.** The planar-distance input is 2.2–4.2 m, outside anything
  demonstrated (≤ 2.8 m), and the MLP does not care: 196–205 at 1 cm, the expert's own count. The drive is bearing-driven
  (cos / sin of the heading error) and the stop is decided by the target-in-base-frame features, which are in range once the
  robot is near; the out-of-range distance only has to say "far, keep driving". The GRU loses (61–118 at 1 cm vs 121–151
  nominal): its hidden state integrates a longer drive than any it was trained on, and the stop fires late or early
  (5 cm 160–205). This is the clearest generalisation result of the programme and it separates the two architectures.
- **Target height: both halves fail, as predicted, and the failure is in the arm.** Split by half (seeds 0 / 1 / 2 of
  K 144, 103 high and 102 low geometries): high 12 / 31 / 3 at 1 cm and 69 / 52 / 47 at 5 cm; low 6 / 10 / 13 and
  88 / 25 / 82. The base parks in 39–79 of 205 (vs 145–175 nominal) and the arm extends in 185–200, so the base loses its
  spot and the arm, asked for a configuration outside the demonstrated 0.3–0.8 m band, reaches the 5 cm shell in 77–157 and
  the 1 cm one almost never. The GRU is no better (6–16 at 1 cm). Height is the dimension the demonstrations do not cover,
  and no input transform can supply the missing arm configurations: this is the honest limit of the dataset, and the
  remedy (collecting demonstrations at those heights) is a data decision, not a model one.

### 7.2 E12, first round (4 Nov 04:40): two adversities calibrated wrong, two results

| adversity | expert | K 144 (3 seeds) | K 144 + band | GRU 28, patience 40 | what happened |
|---|---|---|---|---|---|
| `gain_left095` | 182 / 182 / 183 | **199 / 201 / 203**, 194 / 200 / 203, 180 / 198 / 199 | 194 / 197 / 198, 189 / 203 / 203, 191 / 197 / 199 | 133 / 157 / 175, 107 / 142 / 190, 150 / 202 / 205 | the policies beat the expert: the expert's stall / timeout logic gives up on 23 geometries where its curved drive is slow, the policies have no timeout and keep correcting through the heading inputs; GRU unchanged from nominal |
| `obsnoise` (2 cm, 0.02 rad, 0.01 rad, 0.05 rad/s) | 205 (clean reference) | 7 / 13 / 16, 8 / 12 / 15, 5 / 9 / 11 | 7 / 12 / 16, 11 / 15 / 16, 4 / 8 / 9 | **127 / 181 / 195, 149 / 178 / 187, 119 / 175 / 198** | the MLP is thrown: arm torques clipped 33–67% of the time, joint-limit violations, final distance 10–148 m (the robot is flung by chattering torques); the GRU keeps 187–198 at 5 cm with final distances 0.1–1 m and no clipping. Memory filters the noise; a 1 kHz stateless regressor amplifies it |
| `payload1kg` | 154 / 170 / 183 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | the expert's PD absorbs 1 kg (final 0.23 m mean); every policy saturates the arm torques (clipping 70–99%), leaves the demonstrated arm states and is flung (final 10–1700 m). Level too high for a torque regressor with no feedback law of its own; **re-run at 0.1 / 0.25 / 0.5 kg** |
| `delay50ms` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | the **expert itself** explodes: its 1 kHz torque loop with 50 ms of latency oscillates (clipping 37–74%, 2 km final distance). The prediction (base tolerates, arm loses 1 cm) was wrong by an order of magnitude; **calibrating on the expert at 1 / 2 / 5 / 10 / 20 ms** before re-running |

Predictions scored: gain (policies lose 1 cm arrivals) wrong, they gained; noise (K 144 keeps igniting, GRU suffers)
inverted: the GRU is the robust one and the MLP's failure is not in the ignition but in the torque chatter; payload and
delay: level mis-set, no policy result yet. The noise result is the one that changes the story: it is the first axis on
which the pure GRU beats the device MLP outright, and it is the robustness axis the brief asks for.

Second round queued (`chain_adv2.sh`, after the first chain finishes): payload 0.1 / 0.25 / 0.5 kg, observation noise at
one tenth and one half of the above, and the delay levels the expert survives, each on K 144 ×3 and GRU patience 40 ×3
(`experiment_specs/evaluation/validation_payload0p{1,25,5}kg.json`, `validation_obsnoise_{tenth,half}.json`).

### 7.3 E14, GPU chain: the first two groups evaluated (4 Nov 05:20)

Stable arrivals at 1 / 2 / 5 cm, nominal validation and the v2 random-start bank (205 geometries each), 3 seeds:

| run | offline NRMSE | nominal 1 cm | nominal 5 cm | random 1 cm | base parked / arm extended (nominal) |
|---|---|---|---|---|---|
| transformer 28, patience 20, no device (`v2_tf28_p20`) | 0.106 / 0.109 / 0.111 | **166** / 85 / 104 | 185 / 137 / 182 | 153 / 73 / 113 | 169–184 / 103–149 |
| MLP, 21 raw inputs + clock, no transforms (`v2_mlp22raw_control`) | 0.169 / 0.151 / 0.158 | 8 / 0 / 0 | 67 / 18 / 16 | 4 / 0 / 0 | 93–137 / 22–169 |
| (reference) GRU 28, patience 40, no device | 0.105–0.108 | 138 / 121 / 151 | 177 / 193 / 205 | 118 / 103 / 140 | |
| (reference) MLP 28 + K 144 | 0.135–0.141 | 199 / 193 / 175 | 202 / 204 / 199 | 200 / 197 / 187 | 145–175 / 180–199 |

- **The transformer ignites on v2.** Trained on expert successes only (the filter it lacked until yesterday) it parks in
  169–184 episodes and extends the arm in 103–149, with no label-derived device: seed 0 gives 166 at 1 cm, more than any
  GRU seed, seeds 1 and 2 give 85 and 104 (GRU 121–151). Final distance 0.07–0.22 m mean, drift after parking 0.07–0.18 m.
  Its limiting factor is the ignition (arm extended 103–149 vs base parked 169–184), the same coupling failure as every
  pure model, only less frequent. Next step, queued with no code change (`chain_tf_p40.sh`, after the story chain): patience
  40, three seeds, the step that gave the GRU +10 / +61 / +1. The trigger weight for the sequence trainer stays as the
  option if that is not enough.
- **Chapter 1 on v2 is confirmed.** The lecturer's 21 inputs plus the clock fit worst offline (0.15–0.17) and reach 0–8
  at 1 cm; the base parks in 93–137 episodes but 0.6–1.5 m from the expert's spot (final distance 0.8–1.9 m), and the
  arm extends in 22–169. Without the target expressed in the base frame the regressor cannot place the stop; the 7
  transforms (chapter 3) are what fix it, not the data.

### 7.4 E13, efficiency (4 Nov 05:20, `scripts/analysis/efficiency_table.py`)

| run | model | parameters | trainer-measured µs median / p95 | in-loop µs (evaluator median) | training min | epochs (best) | val NRMSE |
|---|---|---:|---:|---:|---:|---|---:|
| `v2_mlp22raw_control_seed0` | MLP, 22 inputs | 20 617 | 108 / 115 | | 47 | 40 (32) | 0.169 |
| `v2_mlp28_trig144_seed0` | MLP, 28 inputs, K 144 | 21 385 | 145 / 154 | 82 | 36 | 35 (27) | 0.139 |
| `v2_mlp28_cat41_trig144_seed0` | MLP, categorical wheels, K 144 | 31 705 | 151 / 164 | | 49 | 35 (27) | 0.149 |
| `v2_gru28_patience40_seed2` | GRU, memory every 20 ms | 81 929 | 33 / 43 | 78 | 15 | 253 (213) | 0.108 |
| `v2_tf28_p20_seed0` | transformer, memory every 20 ms, 20 s window | 81 993 | 47 / 154 | 154 | 25 | 192 (172) | 0.106 |
| `v2_gated_trigw144_seed0` | gated GRU, trigger weight 144 | 10 986 | 31 / 37 | | 4 | 65 (57) | 0.244 |

The trainer latencies were measured on a loaded machine (the same MLP reads 35 µs on an idle one, 145 µs beside two
evaluators), so the in-loop column is the fair one: 80 µs for the MLP and the GRU, 150 µs for the transformer, all inside
the 1 000 µs control period on one CPU core with NumPy. The MLP trains in 25–50 min on the 35 M rows; the sequence
models in 15–25 min on 3 221 episodes. The full table over every v2 run: `python3 -I scripts/analysis/efficiency_table.py`.
