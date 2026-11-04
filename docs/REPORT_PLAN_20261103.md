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


### 7.1 E11, the three banks (4 Nov 04:40, far start added 06:45)

Stable arrivals at 1 / 2 / 5 cm on the 205 evaluated geometries of each bank, expert row = the same geometries with the
expert (`runs/evaluation/ood_<bank>_<group>_20261103`):

| bank | expert | K 144 (3 seeds) | K 144 + band 0.45 | categorical + K 144 | GRU 28, patience 40 |
|---|---|---|---|---|---|
| `ood_far_targets` (2.2–3 m) | 205 / 205 / 205 | **205 / 205 / 205**, 196 / 205 / 205, 197 / 205 / 205 | 202 / 203 / 204, 198 / 205 / 205, 205 / 205 / 205 | 178 / 196 / 205, 184 / 196 / 204, 194 / 205 / 205 | 111 / 139 / 160, 61 / 128 / 189, 118 / 186 / 205 |
| `ood_target_height` (0.15–0.28 and 0.85–1.05 m) | 205 / 205 / 205 | 18 / 64 / 157, 41 / 63 / 77, 16 / 84 / 129 | 22 / 72 / 159, 35 / 61 / 75, 17 / 75 / 112 | 31 / 58 / 113, 14 / 39 / 110, 22 / 63 / 151 | 16 / 57 / 131, 6 / 26 / 83, 14 / 46 / 114 |
| `ood_far_start` (start 2.3–3.5 m out, any yaw) | 205 / 205 / 205 | **200 / 200 / 200**, 195 / 204 / 204, 201 / 205 / 205 | 196 / 196 / 196, 198 / 204 / 204, 204 / 205 / 205 | 150 / 165 / 175, 177 / 189 / 202, 198 / 202 / 205 | 85 / 106 / 132, 59 / 104 / 180, 103 / 188 / 205 |

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

- **Far starts (06:45): the K 144 MLP generalises again, 195–201 at 1 cm, and the clock prediction was wrong.** The
  drives are 1.7–6 m (training ≤ 2.8 m) and the robot arrives later than any demonstration, yet the base parks in 181–196
  and the arm fires in 203–205: the stop and the ignition are read from the target-in-base-frame and arm features, not
  from the clock. The GRU falls to 59–103 (its ignition: arm active 128 in seed 0; its parking is fine, 181–200), the
  categorical to 150–198. Two geometries defeat every MLP seed and the band alike: starts at the far corners
  (2.63, 3.47, yaw −2.36) and (3.06, −3.27, yaw −2.93), facing away from the target. There the heading error is near π,
  cos ≈ −1 and sin ≈ 0, so the sign of the turn is undetermined and the demonstrations contain almost no such rows at
  4–5 m; the policy drives straight away for 68–159 m. That is the one failure mode of the far-start bank and the first
  drive-away since v2. The remaining 1–7 misses per seed have the base 3–14 cm from the goal and the arm 0.13–0.46 m
  short: an arm that fires from an approach direction it was not shown.
- **Categorical wheels on the height bank** (06:45): 31 / 14 / 22 at 1 cm, 113 / 110 / 151 at 5 cm, like the others.

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

### 7.5 Two metric questions from Rodrigo (4 Nov 06:10): seed spread, and what "moving the arm" means

**"Arm extended" was mis-defined and under-counted the expert.** The decomposition used the end-effector displacement in
the base frame between the first and last trace sample, > 10 cm. Measured on the nominal validation: the expert's reach
from its standoff moves the end effector as little as 2.8 cm (percentiles 0 / 1 / 5 / 25: 2.8 / 3.4 / 4.9 / 9.8 cm), so
the expert itself counted 151 of 205; while a sleeping arm jostled by the base moves 3.0–3.2 cm, so the two populations
overlap and a 10 cm threshold is arbitrary. In joint space they do not overlap: the largest joint excursion
max_t max_j |q_j(t) − q_j(0)| is ≥ 0.195 rad in every expert episode (nominal and random banks) and ≤ 0.076 rad for every
sleeping arm (the DART-trained hold wobbles 0.053–0.058 rad). New rule, now in `arm_base_decomposition.py`: **arm active
= largest joint excursion > 0.1 rad** (about 6°, in the middle of the gap). Re-counts on the nominal validation:

| policy | old rule (EE > 10 cm) | arm active (q > 0.1 rad) | base parked |
|---|---|---|---|
| expert | 151 | 205 | 205 |
| MLP 28 + K 144 (3 seeds) | 189 / 199 / 180 | 204 / 204 / 203 | 145 / 175 / 150 |
| MLP 28 + K 144 + band | 186 / 199 / 176 | 203 / 205 / 202 | 147 / 161 / 136 |
| categorical + K 144 | | 205 / 205 / 205 | 112 / 137 / 142 |
| GRU 28 patience 40 | 133 / 184 / 158 | 170 / 200 / 205 | 132 / 170 / 146 |
| transformer 28 patience 20 | 147 / 103 / 149 | 174 / 109 / 155 | 169 / 184 / 171 |
| MLP 28 control (no device) | 4 / 1 / 1 | 6 / 1 / 1 | 123 / 137 / 116 |
| gated stop (E8a control) | | 164 / 149 / 202 | 22 / 114 / 70 |

Consequences for the text: the device MLPs ignite in 203–205 of 205, not 180–199; the pure GRU's shortfall at 1 cm is
split between ignition (35 asleep in seed 0, 5 in seed 1, 0 in seed 2) and parking (132–170), and the transformer's is
ignition (109–174). One earlier reading changes: the gated-stop control was described as "closes before the zero but does
not ignite the arm"; by the joint rule its arm does leave home in 149–202 episodes, and its failure is the base (parked
22–114): the latch closes at the wrong place. The 1 cm counts, the metric that selects models, are unaffected. The
`BEST_MODEL_REPORT` table is corrected; §6 tables of the blackbox plan keep the old column and say so.

**Is the seed spread normal?** Yes, and it is a property of closed-loop imitation, not of our training. Offline the three
K 144 seeds are within 4% of each other (NRMSE 0.135–0.141); closed loop they span 175–199 at 1 cm. The failures are not
shared: of the 39 geometries where some K 144 seed fails, 31 fail in exactly one seed and 1 in all three (GRU: 134 / 78 /
15; transformer: 158 / 75 / 19). So there is no "hard geometry" set; each seed draws its own stop and ignition boundary
and a few episodes land on the wrong side of it, and once a decision is wrong the episode is lost (the error compounds,
there is no recovery signal in the demonstrations). A regressor's 4% fit difference becomes a 12% arrival difference
because the arrival hinges on two sharp events learned from one row per episode each. The sequence models spread more
(GRU 59–74%, transformer 41–81%) because their decisions depend on a hidden state trained on 3 221 sequences rather than
35 M rows. This is why every number is reported per seed, and why the final policy should be chosen on the validation
geometries (seed included: that is model selection) and confirmed once on the reserved banks. Ensembling the three seeds
(mean of the outputs) is the standard remedy and would be a one-line evaluator option; not done.

**Gated stop re-read (4 Nov 06:30, `v2_gated_control` on the nominal validation).** With the joint rule the arm leaves home
in 164 / 149 / 202 episodes, at 10–15 s median. The failure is where the latch fires. (a) Early and far: in 41 / 56 / 3
episodes the arm stays asleep and the base is 0.36 / 1.43 / 0.31 m (median) from the expert's goal, so the arm never sees
a stop state. (b) Slightly off: where the arm is active but the base is not "parked" (142 / 73 / 133 episodes) the base
sits 8 / 12 / 5 cm from the goal and the arm, reaching from a pose it was not demonstrated at, lands 13 / 8 / 5 cm from the
target; 5 cm arrivals among them 72 / 39 / 116, 1 cm almost none. Seed 2 shows the ceiling of this design: base within
5 cm median (90th percentile 14 cm) gives 180 at 5 cm and 107 at 1 cm. Because the latch is irreversible there is no
correction afterwards; the K 144 MLP, whose base ends 10–25 cm off in most episodes too, keeps a small wheel command and
the arm converges with it (175–199 at 1 cm). So the earlier sentence "the gate closes the base but does not ignite the
arm" becomes: the gate ignites the arm in 73–99% of episodes, but freezes the base at a point chosen once, and from the
wrong point the arm cannot finish. The coupling diagnosis of §6w stands; its mechanism is the stop point, not the ignition.

### 7.6 E12, second round (4 Nov 10:45): one adversity was an artefact, noise is a dither, delay has no margin

Calibrated levels from §7.2, K 144 ×3 and GRU patience 40 ×3 on the 205 nominal geometries, expert under the same
condition (except noise, which it does not read). Stable arrivals 1 / 2 / 5 cm per seed.

| adversity | expert | K 144 seeds 0 / 1 / 2 | GRU 28 patience 40 seeds 0 / 1 / 2 |
|---|---|---|---|
| none (reference) | 205 / 205 / 205 | 199 / 193 / 175 at 1 cm | 138 / 121 / 151 at 1 cm; 177 / 193 / 205 at 5 cm |
| payload 0.1 kg on the end-effector frame | 204 / 204 / 204 | 0 / 0 / 0 (all tolerances) | 0 / 0 / 0 |
| payload 0.25 kg, same | 201 / 201 / 201 | 0 / 0 / 0 | 0 / 0 / 0 |
| payload 0.5 kg, same | 196 / 198 / 198 | 0 / 0 / 0 | 0 / 0 / 0 |
| observation noise, one tenth (2 mm, 0.002 rad, 0.001 rad, 0.005 rad/s) | 205 (clean) | 34 / 43 / 61, **103 / 131 / 141**, 10 / 14 / 15 | 149 / 172 / 189, 143 / 186 / 204, 148 / 202 / 205 |
| observation noise, one half (1 cm, 0.01 rad, 0.005 rad, 0.025 rad/s) | 205 (clean) | 10 / 20 / 26, 17 / 23 / 28, 4 / 6 / 7 | **199 / 204 / 204, 184 / 201 / 202, 177 / 200 / 204** |
| observation noise, full (§7.2) | 205 (clean) | 7 / 13 / 16, 8 / 12 / 15, 5 / 9 / 11 | 127 / 181 / 195, 149 / 178 / 187, 119 / 175 / 198 |
| command delay 2 ms | 204 / 205 / 205 | 0 / 0 / 1, 0 / 0 / 0, 0 / 1 / 1 | 0 / 0 / 0 ×3 |
| command delay 5 ms | 129 / 152 / 186 | 0 / 0 / 0 ×3 | 0 / 0 / 0 ×3 |

**Payload: the adversity, not the policies, was broken.** 0.1 kg and 0.5 kg give the same 0 / 205, which is not how a
payload behaves, and the expert's own arm is abnormal under it: torques at the 30 N·m limit in 26% of the sampled steps
(median episode; 0% nominally), joint speeds up to 29 rad/s (0.75 nominally), while it still parks and reaches (204 /
201 / 196) because its base loop never reads the arm. The mechanism, measured with `$S/payload_mech.py` on one
geometry, 15 s each: the URDF link `robot_j2s7s300_end_effector` has **no inertial block**; PyBullet loads it as a
1 kg body with a 6.7e-7 kg·m² inertia (its own warning says so). Calling `changeDynamics` on that link with the **mass
unchanged** (1.0 → 1.0 kg) already saturates the expert's torques in 14 979 of 15 000 steps (nominal: 23) with joint
speeds of 33 rad/s; adding 0.1 kg changes nothing further; giving the frame a realistic inertia does not cure it
(14 937 / 15 000). Hanging 0.1 or 0.5 kg on `robot_j2s7s300_link_7`, the last link with an inertial block, leaves the
expert exactly nominal (26–28 saturated steps, 8 rad/s, reaches at 10 mm). So the second-round payload rows measure
"a chattering 1 kg phantom appears at the hand", not a payload. What they do show, again, is the decoupling: the expert
tolerates its own arm chattering at the limit because the wheels are a separate controller; the clones, whose 11 outputs
come from one network, follow the arm into saturation (torques at the limit 86% of steps, joint speeds at the 100 rad/s
cap, wheels at the limit 94% MLP / 73% GRU, final distances 2–1500 m). Fix queued (`chain_payload_link7.sh`, after the
GPU chains, because their evaluators import `src/` live): `adversity.py` gets `PAYLOAD_LINK = robot_j2s7s300_link_7`,
and 0.1 / 0.25 / 0.5 kg are re-run as `adv_payload*_link7_*_20261104`. The `payload1kg` row of §7.2 and the three rows
above are **not** payload results and will not be reported as such.

**Observation noise: the GRU's curve is not monotone, and the reason is the ignition.** GRU at 1 cm: nominal 138 / 121
/ 151 → one tenth 149 / 143 / 148 → one half **199 / 184 / 177** → full 127 / 149 / 119. Two effects pull in opposite
directions, both visible in the decomposition:

| GRU patience 40, seeds 0 / 1 / 2 | nominal | noise 1/10 | noise 1/2 | noise full |
|---|---|---|---|---|
| arm active (joint rule) | 170 / 200 / 205 | 181 / 205 / 205 | 205 / 205 / 205 | 205 / 205 / 205 |
| base parked (8 cm box, 0.5 s) | 132 / 170 / 146 | 109 / 167 / 131 | 65 / 78 / 42 | 68 / 87 / 49 |
| stable 5 cm | 177 / 193 / 205 | 189 / 204 / 205 | 204 / 202 / 204 | 195 / 187 / 198 |
| stable 1 cm | 138 / 121 / 151 | 149 / 143 / 148 | 199 / 184 / 177 | 127 / 149 / 119 |

Noise on q / qd **wakes the GRU's arm**: the sleeping-arm episodes (35 in seed 0 nominally) disappear at one half of
the level. That is the ignition problem of §4 solved by dither instead of by the device: a stateless reading of "arm at
rest" is exactly what the recurrent model latches on, and 0.005 rad of jitter breaks the latch. At the same time noise
on the base pose degrades parking monotonically (132 → 109 → 65 → 68 parked in seed 0), and at the full level the
parking loss outweighs the ignition gain. At one half both halves of the task still hold, hence the peak. The MLP K 144
has no ignition to gain (arm active 204–205 already) and only the loss: its torques chatter and clip 14–47% of the
time at **one tenth** of the level (0.001 rad on q, 0.005 rad/s on qd), 39–52% at one half, and the wheels follow
(at the limit 19–67% of steps). Seed 1 is the exception (103 at one tenth), the other two collapse. The lesson for the
report: the 1 kHz stateless torque regressor is a marginally stable arm loop that amplifies any input perturbation,
and the GRU's 20 ms state is a low-pass filter that both tolerates the noise and profits from it.

**Delay: the expert has no margin either, and the clone inherits that plus the coupling.** With 2 ms of command delay
the expert's arm loop already chatters (torques at the limit 44% of sampled steps, joint speeds 3.9 rad/s against 0.75
nominal; mean |qd| during the drive 1.0 rad/s against 0.02) and it still arrives 204 / 205 because the base is a
separate loop; at 5 ms it loses 76 at 1 cm (129 / 152 / 186) and the hand hunts (final 10 mm median). Both clones are
at 0 / 205 from 2 ms. The MLP blows up (torques at the limit 74%, joint speeds at the 100 rad/s cap, wheels at the
limit 85%). The GRU does **not** blow up: its arm chatters like the expert's (mean |qd| 0.79–1.0 rad/s, torques at the
limit 37%) and its wheels keep driving at 1.5–2.5 rad/s, but the base never parks (0–1 of 205) and the hand never comes
within 5 cm (0–2 episodes): the navigation head reads an arm state (|qd| ≈ 1 rad/s during the drive) that in the
demonstrations only occurs during the reach, and the drive does not converge. Delay is therefore not a graded
robustness test of the clone: the expert's own gains leave zero delay margin at 1 kHz, the clone copies that loop, and
loses in addition the separation that lets the expert finish anyway.

**What goes to the report (Table III).** Gain 0.95: policies ≥ expert. Noise at three levels: MLP collapses from one
tenth, GRU peaks at one half with the ignition explanation. Delay 2 / 5 ms: expert 204 / 129, clones 0, with the
no-margin sentence. Payload: pending the link-7 re-run; until then only the artefact note. Prediction scoring for the
round: "payload 0.1–0.5 kg graded" wrong (artefact); "lower noise helps the MLP" wrong except one seed; "GRU degrades
monotonically with noise" wrong, it improves first; "delay 2 ms survivable for the policies because the expert is
clean there" wrong, the expert is not clean there, it only looks so in the arrival count.
