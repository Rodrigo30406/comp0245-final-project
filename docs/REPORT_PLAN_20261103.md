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

**Gain 0.95, the expert's 23 failures re-read (4 Nov 12:30, `$S/gain_timeout.py`).** Rodrigo asked whether those
failures are the 25 s navigation timeout, since the report blamed "stall and timeout logic". They are not. In the 23
episodes the timeout fires with the base still 2–3 m from its goal (path 5.9 m, the drive curves under the asymmetric
wheels), the arm then reaches from there and the expert declares "not reachable" at 28 s. Expert-only re-run of the
same 23 scenarios (same seeds, gain 0.95) with `nav_timeout` raised to 38 s: parks in **0 of 23**, closest approach of
the hand 0.09–0.46 m (identical minima to the 25 s run, which reproduced the original failures exactly), and the final
distance gets worse, 2.4–3.9 m, because the drive keeps curving away after its closest pass. So the expert's navigation
never satisfies its 8 cm / 0.05 rad parking test with a 5% wheel asymmetry on these geometries; the timeout only
decides where it gives up. The policies win (180–203) because they have no parking test to satisfy. `nav_timeout`
stays at 25 s everywhere (demonstrations and evaluation), as in `AGENTS.md`; raising it for evaluation would be a
declared variant of the expert and, on this evidence, would not change the count.

### 7.7 E14 closed (4 Nov 13:10): the whole GPU chain evaluated on the final mixture

Seed 0 unless three numbers are given; stable arrivals 1 / 2 / 5 cm on the 205 nominal geometries, then on the 205
random-start geometries.

| run | nominal | random start | reading |
|---|---|---|---|
| raw 21 + clock, no device (×3) | 8 / 21 / 67, 0 / 1 / 18, 0 / 0 / 16 | 4 / 15 / 57, 0 / 0 / 17, 0 / 0 / 12 | chapter 1 stands on the final data |
| raw 21 + clock + 144 copies | 15 / 53 / 117 | 26 / 46 / 101 | the ignition device needs the egocentric inputs: same device, 15 against 199 |
| egocentric + 16 copies | 102 / 144 / 195 | 101 / 146 / 196 | K too small: the onset row still loses to the resting rows |
| egocentric + 144 copies (×3, reference) | 199 / 193 / 175 | 200 / 197 / 187 | |
| egocentric + 512 copies | 200 / 204 / 204 | 202 / 205 / 205 | plateau from 144; the "fires early" effect seen on the earlier collection does not appear here |
| 144 copies, 400 training episodes | 142 / 165 / 191 | 137 / 154 / 183 | |
| 800 | 148 / 193 / 200 | 127 / 172 / 194 | |
| 1 600 | 177 / 193 / 201 | 174 / 194 / 204 | |
| 2 400 | 195 / 198 / 202 | 187 / 200 / 202 | |
| 3 221 (all) | 199 / 193 / 175 | 200 / 197 / 187 | 1 cm keeps climbing to the full set; 5 cm saturates at 800 |
| 144 copies, **no DART** (clean mixture, 4 000 episodes, ×3) | 0 / 0 / 0 ×3 | 0 / 0 / 0 ×3 | |
| no device, no DART (×3) | 0 / 0 / 0 ×3 | 0 / 0 / 0 ×3 | |

Three things for the report. (1) The transforms and the device are not additive: raw + device gives 15, egocentric
without device 1, egocentric + device 199. The device makes the onset row learnable; the egocentric inputs make the
rest of the task learnable from any start pose (raw + device on the random bank ends 0.42 m from the base goal, median,
after 2.6 m of path: it drives to the wrong place). (2) DART is a condition, not an option: without the injected
noise at collection even 4 000 clean episodes with the device give 0 arrivals; the policy leaves the demonstrated
states within the first seconds and has no recovery rows. (3) Data size: at 1 cm the curve is still rising at 3 221
(142 → 148 → 177 → 195 → 199), at 5 cm it is flat from 800 episodes; the 1 cm precision is what the data buys. K: 16
→ 102, 144 → 199, 512 → 200 at seed 0, so the device saturates by 144.

Queued (`chain_nominal_only.sh`, after the transformer patience-40 chain): the same policy trained on the nominal-start
successes alone (721 episodes), 3 seeds, both banks. That is the one data ablation the report keeps, at Rodrigo's
request (4 Nov): success-only training and the start-referenced goal are stated as conditions of the dataset, and the
with / without random-start comparison is the ablation.

### 7.8 Transformer at patience 40 (`chain_tf_p40.sh`, done 4 Nov 14:47)

Same recipe as `v2_tf28_p20` (28 inputs, stride 20 ms, window 20 s, 8 episodes per batch, success filter), early
stopping at patience 40 instead of 20, three seeds. Stable arrivals of 205, nominal validation / v2 random-start bank:

| run | epochs | offline NRMSE | nominal 1 / 2 / 5 cm | random 1 / 2 / 5 cm | base parked / arm active (nominal) |
|---|---|---|---|---|---|
| `v2_tf28_p40_seed0` | 172 | 0.106 | 166 / 173 / 185 | 153 / 160 / 176 | 169 / 174 |
| `v2_tf28_p40_seed1` | 128 | 0.109 | 85 / 100 / 137 | 73 / 103 / 128 | 184 / 109 |
| `v2_tf28_p40_seed2` | 259 | 0.103 | 169 / 187 / 197 | 179 / 185 / 199 | 124 / 183 |
| (patience 20, same seeds) | 47 / 52 / 61 | 0.106 / 0.109 / 0.111 | 166 / 85 / 104 | 153 / 73 / 113 | 169–184 / 103–149 |

- Patience 40 is worth +0 / +0 / +65 at 1 cm: seeds 0 and 1 stopped at the same minimum (the patience-20 stop was
  already the best offline fit), seed 2 trained 4× longer and gained 65 arrivals. Same picture as the GRU (+10 / +61 / +1):
  the offline minimum is flat and which side of it the stop falls on decides the closed-loop result by 60 arrivals.
- Ignition is still the limiting factor: seed 1 ignites the arm in 109 episodes and arrives in 85; seeds 0 and 2 ignite in
  174 / 183 and arrive in 166 / 169. The base parks more often than the arm extends in seeds 0 and 1 (169 / 184 vs
  174 / 109), the coupling failure of every model without the device.
- Report: TF-Ego now means this run (patience 40, like GRU-Ego); Table II row 166 / 85 / 169 and 185 / 137 / 197;
  §III-D updated. `final_table.py` registers `v2_tf28_p40_on_clean_validation_20261104`. Seeds 3 and 4 of the
  transformer (`chain_seeds34.sh`, relaunched 14:55) also train at patience 40 so the five seeds share the recipe.

### 7.9 E12 payload on the last arm link (`chain_payload_link7.sh`, done 4 Nov 16:33)

`adversity.py` patched and committed: the payload goes on `robot_j2s7s300_link_7` (the last link with an inertial block),
not on the massless end-effector frame. The mechanism check of §7.6 reproduced before the runs: +0.1 kg on the
end-effector frame saturates the expert's torques in 14 937 of 15 000 steps, +0.1 kg on link 7 in 26 (nominal 23), and
the expert reaches the target in all three cases. The `adv_payload*_20261104` rows (end-effector frame) stay excluded.

Stable arrivals at 1 / 2 / 5 cm, nominal validation bank, expert and three seeds each:

| payload | expert | MLP-Ego-IC (K 144) 1 cm | 2 cm | 5 cm | GRU-Ego 1 cm | 2 cm | 5 cm |
|---|---|---|---|---|---|---|---|
| none | 205 | 199 / 193 / 175 | 202 / 198 / 204 | 202 / 204 / 199 | 138 / 121 / 151 | 160 / 154 / 198 | 177 / 193 / 205 |
| 0.1 kg | 204 | 201 / 199 / 188 | 202 / 201 / 203 | 203 / 203 / 204 | 141 / 109 / 157 | 159 / 147 / 200 | 177 / 190 / 205 |
| 0.25 kg | 205 | 191 / 195 / 153 | 199 / 204 / 204 | 203 / 205 / 204 | 121 / 80 / 119 | 161 / 137 / 188 | 187 / 190 / 205 |
| 0.5 kg | 205 | 50 / 36 / 37 | 186 / 196 / 184 | 202 / 200 / 199 | 69 / 18 / 64 | 145 / 122 / 157 | 191 / 190 / 203 |

Closest approach of the hand to the target (`minimum_distance_m`), three seeds pooled (615 episodes):

| policy | payload | 10th / 25th / median / 75th / 90th percentile (mm) | < 10 mm | 10–20 mm | > 20 mm |
|---|---|---|---|---|---|
| MLP-Ego-IC | none | 1.6 / 2.5 / 3.8 / 5.5 / 8.6 | 570 | 30 | 15 |
| MLP-Ego-IC | 0.25 kg | 2.4 / 3.9 / 5.5 / 7.0 / 8.8 | 574 | 34 | 7 |
| MLP-Ego-IC | 0.5 kg | 3.3 / 5.2 / 7.3 / 10.0 / 14.6 | 460 | 124 | 31 |
| GRU-Ego | none | – / 2.2 / 5.7 / 13.1 / – | 416 | 101 | 98 |
| GRU-Ego | 0.5 kg | – / 4.5 / 8.6 / 17.4 / – | 346 | 143 | 126 |

- **The load is a steady-state offset, not a loss of the task.** With 0.5 kg the MLP still arrives at 5 cm in 199–202
  and at 2 cm in 184–196, and the closest approach moves from 3.8 to 7.3 mm at the median and from 8.6 to 14.6 mm at
  the 90th percentile: 124 episodes bottom out between 1 and 2 cm, which is the 1 cm column falling from 175–199 to
  36–50. That is what a torque clone does under an unmodelled load: the arm torques were demonstrated for the unloaded
  arm, the policy has no integral action, and the extra gravity moment leaves a residual proportional to the mass
  (about 2 mm per 0.25 kg at the median). The expert's loop closes it (204 / 205 / 205; its one failure is at 0.1 kg).
- **GRU-Ego carries the same offset on top of its parking variance**: median closest approach 5.7 → 8.6 mm, 1 cm
  arrivals 138 / 121 / 151 → 69 / 18 / 64, 5 cm unchanged (190–203). Seed 1, the worst parker, loses most.
- At 0.1 kg both models are unchanged from nominal (MLP 201 / 199 / 188, GRU 141 / 109 / 157). The 0.1 kg MLP
  evaluation was killed at 14:57 by the clean-up of the aborted nominal-only chain (my error: three evaluators were
  running, one was this one) and re-run alone 16:40–17:12.
- Report: Table III gets the payload rows (expert / MLP-Ego-IC / GRU-Ego at 1 cm) and the adversity paragraph the
  offset explanation; the artefact sentence stays as one line.

### 7.10 Nominal starts only: the random-start condition measured (`chain_nominal_only.sh`, done 4 Nov 17:09)

MLP-Ego-IC (K 144) trained on the DART nominal collection alone, `runs/splits/dart_low_geometry_819_205.json`: 819
training geometries, 721 expert successes kept, same 205 validation geometries as every other row. Offline NRMSE 0.179 /
0.183 / 0.180 (mixture: 0.135–0.141), best epoch 23 / 15 / 22. Stable arrivals 1 / 2 / 5 cm:

| policy (3 seeds) | nominal bank | random-start bank | base parked nominal / random | drift after parking (median) nominal |
|---|---|---|---|---|
| nominal starts only (721 ep.) | 178 / 115 / 170; 194 / 148 / 184; 199 / 195 / 193 | **40 / 58 / 62**; 55 / 91 / 78; 84 / 127 / 84 | 85–124 / 7–29 | 0.10 / 1.11 / 0.10 m |
| full mixture (3 221 ep.) | 199 / 193 / 175; 202 / 198 / 204; 202 / 204 / 199 | 200 / 197 / 187; 203 / 203 / 196; 204 / 203 / 199 | 145–175 / 133–169 | 0.06 / 0.25 / 0.14 m |

- **From nominal starts the nominal-only policy still works** (178 / 115 / 170 at 1 cm, 193–199 at 5 cm): the 721
  episodes carry the geometry of the task. What they do not carry is the stop from any pose: on the random-start bank
  the hand passes within 5 cm in 84–129 episodes but the base parks in 7–29, and the median final distance is 1.0–2.1 m,
  the robot reaches and then drives on. Even from nominal starts it parks in 85–124 episodes (full set 145–175) and seed
  1 drifts 1.1 m after parking: fewer stop rows, weaker latch.
- So the random-start repeats buy two things: the task from any pose (40–62 → 187–200 on the random bank) and a firmer
  stop on the nominal starts themselves. With the no-DART result (0 everywhere) this closes the two measured
  conditions; success-only training is stated as a condition and not measured, at Rodrigo's request.
- Report: Table II row "MLP-Ego-IC, nominal starts only" 178 / 115 / 170 and 199 / 195 / 193; §III-C paragraph.
  `final_table.py` registers `v2_mlp28nom_trig144_on_clean_validation_20261104`.

### 7.11 The expert's 129 nominal failures are not the timeout (4 Nov 19:03, `scripts/analysis/expert_failures_timeout.py`)

Rodrigo: the data section still said the expert "times out at 25 s". Measured on the 129 failures of the nominal DART
collection (12.6 % of 1 024), re-run with the same DART noise realisation (same `(noise seed, scenario seed)` stream,
same `np.random.seed`) and `nav_timeout` 38 s instead of 25 s; three 25 s controls reproduce the recorded minimum
distance exactly.

| group (from the metadata) | n | at 25 s (collection) | at 38 s |
|---|---|---|---|
| navigation timed out: base circling 10–50 cm from its goal, arm then "closest pose reached" from ~2 m | 110 | min distance median 0.28 m, final 2.0 m | parks in **0**, min distance unchanged in all 110 (|Δ| < 5 mm), final distance median 4.5 m |
| navigation done in 0–3 s (target within reach of the start), hand 1–6 mm away, expert's stop test declares "closest pose reached" | 19 | min distance median 3 mm | unchanged (independent of the timeout); 14 of 19 hold 1 cm for 0.5 s by our metric |

- The timeout decides when the expert gives up, not why it fails: the base never satisfies its 8 cm / 0.05 rad parking
  test on these geometries under the DART wheel noise (it circles at a fixed radius, so more time only adds distance),
  the same mechanism as the gain-0.95 failures (§7.6). A successful drive takes at most 11 s (99th percentile), so 25 s
  was already 2.3× that and 38 s changes nothing.
- The 19 arm cases are the expert's own termination under torque noise, a label failure at 1–6 mm; they are also dropped
  by the success filter (they would teach "closest pose reached" as a stop 1 cm short).
- Report: data section (ii) rewritten with these numbers; no sentence attributes a failure to the timeout.
  `nav_timeout` stays 25 s everywhere (AGENTS.md).

**Far targets vs far starts (Rodrigo, 4 Nov 19:20): are they the same axis?** Measured on the bank metadata, base-frame
range start→target: training successes (nominal + random starts, 3 701) median 2.0 m, max 4.9 m, 594 above 3 m, 81
above 4 m; `ood_far_targets` 2.3–4.1 m (median 3.0; 104 of 210 above 3 m), start at the origin; `ood_far_start`
0.8–6.7 m (median 3.7; 88 above 4 m, **32 beyond anything demonstrated**), start 3.5–4.8 m from the origin, any yaw.
So far targets is **not** out of distribution in the base frame (the random-start repeats already cover it); it is a
world-frame test (parking 2.3–4.1 m from the origin, raw pose inputs outside the trained box). Far starts is the true
long-range test plus the novel start pose. The report said "planar range up to 4.2 m against 2.8" for far targets: that
2.8 was the nominal set's maximum, wrong once the random-start repeats are in the training set; paragraph rewritten and
the Fig. 2 caption ranges corrected (starts 3.5–4.8 m, not 2.3–3.5). The GRU's loss on far targets, where the drive
length is demonstrated, says its loss is not the length alone (its state also reads the raw world pose).

### 7.12 Seeds 3 and 4 of the three main models (`chain_seeds34.sh`, done 4 Nov 19:52)

Same recipes as seeds 0–2 (MLP-Ego-IC K 144; GRU-Ego patience 40; TF-Ego patience 40, so the five transformer seeds
share the recipe). Stable arrivals at 1 / 2 / 5 cm of 205, both banks, pooled with `seed_stats.py`:

| policy | bank | seeds 0 / 1 / 2 / 3 / 4 at 1 cm | 1 cm mean ± sd (range) | 5 cm mean ± sd (range) |
|---|---|---|---|---|
| MLP-Ego-IC | nominal | 199 / 193 / 175 / 193 / 185 | 189.0 ± 9.3 (175–199) = 92 ± 5 % | 201.0 ± 2.0 (199–204) |
| MLP-Ego-IC | random start | 200 / 197 / 187 / 199 / 180 | 192.6 ± 8.7 (180–200) = 94 ± 4 % | 202.0 ± 2.0 (199–204) |
| GRU-Ego | nominal | 138 / 121 / 151 / 150 / 186 | 149.2 ± 23.9 (121–186) = 73 ± 12 % | 196.8 ± 12.2 (177–205) |
| GRU-Ego | random start | 118 / 103 / 140 / 133 / 168 | 132.4 ± 24.5 (103–168) | 194.4 ± 13.6 (172–205) |
| TF-Ego | nominal | 166 / 85 / 169 / 172 / 200 | 158.4 ± 43.2 (85–200) = 77 ± 21 % | 185.0 ± 27.9 (137–205) |
| TF-Ego | random start | 153 / 73 / 179 / 179 / 197 | 156.2 ± 49.1 (73–197) | 181.6 ± 31.9 (128–203) |

- Offline NRMSE of the new seeds: MLP 0.134 / 0.140, GRU 0.108 / 0.105, TF 0.104 / 0.103, all inside the ranges of
  seeds 0–2. The new seeds did not change the MLP's picture (sd 9 of 205) and widened the memory models' upward: GRU seed
  4 reaches 186 and TF seed 4 reaches 200 at 1 cm (arm active 205, base parked 142; its parking is the MLP's), so a
  transformer seed can equal the device. The seed spread of the sequence models (sd 24 and 43) is 3–5× the MLP's.
- Report: Table II rows marked "(5 seeds)"; abstract (GRU 59–91 %), §III-D (ranges over five seeds; the device is
  worth 30–40 arrivals per seed on top of memory, not 50), conclusion. Table III and Fig. 2 stay at seeds 0–2
  (the adversity and bank evaluations were run on those). `weights/v2/` gains the six new runs and the five
  patience-40 transformers; `final_table.py` registers the `_s34_` evaluations.
- Queue after this chain: empty. GPU and CPU free.

## 8. Step 9: the obstacle task with a 2D lidar (Rodrigo, 4 Nov 20:30; night chain)

Rodrigo: a teammate's branch `feature/step9-obstacle-lidar` adds a harder task and a new expert; re-collect data with
that expert and train our most robust models on it, overnight. What the branch holds (`final/robotnik_arm_task/`,
plan `STEP9_PLAN_PROMPT.md`): one static cylinder (r 0.25–0.35 m, 1.5 m tall) between the start and the docking point,
start pose in a 2 m disc with any yaw, target 3.5–5 m from the start; the expert v2 drives two segments with the
UNCHANGED template base controller (segment 0 to a waypoint beside the obstacle, segment 1 the A-to-B docking),
nav_timeout 45 s, step cap 60 s, a robot-obstacle contact ends the episode; inside the ambiguous band (|lateral offset|
< 0.2 m) the detour side is a fair coin (deliberately multimodal demonstrations). The sensor is an analytic 64-beam
planar lidar in the base frame (beam 0 forward, ccw, clipped at 4 m), `obs["lidar"]`, the only obstacle information a
learned policy gets. Their DART noise (v 0.02 m/s, w 0.05 rad/s, torque 2 N m, held 0.3 s, fading near the goals) on
50 % of the episodes; sets train2 (seed 3, 2000 episodes), dev2 (seed 300, 300 clean episodes, every decision),
confirm2, test2 (sealed). Their harness `obstacle_task/evaluate.py` (protocol v2.0: T_MAX 60 s, contact = failure,
success = EE within 10 mm at some moment, docked_clean, collision, passed_obstacle, side_taken, and v1.1's stable
arrivals) runs any policy with `reset(target, info)` / `act(obs)`. Their learners L0/L1/L2 (sklearn MLPs on
ego + lidar features) reached success 60 % on dev2 (L2 seed 0; expert 100 %).

Integration chosen (no edits to their code; worktree `~/repos/comp0245-step9`, robot models linked from RoboEnv):
1. **Collection with their collector and expert** (`obstacle_task.collect`, 16 workers): train2 and dev2.
   Smoke: 2 pilot2 episodes reached in 20–24 s simulated.
2. **Conversion** to this project's dataset layout (`scripts/obstacle/convert_step9_dataset.py`): observation 21 =
   base xyz, bearing, 7 arm q, 7 arm qd, target; action 11 = 4 wheels + 7 torques (clean expert labels, same-side
   identity checked); plus a per-row `lidar` (N, 64) recomputed from the logged base pose and the episode's obstacles
   with the vendored ray caster (checked equal to `obstacle_task/lidar.py` on 500 poses); `success` = reached, no
   contact, navigation not timed out (their keep rule); explicit split 85 / 15 by episode, seed 20261104; expert
   failures kept in the split and dropped by the trainers (success-only rule).
3. **Lidar input variant** `raw21_targetpolar_clock_lidar64_v1` (92 inputs = MLP-Ego's 28 + 64 ranges) in
   `learning/egocentric.py`, shared by the GRU and the transformer through `features_for`; `encode`, `predict`,
   `predict_action` take `lidar=`; the trainers' live-vs-batch checks pass the scan through. 51 learning tests pass;
   end-to-end smoke (convert 2 episodes, train 2 epochs of each model, run each in their harness) passes.
4. **Harness adapter** `scripts/obstacle/rl_obstacle_policy.py`: our NumPy policies behind their interface, clock
   reset per episode, commands clipped to the declared envelope (±5.27 rad/s, ±30 N m); policy step 0.12–0.14 ms.
5. **Night chain** (`chain_step9_night.sh`): after the collection, convert train2; train MLP-Ego-IC (K 144), GRU-Ego
   (patience 40) and TF-Ego (patience 40) with the lidar variant, clock horizon 60 s, seed 0 of each first, then
   seeds 1 and 2 (one trainer on the GPU at a time); evaluate every model and the expert on dev2 (300 episodes) in
   their harness as soon as trained (`runs/obstacle/step9_<tag>_dev2/`). Memory: ~40 M training rows × 92 inputs;
   MLP peak ~60 GB RAM, sequence models ~18–54 GB (125 GB machine).

**First results (4 Nov 22:05).** Collection: train2 2000 / 2000 reached (0 contacts, 0 navigation timeouts, 1000
noisy), dev2 300 / 300; conversion 2000 episodes, 1700 / 300 split; the expert in their harness on dev2: success
300 / 300, docked_clean 300, 0 collisions, side as planned 300 / 300. MLP-Ego-IC + lidar seed 0 (21 min, 27 epochs,
validation NRMSE 0.157) on dev2:

| metric (their harness, protocol v2.0) | MLP-Ego-IC + lidar, seed 0 | expert | their L2 seed 0 |
|---|---|---|---|
| success (EE within 10 mm at some moment, no contact) | 182 / 300 = 60.7 % | 300 | 180 / 300 |
| stable arrival 1 / 2 / 5 cm (0.5 s, by 40 s) | 160 / 245 / 292 | 300 / 300 / 300 | 148 / 233 / 295 |
| docked_clean (within 0.2 m of the ring, still 1 s, before any contact) | 291 / 300 | 300 | 291 |
| collision / passed obstacle / no_pass | 0 / 300 / 0 | 0 / 300 / 0 | 0 / 300 / 0 |
| side taken = scenario side | 207 / 300 | 300 | 227 |
| min clearance (min / median) | 0.10 / 0.38 m | 0.17 / 0.38 m | 0.06 / 0.39 m |
| inside the ambiguous band: success / docked_clean | 62.0 % / 99.3 % | 100 / 100 | 57.7 % / 97.8 % |
| outside the band | 59.5 % / 95.1 % | 100 / 100 | 62.0 % / 96.3 % |

- **The obstacle is not the problem.** 0 contacts in 300 episodes, the obstacle passed in every one, min clearance
  10 cm, docked_clean 97 %; inside the ambiguous band (the multimodal demonstrations, their hypothesis H-MM) the policy
  docks in 136 / 137 and succeeds as often as outside. A unimodal regressor with the lidar in its inputs does not
  average the two detours into the obstacle here; it commits to one side (its side differs from the recorded coin in
  93 episodes, which is what a coin looks like from the outside).
- **The failures are the last centimetre, as on the A-to-B task.** 118 failures: closest approach of the hand median
  1.6 cm (78 within 2 cm, 33 in 2–5 cm, 7 beyond 5 cm); 113 of them docked (ring distance at the arm's extension 5 cm,
  bearing error 2.6°), the arm extends at 22.4 s, the same time the expert docks (22.6 s), and the episode ends by their
  stall rule (41) or the 60 s cap (77). Stable 1 cm 160 / 300 against 189 / 205 on the A-to-B nominal bank: a longer
  drive (3.7 m, two segments) and a wider start distribution, with the same parking residual.
- On their own scale (success 60.7 %) MLP-Ego-IC + lidar equals their best learner (L2 60 %) from the first seed, with
  one network and no side classifier. GRU-Ego and TF-Ego, and seeds 1–2, follow in the chain.

**22:15, transformer memory.** GRU-Ego + lidar seed 0 trained (13 min, 267 epochs, validation NRMSE 0.103) and is on
dev2. The transformer trainer on the full split was killed by the system for memory at 22:12: it pads every training
episode to the longest (1700 × 29 021 steps × 92 inputs in float64 = 36 GB, then float32 copies) while the MLP seed 1
trainer and a 16-worker evaluation were running on the same machine. No source edit is possible while trainers run
(they hash `learning/*.py`), so TF-Ego trains after the night chain on a capped split: 700 of the 1700 training
episodes (uniform, seed 20261104; 16.5 M rows), the same 300 validation episodes
(`runs/splits/step9_train2_split_tfcap700.json`, `chain_step9_tf.sh`, seeds 0–2, each evaluated on dev2). The
night chain skips its own transformer attempts (placeholder dirs). The data-size difference is declared in the table.
Fix for later: stream the padded float32 episodes as `gru_policy.normalised_group` does.

Open points for the morning: the clock input runs to 60 s here (40 s on the A-to-B task); the ignition copies rule is
unchanged (first row of the terminal zero-wheel run, i.e. the expert's dock); the ambiguous band is where a unimodal
regressor should average left and right (their H-MM): our models have no side latch, so the band is the first thing to
read in the dev2 rows (`ambiguous` column).

**5 Nov 04:30, dev2 results (MLP-Ego-IC + lidar and GRU-Ego + lidar, three seeds each; TF-Ego on the 700-episode
cap training now).** Harness protocol v2.0 on dev2 (300 clean scenarios, seed 300; T_MAX 60 s; success = EE within
10 mm at some moment with no contact). Percent of 300, mean ± s.d. over seeds 0–2 (min–max); `scripts/obstacle/dev2_table.py`.

| policy | success | stable 1 cm | 2 cm | 5 cm | docked clean | collision | same side as expert | band / outside |
|---|---|---|---|---|---|---|---|---|
| expert v2 (clean) | 100 | 100 | 100 | 100 | 100 | 0 | 100 | 100 / 100 |
| MLP-Ego-IC + lidar (K=144) | 54±6 (48–61) | 45±8 (39–53) | 71±12 (59–82) | 91±7 (83–97) | 93±5 (87–97) | 0 (0–1) | 70±2 | 55±7 / 53±6 |
| GRU-Ego + lidar | 13±3 (9–15) | 7±2 (6–9) | 25±2 (23–27) | 67±11 (60–79) | 97±1 (96–98) | 8±8 (0–17) | 78±2 | 13±4 / 12±3 |
| teammate's L2 (their table, seed 0) | 60.0 | 49.3 | 77.7 | 98.3 | 97.0 | 0 | 75.7 | 57.7 / 62.0 |

Per seed: MLP success 182 / 159 / 144, stable 1 cm 160 / 127 / 116, collisions 0 / 2 / 0; GRU success 44 / 26 / 43,
stable 1 cm 20 / 17 / 27, collisions 22 / 51 / 1. Every policy passes the obstacle in 300 / 300 episodes and docks
clean in 87–98 %: navigation with the 64-beam scan is learnt, and the ambiguous band is not the problem (success inside
the band equals success outside it for every seed, so the unimodal regressor does not average the two detours; it
takes the expert's side in 70–78 % of the scenarios and the other side otherwise, both clear of the cylinder,
min clearance median 0.37–0.38 m against the expert's 0.38 m).

**What fails is the reach and, above all, the hold, and the cause is in the demonstrations.** The group's collector
stops an episode the moment the end-effector is within the reach tolerance (`collect.py`, `StopCriteria(reach_tolerance
...)`): the 2000 training episodes end at the expert's reach (last one at 29.5 s, median 23.6 s) and contain no
holding at all, where the A-to-B episodes held the pose until 40 s. The harness, by contrast, runs on after the first
reach: the expert holds 300 / 300 (EE < 15 mm for 10 s), our MLP holds in 33 / 30 / 10 of its 182 / 159 / 144 reaches
and leaves the target a median 2.2–3.0 s after arriving (EE at the end of the 10 s window: median 22–28 mm), the GRU
leaves after 0.7–2.1 s (35–92 mm). After its last demonstration the policy also runs with a clock input beyond anything
in the training set (every episode is shorter than 30 s; T_MAX is 60 s): the GRU's 22 / 51 / 1 collisions all happen
after a clean dock (49 of seed 1's 51, 22 of 22 for seed 0; collision at 38–48 s against a dock at 21–22 s), when the
base leaves the dock and wanders into the cylinder (final ring distance 0.6–0.95 m, wheels and fingers the links in
contact). Failures that never reach are the last centimetres as on the A-to-B task: among failures the closest
approach is median 15.6 / 17.3 / 26.7 mm for the MLP (first quartile 11–15 mm) and 32–42 mm for the GRU, with the base
docked in 95 % of them (ring distance 0.05–0.09 m); the harness's "stalled" stop (error no longer improving while the
robot is still) accounts for 37–44 MLP and 112–166 GRU episodes. The GRU's obstacle result is far below its A-to-B
one (73 ± 12 % at 1 cm) for the same reason the A-to-B GRU sleeps on nominal starts (section 7): its memory makes it
the most sensitive model to a horizon it has never seen.

Reading for the report: the obstacle experiment is the cleanest evidence that behaviour cloning copies the demonstration
protocol, not the task. The same MLP-Ego-IC recipe that holds 1 cm for 15 s on 92 % of the A-to-B geometries reaches
10 mm on 54 % of the obstacle scenarios and keeps the pose on 13 %, because nobody showed it what to do after the
reach. Next step (not run; needs the user's go-ahead because it is a new collection of ~45 min plus ~3 h of
training and evaluation): re-run the same expert on the same 2000 scenarios with a hold tail (keep stepping the
expert for 10 s after the reach, as the harness itself does), convert, retrain the three MLP seeds and the three GRU
seeds, and read the hold column. A cheaper control: train the same models without the clock input (variant without
`clock`) to separate the clock extrapolation from the missing hold data.

Transformer: the cap-700 chain was stalled from 23:27 to 04:08 by the same process-guard pattern as on 4 Nov (a shell
wrapper whose text contained the trainer names); released at 04:08, seed 0 training.

**Seed reporting convention (Rodrigo, 4 Nov 19:00).** Tables II and III give stable arrivals as a percentage of the
205 geometries, mean ± s.d. over the seeds with the min–max range in brackets (`scripts/analysis/seed_stats.py`
computes the cells); single-seed rows are marked "seed 0". No cell reports the best seed alone. The prose keeps the
per-seed counts where the mechanism is the point (which seed sleeps, which parks). With seeds 3 and 4 the same cells
go to five seeds.

**Chain order, 4 Nov 15:00.** The nominal-only and payload chains were stalled from 14:47 to 14:55 by their own
process check: a shell wrapper whose command text contained `train_egocentric.py` matched the `ps | grep` guard, so
both waited on a process that was only a string. Removed the wrapper, and the nominal-only chain then failed twice in
its first minute (training split pointed at the clean stratified dataset instead of the DART one; a helper function
missing), fixed and relaunched at 14:57 with `runs/splits/dart_low_geometry_819_205.json` (same 205 validation
geometries as the evaluation bank, checked by episode id; 819 training geometries of which the trainer keeps the 721
expert successes). Order now: nominal-only (`v2_mlp28nom_trig144_seed{0,1,2}`) → payload on link 7 → seeds 3 / 4
(MLP-Ego-IC, GRU-Ego, TF-Ego at patience 40).

**Rodrigo, 4 Nov 14:30.** (1) The start-referenced goal is not a "rule" of the dataset: the world-origin bearing was a
mis-generalisation of the first random-start collector, and referencing the goal to the start pose is simply the
correct expert. The report states three collection conditions (DART, successes only, random starts) and measures two
of them: no DART (0 / 205) and nominal starts only (`chain_nominal_only.sh`, queued). (2) Model names adopted:
MLP-Base (21 obs. + clock), MLP-Ego (+7 egocentric transforms), MLP-Ego-IC (+K ignition copies), GRU-Ego, TF-Ego,
+DB and MLP-Ego-Cat for the parking variants. (3) Two more seeds (3, 4) of MLP-Ego-IC, GRU-Ego and TF-Ego queued
(`chain_seeds34.sh`, after the nominal-only chain; configs `mlp_pytorch_successes{,_patience40,_patience20}_seed{3,4}.json`),
evaluated on both banks as `v2_<tag>_s34_on_*_20261104`, so the report can give mean ± sd over five seeds.
