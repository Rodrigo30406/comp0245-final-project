# Second experiment set: the expert as a black box

Date: 27 October 2026, afternoon. Companion to `docs/MECHANISMS_20261027.md`
(why the first set behaves as it does) and `docs/NIGHT_RUN_20261027.md` (what
was run). This file records (1) the decomposition of the input-set gain into
"neutral maths" and "expert knowledge", measured today, and (2) the plan for
the second set, where the expert is treated as a black box and the network has
to learn its rules from data. Experiments are added one at a time so that each
isolates one factor.

## 0. The question

The best recipe so far (MLP, 32 inputs, DART low) extends the lecturer's 21
inputs with 10 "egocentric" features and a clock. Those 10 features are not all
of the same kind:

| Feature | Built from | Knowledge used |
|---|---|---|
| `target_in_base_x/y` | rotate `target − base` by `−yaw` | rigid transform only (robotics, not the expert) |
| `target_height_above_base` | `target_z − base_z` | rigid transform only |
| `goal_in_base_x/y`, `goal_distance_xy` | the expert's navigation **goal** | the standoff table of the expert configuration |
| `cos/sin_heading_error_to_goal` | angle from the yaw to the goal | the standoff table |
| `cos/sin_goal_bearing_error` | yaw vs the arrival orientation | the expert's rule `goal_bearing = atan2(target_y, target_x)` |

The goal is **not observable**: the observation contains the target, and the
expert decides where to park relative to it (0.35–0.8 m behind, along the
world-origin bearing). Those two rules live in `experiment_specs/expert/default.json`
and `Episode._initialize`; computing them for the policy is reading the expert's
design, not transforming its observations. The clock is neutral (a counter
inside the policy). The 14 joint inputs are untouched.

Rule adopted for the second set: **policy inputs may be functions of the
observations and of the task specification, never of the expert's control law**
(no standoff table, no bearing rule, no thresholds, no phase bit). The
alternative, giving the policy the expert's goal and then its thresholds and
then its phases, has no natural stopping point.

## 1. Decomposition measured today (DART low, 5 seeds each, 205 clean validation goals)

New variant `raw21_target_clock_v1` (commit `94330c7`): the 21 explicit inputs
+ the 3 rigid-transform features + clock (25 inputs). Its unit test checks that
changing the standoff table does not change a single column.

| Inputs | Knowledge | Stable arrival 5 cm, min / median / max | Final within 5 cm | Held ≥ 10 s | EE ever < 15 cm | Final > 1 m |
|---|---|---|---:|---:|---:|---:|
| 21 explicit + clock (22) | none | 5 / 9 / 10 | 0 | 0 | — | — |
| 21 + 3 rigid-transform + clock (25) | none (frame only) | 25 / 29 / 32 | 0–1 | 0 | 85–87 | 154–205 |
| 21 + 10 geometry + clock (32) | expert goal construction | 115 / 138 / 162 | 17–96 | 61–92 | — | — |
| 25 egocentric (replaces the 7 absolutes) | expert goal construction | 23 / 67 / 100 | 0–63 | 1–54 | — | — |

Offline validation NRMSE: 0.135–0.139 (21+3) vs 0.114–0.120 (32) vs 0.172–0.177 (22).

Reading:

- The frame change alone triples the arrival count (9 → 29) and makes the
  approach work: the end effector passes within 0.5 m of the target in
  202–204 of 205 episodes and within 15 cm in 85–87, versus 14–21 for the 22-input
  model. This is the align/drive law (`docs/MECHANISMS_20261027.md` §1): with
  the target in the robot frame the turn law becomes near-linear.
- Almost none of it survives the stop: 154–205 episodes end more than 1 m away.
  Without the goal features the network has to learn *where to stop relative
  to the target* from data alone, and the orient/stop regime is the least
  covered one (one approach direction per demonstration). The 7 goal-derived
  features give that stopping point explicitly, and that is where most of the
  32-input gain comes from (29 → 138).
- Consequence for the report: the 32-input result must be presented as
  "21 explicit inputs + expert-derived goal geometry", not as a neutral
  feature engineering. The honest black-box baseline today is 25–32 / 205.

The first-set result and this decomposition are the starting point of the plan
below. The open question of §7 of the mechanisms file (why 32 > 25 egocentric)
is a white-box question and is parked.

## 2. What the black-box learner has to learn, and which tool fits each part

| Missing piece | Nature | Does memory help? | What helps |
|---|---|---|---|
| Turn/approach law from absolute coordinates (`atan2`, rotation) | representation off the data manifold | no (GRU 21 inputs: 1–2 / 205) | state coverage: diverse initial poses; the 3 rigid-transform features are allowed |
| Where to stop relative to the target (the goal) | a smooth function of the target, never demonstrated except at the one approach direction | partly (a latch helps the stop, not its location) | near-target starts covering the orient/stop regime from many directions |
| The latched `done` (exact zero wheels for 30 s) | discontinuity + memory | yes | GRU/transformer with a gradient horizon long enough to see the latch |
| The arm's integrated reference `q_des` | hidden integrator | yes | memory; attention over "time since arrival" may fit better than a GRU |

The first two are **data** problems; the second-set budget goes there first.
Architecture only after the data covers what the architecture must represent,
otherwise a transformer on 819 nominal-start episodes with 21 inputs would fail
for the same reason as the MLP and teach nothing.

Fact that motivates E1 (checked in `datasets/*/scenarios.json`): every training
episode so far starts with the base at the origin with yaw 0 (`initial_base:
nominal`). Only the target varies (4 × 4 m box, z 0.3–0.8). All yaw variety in
the data is the expert's own turn; the network has never seen the robot
anywhere else. The collector already supports `initial_base: {"mode":
"uniform_xy", x, y, yaw}` (the reserved random-start bank uses it).

## 3. The experiments, one factor at a time

Common protocol for every item: DART low noise (`wheel_std 0.5, torque_std
1.0, hold 50 steps`, clean labels), 1 ms recording, 40 s horizon; MLP recipe
`mlp_pytorch.json` (+ seeds 1–4), 5 seeds; closed-loop evaluation on the same
**205 clean validation goals, nominal start** (`runs/training/mlp_stratified1024_torch_baseline/split.json`,
`--allow-cross-split`), so every row of every table stays comparable with the
first set. Input sets trained on each new dataset: 22 (21+clock, the lecturer's
line), 25 (21 + 3 rigid-transform + clock, the black-box line), and 32 (the
white-box reference, to see whether the data change also helps it). The two
reserved test banks remain untouched.

### E1 — Random initial pose (coverage of the turn law)

- Change: one new dataset, 1024 episodes, `initial_base: uniform_xy`, x, y in
  [−2, 2], yaw in [−π, π]; targets as the stratified-1024 plan. Everything else
  as `dart_stratified1024_low.json`.
- Hypothesis: closes most of the 22-input vs 25-input gap on the approach
  (align/drive) and raises both; the orient/stop regime should barely move.
- Decision: if 22 inputs now reach ≥ 60 / 205 the frame transform is no
  longer necessary; if 25 moves and 22 does not, the transform stays as the
  neutral feature of the black-box line.
- Note: the expert's path from a random pose is longer; expect a few more
  `nav_timeout` demonstrations (25 s) — count them, the medium-noise DART set
  lost 46 % of its demonstrations that way and that was fatal.
- Cost: collection ≈ 6–8 min (24 workers), 15 MLP trainings ≈ 7 min each in
  parallel, evaluation ≈ 10 min.

### E2 — Near-target starts (coverage of the orient/stop regime)

- Change: a new collector mode `initial_base: {"mode": "uniform_polar_around_target",
  "radius": [0.3, 1.5], "heading": [−π, π]}`: sample the start on a ring around
  the target with a random yaw. This is a small `src` change (config validation +
  `generate_scenarios`) with a unit test; it uses the target only, not the
  expert's goal. Dataset: 1024 episodes of this mode alone, then a 50/50 mix
  with E1 (2048).
- Hypothesis: the expert now demonstrates the stop from every approach
  direction, and from inside the stop disc with heading errors; drive-through
  failures (the dominant residual, mechanisms §5) fall. Side effect to check:
  an expert that starts 5 cm from its goal *returns* to it, so for the first
  time the data contains "at the goal and correcting" labels, with the arm at
  rest. That is the restoring force the hold regime lacks; it must not create
  label conflicts with latched `done` states (arm extended).
- Decision: compare drive-through counts and "held ≥ 10 s" between E1 and
  E1+E2. If E2 alone helps the stop but hurts the approach, keep the mix.
- Cost: as E1 plus ~1 h of implementation.

### E3 — Scale (learning curve on the best mixture)

- Change: 1024 → 4096 → 8192 episodes of the E1+E2 mixture, same seeds for the
  scenario generator family, same split fraction.
- Hypothesis: the black-box line keeps improving with data while the
  white-box line saturates; the per-seed spread narrows.
- Decision: pick the size where the 25-input curve flattens; that dataset
  becomes the base for E4/E5.
- Cost: collection ≈ 45 min for 8192; MLP training scales linearly
  (≈ 1 h at 8192); keep 5 seeds.

### E4 — Memory with a gradient horizon that can see the latch (GRU, fixed)

- Change: GRU on the E3 dataset, 25 black-box inputs, memory stride 20 ms,
  **full-episode BPTT** (`--chunk-steps 2000`, i.e. 40 s) instead of 1000
  decisions; hidden 128 (as today) and 256; 5 seeds; at most 3 trainers at a
  time (system RAM). Also one control: stride 1 ms with `--chunk-steps 20000`
  if it fits, to settle the 1 ms / 20 ms confound of mechanisms §6.
- Hypothesis: with the whole episode inside the gradient window the GRU learns
  the latch; "held ≥ 10 s" and "final within 5 cm" rise above the MLP on the
  same data; the arm regime (torque NRMSE 0.3 in done) improves only partly.
- Decision: if the hold is fixed but the arm is not, go to E5; if neither
  moves, the data still does not expose the latch and E2's near-target starts
  need more weight.

### E5 — Attention over the episode (transformer), only if E4 leaves the arm unsolved

- Change: causal transformer over 20 ms tokens (2000 per episode), inputs as
  E4, small (4 layers, 128 wide), same TBPTT-free full-episode training, 5
  seeds, same NumPy-exportable inference contract (or torch CPU inference
  measured for latency).
- Hypothesis: "time since arrival" is directly readable by attention, which
  is what the integrated arm reference depends on.
- Not before E3: transformers are data-hungry and the first set has 819
  training episodes.

### E0 (white-box contrast, optional, cheap) — the latched `arrived` bit

- The counterpart to the black-box line: 32 inputs + the expert's `done`
  recomputed from observations (`docs/MECHANISMS_20261027.md` §8). It breaks
  the rule of §0 and is reported only as the ablation that confirms the
  mechanism: if the hold is fixed with the bit, the latch was the cause; if
  not, the arm integrator is. 5 seeds, same data as the first set.

## 4. Order and gating

```
E1 (random pose)  ─►  E2 (near-target)  ─►  E3 (scale)  ─►  E4 (GRU, full BPTT)  ─►  E5 (transformer, if needed)
                                                             E0 runs anytime as the white-box contrast
```

Each step reuses the previous dataset family and adds one factor; each is
evaluated closed loop before the next is launched. Every table carries the
first-set rows (22, 25 black-box, 32 white-box on nominal-start data) as the
reference, so the gain of each factor is visible per input set.

## 5. Reporting template (per experiment)

| Inputs | Data | Stable arrival 1 / 2 / 5 cm (min / median / max over 5 seeds) | Final within 5 cm | Held ≥ 10 s | Drive-throughs | Demos lost to `nav_timeout` |
|---|---|---|---:|---:|---:|---:|

Plus, for each dataset, the coverage check that motivated it: yaw histogram at
t = 0 (E1), approach-direction histogram at `done` (E2), episode count (E3).

## 6. E1 result (16:54–19:22): random initial pose makes everything worse

Dataset `datasets/e1_randompose_dart_low/20261027T215423Z_…` from
`experiment_specs/collection/e1_randompose_dart_low.json`: 1024 episodes, same
stratified targets, base start uniform in x, y ∈ [−2, 2], yaw ∈ [−π, π],
DART low, 7 min with 24 workers. Expert outcome 981 success / 43 stalled
(the nominal-start set: 895 / 129), so the demonstrations are not the
problem. Split: the same geometry plan (819 / 205). Fifteen MLPs (3 input sets
× 5 seeds), evaluated on the 205 clean nominal-start validation goals
(`--allow-external-holdout`: the evaluator's leakage guard keys on target +
initial state, and no training condition is shared).

| Inputs | Data | Offline val NRMSE | Stable arrival 5 cm, min / median / max | Final within 5 cm | Held ≥ 10 s | EE ever < 15 cm |
|---|---|---:|---|---:|---:|---:|
| 22 (21 + clock) | nominal starts | 0.172–0.177 | 5 / 9 / 10 | 0 | 0 | 14–21 |
| 22 | **E1 random starts** | 0.212–0.218 | **2 / 4 / 7** | 0 | 0 | 8–14 |
| 25 (21 + 3 rigid + clock) | nominal starts | 0.135–0.139 | 25 / 29 / 32 | 0–1 | 0 | 85–87 |
| 25 | **E1 random starts** | 0.181–0.186 | **11 / 12 / 17** | 0 | 0–4 | 73–83 |
| 32 (21 + 10 geometry + clock) | nominal starts | 0.114–0.120 | 115 / 138 / 162 | 17–96 | 61–92 | 160–188 |
| 32 | **E1 random starts** | 0.115–0.122 | **24 / 26 / 46** | 0–31 | 0–19 | 81–106 |

The hypothesis of E1 (random poses close the 22-vs-25 gap on the approach) is
rejected: every input set, including the white-box 32, loses most of its
closed-loop performance with the same 1024-episode budget, while the offline
NRMSE of the 32-input set is unchanged (0.115–0.122). Two cross evaluations
on E1's own 205 random-start validation scenarios separate the explanations:

| Model (32 inputs) | On nominal-start goals | On random-start goals (E1 validation) | Failure shape on random starts |
|---|---|---|---|
| trained on nominal starts | 115–162 | **36–70** | EE ever < 15 cm 81–104; final > 1 m in 82–128 (runs away when it fails) |
| trained on E1 random starts | 24–46 | **25–42** | EE ever < 15 cm 83–101; final > 1 m in 35–73 (gets near, never finishes) |

Reading:

- Not a train/test mismatch: the E1-trained models are weak on their own
  distribution too (25–42). The nominal-trained models beat them even there.
- Dilution: with the start pose free, 819 demonstrations cover a 4-D family
  of situations instead of a 3-D one, and the demonstrations are longer
  (median 13.5 k vs 11.1 k samples, `done` at 10.8 s vs 8.7 s). The
  per-situation density that the stop regime needs is gone. Offline this is
  invisible (same NRMSE); closed loop it costs everything, which is the same
  lesson as the first night: offline error does not predict closed-loop
  success.
- Generalisation of the nominal recipe to random starts is partial (36–70,
  with the runaway failure that the yaw-sweep probe predicted in
  `docs/MECHANISMS_20261027.md` §1). This is a first, non-reserved estimate
  of what the reserved random-start bank would show.
- Random-pose data is therefore **extra** data, not a replacement, and only
  at a larger budget. The next step is the dilution test, not E2.

Dataset facts recorded on the way (300 episodes each): phase fractions by
sample, nominal align / drive / orient = 0.19 / 0.62 / 0.19 versus E1
0.11 / 0.63 / 0.26; demonstrations end at the expert's success, so the
post-`done` hold is only a few seconds per demonstration while the evaluation
asks for 30 s.

### Revised order after E1

1. **E3' — dilution test (no code):** 3072 more random-start episodes
   (`--resume` on the same spec family or a 4096-episode spec), train the
   32-input set on 4096 random-start demonstrations, evaluate on both goal
   sets. If 4096 random-start demos recover ≥ the nominal 115–162 on nominal
   goals, the cause was density and the black-box line scales with data; if
   not, the random-start task needs the nominal core.
2. **Mixture (small trainer change):** nominal 1024 + random 1024 in one
   training set, 32 and 25 inputs, 5 seeds; this is the "extra data, not
   replacement" setting and the honest baseline for the reserved random bank.
3. Then E2 (near-target starts) on top of whichever of the two wins.

## 6b. Inverted experiment (19:49–20:31): one target, 512 random starts — it is not density

Rodrigo's check: fix one target (validation target `[0.96, 0.83, 0.55]`),
collect 512 DART-low demonstrations from uniformly random base poses
(`experiment_specs/collection/inv_onetarget_randomstart_dart_low.json`,
expert 486 / 512 successes), random split 448 / 64, three input sets × 3
seeds, evaluated on the 64 held-out random starts of the same target. With
one target the start-pose family is 3-D and 448 demonstrations cover it
densely, so if E1 had failed for lack of density this should work.

| Inputs (one target, random starts) | Stable arrival 1 / 2 / 5 cm of 64 | EE min distance (q25 / q50 / q75) |
|---|---|---|
| 22 (21 + clock) | 0 / 0 / 0 (3 seeds) | — |
| 25 (21 + 3 rigid + clock) | 0 / 0 / 0 (3 seeds) | 0.169 / 0.171 / 0.173 m |
| 32 (21 + 10 geometry + clock) | 2 / 5 / 12, 0 / 0 / 0, 0 / 0 / 2 | 0.168 / 0.169 / 0.171 m |
| expert | 63 / 63 / 63 | 0.002 m |

Density is not the explanation. The base-navigation diagnostics of the
evaluator say where the failure is:

| Policy (32 inputs) on … | base ever < 8 cm of its goal | stable base pose | base drift after arrival (median) | base finally in pose tolerance | final EE distance when the base is parked (q25 / q50 / q75) |
|---|---:|---:|---:|---:|---|
| nominal-trained, nominal goals | 0.94 | 0.84 | 0.13 m | 105 / 1025 | 0.025 / 0.039 / 0.062 m |
| E1-trained, nominal goals | 0.85 | 0.72 | 0.07 m | 373 / 1025 | 0.083 / 0.145 / 0.267 m |
| E1-trained, random-start goals | 0.90 | 0.77 | 0.05 m | 464 / 1025 | 0.085 / 0.175 / 0.313 m |
| one-target-trained, its random starts | 0.92 | 0.83 | 0.24 m | 35 / 192 | 0.176 / 0.181 / 0.187 m |

**The base is fine; the arm is what random-start training breaks.** Models
trained on random starts park the base at its goal as often as the nominal
ones, and hold it better on the nominal goals (drift 7 cm vs 13 cm, half the
episode inside tolerance vs 12 %). But with the base correctly parked the
end effector stops 15–18 cm from the target instead of 4 cm; in the
one-target case it ends 17 cm away in every episode, i.e. the arm converges to
one fixed wrong configuration. Offline, the torque error in the `done`
regime (the only regime where the arm moves) rises from 0.31 (nominal data)
to 0.46–0.53 (E1 data) and 0.35–0.36 (one-target data), while the wheel
errors stay in the same range; the overall validation NRMSE hid this because
`done` is 15 % of the samples and the torques are 7 of 9 outputs averaged
with 2 nearly perfect wheel channels. (`regime_nrmse.py` in the scratchpad;
60 validation episodes each.)

What this is *not*: the base pose at `done` is equally tight in all three
datasets (target in the base frame at `done`: x / y std 3 / 2 cm nominal,
3 / 4 cm E1, 3 / 4 cm one-target; base–goal distance at `done` median 7 cm,
90th percentile 8–10 cm), so the arm's reach target in the robot frame is
not more scattered with random starts. The orient phase is four times larger
(64–73 k vs 16 k validation samples; arrival heading error median 0.86–1.17
rad vs 0.16 rad; base slip during the turn 7–10 cm vs 2 cm), which is the
intended coverage gain, and the wheel heads use it (better hold).

Open question, with the test that settles it: why does the reach regime get
harder to imitate when the approach is random? Candidates: (i) the arm head
in nominal data indexes the reach trajectory with the clock (`done` at
8.7 ± 2 s) and loses that index when `done` spreads over 6–15 s; (ii) the arm
imitation only works when the base is as still as in the demonstrations and
the small residual base motion after a long orient turn is enough to push
the integrated reach off its demonstrated curve. Test (i): retrain E1 32
inputs without the clock column, 3 seeds. Test (ii): roll out the
E1-trained policy's arm with the base frozen at the expert's `done` pose
(a small harness on `Episode`). Both are cheap; neither was run tonight.

### E3 arms launched (20:38, running overnight)

Datasets: `e3_multistart3x_dart_low` (3072 episodes, every plan target with 3
random starts, expert outcome in the dataset manifest) plus the existing
nominal and E1 sets. Two splits built with the multi-dataset support added in
commit `ff0f120` (`runs/splits/e3_randomonly_4096.json`: E3 + E1, 3276 / 820;
`runs/splits/e3_mixture_4096.json`: nominal + E3, 3276 / 820), both holding
out the plan's 205 target indices in every dataset. 32-input MLPs, 3 seeds per
arm, evaluated on the nominal goals and on E1's random-start goals. Given
§6b, the arm regime, not density, is the thing to watch in these results.
Operational note: three concurrent trainers on 4096 episodes exhausted
system RAM and one was OOM-killed; the chain was resumed with at most two
trainers at a time (`chain_e3b.sh`).

## 6c. The 17 cm explained (28 Oct, 00:50): the arm never leaves its hold posture

Measured on the inverted experiment's traces
(`runs/evaluation/inv_onetarget_randomstart_on_own_validation_20261027/traces`,
sampled every 100 ms) and on the demonstrations of the one-target dataset.

- In the demonstrations, at the instant `done` switches on (base parked at the
  standoff, arm still at `q_hold`), the end effector is 0.174 m from the target
  (median; p10 0.169, p90 0.200). The expert then extends the arm and ends at
  0.010 m.
- The closed-loop policies' end effector ends at exactly that 0.17 m because the
  arm does not move. End-effector displacement in the base frame between the
  first and the last sample, median over the 64 episodes:

  | Policy | EE displacement in base frame (m) | Change of arm joints 2/4/6 (rad) | Min EE–target dist. (m) |
  |---|---:|---|---:|
  | expert | 0.208 | 0.56 / 0.42 / 0.16 | 0.000 |
  | 21 + clock, seeds 0–2 | 0.027–0.030 | 0.05 / 0.01 / 0.00 | 0.170–0.173 |
  | 21 + 3 target-in-base + clock, seeds 0–2 | 0.032–0.040 | 0.05–0.07 / 0.01 / 0.00 | 0.171–0.172 |
  | 32 inputs, seeds 0 and 1 | 0.031–0.035 | 0.05 / 0.02 / 0.00 | 0.169–0.170 |
  | 32 inputs, seed 2 | 0.251 | 0.70 / 0.48 / 0.21 | 0.169 |

  Eight of the nine models keep the arm within 3–4 cm of the hold posture for
  the whole 40 s; the base parks correctly underneath. Only one model (32
  inputs, seed 2) extends the arm as the expert does, and even it ends 25 cm
  off (it extends to the right height but not to the right point).
- The E1 numbers are the same effect on several targets: with the base parked,
  the end effector of E1-trained models ends 8–27 cm from the target, which is
  the spread of the hold-posture distance across the standoff table (0.35–0.80 m).

So "a stable 17 cm error" was the wrong description. The arm does not go to a
wrong posture; it never starts the reach. The failure is the *trigger* of the
reach phase, not the reach trajectory. This narrows the two hypotheses of §6b:

- Clock as trigger. In nominal demonstrations `done` happens at 8.7 ± 2 s, so
  the clock alone predicts the reach onset, and a clock trigger is robust in
  closed loop (time always advances, whatever the base does). With random
  starts `done` happens anywhere between 6 and 15 s; the clock is useless and
  the only remaining cue is the state "base parked at the standoff with zero
  velocity", which the regressor evidently does not isolate sharply enough,
  except by luck in one seed.
- Test (E6, nominal data): train the 32-input model *without* the clock (31
  inputs). If the arm stops extending on nominal starts, the nominal models'
  arm was clock-triggered and the black-box line needs a state-based trigger
  (memory or an explicit `arrived` feature). Needs a new input variant in
  `egocentric.py`; it is applied only when no trainer/evaluator is running.
- Implication for E3 (mixture of nominal and random starts): the nominal demos
  support a clock trigger that the random demos contradict, so the mixture is
  expected to lose the arm on nominal starts too; the evaluations running now
  decide.

## 6d. E3 phase A (28 Oct, 11:49–13:25): the mixture can do both, but only in one seed of two

The machine rebooted at 10:56 (kernel update; the NVIDIA module is not installed for the new kernel, so
everything below ran on CPU). Three E3 trainers were complete before the reboot (random-only seed 0,
mixture seeds 0 and 1); they were evaluated on the 205 nominal goals (`*_on_clean_validation_20261027`)
and on the 205 random-start goals of E1 (`*_on_e1_random_validation_20261027`). Stable arrival at 5 cm,
plus the same base/arm decomposition as §6c (base parked = stable pose arrival at the expert's goal;
arm extended = end-effector displacement in the base frame > 10 cm over the episode):

| Training data (MLP 32 inputs) | Validation | Stable arrival 5 cm | Base parked | Arm extended |
|---|---|---:|---:|---:|
| nominal 1024, seeds 0–4 (reference) | nominal starts | 115–162 | 154–194 | 119–190 |
| nominal 1024, seeds 0–4 (reference) | random starts | 36–70 | 21–56 | 60–148 |
| E1 random 1024, seeds 0–4 (reference) | nominal starts | 24–46 | 127–161 | 0–27 |
| random-only 4096, seed 0 | nominal starts | 56 | 147 | 42 |
| random-only 4096, seed 0 | random starts | 55 | 154 | 31 |
| nominal + random 4096, seed 0 | nominal starts | 51 | 177 | 26 |
| nominal + random 4096, seed 0 | random starts | 51 | 184 | 18 |
| nominal + random 4096, seed 1 | nominal starts | 93 | 176 | 134 |
| nominal + random 4096, seed 1 | random starts | **113** | 176 | 136 |

Reading:

- The two halves of the task come from the two kinds of demonstrations. Random-start demonstrations
  teach the base to park from anywhere (147–184 parked, against 21–56 for nominal-trained models on
  random starts) and do not teach the arm to start (18–42 extended). Nominal demonstrations teach the
  arm to start (119–190) and not the base from random starts. This is §6c again, on 4× the data.
- Four times more random-start data does not fix the arm (random-only 4096 seed 0: 42 extended,
  against 0–27 for the 1024-episode E1 models). Density is not the issue, as the inverted experiment said.
- The mixture *can* learn both: seed 1 parks in 176 and extends the arm in 134–136, and on random
  starts it makes 113 stable arrivals, the best of any model so far on that validation (nominal
  reference 36–70). But seed 0 of the same data loses the arm (18–26 extended) and lands at 51.
  The reach trigger is learnt or not learnt as a whole, per seed. That is the signature of a
  threshold event that the regressor has no reliable cue for: when the clock is informative
  (nominal data) every seed finds it; when the clock is ambiguous (random starts in the data) it
  is a coin toss.
- On nominal starts the mixture seed 1 (93) is still below the nominal-only models (115–162): the
  random demonstrations cost accuracy on the nominal regime even in the good seed.

The remaining seeds (random-only 1, 2; mixture 2) retrain once the GPU is back (CPU would take ~3 h
each on 4096 episodes); their evaluations go to `*_20261027_b` directories.

## 6e. Pure transforms, 28 inputs (28 Oct, 11:44–14:03): they unlock the base and the arm, not the hold

Variant `raw21_targetpolar_clock_v1` = the 21 explicit inputs + target in the base frame (x, y,
height) + planar distance base→target + cos/sin of the heading error to the target + clock. Every
added column is a function of the observations alone (test
`test_targetpolar_variant_is_pure_transform_of_the_target_in_base`: a different standoff table changes
nothing). Trained like the 25- and 32-input models (nominal DART low 1024, split 819/205, 5 seeds; on
CPU, 7–17 min each), evaluated on the 205 nominal goals
(`runs/evaluation/dart_low_mlp_raw21polar_seeds_on_clean_validation_20261028`).

| Inputs | Offline NRMSE | Stable arrival 5 cm | Base parked | Arm extended | Drift after parking (median) | Held ≥ 10 s |
|---|---:|---:|---:|---:|---:|---:|
| 22 = 21 + clock | 0.172–0.177 | 5–10 | 11–36 | 80–161 | — | — |
| 25 = 21 + target in base + clock | 0.135–0.139 | 25–32 | 138–160 | 0–60 | — | — |
| **28 = 25 + distance and heading to target** | 0.129–0.137 | **129 / 38 / 33 / 66 / 57** | 158–183 | 32–197 | 0.36 / 3.1 / 1.6 / 1.4 / 1.4 m | 21 / 4 / 0 / 1 / 4 |
| 32 = 21 + 10 goal geometry (expert rules) + clock | 0.114–0.120 | 115–162 | 154–194 | 119–190 | 0.07–0.50 m | 61–92 |

(Base parked = stable pose arrival at the expert's goal; arm extended = end-effector displacement in
the base frame > 10 cm; drift = maximum base displacement after the stable arrival; held = end
effector within 5 cm for 10 s or more.)

Reading:

- The two polar features do what Rodrigo expected for the *approach*: the base parks in 158–183
  goals (25 inputs: 138–160) and the arm now starts in most seeds (32–197 against 0–60). One seed
  reaches 129 stable arrivals, inside the range of the 32-input models, with no expert rule in
  the inputs. The scalar distance is a far easier stop cue for the regressor than the two
  coordinates it is computed from, although the offline error barely moves (0.134 vs 0.137): the
  gain is in closed loop, where a sharp threshold matters and a 2 % fit does not.
- What the pure features do not give is the *hold*. After parking, the 28-input models drift
  1.4–3.1 m (median) in four seeds of five; the 32-input models drift 0.07–0.17 m in four of five.
  Held ≥ 10 s: 0–21 against 61–92. The expert-rule features describe the parked state as a fixed
  point (goal in base ≈ 0, goal distance ≈ 0, bearing error ≈ 0, the same for every goal); the
  target-relative features describe it as "distance = standoff(height), heading ≈ 0", a different
  point for each height, next to the approach regime. The regressor's zero-wheel output is not
  anchored there, and the integration argument of `docs/MECHANISMS_20261027.md` applies: a few mm/s of wheel
  command with no restoring force walks the base away in 30 s.
- Seed variance again dominates (33 to 129). Same signature as E3 §6d: the trigger and the hold are
  threshold events the regressor finds or misses as a whole.

Consequence for the black-box line: with observations only, the missing piece is a *stationary
cue for "parked"* that does not depend on the expert's goal. Candidates, in order of cost: (a) base
speed and yaw rate already in the 21 inputs but perhaps too small in scale after normalisation — a
"base is still" feature (|v|, |ω|) as a pure transform; (b) memory (E4) so the policy can latch;
(c) the white-box contrast `arrived` bit (E0) to bound what any cue could give.

## 6f. E3 complete (29 Oct, 20:42): three seeds per arm, both validations

Six MLP-32 models, 4096 training episodes each, trained with the memory-fixed trainer (seeds 1, 2 of
random-only and seed 2 of the mixture) or before it (the others; the training loop is identical).
Stable arrival at 5 cm on 205 goals; base parked / arm extended / median drift after parking as in §6c–6e.
Evaluations `e3_*_on_{clean,e1_random}_validation_20261027` and `*_b`.

| Training data | Seed | Nominal starts: arrivals | parked | arm | drift | Random starts: arrivals | parked | arm | drift |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random-only 4096 | 0 | 56 | 147 | 42 | — | 55 | 154 | 31 | — |
| random-only 4096 | 1 | 26 | 146 | 0 | 0.02 m | 29 | 169 | 0 | 0.02 m |
| random-only 4096 | 2 | 23 | 163 | 45 | 0.09 m | 33 | 169 | 24 | 0.03 m |
| nominal + random 4096 | 0 | 51 | 177 | 26 | — | 51 | 184 | 18 | — |
| nominal + random 4096 | 1 | 93 | 176 | 134 | — | **113** | 176 | 136 | — |
| nominal + random 4096 | 2 | 78 | 171 | 71 | 0.18 m | 71 | 174 | 58 | 0.13 m |
| reference: nominal 1024, seeds 0–4 | | 115–162 | 154–194 | 119–190 | 0.07–0.50 m | 36–70 | 21–56 | 60–148 | |

Reading, adding to §6d:

- Random-start demonstrations teach the base *and its hold*: the random-only models park in 146–169 goals
  and then drift only 2–9 cm (the nominal-trained models drift 7–50 cm; the 28-input pure-transform
  models 36–310 cm). The latch that the nominal data never pins down, random-start data does. The
  mechanism fits §6c: with random starts the clock carries no information about the phase, so the
  regressor is forced to read the stop from the state, and it does.
- The same data never teaches the arm: 0–45 episodes with the arm extended, in two seeds literally 0.
  So the arm trigger in nominal-trained models *is* clock-driven (prediction of §6c, now supported by
  three seeds × two validations), and the clock is exactly what random starts destroy.
- The mixture trades one for the other with the seed deciding the balance: seed 1 keeps the arm (134)
  and the hold, seeds 0 and 2 keep the hold and partly lose the arm (26, 71). On random starts the
  mixture is the best family so far (51–113 against 36–70 for the nominal models), on nominal starts
  it is below the nominal models (51–93 against 115–162).

Taken together with §6e, the two halves now have a cause each, and they pull in opposite directions
for a stateless regressor on these inputs: the hold needs a stop cue that is not the clock (random
starts supply it, pure target features do not), the arm trigger needs a cue the clock provides in
nominal data and nothing else provides. A one-frame policy without base velocity cannot have both
from observations alone. The next step is therefore not more data or more features of the current
frame but a base-velocity signal: finite difference of two consecutive observations (minimal memory,
pure transform), then the GRU if that is not enough.

## 6g. E4 phase 1 (30 Oct, 00:14): finite-difference base velocity makes a copycat, not a stop cue

Three MLP-31 models (`raw21_targetpolar_vel_clock_v1`: the 28 pure transforms of §6e plus the finite difference of
the lecturer's base position and bearing, in the base frame) trained on the nominal + random mixture of 4096
episodes, 205 nominal and 205 random-start goals. Evaluations `e4_mlp_polarvel_on_{clean,e1_random}_validation_20261029`.

| Seed | offline NRMSE | Nominal: arrivals | parked | arm | drift | held ≥ 10 s | Random: arrivals | parked | arm | drift | held |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.108 | 40 | 107 | 130 | 0.90 m | 7 | 39 | 14 | 111 | 1.06 m | 1 |
| 1 | 0.107 | 38 | 116 | 0 | 0.20 m | 34 | 36 | 4 | 0 | 0.12 m | 26 |
| 2 | 0.106 | 37 | 98 | 0 | 0.09 m | 19 | 33 | 12 | 0 | 0.08 m | 8 |
| ref. MLP 28, nominal data (§6e) | 0.129–0.137 | 33–129 | 158–183 | 32–197 | 0.36–3.1 m | 0–21 | | | | | |
| ref. MLP 32, same mixture (§6f) | 0.114–0.120 | 51–93 | 171–177 | 26–134 | 0.18 m | | 51–113 | 174–184 | 18–136 | 0.13 m | |

The offline error is the best of every variant trained so far and the closed loop is the worst of the
black-box line: 37–40 arrivals on either validation, the base parked in 4–14 of 205 random-start goals.
The two facts have one cause, measured with `scripts/analysis/velocity_copycat.py` on 60 validation
episodes (778k rows) and `scripts/analysis/stop_profile.py` on the evaluations:

- **The velocity is the expert's previous command in disguise.** A linear map from the three velocity
  features alone recovers 90% of the variance of the expert's two wheel commands (R² 0.904 / 0.896); the
  other 28 features together recover 55% (0.550 / 0.538). The base is a first-order system driven by a
  smooth expert, so its velocity at *t* is a near copy of the command at *t* − 1 ms. Regressing on it is
  easy, which is why the offline error drops to 0.106; it is also the copycat shortcut of behaviour
  cloning from observation histories (Wen et al., NeurIPS 2020), learned here from a single extra frame.
- **The policy did take the shortcut.** On the rows where the expert drives (542k rows, mean |wheel| 2.27),
  the three models reproduce the command (2.27); with the velocity features set to zero, the same rows and
  the same geometry, they keep 38%, 30% and 33% of it. Two thirds of the drive command is read from "I am
  already moving", one third from where the target is.
- **Closed loop, a copycat latches.** Whatever the base is doing is fed back as its own cause. Seed 0 latched
  on "moving": at 40 s the base is still moving in 187 of 205 nominal episodes (closest approach to the goal
  median 8 cm, final median 1.46 m past it). Seeds 1 and 2 latched on "stopped": 152 of 205 nominal
  episodes end stopped outside the pose tolerance, at a median 0.10–0.16 m from the goal, so the hold is
  there (drift 9–20 cm, held ≥ 10 s in 19–34) but the pose is a few centimetres off and the arm never fires
  (0 extended). From random starts the same two seeds stop at a median 0.37–0.42 m from the goal and never
  got closer than 0.35–0.39 m: the expert's align phase rotates in place at zero linear speed, the policy
  reads "at rest" and stays at rest. The reference MLP 28 on nominal data has the opposite profile (97–199
  episodes still moving at 40 s, closest approach 1–2 cm): it has no stop at all.
- **The wheels read the arm as a phase flag.** The largest standardised sensitivities of the wheel outputs
  are to `arm_q_2`, `arm_q_6`, `arm_qd_6` and the lateral velocity, not to the target distance (seed 1 and 2
  do rank `target_distance_xy` first or second). In the demonstrations the arm leaves its hold posture only
  after the base has latched `done`, so the arm configuration is a perfect, non-privileged indicator of
  "wheels are zero". It is another shortcut of the same kind: the policy infers the phase from the
  consequences of its own past actions rather than from the task geometry.

What this settles: the base velocity is observable without privileged information and it does carry the
phase, but handing it to a one-frame regressor as a plain input turns the regressor into a copycat of its
last command. "Memory" for this task cannot be the previous frame; it has to be a latent that is trained to
predict the expert's phase, not to continue the robot's motion. That is what phase 2 (GRU over the 28 pure
transforms, 20 ms stride, gradient horizon 20 s) and phase 3 (MLP 28 control on the same data) are
running to test. If the GRU copies as well, the known remedies are feature dropout on the velocity
(Wen et al.) or an explicit phase head; both stay black-box.

## 6h. E4 phase 2 (30 Oct, 01:38): the GRU learns the stop and never reaches it

Three GRUs over the 28 pure transforms (hidden 128, head 128, memory stride 20 ms, TBPTT 1000 decisions = 20 s,
so the gradient sees whole demonstrations), same mixture split of 4096 episodes, 7–9 min of training each on the GPU.
Evaluations `e4_gru_polar_on_{clean,e1_random}_validation_20261029`.

| Seed | offline NRMSE | Nominal: arrivals | parked | arm | drift | held ≥ 10 s | Random: arrivals | parked | arm | drift | held |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.136 | 32 | 13 | 149 | 3.27 m | 9 | 29 | 7 | 155 | 2.95 m | 8 |
| 1 | 0.153 | **102** | 21 | 101 | 1.55 m | 38 | **99** | 9 | 96 | 1.69 m | 36 |
| 2 | 0.145 | 27 | 44 | 16 | 2.55 m | 2 | 26 | 12 | 14 | 2.07 m | 1 |
| ref. MLP 32, same mixture (§6f) | 0.114–0.120 | 51–93 | 171–177 | 26–134 | 0.18 m | | 51–113 | 174–184 | 18–136 | 0.13 m | |
| ref. first-night GRU 32 / GRU 25, nominal data | | 122–163 | | | | | | | | | |

Three measurements (`scripts/analysis/openloop_phase_error.py`, `clock_extrapolation.py`, `stop_profile.py`):

- **Open loop, every model has the stop.** Teacher-forced on 40 validation demonstrations, the predicted wheel
  command after the expert's stop is 1–3% of the drive level for the GRU, 0–5% for every MLP family, and it
  stays below 10% for 96–100% of the post-stop time. Drive-phase error relative to the expert: MLP 32
  mixture 0.03–0.04, MLP 31 velocity 0.05, **GRU 0.06**, MLP 28 mixture 0.09, MLP 28 nominal 0.11.
- **The clock extrapolation is real, and the mixture data removes it, not the memory.** Freezing the final
  (parked) observation of a demonstration and letting the clock run to 40 s, the models trained on nominal
  data start driving again: MLP 28 nominal 24–33% of the drive level at +20–30 s, MLP 32 nominal 14–28%.
  Nominal demonstrations last 9.3 s (median), so a clock past ~15 s is outside their support while the
  evaluation runs 40 s; this is the measured cause of the 0.36–3.1 m drift after parking in §6e. Models
  trained on the mixture do not restart (GRU 1–4%, MLP 32 mixture 1–4%, MLP 28 mixture 3–8% in two seeds,
  47% in one): random-start demonstrations last 13.2 s median and up to 29 s, and in them the clock carries
  no phase, so the regressor stops reading it.
- **Closed loop, the GRU rarely reaches the state where its stop fires.** Its closest approach to the goal is
  a median 10–18 cm on nominal starts and 20–36 cm on random starts, against an 8 cm / 0.05 rad box and the
  expert's 7 cm; it is still moving at 40 s in 103–196 of 205 episodes, so the hold it has open loop is never
  exercised (parked 13–44). Across the five families the open-loop precision does not predict the
  closed-loop approach: the MLP 28 on nominal data has the worst drive-phase error (0.11) and the tightest
  closest approach (1–2 cm, §6e); the GRU has the second-best error and the loosest approach. The latch is a
  knife edge in state space, and what decides whether it fires is where the policy's own trajectory passes,
  not how well it fits the expert's trajectory.

Two things the memory did buy. The GRU is the first family whose behaviour does not depend on the start
pose (32 / 102 / 27 nominal against 29 / 99 / 26 random, the MLP 32 mixture loses 0–20 between the two and
the nominal models lose half). And it brings the arm back on random-start data: 149, 101 and 16 episodes
with the arm extended, where the stateless models trained on random starts had 0–45 (§6f), so the
reach is being timed from the history rather than from the clock. Seed 1's 102 / 99 arrivals come from
exactly that: the arm reaching the target while the base creeps past it (parked 21, arm 101, held ≥ 10 s 38).

What this settles, with §6g: neither one-step velocity nor a 20 s recurrent memory turns the pure
28 inputs into a reliable hold on these data. The hold fails for two measured reasons that need two
different fixes: (1) the nominal demonstrations end too early for the clock (fixed by any data whose clock
is uninformative or longer, the mixture already does this); (2) the stop condition is an 8 cm / 0.05 rad box
that the learner's own approach misses by 10–30 cm. The second is a coverage problem around the goal, which
is what E2 (near-target starts: demonstrations that begin a few decimetres from the goal with random
heading, so the stop is demonstrated from many approach states) was designed for, and it is now the
first candidate. The phase 3 control (MLP 28 on the mixture) is being evaluated to close E4.

## 6i. E4 closed (30 Oct, 02:20–03:10): the control, and the measured cause of the lost hold

Phase 3, the control: three MLP-28 (pure transforms, no velocity) on the same mixture split. Evaluations
`e4_mlp_polar_on_{clean,e1_random}_validation_20261029`. Full E4 table (stable arrivals at 5 cm of 205):

| Model on the nominal + random mixture (4096) | offline NRMSE | Nominal: arrivals | parked | arm | drift | Random: arrivals | parked | arm | drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP 28 pure (control), seeds 0–2 | 0.145–0.149 | 23 / 30 / 27 | 119–126 | 3 / 103 / 15 | 2.0–2.3 m | 11 / 15 / 17 | 33–68 | 3 / 105 / 11 | 1.8–2.3 m |
| MLP 31 = 28 pure + FD velocity (§6g) | 0.106–0.108 | 40 / 38 / 37 | 98–116 | 130 / 0 / 0 | 0.09–0.90 m | 39 / 36 / 33 | 4–14 | 111 / 0 / 0 | 0.08–1.06 m |
| GRU 28 pure, memory 20 ms (§6h) | 0.136–0.153 | 32 / **102** / 27 | 13–44 | 149 / 101 / 16 | 1.6–3.3 m | 29 / **99** / 26 | 7–12 | 155 / 96 / 14 | 1.7–3.0 m |
| MLP 32 (21 + 10 goal-based + clock, §6f) | 0.114–0.120 | 51 / 93 / 78 | 171–177 | 26 / 134 / 71 | 0.13–0.18 m | 51 / **113** / 71 | 174–184 | 18 / 136 / 58 | 0.13 m |

The control settles the attribution. Same data, same trainer: the 28 pure inputs give 23–30 arrivals, the 32
inputs with the goal-based features give 51–93. Neither memory variant closes that gap (velocity 37–40, GRU
27–102 with one lucky seed). The 28-input control reaches the box as precisely as anything (closest approach
2–4 cm, parked 119–126) and then does not stay: moving at 40 s in 199–203 of 205 episodes, held ≥ 10 s in 0–5.

What makes the pure-input models leave, measured on the evaluation traces (`creep_profile.py`,
`arm_drift_at_restart.py`) and on open-loop probes at the expert's stop pose (`stop_basin.py`,
`stop_arm_sensitivity.py`, `clock_extrapolation.py --freeze-at stop`):

- **It is not the arm.** When a parked base restarts, the arm is 0.020 rad rms from its hold posture, the same
  value the expert shows at rest (the arm never moved). The wheel outputs *are* very sensitive to the arm
  (an rms offset of 0.1 rad per joint at the stop pose provokes 30–48% of the drive command in the MLP 28,
  156% in one GRU; 0.2 rad gives 64–119%): the wheels do read the arm as the phase flag (§6g), but in closed
  loop the arm stays where the demonstrations had it, so this sensitivity is not what fires.
- **It is not the clock, and not the pose inside the box.** With the observation frozen at the expert's stop
  (arm at hold) and the clock run to 40 s, the mixture-trained MLP 28 outputs 2–7% of the drive command
  (one seed 17–23% after 10 s); displacing the base anywhere within ±15 cm or rotating it ±0.1 rad at the stop
  changes that to 2–5%. The restarts happen at a median 15.7–20 s, inside the clock support of the mixture
  (random-start demonstrations last 13 s median, up to 29 s).
- **It is a creep.** The regressor's "zero" is 6–10% of its driving command (the expert's is exactly zero: a
  latched phase, not a regression). That bias integrates: in the 120–142 nominal episodes per seed where the
  MLP 28 parks and restarts, the base creeps 5.8–9.2 cm over 7.5–11.9 s with the heading fixed (2.6–5.6°),
  **towards the target in 95–100% of the episodes**, and then drives. Nominal-trained MLP 28 (§6e): the same,
  5.5–10 cm towards the target in 86–97% of 157–198 episodes, restart at 13–16 s. The expert parks at the
  standoff radius (0.70–0.72 m from the target) approaching from outside, so the demonstrated stop states form a
  thin shell at that radius; nothing in the data says what to do 10 cm inside it. The open-loop basin shows the
  asymmetry: displaced 20–40 cm *towards* the target the MLP 28 outputs 13–21% of the drive command, displaced
  *away* 4–7%. The GRU is the extreme case: it pauses 0.5–2.8 s with 19–33% of the drive command still on,
  creeps 1–7 cm and goes. The MLP 32 creeps too (7% while parked, 10–26 cm) but its goal-centred features keep
  the command at 2–3% up to 40 cm inside (seed 0), so it drifts 13–18 cm and mostly stays.

So the hold is lost through a different mechanism than the one §6h proposed: not an approach that misses the
box, but a parked regressor whose residual command is a slow forward creep, into a region of state space no
demonstration covers, where it extrapolates to "drive". It is the same compounding-error structure as the
copycat of §6g and the clock of §6h: a stateless regressor reads the consequences of its own small errors as
a new situation.

Consequences for the black-box line (decision pending):

1. **Data that covers the inside of the shell** is the clean remedy: demonstrations that start within a few
   decimetres of the goal with random heading (E2). From inside the box the expert latches `done` at once
   (zero wheels everywhere inside, the basin becomes a disc); from just inside the standoff radius it turns
   and drives back to the goal, which is the recovery a crept base needs. DART noise during the stop phase
   would do the same more cheaply if the collector can inject it after the stop (today's DART perturbs during
   driving; at 0.5 wheel noise the parked base is not displaced enough).
2. **A latched stop in the policy** (a stop head trained on the demonstrations' own wheel commands, with
   hysteresis) is the architectural remedy; it uses only demonstrated actions, not the expert's phases, but it
   is a design decision beyond the lecturer's MLP recipe.
3. The GRU keeps its two genuine gains (start-pose invariance, arm timing from history) and would benefit
   from either remedy as much as the MLP.

## 6j. E5 (30 Oct, 09:15–10:28): the transformer fits the demonstrations best and holds worst

Three causal transformers on the 28 pure transforms (§7 for the architecture), same mixture split as E4, same
two validations. Runs `tf_e5_mixture4096_raw21polar_stride20_seed{0,1,2}` (575 / 768 / 455 s on the GPU, best
epoch 67 / 97 / 49), evaluations `e5_tf_polar_on_{clean,e1_random}_validation_20261030`. Stable arrivals at
5 cm of 205, seeds 0 / 1 / 2:

| Model on the nominal + random mixture (4096) | offline NRMSE | Nominal: arrivals | parked | arm | held ≥ 10 s | Random: arrivals | parked | arm | held ≥ 10 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP 28 pure (control, §6i) | 0.145–0.149 | 23 / 30 / 27 | 119–126 | 3 / 103 / 15 | 0–5 | 11 / 15 / 17 | 33–68 | 3 / 105 / 11 | — |
| GRU 28 pure, memory 20 ms (§6h) | 0.136–0.153 | 32 / **102** / 27 | 13–44 | 149 / 101 / 16 | — | 29 / **99** / 26 | 7–12 | 155 / 96 / 14 | — |
| **Transformer 28 pure, attention 20 s** | **0.130–0.142** | 81 / 76 / 47 | 59 / 85 / 42 | 57 / 44 / 30 | 16 / 35 / 9 | 60 / 32 / 29 | 10 / 6 / 12 | 40 / 48 / 42 | 11 / 6 / 1 |
| MLP 32 (goal-based features, §6f) | 0.114–0.120 | 51 / 93 / 78 | 171–177 | 26 / 134 / 71 | — | 51 / **113** / 71 | 174–184 | 18 / 136 / 58 | — |

"Arrivals" here is the 5 cm criterion (EE within 5 cm of the target for 0.5 s at some point of the 40 s), the
one all the tables use; at 1 cm the transformer gets 25 / 29 / 8 nominal and 9 / 5 / 3 random. Median final
distance 1.5–1.7 m nominal, 1.6–1.9 m random: in tolerance at 40 s in 0–2 episodes, moving at 40 s in 162–193.

Three facts, in the order they were measured:

1. **Offline it is the best pure-input model.** Validation NRMSE 0.130–0.142 against 0.136–0.153 for the GRU
   and 0.145–0.149 for the MLP 28 on the same split, and the cleanest stop ever measured open loop: teacher-forced
   after the expert's stop the residual wheel command is 1–2% at 0.5, 2 and 5 s (MLP 28: 6–10%, GRU: 5–20%),
   and above 10% in 0–1% of the post-stop time. The clock does not move it: frozen at the stop pose and run to
   40 s, 2–5% (one seed 10% in the last ten seconds when frozen at the episode end). Attention over 20 s does
   what it was asked: it reads the stop from the history.
2. **Closed loop it reaches the box and leaves it, by the same creep as everything else, only larger.** On the
   evaluation traces (`creep_profile.py`) the transformer parks and restarts in 105–148 of 205 nominal episodes:
   parked 5.0–9.5 s, residual wheel command while parked 11–17% of its driving command (MLP 28: 6–10%), creep
   7.5–15.3 cm, towards the target in 93–100% of the episodes, heading fixed (4–12°), target distance at the
   park 0.73 m (the standoff shell). The arm is at hold when it restarts (0.020 rad rms, the expert's resting
   value), so again it is not the arm. Open loop the model is quiet because the teacher keeps it on the
   demonstrated shell; closed loop its own 11–17% pushes it inside the shell within seconds, where no token in
   its 20 s window has ever been seen. Attention does not generalise to an unseen token any better than an MLP
   generalises to an unseen input: the stop is learned as a function of the trajectory *so far*, and a
   trajectory that enters the shell was never demonstrated.
3. **It loses the GRU's start-pose invariance.** The GRU gave the same arrivals on nominal and random starts
   (32 / 102 / 27 vs 29 / 99 / 26); the transformer drops from 81 / 76 / 47 to 60 / 32 / 29, base parked 6–12 on
   random starts, median closest approach 0.26–0.41 m (nominal 0.07–0.08 m). With no positional encoding and
   attention weights that are functions of the token contents, the recurrent summary that the GRU builds (which
   does not care where the episode started) is replaced by a set of matches against the stored tokens, and a
   random start produces tokens the nominal half of the data never contained. The arm follows the base: extended
   in 30–57 nominal episodes (GRU 149 / 101 / 16, MLP 28 3 / 103 / 15), arrivals held ≥ 10 s in 1–35.

Where this leaves the memory question. The three memories tried on the mixture (velocity, GRU, attention) all
learn the stop open loop and all lose it closed loop through a residual command that was never zero because the
expert's zero is a latch, not a regression target; the memory with the longest horizon (attention) has the
smallest open-loop residual and the largest closed-loop one. What varies between them is the second-order
behaviour (the GRU's pose invariance, the transformer's precision) and none of it touches the hold. The hold is a
coverage problem of the training distribution (states inside the standoff shell, §6i remedy 1) or a structural
one (a latched stop head, remedy 2); it is not an architecture problem, and E5 is the third architecture that
says so. The MLP 32 holds for a different reason: its goal-centred inputs (distance and bearing to the
*expert's goal*, which encode the standoff rule) make "inside the shell" look like "at the goal", a shortcut the
pure inputs cannot offer.

Cost: 455–768 s per seed on the GPU (6 s per epoch, 18.7 GB), 46 µs per physics step at inference, so a
transformer is not expensive here; it is just not what is missing.

## 6k. Remedies for the hold without a third output (plan, 1 Nov)

The assignment brief fixes the regressor's outputs: arm commands and wheel commands. A separate stop head
(§6i remedy 2) is therefore out. What the brief leaves open, and asks about, is the data ("quantity and quality",
"extra sensors/input data"), the input window, the regressor architecture and the training methodology. The four
remedies below each live in one of those boxes, all on the 4096 mixture of §6f as base data, 3 seeds each, pure
28 inputs unless stated, both validations (205 nominal, 205 random starts) as in E4/E5. Decision (Rodrigo, 1 Nov):
run all of them.

| Arm | Box in the brief | What changes | Expected mechanism |
|---|---|---|---|
| **E7a** stop-weighted loss | training methodology | wheel loss of rows whose demonstrated wheels are exactly zero weighted ×8 (20% of rows) | the regressor's zero moves closer to zero; does not touch states outside the demonstrations |
| **E7b** L1 wheel loss | training methodology | L1 instead of L2 on the wheel outputs (arm stays L2) | L1 fits the conditional median, which is exactly zero where most of the mass is zero; L2 fits the mean, pulled by neighbouring driving rows |
| **E7c** categorical wheels | regressor architecture | the wheel output is a softmax over 41 velocity bins (−3…3, one bin exactly zero), command = argmax; arm unchanged; still two outputs | zero becomes a class that wins or loses, not a residual to integrate; nothing to creep with |
| **E7d** previous wheel command as input | extra input data | 30 inputs = 28 pure + the policy's own wheel command of the previous step (demonstrated command during training, own output in closed loop); anti-copycat: in training the two columns are shuffled across the batch in 25% of the rows and get N(0, 0.1) noise | a latch implemented through the input ("I was stopped, I stay stopped"); the copycat risk of §6g measured, not assumed |
| **E2** near-target starts | data | 2048 new demonstrations (2 per plan target) with the base placed 0.25–1.3 m from the target at a random heading, DART low as before; mixture 4096 + 2048 = 6144; trained with MLP 28, GRU 28 and E7c | the demonstrated stop states stop being a thin shell: from inside the shell the expert turns and drives back, from the box it latches at once |

Black-box check: every input is still a function of observations, the task target and the policy's own past
outputs; the stop-weight label is "demonstrated wheels are zero", read from the recorded actions, not from the
expert's phase. Placing the base near the target in the collector is a choice of the data designer, as the random
starts of E1/E3 were.

Order: E7a–d first (they only need the existing data, ~30 min per seed on the GPU), E2 collected in parallel
(~20 min for 2048 episodes at 24 workers), then the E2 arms, one trainer at a time. The code for all of it goes in
before anything is launched (provenance guards). Results in §6l onwards.

## 7. Status

- Done: decomposition (§1), variant `raw21_target_clock_v1`; E1 collected,
  trained (15 runs `runs/training/mlp_e1_*`), evaluated
  (`runs/evaluation/e1_mlp_*_seeds_on_clean_validation_20261027`,
  `…_on_e1_random_validation_20261027`,
  `dart_low_mlp_raw21plus_seeds_on_e1_random_validation_20261027`).
- E3: three of the six trainers were OOM-killed (the final full-dataset metric pass peaks at ~92 GB RSS on
  4096 episodes; two concurrent trainers exceed the 125 GB machine). Rule from now on: one 4096-episode trainer
  at a time. Relaunched 28 Oct 00:40 (`chain_e3c.sh`): the three complete seeds (random-only seed 0, mixture
  seeds 0 and 1) are being evaluated now; seeds random-only 1, 2 and mixture 2 retrain sequentially (~4.5 h) and
  are evaluated into `*_20261027_b` directories.
- E3 complete and reported in §6f (six models, two validations).
- The 21 lecturer inputs carry no base velocity (position and bearing only; the arm does get its joint
  velocities), so "the base is parked" is not observable from one frame.
- E4 launched 29 Oct 22:21 (`chain_e4.sh`, GPU, one trainer at a time), all on the nominal + random
  mixture split (`runs/splits/e3_mixture_4096.json`), 3 seeds each, evaluated on the 205 nominal and the
  205 random-start goals (`runs/evaluation/e4_*_20261029`):
  1. MLP 31 = 28 pure transforms + finite-difference base velocity (`raw21_targetpolar_vel_clock_v1`,
     commit `c3d67a4`): the one-step memory. Runs `mlp_e4_mixture4096_raw21polarvel_seed*`.
  2. GRU on the 28 pure transforms, memory stride 20 ms, TBPTT 1000 decisions (20 s): the long memory.
     Runs `gru_e4_mixture4096_raw21polar_stride20_seed*`.
  3. Control: MLP 28 without velocity on the same data, to separate the effect of the velocity from the
     effect of the mixture data. Runs `mlp_e4_mixture4096_raw21polar_seed*`.
  Nothing in these inputs comes from the expert: the velocity is the finite difference of the lecturer's
  own position and bearing inputs (what odometry gives a real robot), the GRU sees the same 28 inputs.
- E4 phase 1 done and reported in §6g (30 Oct 00:14): the finite-difference velocity is a copycat shortcut,
  37–40 arrivals.
- E4 phase 2 done and reported in §6h (30 Oct 01:38): GRU 28 pure on the mixture, 32 / 102 / 27 arrivals, the
  stop is learned open loop but the hold is lost in closed loop.
- E4 closed and reported in §6i (30 Oct 03:10): the control MLP 28 on the mixture gives 23–30 arrivals against
  51–93 for the MLP 32 on the same data; the lost hold is a measured creep (6–10% residual wheel command,
  6–9 cm towards the target in 8–12 s, then drive), not the arm, not the clock, not the approach. Remedies
  listed in §6i, decision pending.
- Decision 30 Oct (Rodrigo): keep the 4096 mixture as the base data for everything that follows; add E5 now.
- E5 launched 30 Oct 09:15 (`chain_e5.sh`, GPU, one trainer at a time): causal transformer over the 28 pure
  transforms (`src/robot_learning/learning/transformer_policy.py`, commit `d1c30ec`): one token every 20 ms,
  causal attention over the last 1000 tokens (20 s, the GRU's gradient horizon), 2 layers, d 64, 4 heads, FFN 128,
  no positional encoding (order from the causal mask, time from the clock input, so nothing extrapolates past the
  demonstration lengths), 1 ms reaction head on memory + current features as in the GRU. Same mixture split,
  3 seeds, runs `tf_e5_mixture4096_raw21polar_stride20_seed*`, evaluations `e5_tf_polar_on_{clean,e1_random}_validation_20261030`.
  NumPy inference with a key/value cache: 46 µs per physics step with a full window (GRU ≈ 120 µs); verified against
  the torch network to 1e-9 by test and at the end of every run. Black-box: same inputs as the GRU, nothing from the expert.
- E5 done and reported in §6j (30 Oct 10:28): transformer 28 pure on the mixture, best offline NRMSE of the pure
  models (0.130–0.142) and the cleanest open-loop stop (1–2% residual), 81 / 76 / 47 arrivals nominal, 60 / 32 / 29
  random; parked and restarted in 105–148 episodes by the §6i creep (11–17% residual while parked, 7.5–15 cm
  towards the target). Third architecture to show the hold is a coverage/latch problem, not a memory problem.
- 1 Nov: stop head ruled out (the brief fixes two outputs); four remedies planned in §6k (stop-weighted loss, L1
  wheels, categorical wheels, previous wheel command as input) plus E2 near-target starts. Decision: run all.
- E7 + E2 launched 1 Nov 15:10 (`chain_e7.sh`, commit `9fc5ec0`): E2 collection (2048 near-target starts, 24 workers) in the
  background; then on the GPU, one trainer at a time, 3 seeds each on the 4096 mixture: E7a stop_weight 8, E7b L1 wheels,
  E7c categorical wheels (41 bins), E7d previous wheel command input (shuffle 0.25, noise 0.1); then on the 6144 mixture
  (`runs/splits/e2_mixture_6144.json`): MLP 28 pure, categorical, GRU 28 pure. Every arm evaluated on the 205 nominal and
  205 random-start goals (`runs/evaluation/{e7a,e7b,e7c,e7d,e2}_*_20261101`). Trainer options are recorded in each run's
  `config.json` / `model_info.json` (`egocentric_options`). Expected to finish 2 Nov morning.
- 28-input pure-transform variant done and reported in §6e (branch `rodrigo-pure28` merged).
- Next, on request: E6 (§6c, nominal data without the clock) — the cheapest test of the reach-trigger mechanism.
- E2 needs a collector mode; nothing in `src/` is edited while a collector,
  trainer or evaluator is running (provenance guards).
