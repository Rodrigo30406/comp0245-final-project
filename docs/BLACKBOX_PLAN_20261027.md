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

## 6l. E7a and E7b (1 Nov, 15:10–20:13): fixing the zero where there is data does not fix the creep

Three MLP-28 with the wheel loss weighted ×8 on the rows whose demonstrated wheels are exactly zero (27.6% of the
training rows; `mlp_e7a_mixture4096_polar_stopw8_seed*`) and three with L1 instead of L2 on the wheels
(`mlp_e7b_mixture4096_polar_l1_seed*`), same 4096 mixture, same validations. Stable arrivals at 5 cm of 205:

| Model on the 4096 mixture | offline NRMSE | open-loop residual after the expert's stop | Nominal: arrivals | parked | arm | moving at 40 s | Random: arrivals | parked | arm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP 28 pure (control, §6i) | 0.145–0.149 | 1–5% | 23 / 30 / 27 | 119–126 | 3 / 103 / 15 | 199–203 | 11 / 15 / 17 | 33–68 | 3 / 105 / 11 |
| **E7a** stop weight ×8 | 0.148–0.150 | **0–1%** | 19 / 44 / 30 | 93–131 | 7 / 113 / 2 | 196–205 | 14 / 38 / 16 | 39–45 | 3 / 107 / 0 |
| **E7b** L1 wheels | 0.149–0.150 | **0–1%** | 20 / 18 / 9 | 58–131 | 148 / 25 / 136 | 202–204 | 8 / 7 / 7 | 21–78 | 144 / 27 / 125 |

Both remedies do exactly what they were designed to do and it changes nothing in closed loop:

- **Open loop the zero is now zero.** Teacher-forced after the expert's stop, the residual wheel command is 0–1% at
  0.5, 2 and 5 s for all six models (control: 1–5%, one seed 5% at 5 s). Frozen at the stop pose and run to 40 s:
  1–2% up to 20 s, 3–9% in the last ten seconds (the clock, as in §6i). The fit on the demonstrated stop states is
  as clean as the transformer's (§6j).
- **Closed loop the creep is unchanged.** On the evaluation traces (`creep_profile.py`) the E7a models park and
  restart in 62–168 nominal episodes with 6–14% of their driving command still on while parked, creeping 3.6–10 cm
  towards the target in 90–99% of the cases; E7b: 33–127 episodes, 7–16%, 5–7.6 cm. Control: 120–142 episodes,
  6–8%, 5.8–9.2 cm. The best seed (E7a seed 2: 6%, 3.6 cm, parked 10.7 s) still leaves in 168 episodes; it only
  takes longer. Moving at 40 s in 196–205 of 205.
- **Why.** The loss weighting and the L1 act on the training rows, and every training row sits on the demonstrated
  stop shell (base at the standoff radius, arm at hold). The parked base in closed loop is a few centimetres and
  degrees off that shell from the first instant (its own approach error), and the 6–16% it outputs there is an
  extrapolation that no loss term saw. Making the fitted zero more exact on the shell sharpens the function on the
  shell; it says nothing about the function 5 cm inside it. The remedy has to put data (E2) or structure that is
  insensitive to small displacements (E7c categorical, E7d latch through the input) where the robot actually ends up.
- Arrivals move with the arm, not with the base: E7b extends the arm in 148 / 25 / 136 nominal episodes (control
  3 / 103 / 15), E7a in 7 / 113 / 2, and the 5 cm arrivals follow the seed-to-seed arm behaviour rather than the
  wheel loss. The arm head shares the hidden layers with the wheels, so a change in the wheel loss reshuffles which
  seed happens to learn the arm timing (§6c: the arm reads "base parked" from the posture it never changes). That
  variance is a separate problem the E2 data should also reduce, by giving the arm more demonstrations of the
  extension from varied base poses.

E2 collection, same window (15:10–18:49, 24 workers): 2048 near-target starts, 1775 expert successes and 273
stalls (13.3%; the expert's navigation times out at 25 s). The same stall exists in the base data at the same rate
(nominal DART-low 129 / 1024 = 12.6%, E3 random starts 149 / 3072, E1 43 / 1024) and every mixture so far has
kept the stalled episodes (`keep_expert_failures`), so E2 is treated the same: `runs/splits/e2_mixture_6144.json`
holds 4914 training / 1230 validation episodes over the three datasets, validation = the 205 held-out geometries
from every start pose, as before. Successful E2 episodes last 11.1 s median (p95 15.2 s), start 0.25–1.3 m from
the target (median 0.80 m).

## 6m. E7c (1 Nov, 19:14–22:09): categorical wheels remove the creep and replace it with chatter and a clock-gated stop

Three MLP-28 whose wheel head is a 41-way classifier per wheel (bins of 0.156 over the demonstrated range ±3.12,
cross-entropy, argmax at inference; `mlp_e7c_mixture4096_polar_cat41_seed*`, 31,705 parameters), same 4096 mixture,
same two validations. Offline NRMSE 0.170 / 0.162 / 0.161 (control 0.145–0.149: the quantisation costs 0.015–0.025).
Stable arrivals at 5 cm of 205:

| Model on the 4096 mixture | open-loop residual after the stop | clock frozen at the stop, 20–30 s | Nominal: arrivals | parked | arm | moving at 40 s | stopped outside tolerance | Random: arrivals | parked | arm | moving at 40 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP 28 pure (control, §6i) | 1–5% | 3–9% | 23 / 30 / 27 | 119–126 | 3 / 103 / 15 | 199–203 | 2–6 | 11 / 15 / 17 | 33–68 | 3 / 105 / 11 | 199–204 |
| **E7c** categorical 41 bins | **0%** | **0–2%** | 39 / 24 / 32 | 103–105 | 124 / 60 / 165 | **103 / 89 / 152** | **93 / 114 / 46** | 31 / 25 / 31 | 79–107 | 137 / 68 / 161 | 106 / 72 / 131 |

The arrivals do not move (39 / 24 / 32 against 23 / 30 / 27), but the failure changes shape: for the first time half of
the robots are **stopped** at 40 s, just not in the right place (median distance to the base goal 0.39–0.78 m, after
having passed within 2–3 cm of it). The traces say why, in three measurements (`scripts/analysis/restart_trigger.py`,
`restart_cause.py`, `zero_by_clock.py`; logs in the scratchpad `e7c_*.log`):

- **The zero bin is never chosen near the goal.** While the base is "parked" (speed < 0.02 m/s, 111–123 nominal
  episodes), 100% of the sampled wheel outputs are non-zero bins: median 0.47 (three bins from zero), p90 0.62–0.70,
  with the sign changing from sample to sample (`[+0.47, +0.62]`, `[−0.47, −0.47]`, `[−0.47, +0.62]`, `[+0.47, −0.47]`
  in consecutive 100 ms samples of episode 0). The base does not move because the commands cancel on average; the
  creep is 1.1–1.4 cm (control 6–9 cm), heading change 1.3–2.5°. In open loop, on the demonstrated stop states, the
  same models output exactly zero 100% of the time, so this is again the off-shell state: a few centimetres inside
  the standoff radius the posterior over the 41 bins has no dominant mode, and the argmax jumps between the small
  "orient"-like corrections of either sign (the expert's orient commands are 0.76–3.0 in magnitude, p10–p90).
- **The restart is not a drift of a zero; it is a run of same-sign chatter.** Re-running the policy from the
  parked instant on three input streams (observation held and clock running; real observations with the clock
  frozen; full replay), the 0.5 s moving average of the wheel command exceeds a drive level of 0.3 within 0.4 s on
  all three streams in 91–100% of the episodes: the parked output is at drive level from the first instant, in
  any of the three decompositions. The recorded restart happens after 2.3–6.0 s, when the signs stop cancelling
  (`[+0.78, +0.47]` for 2 s in episode 0: a slow forward arc). The control's regression restart behaves differently:
  its output is a small residual (duty cycle above 0.3: 5–8%) that grows with the clock and drifts the base for
  6–9 s before the 0.3 level is reached (clock-only stream reproduces the restart in 59–91% of episodes at 6–9 s).
- **The stop that finally happens is the 25 s rule of the stalled demonstrations.** Episodes that end stopped
  outside tolerance stop at a median clock of 25.2 s (seed 0; 55% within 24–27 s) and 26.1 s (seed 1), 0.22–0.28 m
  from the target, i.e. inside the standoff shell (0.65–0.80 m), where the chatter run had driven them. The zero bin
  is output in 2–6% of the samples between 12 and 24.5 s and in 34–49% after 25.5 s. In the training split, every
  row with clock ≥ 25 s is a stalled episode whose wheels are exactly zero (114k rows, 100%; the expert's navigation
  timeout), and rows of successful episodes are exactly zero in 1% (0–5 s), 23% (5–10 s), 53% (10–15 s), 78%
  (15–20 s) and 100% (20–25 s, 7k rows) of the cases. A classifier learns the hard boundary "clock ≥ 25 s → zero"
  that the data presents, independent of the base's position; the regression models never output an exact zero
  there (0% of samples) and keep moving at 40 s in 199–204 episodes. Seed 2 is in between (stop at 30.4 s median,
  0.70 m from the target, 152 still moving).
- The arm fires more often (124 / 60 / 165 nominal, control 3 / 103 / 15) because the base is still more often and
  for longer: the arm's cue is the posture (§6c), and a chattering base holds that posture. "Both parked and extended"
  reaches 98 / 41 / 90 episodes, but the base then leaves, so the stable arrivals stay at 24–39.

Conclusion for E7c: the categorical head does solve the exact problem it was built for (no small residual: the zero
is exactly zero wherever the model chooses it) and exposes the real one more clearly than the regression did. Off the
demonstrated shell the model has no basis for choosing zero, whatever the output parametrisation; it chooses zero
only where the data forces it, and in this data that place is the clock, not the pose. The fix remains data in the
off-shell states (E2, phase B, which also runs the categorical head on the 6144 mixture so the two effects can be
separated) or an input that carries the latch (E7d).

## 6n. E7d (1 Nov, 20:13–00:14): the previous wheel command as input gives either a copycat or nothing

Three MLP-30 = the 28 pure transforms + the policy's own previous wheel command (right, left), trained with the
anti-copycat augmentation (previous command swapped across rows with p = 0.25, Gaussian noise 0.1 in normalised units;
`mlp_e7d_mixture4096_prevwheel_seed*`), same 4096 mixture, same validations. Offline NRMSE 0.108 / 0.104 / 0.105, the
best of every arm so far (control 0.145–0.149). Stable arrivals at 5 cm of 205:

| Model on the 4096 mixture | offline NRMSE | Nominal: arrivals | parked | arm | moving at 40 s | Random: arrivals | parked | arm | moving at 40 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP 28 pure (control, §6i) | 0.145–0.149 | 23 / 30 / 27 | 119–126 | 3 / 103 / 15 | 199–203 | 11 / 15 / 17 | 33–68 | 3 / 105 / 11 | 199–204 |
| **E7d** + previous wheel command | **0.104–0.108** | **11 / 19 / 16** | 38–66 | 1 / 5 / 67 | 203–205 | 12 / 7 / 7 | 24–39 | 0 / 28 / 79 | 204–205 |

The worst closed-loop result of the programme with the best offline number. Measurements (`scripts/analysis/
prevwheel_copycat.py`, `prevwheel_loop_gain.py`, `wheel_sensitivity.py`; logs `e7d_*.log` in the scratchpad):

- **The input is a perfect predictor that the model learned to ignore.** On teacher-forced validation rows a linear
  map recovers the expert's wheel command from the previous command alone with R² = 0.999 (from the other 28 features:
  0.46 / 0.51): at 1 ms the command almost never changes, so without augmentation the regressor would become a pure
  copycat (§6g, the velocity). The augmentation did its job: with the previous command zeroed on driving rows the
  policy keeps 97–98% of its command; with the median driving command injected into parked rows it still outputs
  0.02–0.03; the previous command carries 3% of the total wheel sensitivity while driving and 0% while parked; and
  the sensitivity profile is otherwise the control's (target geometry 70–80% of the parked-row sensitivity, arm
  posture 13–22%). In closed loop, re-running the policy at the arrival states with its own previous command and with
  the previous command zeroed differ by 0.1 (1.10 against 1.00), and d(output)/d(previous command) is 0.13–0.17
  (p90 0.29–0.36): no latch formed, and no unstable action loop either.
- **So the closed loop is the control's off-shell extrapolation, only shallower.** The base reaches the goal
  (185–199 of 205 within 10 cm, median closest 0.03–0.04 m, as the control) and slows to 0.05–0.06 m/s, but the
  command around the arrival instant is 0.81–1.13 (control 0.45–0.58) and the lowest 1 s mean |wheel| in an episode
  is 0.38–0.63 (control 0.06–0.14; below 0.1 in 12–20 episodes against 76–119). It re-accelerates within 1–2 s and
  leaves: 38–66 parks (control 120–142), 203–205 moving at 40 s, the arm fires in 1 / 5 / 67 episodes because the
  posture cue never appears. The residual that the regression leaves off the shell is two to three times the
  control's; the training rows it was fitted on are the same, so the extra input (with its augmentation noise)
  simply gave the stop rows a slightly different, equally off-shell, function.
- **Open loop everything is fine**, as for every arm: teacher-forced residual after the expert's stop 1%, sequential
  on the policy's own outputs over the demonstration's observations 1%, 33–38 of 40 sequential runs reach a stop.
  Offline NRMSE 0.104 is the fourth time (§6c velocity, §6g, §6j transformer, here) that the offline metric moves
  opposite to the closed-loop result.

Conclusion for E7d: with a single regressor and the demonstrations as the only data, an action input has two
regimes and no useful middle. Unaugmented it reproduces the previous command (R² 0.999) and the stop becomes an
accident of the first zero; augmented until the copycat disappears, the model stops using it and nothing is latched.
A latch has to be a function of the state history (the GRU's hidden state, phase B) or of data that covers the states
where the decision is actually taken (E2). E7d is closed.

## 6o. E2 on the MLP 28 (1 Nov 22:09 – 2 Nov 02:34): near-target starts teach "rotate" inside the tolerance box

Three MLP-28 on the 6144 mixture (nominal 1024 + random starts 3072 + near-target starts 2048; split
`runs/splits/e2_mixture_6144.json`, 4914 / 1230 episodes; `mlp_e2_mixture6144_polar_seed*`), same two validations.
Offline NRMSE 0.148 / 0.151 / 0.150 (the 4096 control: 0.145–0.149). Stable arrivals at 5 cm of 205:

| MLP 28 pure | data | Nominal: arrivals | within 10 cm of the base goal | parked | arm | moving at 40 s | Random: arrivals | within 10 cm | parked | moving at 40 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control (§6i) | 4096 | 23 / 30 / 27 | 195 / 198 / 190 | 119–126 | 3 / 103 / 15 | 199–203 | 11 / 15 / 17 | 137 / 137 / 123 | 33–68 | 199–204 |
| **E2** | 6144 | **14 / 17 / 13** | 150 / 156 / 185 | 42–67 | 1 / 31 / 48 | 198–203 | 12 / 10 / 9 | 112 / 120 / 134 | 35–47 | 197–203 |

More data, from the region where the robot fails, made the base stop less often (42–67 parks against 120–142) and
even reach the goal less often on the nominal starts. Measurements (`scripts/analysis/arrival_profile.py`,
`arrival_heading.py`, `zero_by_park_distance.py`, `stop_basin.py`; logs `e2_*.log`):

- **The stop basin on the demonstrated states is unchanged.** Displacing the demonstrated stop pose (teacher-forced,
  arm at hold) gives 2% |wheel| inside the 8 cm box, 2–5% in the 8–15 cm ring, 2–3% with a yaw offset: identical to
  the control. The data did not narrow the stop; it changed what the model does on the way in.
- **The residual at the arrival doubled, and the new part is a rotation.** In the second around the closest approach
  the wheel command is 0.73–0.83 (control 0.45–0.58). Decomposed into a drive (right + left)/2 and a rotation
  (right − left)/2: control 0.28–0.42 drive, 0.20–0.23 rotation; E2 0.45–0.56 drive, **0.43–0.49 rotation**. The
  heading error to the target at that instant is 1.1–2.2° median, inside the expert's 2.9° tolerance for 130–160 of
  the 205 episodes; the model rotates anyway. The lowest 1 s mean |wheel| in an episode is 0.12–0.39 (control
  0.06–0.14; below 0.1 in 17–87 episodes against 76–119). Where the base does park the creep is the control's
  (10–12% of the driving command, 7–15 cm towards the target).
- **What the near-target data says inside the tolerance box.** Successful episodes, rows binned by distance between
  the base and the expert's own base goal (`base_goal_xy`), rows within 8 cm of it:

  | dataset | rows within 8 cm of the goal | of which wheels exactly zero | non-zero rows: median \|wheel\| | phase |
  |---|---:|---:|---:|---|
  | nominal 1024 | 25% of all rows | **82%** | 1.40 | 100% orient |
  | E3 random starts | 22% | 63% | 3.00 | 100% orient |
  | E2 near-target starts | 24% | **44%** | **3.00** | 100% orient |

  A near-target start has a random yaw and often lies on the far side of the target, so the expert reaches its base
  goal facing the wrong way and spends seconds rotating in place at full speed (the `orient` phase, |wheel| = 3.0)
  before the latch; in E2 more than half of the rows inside the tolerance box are such rotations, against 18% in
  the nominal data. In the 6144 mixture the "inside the box" rows went from roughly 70% zero to 60% zero with the
  non-zero mode at saturation. The expert resolves this with its 0.05 rad heading test and a latch; a regressor
  fitted to these rows learns a blend "inside the box → some rotation" whose weight grows with the share of
  rotation rows, and that blend is what appears at the closed-loop arrival with a 1–2° heading error. The same data
  also added short drives from arbitrary headings around the goal (29–38% drive rows at 8–20 cm), which is where
  the nominal approach got worse (150–185 within 10 cm).
- Offline the three models are as good as the control (0.148–0.151) and the open-loop stop is as clean: the fourth
  arm in a row whose offline metric does not see the closed-loop change.

Conclusion for E2: coverage of the near-goal region does not help if the expert's behaviour there is dominated by a
rotation that a 0.05 rad threshold and a latch turn off. The demonstrations contain the decision but not its
boundary at a resolution the regressor can learn from 1 ms rows (the stop rows and the rotate rows differ by a
heading of a few degrees and by the latch state, which is not in the inputs). The E2 categorical models (same data,
evaluating) will say whether the argmax sharpens that boundary; the GRU on this mixture (the only arm with a state
that can hold the latch) is relaunched below.

**Why the box fills with rotations (measured 2 Nov, 354 E2 and 339 nominal successful episodes).** The expert's goal is
0.65 m from the target in a direction that does not depend on where the robot comes from, and the expert drives facing
the goal, not the target. From the nominal start pose the line robot→goal happens to point almost at the target, so the
base enters the 8 cm box with a heading error of 8° median and the `orient` spin is short (27% of the in-box rows are
moving). From the E2 ring (0.25–1.3 m from the target, any direction) the approach comes from the side or from inside
the standoff radius, and the base enters the box with a heading error of **141° median (84% above 90°)**: it then spins
at full speed inside the box for ~2.4 s (74% of the in-box rows are moving; 81% for starts inside the standoff radius,
69% for starts outside). The rotation is not an artefact of the random initial yaw (the `align` phase fixes that at the
start); it is the expert's own `orient` phase, which E2 makes long exactly at the place where the stop should be
demonstrated.

## 6p. E2 on the categorical head (2 Nov 00:14–03:17): the rotation rows reach the argmax too

Three MLP-28 with the 41-bin wheel head on the 6144 mixture (`mlp_e2_mixture6144_polar_cat41_seed*`), offline NRMSE
0.157 / 0.160 / 0.161 (E7c on 4096: 0.161–0.170). Stable arrivals at 5 cm of 205:

| categorical 41 bins | data | Nominal: arrivals | parked | arm | moving at 40 s | stopped outside | Random: arrivals | parked | moving at 40 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E7c (§6m) | 4096 | 39 / 24 / 32 | 103–105 | 124 / 60 / 165 | 103 / 89 / 152 | 93 / 114 / 46 | 31 / 25 / 31 | 79–107 | 106 / 72 / 131 |
| **E2 categorical** | 6144 | **20 / 14 / 23** | 69 / 57 / 112 | 27 / 41 / 90 | 199 / 179 / 181 | 6 / 26 / 23 | 20 / 15 / 30 | 63–93 | 190 / 176 / 164 |

Same direction as the regression (§6o): the near-target data costs arrivals (20 / 14 / 23 against 39 / 24 / 32), parks
and arm extensions. The probes (`arrival_profile`, `arrival_heading`, `creep_profile`, `restart_trigger` on
`e2_mlp_cat41_on_clean_validation_20261101`):

- The arrival residual is the E2 regression's: 0.70–0.95 around the closest approach, rotation component 0.40–0.49,
  heading error at arrival 1.2–2.5° median. The argmax does not sharpen the boundary; with 56% of the in-box rows
  being full-speed rotations, the rotation bins win the argmax more often than before, so the chatter of §6m
  (100% non-zero samples while "parked", first non-zero bin 0.47–0.55, 2 cycles median, up to 6) now resolves into
  a drive sooner (restart after 1.6–4.9 s) and the base ends moving in 179–199 episodes.
- The 25 s clock stop of §6m almost disappears (stopped outside tolerance 6–26 against 46–114): the E2 episodes are
  short (11 s median) and the mixture's rows after 25 s are now a smaller share of what the zero bin sees, so the
  classifier's "clock → zero" rule is weaker; it was never the stop we wanted anyway.

E2 is closed for the MLP family: on either head, data from the region where the robot fails makes the failure worse,
because the expert's behaviour there is the orient rotation and not the stop. The GRU on the same mixture is the one
remaining arm (relaunched below with the memory fix).

## 6q. E2 on the GRU (2 Nov 03:20–04:36): the memory holds the stop open loop, and the near-target data moves the approach away from it

Three GRU 28 pure (hidden 128, head 128, memory stride 20 ms, TBPTT 1000 decisions = 20 s) on the 6144 mixture
(`gru_e2_mixture6144_polar_stride20_seed*`), run alone after the memory fix (§7), 9–13 min each on the GPU, peak 43 GB RSS.
Offline NRMSE 0.132 / 0.123 / 0.127: the best of every pure-28 model of the programme (GRU on 4096: 0.136–0.153,
transformer 0.130–0.142, MLP 28 0.145–0.151). Evaluations `e2_gru_polar_on_{clean,e1_random}_validation_20261101`.
Stable arrivals at 5 cm of 205:

| GRU 28 pure, memory 20 ms | data | offline NRMSE | Nominal: arrivals | parked | arm | moving at 40 s | within 10 cm of the base goal | median closest approach | Random: arrivals | parked | moving at 40 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E4 (§6h) | 4096 | 0.136–0.153 | 32 / **102** / 27 | 13–44 | 149 / 101 / 16 | 103–196 | 64 / 111 / 106 | 0.10–0.18 m | 29 / **99** / 26 | 7–12 | 100–190 |
| **E2 GRU** | 6144 | **0.123–0.132** | **41 / 43 / 37** | 11 / 10 / 4 | 117 / 203 / 85 | 147 / 132 / 181 | 48 / 47 / 50 | 0.23 / 0.44 / 0.18 m | 39 / 53 / 28 | 8–18 | 134 / 94 / 168 |
| ref. MLP 28 control (§6i) | 4096 | 0.145–0.149 | 23 / 30 / 27 | 119–126 | 3 / 103 / 15 | 199–203 | 190 / 198 / 195 | 0.02–0.04 m | 11 / 15 / 17 | 33–68 | |
| ref. transformer (§6j) | 4096 | 0.130–0.142 | 81 / 76 / 47 | | | | 147 / 160 / 120 | | 60 / 32 / 29 | | |

The seed spread collapses (41 / 43 / 37 against 32 / 102 / 27) and the mean does not move. The probes
(`openloop_phase_error`, `stop_basin`, `clock_extrapolation --freeze-at stop` on the runs; `arrival_profile`,
`arrival_heading`, `creep_profile`, `restart_trigger` on the traces; logs `e2gru_*.log`) separate what the memory
does from what the data did:

- **Open loop, the GRU has the latch — in its state, not in the pose.** Teacher-forced on 40 validation
  demonstrations the residual wheel command after the expert's stop is 1% (0% of the post-stop time above 10%; drive-phase
  error 0.05–0.06, as on 4096). New measurement: `stop_basin` feeds the demonstration history up to the stop and then
  displaces the *current* base pose. For the GRU the wheel command stays at **1–4% of the drive level for every
  displacement, up to 40 cm towards or away from the target and ±0.1 rad of yaw, in all three seeds**. The stateless
  MLP 28 on the same data rises to 13–23% at 40 cm towards the target (§6i: the off-shell extrapolation that fires
  the restart). Once the history says "stopped", the recurrent state keeps the wheels at zero whatever the pose
  reads: this is the latch the regression on the current frame cannot represent, and the trained network has it.
  With the observation frozen at the stop and the clock run to 40 s: 3–14%, no clock restart.
- **Closed loop, the GRU does not pass through the state that sets the latch.** It comes within 10 cm of the base goal
  in 47–50 episodes of 205 (E4 GRU 64–111, MLP 28 control 190–198, transformer 120–160); its median closest approach is
  18–44 cm against the expert's 8 cm box (E4 GRU 10–18 cm, control 2–4 cm). It is still moving at 40 s in 132–181
  episodes. The latch was learned from the demonstrations' own approach histories; a drive-phase error of 0.05–0.06
  integrated over 10–20 s of closed loop puts the policy's trajectory 20–40 cm from where those histories end, and
  the history it then has is not one the training set followed with a stop.
- **The near-target data moved the arrival from "stop" to "rotate", as for the MLP (§6o).** Around the closest approach
  the wheel command is 0.70 / 0.69 / 1.06 with a rotation component of 0.51 / 0.65 / 0.94 and a drive component of
  0.05–0.07, at a heading error to the target of 1.5–2.0° median (inside the 2.9° tolerance). E4 GRU: rotation
  0.99 / 0.10 / 0.30. Within 10 cm of the base goal the count halves from the E4 GRU (64–111 → 47–50): the E2
  episodes (short, starting inside the standoff radius, 56% of their in-box rows full-speed `orient` rotations) teach the
  memory that a short history ending near the target continues as a rotation, not as a stop.
- **When it does pause, it pauses like the §6i GRU.** 87–125 park→restart events per seed, at 0.72–0.74 m from the
  target (the standoff shell), with 12–19% of the drive command still on, a creep of 1.7–5.8 cm towards the target
  (71–100% of the events) and a restart after 1.4–2.5 s; the arm did not move before the restart (0.004–0.012 rad).

What this settles. The hold problem has two halves and the two architectures now hold one each: the stateless MLP 28
reaches the box (190–198 within 10 cm) and cannot latch (its "zero" is a 6–10% creep that extrapolates to "drive" a
few centimetres inside the shell); the GRU latches (basin flat to 40 cm) and does not reach the box (47–111 within
10 cm). Neither half is fixed by the data we can collect from the black-box expert: random starts (E1/E3) gave the
mixture its long clock support, near-target starts (E2) demonstrate the expert's `orient` rotation rather than its
stop and make both the MLP and the GRU rotate at the arrival. And for the seventh time the offline metric points the
wrong way: the E2 GRU is the best fit of the programme (0.123) and arrives 41 / 43 / 37 times.

The E7 + E2 programme is complete; §8 gathers the whole picture.

## 8. Closing synthesis (2 Nov 04:50): what the black-box programme established

Every arm below was trained three times (seeds 0, 1, 2) and evaluated closed loop on the same 205 held-out geometries
from the nominal start pose ("nominal") and from random start poses ("random", the E1 bank). Numbers are stable
arrivals at 5 cm of 205, seed by seed; the expert makes 205 / 201. Full per-run table: `python3 -I scripts/analysis/final_table.py`.

| Arm | inputs | data | offline NRMSE | nominal | random | measured mechanism |
|---|---|---|---:|---|---|---|
| MLP 22 (lecturer's 21 + clock) | 22 | nominal 1024 | 0.17 | 5–10 (5 seeds) | — | no goal geometry in the inputs: the base never finds the standoff (§1) |
| MLP 25 (+3 target in base frame) | 25 | nominal 1024 | 0.135 | 25–32 (5 seeds) | — | base arrives, arm never leaves the hold posture (§6c) |
| MLP 28 pure (+3 polar) | 28 | nominal 1024 | 0.13 | 33–129 (5 seeds) | — | unlocks base and arm; clock extrapolation restarts the base after 15 s (§6e, §6h) |
| MLP 32 (+10 goal-based features, "cheat") | 32 | nominal 1024 | 0.115 | 115–162 (5 seeds) | 24–46 | goal-centred features keep the command at 2–3% inside the shell (§1, §6i) |
| MLP 32 | 32 | mixture 4096 | 0.094–0.102 | 51 / 93 / 78 | 51 / 113 / 71 | start-pose invariance bought with nominal arrivals (§6f) |
| **MLP 28 pure (control)** | 28 | mixture 4096 | 0.145–0.149 | 23 / 30 / 27 | 11 / 15 / 17 | reaches the box (2–4 cm), creeps 6–9 cm inside the shell, extrapolates to drive (§6i) |
| MLP 31 (28 + FD base velocity) | 31 | mixture 4096 | 0.106–0.108 | 40 / 38 / 37 | 39 / 36 / 33 | copycat of its own last motion (§6g) |
| GRU 28, memory 20 ms | 28 | mixture 4096 | 0.136–0.153 | 32 / 102 / 27 | 29 / 99 / 26 | has the stop open loop, approach 10–18 cm off (§6h) |
| Transformer 28, attention 20 s | 28 | mixture 4096 | 0.130–0.142 | 81 / 76 / 47 | 60 / 32 / 29 | best fit, same creep (11–17% while parked) (§6j) |
| E7a: stop rows weighted ×8 | 28 | mixture 4096 | 0.148–0.150 | 19 / 44 / 30 | 14 / 38 / 16 | open-loop residual 0–1%, closed-loop creep unchanged (§6l) |
| E7b: L1 wheel loss | 28 | mixture 4096 | 0.149–0.150 | 20 / 18 / 9 | 8 / 7 / 7 | same (§6l) |
| E7c: categorical wheels, 41 bins | 28 | mixture 4096 | 0.161–0.170 | 39 / 24 / 32 | 31 / 25 / 31 | exact zero exists; replaced by ±3-bin chatter and a 25 s clock-gated stop (§6m) |
| E7d: + previous wheel command | 30 | mixture 4096 | 0.104–0.108 | 11 / 19 / 16 | 12 / 7 / 7 | copycat (R² 0.999) or, augmented, ignored (3% sensitivity) (§6n) |
| E2: MLP 28 pure | 28 | mixture 6144 (+near-target) | 0.148–0.151 | 14 / 17 / 13 | 12 / 10 / 9 | in-box rows become `orient` rotations; arrival residual doubles (§6o) |
| E2: categorical 41 bins | 28 | mixture 6144 | 0.157–0.161 | 20 / 14 / 23 | 20 / 15 / 30 | same rotation reaches the argmax (§6p) |
| E2: GRU 28, memory 20 ms | 28 | mixture 6144 | 0.123–0.132 | 41 / 43 / 37 | 39 / 53 / 28 | latch in the state (basin flat to 40 cm), approach 18–44 cm off, rotates at arrival (§6q) |

What is established, with the measurement that establishes it:

1. **The lecturer's 21 inputs plus a clock cannot encode the task** (5–10 arrivals, §1): the goal is a function of the
   target and the standoff that the regressor must reconstruct from absolute coordinates. Three rigid-frame transforms of
   the lecturer's own inputs (target in the base frame) give the base; three more (range, bearing, standoff-relative range)
   give the arm as well (§6c, §6e). These six features use nothing from the expert.
2. **The hold is a latch, and a regression on the current frame cannot represent a latch.** The expert's stop is a
   phase bit set once, with zero wheels afterwards whatever the pose does. Every stateless model's "zero" is a residual
   of 6–10% of the drive command (§6i), which integrates into a creep towards the target that leaves the thin shell of
   demonstrated stop states in 8–12 s and extrapolates to "drive". Weighting the stop rows (E7a), an L1 loss (E7b) and an
   exact-zero categorical bin (E7c) all give an open-loop residual of 0–1% and leave the closed-loop failure in place,
   because the failure happens in states no demonstration covers. Feeding the policy its own previous command (E7d)
   gives it the latch only as a copycat that cannot be stabilised.
3. **Memory represents the latch but not the approach.** The GRU keeps its wheels at 1–4% for any displacement up to
   40 cm once its history says "stopped" (§6q), and brings the arm back on random-start data (§6h); but its closed-loop
   approach passes 10–44 cm from an 8 cm box, so the latch is rarely set. The transformer fits best and creeps like the MLP.
4. **The data we can collect from the black-box expert do not fill the gap.** Random starts (E1, E3) give long clock
   support and start-pose invariance at the cost of nominal arrivals; the mixture is the right base. Near-target starts
   (E2), designed to demonstrate the stop from many approach states, demonstrate the expert's `orient` rotation instead
   (56% of the in-box rows are full-speed rotations, §6o), and every model trained on them rotates at the arrival.
   Scale alone (1024 → 4096) does not change the ranking (§6f).
5. **The offline validation metric is not a selection criterion for this task.** In seven comparisons the best offline
   fit was the worst or an unchanged closed-loop policy (E7d 0.104 → 11–19 arrivals; E2 GRU 0.123 → 41–43; MLP 32
   mixture 0.094 → 51–93 against 115–162 for 0.115 on nominal data). Model selection must be closed loop.
6. **The best black-box policy of the programme remains the GRU seed 1 on the 4096 mixture (102 / 99)** and the best
   pure-28 family on average is the transformer (81 / 76 / 47); the best policy overall is the MLP 32 on nominal data
   (115–162), whose 10 extra features encode the expert's goal construction and were therefore set aside as
   privileged (§0, §1).

The stop region, the demonstrated zero rows and the policy's creep are drawn in `STOP_ZONE_EXPLAINED.md`
(`scripts/analysis/stop_zone_figures.py`, `docs/figures/stop_zone/`).

Open decisions (Rodrigo's):

- **E0, the white-box contrast**: a single latched `arrived` bit from the expert's phase as an extra input would show
  how many arrivals the latch alone is worth. It is privileged information and was excluded on the black-box rule;
  it is cheap (one collector field, one input variant) and would make the report's argument quantitative.
- **DAgger / expert relabelling** of the policy's own closed-loop states is the textbook fix for both halves of the
  problem (the crept states inside the shell and the GRU's off-track approaches would get the expert's `done` or
  `drive` labels). It was excluded by Rodrigo's decision at the start of the programme; the measurements above are the
  argument for or against reopening it.
- **Reserved test banks** (`datasets/test_reserved/*`) have never been used for training or selection; evaluating the
  chosen policies on them is the last step before the report and is Rodrigo's call.
- **E6** (nominal data without the clock, §6c) is the cheapest remaining test of the reach-trigger mechanism and does
  not depend on the hold.

## 6r. Is the policy's "zero" separable from its real commands? (2 Nov 06:50)

Rodrigo's question: when the expert sends zero, what is the largest command the policy sends, and does it overlap with
the smallest command the policy sends when the base has to move? Measured by `scripts/analysis/zero_separability.py`
(40 validation demonstrations teacher-forced per run; closed-loop traces of the clean validation), |wheel| = max of the
two wheels in rad/s. The expert's own commands have a floor: **no non-zero wheel command below 0.450 rad/s** in 4.0 M
nominal rows (`min_lin_vel / wheel_radius = 0.05 / 0.11` and `kp_ori · ang_tol · half_width / wheel_radius` both give 0.45),
so a dead band below that value clips nothing the expert ever does.

| model (4096 mixture) | offline: predicted on expert-zero rows, median / p90 / p99 | offline: predicted on moving rows, p1 / p5 | closed loop, base still: median / p90 / p99 | closed loop, driving: p1 |
|---|---|---|---|---|
| MLP 28 control, seeds 0–2 | 0.03–0.06 / 0.07–0.10 / 0.24–0.40 | 0.71 / 1.08–1.14 | 0.11–0.14 / 0.28–0.33 / 0.59–0.63 | 0.96–1.07 |
| GRU 28, seeds 0–1 | 0.04 / 0.08–0.11 / 0.25–0.34 | 0.70–0.80 / 1.10–1.13 | 0.05–0.06 / 0.11–0.44 / 0.49–0.70 | 0.02 (seed 0, the pausing GRU) / 1.11 |
| transformer 28, seeds 0–1 | 0.04–0.05 / 0.09–0.10 / 0.25–0.28 | 0.73–0.74 / 1.13–1.14 | 0.10–0.11 / 0.29–0.30 / 0.56–0.57 | 1.03–1.04 |

Consequences of a dead band at 0.45 rad/s (the expert's floor), per model:

| model | zero rows left non-zero (offline) | moving rows clipped (offline) | still samples left non-zero (closed loop) | driving samples clipped (closed loop) |
|---|---:|---:|---:|---:|
| MLP 28 control | 0.2–0.5% | 0.2–0.3% | 2.5–3.8% | 0.0–0.2% |
| GRU 28 seed 1 | 0.5% | 0.1% | 1.2% | 0.0% |
| transformer 28 | 0.4% | 0.1–0.2% | 2.5–2.7% | 0.0–0.3% |

So the two populations are separable: the policies' "zero" lives below 0.35 rad/s in 99% of the rows and their real
commands above 0.7 in 99%; the expert's floor of 0.45 sits in the gap. A dead band at 0.45 would turn 96–99% of the
closed-loop residual while parked into an exact zero at no cost on the drive. What it does not tell us yet is whether a
base that is held exactly still stays in the policy's zero region (the input then changes only through the clock and the
arm's reach), i.e. whether the 2.5–3.8% of still samples above the threshold are the restart itself or noise before it.
That is the closed-loop test (E8d below): the dead band as a declared output filter on the existing models, no retraining.
Its learned counterpart is a soft-threshold output (`wheel = sign(u) · max(|u| − τ, 0)`), which makes every |u| < τ an
exact zero during training as well, so the zero rows stop pulling the regression towards a point.

## 6s. The transformer's stop is also in its context, and the E8 programme (2 Nov 07:40)

`stop_basin` on the three E5 transformers (teacher-forced history up to the expert's stop, then the current pose
displaced): **2–3% of the drive level for every displacement up to 40 cm towards or away from the target**, exactly like
the GRU (1–2%) and unlike the MLP 28 (13–23% at 40 cm towards). So the transformer *does* hold a stop it has seen in its
context, whatever the pose says; the earlier reading of §6j ("creeps like the MLP because it has no latch") was wrong
about the cause. What differs is closed loop: its own residual while parked is 0.10–0.11 rad/s median (§6r), larger than
the GRU seed 1's 0.06, so its context never contains a clean stop event and the latch it can hold is never triggered. The
memory models need to *see themselves stopped* before they can hold the stop. That is a second argument for the dead
band (E8d): an exact zero on the wheels puts a real stop into the context.

E8, decided with Rodrigo on 2 Nov, all on the 4096 mixture with the 28 pure inputs, 3 seeds, both validations:

- **E8d — dead band 0.45 rad/s at the output** of the trained models (MLP 28 control, GRU 28, transformer 28, MLP 32), a
  declared protocol variant (`experiment_specs/evaluation/validation_deadband045.json`, `wheel_dead_band_rad_s`; the
  evaluator zeroes the four wheels when both commands are below the band, learned policies only, recorded per episode).
  No retraining. Launched 2 Nov 07:40 (`chain_e8d.sh`, commit `509c6d4`), outputs `runs/evaluation/e8d_db045_*_20261102`.
- **E8a — gated stop**: wheels = g_t · MLP(x_t), g_t = g_{t−1} · (1 − σ(GRU(x_1..t))), g_0 = 1, reset per episode. The
  memory can only attenuate a command produced from the current frame, never produce it; the gate is non-increasing by
  construction, so a closed stop stays closed as the expert's `done` does; trained end to end from the wheel and arm
  labels only (no stop label, no extra output).
- **E8b — MLP → GRU** (Rodrigo's hybrid): a learned encoder of the current frame feeding the GRU, command from the GRU.
- **E8c — GRU with geometry by default**: command = MLP(x_t) + GRU correction with an L2 penalty on the correction, and
  hidden-state resets at random times plus hidden-state noise during training (broken histories), so the memory cannot
  time the phases by counting.
- **E8e — soft-threshold output** on the MLP 28: wheel = sign(u) · max(|u| − τ, 0), τ = 0.45 fixed (learned variant if
  the fixed one works), so |u| < τ is an exact zero during training too.
- **E8f — transformer with relative position**, conditional on E8d: if the dead band alone lets the transformer latch, the
  positional change is not needed.

Pause lengths before the restart (closed-loop traces, base below 2 cm/s for ≥ 0.5 s and moving again afterwards):
transformer 3.2 / 4.2 / 1.7 s median (p75 5.3 / 7.2 / 2.8 s, up to 32 s; 201 / 226 / 110 pauses), GRU 0.7 / 1.4 / 1.1 s
(p75 ≤ 2.5 s), MLP 28 2.3–2.8 s (p75 7–9 s). So the transformer does see its base nearly still for several seconds and
restarts anyway: a near-still base creeping at ~0.7 cm/s is not the exactly-still base of the demonstrations (2 cm of
drift in 3 s is visible in the 28 features), which is what E8d tests with an exact zero; if an exact stop of several
seconds in its own context still does not latch it, the gate with the transformer as memory (E8a with attention) is the
next arm.

New probe for every arm: the instant and the distance to the base goal at which the wheel command becomes exactly zero
and whether it stays zero (closed early / on time / never).

## 6t. E8d, dead band 0.45 rad/s at the evaluator output (2 Nov 07:40–10:40)

Setting: the trained E4 models unchanged; the evaluator zeroes the four wheel commands whenever
max|wheel| < 0.45 rad/s (`experiment_specs/evaluation/validation_deadband045.json`, §6r; `wheel_dead_band_rad_s` recorded
in every episode record; the expert is never filtered). Same 205 nominal goals and 205 random-start goals as every
other arm. Four families, 12 models, 24 evaluations (`runs/evaluation/e8d_db045_{mlp28,gru28,tf28,mlp32}_on_{clean,e1_random}_validation_20261102`).

Stable arrivals 1/2/5 cm, seeds 0/1/2, with the dead band vs without (E4, §6g/§6i):

| model | validation | without band | with band 0.45 |
|---|---|---|---|
| MLP 28 control | nominal | 0/0/23, 0/0/30, 0/1/27 | 0/1/24, 0/0/33, 1/1/27 |
| MLP 28 control | random | 0/0/11, 0/0/15, 0/0/17 | 0/0/13, 0/0/16, 0/1/14 |
| GRU 28, memory 20 ms | nominal | 6/6/32, 19/50/102, 4/8/27 | 0/3/28, 25/50/94, 5/9/29 |
| GRU 28, memory 20 ms | random | 6/9/29, 29/57/99, 5/8/26 | 3/6/23, 35/55/92, 4/10/28 |
| transformer 28, attention 20 s | nominal | 25/42/81, 29/43/76, 8/13/47 | 12/24/68, 10/14/40, 6/14/41 |
| transformer 28, attention 20 s | random | 9/29/60, 5/14/32, 3/4/29 | 9/26/54, 7/11/33, 3/5/27 |
| MLP 32 mixture (E3, rule-based features) | nominal | 22/32/51, 34/56/93, 15/38/78 | 23/42/68, 8/22/55, 1/12/55 |
| MLP 32 mixture (E3, rule-based features) | random | 16/22/51, 37/65/113, 10/30/71 | 19/37/64, 9/12/46, 3/8/45 |

No gain anywhere, and a loss where the stop used to be jostled into reaching: the transformer drops 81/76 → 68/40 and
the MLP 32 drops 93/113 → 55/46 (seed 1). The probes say why, and the stateless families and the GRU fail for opposite
reasons (`scripts/analysis/deadband_raw_drift.py`, `creep_profile.py`, `restart_trigger.py`, nominal
validation, medians over episodes):

- **MLP 28: the dead band freezes the base at the edge of the policy's zero, not inside it.** The policy ramps its
  wheel command down smoothly where the expert steps from ≥ 0.45 to 0; the band cuts the ramp at the first sample
  under 0.45, so the base stops where the raw command is just under the threshold. While filtered to exactly zero the
  raw max|wheel| is 0.31–0.38 (median; p90 0.42–0.44), the longest exact-zero spell lasts 8–9 s nominal, 4–5 s on
  random starts, and ends when the raw command crosses the band (0.47–0.48 on the first sample over it); only 1–2 % of
  the spells last to 40 s. The base is exactly still during the spell (0.2 cm/s), the arm joints move 0.001–0.002 rad,
  the target does not move: the only input that changes is the clock, and it is enough to push a command sitting at
  0.40–0.44 over 0.45. The park lasts longer than without the band (10–13 s to restart vs 2–3 s, §6g) but it still ends
  in a restart in 93 % of the episodes, and once the base moves again the stateless policy is back in §6g. Arrivals are
  unchanged also because this family's arm is poor (arm extended 12–18 of 205 for seeds 0 and 2), so even a permanent
  base stop would not reach 5 cm.
- **GRU 28 seed 1: the dead band gives a permanent, exact stop, and the arrivals do not move.** While filtered to zero
  its raw max|wheel| is 0.07 (p90 0.11), far under the band and falling (0.12 → 0.07 over the spell); the longest
  exact-zero spell lasts 25 s and reaches the end of the episode in 90 % of the episodes; the base spends 59 % of the
  40 s at an exact zero. This is the latch of §6g/§6q holding with a clean zero under it. The arrivals stay at 94–102
  because the stop is in the wrong place: `stop_profile` counts 99 episodes stopped outside the tolerance with a
  median closest EE distance of 10 cm, and the base is within 10 cm of its goal in only 17 episodes. The dead band
  therefore isolates the GRU's remaining defect to the approach, exactly as §6q concluded.
- GRU seeds 0 and 2 behave as MLPs here (raw 0.38 under the band for seed 2, 88 % of the zero samples above 0.3; seed 0
  rarely stops at all, 10 % of the time at zero and most of it while the base is actually moving on a transient dip).

### 6t.1 Why the MLP does not reach when the dead band stops it: the arm's start is triggered by the arm's own velocity

With the band the MLP 28 stops in the right place: first exact stop at 7.1–7.4 s, base 5 cm from where the expert
parks (within 10 cm in 87–93 % of the episodes), end effector 19–21 cm from the target, exactly what the expert has when
its navigation finishes (20 cm). The stop lasts 6–9 s. During it the arm does not move (0.001 rad) and the policy's arm
torque is the expert's HOLD torque (distance 0.28 to the hold torque vs 2.4 to the first reach torque, 2–3 % along the
hold→reach axis), at every heading error, including the 60 episodes within 1° of the required bearing. So the base is
parked and the policy keeps the arm at home until the clock pushes the wheel command over the band and the base leaves.

Offline swap test (`scripts/analysis/arm_switch_probe.py`, 71 nominal episodes, MLP 28 seeds 0/1/2, arm output as a
fraction along hold→reach): on the expert's own row at the done instant the policy outputs reach (+1.6, overshooting);
the same row with the policy's base pose, +1.3; with the policy's clock, +1.6; with the policy's arm state, −0.02. Within
the arm state it is the velocities: angles from the resting sample before done, +2.2; velocities set to zero, −0.08
(seed 2: +0.01). The policy's arm therefore reaches only once the arm is already moving. The reason is in the
demonstrations: the expert's reach torque starts the instant navigation finishes and the joints exceed 0.05 rad/s within
the same millisecond (0.09 rad/s on the done row, 0.54 rad/s 20 ms later), so "arm at rest at home, base parked, label =
reach" exists for about one row per 12 000, while "arm at rest at home, base in the box, label = hold" fills the whole
orient phase. The regressor learns hold for a resting arm and reach for a moving arm: a self-sustaining loop with no
ignition. Without the band the ignition came from the base: in the E4 evaluation the arm first exceeds 0.1 rad/s at 18 s
median, while the base moves at 7.8 cm/s (still in only 12 % of the episodes), i.e. the creeping base jostles the arm
and the loop catches. The band removes the jostle. Knowing that the base has stopped is not what the arm lacks: the
MLP 31 of E4 (28 pure inputs + finite-difference base velocity) parks with 9–20 cm of drift and extends the arm in 0 of
205 episodes for seeds 1 and 2 (130 for seed 0). The defect is the single resting-arm reach row per episode against the
whole orient phase of resting-arm hold rows, which no stateless regressor resolves whatever it is told about the base.

This is the same thin-shell defect as the base stop (§6g), one level up: the arm's phase switch is a step in the labels
at an input configuration the demonstrations visit for a millisecond. GRU seed 1 carries it in its state (its arm moves
0.30 rad during its 21 s stop and reaches 4 cm at best), which is why it is the only arm of the programme with a
working arm-after-stop. It also says what the gate must do: switching the wheels off is not enough, the stop state must
also hand the arm its first push (or the arm head must see a "base stopped" signal), otherwise E8a inherits this.

- **Transformer 28 and MLP 32 under the band: the same two defects, in different proportions.** Both stop at the edge
  of their zero (raw max|wheel| 0.33–0.36 → 0.43–0.44 over the spell for the transformer, 0.35–0.41 → 0.43–0.44 for the
  MLP 32) and both keep the arm at home while stopped (arm joints move 0.005–0.009 rad over the spell). The transformer's
  spell lasts 3–9 s and ends in a restart in 82–91 % of the episodes; its arm output on its own stop rows is −0.03 on the
  hold→reach axis, and on the expert's done row with its own arm state +0.18 (base pose and clock irrelevant, as for
  the MLP 28). Even open loop on the expert's rows its arm is slow: −0.24 in the first second after done, +0.78 at 1–3 s.
  The MLP 32 holds the exact stop much longer (16–19 s median, to the end of the episode in 9–37 %, base parked in
  170–182 of 205 with 3–15 cm of drift after parking) because its rule-based features put its zero deeper under the
  band, but the arm extends in only 37–47 episodes (134 for seed 1 without the band): its arm output on its own stop
  rows is +0.12, with the expert's arm velocities +1.6. So for every stateless model the band converts "creep and
  restart" into "stand still with the arm at home", and the arrivals it used to get from a jostled arm disappear.

Conclusion for the E8 programme: an output-side threshold cannot rescue a policy whose zero sits at the threshold, and
the clock makes any fixed threshold a timer. More important, E8d shows that an exact zero alone is not the goal: once
the base is exactly still, every stateless policy keeps the arm at home, because the arm's start was learned from its
own velocity (§6t.1). The stop state therefore has to do two things, switch the wheels off and start the arm. That is
what E8a's gate must be wired to (the gate value as an input of the arm head, or the arm head conditioned on the stop
state), and it is why GRU seed 1, which carries the stop in its state, is the only model whose arm works after the stop.
The transformer did not latch with an exact stop in its context (82–91 % restarts), so the gate is built on a GRU, not
on the transformer; E8f (relative positions) is dropped.

## 6u. E8a (gated stop) and E8b (MLP→GRU): design and launch (2 Nov 11:40)

Implementation: `src/robot_learning/learning/gated_policy.py` (kind `gru_gated_regressor`, trainer
`scripts/train_gated.py --architecture gated|mlp_gru`), NumPy inference verified against the torch twin by test and at the
end of each run (`tests/test_gated_policy.py`, 105 tests pass). Both keep the brief's two outputs.

- **E8a `gated`.** u_t = head([x_t, g_t]) is a stateless MLP on the 28 pure features plus the scalar gate; a GRU of 32
  units reads the feature history every 20 ms and emits a stop score s_t = σ(w_s·h_t + b_s); the gate is
  g_t = 1 − max_{τ≤t} s_τ, so it starts at 1 (bias init −4), can only fall and never reopens within an episode. Wheels =
  g_t · (head wheels in rad/s), so g_t = 0 is an exact zero whatever the head says; arm torques = head arm torques, and
  because the head sees g_t the arm has the "base stopped" signal it never had (§6t.1). Trained end to end from the wheel
  and arm labels only (masked MSE in normalised units, the gate acting in physical units inside the network); nothing
  tells the gate when to close except the labels going to zero. The trainer records where the gate closes on 64
  validation episodes (`metrics.json → validation.gate`) against the time the wheel labels go to zero.
- **E8b `mlp_gru`** (Rodrigo's hybrid). e_t = relu(W_e x_t + b_e) with 64 units feeds a GRU of 128 units (20 ms); the head
  reads [h_t, x_t] as in the base GRU. Same data, same loss.

Both on the 4096 nominal + random mixture (`runs/splits/e3_mixture_4096.json`), 3 seeds, one trainer at a time, then the
two validations (205 nominal goals, 205 random-start goals) with the standard protocol (no dead band). Runs
`runs/training/gru_e8a_gated_mixture4096_polar_seed{0,1,2}`, `runs/training/gru_e8b_mlpgru_mixture4096_polar_seed{0,1,2}`;
evaluations `runs/evaluation/e8a_gated_on_{clean,e1_random}_validation_20261102`, `e8b_mlpgru_on_…`.
Predictions, written before the results: (i) E8a's gate closes within 0.5 s of the label zero on the validation
episodes if the stop is learnable from history at all; (ii) in closed loop E8a stops exactly and the arm starts (both ≥
GRU seed 1's 11 "both" per 205), the open question being whether its stateless head drives into the box better than the
GRU (§6q: approach is the GRU's defect); (iii) E8b behaves like the base GRU (one seed latching, two not) unless the
encoder makes the latch easier to find, which would show as all three seeds latching.

## 6v. E8a results (2 Nov 11:54–12:50): the gate confirms a stop it never causes

Runs `gru_e8a_gated_mixture4096_polar_seed{0,1,2}` (96 / 148 / 96 epochs of 582 updates, patience 8, 6–9 min each on the
GPU), evaluations `runs/evaluation/e8a_gated_on_{clean,e1_random}_validation_20261102` (205 goals each, no dead band).
Probes: `scripts/analysis/gate_closure_probe.py` (new: box entry, stop-condition instants, where the wheels go to zero
for good, arm after the zero), `stop_profile.py`, `arm_base_decomposition.py`, open-loop gate ablations and gate
profiles on the validation demonstrations, and a re-simulation of the 205 nominal episodes of seed 1 logging the gate
and the stop score every 100 ms (scratch scripts, numbers below).

| E8a gated | offline NRMSE | nominal 1 / 2 / 5 cm | parked / arm ext. / both | held ≥ 10 s | in tol. / stopped outside (median dist) / moving at 40 s | random 1 / 2 / 5 cm | parked / arm ext. / both | stopped outside / moving |
|---|---:|---|---|---:|---|---|---|---|
| seed 0 | 0.219 | 5 / 11 / 23 | 19 / 39 / 3 | 8 | 0 / 35 (0.54 m) / 170 | 2 / 5 / 17 | 9 / 59 / 4 | 49 (0.60 m) / 156 |
| seed 1 | 0.233 | 8 / 18 / 49 | 28 / 126 / 14 | 17 | 1 / 106 (0.48 m) / 98 | 3 / 8 / 34 | 19 / 148 / 19 | 129 (0.40 m) / 74 |
| seed 2 | 0.230 | 0 / 1 / 5 | 26 / 15 / 1 | 2 | 0 / 14 (0.60 m) / 191 | 0 / 0 / 8 | 7 / 12 / 0 | 10 (0.34 m) / 195 |
| E4 MLP 28 control (same data) | 0.145–0.149 | 0 / 0–1 / 23–30 | 0 / 12–18 / 0 | 0–5 | 0 / 0 / 205 | — | — | — |
| E4 GRU 28 seed 1 | 0.153 | 19 / 50 / 102 | 29 / 38 / 11 | 38 | — | — | — | — |

Worse than every earlier arm on arrivals (best seed 49 at 5 cm, 8 at 1 cm), but with a new failure signature: the first
pure-28 model whose base is *parked* in 19–28 nominal episodes and whose arm *extends* in 126 (seed 1) — and whose stops
are all in the wrong place. The predictions of §6u, tested:

**(i) Where the gate closes on the demonstrations — partly as predicted, and the measure hid the mechanism.** On the 64
validation demonstrations the gate falls below 0.5 in 63 / 63 / 62 episodes, never later than the label zero, within 0.5 s
of it in 33 / 44 / 49 and more than 0.5 s early in 30 / 19 / 13 (up to 19 s early on stalled demonstrations). That looked
like a predictor. The gate's *profile* against the time of the label zero (median over 40 demonstrations) says otherwise:

| time relative to the label zero | −10 s | −5 s | −2 s | −1 s | −0.5 s | 0 | +0.5 s | +2 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gate, seed 0 | 0.84 | 0.83 | 0.82 | 0.66 | 0.50 | 0.44 | 0.002 | 0.002 |
| gate, seed 1 | 0.91 | 0.92 | 0.92 | 0.77 | 0.62 | 0.39 | 0.002 | 0.002 |
| gate, seed 2 | 0.94 | 0.76 | 0.74 | 0.66 | 0.60 | 0.53 | 0.001 | 0.001 |

The stop score peaks at 0.23–0.34 (median; p90 0.46–0.50) somewhere during the drive, so the cummax gate *leaks*: it sits
at 0.75–0.92 for the whole approach (below 0.9 on 22% of the drive rows of seed 1), slides to ≈ 0.5 as the expert
rotates in its last half second, and reaches its closed value (0.002) only *after* the labels are already zero, when the
stop score saturates at 0.998. The gate does not anticipate the stop from the approach; it recognises the stopped state
once it is there. Ablations on 24 demonstrations (seed 1, closure = gate < 0.5): clock frozen at zero, arm velocities
zeroed or arm angles frozen — still 24 / 24 closures at the label zero (the clock and the arm are not the cue); the
observation frozen from the start of the expert's `orient` phase (base still at the box, heading not yet aligned, no
rotation) — 8 / 23; base still but rotating forever at 0.5 rad/s — 6 / 23; sequence cut before `orient` — 3 / 23. The cue
is the base standing still at the goal with the heading aligned, i.e. the state the expert's controller produces, read
off the history (velocity is not in the 21 inputs, but two frames 20 ms apart give it to the GRU). It is the same shortcut
as the GRU's open-loop stop in §6h: the labels are zero exactly when the base is still, so "still ⇒ zero" is the cheapest
fit, and nothing in the loss asks the gate to close *before* the stillness it is supposed to cause.

**(ii) Closed loop — refuted for the stop, confirmed for the arm.** `gate_closure_probe.py` on the 205 nominal goals:

| nominal, 205 goals | entered the box (< 8 cm) | heading < 0.05 rad while inside | wheels zero to the end | … inside the box | … outside (median dist) | arm moved > 10 cm after the zero |
|---|---:|---:|---:|---:|---|---:|
| E8a seed 0 | 49 | 25 | 4 | 0 | 4 (0.64 m) | 3 |
| E8a seed 1 | 88 | 47 | 54 | 0 | 54 (0.45 m) | 43 |
| E8a seed 2 | 67 | 34 | 7 | 0 | 7 (0.57 m) | 2 |
| E4 MLP 28 control, seeds 0–2 | 186 / 195 / 185 | 132 / 148 / 136 | 0 | 0 | 0 | — |
| E4 GRU 28, seeds 0–2 | 57 / 89 / 93 | 30 / 39 / 63 | 1 / 2 / 3 | 0 | 1 / 2 / 3 | 0 / 0 / 1 |

Random starts: E8a entered 33 / 56 / 63, met the stop condition in 16 / 24 / 24, wheels zero to the end in 0 / 71 / 3
(one inside the box), arm after the zero in 0 / 59 / 2; the MLP 28 control entered 106–134 and never zeroed.

- The head drives the base into the box in 49–88 episodes and even through the expert's exact stop condition (inside
  8 cm with |heading error| < 0.05 rad at some sampled instant) in 25–47 — and the gate closes inside the box in **none**
  of them. Inside the box the base keeps a median speed of 0.04–0.10 m/s and 0.01–0.06 rad/s, stays 1.2–2.7 s (p90 2.2–6.1)
  and leaves; the expert's `orient` phase before `done` lasts 0.5 s (0.3–1.4) at 0.36 rad/s with 0.015 m/s of translation
  and ends with the base still. The stillness the gate waits for is the stillness only the gate can produce: a
  confirmer in a loop with its own effect never gets its cue. The 54 / 71 zeros of seed 1 happen outside, at 0.40–0.48 m
  from the goal, with the base still moving (§6v.1 measures what the stop score does there).
- Once the gate has closed, the arm starts: it moves more than 10 cm after the zero in 43 of 54 nominal and 59 of 71
  random closures (80%), against 0 of 205 for every stateless model under an exact stop in §6t.1. The gate as an input
  to the arm head is the "base stopped" signal the arm lacked. The stops being 0.4–0.6 m off, "both" is 14 / 19 (seed 1)
  against 11 for the GRU seed 1 — the prediction's threshold is met by seed 1 only, and for the wrong reason.
- The gate's drift after closing is 0.01–0.05 m (median), against 9–20 cm for the MLP 31 of §6g and the 6–10 cm creep of
  §6i: the exact zero holds, as designed (E8d's GRU seed 1 held 25 s; E8a holds to the end in every closure).

**The head itself fits worse than the MLP 28 on the same rows.** Open loop on 24 validation demonstrations
(episode-median RMSE): wheels on drive rows 0.43 rad/s for E8a seed 1 against 0.23 for the E4 MLP 28 control and 0.23 for
the E4 GRU; on the rows where the gate is still above 0.9, 0.41 — the head, not the leak; arm before the stop 0.14 N·m
against 0.04 (MLP 28) and 0.13 (GRU); arm after the stop 0.73 against 0.41 and 0.61. Offline NRMSE 0.22–0.23 against
0.145–0.149 for the control and 0.14–0.15 for the GRU family. The head sees the same 28 features as the control plus the
gate, and is trained by the recurrent recipe (8 whole episodes per update, 1000-decision chunks) instead of the control's
1024 shuffled rows from ~1000 episodes per update; the GRU family pays the same price on the arm (0.13 vs 0.04) and enters
the box in 57–93 episodes against 185–195 for the control. The approach defect of the recurrent models (§6q) and of E8a's
stateless head is at least partly a training-recipe effect, not an architecture effect — a testable claim (train a plain
MLP 28 with the recurrent recipe, or the recurrent models with more, shorter, shuffled episode chunks).

**What E8a establishes.** A latch that cannot reopen holds an exact zero and starts the arm — the two things no earlier
arm did. Its learned trigger is the wrong one: trained end to end from zero labels that coincide perfectly with a still
base, the stop score becomes a stillness detector and never fires in a loop where the stillness is its own output. The
closed-loop system has no stop until the base stalls by accident. The next gate must be unable to see stillness: a
*stateless* stop score on the current geometry (the 21 inputs carry no base velocity, so a single frame cannot tell
still from moving), latched by the same cummax, and trained with a target derived from the labels alone (1 from the first
zero-wheel row of the demonstration onwards, 0 before — a function of the labels, not of the expert's phases, so inside
the black-box rule). With teacher forcing (the label's own gate during training) the head and the gate become row-wise
models trainable by the MLP recipe, which also removes the recipe handicap above. Proposed as E8g; not launched.

### 6v.1 The gate in closed loop, re-simulated (seed 1, 205 nominal goals, gate and stop score logged every 100 ms)

The re-simulation reproduces the evaluation exactly (final goal distances identical to 1e-5 m in all 205 episodes).

| time window | 0–5 s | 5–10 s | 10–20 s | 20–40 s |
|---|---:|---:|---:|---:|
| stop score, median | 0.001 | 0.31 | 0.34 | 0.43 |
| gate, median (p10) | 0.94 (0.67) | 0.57 (0.002) | 0.42 (0.002) | 0.24 (0.002) |

- **The gate does fall in every episode, slowly, and not at a stop.** It passes 0.5 in all 205 episodes, at 7.8 s (median)
  with the base 0.30 m from the goal, 0.25 rad off the goal bearing and moving at 0.125 m/s over the previous second
  (0.24 m/s two to three seconds earlier); in none of the 205 was the base still (< 2 cm/s) when the gate passed 0.5, and
  only 38 of the 205 half-closures happened inside the box. The per-episode peak of the stop score (0.87 median) comes
  at 31.5 s, 1.2 m from the goal, with the base moving at 8 cm/s — nothing resembling a stop. The score is a function of
  the policy's own trajectories that the demonstrations never showed it: during the drive it is 0.03 when the heading is
  within 0.1 rad of the goal bearing and 0.37–0.50 (p90 0.91) when it is off by more than 0.3 rad. The expert drives
  aligned (median |heading error to the goal bearing| 0.05 rad while moving, nominal goals, because it aligns first and
  then drives); every policy drives in arcs instead (E8a head 0.45–0.99 rad, MLP 28 control 1.03, GRU seed 1 0.74), so
  the detector spends the whole approach in states off its training distribution, and the cummax turns each excursion of
  the score into a permanent loss of gate.
- **Inside the box the score never saturates.** In the 47 episodes where the head met the expert's stop condition the
  score peaks at 0.57 (p90 0.65) with the base down to 1.4 cm/s; on the demonstrations the same stillness at the goal gives
  0.998. The confirmer (§6v) needs the demonstration's exact stillness and the demonstration's approach history; the
  policy's approach supplies neither, so the gate half-closes (0.42 median) and the base leaves the box at half command.
- **The leak brings the creep back.** Final gate below 0.05 in 115 of 205 episodes, but the wheels are exactly zero to the
  end in only 54: in 61 episodes the gate sits at 0.002–0.05 of a head that still commands 1–5 rad/s, so the base crawls
  at 0.04 rad/s of wheel command (median after 35 s), the evaluator's "moving at 40 s" (98 of 205 for this seed). The
  demonstrations' closed value is 0.002 because the score saturates there; off distribution it does not.

Mechanism, in one line: a learned stop detector with a non-reopening latch is only as good as its false-positive rate on
the policy's own states, and end-to-end training from zero labels gives it no false positives to learn from — every
training history is the expert's. (A DAgger-style relabelling would supply them; the brief excludes it, §6k.) The stateless
geometry gate of E8g has no history to drift on and an explicit 0/1 target on every row, including the rows where the
expert drives misaligned in the random-start demonstrations.

## 6w. E8b results (2 Nov 12:34): the encoder does not make the latch easier to find

Runs `gru_e8b_mlpgru_mixture4096_polar_seed{0,1,2}` (44 / 97 / 75 epochs, 5–10 min each), evaluations
`runs/evaluation/e8b_mlpgru_on_{clean,e1_random}_validation_20261102`. Architecture: relu encoder of 64 units → GRU 128
(20 ms) → head on [h, x], 28 pure features, the 4096 mixture, the same recipe as the E4 GRU.

| E8b MLP→GRU | offline NRMSE | nominal 1 / 2 / 5 cm | parked / arm ext. / both | held ≥ 10 s | stopped outside (median dist) / moving at 40 s | random 1 / 2 / 5 cm | parked / arm ext. / both |
|---|---:|---|---|---:|---|---|---|
| seed 0 | 0.162 | 23 / 31 / 55 | 29 / 50 / 1 | 24 | 47 (0.24 m) / 157 | 6 / 24 / 62 | 7 / 61 / 0 |
| seed 1 | 0.136 | 7 / 17 / 44 | 31 / 157 / 26 | 16 | 24 (0.21 m) / 180 | 7 / 20 / 49 | 8 / 142 / 7 |
| seed 2 | 0.144 | 9 / 33 / 79 | 10 / 50 / 2 | 20 | 37 (0.24 m) / 167 | 3 / 13 / 43 | 2 / 27 / 0 |
| E4 GRU 28 (same data, no encoder) | 0.136–0.153 | 6 / 6 / 32 · 19 / 50 / 102 · 4 / 8 / 27 | 4 / 9 · 29 / 38 · 0 / 2 (parked / arm) | 38 (seed 1) | — | 29 / 99 / 26 (5 cm) | — |

Prediction (iii) of §6u was "like the base GRU — one seed latching, two not — unless the encoder makes the latch easier
to find, which would show as all three seeds latching". Neither: **no seed latches.** Wheels exactly zero to the end of the
episode in 0 / 10 / 0 nominal episodes (`gate_closure_probe.py`), against 25 s holds for the GRU seed 1 under the band
(§6t); the base parks and restarts in 38 / 81 / 120 episodes (`creep_profile.py`) with 13–25% of the drive command while
parked, creeping 2.7–7.6 cm towards the target in 2.3–3.5 s before the restart — the stateless creep of §6i, not the
recurrent latch of §6q. The box is entered in 99 / 57 / 82 episodes with the stop condition met in 47 / 43 / 22, and the
wheels never go to zero there. Open loop on the validation demonstrations the residual after the stop is 1–2% of the drive
command for E8b, for the base GRU and for the MLP 28 alike (all on the expert's states), so the open-loop residual does not
separate the families; the closed-loop residual does (13–25% here, 6–10% MLP, 1–4% GRU seed 1 and the E2 GRU in §6q).

The arrivals are the more even of the recurrent family (44–79 nominal, 43–62 random, no 102 and no 27): the encoder gives
the three seeds the same, middling policy — the base GRU's seeds 0 and 2 without the seed-1 latch. The arm extends in
157 nominal episodes for seed 1 (as the GRU seed 1 and E8a seed 1 do: the recurrent models that get the arm moving are those
whose base decelerates most), but "both" stays at 26 because the base is parked 21–24 cm off the goal. Conclusion: a
feature encoder in front of the GRU changes neither of the two measured defects (approach 16–17 cm off the box at closest,
no latch in closed loop); the latch of §6q was a seed-dependent solution of the same optimisation, not something the
input layer withheld. E8b closed.

## 6x. The teammate's C3 "trigger-frame copies" (branch `feature/step3c-trigger-copies`, 31 Oct): why it works

Read on 2 Nov from Whitea02's branch (`final/robotnik_arm_task/`, report `report.txt` §"实验 C3", code `train_policy.py
--trigger-copies`, features `bc_features.py`, policy `policies/bc_policy.py`). Their pipeline is independent of ours (same
lecturer expert and parameters: pos_tol 0.08, ang_tol 0.05, standoff interp, 25 s navigation timeout, 0.25 m/s arm; own
collector, sklearn MLP, own evaluator), so the comparison is at the level of mechanisms, not of files.

**What they changed.** One thing: the training rows. Their baseline `ego-novw` is a stateless MLP 128×128 (Adam, batch 200,
early stopping on the validation loss) on 20 pure features (target in the base frame dx, dy, target z, planar distance,
cos/sin of its bearing, arm q, arm qd; no base speed, no clock, no world pose), trained on 1 613 episodes (2 000 collected,
seed 1, random start within 2 m and any yaw, 50% of them with DART-style command noise, expert failures filtered out), one
row every 9 ms. C3 appends to those rows **K = 16 copies per episode of the recorded frame k0 = the first `nav_done` step**
(wheel labels exactly 0, arm torque label = the first reach torque, median jump 27.9 N·m), inputs and labels exactly as
recorded, with Gaussian jitter on the 7 arm velocities only (σ per joint = std of arm qd over the navigation rows, one
fixed rng). Normalisation, features, network, split, validation loss and early stopping are unchanged; 25 808 rows added
(1.5% of the training rows). K was picked among {16, 64, 256} as the smallest that extends the arm in >30% of a pilot
closed-loop run with the expert base (rule amended after seeing the pilot; recorded as such in their report).

**What they got** (their protocol v1.1; `success` = EE within 10 mm at some instant ≤ 40 s with the episode ending as
`reached`, no dwell; 5 seeds):

| Set (300 / 200 episodes, random starts) | success | stable 5 / 2 / 1 cm (our criterion, 0.5 s dwell) | held 10 s | parked | arm extended | tipped |
|---|---:|---|---:|---:|---:|---:|
| baseline `ego-novw` DEV, seed 0 | 1 | 15 / 1 / 0 | 0 | 0 | 0 | 0 |
| C3 DEV, seeds 0–4 | 210 / 157 / 190 / 180 / 170 | 267–275 / 219–241 / 151–198 | 119 / 42 / 72 / 84 / 86 | 208–276 | 236–254 | 23 / 7 / 8 / 10 / 0 |
| C3 HARD, seeds 0–4 | 128 / 102 / 110 / 111 / 112 | 167–182 / 136–146 / 94–113 | 60 / 34 / 45 / 57 / 58 | 131–175 | 163–178 | 0–1 |
| C3 CONFIRM (new seeds 100–104), mean | 64.3% | reach 5 cm 92.7%, 2 cm 81.3% | 26.5% | — | 84.1% | 1.4% |

Their expert on DEV: success 298/300, held 10 s 296, arm extended 245 (short reaches do not move the EE 10 cm).

**Why it works: it is the ignition row of §6t.1, re-weighted.** Our §6t.1 swap test showed the MLP's arm reaches only once
the arm is already moving: "arm at rest at home, base parked, label = reach" exists for about one row per 12 000 in the
demonstrations, while "arm at rest at home, base in the box, label = hold" fills the whole orient phase, so the regressor
learns hold for a resting arm and reach for a moving arm, a self-sustaining loop with no ignition. Their offline analysis
(C0) found the same thing independently: in their stride-9 rows the trigger state is 147 rows of 1.8 M (0.008%), each
episode has ~74 look-alike rows labelled hold, the MSE conditional mean on that state is therefore "hold", and both of
their networks output r = 0.00 at the recorded k0 frame (r = fraction of the hold→reach torque jump produced) and
0.76–0.96 one millisecond later, when the arm is already moving. Their "push control" (same parked state, arm velocities
replaced by the recorded values 1 ms after k0) gives r = 0.67–0.87: the loop catches from the smallest nudge. C3 adds 16
labelled copies of the ignition state against ~74 look-alikes, so the conditional mean at that state moves ~0.1–0.2 toward
reach (their measured r on the training copies: 0.12; on the recorded k0 frames of the validation set: 0.12 for K 16, 0.30
for K 64, 0.48 for K 256). Offline that looks like nothing (r ≈ 0 on the frozen parked state, validation loss *doubles*:
0.026 → 0.055–0.063, early stopping 2–3× earlier), but in closed loop a 0.1–0.2 fraction of the reach torque for a few
milliseconds is enough to move the joints past the ~0.05 rad/s where the regressor switches to reach, and the arm then
drives itself to the target: arm extended 0% → 79% with the expert base (their EM cell) and 0.4% → 81% with the learned
base. The jitter on arm qd matters for the same reason: it widens the ignition state from "exactly at rest" to the band of
small velocities the arm actually has when the base parks (their fixed input: arm qd when the base has just stopped is
0.1–0.5 rad/s, not 0).

**Why the base then stops.** Our §6h finding: the base reads the arm as its phase flag (arm moving ⇒ navigation is over ⇒
wheels zero). Once the arm ignites, their parked rate goes 12% → 81% and `parked_no_arm` 54% → 0%, with no change to the
base's own rows. The residual creep is still there: half of their successes do not hold 10 s (`held_10s` 27% vs success
60%; base displacement during the hold median 6 cm, EE leaves the 15 mm band at median 3.2 s), the same 6–10% residual
command we measured in §6h–§6i. Their circling (base never parks) only drops 32% → 12%, confirming that part of the base
failures is the base's own (their estimate ~10%), not the arm's.

**Is it legitimate under our black-box rule?** Yes, with one wording caveat. The policy is a pure function of the current
observation (no clock, no expert state, no base speed; they dropped the measured wheel speeds on purpose). The only use of
the expert's internals is in *selecting* which training row to duplicate: the first `nav_done` row, read from the
recording. That row is identifiable from the labels alone (first row whose wheel labels are exactly zero after the drive,
where the arm torque jumps), so it is label-derived re-weighting, the same class as our E7a stop weight, not privileged
input. The jittered copies are input-side augmentation with the original label. For the report it should be described as
"K copies of the first zero-wheel row per demonstration", not "of the expert's done flag".

**Why our E7a did not do this.** E7a weighted ×8 the *wheel* loss on *all* rows with zero demonstrated wheels (27.6% of the
rows, almost all with the arm already reaching). It left the arm loss untouched and did not single out the one resting-arm
row, so the ignition state kept its conditional mean "hold" and the arm never started (3 / 103 / 15 → 7 / 113 / 2 arms
extended). C3 duplicates the whole row (both heads) and only that row.

**Costs they measured.** New failure classes: `extended_not_reached` 6% → 19% (EE enters 5 cm, not 1 cm: the arm fires
from a base pose that is not exactly the expert's), `tipped` 0.1% → 3.2% (48 sideways tip-overs, 23 in seed 0), early
reaches 5.5% before the shadow FSM's `nav_done` (never for the expert) but 96% on the ring and oriented. K 64 / 256 trigger
as often (79%) but succeed less (52% / 36% on the pilot) with more `extended_not_reached`: more weight fires earlier and
less precisely. Confound they flag themselves: the added rows also change the optimisation (more Adam steps per epoch,
earlier early stopping, doubled validation loss), so "trigger samples" and "training-process change" are not separated.

**For us (E9, proposed, not launched).** The same intervention fits our trainer as an `egocentric_options` entry
(`trigger_copies: K`, `trigger_qd_jitter: auto`): per training episode, take the first row with `base_navigation_done`
(equivalently the first exactly-zero wheel label), append K copies with N(0, σ_j) on the 7 arm velocities, labels
untouched, scaler from the original rows only. Test on MLP 28 pure, 4096 mixture, 3 seeds, both validations, with the
§6t.1 swap probe (`arm_switch_probe.py`) before and after as the mechanistic check: the prediction is r on the resting-arm
parked row going from −0.02 to ≈ +0.1–0.2 and the arm extending in most nominal episodes; the creep (§6h) should remain,
so stable 1 cm arrivals will trail 5 cm arrivals by their held-10 s gap. It is the cheapest test of the §6t.1 mechanism we
have (no new data, no memory, ~30 min per seed), and it combines with E8d/E8e (dead band / soft threshold) which remove the
creep but left the arm asleep precisely for lack of this row.

## 6y. E9, trigger copies in our trainer (2 Nov 15:20, pre-registered before any result)

**Implementation** (`egocentric_options`: `trigger_copies` K, `trigger_qd_jitter` "auto" | rad/s; commit `d5df73c`, 109 tests).
Per training episode the trigger row is **the first row of the terminal run of exactly-zero wheel labels**; K copies of it
are appended with their recorded features and labels and N(0, σ_j) on the seven arm-velocity features only, σ_j = std of
arm qd over the rows whose wheels are not both zero, one fixed rng (seed 0) shared by every training seed. Input and
output scalers are fitted on the recorded rows before the copies are appended. Nothing else changes (MLP 128×128, batch
1024, patience 8, 28 pure inputs, 4096 mixture, same split). Label-derived only: no expert phase or done flag is read.
Check on 80 training episodes: the label-only row equals the first `base_navigation_done` row in 78; the other 2 are 25 s
navigation timeouts where the zero run starts 1 ms after the flag. At that row the arm torque jumps 26.6 N·m median
(p10 19.0) and max |arm qd| is 0.099 rad/s median (p90 0.18): the ignition state of §6t.1.

**Why a K sweep and not their K = 16.** The weight that matters is copies against look-alike hold rows in the same input
region. They train on one row every 9 ms (~1 300 rows per episode, ~74 look-alikes); we train on every 1 ms row (11 952
rows per episode median, 3 126 orient-phase rows median). Their K = 16 is 1.5% of the rows and ~0.2 of the look-alikes;
for us K = 16 is 0.13% and 0.005. So: seed 0 at **K ∈ {16, 144, 512}** (144 ≈ 16 × 9 keeps their row fraction, 1.2%;
512 is 4.3%, between their K 64 and K 256 in relative weight), evaluated on both validations (205 nominal, 205 random
starts, 1/2/5 cm stable arrivals, parked, arm extended, drift). Their K 64 / 256 fired as often as K 16 but reached less
precisely, so the sweep is expected to trade ignition rate against precision.

**Rule for K, fixed now:** the smallest K whose seed-0 model extends the arm (EE > 10 cm from its start in the base frame,
`arm_base_decomposition.py`) in more than 50% of the nominal episodes; among several, the one with more stable 1 cm
arrivals (nominal + random). Seeds 1 and 2 are then trained at that K and the three are reported like every other arm
(§7 table rows). If no K passes, report and stop. Mechanistic check before adopting: `arm_switch_probe.py` on the
resting-arm parked row (control −0.02 on the hold→reach axis) and the arm-start time relative to the first exact stop.

**Predictions.** Offline NRMSE worse than the control (their validation loss doubled); arm extended in most nominal
episodes (control 3 / 103 / 15 of 205); parked rate up because the base reads the moving arm as its phase flag (§6h);
creep unchanged, so held-10 s and 1 cm arrivals trail the 5 cm arrivals; some early reaches from a misplaced base (their
`extended_not_reached` 19%, tip-overs 3%). Runs: `runs/training/mlp_e9_mixture4096_polar_trig{K}_seed{s}`,
evaluations `runs/evaluation/e9_ksweep_mlp28_on_{clean,e1_random}_validation_20261102`, then the chosen K's three seeds.

### 6y.1 K sweep results (2 Nov 15:41–17:04): the ignition row is the missing piece, and K = 144 holds

Seed 0, 4096 mixture, both validations (`e9_ksweep_mlp28_on_{clean,e1_random}_validation_20261102`). Stable arrivals of 205:

| K (seed 0) | offline NRMSE | Nominal 1 / 2 / 5 cm | parked | arm ext. | both | drift after parking | held ≥ 10 s | moving at 40 s | Random 1 / 2 / 5 cm | parked | arm ext. | held ≥ 10 s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| control MLP 28 (§6i, seeds 0/1/2) | 0.145–0.149 | 0 / 0–1 / 23–30 | 119–126 | 3 / 103 / 15 | | | | 199–203 | 0 / 0 / 11–17 | 33–68 | 3 / 105 / 11 | |
| 16 | 0.159 | 48 / 84 / 141 | 100 | 197 | 98 | 1.22 m | 22 | 187 | 47 / 61 / 89 | 44 | 194 | 23 |
| **144** | 0.166 | **96 / 124 / 140** | 101 | 198 | 97 | 0.22 m | **107** | 111 | **103 / 116 / 121** | 36 | 197 | **103** |
| 512 | 0.177 | 86 / 101 / 119 | 102 | 149 | 80 | 1.62 m | 79 | 183 | 66 / 83 / 98 | 31 | 145 | 58 |
| expert | | 205 / 205 / 205 | | | | | | 0 | 201 / 201 / 201 | | | |

**Rule applied:** every K extends the arm in > 50% of the nominal episodes (197 / 198 / 149 of 205); the tie-break (most
stable 1 cm arrivals, nominal + random) picks **K = 144** (199 against 95 and 152). Seeds 1 and 2 at K = 144 launched
17:20 (`chain_e9c.sh`), evaluations `e9_trig144_mlp28_on_{clean,e1_random}_validation_20261102`.

**What the numbers say.**

- **Ignition, measured.** `arm_switch_probe.py` on the expert's done row with the policy's own resting-arm state (the
  §6t.1 swap; fraction along hold→reach, control −0.02): K 16 **+0.12**, K 144 **+0.96**, K 512 **+1.27**. On the
  policy's own parked row in closed loop: +0.14 / +0.76 / +0.81. The copies move the regressor's output at the resting-arm
  parked state from "hold" to "reach"; the arm then extends in 197–198 of 205 nominal episodes (control 3–103). Even
  K 16's +0.12 is enough to start it (their K 16 had r 0.12 too, §6x): the loop catches from the smallest nudge, as the
  teammate's push control predicted.
- **1 cm arrivals go from 0 to 96 / 103** (nominal / random), the best of the programme by a factor of four (previous
  best: E8b seed 0, 23). 5 cm arrivals 140 / 121 against the GRU's best single seed 102 / 99.
- **The base parks because the arm moves (§6h), and the creep is what is left.** Parked 100–102 nominal (control
  119–126, so not more parking: *better* parking). K 144's drift after parking is 0.22 m median and 107 episodes hold the
  target 10 s; K 16 drifts 1.22 m and holds in 22, K 512 drifts 1.62 m and holds in 79. Still moving at 40 s: 111 / 187 /
  183 (control 199–203). The remaining failures are the §6h creep (base leaves after the arm arrived), now visible as
  "arrived, then left": 5 cm arrivals 140 but held 10 s 107.
- **Random starts: the gap to nominal closes.** 103 / 116 / 121 random against 96 / 124 / 140 nominal; the control had
  11–17 at 5 cm on random starts. The ignition does not depend on the approach history, only on the parked state.
- **Too much weight fires early.** K 512 extends the arm in fewer episodes (149 / 145) with the worst drift (1.62 m) and
  the worst offline fit (0.177); its stop rows are 0.24 m / 0.49 rad from the expert's done pose (K 144: 0.06 m / 0.08
  rad). The teammate saw the same at their K 256 (§6x).
- **Offline fit is anti-correlated with the gain**, again (0.145 → 0.159 → 0.166 → 0.177 while closed loop improves then
  degrades): the validation NRMSE is dominated by the orient-phase look-alike rows whose conditional mean the copies
  deliberately move.

**Under the teammate's protocol** (`scripts/analysis/teammate_protocol_metrics.py`: success = EE within 10 mm at some instant,
no dwell; settled = < 15 mm for 1 s after; held 10 s; reach 2 / 5 cm = minimum error; arm extended = EE moved > 10 cm in the
base frame; parked = ring distance < 0.2 m and still 1 s; our traces are 100 ms, theirs 1 ms), seed 0 of 205:

| policy | success | settled | held 10 s | reach 2 cm | reach 5 cm | arm extended | parked |
|---|---:|---:|---:|---:|---:|---:|---:|
| K 144, nominal | 49.3% | 48.8% | 28.8% | 62.0% | 68.8% | 97.6% | 72.7% |
| K 144, random starts | 51.2% | 50.7% | 33.2% | 57.1% | 60.5% | 97.6% | 62.0% |
| K 16 / K 512, nominal | 30.7% / 43.4% | 24.4% / 42.4% | 2.4% / 10.2% | 47.3% / 50.7% | 71.2% / 59.5% | 96.1% / 73.2% | 70.7% / 48.8% |
| MLP 28 control, nominal (3 seeds) | 0–1.5% | 0% | 0% | 1.5% | 14–16% | 1.5–53% | 28–54% |
| their C3 (K 16 at stride 9), DEV random starts, 5 seeds | 52–70% | 46–65% | 14–40% | 73–80% | 89–92% | 79–85% | 69–92% |
| their baseline `ego-novw`, DEV | 0.3% | 0% | 0% | 0.3% | 5% | 0% | 0% |
| our expert, nominal / random | 100% / 98% | 100% / 98% | 99.5% / 97.6% | 100% / 98% | 100% / 98% | 75% / 75% | 100% / 98% |

Same picture on both pipelines: from ~0 to ~50% touch-success, with the arm extending in nearly every episode and about a
third of the reaches held for 10 s. Where theirs is ahead is the base: reach 5 cm 90% against our 61–69%, i.e. their base
brings the arm within 5 cm more often. Candidates, all addressed by mixture v2 (§6z): our random-start data drives to the
origin-referenced goal (theirs to the start-referenced one), and our training set carries the 217 expert failures (theirs
filters them). "Arm extended" is an ignition flag, not a quality measure: it saturates (their expert 82%, ours 75%, because short
reaches move the EE less than 10 cm) and it rewards moving too much. Our K 144 trips it in 98% because its arm travels a
median 30 cm *more* than the expert on the same target: 23 cm where the expert moves 7 cm (51 short reaches, still touched
in 80%), 70 cm where the expert moves 28 cm (154 long reaches, touched in 39%). Quality is read from reach 5 cm, stable
1 cm and held 10 s. The settled / held / parked columns of this table are computed from our 100 ms traces (theirs at 1 ms),
so they can be slightly optimistic; success and reach use the 1 ms minimum distance.

**Caveats.** One seed per K (seeds 1–2 at K 144 running); the training set still holds the 217 expert failures (the
success-only retraining of §6z comes next); the K rule was pre-registered but the three K values were a guess from the
row-count argument, so 144 is "the best of three", not an optimum.

## 6z. Mixture v2 (2 Nov 17:30, pre-registered): goal from the start pose, expert successes only

**Why.** The lecturer's expert takes the goal bearing from the world origin (`atan2(target_y, target_x)`), where his robot
always starts. With a random start the goal is still "the point of the standoff ring facing the origin", which can be the
far side of the target from where the robot is, so the expert drives past the target and around it. 75% of mixture v1
(multistart, E1, E2) and the v1 random validation carry that geometry. Rodrigo's call: irrational, fix it. Second rule
(§7, 17:15): train and select on expert successes only.

**Change.** `goal_reference` in the expert config (`experiment_specs/expert/default.json`): `origin` (default, the
lecturer's rule, bit-identical to before) or `start` (bearing from the base start position; the same goal for the nominal
start, the near side of the ring for any other). Implemented in `simulation/episode.py::base_goal`, tested
(`tests/test_episode.py::GoalReferenceTests`). It is a generalisation of the goal's reference point, not a change to the
control law; the teammate's collector made the same choice (§6x).

**Data.** `e3_multistart3x_dart_low_v2_startgoal` (3072 episodes, same targets, start poses and noise seeds as v1) and
`e1_randompose_dart_low_v2_startgoal` (1024, the random validation bank), collected 17:30– with 14 workers; then the
replacement collection of their expert failures (new start / noise seed; a scenario that fails twice is discarded).
Split v2 = nominal (819 training / 205 validation geometries, unchanged) + multistart v2, validation by geometry as in v1
(`make_mixture_split.py`, extended to map replacement episodes to their plan target); trainers run with
`keep_expert_failures: false`. The v1 replacement collection (`e10`, 278 scenarios, origin goal) gave 262 successes and
16 second failures; it is not used in v2 (origin geometry).

**Runs (pre-registered).** On split v2, 3 seeds each, both v2 validations (nominal unchanged, random v2):
(1) MLP 28 pure control; (2) E9 K = 144. Predictions: the control's base failures on random starts shrink (the goal is now
where the approach ends) but the arm still does not ignite (§6t.1 is about the parked state, not the path); E9 keeps its
ignition (96–103 at 1 cm on v1) and gains on random starts; dropping the 605k post-timeout rows (zero wheels 2.5 m from the
goal, §7 17:15) should reduce the stops far from the box (E8a closed at 0.45–0.64 m; MLP 28 stopped outside in 76–96
episodes). All v1 numbers stay in this document as "mixture v1, origin goal"; v1 and v2 are not compared across.

## 6aa. Second wave on mixture v2 (decided 2 Nov 17:50, pre-registered): ignition for the models that already stop

Rodrigo rejected phase-balanced sampling (it uses the expert's phase labels: "medio trampa") and approved, in this order,
all on mixture v2 (§6z) unless stated:

1. **K 144 + dead band 0.45 rad/s** (evaluator option of §6t, no retraining): the three K 144 v1 seeds evaluated with
   `validation_deadband045.json` on both validations (`e9_trig144_db045_on_*`), queued after the E9c chain. Prediction:
   the base stops where E8d stopped it (within 10 cm of the expert's spot in 87–93%), the arm now ignites inside that
   window (+0.96 on the resting-arm row), the moving arm keeps the wheel command under the band (§6h), so held-10 s
   approaches the number of arrivals (v1: 107 of 140) and drift after parking drops below 0.22 m.
2. **Trigger weight instead of copies** (`trigger_weight: K`, loss of the trigger row ×K, no duplication, no jitter): on the
   MLP 28 (same K = 144) to show the equivalence, and in the recurrent trainers for **GRU 28** (§6h recipe, the model with the
   learned latch) and **E8a gated** (exact zero outside the box): the trigger *step* of each training sequence weighted ×K in
   the sequence loss. Prediction: MLP ≈ copies; GRU seed-dependent latch + ignition; gated head reaches when the gate closes.
3. **Action chunking** on the MLP 28: the network predicts the commands of the next H = 200 ms (9 × 200 outputs at 1 ms, or
   9 × 10 at a 20 ms chunk stride), the policy executes the first; loss on the whole chunk. The reach onset then appears in
   the labels of the ~200 rows before it instead of one, with no change to the data. The outputs remain wheel and arm
   commands (a short horizon of them). Prediction: ignition without copies; precision at least that of K 144.
4. ~~DAgger~~ **deferred (Rodrigo, 18:05)**. Not impossible: a shadow expert run on the learner's trajectory (as the
   teammate's hybrid) gives causal labels, but the done latch, the 25 s timeout and which rows to keep need rules that
   amount to designing a new expert (`docs/BASELINE_LIMITATIONS.md` L2). If taken up later: first run the shadow on the
   existing K 144 / GRU traces and check that its labels at the parked-asleep and creeping states are coherent.

Rule of the wave: one factor per arm, 3 seeds, both v2 validations, 1 / 2 / 5 cm stable arrivals + held 10 s + the
teammate's columns, mechanistic probe (`arm_switch_probe.py`) on every arm. `src/` is edited only between the running
chains (collectors and trainers hash their sources).

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
- E7a/E7b done and reported in §6l (1 Nov 20:13): open-loop residual 0–1% but the closed-loop creep is unchanged;
  arrivals 19–44 (stop weight) and 9–20 (L1) against 23–30 for the control. E2 collected (1775 successes, 273 stalls,
  same stall rate as the base data).
- E7c done and reported in §6m (1 Nov 22:09): arrivals unchanged (24–39), but the categorical head replaces the creep
  with sign-alternating chatter at three bins from zero and a stop gated by the 25 s clock of the stalled
  demonstrations (every training row after 25 s has zero wheels). E7d trained (offline NRMSE 0.104–0.108, the
  copycat signature) and evaluating; phase B (6144 mixture with E2) training, one trainer at a time, RAM logged every
  30 s (`rss_monitor.log` in the scratchpad).
- E7d done and reported in §6n (2 Nov 00:14): 11 / 19 / 16 arrivals with the best offline NRMSE (0.104–0.108); the
  augmentation removed the copycat and with it any effect of the input (3% of the wheel sensitivity, loop gain
  0.13–0.17); the off-shell residual at arrival is twice the control's and the base never parks. E7d closed.
  Phase B: MLP 28 on the 6144 mixture trained (offline 0.148 / 0.151 / 0.150, peak RSS 46 GB) and evaluating;
  categorical 41 bins on the 6144 mixture training; GRU next.
- E2 MLP 28 done and reported in §6o (2 Nov 02:34): 14 / 17 / 13 arrivals, worse than the 4096 control; the
  near-target starts fill the tolerance box with full-speed `orient` rotations (56% of the rows inside 8 cm are
  non-zero, median 3.0) and the closed-loop arrival residual doubles with a rotation component. E2 categorical
  (offline 0.157 / 0.160 / 0.161) evaluating.
- 2 Nov 02:38: the GRU trainer on the 6144 mixture was OOM-killed while loading (RSS 50 → 77 GB in one minute, the
  machine at 0 available; `padded_group` materialises the training set as a float64 array padded to the longest
  episode, 28 s of stalls, and normalises it in two more float64 copies: ~110 GB on 4914 episodes). The chain was
  stopped before the second seed repeated it; the categorical evaluators were left running. Fix (after the
  evaluators finish, nothing in `src/` is edited while they run): fill the normalised float32 tensors per episode
  from the same statistics, which reproduces the current numbers at a third of the memory; then the three
  GRU seeds run alone, one at a time.
- E2 categorical done and reported in §6p (2 Nov 03:17): 20 / 14 / 23 arrivals, same rotation at the arrival as the
  regression. E2 closed for the MLP family.
- 2 Nov 03:20: GRU trainer memory fix applied (`normalised_group` in `gru_policy.py`: per-episode normalisation and
  float32 cast, statistics from the same valid rows, test `GroupLoadingTests`; 98 tests OK, commit `4f4e95c`) and the
  three E2 GRU seeds relaunched alone (`chain_gru_e2.sh`), evaluations after the third seed. Verified: the load of the
  4914-episode group peaks at 43 GB RSS (was killed past 77 GB), 24 GB during training, 29 GB on the GPU; the first
  epoch starts 90 s after launch.
- E2 GRU done and reported in §6q (2 Nov 04:36): 41 / 43 / 37 nominal, 39 / 53 / 28 random with the best offline NRMSE of
  the programme (0.123–0.132); open loop the recurrent state latches the stop (1–4% of the drive command for any
  displacement up to 40 cm), closed loop the approach passes 18–44 cm from the box and rotates at the arrival as in §6o.
  E7 + E2 complete. §8 gathers the whole programme (table of every arm, six established points, open decisions).
- 28-input pure-transform variant done and reported in §6e (branch `rodrigo-pure28` merged).
- 2 Nov 07:40: transformer stop basin flat (§6s); E8 programme agreed; E8d (dead band 0.45 on trained models) launched,
  12 models × 2 validations, 8 workers per evaluation, two evaluations at a time.
- 2 Nov 06:50: zero separability measured (§6r): the expert never emits a non-zero wheel command below 0.45 rad/s; the
  policies' zero predictions sit below 0.35 (p99) and their drive commands above 0.7 (p1). A dead band at 0.45 is free offline.
- 2 Nov 10:40: E8d complete (§6t): dead band 0.45 on MLP 28, GRU 28, transformer 28, MLP 32; no gain, losses where
  a jostled arm used to reach (transformer 81/76 → 68/40, MLP 32 93/113 → 55/46). The MLP freezes at the edge of its
  zero (raw 0.31–0.38 under the band) and the clock pushes it over in 8–9 s; GRU seed 1 holds an exact zero for 25 s
  to the end of the episode and still arrives 94, because the stop is 10 cm off. §6t.1: the MLP's arm starts only when it is
  already moving (reach is triggered by the arm velocities; the resting-arm reach rows are ~1 per 12 000); the creeping base used
  to jostle it, the band removes the jostle. Same for the transformer and the MLP 32.
- 2 Nov 11:40: E8a (gated stop, gate seen by the arm head) and E8b (MLP→GRU) implemented, tested (105), launched (§6u).
- 2 Nov 12:50: E8a done and reported in §6v: 23 / 49 / 5 nominal, 17 / 34 / 8 random. The gate holds an exact zero and
  starts the arm (80% of closures) but it learned to *confirm* a still base, not to predict the stop: it reaches its closed
  value only after the labels are zero, and in closed loop it never closes inside the box (0 of the 25–47 episodes where
  the head met the expert's stop condition), only where the base stalls by itself at 0.4–0.6 m. The gated head also fits
  the demonstrations worse than the MLP 28 control (drive wheels 0.43 vs 0.23 rad/s) under the recurrent training recipe.
  Proposal E8g (stateless geometry-only gate, label-derived target, teacher forcing) written, not launched.
- 2 Nov 12:34: E8b done and reported in §6w: 55 / 44 / 79 nominal, 62 / 49 / 43 random, offline 0.136–0.162. No seed
  latches (wheels zero to the end in 0 / 10 / 0), parks and restarts with 13–25% residual like the stateless models; the
  encoder evens the seeds out at the base GRU's non-latching level. E8a + E8b closed; E8c and E8e next, one trainer at a time.
- 2 Nov 13:40: read the teammate's `feature/step3c-trigger-copies` (C3: 16 copies per episode of the first zero-wheel row with
  arm-qd jitter, stateless MLP 20 pure inputs): success 52–70% DEV, stable 5 cm ~90%, held 10 s 14–40%, from a 0.3% baseline.
  §6x explains it with §6t.1: it re-weights the one resting-arm reach row per episode, the arm ignites and the base reads it as
  the phase flag (§6h); creep remains (half the successes do not hold 10 s). Label-derived re-weighting, not privileged input.
  E9 (same option in our trainer, MLP 28, 4096 mixture) proposed, not launched.
- 2 Nov 15:20: E9 launched (§6y): trigger copies in our trainer, K sweep 16 / 144 / 512 at seed 0 first (our 1 kHz rows make
  their K = 16 nine times lighter), both validations, K rule pre-registered; then 3 seeds at the chosen K. E8c / E8e wait.
- 2 Nov 17:04: E9 K sweep done (§6y.1): seed 0, nominal 1/2/5 cm = 48/84/141 (K 16), **96/124/140 (K 144)**, 86/101/119
  (K 512); random 47/61/89, **103/116/121**, 66/83/98; arm extended 197/198/149 of 205 (control 3–103); held 10 s 22/107/79.
  Probe: output at the resting-arm parked row −0.02 (control) → +0.12 / +0.96 / +1.27. K = 144 chosen by the pre-registered
  rule; seeds 1–2 launched 17:20 with the replacement collection of the 278 failed scenarios in parallel.
- 2 Nov 17:15: rule from Rodrigo: **train and select on expert successes only**; failed scenarios are re-collected with
  another start pose / noise seed and discarded if they fail again. The 4096 mixture holds 217 training + 61 validation
  expert failures (195 navigation timeouts: 25 s circling 8.0–8.5 cm from the goal, then zero wheels with the base 2.5 m
  away and the arm reaching 1.1 m toward a 2 m target: 605k rows, 5% of all zero-wheel rows). `keep_expert_failures=false`
  implemented in the egocentric trainer (it was declared but unused), configs `mlp_pytorch_successes*.json`, spec
  `e10_replacement_expert_failures_dart_low.json` (278 scenarios). Next: §6z, control + E9 K 144 on success-only data.
- 2 Nov 17:30: mixture v2 decided and launched (§6z): `goal_reference: start` for random-start data, expert successes only.
  Collections running (3072 + 1024 episodes, 14 workers) beside the E9 K 144 seeds 1–2 chain. Then replacements, split v2,
  control + E9 K 144 (3 seeds each), both validations. v1 replacement collection e10: 262 / 278 succeeded.
- 2 Nov 17:50: second wave decided (§6aa): K 144 + band (queued), trigger weight on MLP / GRU / gated, action chunking;
  phase-balanced sampling rejected; DAgger deferred (18:05). Implementation starts when the running trainer and collectors end.
- Next, on request: E6 (§6c, nominal data without the clock) — the cheapest test of the reach-trigger mechanism.
- E2 needs a collector mode; nothing in `src/` is edited while a collector,
  trainer or evaluator is running (provenance guards).
