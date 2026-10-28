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

## 7. Status

- Done: decomposition (§1), variant `raw21_target_clock_v1`; E1 collected,
  trained (15 runs `runs/training/mlp_e1_*`), evaluated
  (`runs/evaluation/e1_mlp_*_seeds_on_clean_validation_20261027`,
  `…_on_e1_random_validation_20261027`,
  `dart_low_mlp_raw21plus_seeds_on_e1_random_validation_20261027`).
- Next, on request: the dilution test (E3'), then the mixture.
- E2 needs a collector mode; nothing in `src/` is edited while a collector,
  trainer or evaluator is running (provenance guards).
