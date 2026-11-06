# 1. Data collection

How the demonstrations of the report were produced: what the collectors do, the criteria behind every choice, and the
exact order to reproduce the datasets.

The chapter has two kinds of content. Two rules are **definitions** fixed from the start (1.2): the base goal is
computed from the robot's own start pose, and models only learn from episodes where the expert succeeded. Three
ingredients were **experimental**, each tested against a control (1.3–1.5): DART noise on the executed command,
randomised start poses, and the amount of data.

| collector | task | code | output |
|---|---|---|---|
| reaching collector | reach a target with the mobile manipulator (the assignment task) | `scripts/collect_dataset.py` → `src/robot_learning/data/` | `datasets/<id>/` (project dataset layout) |
| Static-obstacle hold collector | reaching with one cylinder between start and target, 64-beam lidar | `scripts/obstacle/collect_static_obstacle.py` (wraps the group's `obstacle_task.collect`) | raw set → `scripts/obstacle/convert_obstacle_dataset.py` → project layout |
| Dynamic-obstacle collector | reaching with one moving cylinder crossing the path | `scripts/obstacle/dynamic_obstacle.py --dart` | raw set → same converter |

All three record at the simulator rate (1 kHz, every physics step), store the expert's **clean** command as the label
and the command actually executed separately, and keep every episode (successes and failures) with its metadata, so
any filtering is explicit and counted.

---

## 1.1 The expert and the episode (what one demonstration is)

The expert is the lecturer's controller, unchanged (`src/robot_learning/expert/controllers.py`, rules in
`experiment_specs/expert/default.json`):

- **Base**: a skid-steer state machine *align → drive → orient → done* towards a **base goal** ("dock") placed at a
  standoff distance behind the target (1.2). The standoff depends on the target height (`standoff`: 0.8 m at
  z = 0.1 m, 0.65 m for z in [0.5, 1.1] m, 0.35 m at 1.25 m, linear in between). Parking tolerance 8 cm / 0.05 rad,
  navigation timeout 25 s, speeds ≤ 0.25 m/s and ≤ 1 rad/s.
- **Arm**: held at its initial pose while the base drives; once navigation is done, differential IK (damped, joint-limit
  aware) generates a joint reference tracked by a gravity-compensated PD (kp 100, kd 30, |τ| ≤ 30 N·m). The arm is only
  asked to reach 0.85 m from the shoulder.
- **Stop rule** (the lecturer's `check_stop`): success when the end effector is within **1 cm** of the target, after at
  least 8 s, with navigation done; "stalled" when the error has not improved by 2 mm in 3 s; cap 40 000 steps (40 s).

One episode = one target and one start pose. The **observation** is the 21-vector the brief allows (base x, y, z,
bearing, 7 arm joint angles, 7 arm joint velocities, target x, y, z); the **label** is the 11-command action (4 wheel
velocities in rad/s, which the expert always sets pairwise equal per side, and 7 arm torques in N·m). The schema, units
and the observation → action → next-observation alignment are written into every dataset (`schema.json`,
`src/robot_learning/data/schema.py`). Fields marked `diagnostics_only` (expert phase, navigation flag, the arm's
internal reference) are recorded for analysis and are **never** model inputs.

---

## 1.2 Two definitions

### The base goal is computed from the start pose

The dock lies at the standoff distance behind the target, along the bearing **from the robot's start position** to the
target (`"goal_reference": "start"`, `base_goal` in `src/robot_learning/simulation/episode.py`). The lecturer's code
takes this bearing from the world origin, which is the same thing for his robot, because it always starts at the
origin: at the nominal start the two rules give the identical goal, so the lecturer's demonstrations are unchanged. For
any other start pose the start-referenced rule is the only sensible one: the robot approaches the target from its own
side and parks on the near side of the standoff ring.

### Learn only from expert successes

A failed demonstration teaches the failure: an expert that circles its goal for 25 s and gives up, or stops short of the
target, would be imitated just as faithfully as a successful one. So **models are trained and selected on expert
successes only**:

- the collectors keep every episode with its outcome (`success`, `termination`, `navigation_timeout` in the episode
  metadata), and the splits list every episode, failures included, so nothing disappears silently;
- the trainers drop the failures (`keep_expert_failures: false` → `robot_learning.data.splits.drop_expert_failures`)
  and record how many;
- a scenario whose expert run failed is **re-collected** with a new start pose and a new noise realisation (failures
  depend on the start and the noise, not on the target); if it fails again it is discarded.

| collection | episodes | expert failures | of which navigation timeout |
|---|---|---|---|
| nominal start, DART | 1024 | 129 (12.6 %) | 110 |
| random start, DART | 3072 | 266 (8.7 %) | 214 |
| random-start validation bank, DART | 1024 | 87 (8.5 %) | 68 |
| re-collection of the 266 + 87 | 353 | 31 (discarded) | 23 |

**What the failures are** (measured on the 129 nominal ones, re-run with the same noise and a 38 s timeout instead of
25 s): 110 are navigation failures in which the base circles 10–50 cm from its goal and never satisfies the
8 cm / 0.05 rad parking test under the wheel noise (more time changes nothing: the closest approach is unchanged in all
110); 19 are the expert's own stop declaring "closest pose reached" at 1–6 mm. The timeout decides *when* the expert
gives up, not *why* it fails; a successful drive takes at most 11 s (99th percentile).

---

## 1.3 Experiment 1: learning to recover with DART noise

**Problem.** The expert's trajectories are too clean: every demonstration follows the ideal path, so the data never show
the expert correcting a deviation. A cloned policy makes small errors, leaves the demonstrated states within a second,
and has no example of how to come back (the covariate shift of behaviour cloning).

**Idea.** Perturb the command the simulator *executes* while recording the expert's *clean* command as the label
(DART, Laskey et al. 2017). The expert sees the perturbed state at the next step and corrects it, so the data contain
the corrections, labelled with what the expert would do from there.

**Implementation** (`action_noise` in a collection spec; `make_perturbation` in `src/robot_learning/data/worker.py`):
held Gaussian noise, one new draw every `hold_steps` = 50 steps (50 ms), added to the executed command:

| noise | value | why |
|---|---|---|
| wheels | σ = 0.5 rad/s, **one draw per side** (applied as [right, left, left, right]) | keeps the skid-steer pair identity of the expert's command; 0.5 rad/s per side ≈ 0.04 m/s of linear speed and 0.12 rad/s of yaw rate |
| arm torques | σ = 1.0 N·m per joint | about 3 % of the torque envelope (30 N·m) |
| hold | 50 ms | independent noise every 1 ms averages out through the robot's inertia and barely moves it; holding each sample produces real deviations the expert must correct |
| labels | clean | `action` = expert command; `applied_action` = what was executed; `previous_action` = the previous *executed* command |

The noise stream is seeded from (noise seed, scenario seed), so a collection is reproducible bit for bit.

**Noise level.** Doubling the noise (σ 1.0 rad/s / 2.0 N·m, same nominal scenarios) made the expert itself fail in 46 %
of the episodes (555 / 1024 successes, against 895 / 1024 at the chosen level): with the success rule, the data would
lose half of the demonstrations, mostly their final approach, which is what the noise is meant to enrich. The lower
level was kept.

**Result** (five seeds, 205 held-out geometries, stable arrival within 1 cm; the control is the same scenarios
collected without noise: `reaching_nominal_clean.json`, `reaching_random_start_clean.json`,
`reaching_validation_random_start_clean.json`):

| same model, same scenarios | with DART | without DART |
|---|---|---|
| MLP-Ego (no ignition copies) | 2.0 ± 1.6 | 0.6 ± 0.9 |
| MLP-Ego-IC (the report's MLP) | **189 ± 9** | **0.0 ± 0.0** |
| offline validation error (NRMSE) of MLP-Ego-IC | 0.13–0.14 | **0.09–0.10** |

Without DART the policy fits the demonstrations *better* offline and never arrives in closed loop: its runs leave the
workspace (median final distance 250–1500 m, wheels at the envelope 75–88 % of the time).

DART also matters during the final **hold** (base parked, arm reaching): with the noise on, the base keeps being nudged
while the arm reaches, so the demonstrations show the arm compensating a moving base. That is why the reaching MLP
tolerates the few centimetres of base creep it produces itself (chapter 6), and why the obstacle sets needed the noise
kept on during the hold (1.6).

---

## 1.4 Experiment 2: randomised targets and start poses

### Targets: a grid over a bounded box

**Bounds.** The brief does not fix a range ("generalise to new targets and/or initial conditions"). In the plane the box
is x, y ∈ [−2, 2] m (drives of up to 2.8 m from the origin). The **height range follows a reachability criterion**: once
the base is parked, the target must be within the arm's reach (0.85 m from the shoulder) given the height of the
shoulder on the base (≈ 0.49 m), so z ∈ [0.3, 0.8] m, a band of about −0.2 / +0.3 m around the shoulder, keeps every
target reachable with margin. Measured check on 303 successful nominal episodes: with the base parked, the shoulder is
0.42–0.71 m from the target over the whole box (≥ 14 cm below the reach limit at every height band). Everything outside
the box is reserved for the generalisation banks (1.7).

**Target grid** (`experiment_specs/collection/target_grid_16x8x8.json`, made by `robot_learning.data.sampling.stratified_grid`,
regenerated and checked by `scripts/data/make_target_grid.py`): the box is cut into a **16 × 8 × 8 grid** (cells of
0.25 m × 0.5 m × 0.0625 m) and one target is drawn uniformly inside each of the 1024 cells. Compared with 1024 uniform
draws this guarantees that every region of the box is demonstrated (no holes, no clusters).

**Holdout chosen from geometry only.** 205 cells (20 %) are held out for validation, chosen *before* any target is
jittered or any episode is simulated, so no outcome can influence the choice. The selection is a small integer program
(`scipy.optimize.milp`) with these constraints, then the most spread-out feasible subset (maximin distance between cell
centres) out of 32 random objectives:

- each level of each axis holds out the same number of cells (±1): no slab of x, y or z is over- or under-validated;
- every octant of the box has at least one held-out cell;
- in every vertical pair of cells at most one is held out: every (x, y) column keeps training data at every height band.

The validation side is defined by the **target**: an episode whose target lies in a held-out cell is validation,
whatever its start pose (`scripts/data/make_train_val_split.py`). No target geometry is ever seen in both training and
validation, also across the nominal-start and random-start collections.

### Start poses

| condition | spec | bounds |
|---|---|---|
| nominal | `initial_base: {"mode": "nominal"}` | the lecturer's start: origin, facing +x |
| random start | `initial_base: {"mode": "uniform_xy", "x": [-2, 2], "y": [-2, 2], "yaw": [-π, π]}`, `"goal_reference": "start"` | position uniform in the same square as the targets, any heading; base goal from the start pose (1.2) |

The random-start collection demonstrates every grid target **three times** with three different start poses
(`reaching_random_start_dart`: 3 × 1024 episodes, the grid targets repeated three times).

**Result** (MLP-Ego-IC, five seeds, 1 cm; control: the nominal-start collection alone):

| training data | nominal validation | random-start validation |
|---|---|---|
| nominal starts only (721 successful episodes) | 152 ± 25 | **62 ± 15** |
| nominal + random starts (3 221) | 189 ± 9 | **193 ± 9** |

A policy trained on the lecturer's single start pose does not transfer to other start poses; with the random starts it
is as good from anywhere as from the origin, and better from the origin too.

---

## 1.5 Experiment 3: how much data

**How the amount was set.** The size is not a free number: it follows from two coverage choices, and a learning curve
then checks that it is enough.

1. *Target coverage*: one target per cell of the 16 × 8 × 8 grid, i.e. 1024 target positions, one every 25 cm in x,
   50 cm in y and 6 cm in height; 819 of them are for training.
2. *Start coverage*: each target demonstrated once from the nominal start and three times from random starts, i.e.
   4 × 1024 = 4096 episodes (about 40 min of collection with 28 processes).
3. Removing the validation cells and the expert failures (re-collected once) leaves **3 221 training episodes**.
4. *Check*: train the final model on growing random subsets of these 3 221 episodes (below). If the result were still
   climbing steeply at the full size, more data would be needed; it is not: the last 821 episodes (+34 %) add 4
   arrivals on the nominal validation (195 → 199) and the offline error stopped improving at 1 600 episodes. The
   amount is therefore at the knee of the curve: enough, with little to gain from collecting more.

In one sentence, for the report: *the dataset size was set by coverage (one target per cell of a 16 × 8 × 8 grid, four
start poses per target), and a learning curve shows it sits at saturation: 34 % more episodes add 2 % more arrivals.*

**The training mixture** (every reaching model of the report):

| dataset (spec) | start | episodes | expert successes | used for |
|---|---|---|---|---|
| `reaching_nominal_dart` | nominal | 1024 (grid, once) | 895 | training (819 geometries) + nominal validation (205) |
| `reaching_random_start_dart` | random | 3072 (grid × 3) | 2806 | training + validation |
| `reaching_recollect_failures_dart` | random | 353 (the failed scenarios, new start and noise) | 322 | replaces failures (1.2) |

After the train/validation split and the success rule: **3 221 training episodes** and 802 validation episodes. Three
levels, not to be confused:

- **target positions**: 819 grid cells for training (205 held out), one fixed target each;
- **episodes** (demonstrations): 3 221, i.e. each training target demonstrated about four times, once from the nominal
  start (721 episodes) and three times from random starts (2 500 episodes, re-collections included);
- **rows** (state → command pairs, what the network is fitted on): about 35 M, since an episode lasts ~12.5 s at 1 kHz.

 The closed-loop validation sets are the 205 nominal held-out
geometries and the 205 held-out geometries of the random-start bank `reaching_validation_random_start_dart`.
Collection cost: about 10 min per 1024 episodes with 28 worker processes.

**Learning curve** with the final recipe (MLP-Ego-IC, seed 0). Each point is a random subset of the 3 221 training
episodes (one fixed permutation, `training_episode_cap` option of the trainer), so every point keeps the mixture's
proportion of about 22 % nominal and 78 % random-start episodes; stable arrivals within 1 cm of 205:

| training episodes | 400 | 800 | 1 600 | 2 400 | 3 221 |
|---|---|---|---|---|---|
| nominal / random-start episodes | 93 / 307 | 193 / 607 | 370 / 1230 | 549 / 1851 | 721 / 2500 |
| distinct target cells (of 819) | 331 | 553 | 772 | 818 | 819 |
| nominal validation | 142 | 148 | 177 | 195 | 199 |
| random-start validation | 137 | 127 | 174 | 187 | 200 |
| offline NRMSE | 0.209 | 0.189 | 0.142 | 0.142 | 0.139 |

- The closed-loop result keeps improving up to the full set while the offline error has already flattened at 1 600:
  the extra episodes add coverage of rare states (the arm's ignition, the parked base), which the average error hardly
  sees.
- Up to ~2 400 episodes the curve mixes two effects, more distinct targets and more start poses per target; from 2 400
  on every training cell is covered and the last gain (195 → 199 nominal, 187 → 200 random start) comes from more start
  poses per target only.
- Read together with experiment 2: at about the same size, 721 nominal-only episodes give 62 ± 15 on the random-start
  validation while 800 mixed episodes give 127 (nominal validation 152 ± 25 against 148). The diversity of start poses
  matters more than the number of episodes.
- The curve is one seed (the end points have five); it is a sizing argument, not a precise scaling law.

---

## 1.6 The hold: extending the episodes for stability

**Problem.** A demonstration that ends at the moment of the reach shows nothing about *staying* there. Cloned policies
reach the target and leave it 2–3 s later.

- **reaching**: not extended. An episode ends at the first reach within 1 cm after at least 8 s; its last seconds are the
  reach phase with the base parked and, with DART, perturbed, which is what the reaching policies learn to hold from.
- **Static obstacle**: the group's collector stops at the reach. `collect_static_obstacle.py` keeps the expert running for
  **10 s after the first reach** (`OBSTACLE_HOLD_S`, default 10) without editing the group's code (it wraps
  `obstacle_task.collect.check_stop`). Holding alone was not enough: the group's collector switches the base noise off
  once navigation is done, so the base is still in 99 % of the hold rows and the arm is never seen compensating
  anything. `OBSTACLE_HOLD_WHEEL_NOISE=0.5` adds our wheel noise (0.5 rad/s per side, 50 ms) whenever the expert's wheel
  command is exactly zero, i.e. during the hold.

  | set | noise during navigation | noise during the hold | MLP held 10 s (3 seeds) |
  |---|---|---|---|
  | recorded (group, stops at the reach) | group's | none | 8 ± 4 % |
  | hold, group's noise | group's | arm only, faded near the target | 18 ± 17 % |
  | D: hold, our DART | ours (v 0.0389 m/s, w 0.1178 rad/s = 0.5 rad/s per wheel side; torque 1.0 N·m; 50 ms) | arm only | 59 ± 3 % |
  | **B**: hold, our DART + base noise in the hold | ours | arm + base | **77 ± 9 %** |
  | F: B + base noise faded within 0.5 m of the dock | ours, faded | arm + base | 71 ± 26 % |

- **Dynamic obstacle**: `dynamic_obstacle.py` holds 10 s after the first reach (the template's stall rule is ignored
  during the hold), and `--dart` applies the set-F noise (base noise faded near the dock, 0.5 rad/s per wheel side in the
  hold, torque noise throughout).

---

## 1.7 Evaluation banks (collected, never trained on)

Out-of-distribution banks, 210 episodes each (205 evaluated, `scripts/data/make_bank_split.py`), expert recorded on each
as the reference (it succeeds in 210/210, 209/210 and 210/210, so the banks test the learner, not the expert): `ood_far_targets` (max(|x|, |y|) from 2.2 to 3 m), `ood_target_height` (z in [0.15, 0.28] and
[0.85, 1.05] m), `ood_far_start` (start x in [2.3, 3.5] m, y in [−3.5, 3.5] m). Reserved test banks
(`test_reserved_nominal`, `test_reserved_random_start`, 64 episodes each, split `test`) are only for the final table.

---

## 1.8 The scripts, what is configurable, and how to use them

### Layout

```
scripts/collect_dataset.py              reaching collector (entry point of src/robot_learning/data/cli.py)
scripts/simulate.py                     watch the expert on one target in the PyBullet GUI
scripts/data/make_target_grid.py        regenerate the 16x8x8 target grid and check it against the frozen one
scripts/data/make_train_val_split.py     train/validation split of reaching datasets by held-out target cell
scripts/data/make_bank_split.py         split of an evaluation bank (all episodes evaluated but the last 5)
scripts/obstacle/collect_static_obstacle.py  static-obstacle collector: the group's collector + 10 s hold (+ base noise in the hold)
scripts/obstacle/dynamic_obstacle.py      moving-obstacle expert: development runs, training collection (--dart), policy test (--policy)
scripts/obstacle/convert_obstacle_dataset.py  raw obstacle set (group format) -> project dataset layout + split
scripts/run_tests.py                    unit and integration tests

src/robot_learning/
  settings.py            where RoboEnv is (ROBOENV_ROOT or ../RoboEnv), optional .env
  expert/controllers.py  the lecturer's base and arm controllers and stop rule (unchanged)
  simulation/            setup.py (robot + physics), config.py (episode rules), episode.py (one episode, the lecturer's
                         step order; used by the collector, the viewer and later the evaluator), interactive.py (viewer)
  data/                  cli.py (arguments, dataset directory), config.py (spec validation, scenario generation),
                         scheduler.py (parallel episode processes, resume, lock), worker.py (one episode + DART noise),
                         storage.py (recorder, hashes, provenance), schema.py (fields and units), sampling.py (target
                         grid), splits.py (episode inventory, splits, expert-success filter)

experiment_specs/expert/default.json      the expert's rules (gains, tolerances, timeout, standoff table)
experiment_specs/collection/*.json        one file per collection of the report
```

### reaching collector: `scripts/collect_dataset.py`

| option | effect |
|---|---|
| `--config SPEC.json` | the collection spec (below) |
| `--episodes N` | override the number of episodes: always a prefix of the full set (scenario i depends only on (seed, i); a spec with an explicit `targets` list keeps its first N targets, and cannot grow beyond the list) |
| `--max-steps N` | override the step cap (1 kHz steps; default 40 000 = 40 s) |
| `--workers N` | parallel episode processes (headless only) |
| `--gui` | show the episodes in PyBullet (use with `--workers 1`) |
| `--plan-only` | freeze config and scenarios without simulating (inspect `scenarios.json`, then `--resume`) |
| `--resume DIR` | continue an interrupted collection with its frozen config (refuses if any source file changed) |
| `--output-dir DIR` | where the dataset directory is created (default `datasets/`) |
| `--roboenv-root DIR` | RoboEnv location (default `ROBOENV_ROOT` or `../RoboEnv`) |

**Collection spec** (JSON; every field optional, defaults in `src/robot_learning/data/config.py`):

| field | default | meaning |
|---|---|---|
| `name` | `"dataset"` | part of the dataset directory name |
| `episodes` | 64 | number of episodes |
| `seed` | 20261023 | scenario seed (targets and start poses) |
| `split` | `"development"` | label only (`development`, `train`, `validation`, `test`) |
| `targets` | `null` | explicit list of [x, y, z], one per episode (the report's specs copy the grid targets) |
| `target_box` | x, y ∈ [−2, 2], z ∈ [0.3, 0.8] | uniform sampling box, used when `targets` is null |
| `initial_base` | `{"mode": "nominal"}` | or `{"mode": "uniform_xy", "x": [lo, hi], "y": [lo, hi], "yaw": [lo, hi]}` |
| `goal_reference` | `"origin"` | `"start"` for random starts (1.2; identical at the nominal start) |
| `action_noise` | `null` | DART: `{"wheel_std": 0.5, "torque_std": 1.0, "hold_steps": 50, "seed": N}` |
| `record_every_steps` | 1 | keep one transition every N steps (does not change the control rate) |
| `max_steps` | 40000 | step cap |
| `headless` | true | false = GUI |
| `base_controller`, `arm_controller`, `stop`, `standoff`, `physics` | `experiment_specs/expert/default.json` | override single expert rules, e.g. `"stop": {"reach_tolerance": 0.02}` or `"base_controller": {"nav_timeout": 38.0}` |

Examples:

```bash
PY="pixi run --manifest-path ../RoboEnv/pixi.toml python"
$PY scripts/collect_dataset.py --config experiment_specs/collection/smoke.json            # 2 short episodes, pipeline check
$PY scripts/collect_dataset.py --config experiment_specs/collection/reaching_nominal_dart.json --episodes 64 --workers 8
$PY scripts/collect_dataset.py --config experiment_specs/collection/reaching_random_start_dart.json --plan-only
$PY scripts/collect_dataset.py --resume datasets/<id> --workers 28
$PY scripts/collect_dataset.py --config experiment_specs/collection/smoke.json --gui --workers 1
$PY scripts/simulate.py --target 0 1.1 0.5                                                  # the expert on one target
```

A new collection = a new spec file: copy one of `experiment_specs/collection/`, change `name`, `seed`, `episodes`,
the start or the noise, and remove `targets` to sample uniformly in `target_box`.

### Splits

```bash
$PY scripts/data/make_target_grid.py [OUT.json]                              # regenerate + compare the frozen grid
$PY scripts/data/make_train_val_split.py OUT.json PRIMARY_DIR [EXTRA_DIR ...]  # reaching datasets on the grid targets
$PY scripts/data/make_bank_split.py BANK_DIR OUT.json [HOLDBACK=5]            # evaluation banks
```

### Static obstacle: `scripts/obstacle/collect_static_obstacle.py`

Runs the group's collector (`python -m obstacle_task.collect`, from the group's `final/robotnik_arm_task`) with the hold
added; it takes all of the group's options, the useful ones being:

| option | meaning |
|---|---|
| `--set-name` | `train2` (2000 episodes, seed 3), `dev2` (300, seed 300), `confirm2` (300, seed 302), ... |
| `--n-episodes N`, `--seed S`, `--episodes 0-99` | size, scenario seed, subset |
| `--noise-frac F` | fraction of episodes with noise (1.0 for ours) |
| `--noise-v`, `--noise-w`, `--noise-torque`, `--noise-hold` | DART noise: m/s, rad/s, N·m, s (ours: 0.0389, 0.1178, 1.0, 0.05) |
| `--noise-fade-base`, `--noise-fade-arm` | distance (m) within which the noise fades to zero near the dock / target (0 = no fade) |
| `--workers N`, `--out-dir DIR`, `--resume` | parallelism, output set folder, continue a set |

Environment variables of the wrapper: `OBSTACLE_HOLD_S` (hold after the first reach, default 10 s, 0 = stop at the reach),
`OBSTACLE_HOLD_WHEEL_NOISE` (rad/s per wheel side during the hold, default 0 = off; 0.5 for set B),
`OBSTACLE_TASK_DIR` (the group's task folder). Then `convert_obstacle_dataset.py --source RAW --output DS --split-out SPLIT
[--validation-fraction 0.15] [--split-seed 20261104] [--workers 8]`.

### Dynamic obstacle: `scripts/obstacle/dynamic_obstacle.py`

| option | meaning |
|---|---|
| `--seed S`, `--episodes 0-99` | scenario seed and episode indices (seeds 1–2: expert development, 3: training set, 4: test) |
| `--mode dwa\|blind` | the expert (default) or the plain reaching template as control |
| `--dart` | set-F noise on the executed command (training collections) |
| `--policy RUN_DIR[,db=0.45]` | a learned policy drives, the expert only monitors and labels (closed-loop test) |
| `--hold-s S` | hold after the first reach (default 10 s) |
| `--workers N`, `--out-dir DIR` | parallelism, output folder (npz per episode, index.csv, scenarios.json, config.json) |
| `--gif`, `--gui` | top-view animation per episode, PyBullet window |

Scenario ranges (cylinder speed 0.10–0.25 m/s, crossing point, angle, timing) and the expert's cost weights are the
constants at the top of the file, each with its reason.

---

## 1.9 Reproducing the data, in order

Environment: the RoboEnv pixi environment (`ROBOENV_ROOT`, default `../RoboEnv`; `default` env for collection and
simulation, `ml` env for training). From the project root, with
`PY="pixi run --manifest-path ../RoboEnv/pixi.toml python"`:

```bash
# 0. check the pipeline and the frozen target grid
$PY scripts/run_tests.py
$PY scripts/data/make_target_grid.py                      # -> OK: identical to the frozen target grid

# 1. reaching collections (each prints DATASET: datasets/<timestamp>_<name>_<hash>)
$PY scripts/collect_dataset.py --config experiment_specs/collection/reaching_nominal_dart.json --workers 28
$PY scripts/collect_dataset.py --config experiment_specs/collection/reaching_random_start_dart.json --workers 28
$PY scripts/collect_dataset.py --config experiment_specs/collection/reaching_validation_random_start_dart.json --workers 28
#    the failed scenarios of the two random-start collections, re-collected with new start / noise:
$PY scripts/collect_dataset.py --config experiment_specs/collection/reaching_recollect_failures_dart.json --workers 28
#    controls and banks: reaching_nominal_clean, reaching_random_start_clean, reaching_validation_random_start_clean,
#    ood_far_targets, ood_target_height, ood_far_start

# 2. splits
$PY scripts/data/make_train_val_split.py runs/splits/reaching_train_val.json NOMINAL_DART_DIR RANDOM_START_DART_DIR RECOLLECT_DIR
$PY scripts/data/make_train_val_split.py runs/splits/reaching_validation_random_start.json VALIDATION_RANDOM_START_DIR
$PY scripts/data/make_bank_split.py OOD_DIR runs/splits/bank_<name>.json

# 3. static obstacle, set B (needs the group's final/robotnik_arm_task with obstacle_task: OBSTACLE_TASK_DIR)
cd $OBSTACLE_TASK_DIR && OBSTACLE_HOLD_WHEEL_NOISE=0.5 $PY <project>/scripts/obstacle/collect_static_obstacle.py \
    --set-name train2 --workers 12 --noise-frac 1.0 --noise-v 0.0389 --noise-w 0.1178 --noise-torque 1.0 \
    --noise-hold 0.05 --noise-fade-base 0 --noise-fade-arm 0 --out-dir <project>/datasets/static_obstacle/raw/set_B
cd <project> && pixi run --manifest-path ../RoboEnv/pixi.toml -e ml python scripts/obstacle/convert_obstacle_dataset.py \
    --source datasets/static_obstacle/raw/set_B --output datasets/static_obstacle/set_B \
    --split-out runs/splits/static_obstacle_set_B.json --validation-fraction 0.15 --split-seed 20261104
#    set F: the same with --noise-fade-base 0.5

# 4. dynamic obstacle: training set (seed 3) and the converter
$PY scripts/obstacle/dynamic_obstacle.py --seed 3 --episodes 0-1599 --dart --workers 28 --out-dir runs/obstacle/dyn_train_seed3_dart1600
pixi run --manifest-path ../RoboEnv/pixi.toml -e ml python scripts/obstacle/convert_obstacle_dataset.py \
    --source runs/obstacle/dyn_train_seed3_dart1600 --output datasets/dynamic/dyn_seed3_dart1600_lidar \
    --split-out runs/splits/dyn_seed3_dart1600_split.json
```

The split scripts were checked against the splits used for the report (episode for episode, identical inventory
fingerprints), and `dynamic_obstacle.py` reproduces the report's expert, blind and DART recordings bit for bit.

### Names used in the experiment records

The specs were renamed for this branch; the `name` field (and so the dataset directory name) changed, the scenarios
did not (they depend on the seed only). The runs and splits of the report, in the `rodrigo` branch, use the old names:

| spec here | dataset / split name in the experiment records |
|---|---|
| `target_grid_16x8x8.json` | `stratified1024_plan.json` |
| `reaching_nominal_dart` | `dart_stratified1024_low` (`dart_stratified1024_wheel0_5_torque1_0_hold50`) |
| `reaching_nominal_clean` | `stratified1024_nominal` |
| `reaching_random_start_dart` | `e3_multistart3x_dart_low_v2_startgoal` |
| `reaching_random_start_clean` | `e3_multistart3x_v2_startgoal_clean` |
| `reaching_validation_random_start_dart` | `e1_randompose_dart_low_v2_startgoal` |
| `reaching_validation_random_start_clean` | `e1_randompose_v2_startgoal_clean` |
| `reaching_recollect_failures_dart` | `e10b_replacement_v2_expert_failures_startgoal_dart_low` |
| split `reaching_train_val.json` | `v2_mixture.json` |
| split `reaching_validation_random_start.json` | `e1_randompose_v2_startgoal_819_205.json` |
| static obstacle, sets D / B / F | `step9_train2_hold10_ourdart`, `..._ourdart_basenoise`, `..._ourdart_basenoise_fade` |

### What a dataset directory contains

`config.json` (resolved spec, frozen), `scenarios.json` (target, start pose and seed of every episode), `schema.json`,
`provenance.json` (hashes of every source file, robot asset and package version: a collection refuses to resume if any
changed), `source_snapshot/` (copy of the code that produced it), `episodes/<id>.npz` + `<id>.json` (arrays and
metadata: outcome, termination, durations, the expert's base goal), `summary.json`, `logs/`. A collection can be
stopped and resumed (`--resume DIR`); `--plan-only` freezes the scenarios without simulating; `--gui --workers 1`
watches the episodes. `scripts/simulate.py` runs the expert interactively on one target.
