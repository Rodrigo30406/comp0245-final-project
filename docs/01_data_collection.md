# 1. Data collection

How the demonstrations of the report were produced: what the collectors do, the criteria behind every choice, and the
exact order to reproduce the datasets. Everything here is the **v2** data definition (from 2 Nov 2026); section 1.8
explains in two paragraphs what was wrong with v1 and why it is not used anywhere in the report.

| collector | task | code | output |
|---|---|---|---|
| A-to-B collector | reach a target with the mobile manipulator (the assignment task) | `scripts/collect_dataset.py` → `src/robot_learning/data/` | `datasets/<id>/` (project dataset layout) |
| Static-obstacle hold collector | A-to-B with one cylinder between start and target, 64-beam lidar | `scripts/obstacle/collect_step9_hold.py` (wraps the group's `obstacle_task.collect`) | raw set → `scripts/obstacle/convert_step9_dataset.py` → project layout |
| Dynamic-obstacle collector | A-to-B with one moving cylinder crossing the path | `scripts/obstacle/dynamic_expert.py --dart` | raw set → same converter |

All three record at the simulator rate (1 kHz, every physics step), store the expert's **clean** command as the label
and the command actually executed separately, and keep every episode (successes and failures) with its metadata so the
filtering is explicit and counted (section 1.5).

---

## 1.1 The expert and the episode (what one demonstration is)

The expert is the lecturer's controller, unchanged (`src/robot_learning/expert/controllers.py`, rules in
`experiment_specs/expert/default.json`):

- **Base**: a skid-steer state machine *align → drive → orient → done* towards a **base goal** ("dock") placed at a
  standoff distance behind the target, along the bearing from the start position to the target. The standoff depends on
  the target height (`standoff`: 0.8 m at z = 0.1 m, 0.65 m for z in [0.5, 1.1] m, 0.35 m at 1.25 m, linear in between).
  Parking tolerance 8 cm / 0.05 rad, navigation timeout 25 s, speeds ≤ 0.25 m/s and ≤ 1 rad/s.
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

## 1.2 Learning to recover: DART noise on the executed command

**Problem.** The expert's trajectories are too clean: every demonstration follows the ideal path, so the data never show
the expert correcting a deviation. A cloned policy makes small errors, leaves the demonstrated states within a second,
and has no example of how to come back (the classic covariate shift of behaviour cloning).

**Criterion.** Perturb the command the simulator *executes* while recording the expert's *clean* command as the label
(DART, Laskey et al. 2017). The expert sees the perturbed state at the next step and corrects it, so the data contain
the corrections, labelled with what the expert would do from there.

**Implementation** (`action_noise` in a collection spec; `make_perturbation` in `src/robot_learning/data/worker.py`):
held Gaussian noise, one new draw every `hold_steps` = 50 steps (50 ms), added to the executed command:

| noise | value | why |
|---|---|---|
| wheels | σ = 0.5 rad/s, **one draw per side** (applied as [right, left, left, right]) | keeps the skid-steer pair identity of the expert's command; 0.5 rad/s ≈ 0.04 m/s of linear speed, 0.12 rad/s of yaw rate |
| arm torques | σ = 1.0 N·m per joint | about 3 % of the torque envelope (30 N·m); visible against the PD's corrections |
| hold | 50 ms | independent noise every 1 ms averages out through the robot's inertia and barely moves it; holding each sample produces real deviations the expert must correct |
| labels | clean | `action` = expert command; `applied_action` = what was executed; `previous_action` = the previous *executed* command |

The noise stream is seeded from (noise seed, scenario seed), so a collection is reproducible bit for bit.

**Why "low" noise.** Two levels were compared on the first data definition: σ 0.5 / 1.0 ("low") and σ 1.0 / 2.0.
The stronger noise made the expert itself fail in 46 % of the episodes (555 / 1024 successes against 895 / 1024 at low
noise), so the data lost most of the complete final approaches it was meant to enrich; the low level was kept.

**Evidence that it is needed (v2, five seeds, 205 held-out geometries, stable arrival within 1 cm):**

| same model, same 3 221 scenarios | with DART | without DART |
|---|---|---|
| MLP-Ego (no ignition copies) | 2.0 ± 1.6 | 0.6 ± 0.9 |
| MLP-Ego-IC (the report's MLP) | **189 ± 9** | **0.0 ± 0.0** |
| offline validation error (NRMSE) of MLP-Ego-IC | 0.13–0.14 | **0.09** |

Without DART the policy fits the demonstrations *better* offline and never arrives in closed loop: its runs leave the
workspace (median final distance 250–1500 m, wheels at the envelope 75–88 % of the time). The clean twin collections
(`stratified1024_nominal.json`, `e3_multistart3x_v2_startgoal_clean.json`, `e1_randompose_v2_startgoal_clean.json`:
same scenarios and seeds, no `action_noise`) exist only for this control.

DART also matters during the final **hold** (base parked, arm reaching): with the noise on, the base keeps being nudged
while the arm reaches, so the demonstrations show the arm compensating a moving base. That is why the A-to-B MLP tolerates
the few centimetres of base creep it produces itself (see the base-drift chapter), and why the obstacle sets needed the
noise kept on during the hold (section 1.6).

---

## 1.3 Randomisation: targets and start poses, and their bounds

### Targets: a stratified plan instead of uniform sampling

**Bounds.** The brief does not fix a range ("generalise to new targets and/or initial conditions"); the box was set at
the start of the project: x, y ∈ [−2, 2] m (drives of up to 2.8 m from the origin), z ∈ [0.3, 0.8] m. It is inside what
the expert can solve with margin, measured on 303 successful nominal episodes: when the base has parked, the shoulder
(at 0.49 m height) is 0.42–0.71 m from the target over the whole box, against the arm's 0.85 m reach limit (≥ 14 cm of
margin at every height band; the standoff table puts the dock 0.72–0.78 m from the target in the plane). The z range
spans roughly −0.2 / +0.3 m around the shoulder height. Everything outside the box is reserved for the generalisation
banks (section 1.7), where the expert still succeeds (210/210, 209/210, 210/210), so the banks test the learner, not the
expert.

**Plan** (`experiment_specs/collection/stratified1024_plan.json`, made by `robot_learning.data.sampling.stratified_grid`,
regenerated and checked by `scripts/data/make_target_plan.py`): the box is cut into a **16 × 8 × 8 grid** (cells of
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
whatever its start pose (`scripts/data/make_geometry_split.py`). So no target geometry is ever seen in both training and
validation, also across the nominal-start and random-start collections.

### Start poses

| condition | spec | bounds |
|---|---|---|
| nominal | `initial_base: {"mode": "nominal"}` | the lecturer's start: origin, facing +x |
| random start | `initial_base: {"mode": "uniform_xy", "x": [-2, 2], "y": [-2, 2], "yaw": [-π, π]}` | position uniform in the same square as the targets, any heading |

With a random start the base goal is placed along the bearing **from the start position** to the target
(`"goal_reference": "start"`). At the nominal start this is exactly the lecturer's rule (bearing from the origin, where
his robot always starts), so the nominal data are unchanged; for any other start it keeps the dock on the near side of
the standoff ring. (Using the world origin, as v1 did, parked a randomly started robot on the far side of the ring: an
irrational expert, see 1.8.)

Each random-start collection demonstrates every plan target **three times** with three different start poses
(`e3_multistart3x_*`: 3 × 1024 episodes, targets = plan targets repeated three times).

**Why random starts are needed (v2, MLP-Ego-IC, five seeds, 1 cm):**

| training data | nominal validation | random-start validation |
|---|---|---|
| nominal starts only (721 successful episodes) | 152 ± 25 | **62 ± 15** |
| nominal + random starts (3 221) | 189 ± 9 | **193 ± 9** |

A policy trained on the lecturer's single start pose does not transfer to other start poses; with the random starts it
is as good from anywhere as from the origin.

---

## 1.4 How much data

**Mixture v2** (the training data of every A-to-B model in the report):

| dataset (spec) | start | episodes | expert successes | used for |
|---|---|---|---|---|
| `dart_stratified1024_low` | nominal | 1024 (plan, once) | 895 | training (819 geometries) + nominal validation (205) |
| `e3_multistart3x_dart_low_v2_startgoal` | random | 3072 (plan × 3) | 2806 | training + random-start validation |
| `e10b_replacement_v2_expert_failures_startgoal_dart_low` | random | 353 (the failed scenarios, new start and noise) | 322 | replaces failures (section 1.5) |

After the geometry split and the success filter: **3 221 training episodes** (721 nominal + 2 500 random start;
about 35 M rows at 1 kHz) and 802 validation episodes, of which the 205 nominal ones and the 205 random-start ones of
the validation bank `e1_randompose_dart_low_v2_startgoal` are the closed-loop validation sets. Collection cost: about
10 min per 1024 episodes with 28 worker processes.

**Why this much.** Learning curve with the final recipe (MLP-Ego-IC, seed 0, fixed random subsets of the 3 221
successes, stable arrivals within 1 cm of 205):

| training episodes | 400 | 800 | 1 600 | 2 400 | 3 221 |
|---|---|---|---|---|---|
| nominal validation | 142 | 148 | 177 | 195 | 199 |
| random-start validation | 137 | 127 | 174 | 187 | 200 |
| offline NRMSE | 0.209 | 0.189 | 0.142 | 0.142 | 0.139 |

The closed-loop result keeps improving up to the full set while the offline error has already flattened at 1 600: the
extra episodes add coverage of rare states (the arm's ignition, the parked base), which the average error hardly sees.
The curve is one seed; it is a sizing argument, not a precise scaling law.

---

## 1.5 Cutting the episodes where the expert failed

**Rule.** Policies are trained and selected on **expert successes only**. A failed demonstration teaches the failure: in
the first data definition 6.6 % of the training episodes (13 % of the rows) were expert timeouts, 25 s of circling just
outside the parking tolerance followed by zero wheels with the base far from the goal and the arm reaching towards a
target it could not reach.

**How.** The collectors keep every episode and its outcome (`success`, `termination`, `navigation_timeout` in the
episode metadata); the splits list every episode (failures included, so nothing disappears silently); the trainers drop
the failures (`keep_expert_failures: false` → `robot_learning.data.splits.drop_expert_failures`) and record how many.
Scenarios whose expert run failed are **re-collected** with a new start pose and a new noise realisation (failures
depend on the start and on the noise, not on the target); if they fail again they are discarded:

| collection | episodes | expert failures | of which navigation timeout |
|---|---|---|---|
| nominal, DART | 1024 | 129 (12.6 %) | 110 |
| random start, DART | 3072 | 266 (8.7 %) | 214 |
| random-start validation bank, DART | 1024 | 87 (8.5 %) | 68 |
| replacement of the 266 + 87 | 353 | 31 (discarded) | 23 |

**What the failures are** (measured on the 129 nominal ones, re-run with the same noise and a 38 s timeout instead of
25 s): 110 are navigation failures in which the base circles 10–50 cm from its goal and never satisfies the
8 cm / 0.05 rad parking test under the wheel noise (more time changes nothing: min distance unchanged in all 110); 19 are
the expert's own stop declaring "closest pose reached" at 1–6 mm. The timeout decides *when* the expert gives up, not
*why* it fails; a successful drive takes at most 11 s (99th percentile).

---

## 1.6 Extending the episodes for stability: the hold

**Problem.** A demonstration that ends at the moment of the reach shows nothing about *staying* there. Cloned policies
reach the target and leave it 2–3 s later.

- **A-to-B**: not extended. An episode ends at the first reach within 1 cm after at least 8 s; its last seconds are the
  reach phase with the base parked and, with DART, perturbed, which is what the A-to-B policies learn to hold from.
- **Static obstacle (step 9)**: the group's collector stops at the reach. `collect_step9_hold.py` keeps the expert
  running for **10 s after the first reach** (`STEP9_HOLD_S`, default 10) without editing the group's code (it wraps
  `obstacle_task.collect.check_stop`). Holding alone was not enough: the group's collector switches the base noise off
  once navigation is done, so the base is still in 99 % of the hold rows and the arm is never seen compensating
  anything (the MLP trained on that hold data held 10 s in 18 ± 17 % of the episodes). `STEP9_HOLD_WHEEL_NOISE=0.5` adds our wheel noise
  (0.5 rad/s per side, 50 ms) whenever the expert's wheel command is exactly zero, i.e. during the hold.

  | set | noise during navigation | noise during the hold | MLP held 10 s (3 seeds) |
  |---|---|---|---|
  | recorded (group, stops at reach) | group's | none | 8 ± 4 % |
  | D: hold, our DART | ours (v 0.0389 m/s, w 0.1178 rad/s = 0.5 rad/s per wheel side; torque 1.0 N·m; 50 ms) | arm only | 59 ± 3 % |
  | **B**: hold, our DART + base noise in the hold | ours | arm + base | **77 ± 9 %** |
  | F: B + base noise faded within 0.5 m of the dock | ours, faded | arm + base | 71 ± 26 % |

- **Dynamic obstacle**: `dynamic_expert.py` holds 10 s after the first reach (the template's stall rule is ignored
  during the hold), and `--dart` applies the set-F noise (base noise faded near the dock, 0.5 rad/s per wheel side in the
  hold, torque noise throughout).

---

## 1.7 Evaluation banks (collected, never trained on)

Out-of-distribution banks, 210 episodes each (205 evaluated, `scripts/data/make_bank_split.py`), expert recorded on each
as the reference: `ood_far_targets` (max(|x|, |y|) from 2.2 to 3 m), `ood_target_height` (z in [0.15, 0.28] and
[0.85, 1.05] m), `ood_far_start` (start x in [2.3, 3.5] m, y in [−3.5, 3.5] m). Reserved test banks
(`test_reserved_nominal`, `test_reserved_random_start`, 64 episodes each, split `test`) are only for the final table.

---

## 1.8 Why v1 is not used

The first data definition (27 Oct – 2 Nov) had two defects that make its results unusable as evidence:

1. **The base goal of random-start episodes was referenced to the world origin**, as in the lecturer's code where the
   robot always starts there. From any other start this placed the dock on the far side of the standoff ring, so 75 % of
   the random-start demonstrations taught an approach to a point defined by the origin, and the random-start validation
   measured the same artefact.
2. **Expert failures were trained on** (`keep_expert_failures: true` in the configs but never implemented): 6.6 % of the
   episodes, 13 % of the rows, taught circling and giving up.

v2 fixes both (`goal_reference: "start"`, success filter with re-collection) on the same plan targets, start-pose
distribution and noise seeds, so v2 numbers are never compared with v1 numbers. The v1 code and notes remain in the
`rodrigo` branch for the record.

---

## 1.9 Reproducing the data, in order

Environment: the RoboEnv pixi environment (`ROBOENV_ROOT`, default `../RoboEnv`; `default` env for collection and
simulation, `ml` env for training). From the project root, with
`PY="pixi run --manifest-path ../RoboEnv/pixi.toml python"`:

```bash
# 0. check the pipeline and the frozen target plan
$PY scripts/run_tests.py
$PY scripts/data/make_target_plan.py                      # -> OK: identical to the frozen plan

# 1. A-to-B collections (each prints DATASET: datasets/<timestamp>_<name>_<hash>)
$PY scripts/collect_dataset.py --config experiment_specs/collection/dart_stratified1024_low.json --workers 28
$PY scripts/collect_dataset.py --config experiment_specs/collection/e3_multistart3x_dart_low_v2_startgoal.json --workers 28
$PY scripts/collect_dataset.py --config experiment_specs/collection/e1_randompose_dart_low_v2_startgoal.json --workers 28
#    the failed scenarios of the two random-start collections, re-collected with new start / noise:
$PY scripts/collect_dataset.py --config experiment_specs/collection/e10b_replacement_v2_expert_failures_startgoal_dart_low.json --workers 28
#    controls and banks (optional): stratified1024_nominal, e3_multistart3x_v2_startgoal_clean,
#    e1_randompose_v2_startgoal_clean, ood_far_targets, ood_target_height, ood_far_start

# 2. splits
$PY scripts/data/make_geometry_split.py runs/splits/v2_mixture.json NOMINAL_DIR E3_DIR E10B_DIR
$PY scripts/data/make_geometry_split.py runs/splits/e1_randompose_v2_startgoal_819_205.json E1_DIR
$PY scripts/data/make_bank_split.py OOD_DIR runs/splits/bank_<name>.json

# 3. static obstacle (needs the group's final/robotnik_arm_task with obstacle_task: STEP9_TASK_DIR)
cd $STEP9_TASK_DIR && STEP9_HOLD_WHEEL_NOISE=0.5 $PY <project>/scripts/obstacle/collect_step9_hold.py \
    --set-name train2 --workers 12 --noise-frac 1.0 --noise-v 0.0389 --noise-w 0.1178 --noise-torque 1.0 \
    --noise-hold 0.05 --noise-fade-base 0 --noise-fade-arm 0 --out-dir <project>/datasets/step9/raw/train2_hold10_B
cd <project> && pixi run --manifest-path ../RoboEnv/pixi.toml -e ml python scripts/obstacle/convert_step9_dataset.py \
    --source datasets/step9/raw/train2_hold10_B --output datasets/step9/train2_hold10_B_lidar \
    --split-out runs/splits/step9_train2_hold10_B_split.json --validation-fraction 0.15 --split-seed 20261104

# 4. dynamic obstacle: training set (seed 3) and the converter
$PY scripts/obstacle/dynamic_expert.py --seed 3 --episodes 0-1599 --dart --workers 28 --out-dir runs/obstacle/dyn_train_seed3_dart1600
pixi run --manifest-path ../RoboEnv/pixi.toml -e ml python scripts/obstacle/convert_step9_dataset.py \
    --source runs/obstacle/dyn_train_seed3_dart1600 --output datasets/dynamic/dyn_seed3_dart1600_lidar \
    --split-out runs/splits/dyn_seed3_dart1600_split.json
```

The split scripts were checked against the splits used for the report: `make_geometry_split.py` reproduces
`v2_mixture.json` and `e1_randompose_v2_startgoal_819_205.json`, and `make_bank_split.py` reproduces the bank splits,
episode for episode with identical inventory fingerprints. The cleaned `dynamic_expert.py` reproduces the report's
expert, blind and DART recordings bit for bit.

### What a dataset directory contains

`config.json` (resolved spec, frozen), `scenarios.json` (target, start pose and seed of every episode), `schema.json`,
`provenance.json` (hashes of every source file, robot asset and package version: a collection refuses to resume if any
changed), `source_snapshot/` (copy of the code that produced it), `episodes/<id>.npz` + `<id>.json` (arrays and
metadata: outcome, termination, durations, the expert's base goal), `summary.json`, `logs/`. A collection can be
stopped and resumed (`--resume DIR`); `--plan-only` freezes the scenarios without simulating; `--gui --workers 1`
watches the episodes. `scripts/simulate.py` runs the expert interactively on one target.
